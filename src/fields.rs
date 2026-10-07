//! Traversals that read a node's children by its type's `_fields` names.
//!
//! `walk_dfs` reads the first `len(_fields)` entries of the instance dict,
//! which holds for a parsed tree. A transformer can build a node whose dict
//! has another order (`ast.Call(lineno=1, func=f)` stores `lineno` first),
//! and these functions run on such trees, so each field is looked up by
//! name instead.
//!
//! A field read can run Python code (a property, a `__getattr__`, a value's
//! `__hash__` or `__repr__`) that drops the tree's own references, so every
//! node a walk will still touch is held as a `Strong`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use pyo3::ffi::{self, PyObject, PyTypeObject};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple, PyType};
use pyo3::{PyTypeInfo, intern};

use crate::{Edge, get_instance_dict_fast, issubclass_of_ast, resolve_base_types};

/// An owned reference, dropped with the GIL held.
pub(crate) struct Strong(*mut PyObject);

impl Strong {
    /// # Safety
    /// `object` is a live object and the GIL is held.
    #[inline(always)]
    pub(crate) unsafe fn borrowed(object: *mut PyObject) -> Self {
        unsafe { ffi::Py_INCREF(object) };
        Self(object)
    }

    /// # Safety
    /// `object` is a new reference the caller gives up.
    #[inline(always)]
    unsafe fn owned(object: *mut PyObject) -> Self {
        Self(object)
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *mut PyObject {
        self.0
    }

    #[inline(always)]
    pub(crate) fn into_ptr(self) -> *mut PyObject {
        let object = self.0;
        std::mem::forget(self);
        object
    }
}

impl Clone for Strong {
    #[inline(always)]
    fn clone(&self) -> Self {
        unsafe { Self::borrowed(self.0) }
    }
}

impl Drop for Strong {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe { ffi::Py_DECREF(self.0) };
    }
}

/// A list holding `items`, whose references it takes over.
pub(crate) fn into_pylist<'py>(
    py: Python<'py>,
    items: impl ExactSizeIterator<Item = Strong>,
) -> PyResult<Bound<'py, PyAny>> {
    unsafe {
        let list =
            Bound::from_owned_ptr_or_err(py, ffi::PyList_New(items.len() as ffi::Py_ssize_t))?;
        let slots = (*(list.as_ptr() as *mut ffi::PyListObject)).ob_item;
        for (position, item) in items.enumerate() {
            *slots.add(position) = item.into_ptr();
        }
        Ok(list)
    }
}

#[derive(Default)]
struct PointerHasher(u64);

impl Hasher for PointerHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 << 8 | byte as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    fn write_usize(&mut self, value: usize) {
        self.0 = ((value as u64) >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type PointerMap<V> = HashMap<usize, V, BuildHasherDefault<PointerHasher>>;

const LINENO: u8 = 1;
const COL_OFFSET: u8 = 2;
const END_LINENO: u8 = 4;
const END_COL_OFFSET: u8 = 8;

/// An AST type's `_fields` names, and which location `_attributes` it has.
/// It holds the type, so the address it is cached under cannot be reused by
/// another type.
struct TypeInfo {
    _kind: Py<PyAny>,
    fields: Vec<Py<PyAny>>,
    locations: u8,
}

thread_local! {
    static TYPE_INFO: RefCell<PointerMap<Box<TypeInfo>>> = RefCell::new(PointerMap::default());
}

fn names_of(py: Python<'_>, kind: &Bound<'_, PyAny>, attribute: &str) -> Vec<Py<PyAny>> {
    let Ok(names) = kind.getattr(attribute) else {
        return Vec::new();
    };
    let Ok(names) = names.cast_into::<PyTuple>() else {
        return Vec::new();
    };
    names
        .iter()
        .filter_map(|name| {
            let name = name.cast_into::<pyo3::types::PyString>().ok()?;
            let interned = pyo3::types::PyString::intern(py, name.to_str().ok()?);
            Some(interned.into_any().unbind())
        })
        .collect()
}

/// The cached `TypeInfo` for `kind`. Entries are boxed and never removed,
/// so the pointer stays valid for the life of the thread. A miss reads the
/// type's attributes, which can run Python code that walks a tree, so no
/// borrow of the cache is held across it.
fn type_info(py: Python<'_>, kind: *mut PyTypeObject) -> *const TypeInfo {
    let cached = TYPE_INFO.with(|cache| {
        cache
            .borrow()
            .get(&(kind as usize))
            .map(|entry| &**entry as *const TypeInfo)
    });
    if let Some(info) = cached {
        return info;
    }
    let kind_object = unsafe { Bound::from_borrowed_ptr(py, kind.cast::<PyObject>()) };
    let mut locations = 0;
    for name in names_of(py, &kind_object, "_attributes") {
        locations |= match name.bind(py).extract::<&str>().unwrap_or_default() {
            "lineno" => LINENO,
            "col_offset" => COL_OFFSET,
            "end_lineno" => END_LINENO,
            "end_col_offset" => END_COL_OFFSET,
            _ => 0,
        };
    }
    let info = Box::new(TypeInfo {
        fields: names_of(py, &kind_object, "_fields"),
        locations,
        _kind: kind_object.unbind(),
    });
    TYPE_INFO
        .with(|cache| &**cache.borrow_mut().entry(kind as usize).or_insert(info) as *const TypeInfo)
}

/// Call `visit(field_index, value)` for each field `node` has, in `_fields`
/// order, as `ast.iter_fields` reads them: from the node's dict, else by
/// attribute lookup, which finds the class default of an optional field a
/// constructor left out (`ast.Constant(value=1)` has no `kind` entry).
/// `node` must be kept alive by the caller. The value is borrowed: a `visit`
/// that runs Python code holds it first.
unsafe fn for_each_field(
    py: Python<'_>,
    node: *mut PyObject,
    info: &TypeInfo,
    mut visit: impl FnMut(usize, *mut PyObject) -> PyResult<()>,
) -> PyResult<()> {
    let dict = get_instance_dict_fast(node).map(|dict| unsafe { Strong::borrowed(dict) });
    for (index, name) in info.fields.iter().enumerate() {
        if let Some(dict) = &dict {
            let value = unsafe { ffi::PyDict_GetItemWithError(dict.as_ptr(), name.as_ptr()) };
            if !value.is_null() {
                visit(index, value)?;
                continue;
            }
            if unsafe { !ffi::PyErr_Occurred().is_null() } {
                return Err(PyErr::fetch(py));
            }
        }
        let inherited = unsafe { ffi::PyObject_GetAttr(node, name.as_ptr()) };
        if inherited.is_null() {
            let error = PyErr::fetch(py);
            if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) {
                continue;
            }
            return Err(error);
        }
        let inherited = unsafe { Strong::owned(inherited) };
        visit(index, inherited.as_ptr())?;
    }
    Ok(())
}

/// The items of `list`, borrowed, read one at a time so Python code run
/// between them cannot leave the loop reading a resized list.
#[inline(always)]
unsafe fn for_each_item(
    list: *mut PyObject,
    mut visit: impl FnMut(ffi::Py_ssize_t, *mut PyObject) -> PyResult<()>,
) -> PyResult<()> {
    let mut index = 0;
    while index < unsafe { ffi::PyList_GET_SIZE(list) } {
        visit(index, unsafe { ffi::PyList_GET_ITEM(list, index) })?;
        index += 1;
    }
    Ok(())
}

struct Walk<'py> {
    py: Python<'py>,
    base: (*mut PyTypeObject, *mut PyTypeObject),
    list_type: *mut PyTypeObject,
}

impl<'py> Walk<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            py,
            base: resolve_base_types(py)?,
            list_type: PyList::type_object_raw(py),
        })
    }

    /// Whether `value` is an `ast.AST` instance, at any depth of subclass.
    #[inline(always)]
    fn is_ast(&self, value: *mut PyObject) -> bool {
        let kind = unsafe { ffi::Py_TYPE(value) };
        if issubclass_of_ast(kind, self.base) {
            return true;
        }
        let first = unsafe { (*kind).tp_base };
        if first.is_null() || first == &raw mut ffi::PyBaseObject_Type {
            return false;
        }
        unsafe { ffi::PyType_IsSubtype(kind, self.base.0) != 0 }
    }

    /// Push the AST children of `node` onto `stack` so they pop in
    /// `ast.iter_child_nodes` order.
    fn push_children(&self, node: &Strong, stack: &mut Vec<Strong>) -> PyResult<()> {
        let info = unsafe { &*type_info(self.py, ffi::Py_TYPE(node.as_ptr())) };
        let start = stack.len();
        unsafe {
            for_each_field(self.py, node.as_ptr(), info, |_, value| {
                if ffi::Py_TYPE(value) == self.list_type {
                    for_each_item(value, |_, item| {
                        if self.is_ast(item) {
                            stack.push(Strong::borrowed(item));
                        }
                        Ok(())
                    })?;
                } else if self.is_ast(value) {
                    stack.push(Strong::borrowed(value));
                }
                Ok(())
            })?;
        }
        stack[start..].reverse();
        Ok(())
    }

    /// Every node under `root`, `root` included, in depth-first pre-order.
    fn pre_order(&self, root: &Bound<'_, PyAny>) -> PyResult<Vec<Strong>> {
        let mut result = Vec::new();
        let mut stack = vec![unsafe { Strong::borrowed(root.as_ptr()) }];
        while let Some(node) = stack.pop() {
            self.push_children(&node, &mut stack)?;
            result.push(node);
        }
        Ok(result)
    }
}

/// A set of exact types, sorted for a binary search.
pub(crate) struct TypeSet(Vec<usize>);

impl TypeSet {
    pub(crate) fn new(kinds: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let mut pointers = kinds
            .iter()
            .map(|kind| Ok(kind.cast_into::<PyType>()?.as_type_ptr() as usize))
            .collect::<PyResult<Vec<_>>>()?;
        pointers.sort_unstable();
        pointers.dedup();
        Ok(Self(pointers))
    }

    #[inline]
    pub(crate) fn contains(&self, node: &Strong) -> bool {
        self.0
            .binary_search(&(unsafe { ffi::Py_TYPE(node.as_ptr()) } as usize))
            .is_ok()
    }
}

/// Return every node under `node` (`node` included) whose exact type is in
/// `kinds`, in depth-first pre-order. Unlike `walk_frontier`, matches are
/// descended into, except a node whose exact type is in `prune`: it is
/// returned if it matches, and its children are skipped. `node` itself is
/// always descended.
#[pyfunction]
#[pyo3(signature = (node, kinds, prune=None))]
pub fn walk_of_types<'py>(
    py: Python<'py>,
    node: Bound<'py, PyAny>,
    kinds: Bound<'py, PyTuple>,
    prune: Option<Bound<'py, PyTuple>>,
) -> PyResult<Bound<'py, PyAny>> {
    let kinds = TypeSet::new(&kinds)?;
    let prune = prune.map(|prune| TypeSet::new(&prune)).transpose()?;
    let walk = Walk::new(py)?;
    let root = node.as_ptr();
    let mut result = Vec::new();
    let mut stack = vec![unsafe { Strong::borrowed(root) }];
    while let Some(current) = stack.pop() {
        let pruned = current.as_ptr() != root
            && prune.as_ref().is_some_and(|prune| prune.contains(&current));
        if !pruned {
            walk.push_children(&current, &mut stack)?;
        }
        if kinds.contains(&current) {
            result.push(current);
        }
    }
    into_pylist(py, result.into_iter())
}

/// Set `child.parent = parent` for every node under `node`, visiting parents
/// in depth-first pre-order and each parent's children in
/// `ast.iter_child_nodes` order. `node` itself is left alone.
#[pyfunction]
pub fn set_parents(py: Python<'_>, node: Bound<'_, PyAny>) -> PyResult<()> {
    let walk = Walk::new(py)?;
    let name = intern!(py, "parent");
    let mut children = Vec::new();
    let mut stack = vec![unsafe { Strong::borrowed(node.as_ptr()) }];
    while let Some(parent) = stack.pop() {
        walk.push_children(&parent, &mut children)?;
        for child in children.iter().rev() {
            if unsafe { ffi::PyObject_SetAttr(child.as_ptr(), name.as_ptr(), parent.as_ptr()) }
                == -1
            {
                return Err(PyErr::fetch(py));
            }
        }
        stack.append(&mut children);
    }
    Ok(())
}

const VISIT: i64 = 0;
const ENTER: i64 = 1;
const LEAVE: i64 = 2;

enum Step {
    Child(Strong),
    Leave(Strong),
}

/// The calls a `NodeVisitor` makes descending from `node`, as a list of
/// `(event, node)` pairs: `(0, n)` for a descendant whose exact type is in
/// `kinds` (dispatched, not descended), and `(1, n)` / `(2, n)` around the
/// descent of a node whose exact type is in `bracket`. `node` itself is
/// always descended, and bracketed if its type is in `bracket`.
#[pyfunction]
pub fn walk_frontier_events<'py>(
    py: Python<'py>,
    node: Bound<'py, PyAny>,
    kinds: Bound<'py, PyTuple>,
    bracket: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let kinds = TypeSet::new(&kinds)?;
    let bracket = TypeSet::new(&bracket)?;
    let walk = Walk::new(py)?;
    let mut events: Vec<(i64, Strong)> = Vec::new();
    let mut stack: Vec<Step> = Vec::new();
    let mut children = Vec::new();
    let mut open =
        |current: Strong, stack: &mut Vec<Step>, events: &mut Vec<(i64, Strong)>| -> PyResult<()> {
            children.clear();
            walk.push_children(&current, &mut children)?;
            if bracket.contains(&current) {
                events.push((ENTER, current.clone()));
                stack.push(Step::Leave(current));
            }
            stack.extend(children.drain(..).map(Step::Child));
            Ok(())
        };
    open(
        unsafe { Strong::borrowed(node.as_ptr()) },
        &mut stack,
        &mut events,
    )?;
    while let Some(step) = stack.pop() {
        match step {
            Step::Leave(current) => events.push((LEAVE, current)),
            Step::Child(current) if kinds.contains(&current) => events.push((VISIT, current)),
            Step::Child(current) => open(current, &mut stack, &mut events)?,
        }
    }
    let codes =
        [VISIT, ENTER, LEAVE].map(|code| code.into_pyobject(py).unwrap().into_any().unbind());
    unsafe {
        let list =
            Bound::from_owned_ptr_or_err(py, ffi::PyList_New(events.len() as ffi::Py_ssize_t))?;
        let slots = (*(list.as_ptr() as *mut ffi::PyListObject)).ob_item;
        for (position, (event, current)) in events.into_iter().enumerate() {
            let pair = Bound::from_owned_ptr_or_err(py, ffi::PyTuple_New(2))?;
            ffi::PyTuple_SET_ITEM(
                pair.as_ptr(),
                0,
                codes[event as usize].clone_ref(py).into_ptr(),
            );
            ffi::PyTuple_SET_ITEM(pair.as_ptr(), 1, current.into_ptr());
            *slots.add(position) = pair.into_ptr();
        }
        Ok(list)
    }
}

/// `ast.fix_missing_locations`, without recursing per tree level. Returns
/// `node`.
#[pyfunction]
pub fn fix_missing_locations<'py>(
    py: Python<'py>,
    node: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let walk = Walk::new(py)?;
    let names = [
        (LINENO, intern!(py, "lineno"), false),
        (COL_OFFSET, intern!(py, "col_offset"), false),
        (END_LINENO, intern!(py, "end_lineno"), true),
        (END_COL_OFFSET, intern!(py, "end_col_offset"), true),
    ];
    let one = 1i64.into_pyobject(py)?.into_any().unbind();
    let zero = 0i64.into_pyobject(py)?.into_any().unbind();
    let mut locations: Vec<[Py<PyAny>; 4]> =
        vec![[one.clone_ref(py), zero.clone_ref(py), one, zero]];
    let mut stack: Vec<(Strong, usize)> = vec![(unsafe { Strong::borrowed(node.as_ptr()) }, 0)];
    let mut children = Vec::new();
    while let Some((current, inherited)) = stack.pop() {
        let info = unsafe { &*type_info(py, ffi::Py_TYPE(current.as_ptr())) };
        let mut own = inherited;
        if info.locations != 0 {
            let dict = get_instance_dict_fast(current.as_ptr())
                .map(|dict| unsafe { Strong::borrowed(dict) });
            let mut values: [Py<PyAny>; 4] =
                locations[inherited].each_ref().map(|v| v.clone_ref(py));
            for (slot, (flag, name, none_is_missing)) in names.iter().enumerate() {
                if info.locations & flag == 0 {
                    continue;
                }
                let found = match &dict {
                    Some(dict) => unsafe {
                        ffi::PyDict_GetItemWithError(dict.as_ptr(), name.as_ptr())
                    },
                    None => std::ptr::null_mut(),
                };
                if found.is_null() && unsafe { !ffi::PyErr_Occurred().is_null() } {
                    return Err(PyErr::fetch(py));
                }
                let missing =
                    found.is_null() || (*none_is_missing && unsafe { found == ffi::Py_None() });
                if missing {
                    unsafe { Bound::from_borrowed_ptr(py, current.as_ptr()) }
                        .setattr(name, values[slot].bind(py))?;
                } else {
                    values[slot] = unsafe { Bound::from_borrowed_ptr(py, found) }.unbind();
                }
            }
            locations.push(values);
            own = locations.len() - 1;
        }
        children.clear();
        walk.push_children(&current, &mut children)?;
        stack.extend(children.drain(..).map(|child| (child, own)));
    }
    Ok(node)
}

#[inline(always)]
fn mix(hash: u64, value: u64) -> u64 {
    (hash.rotate_left(5) ^ value).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// A structural hash per node, keyed by `id(node)`. Two subtrees that would
/// give the same `ast.dump` get the same hash; a leaf value's type is mixed
/// in, so `1`, `1.0` and `True` (equal hashes in Python) stay apart. A leaf
/// that is not a str, bytes, int, bool, None or Ellipsis is hashed by its
/// `repr`, as `ast.dump` prints it: NaNs compare equal there and `-0.0` and
/// `0.0` do not, and a list constant has a repr where it has no hash.
#[pyfunction]
pub fn subtree_hashes<'py>(
    py: Python<'py>,
    node: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let hashes = hash_subtrees(py, &node)?;
    let result = pyo3::types::PyDict::new(py);
    for (&pointer, &hash) in &hashes {
        let key = unsafe {
            Bound::from_owned_ptr_or_err(
                py,
                ffi::PyLong_FromVoidPtr(pointer as *mut std::ffi::c_void),
            )?
        };
        result.set_item(key, hash as i64)?;
    }
    Ok(result.into_any())
}

/// `subtree_hashes(node)[id(node)]`, without building the dict.
#[pyfunction]
pub fn subtree_hash(py: Python<'_>, node: Bound<'_, PyAny>) -> PyResult<i64> {
    let hashes = hash_subtrees(py, &node)?;
    Ok(hashed(&hashes, node.as_ptr())? as i64)
}

fn hashed(hashes: &PointerMap<u64>, node: *mut PyObject) -> PyResult<u64> {
    hashes.get(&(node as usize)).copied().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("the tree changed while it was being hashed")
    })
}

/// Whether `value`'s hash is what `ast.dump` compares: its type's hash is
/// C code that agrees with its repr.
#[inline(always)]
fn hashes_as_dumped(value: *mut PyObject) -> bool {
    unsafe {
        let kind = ffi::Py_TYPE(value);
        kind == &raw mut ffi::PyUnicode_Type
            || kind == &raw mut ffi::PyLong_Type
            || kind == &raw mut ffi::PyBool_Type
            || kind == &raw mut ffi::PyBytes_Type
            || value == ffi::Py_None()
            || value == ffi::Py_Ellipsis()
    }
}

fn hash_subtrees(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<PointerMap<u64>> {
    let walk = Walk::new(py)?;
    let nodes = walk.pre_order(node)?;
    let mut hashes: PointerMap<u64> = PointerMap::default();
    hashes.reserve(nodes.len());
    let leaf = |value: *mut PyObject, hashes: &PointerMap<u64>| -> PyResult<u64> {
        if walk.is_ast(value) {
            if let Some(&hash) = hashes.get(&(value as usize)) {
                return Ok(hash);
            }
            // A field that computes its node gives a new one on every read.
            let child = unsafe { Bound::from_borrowed_ptr(py, value) };
            return hashed(&hash_subtrees(py, &child)?, value);
        }
        let kind = unsafe { ffi::Py_TYPE(value) } as u64;
        let hash = unsafe {
            if hashes_as_dumped(value) {
                ffi::PyObject_Hash(value)
            } else {
                let repr = Bound::from_owned_ptr_or_err(py, ffi::PyObject_Repr(value))?;
                ffi::PyObject_Hash(repr.as_ptr())
            }
        };
        if hash == -1 && unsafe { !ffi::PyErr_Occurred().is_null() } {
            return Err(PyErr::fetch(py));
        }
        Ok(mix(kind, hash as u64))
    };
    for current in nodes.iter().rev() {
        if hashes.contains_key(&(current.as_ptr() as usize)) {
            continue;
        }
        let kind = unsafe { ffi::Py_TYPE(current.as_ptr()) };
        let info = unsafe { &*type_info(py, kind) };
        let mut hash = mix(0, kind as u64);
        unsafe {
            for_each_field(py, current.as_ptr(), info, |index, value| {
                let value = Strong::borrowed(value);
                let value_hash = if ffi::Py_TYPE(value.as_ptr()) == walk.list_type {
                    let mut items = mix(u64::MAX, ffi::PyList_GET_SIZE(value.as_ptr()) as u64);
                    for_each_item(value.as_ptr(), |_, item| {
                        let item = Strong::borrowed(item);
                        items = mix(items, leaf(item.as_ptr(), &hashes)?);
                        Ok(())
                    })?;
                    items
                } else {
                    leaf(value.as_ptr(), &hashes)?
                };
                hash = mix(mix(hash, index as u64), value_hash);
                Ok(())
            })?;
        }
        hashes.insert(current.as_ptr() as usize, hash);
    }
    Ok(hashes)
}

impl Walk<'_> {
    /// Push the AST children of `parent` onto `stack` so they pop in
    /// `ast.iter_child_nodes` order, each with the slot it was read from.
    fn push_child_edges(&self, parent: &Strong, stack: &mut Vec<Edge>) -> PyResult<()> {
        let info = unsafe { &*type_info(self.py, ffi::Py_TYPE(parent.as_ptr())) };
        let start = stack.len();
        unsafe {
            for_each_field(self.py, parent.as_ptr(), info, |index, value| {
                let key = info.fields[index].as_ptr();
                if ffi::Py_TYPE(value) == self.list_type {
                    for_each_item(value, |item, child| {
                        if self.is_ast(child) {
                            stack.push(Edge {
                                node: Strong::borrowed(child),
                                parent: parent.clone(),
                                key,
                                index: item,
                            });
                        }
                        Ok(())
                    })?;
                } else if self.is_ast(value) {
                    stack.push(Edge {
                        node: Strong::borrowed(value),
                        parent: parent.clone(),
                        key,
                        index: -1,
                    });
                }
                Ok(())
            })?;
        }
        stack[start..].reverse();
        Ok(())
    }
}

/// Descendants of `node` (excluding `node`) whose exact type is in `kinds`,
/// in depth-first pre-order, without descending below any of them. This is
/// the set `ast.NodeVisitor.generic_visit` dispatches to `visit_<kind>`
/// methods when every other node falls through to the generic descent.
pub(crate) fn frontier_edges(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    kinds: &Bound<'_, PyTuple>,
) -> PyResult<Vec<Edge>> {
    let kinds = TypeSet::new(kinds)?;
    let walk = Walk::new(py)?;
    let mut result = Vec::new();
    let mut stack = Vec::new();
    walk.push_child_edges(&unsafe { Strong::borrowed(node.as_ptr()) }, &mut stack)?;
    while let Some(edge) = stack.pop() {
        if kinds.contains(&edge.node) {
            result.push(edge);
        } else {
            walk.push_child_edges(&edge.node, &mut stack)?;
        }
    }
    Ok(result)
}
