//! Traversals that read a node's children by its type's `_fields` names.
//!
//! `walk_dfs` reads the first `len(_fields)` entries of the instance dict,
//! which holds for a parsed tree. A transformer can build a node whose dict
//! has another order (`ast.Call(lineno=1, func=f)` stores `lineno` first),
//! and these functions run on such trees, so each field is looked up by
//! name instead.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use pyo3::ffi::{self, PyObject, PyTypeObject};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple, PyType};
use pyo3::{PyTypeInfo, intern};

use crate::{Edge, get_instance_dict_fast, issubclass_of_ast, resolve_base_types, vec_into_pylist};

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
struct TypeInfo {
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
/// so the pointer stays valid for the life of the thread.
fn type_info(py: Python<'_>, kind: *mut PyTypeObject) -> *const TypeInfo {
    TYPE_INFO.with(|cache| {
        let mut cache = cache.borrow_mut();
        let entry = cache.entry(kind as usize).or_insert_with(|| {
            let kind = unsafe { Bound::from_borrowed_ptr(py, kind.cast::<PyObject>()) };
            let attributes = names_of(py, &kind, "_attributes");
            let mut locations = 0;
            for name in &attributes {
                locations |= match name.bind(py).extract::<&str>().unwrap_or_default() {
                    "lineno" => LINENO,
                    "col_offset" => COL_OFFSET,
                    "end_lineno" => END_LINENO,
                    "end_col_offset" => END_COL_OFFSET,
                    _ => 0,
                };
            }
            Box::new(TypeInfo {
                fields: names_of(py, &kind, "_fields"),
                locations,
            })
        });
        &**entry as *const TypeInfo
    })
}

/// Call `visit(field_index, value)` for each field `node` has set, in
/// `_fields` order. Values are borrowed from the node's dict.
unsafe fn for_each_field(
    py: Python<'_>,
    node: *mut PyObject,
    info: &TypeInfo,
    mut visit: impl FnMut(usize, *mut PyObject) -> PyResult<()>,
) -> PyResult<()> {
    let Some(dict) = get_instance_dict_fast(node) else {
        return Ok(());
    };
    for (index, name) in info.fields.iter().enumerate() {
        let value = unsafe { ffi::PyDict_GetItemWithError(dict, name.as_ptr()) };
        if value.is_null() {
            if unsafe { !ffi::PyErr_Occurred().is_null() } {
                return Err(PyErr::fetch(py));
            }
            continue;
        }
        visit(index, value)?;
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

    fn is_ast(&self, value: *mut PyObject) -> bool {
        issubclass_of_ast(unsafe { ffi::Py_TYPE(value) }, self.base)
    }

    /// Push the AST children of `node` onto `stack` so they pop in
    /// `ast.iter_child_nodes` order.
    fn push_children(&self, node: *mut PyObject, stack: &mut Vec<*mut PyObject>) -> PyResult<()> {
        let info = unsafe { &*type_info(self.py, ffi::Py_TYPE(node)) };
        let start = stack.len();
        unsafe {
            for_each_field(self.py, node, info, |_, value| {
                if ffi::Py_TYPE(value) == self.list_type {
                    let list = value as *mut ffi::PyListObject;
                    let length = (*(list as *mut ffi::PyVarObject)).ob_size;
                    for index in 0..length {
                        let item = *(*list).ob_item.offset(index);
                        if self.is_ast(item) {
                            stack.push(item);
                        }
                    }
                } else if self.is_ast(value) {
                    stack.push(value);
                }
                Ok(())
            })?;
        }
        stack[start..].reverse();
        Ok(())
    }

    /// Every node under `root`, `root` included, in depth-first pre-order.
    fn pre_order(&self, root: *mut PyObject) -> PyResult<Vec<*mut PyObject>> {
        let mut result = Vec::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            result.push(node);
            self.push_children(node, &mut stack)?;
        }
        Ok(result)
    }
}

fn type_pointers(kinds: &Bound<'_, PyTuple>) -> PyResult<Vec<*mut PyTypeObject>> {
    kinds
        .iter()
        .map(|kind| Ok(kind.cast_into::<PyType>()?.as_type_ptr()))
        .collect()
}

/// Return every node under `node` (`node` included) whose exact type is in
/// `kinds`, in depth-first pre-order. Unlike `walk_frontier`, matches are
/// descended into.
#[pyfunction]
pub fn walk_of_types<'py>(
    py: Python<'py>,
    node: Bound<'py, PyAny>,
    kinds: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let kinds = type_pointers(&kinds)?;
    let walk = Walk::new(py)?;
    let mut result = Vec::new();
    let mut stack = vec![node.as_ptr()];
    while let Some(current) = stack.pop() {
        if kinds.contains(&unsafe { ffi::Py_TYPE(current) }) {
            result.push(current);
        }
        walk.push_children(current, &mut stack)?;
    }
    vec_into_pylist(py, &result)
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
    let mut stack: Vec<(*mut PyObject, usize)> = vec![(node.as_ptr(), 0)];
    let mut children = Vec::new();
    while let Some((current, inherited)) = stack.pop() {
        let info = unsafe { &*type_info(py, ffi::Py_TYPE(current)) };
        let mut own = inherited;
        if info.locations != 0 {
            let dict = get_instance_dict_fast(current);
            let mut values: [Py<PyAny>; 4] =
                locations[inherited].each_ref().map(|v| v.clone_ref(py));
            for (slot, (flag, name, none_is_missing)) in names.iter().enumerate() {
                if info.locations & flag == 0 {
                    continue;
                }
                let found = match dict {
                    Some(dict) => unsafe { ffi::PyDict_GetItemWithError(dict, name.as_ptr()) },
                    None => std::ptr::null_mut(),
                };
                if found.is_null() && unsafe { !ffi::PyErr_Occurred().is_null() } {
                    return Err(PyErr::fetch(py));
                }
                let missing =
                    found.is_null() || (*none_is_missing && unsafe { found == ffi::Py_None() });
                if missing {
                    unsafe { Bound::from_borrowed_ptr(py, current) }
                        .setattr(name, values[slot].bind(py))?;
                } else {
                    values[slot] = unsafe { Bound::from_borrowed_ptr(py, found) }.unbind();
                }
            }
            locations.push(values);
            own = locations.len() - 1;
        }
        children.clear();
        walk.push_children(current, &mut children)?;
        stack.extend(children.iter().map(|&child| (child, own)));
    }
    Ok(node)
}

#[inline(always)]
fn mix(hash: u64, value: u64) -> u64 {
    (hash.rotate_left(5) ^ value).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// A structural hash per node, keyed by `id(node)`. Two subtrees that would
/// give the same `ast.dump` get the same hash; a leaf value's type is mixed
/// in, so `1`, `1.0` and `True` (equal hashes in Python) stay apart.
#[pyfunction]
pub fn subtree_hashes<'py>(
    py: Python<'py>,
    node: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let walk = Walk::new(py)?;
    let nodes = walk.pre_order(node.as_ptr())?;
    let mut hashes: PointerMap<u64> = PointerMap::default();
    hashes.reserve(nodes.len());
    let leaf = |value: *mut PyObject, hashes: &PointerMap<u64>| -> PyResult<u64> {
        if walk.is_ast(value) {
            return Ok(hashes[&(value as usize)]);
        }
        let hashed = unsafe { ffi::PyObject_Hash(value) };
        if hashed == -1 && unsafe { !ffi::PyErr_Occurred().is_null() } {
            return Err(PyErr::fetch(py));
        }
        Ok(mix(unsafe { ffi::Py_TYPE(value) } as u64, hashed as u64))
    };
    for &current in nodes.iter().rev() {
        if hashes.contains_key(&(current as usize)) {
            continue;
        }
        let kind = unsafe { ffi::Py_TYPE(current) };
        let info = unsafe { &*type_info(py, kind) };
        let mut hash = mix(0, kind as u64);
        unsafe {
            for_each_field(py, current, info, |index, value| {
                let value_hash = if ffi::Py_TYPE(value) == walk.list_type {
                    let list = value as *mut ffi::PyListObject;
                    let length = (*(list as *mut ffi::PyVarObject)).ob_size;
                    let mut items = mix(u64::MAX, length as u64);
                    for item in 0..length {
                        items = mix(items, leaf(*(*list).ob_item.offset(item), &hashes)?);
                    }
                    items
                } else {
                    leaf(value, &hashes)?
                };
                hash = mix(mix(hash, index as u64), value_hash);
                Ok(())
            })?;
        }
        hashes.insert(current as usize, hash);
    }
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

impl Walk<'_> {
    /// Push the AST children of `parent` onto `stack` so they pop in
    /// `ast.iter_child_nodes` order, each with the slot it was read from.
    fn push_child_edges(&self, parent: *mut PyObject, stack: &mut Vec<Edge>) -> PyResult<()> {
        let info = unsafe { &*type_info(self.py, ffi::Py_TYPE(parent)) };
        let start = stack.len();
        unsafe {
            for_each_field(self.py, parent, info, |index, value| {
                let key = info.fields[index].as_ptr();
                if ffi::Py_TYPE(value) == self.list_type {
                    let list = value as *mut ffi::PyListObject;
                    let length = (*(list as *mut ffi::PyVarObject)).ob_size;
                    for item in 0..length {
                        let child = *(*list).ob_item.offset(item);
                        if self.is_ast(child) {
                            stack.push(Edge {
                                node: child,
                                parent,
                                key,
                                index: item,
                            });
                        }
                    }
                } else if self.is_ast(value) {
                    stack.push(Edge {
                        node: value,
                        parent,
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
    let kinds = type_pointers(kinds)?;
    let walk = Walk::new(py)?;
    let mut result = Vec::new();
    let mut stack = Vec::new();
    walk.push_child_edges(node.as_ptr(), &mut stack)?;
    while let Some(edge) = stack.pop() {
        if kinds.contains(&unsafe { ffi::Py_TYPE(edge.node) }) {
            result.push(edge);
        } else {
            walk.push_child_edges(edge.node, &mut stack)?;
        }
    }
    Ok(result)
}
