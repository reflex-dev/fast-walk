//! Fast reimplementation of `ast.walk`.
//!
//! The public API exposes two traversal variants:
//!
//! - [`walk_dfs`] — strict depth-first pre-order.
//! - [`walk_unordered`] — faster; same set of nodes, implementation-defined
//!   order. Uses batched prefetching to hide cache-miss latency on the
//!   scattered `PyDictKeysObject` loads that dominate the DFS profile.
//!
//! `ast.walk` makes no ordering guarantee, so `walk_unordered` is a drop-in
//! replacement wherever order doesn't matter.

mod pydict;

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::PyDeprecationWarning;
use pyo3::ffi::{self, PyListObject, PyObject, PyTypeObject};
use pyo3::types::{PyList, PyModule, PyType};
use pyo3::{PyTypeInfo, prelude::*};

/// Open-addressed, direct-mapped lookup from `*mut PyTypeObject` to an
/// AST-classification code. Specialized for the ~130 `ast.AST`
/// subclasses; populated once per thread at first walk and then
/// read-only on the hot path.
///
/// **Value encoding** (chosen to collapse the two hot-path predicates
/// "is this an AST subclass?" and "how many `_fields` does it have?"
/// into one L1 load):
///
/// - `0` — not an AST type (or absent from the table). Used for
///   primitive values like `str`, `int`, `None`, `bytes`, and also for
///   any type we haven't seen, so a missing key is indistinguishable
///   from an explicit "not AST" entry — which is exactly what we want.
/// - `n` for `1 <= n <= 8` — AST type with `n - 1` `_fields`. This
///   lets a single `table.get(t) > 0` check replace the old
///   `issubclass_of_ast` walk over `tp_base`, and `table.get(t) - 1`
///   give the `_fields` count.
///
/// Stdlib AST types top out at 7 `_fields`, so `u8` is more than
/// enough; the table stays at 256 bytes of values + 2 KB of keys = one
/// L1-resident data structure for the whole walk.
///
/// Layout: parallel `keys` and `values` arrays of `SIZE` slots each.
/// Keys are u64 (pointer as integer); empty slot is `key == 0`. Values
/// are u8. Index function: `(ptr >> 4) & (SIZE - 1)` — type objects
/// are allocator-aligned, so shifting right by 4 drops the alignment
/// zeros and gives well-distributed indices across our ~130 live
/// types.
const FIELD_TABLE_SIZE: usize = 256; // power of two, load factor ~0.5
const FIELD_TABLE_MASK: usize = FIELD_TABLE_SIZE - 1;

struct FieldTable {
    keys: [u64; FIELD_TABLE_SIZE],
    values: [u8; FIELD_TABLE_SIZE],
}

impl FieldTable {
    fn new() -> Self {
        Self {
            keys: [0; FIELD_TABLE_SIZE],
            values: [0; FIELD_TABLE_SIZE],
        }
    }

    /// Store an AST type with its `_fields` length. The value written
    /// is `n_fields + 1`; see the struct-level docs for the encoding.
    fn insert_ast(&mut self, ptr: *mut PyTypeObject, n_fields: u8) {
        let key = ptr as u64;
        debug_assert!(key != 0, "null type pointer");
        debug_assert!(n_fields < u8::MAX, "n_fields would overflow encoding");
        let encoded = n_fields + 1;
        let mut idx = ((key >> 4) as usize) & FIELD_TABLE_MASK;
        // Linear probe until we find the key or an empty slot. Load
        // factor 0.5 keeps the expected probe length ≲ 2.
        loop {
            let k = self.keys[idx];
            if k == key {
                self.values[idx] = encoded;
                return;
            }
            if k == 0 {
                self.keys[idx] = key;
                self.values[idx] = encoded;
                return;
            }
            idx = (idx + 1) & FIELD_TABLE_MASK;
        }
    }

    /// Raw encoded value: `0` if not an AST type, `n_fields + 1`
    /// otherwise. Hot path uses this directly so "is AST?" (`> 0`) and
    /// "how many fields?" (`- 1`) share a single load.
    #[inline(always)]
    fn lookup(&self, ptr: *mut PyTypeObject) -> u8 {
        let key = ptr as u64;
        let mut idx = ((key >> 4) as usize) & FIELD_TABLE_MASK;
        loop {
            // SAFETY: `idx` is always masked to `FIELD_TABLE_MASK`, so it
            // stays within the fixed-size array.
            let k = unsafe { *self.keys.get_unchecked(idx) };
            if k == key {
                return unsafe { *self.values.get_unchecked(idx) };
            }
            if k == 0 {
                return 0;
            }
            idx = (idx + 1) & FIELD_TABLE_MASK;
        }
    }
}

/// Reverse iterator over the first `limit` values of a Python dict whose
/// keys are all strings — the layout used by instance `__dict__`s. Reads
/// the `PyDictKeysObject` entry table directly and skips null (deleted)
/// slots.
///
/// The `limit` parameter is how we walk only the `_fields` portion of an
/// AST node's dict: for parsed ASTs, CPython stores keys in the order
/// `_fields ++ _attributes ++ user_added`, so passing `limit = len(_fields)`
/// guarantees we only see syntactic children — no `lineno`/`col_offset`
/// ints and no user-attached `.parent` cycles.
pub struct ReverseDictValuesIter {
    entries: *const pydict::PyDictUnicodeEntry,
    current: usize,
}

impl ReverseDictValuesIter {
    /// # Safety
    ///
    /// - `obj` must be a valid pointer to a `PyDictObject` whose keys are
    ///   all unicode strings (i.e. a combined-unicode dict).
    /// - The dictionary must outlive the iterator and must not be mutated
    ///   while iterating.
    /// - `limit` must not exceed `dk_nentries`; caller is responsible for
    ///   clamping when working from an external count (e.g. `_fields` len).
    pub unsafe fn new(obj: *mut ffi::PyDictObject, limit: usize) -> Self {
        unsafe {
            let dict = &*obj;
            let keys = &*dict.ma_keys.cast::<pydict::PyDictKeysObject>();
            let entries = keys.unicode_entries();
            let n = (keys.dk_nentries as usize).min(limit);
            Self {
                entries,
                current: n,
            }
        }
    }
}

impl Iterator for ReverseDictValuesIter {
    type Item = *mut PyObject;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current > 0 {
            self.current -= 1;
            let entry: &pydict::PyDictUnicodeEntry = &unsafe { *self.entries.add(self.current) };
            if !entry.me_value.is_null() {
                return Some(entry.me_value);
            }
        }
        None
    }
}

/// Return an object's instance `__dict__` pointer via `tp_dictoffset`, or
/// `None` if the type has no dict offset or the slot is null.
fn get_instance_dict_fast(obj: *mut PyObject) -> Option<*mut PyObject> {
    unsafe {
        let dict_offset = (*obj).ob_type.as_ref()?.tp_dictoffset;
        if dict_offset == 0 {
            return None;
        }
        let dict_ptr_addr = (obj as *mut u8).offset(dict_offset) as *mut *mut ffi::PyObject;
        let dict = *dict_ptr_addr;
        if dict.is_null() { None } else { Some(dict) }
    }
}

/// L1 prefetch hint. No-op on non-x86_64 targets — the Python extension
/// builds and runs identically without it, just without the cache-miss
/// hiding that benefits `walk_unordered`.
#[inline(always)]
unsafe fn prefetch_l1(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

/// Resolve the `ma_keys` pointer of a node's instance dict. Used to
/// prefetch the `PyDictKeysObject` — the intermediate reads (object
/// header, type object, dict slot) are hot; only the final `ma_keys`
/// target typically misses cache.
#[inline(always)]
unsafe fn ma_keys_of(node: *mut PyObject) -> Option<*const u8> {
    unsafe {
        let type_ptr = (*node).ob_type;
        if type_ptr.is_null() {
            return None;
        }
        let dict_offset = (*type_ptr).tp_dictoffset;
        if dict_offset == 0 {
            return None;
        }
        let dict_ptr_addr = (node as *const u8).offset(dict_offset) as *const *mut ffi::PyObject;
        let dict = *dict_ptr_addr;
        if dict.is_null() {
            return None;
        }
        let ma_keys = (*(dict as *const ffi::PyDictObject)).ma_keys;
        Some(ma_keys as *const u8)
    }
}

/// Per-node body shared by both traversals: enumerate the `_fields`
/// slots of the node's instance dict and push AST children onto
/// `stack`. For parsed ASTs, CPython stores dict keys in the order
/// `_fields ++ _attributes ++ user_added`, so limiting the scan to the
/// first `len(_fields)` entries skips the `_attributes` ints (lineno/
/// col_offset/...) and any user-attached metadata (including cycle-
/// inducing `.parent` back-references) in a single loop bound.
///
/// Per-value "is this AST?" checks stay on `issubclass_of_ast` rather
/// than the table: the `tp_base == PyBaseObject_Type` early-exit
/// already catches every primitive (str/int/None/...) in one load, and
/// a table probe costs the same on average — substituting one for the
/// other was measured to regress.
#[inline(always)]
unsafe fn process_node(
    current_node: *mut PyObject,
    base_ast_and_expr_type: (*mut PyTypeObject, *mut PyTypeObject),
    py_list_type: *mut PyTypeObject,
    field_table: &FieldTable,
    stack: &mut Vec<*mut PyObject>,
) {
    let type_ptr = unsafe { ffi::Py_TYPE(current_node) };
    let encoded = field_table.lookup(type_ptr);
    // 0 == not an AST type we know about. Shouldn't normally happen
    // (only AST nodes reach the stack) but guards any caller that
    // seeds the walk with a non-AST root.
    if encoded == 0 {
        return;
    }
    let n_fields = (encoded - 1) as usize;
    if n_fields == 0 {
        return;
    }

    let Some(dict) = get_instance_dict_fast(current_node) else {
        return;
    };

    for item_ptr in
        unsafe { ReverseDictValuesIter::new(dict.cast::<ffi::PyDictObject>(), n_fields) }
    {
        let item_type = unsafe { ffi::Py_TYPE(item_ptr) };
        if item_type == py_list_type {
            let list = item_ptr as *mut PyListObject;
            let length = unsafe { (*(list as *mut ffi::PyVarObject)).ob_size };
            let ob_item = unsafe { (*list).ob_item };
            for i in (0..length).rev() {
                let child = unsafe { *ob_item.offset(i) };
                let child_type = unsafe { ffi::Py_TYPE(child) };
                if issubclass_of_ast(child_type, base_ast_and_expr_type) {
                    stack.push(child);
                }
            }
        } else if issubclass_of_ast(item_type, base_ast_and_expr_type) {
            stack.push(item_ptr);
        }
    }
}

/// Check whether `subtype` is a subclass of `ast.AST` within the first two
/// levels of the MRO. Every stdlib AST node is `Concrete -> ast.expr/stmt
/// -> ast.AST` or `Concrete -> ast.AST`, so two hops suffice.
///
/// Performance notes baked in here:
/// - Early-exit on `first_supertype == PyBaseObject_Type`: primitives like
///   `str`, `NoneType`, `float`, `bytes` inherit directly from `object`,
///   AST subclasses never do. This skips the scattered second `tp_base`
///   load on ~40% of items (non-AST values in `_fields` slots).
fn issubclass_of_ast(
    subtype: *mut PyTypeObject,
    base_ast_and_expr_type: (*mut PyTypeObject, *mut PyTypeObject),
) -> bool {
    let first_supertype = unsafe { (*subtype).tp_base };
    if first_supertype.is_null() {
        return false;
    }
    let py_object_type = &raw mut ffi::PyBaseObject_Type;
    if first_supertype == py_object_type {
        return false;
    }
    let (base_ast_type, base_expr_type) = base_ast_and_expr_type;
    if first_supertype == base_ast_type {
        return true;
    }
    let second_supertype = unsafe { (*first_supertype).tp_base };
    second_supertype == base_ast_type || second_supertype == base_expr_type
}

/// Strict depth-first pre-order traversal.
fn walk_node_dfs(
    node: *mut PyObject,
    base_ast_and_expr_type: (*mut PyTypeObject, *mut PyTypeObject),
    py_list_type: *mut PyTypeObject,
    field_table: &FieldTable,
    result_list: &mut Vec<*mut PyObject>,
) -> PyResult<()> {
    let mut stack = vec![node];

    while let Some(current_node) = stack.pop() {
        result_list.push(current_node);
        unsafe {
            process_node(
                current_node,
                base_ast_and_expr_type,
                py_list_type,
                field_table,
                &mut stack,
            );
        }
    }

    Ok(())
}

/// Batched traversal with prefetching.
///
/// Drains up to `BATCH` nodes from the stack, issues an L1 prefetch for
/// each node's `PyDictKeysObject` in a tight loop, then processes each
/// node in turn. Prefetches issued in parallel hide the latency of the
/// scattered dict-keys loads that dominate the DFS profile (~20% of
/// function time). Visits the same set of nodes as `walk_node_dfs` but
/// not in strict DFS order.
fn walk_node_unordered(
    node: *mut PyObject,
    base_ast_and_expr_type: (*mut PyTypeObject, *mut PyTypeObject),
    py_list_type: *mut PyTypeObject,
    field_table: &FieldTable,
    result_list: &mut Vec<*mut PyObject>,
) -> PyResult<()> {
    const BATCH: usize = 4;
    let mut stack = vec![node];
    let mut batch: [*mut PyObject; BATCH] = [std::ptr::null_mut(); BATCH];

    while !stack.is_empty() {
        let take = stack.len().min(BATCH);
        for slot in batch.iter_mut().take(take) {
            *slot = stack.pop().unwrap();
        }

        for &node_ptr in batch.iter().take(take) {
            if let Some(p) = unsafe { ma_keys_of(node_ptr) } {
                unsafe { prefetch_l1(p) };
            }
        }

        for &current in batch.iter().take(take) {
            result_list.push(current);
            unsafe {
                process_node(
                    current,
                    base_ast_and_expr_type,
                    py_list_type,
                    field_table,
                    &mut stack,
                );
            }
        }
    }

    Ok(())
}

thread_local! {
    static BASE_AST_TYPE_AND_EXPR: Cell<Option<(*mut PyTypeObject, *mut PyTypeObject)>> =
        const { Cell::new(None) };
    static AST_FIELD_TABLE: RefCell<Option<Box<FieldTable>>> =
        const { RefCell::new(None) };
}

/// Resolve `ast.AST` and `ast.expr` to their raw type pointers. Kept out
/// of the hot path so the importlib work doesn't inline into the walk.
#[inline(never)]
fn get_base_ast_type(py: Python<'_>) -> PyResult<(*mut PyTypeObject, *mut PyTypeObject)> {
    let ast_module = py.import("ast")?;
    let ast_class = ast_module.getattr("AST")?.cast_into::<PyType>()?;
    let expr_class = ast_module.getattr("expr")?.cast_into::<PyType>()?;
    Ok((ast_class.as_type_ptr(), expr_class.as_type_ptr()))
}

#[inline(always)]
fn resolve_base_types(py: Python) -> PyResult<(*mut PyTypeObject, *mut PyTypeObject)> {
    BASE_AST_TYPE_AND_EXPR.with(|cache| match cache.get() {
        Some(v) => Ok(v),
        None => {
            let v = get_base_ast_type(py)?;
            cache.set(Some(v));
            Ok(v)
        }
    })
}

/// Walk every subclass of `ast.AST` at first-use and record each type's
/// `len(_fields)`. The resulting direct-mapped table answers the hot-loop
/// lookup in one L1 load per node — no Python calls, no `_attributes`
/// scanning, no hashing.
#[inline(never)]
fn prebuild_field_table(py: Python<'_>) -> PyResult<Box<FieldTable>> {
    let ast_module = py.import("ast")?;
    let ast_class = ast_module.getattr("AST")?.cast_into::<PyType>()?;

    let mut table = Box::new(FieldTable::new());
    let mut stack: Vec<Bound<'_, PyType>> = vec![ast_class];
    while let Some(t) = stack.pop() {
        let n_fields = t
            .getattr("_fields")
            .ok()
            .and_then(|f| f.len().ok())
            .unwrap_or(0);
        // `_fields` tuples in the stdlib top out at 7 entries. Saturate
        // for safety so a rogue subclass with a huge `_fields` tuple
        // can't break the u8 encoding.
        // Saturate to u8::MAX - 1: the table reserves n_fields+1 as the
        // encoded value so we need headroom below 255.
        let n = n_fields.min((u8::MAX - 1) as usize) as u8;
        table.insert_ast(t.as_type_ptr(), n);
        let subs = t.call_method0("__subclasses__")?;
        for sub in subs.try_iter()? {
            stack.push(sub?.cast_into::<PyType>()?);
        }
    }
    Ok(table)
}

/// Run `body` with a `&FieldTable` pinning the prebuilt `_fields`-length
/// cache. The table is built on first use per thread and reused for all
/// subsequent walks on that thread.
#[inline(always)]
fn with_field_table<R>(
    py: Python<'_>,
    body: impl FnOnce(&FieldTable) -> PyResult<R>,
) -> PyResult<R> {
    AST_FIELD_TABLE.with(|cache| {
        let mut borrow = cache.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(prebuild_field_table(py)?);
        }
        body(borrow.as_ref().unwrap())
    })
}

/// Construct a Python list from a Vec of owned-reference pointers, going
/// directly through the FFI `PyList_New` + `PyList_SET_ITEM` path. Avoids
/// the per-item `Bound` allocation in `PyList::new(iter)`.
fn vec_into_pylist<'py>(py: Python<'py>, items: &[*mut PyObject]) -> PyResult<Bound<'py, PyAny>> {
    let len = items.len() as ffi::Py_ssize_t;
    unsafe {
        let list_ptr = ffi::PyList_New(len);
        if list_ptr.is_null() {
            return Err(PyErr::fetch(py));
        }
        let ob_item = (*(list_ptr as *mut ffi::PyListObject)).ob_item;
        for (i, &ptr) in items.iter().enumerate() {
            ffi::Py_INCREF(ptr);
            *ob_item.add(i) = ptr;
        }
        Ok(Bound::from_owned_ptr(py, list_ptr))
    }
}

/// Walk the AST rooted at `node` in strict depth-first pre-order and
/// return every descendant (including `node` itself) as a list.
///
/// Semantically equivalent to `list(ast.walk(node))` but ~100× faster.
/// Use `walk_unordered` if traversal order doesn't matter — it's faster
/// still.
#[pyfunction]
fn walk_dfs<'py>(py: Python<'py>, node: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let base = resolve_base_types(py)?;
    let py_list_type = PyList::type_object_raw(py);
    let node_ptr = node.as_ptr();
    with_field_table(py, |table| {
        let mut result_list = Vec::new();
        walk_node_dfs(node_ptr, base, py_list_type, table, &mut result_list)?;
        vec_into_pylist(py, &result_list)
    })
}

/// Walk the AST rooted at `node` and return every descendant (including
/// `node` itself) as a list, in an implementation-defined order.
///
/// The set of returned nodes is identical to `walk_dfs` and to
/// `ast.walk`; only the order differs. Use this whenever order is not
/// significant — batched prefetching makes it ~25% faster than
/// `walk_dfs`.
#[pyfunction]
fn walk_unordered<'py>(py: Python<'py>, node: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let base = resolve_base_types(py)?;
    let py_list_type = PyList::type_object_raw(py);
    let node_ptr = node.as_ptr();
    with_field_table(py, |table| {
        let mut result_list = Vec::new();
        walk_node_unordered(node_ptr, base, py_list_type, table, &mut result_list)?;
        vec_into_pylist(py, &result_list)
    })
}

static DEPRECATED_WALK_WARNED: AtomicBool = AtomicBool::new(false);

/// Deprecated. Use `walk_dfs` for explicit depth-first order or
/// `walk_unordered` for the faster order-agnostic variant.
///
/// Emits a `DeprecationWarning` once per process on the first call, then
/// delegates to `walk_dfs`. The warning is gated behind an atomic flag
/// so repeated calls don't pay the cost of the `warnings.warn` machinery.
#[pyfunction]
fn walk<'py>(py: Python<'py>, node: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if !DEPRECATED_WALK_WARNED.swap(true, Ordering::Relaxed) {
        let category = py.get_type::<PyDeprecationWarning>();
        PyErr::warn(
            py,
            &category,
            c"fast_walk.walk is deprecated; use walk_dfs for strict depth-first order or walk_unordered for the faster order-agnostic variant",
            1,
        )?;
    }
    walk_dfs(py, node)
}

/// Benchmarking-only. Traverse the AST and return the node count without
/// materializing a result list. Isolates traversal cost from list-build
/// cost for profiling deltas.
#[pyfunction]
fn _walk_count<'py>(py: Python, node: Bound<'py, PyAny>) -> PyResult<usize> {
    let base = resolve_base_types(py)?;
    let py_list_type = PyList::type_object_raw(py);
    let node_ptr = node.as_ptr();
    with_field_table(py, |table| {
        let mut result_list = Vec::new();
        walk_node_dfs(node_ptr, base, py_list_type, table, &mut result_list)?;
        Ok(result_list.len())
    })
}

#[pymodule]
fn fast_walk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(walk, m)?)?;
    m.add_function(wrap_pyfunction!(walk_dfs, m)?)?;
    m.add_function(wrap_pyfunction!(walk_unordered, m)?)?;
    m.add_function(wrap_pyfunction!(_walk_count, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyDict;

    use super::*;

    #[test]
    fn test_empty_dict_no_values() {
        Python::initialize();

        Python::attach(|py| {
            let dict = PyDict::new(py);
            let dict_ptr = dict.as_ptr() as *mut pyo3::ffi::PyDictObject;
            let values =
                unsafe { ReverseDictValuesIter::new(dict_ptr, usize::MAX) }.collect::<Vec<_>>();
            assert_eq!(values.len(), 0);
        });
    }

    #[test]
    fn test_string_keys_dict_values() {
        Python::initialize();

        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("a", 1).unwrap();
            dict.set_item("b", 2).unwrap();
            dict.set_item("c", 3).unwrap();

            let dict_ptr = dict.as_ptr() as *mut pyo3::ffi::PyDictObject;
            let values =
                unsafe { ReverseDictValuesIter::new(dict_ptr, usize::MAX) }.collect::<Vec<_>>();
            assert_eq!(values.len(), 3);
        });
    }
}
