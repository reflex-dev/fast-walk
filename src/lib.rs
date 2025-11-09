mod pydict;

use std::cell::RefCell;

use pyo3::ffi::{self, Py_ssize_t, PyDictObject, PyObject, PyTypeObject};
use pyo3::types::{PyList, PyModule, PyType};
use pyo3::{PyTypeInfo, prelude::*};

pub struct DictValuesIter {
    entries: *const pydict::PyDictUnicodeEntry,
    current: usize,
    end: usize,
}

impl DictValuesIter {
    /// Creates a new iterator over dictionary values
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `obj` is a valid pointer to a `PyDictObject`
    /// - The dictionary remains valid for the lifetime of the iterator
    /// - The dictionary is not modified while iterating
    pub unsafe fn new(obj: *mut PyDictObject) -> Self {
        unsafe {
            let dict = &*obj;
            let keys = &*dict.ma_keys.cast::<pydict::PyDictKeysObject>();
            let entries = keys.unicode_entries();
            let n = keys.dk_nentries as usize;

            Self {
                entries,
                current: 0,
                end: n,
            }
        }
    }
}

impl Iterator for DictValuesIter {
    type Item = *mut PyObject;

    fn next(&mut self) -> Option<Self::Item> {
        // Skip null entries until we find a valid one or reach the end
        while self.current < self.end {
            let entry = &unsafe { *self.entries.add(self.current) };
            self.current += 1;

            if !entry.me_value.is_null() {
                return Some(entry.me_value);
            }
        }

        None
    }
}

fn get_instance_dict_fast(obj: *mut PyObject) -> Option<*mut PyObject> {
    unsafe {
        let dict_ptr = (*obj).ob_type.as_ref()?.tp_dictoffset;

        if dict_ptr != 0 {
            let dict_ptr_addr = (obj as *mut u8).offset(dict_ptr) as *mut *mut ffi::PyObject;
            let dict = *dict_ptr_addr;

            if !dict.is_null() {
                return Some(dict);
            }
        }
        None
    }
}

unsafe fn is_subtype(
    subtype: *mut pyo3::ffi::PyTypeObject,
    base: *mut pyo3::ffi::PyTypeObject,
) -> bool {
    // Walk up the inheritance chain via tp_base, max 3 jumps
    let mut current = subtype;
    for _ in 0..3 {
        current = unsafe { (*current).tp_base };
        if current.is_null() {
            return false;
        }
        if current == base {
            return true;
        }
    }

    false
}

fn isinstance_of_ast(obj: *mut PyObject, base_ast_type: &*mut PyTypeObject) -> bool {
    unsafe { is_subtype(ffi::Py_TYPE(obj), *base_ast_type) }
}

fn is_list(obj: *mut PyObject, py_list_type: *mut PyTypeObject) -> bool {
    unsafe { ffi::Py_TYPE(obj) == py_list_type }
}

fn get_length_of_list(obj: *mut PyObject) -> Py_ssize_t {
    unsafe { ffi::PyList_GET_SIZE(obj) }
}

fn get_item_of_list(obj: *mut PyObject, index: Py_ssize_t) -> *mut PyObject {
    unsafe { ffi::PyList_GET_ITEM(obj, index) }
}

fn walk_node(
    node: *mut PyObject,
    base_ast_type: &*mut PyTypeObject,
    py_list_type: *mut PyTypeObject,
    result_list: &mut Vec<*mut PyObject>,
) -> PyResult<()> {
    result_list.push(node);

    // Recursively walk through child nodes
    let Some(dict) = get_instance_dict_fast(node) else {
        return Ok(());
    };
    for item_ptr in unsafe { DictValuesIter::new(dict.cast::<PyDictObject>()) } {
        if isinstance_of_ast(item_ptr, base_ast_type) {
            walk_node(item_ptr, base_ast_type, py_list_type, result_list)?;
        } else if is_list(item_ptr, py_list_type) {
            let length = get_length_of_list(item_ptr);
            for i in 0..length {
                let item_ptr = get_item_of_list(item_ptr, i);
                if isinstance_of_ast(item_ptr, base_ast_type) {
                    walk_node(item_ptr, base_ast_type, py_list_type, result_list)?;
                }
            }
        }
    }

    Ok(())
}

thread_local! {
    static BASE_AST_TYPE: RefCell<Option<*mut PyTypeObject>> = const { RefCell::new(None) };
}

#[inline(never)]
fn get_base_ast_type<'py>(py: Python<'py>) -> PyResult<*mut PyTypeObject> {
    let ast_module = py.import("ast")?;
    let ast_class = ast_module.getattr("AST")?.cast_into::<PyType>()?;

    Ok(ast_class.as_type_ptr())
}

#[pyfunction]
fn walk<'py>(py: Python, node: Bound<'py, PyAny>) -> PyResult<Py<PyList>> {
    let mut result_list = Vec::new();

    // Initialize if needed (separate step with mutable borrow)
    BASE_AST_TYPE.with(|cache| {
        if cache.borrow().is_none() {
            *cache.borrow_mut() = Some(get_base_ast_type(py)?);
        }
        Ok::<(), PyErr>(())
    })?;

    // Now use immutable borrow for the actual work
    BASE_AST_TYPE.with(|cache| {
        let cache_ref = cache.borrow();
        let base_ast_type = cache_ref.as_ref().unwrap();
        walk_node(
            node.as_ptr(),
            base_ast_type,
            PyList::type_object_raw(py),
            &mut result_list,
        )?;
        Ok::<(), PyErr>(())
    })?;

    Ok(PyList::new(
        py,
        result_list
            .into_iter()
            .map(|ptr| unsafe { Bound::from_borrowed_ptr(py, ptr) }),
    )?
    .into())
}

#[pymodule]
fn fast_walk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(walk, m)?)?;
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
            let values = unsafe { DictValuesIter::new(dict_ptr) }.collect::<Vec<_>>();
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
            let values = unsafe { DictValuesIter::new(dict_ptr) }.collect::<Vec<_>>();
            assert_eq!(values.len(), 3);
        });
    }
}
