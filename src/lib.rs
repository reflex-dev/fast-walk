use std::cell::RefCell;

use fastset::Set;
use pyo3::ffi::{self, Py_ssize_t, PyObject, PyTypeObject};
use pyo3::types::{PyList, PyModule, PyString, PyType};
use pyo3::{PyTypeInfo, prelude::*};

pub struct BorrowedDictIter {
    dict: *mut PyObject,
    ppos: ffi::Py_ssize_t,
    len: ffi::Py_ssize_t,
}

impl Iterator for BorrowedDictIter {
    type Item = *mut PyObject;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let mut value: *mut PyObject = std::ptr::null_mut();

        // Safety: self.dict lives sufficiently long that the pointer is not dangling
        if unsafe { ffi::PyDict_Next(self.dict, &mut self.ppos, std::ptr::null_mut(), &mut value) }
            != 0
        {
            self.len -= 1;
            Some(value)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }
}

impl ExactSizeIterator for BorrowedDictIter {
    fn len(&self) -> usize {
        self.len as usize
    }
}

fn dict_len(dict: *mut PyObject) -> Py_ssize_t {
    unsafe { ffi::PyDict_Size(dict) }
}

impl BorrowedDictIter {
    pub fn new(dict: *mut PyObject) -> Self {
        let len = dict_len(dict);
        BorrowedDictIter { dict, ppos: 0, len }
    }
}

fn get_instance_dict_fast(obj: *mut PyObject) -> Option<*mut PyObject> {
    unsafe {
        let dict_ptr = (*obj).ob_type.as_ref()?.tp_dictoffset;

        if dict_ptr != 0 {
            let dict_ptr_addr =
                (obj as *mut u8).offset(dict_ptr as isize) as *mut *mut ffi::PyObject;
            let dict = *dict_ptr_addr;

            if !dict.is_null() {
                return Some(dict);
            }
        }
        None
    }
}

fn isinstance_of_ast(obj: *mut PyObject, all_ast_classes: &Set) -> bool {
    let el = unsafe { ffi::Py_TYPE(obj) };
    all_ast_classes.contains(&(el as usize))
}

fn isinstance_of_list(obj: *mut PyObject, py_list_type: *mut PyTypeObject) -> bool {
    unsafe { ffi::Py_TYPE(obj) == py_list_type }
}

fn get_length_of_list(obj: *mut PyObject) -> Py_ssize_t {
    unsafe { ffi::PyList_GET_SIZE(obj) }
}

fn get_item_of_list(obj: *mut PyObject, index: Py_ssize_t) -> *mut PyObject {
    unsafe { ffi::PyList_GET_ITEM(obj, index) }
}

fn walk_node<'py>(
    node: *mut PyObject,
    all_ast_classes: &Set,
    py_list_type: *mut PyTypeObject,
    result_list: &mut Vec<*mut PyObject>,
) -> PyResult<()> {
    result_list.push(node);

    // Recursively walk through child nodes
    let Some(dict) = get_instance_dict_fast(node) else {
        return Ok(());
    };
    let values = BorrowedDictIter::new(dict);
    for item_ptr in values {
        if isinstance_of_ast(item_ptr, all_ast_classes) {
            walk_node(item_ptr, all_ast_classes, py_list_type, result_list)?;
        } else if isinstance_of_list(item_ptr, py_list_type) {
            let length = get_length_of_list(item_ptr);
            for i in 0..length {
                let item_ptr = get_item_of_list(item_ptr, i);
                if isinstance_of_ast(item_ptr, all_ast_classes) {
                    walk_node(item_ptr, all_ast_classes, py_list_type, result_list)?;
                }
            }
        }
    }

    Ok(())
}

thread_local! {
    static AST_CLASSES: RefCell<Option<Set>> = RefCell::new(None);
}

#[inline(never)]
fn compute_ast_classes<'py>(py: Python<'py>) -> PyResult<Set> {
    let mut classes = Vec::new();
    let ast_module = py.import("ast")?;
    let ast_class = ast_module.getattr("AST")?.cast_into::<PyType>()?;

    for field in ast_module.dir()? {
        let class = ast_module.getattr(field.cast_exact::<PyString>()?)?;
        if let Ok(class) = class.cast_into::<PyType>() {
            if class.is_subclass(ast_class.cast::<PyType>()?)? {
                classes.push(class.as_type_ptr() as usize);
            }
        }
    }

    Ok(Set::from(classes))
}

#[pyfunction]
fn walk<'py>(py: Python, node: Bound<'py, PyAny>) -> PyResult<Py<PyList>> {
    let mut result_list = Vec::new();

    // Initialize if needed (separate step with mutable borrow)
    AST_CLASSES.with(|cache| {
        if cache.borrow().is_none() {
            *cache.borrow_mut() = Some(compute_ast_classes(py)?);
        }
        Ok::<(), PyErr>(())
    })?;

    // Now use immutable borrow for the actual work
    AST_CLASSES.with(|cache| {
        let cache_ref = cache.borrow();
        let all_ast_classes = cache_ref.as_ref().unwrap();
        walk_node(
            node.as_ptr(),
            all_ast_classes,
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
