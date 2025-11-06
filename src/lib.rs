use pyo3::types::{PyList, PyModule, PyTuple};
use pyo3::{intern, prelude::*};

fn getattr<'py>(
    obj: &Bound<'py, PyAny>,
    attr_name: &Bound<'py, PyAny>,
) -> Option<Bound<'py, PyAny>> {
    let py = obj.py();

    let mut resp_ptr: *mut pyo3::ffi::PyObject = std::ptr::null_mut();
    let attr_ptr = unsafe {
        pyo3::ffi::PyObject_GetOptionalAttr(obj.as_ptr(), attr_name.as_ptr(), &mut resp_ptr)
    };

    if attr_ptr == 1 {
        Some(unsafe { Bound::from_owned_ptr(py, resp_ptr) })
    } else {
        None
    }
}

fn walk_node<'py>(
    node: Bound<'py, PyAny>,
    field_names: Bound<'py, PyTuple>,
    result_list: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    result_list.push(node.clone());

    // Recursively walk through child nodes
    for field in field_names {
        if let Some(child) = getattr(&node, &field) {
            if child.is_exact_instance_of::<PyList>() {
                for item in unsafe { child.cast_unchecked::<PyList>() } {
                    if let Some(subfields) = getattr(&item, intern!(item.py(), "_fields")) {
                        walk_node(
                            item,
                            unsafe { subfields.cast_into_unchecked::<PyTuple>() },
                            result_list,
                        )?;
                    }
                }
            } else if let Some(subfields) = getattr(&child, intern!(child.py(), "_fields")) {
                walk_node(
                    child,
                    unsafe { subfields.cast_into_unchecked::<PyTuple>() },
                    result_list,
                )?;
            }
        }
    }

    Ok(())
}

#[pyfunction]
fn walk<'py>(py: Python, node: Bound<'py, PyAny>) -> PyResult<Py<PyList>> {
    let mut result_list = Vec::new();
    let fields = node.getattr(intern!(py, "_fields"))?;
    let fields = unsafe { fields.cast_into_unchecked::<PyTuple>() };
    walk_node(node, fields, &mut result_list)?;
    Ok(PyList::new(py, result_list)?.into())
}

#[pymodule]
fn fast_walk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(walk, m)?)?;
    Ok(())
}
