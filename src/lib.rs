use pyo3::types::{PyList, PyModule, PyTuple};
use pyo3::{intern, prelude::*};

fn walk_node<'py>(
    node: Bound<'py, PyAny>,
    field_names: Bound<'py, PyTuple>,
    result_list: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    result_list.push(node.clone());

    // Recursively walk through child nodes
    for field in field_names {
        if let Ok(Some(child)) = node.getattr_opt(unsafe { field.cast_unchecked() }) {
            if child.is_exact_instance_of::<PyList>() {
                for item in unsafe { child.cast_unchecked::<PyList>() } {
                    if let Ok(Some(subfields)) = item.getattr_opt(intern!(item.py(), "_fields")) {
                        walk_node(
                            item,
                            unsafe { subfields.cast_into_unchecked::<PyTuple>() },
                            result_list,
                        )?;
                    }
                }
            } else if let Ok(Some(subfields)) = child.getattr_opt(intern!(child.py(), "_fields")) {
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
