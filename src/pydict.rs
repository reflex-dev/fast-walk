// Rust representation of PyDictKeysObject from CPython
// Based on: https://github.com/python/cpython/blob/main/Include/internal/pycore_dict.h

use pyo3::ffi::{Py_ssize_t, PyObject};

// PyDictUnicodeEntry - used for DICT_KEYS_UNICODE
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PyDictUnicodeEntry {
    pub me_key: *mut PyObject,
    pub me_value: *mut PyObject,
}

// Union for dk_indices - the index table can be different sizes
#[repr(C)]
pub union DictIndices {
    pub as_1: [i8; 8],
    pub as_2: [i16; 4],
    pub as_4: [i32; 2],
    pub as_8: [i64; 1],
}

// Main PyDictKeysObject structure
// This matches the layout from Include/internal/pycore_dict.h
#[repr(C)]
pub struct PyDictKeysObject {
    /// Reference count
    pub dk_refcnt: Py_ssize_t,

    /// Log2 of the size of the hash table (dk_indices). Must be a power of 2.
    pub dk_log2_size: u8,

    /// Log2 of the size of the hash table (dk_indices) in bytes
    pub dk_log2_index_bytes: u8,

    /// Kind of keys (General, Unicode, or Split)
    pub dk_kind: u8,

    /// Version number - reset to 0 by any modification to keys
    pub dk_version: u32,

    /// Number of usable entries in dk_entries
    pub dk_usable: Py_ssize_t,

    /// Number of used entries in dk_entries
    pub dk_nentries: Py_ssize_t,

    /// Actual hash table of dk_size entries.
    /// It holds indices in dk_entries, or DKIX_EMPTY(-1) or DKIX_DUMMY(-2).
    ///
    /// The size in bytes of an index depends on dk_size:
    /// - 1 byte if dk_size <= 0xff (i8)
    /// - 2 bytes if dk_size <= 0xffff (i16)
    /// - 4 bytes if dk_size <= 0xffffffff (i32)
    /// - 8 bytes otherwise (i64)
    ///
    /// This is a flexible array member in C, but in Rust we represent it
    /// as a union showing the different interpretations.
    /// In actual usage, this would be followed by additional memory allocated
    /// at runtime containing both the indices array and the entries array.
    pub dk_indices: DictIndices,
    // Note: In the actual C structure, after dk_indices there is dynamically
    // allocated space for:
    // 1. The full indices array (size determined by dk_log2_size and dk_log2_index_bytes)
    // 2. The entries array (PyDictKeyEntry or PyDictUnicodeEntry depending on dk_kind)
    //
    // In Rust, you would typically handle this by:
    // - Allocating the struct with extra space
    // - Using pointer arithmetic to access the variable-length data
    // - Or using a different design pattern like separate allocations
}

impl PyDictKeysObject {
    /// Get a pointer to the start of the entries array
    /// The entries come after the indices array in memory
    ///
    /// This matches CPython's _DK_ENTRIES implementation:
    /// ```c
    /// int8_t *indices = (int8_t*)(dk->dk_indices);
    /// size_t index = (size_t)1 << dk->dk_log2_index_bytes;
    /// return (&indices[index]);
    /// ```
    ///
    /// The calculation is: indices_ptr + (1 << dk_log2_index_bytes)
    /// NOT: indices_ptr + (dk_size * dk_index_bytes)
    ///
    /// This is because dk_log2_index_bytes gives the total size of the indices
    /// array in a single shift operation.
    pub unsafe fn entries_ptr(&self) -> *const u8 {
        let indices_ptr = &self.dk_indices as *const _ as *const i8;
        let index = (1_usize) << self.dk_log2_index_bytes;
        unsafe { indices_ptr.add(index) as *const u8 }
    }

    /// Get a pointer to the entries as PyDictUnicodeEntry
    pub unsafe fn unicode_entries(&self) -> *const PyDictUnicodeEntry {
        unsafe { self.entries_ptr() as *const PyDictUnicodeEntry }
    }
}
