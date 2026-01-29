// Declare modules
mod bpe_encode;
mod bpe_train;

// Re-export types so users can import them directly from the crate root
pub use bpe_encode::{AllowedSpecialTokens, BpeEncode};
pub use bpe_train::BpeTrainer;

// Only compile Python bindings when building with --features python
#[cfg(feature = "python")]
use pyo3::prelude::*;

// Python wrapper class - visible to Python as a class
#[cfg(feature = "python")]
#[pyclass]
struct PyBpeEncode {
    inner: BpeEncode,  // Wraps the actual Rust implementation
}

// Python wrapper for BPE trainer
#[cfg(feature = "python")]
#[pyclass]
struct PyBpeTrainer {
    inner: BpeTrainer,
}

// Methods that Python can call on PyBpeEncode instances
#[cfg(feature = "python")]
#[pymethods]
impl PyBpeEncode {
    // Constructor: called from Python as PyBpeEncode()
    #[new]
    fn new() -> Self {
        PyBpeEncode {
            inner: BpeEncode::new(),  // Create the inner Rust encoder
        }
    }

    // Load a BPE model file - converts Rust errors to Python RuntimeError
    fn load(&mut self, model_file: &str) -> PyResult<()> {
        self.inner
            .load(model_file)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    // Encode text to token IDs - returns Python list[int]
    fn encode(&self, text: &str, allowed_special_tokens: &str) -> PyResult<Vec<u32>> {
        // Convert Python string ("none"/"all"/"none_raise") to Rust enum
        let policy = AllowedSpecialTokens::from_str(allowed_special_tokens)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        // Call inner encoder and convert errors to Python exceptions
        self.inner
            .encode(text, policy)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    // Decode token IDs back to text - Python list[int] -> str
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner
            .decode(&ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
}

// Methods for PyBpeTrainer
#[cfg(feature = "python")]
#[pymethods]
impl PyBpeTrainer {
    #[new]
    fn new(regex_pattern: Option<String>) -> Self {
        PyBpeTrainer {
            inner: BpeTrainer::new(regex_pattern),
        }
    }
    
    fn train(
        &mut self,
        input_file_path: &str,
        vocab_size: usize,
        special_tokens: Vec<String>,
        boundary_split_token: &str,
        num_threads: usize,
        verbose: bool,
    ) -> PyResult<()> {
        self.inner
            .train(input_file_path, vocab_size, special_tokens, boundary_split_token, num_threads, verbose)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
    
    fn register_special_tokens(&mut self, special_tokens: std::collections::HashMap<String, u32>) -> PyResult<()> {
        self.inner.register_special_tokens(special_tokens);
        Ok(())
    }
    
    fn save(&self, file_prefix: &str) -> PyResult<()> {
        self.inner
            .save(file_prefix)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
}

// Define the Python module - the function name becomes the module name
#[cfg(feature = "python")]
#[pymodule]
fn bpe_encode_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBpeEncode>()?;  // Register PyBpeEncode class in the module
    m.add_class::<PyBpeTrainer>()?;  // Register PyBpeTrainer class in the module
    Ok(())
}

