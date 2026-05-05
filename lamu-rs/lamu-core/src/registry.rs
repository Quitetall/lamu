//! Model registry — auto-discover GGUFs, write/read YAML.
//!
//! Port of `lamu/core/registry.py`.
//! TODO: GGUF metadata parsing (struct.unpack equivalent).
//! TODO: VRAM estimation heuristics.

use crate::types::ModelEntry;
use crate::Result;
use std::path::Path;

/// Scan directory recursively for model files.
pub fn scan_directory(_models_dir: &Path) -> Result<Vec<ModelEntry>> {
    todo!("port lamu/core/registry.py::scan_directory")
}

/// Write registry to YAML.
pub fn write_registry(_models: &[ModelEntry], _output: &Path) -> Result<()> {
    todo!("port lamu/core/registry.py::write_registry")
}

/// Load registry from YAML.
pub fn load_registry(_path: &Path) -> Result<Vec<ModelEntry>> {
    todo!("port lamu/core/registry.py::load_registry")
}
