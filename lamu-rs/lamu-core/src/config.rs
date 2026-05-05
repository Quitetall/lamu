//! Configuration constants. Port of `lamu/core/config.py`.

use std::path::PathBuf;

pub fn lamu_root() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join("local-llm")
}

pub fn models_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join("models")
}

pub fn registry_path() -> PathBuf {
    lamu_root().join("config").join("models.yaml")
}

pub fn llama_bin() -> PathBuf {
    dirs::home_dir().unwrap_or_default()
        .join("llama.cpp").join("build").join("bin").join("llama-server")
}

pub const PORT_MAIN: u16 = 8020;
pub const PORT_SIDECAR: u16 = 8001;
pub const PORT_DFLASH: u16 = 8000;

pub const VRAM_RESERVED_MB: u32 = 1500;
pub const DEFAULT_MAX_TOKENS: u32 = 16384;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_CTX_SIZE: u32 = 131072;

// TODO: add `dirs` to Cargo.toml when implementing
