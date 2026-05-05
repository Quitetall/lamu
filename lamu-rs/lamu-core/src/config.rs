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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ports_distinct() {
        assert_ne!(PORT_MAIN, PORT_SIDECAR);
        assert_ne!(PORT_MAIN, PORT_DFLASH);
        assert_ne!(PORT_SIDECAR, PORT_DFLASH);
    }

    #[test]
    fn ports_in_user_range() {
        for p in [PORT_MAIN, PORT_SIDECAR, PORT_DFLASH] {
            assert!(p >= 1024 && p < u16::MAX);
        }
    }

    #[test]
    fn registry_path_under_root() {
        let root = lamu_root();
        let reg = registry_path();
        assert!(reg.starts_with(&root) || reg.to_string_lossy().contains("local-llm"));
    }

    #[test]
    fn defaults_sane() {
        assert!(DEFAULT_MAX_TOKENS > 0);
        assert!(DEFAULT_TEMPERATURE > 0.0 && DEFAULT_TEMPERATURE <= 2.0);
        assert!(DEFAULT_CTX_SIZE >= 4096);
        assert!(VRAM_RESERVED_MB > 0);
    }

    #[test]
    fn paths_are_pathbufs() {
        let _ = lamu_root();
        let _ = models_dir();
        let _ = registry_path();
        let _ = llama_bin();
    }
}
