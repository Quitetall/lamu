//! Crate-wide error type. All public APIs return `Result<T, Error>`.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("backend failed: {0}")]
    Backend(String),

    #[error("vram exhausted: need {need_mb}MB, have {have_mb}MB")]
    VramExhausted { need_mb: u32, have_mb: u32 },

    #[error("invalid config: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, Error>;
