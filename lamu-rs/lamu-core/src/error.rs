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

    #[error("http: {0}")]
    Http(String),

    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("gpu unavailable: {0}")]
    GpuUnavailable(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_from_impl() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "x");
        let e: Error = io.into();
        assert!(matches!(e, Error::Io(_)));
        assert!(format!("{e}").starts_with("io:"));
    }

    #[test]
    fn vram_exhausted_format() {
        let e = Error::VramExhausted { need_mb: 24000, have_mb: 4000 };
        let msg = format!("{e}");
        assert!(msg.contains("24000"));
        assert!(msg.contains("4000"));
    }

    #[test]
    fn model_not_found_format() {
        let e = Error::ModelNotFound("qwen35".into());
        assert!(format!("{e}").contains("qwen35"));
    }

    #[test]
    fn config_format() {
        let e = Error::Config("missing key".into());
        assert!(format!("{e}").contains("missing key"));
    }

    #[test]
    fn backend_format() {
        let e = Error::Backend("OOM".into());
        assert!(format!("{e}").contains("OOM"));
    }
}
