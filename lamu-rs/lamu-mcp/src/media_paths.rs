//! Shared confined-output-path resolver for the media tools (tts, image).
//!
//! Both TTS audio and generated images are written to a CONFINED dir under
//! `<data_dir>/lamu/<subdir>`. A caller-supplied `output_path` may only be a
//! relative name with no `..`; absolute paths and parent traversal are
//! rejected so an MCP caller (the LLM) never gets an arbitrary file-write
//! primitive. When no name is given, a nanosecond-stamped `<prefix>-<ns>.<ext>`
//! is used (nanos, not secs — two calls in the same second would otherwise
//! overwrite each other). `Err` is a ready-to-return error string.

use std::path::{Component, PathBuf};

pub(crate) fn resolve_confined_output_path(
    subdir: &str,
    prefix: &str,
    ext: &str,
    output_path: Option<&str>,
) -> Result<PathBuf, String> {
    let dir = dirs::data_dir()
        .unwrap_or_else(std::env::temp_dir) // never the cwd (CI/containers)
        .join("lamu")
        .join(subdir);
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("error: create {subdir} dir {}: {e}", dir.display()))?;
    match output_path {
        Some(p) if !p.is_empty() => {
            let pb = PathBuf::from(p);
            if pb.is_absolute() || pb.components().any(|c| matches!(c, Component::ParentDir)) {
                return Err(format!(
                    "error: output_path must be a relative name with no '..' (writes are confined to {}): got '{p}'",
                    dir.display()
                ));
            }
            Ok(dir.join(pb))
        }
        _ => {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            Ok(dir.join(format!("{prefix}-{stamp}.{ext}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_traversal_and_absolute() {
        assert!(resolve_confined_output_path("tts", "tts", "mp3", Some("../escape.mp3")).is_err());
        assert!(resolve_confined_output_path("tts", "tts", "mp3", Some("/etc/passwd")).is_err());
        assert!(resolve_confined_output_path("tts", "tts", "mp3", Some("ok.mp3")).is_ok());
        assert!(resolve_confined_output_path("tts", "tts", "wav", None).is_ok());
    }

    #[test]
    fn image_subdir_confined() {
        assert!(resolve_confined_output_path("images", "img", "png", Some("../x.png")).is_err());
        assert!(resolve_confined_output_path("images", "img", "png", Some("/etc/x")).is_err());
        assert!(resolve_confined_output_path("images", "img", "png", Some("ok.png")).is_ok());
        assert!(resolve_confined_output_path("images", "img", "png", None).is_ok());
    }

    #[test]
    fn relative_name_lands_in_confined_dir() {
        let p = resolve_confined_output_path("tts", "tts", "mp3", Some("out.mp3")).unwrap();
        assert!(p.ends_with("lamu/tts/out.mp3"));
    }
}
