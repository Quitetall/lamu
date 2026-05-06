//! User-level lamu config — backend URL, theme choice, etc.
//!
//! Path: `$XDG_CONFIG_HOME/lamu/config.toml` (defaults to
//! `~/.config/lamu/config.toml`). Missing file = baked-in defaults.
//! Bad TOML = warn + use defaults so the TUI keeps working.
//!
//! Distinct from cloud-models.yaml (model registry) and favorites.json
//! (per-row pin state). This file holds *settings* the user might
//! reasonably want to flip from a Settings screen.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const BACKEND_DIRECT: &str = "http://localhost:8020/v1/chat/completions";
pub const BACKEND_BIFROST: &str = "http://localhost:8080/v1/chat/completions";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LamuConfig {
    /// Where the chat REPL POSTs by default. Cycles through
    /// `direct`, `bifrost`, and `custom` via the Settings screen.
    #[serde(default = "default_backend_url")]
    pub backend_url: String,

    /// Name of the active theme — looked up in
    /// ~/.config/lamu/themes/<name>.toml or bundled.
    #[serde(default = "default_theme")]
    pub theme: String,

    /// Editor to spawn for `Edit *` settings items. Falls back to
    /// $EDITOR / $VISUAL / "vi" at run time when this is empty.
    #[serde(default)]
    pub editor: String,
}

fn default_backend_url() -> String { BACKEND_DIRECT.to_string() }
fn default_theme() -> String { "lamu".to_string() }

impl Default for LamuConfig {
    fn default() -> Self {
        Self {
            backend_url: default_backend_url(),
            theme: default_theme(),
            editor: String::new(),
        }
    }
}

impl LamuConfig {
    pub fn path() -> PathBuf {
        if let Some(dir) = dirs::config_dir() {
            return dir.join("lamu").join("config.toml");
        }
        PathBuf::from("./lamu-config.toml")
    }

    pub fn load() -> Self {
        let path = Self::path();
        let bytes = match std::fs::read_to_string(&path) {
            Ok(b) => b,
            Err(_) => return Self::default(),
        };
        match toml::from_str::<LamuConfig>(&bytes) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "lamu: config.toml at {} is corrupt ({}); using defaults.",
                    path.display(),
                    e
                );
                Self::default()
            }
        }
    }

    pub fn save(&self) -> std::io::Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let buf = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, buf)
    }

    /// Cycle backend choice: direct → bifrost → direct. (Custom URLs
    /// stay as-is until the user edits the file directly via Settings.)
    pub fn cycle_backend(&mut self) {
        self.backend_url = if self.backend_url == BACKEND_DIRECT {
            BACKEND_BIFROST.to_string()
        } else {
            BACKEND_DIRECT.to_string()
        };
    }

    pub fn backend_label(&self) -> &str {
        match self.backend_url.as_str() {
            BACKEND_DIRECT => "direct (lamu serve :8020)",
            BACKEND_BIFROST => "bifrost (gateway :8080)",
            _ => "custom",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_use_direct() {
        let c = LamuConfig::default();
        assert_eq!(c.backend_url, BACKEND_DIRECT);
        assert_eq!(c.theme, "lamu");
    }

    #[test]
    fn cycle_swaps_direct_and_bifrost() {
        let mut c = LamuConfig::default();
        assert_eq!(c.backend_label(), "direct (lamu serve :8020)");
        c.cycle_backend();
        assert_eq!(c.backend_url, BACKEND_BIFROST);
        assert_eq!(c.backend_label(), "bifrost (gateway :8080)");
        c.cycle_backend();
        assert_eq!(c.backend_url, BACKEND_DIRECT);
    }

    #[test]
    fn custom_label_when_neither() {
        let mut c = LamuConfig::default();
        c.backend_url = "https://my.proxy/v1/chat/completions".into();
        assert_eq!(c.backend_label(), "custom");
    }

    #[test]
    fn toml_round_trip() {
        let c = LamuConfig::default();
        let s = toml::to_string(&c).unwrap();
        let back: LamuConfig = toml::from_str(&s).unwrap();
        assert_eq!(back.backend_url, c.backend_url);
    }
}
