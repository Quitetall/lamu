//! User favorites — pinned models + harnesses, persisted to disk.
//!
//! Path: `$XDG_CONFIG_HOME/lamu/favorites.json` (defaults to
//! `~/.config/lamu/favorites.json`). Lazy: missing file = empty
//! favorites; bad JSON = empty + a stderr warning. Save errors are
//! swallowed so a read-only home can't kill the TUI.
//!
//! Sets, not lists — toggling is idempotent. JSON shape:
//!   {"models": ["a","b"], "harnesses": ["claude-code"]}

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Favorites {
    #[serde(default)]
    pub models: BTreeSet<String>,
    #[serde(default)]
    pub harnesses: BTreeSet<String>,
}

impl Favorites {
    pub fn path() -> PathBuf {
        if let Some(dir) = dirs::config_dir() {
            return dir.join("lamu").join("favorites.json");
        }
        // No config dir? Fall back to ~/.lamu-favorites.json.
        if let Some(home) = dirs::home_dir() {
            return home.join(".lamu-favorites.json");
        }
        PathBuf::from("./lamu-favorites.json")
    }

    pub fn load() -> Self {
        let path = Self::path();
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return Self::default(),
        };
        serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            eprintln!(
                "lamu: favorites file at {} is corrupt ({}); ignoring.",
                path.display(),
                e
            );
            Self::default()
        })
    }

    pub fn save(&self) {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(buf) = serde_json::to_vec_pretty(self) {
            let _ = std::fs::write(&path, buf);
        }
    }

    pub fn toggle_model(&mut self, name: &str) -> bool {
        if self.models.remove(name) {
            self.save();
            false
        } else {
            self.models.insert(name.to_string());
            self.save();
            true
        }
    }

    pub fn toggle_harness(&mut self, name: &str) -> bool {
        if self.harnesses.remove(name) {
            self.save();
            false
        } else {
            self.harnesses.insert(name.to_string());
            self.save();
            true
        }
    }

    pub fn has_model(&self, name: &str) -> bool {
        self.models.contains(name)
    }

    pub fn has_harness(&self, name: &str) -> bool {
        self.harnesses.contains(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_serde() {
        let mut f = Favorites::default();
        f.models.insert("alpha".into());
        f.harnesses.insert("claude".into());
        let buf = serde_json::to_vec(&f).unwrap();
        let g: Favorites = serde_json::from_slice(&buf).unwrap();
        assert!(g.has_model("alpha"));
        assert!(g.has_harness("claude"));
    }

    #[test]
    fn toggle_is_idempotent() {
        // Use a temp dir for the save side-effect by overriding HOME.
        let tmp = tempdir_for_test();
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", &tmp);
        }
        let mut f = Favorites::default();
        assert!(f.toggle_model("x"));
        assert!(f.has_model("x"));
        assert!(!f.toggle_model("x"));
        assert!(!f.has_model("x"));
        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn missing_file_yields_empty() {
        let f = Favorites::default();
        assert!(!f.has_model("anything"));
        assert!(!f.has_harness("anything"));
    }

    fn tempdir_for_test() -> PathBuf {
        let p = std::env::temp_dir().join(format!("lamu-fav-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&p);
        p
    }
}
