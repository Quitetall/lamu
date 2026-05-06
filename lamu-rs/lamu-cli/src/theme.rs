//! TUI theme — colors, branding, spinner, banner art.
//!
//! Schema mirrors the Hermes/Skynet theme TOML the user shared:
//!   [colors] hex strings (#RRGGBB)
//!   [spinner] waiting_faces / thinking_faces / thinking_verbs
//!   [branding] agent_name / welcome / goodbye / response_label /
//!     prompt_symbol / help_header
//!   tool_prefix / [tool_emojis] / banner_logo / banner_hero
//!
//! Lookup order:
//!   1. `--theme <name>` CLI flag (or LAMU_THEME env var).
//!   2. `~/.config/lamu/themes/<name>.toml`.
//!   3. Bundled `themes/<name>.toml` (compiled into the binary).
//!   4. Bundled `themes/lamu.toml` (default).
//!
//! Bundled themes are baked via `include_str!` so the binary is
//! standalone — no $LAMU_HOME required.

use ratatui::style::Color;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Bundled theme TOML strings — compiled into the binary at build time.
const BUNDLED: &[(&str, &str)] = &[
    ("lamu", include_str!("../themes/lamu.toml")),
    ("skynet", include_str!("../themes/skynet.toml")),
];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Theme {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub colors: Colors,
    #[serde(default)]
    pub spinner: Spinner,
    #[serde(default)]
    pub branding: Branding,
    #[serde(default)]
    pub tool_prefix: String,
    #[serde(default)]
    pub tool_emojis: ToolEmojis,
    #[serde(default)]
    pub banner_logo: String,
    #[serde(default)]
    pub banner_hero: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Colors {
    #[serde(default)] pub banner_border: String,
    #[serde(default)] pub banner_title: String,
    #[serde(default)] pub banner_accent: String,
    #[serde(default)] pub banner_dim: String,
    #[serde(default)] pub banner_text: String,
    #[serde(default)] pub ui_accent: String,
    #[serde(default)] pub ui_label: String,
    #[serde(default)] pub ui_ok: String,
    #[serde(default)] pub ui_error: String,
    #[serde(default)] pub ui_warn: String,
    #[serde(default)] pub prompt: String,
    #[serde(default)] pub input_rule: String,
    #[serde(default)] pub response_border: String,
    #[serde(default)] pub status_bar_bg: String,
    #[serde(default)] pub status_bar_text: String,
    #[serde(default)] pub status_bar_strong: String,
    #[serde(default)] pub status_bar_dim: String,
    #[serde(default)] pub status_bar_good: String,
    #[serde(default)] pub status_bar_warn: String,
    #[serde(default)] pub status_bar_bad: String,
    #[serde(default)] pub status_bar_critical: String,
    #[serde(default)] pub session_label: String,
    #[serde(default)] pub session_border: String,
    // Optional / not all themes use these
    #[serde(default)] pub voice_status_bg: String,
    #[serde(default)] pub completion_menu_bg: String,
    #[serde(default)] pub completion_menu_current_bg: String,
    #[serde(default)] pub completion_menu_meta_bg: String,
    #[serde(default)] pub completion_menu_meta_current_bg: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Spinner {
    #[serde(default)] pub waiting_faces: Vec<String>,
    #[serde(default)] pub thinking_faces: Vec<String>,
    #[serde(default)] pub thinking_verbs: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Branding {
    #[serde(default)] pub agent_name: String,
    #[serde(default)] pub welcome: String,
    #[serde(default)] pub goodbye: String,
    #[serde(default)] pub response_label: String,
    #[serde(default)] pub prompt_symbol: String,
    #[serde(default)] pub help_header: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolEmojis {
    #[serde(default)] pub terminal: String,
    #[serde(default)] pub web_search: String,
    #[serde(default)] pub read_file: String,
    #[serde(default)] pub write_file: String,
    #[serde(default)] pub search_files: String,
    #[serde(default)] pub execute_code: String,
    #[serde(default)] pub browser_navigate: String,
    #[serde(default)] pub delegate_task: String,
    #[serde(default)] pub mixture_of_agents: String,
    #[serde(default)] pub memory: String,
    #[serde(default)] pub clarify: String,
    #[serde(default)] pub cronjob: String,
    #[serde(default)] pub process: String,
    #[serde(default)] pub todo: String,
}

impl Theme {
    /// Resolve a theme by name. None on missing+unknown.
    pub fn load(name: &str) -> Option<Self> {
        // 1. user dir override
        if let Some(dir) = dirs::config_dir() {
            let p = dir.join("lamu").join("themes").join(format!("{name}.toml"));
            if p.exists() {
                if let Ok(s) = std::fs::read_to_string(&p) {
                    if let Ok(t) = toml::from_str::<Theme>(&s) {
                        return Some(t);
                    }
                }
            }
        }
        // 2. bundled
        for (k, body) in BUNDLED {
            if *k == name {
                if let Ok(t) = toml::from_str::<Theme>(body) {
                    return Some(t);
                }
            }
        }
        None
    }

    /// Default theme (always loadable). Reads `--theme` flag → env var
    /// → bundled `lamu`. Falls back to a bare-bones in-memory theme
    /// only if every loader fails (e.g. corrupted bundle).
    pub fn pick(name: Option<&str>) -> Self {
        let chosen = name
            .map(String::from)
            .or_else(|| std::env::var("LAMU_THEME").ok())
            .unwrap_or_else(|| "lamu".to_string());
        Self::load(&chosen).unwrap_or_else(|| Self::load("lamu").unwrap_or_default())
    }

    pub fn list_bundled() -> Vec<&'static str> {
        BUNDLED.iter().map(|(k, _)| *k).collect()
    }

    /// Save bundled themes to ~/.config/lamu/themes/ for users to fork.
    pub fn install_bundled() -> std::io::Result<usize> {
        let dir = dirs::config_dir()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no config dir"))?
            .join("lamu")
            .join("themes");
        std::fs::create_dir_all(&dir)?;
        let mut written = 0;
        for (name, body) in BUNDLED {
            let path = dir.join(format!("{name}.toml"));
            if !path.exists() {
                std::fs::write(&path, body)?;
                written += 1;
            }
        }
        Ok(written)
    }

    pub fn user_themes_dir() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("lamu").join("themes"))
    }
}

/// Hex color → ratatui Color. Falls back to a sensible default when the
/// string is missing or malformed (so a partial theme TOML still works).
pub fn hex_to_color(hex: &str, fallback: Color) -> Color {
    let s = hex.trim().trim_start_matches('#');
    if s.len() != 6 {
        return fallback;
    }
    let r = u8::from_str_radix(&s[0..2], 16).ok();
    let g = u8::from_str_radix(&s[2..4], 16).ok();
    let b = u8::from_str_radix(&s[4..6], 16).ok();
    match (r, g, b) {
        (Some(r), Some(g), Some(b)) => Color::Rgb(r, g, b),
        _ => fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_themes_parse() {
        for (name, _) in BUNDLED {
            let t = Theme::load(name).unwrap_or_else(|| panic!("bundled theme {name} failed to load"));
            assert!(!t.name.is_empty());
        }
    }

    #[test]
    fn list_bundled_is_nonempty() {
        let v = Theme::list_bundled();
        assert!(v.iter().any(|n| *n == "lamu"));
        assert!(v.iter().any(|n| *n == "skynet"));
    }

    #[test]
    fn pick_falls_back_to_lamu_on_unknown() {
        let t = Theme::pick(Some("definitely-not-a-real-theme-xyz"));
        assert_eq!(t.name, "lamu");
    }

    #[test]
    fn hex_parses_six_digit() {
        assert!(matches!(hex_to_color("#FF0000", Color::Reset), Color::Rgb(0xFF, 0x00, 0x00)));
        assert!(matches!(hex_to_color("00BCD4", Color::Reset), Color::Rgb(0x00, 0xBC, 0xD4)));
    }

    #[test]
    fn hex_falls_back_on_garbage() {
        assert_eq!(hex_to_color("nope", Color::Yellow), Color::Yellow);
        assert_eq!(hex_to_color("", Color::Cyan), Color::Cyan);
        assert_eq!(hex_to_color("#zzzzzz", Color::Magenta), Color::Magenta);
    }
}
