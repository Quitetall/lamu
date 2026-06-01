//! Phase 3.3: settings + api-key wizard + first-run helpers.
//!
//! settings_file_path / pick_editor — settings tab "Edit ..." rows.
//! save_api_key — persist API key to ~/.config/lamu/api-keys.env.
//! first_run_checks / prompt_yes — first-launch interactive setup.
//! spawn_detached / run_blocking — CLI subprocess helpers used by
//! the settings page.

use super::{probe_port, SettingFile};
use crate::cloud_models;
use crate::favorites::Favorites;
use crate::lamu_config::LamuConfig;
use crate::mcp_servers;
use crate::theme::Theme;
use lamu_core::config::registry_path;
use lamu_core::registry::load_registry;

pub(super) fn settings_file_path(which: SettingFile) -> std::path::PathBuf {
    match which {
        SettingFile::LamuConfig => LamuConfig::path(),
        SettingFile::CloudModels => cloud_models::config_path(),
        SettingFile::LocalModels => lamu_core::config::registry_path(),
        SettingFile::McpServers => mcp_servers::config_path(),
        SettingFile::Favorites => Favorites::path(),
        SettingFile::ThemesDir => Theme::user_themes_dir().unwrap_or_else(|| {
            // A literal "~/..." never shell-expands — resolve the real dir.
            dirs::config_dir()
                .map(|d| d.join("lamu").join("themes"))
                .unwrap_or_else(std::env::temp_dir)
        }),
    }
}

pub(super) fn pick_editor() -> String {
    let cfg = LamuConfig::load();
    if !cfg.editor.is_empty() {
        return cfg.editor;
    }
    std::env::var("EDITOR")
        .or_else(|_| std::env::var("VISUAL"))
        .unwrap_or_else(|_| "vi".into())
}

pub(super) fn save_api_key(var_name: &str, key_val: &str) -> std::io::Result<std::path::PathBuf> {
    let dir = dirs::config_dir()
        .map(|d| d.join("lamu"))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("api-keys.env");
    // Read existing, replace or append.
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let prefix = format!("export {}=", var_name);
    let new_line = format!("export {}={}", var_name, key_val);
    let updated: String = if existing.lines().any(|l| l.starts_with(&prefix)) {
        existing
            .lines()
            .map(|l| if l.starts_with(&prefix) { new_line.as_str() } else { l })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    } else {
        format!("{}{}\n", existing, new_line)
    };
    std::fs::write(&path, updated)?;
    // M14: api-keys.env holds secrets — make it owner-only (0600). Without this
    // a freshly-created file inherits the umask (typically 0644 = world/group
    // readable), unlike the sibling cloud_config::save_api_key_env writer.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(path)
}

pub(super) fn spawn_detached(argv: &[&str]) {
    use std::process::{Command, Stdio};
    if argv.is_empty() {
        return;
    }
    let _ = Command::new(argv[0])
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

/// Pre-TUI bootstrap. Runs while the terminal is still in cooked mode so
/// the user sees plain `[Y/n]` prompts instead of a half-initialised
/// dashboard. Skips silently when stdin is not a TTY (CI / piped runs).
pub(super) fn first_run_checks() {
    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() {
        return;
    }

    // 1. Empty registry → offer to download Qwen3.6-27B + run scan.
    let entries = load_registry(&registry_path()).unwrap_or_default();
    if entries.is_empty() {
        eprintln!("\nLAMU first-run: no models found in registry.");
        eprintln!("  Suggested: download Qwen3.6-27B-uncensored-heretic-v2 (~16 GB).");
        if prompt_yes("Run `just setup-qwen36` now?", true) {
            run_blocking(&["just", "setup-qwen36"]);
            run_blocking(&["lamu", "scan"]);
        } else {
            eprintln!("  Skipped. You can run it later via `just setup-qwen36`.");
        }
    }

    // 2. LAMU_GATEWAY_URL set but Bifrost down → offer to start it.
    if let Ok(gw) = std::env::var("LAMU_GATEWAY_URL") {
        if gw.contains(":8080") && !probe_port(8080) {
            eprintln!("\nLAMU_GATEWAY_URL points at :8080 but Bifrost is not running.");
            if prompt_yes("Start Bifrost (just serve-bifrost)?", true) {
                run_blocking(&["just", "serve-bifrost"]);
            }
        }
    }
}

pub(super) fn prompt_yes(question: &str, default_yes: bool) -> bool {
    use std::io::{self, Write};
    let suffix = if default_yes { "[Y/n]" } else { "[y/N]" };
    eprint!("  {} {} ", question, suffix);
    let _ = io::stderr().flush();
    let mut buf = String::new();
    if io::stdin().read_line(&mut buf).is_err() {
        return default_yes;
    }
    let answer = buf.trim().to_lowercase();
    if answer.is_empty() {
        default_yes
    } else {
        answer.starts_with('y')
    }
}

pub(super) fn run_blocking(argv: &[&str]) {
    use std::process::Command;
    if argv.is_empty() {
        return;
    }
    let status = Command::new(argv[0]).args(&argv[1..]).status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("  command exited with {}", s),
        Err(e) => eprintln!("  failed to run {:?}: {}", argv, e),
    }
}
