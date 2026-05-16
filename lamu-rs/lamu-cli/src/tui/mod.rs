//! Default-on-no-args TUI dashboard. Ollama-shaped UX for `lamu`.
//!
//! Multi-pane layout:
//!   ┌─ Models ──────────────────────────────────────────────────┐
//!   │ → qwen3.6-27b-...  27.6B  Q4  17.4 GB  [chat code reason] │
//!   │   gpt2-xl.q4_k_m    2.0B  Q4   1.2 GB  [chat]             │
//!   ├─ VRAM ────────────────────────────────────────────────────┤
//!   │ ████████████████░░░░░░░ 17.4 / 24 GB                      │
//!   ├─ Health ──────────────────────────────────────────────────┤
//!   │ qwen3.6-27b-…  HEALTHY  errors=0                          │
//!   ├─ Status ──────────────────────────────────────────────────┤
//!   │ MCP not running · HTTP not running · Bifrost UP           │
//!   └─ [j/k] move  [Enter] chat  [l] load  [u] unload  [q] quit ┘
//!
//! Polls `lamu serve`'s /metrics + /v1/models when reachable; falls back
//! to the on-disk registry when not.

mod render;
mod settings;
mod state;
mod swap;

pub use state::AppState;
use settings::{first_run_checks, pick_editor, run_blocking, save_api_key, settings_file_path, spawn_detached};
use swap::swap_to_model_if_needed;

#[cfg(test)]
use render::{format_ctx, format_params, truncate};

use anyhow::Result;
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use lamu_core::config::registry_path;
use lamu_core::registry::load_registry;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{ModelEntry, VramBudget};
use ratatui::backend::CrosstermBackend;
use ratatui::widgets::ListState;
use ratatui::Terminal;
use std::io;
use std::time::{Duration, Instant};

use crate::cloud_models::{self, CloudModel};
use crate::favorites::Favorites;
use crate::lamu_config::LamuConfig;
use crate::mcp_servers::{self, McpServerEntry, ProbeStatus};
use crate::theme::Theme;

const REFRESH_MS: u64 = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Dashboard,
    Launchers,
    McpServers,
    Settings,
}

/// One row in the Settings screen. Action runs when Enter is pressed.
#[derive(Debug, Clone, Copy)]
pub enum SettingAction {
    /// Cycle backend_url through direct/bifrost.
    CycleBackend,
    /// Open a config file in $EDITOR.
    EditFile(SettingFile),
    /// Theme.install_bundled() — copies bundled themes to user dir.
    InstallBundledThemes,
    /// Delete cloud-models.yaml so the seed regenerates on next load.
    ResetCloudSeed,
}

#[derive(Debug, Clone, Copy)]
pub enum SettingFile {
    LamuConfig,
    CloudModels,
    LocalModels,
    McpServers,
    Favorites,
    ThemesDir,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    /// Favorites pinned, then local→cloud, params↑ vram↑ within group.
    Default,
    Name,
    Params,
    Vram,
    Ctx,
}

impl SortKey {
    fn label(self) -> &'static str {
        match self {
            SortKey::Default => "local→cloud params↑ vram↑",
            SortKey::Name => "name",
            SortKey::Params => "params↑",
            SortKey::Vram => "vram↑",
            SortKey::Ctx => "ctx↑",
        }
    }

    fn cycle(self) -> Self {
        match self {
            SortKey::Default => SortKey::Name,
            SortKey::Name => SortKey::Params,
            SortKey::Params => SortKey::Vram,
            SortKey::Vram => SortKey::Ctx,
            SortKey::Ctx => SortKey::Default,
        }
    }
}

/// Filter input mode — when active, key presses go to the filter buffer
/// instead of triggering keybinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    Filter,
    ApiKey,
}

/// Restricts the dashboard list to a subset of model sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFilter {
    All,
    LocalOnly,
    CloudOnly,
}

impl SourceFilter {
    fn label(self) -> &'static str {
        match self {
            SourceFilter::All => "all",
            SourceFilter::LocalOnly => "local",
            SourceFilter::CloudOnly => "cloud",
        }
    }
}

/// View row — points at either a local registry entry or a cloud entry.
#[derive(Debug, Clone, Copy)]
pub enum ModelRef {
    Local(usize),
    Cloud(usize),
}

/// External CLI harness the TUI can launch. Each entry is detected with
/// `which`. Missing → install hint shown + offer to run it. Add new ones
/// by editing `HARNESSES` below — order matters (it's the on-screen order).
#[derive(Debug, Clone)]
pub struct Harness {
    pub name: &'static str,
    pub bin: &'static str,
    pub install: &'static str,
    pub launch_argv: &'static [&'static str],
    pub notes: &'static str,
    /// Matching entry name in `config/harnesses.yaml`. When `Some`,
    /// the TUI routes launches through `scripts/open-harness.sh` so
    /// LAMU_MODEL pinning + lamu URL env + per-flavor model arg
    /// injection + sandbox + git-snap all fire. When `None`, the TUI
    /// falls back to direct `Command::new(launch_argv[0])` — for
    /// builtins like `lamu repl` and external tools (`gh`) that don't
    /// talk to lamu over HTTP.
    pub slug: Option<&'static str>,
}

pub const HARNESSES: &[Harness] = &[
    Harness {
        name: "lamu repl",
        bin: "lamu",
        install: "(built-in)",
        launch_argv: &["lamu", "repl"],
        notes: "Built-in REPL talking to `lamu serve`",
        slug: None,
    },
    Harness {
        name: "Claude Code",
        bin: "claude",
        install: "npm install -g @anthropic-ai/claude-code",
        launch_argv: &["claude"],
        notes: "Anthropic CLI w/ MCP — best paired with `lamu start`",
        slug: Some("claude-code"),
    },
    Harness {
        name: "Codex",
        bin: "codex",
        install: "npm install -g @openai/codex",
        launch_argv: &["codex"],
        notes: "OpenAI Codex CLI",
        slug: Some("codex"),
    },
    Harness {
        name: "OpenCode",
        bin: "opencode",
        install: "npm install -g opencode-ai",
        launch_argv: &["opencode"],
        notes: "sst/opencode terminal coding agent",
        slug: None,
    },
    Harness {
        name: "Hermes",
        bin: "hermes",
        install: "cargo install hermes-cli  # check the upstream you want",
        launch_argv: &["hermes"],
        notes: "Hermes function-calling CLI (any binary on PATH named `hermes`)",
        slug: Some("hermes"),
    },
    Harness {
        name: "Pi",
        bin: "pi",
        install: "npm install -g @inflection-ai/pi-cli  # if/when published",
        launch_argv: &["pi"],
        notes: "Pi assistant CLI",
        slug: Some("pi"),
    },
    Harness {
        name: "GitHub CLI",
        bin: "gh",
        install: "pacman -S github-cli   # or your distro's package",
        launch_argv: &["gh"],
        notes: "GitHub CLI — `gh issue/pr/repo`. Use `gh copilot` for chat.",
        slug: None,
    },
    Harness {
        name: "GitHub Copilot",
        bin: "gh",
        install: "gh extension install github/gh-copilot",
        launch_argv: &["gh", "copilot"],
        notes: "Requires `gh` first — Copilot is a gh extension",
        slug: None,
    },
];

pub(super) fn probe_port(port: u16) -> bool {
    use std::net::{SocketAddr, TcpStream};
    let addr: SocketAddr = match format!("127.0.0.1:{port}").parse() {
        Ok(a) => a,
        Err(_) => return false,
    };
    TcpStream::connect_timeout(&addr, Duration::from_millis(50)).is_ok()
}

pub fn run() -> Result<()> {
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        eprintln!(
            "lamu: stdout is not a TTY — skipping TUI. Use a subcommand \
             (`lamu scan|status|start|serve|repl`) or run from an interactive shell."
        );
        return Ok(());
    }

    // First-run prompts BEFORE we take over the terminal. Plain stdin/stdout
    // here so the user sees a normal CLI prompt, not a half-initialised TUI.
    first_run_checks();

    let mut state = AppState::new()?;

    if state.entries.is_empty() {
        state.status_msg = "No models in registry. Run `just setup-qwen36` then `lamu scan`.".into();
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = run_loop(&mut terminal, &mut state);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    res
}

fn run_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    state: &mut AppState,
) -> Result<()> {
    state.refresh();
    loop {
        terminal.draw(|f| render::draw(f, state))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                // ApiKey input mode: keys go to the api_key_input buffer until Enter/Esc.
                if state.input_mode == InputMode::ApiKey {
                    match key.code {
                        KeyCode::Esc => {
                            state.input_mode = InputMode::Normal;
                            state.api_key_input.clear();
                            state.api_key_for = None;
                            state.status_msg = "api key entry cancelled.".into();
                        }
                        KeyCode::Enter => {
                            let key_val = state.api_key_input.trim().to_string();
                            if let Some((var, model)) = state.api_key_for.take() {
                                if key_val.is_empty() {
                                    state.status_msg = "api key entry cancelled (empty).".into();
                                } else {
                                    match save_api_key(&var, &key_val) {
                                        Ok(path) => {
                                            // SAFETY: single-threaded TUI event loop; no concurrent threads read env.
                                            unsafe { std::env::set_var(&var, &key_val); }
                                            state.status_msg = format!(
                                                "✓ {} key saved to {} and set for this session.",
                                                model, path.display()
                                            );
                                        }
                                        Err(e) => state.status_msg = format!("save failed: {e}"),
                                    }
                                }
                            }
                            state.api_key_input.clear();
                            state.input_mode = InputMode::Normal;
                        }
                        KeyCode::Backspace => {
                            state.api_key_input.pop();
                            let var = state.api_key_for.as_ref().map(|(v, _)| v.clone()).unwrap_or_default();
                            state.status_msg = format!("API key for {}: [{}]", var, "*".repeat(state.api_key_input.len()));
                        }
                        KeyCode::Char(c) => {
                            state.api_key_input.push(c);
                            let var = state.api_key_for.as_ref().map(|(v, _)| v.clone()).unwrap_or_default();
                            state.status_msg = format!("API key for {}: [{}]", var, "*".repeat(state.api_key_input.len()));
                        }
                        _ => {}
                    }
                    continue;
                }

                // Filter input mode: keys go to the filter buffer until Enter/Esc.
                if state.input_mode == InputMode::Filter {
                    match key.code {
                        KeyCode::Esc => {
                            state.input_mode = InputMode::Normal;
                            match state.mode {
                                Mode::Dashboard => state.model_filter.clear(),
                                Mode::Launchers => state.harness_filter.clear(),
                                Mode::McpServers | Mode::Settings => {}
                            }
                            state.recompute_views();
                        }
                        KeyCode::Enter => {
                            state.input_mode = InputMode::Normal;
                        }
                        KeyCode::Backspace => {
                            match state.mode {
                                Mode::Dashboard => { state.model_filter.pop(); }
                                Mode::Launchers => { state.harness_filter.pop(); }
                                Mode::McpServers | Mode::Settings => {}
                            }
                            state.recompute_views();
                        }
                        KeyCode::Char(c) => {
                            match state.mode {
                                Mode::Dashboard => state.model_filter.push(c),
                                Mode::Launchers => state.harness_filter.push(c),
                                Mode::McpServers | Mode::Settings => {}
                            }
                            state.recompute_views();
                        }
                        _ => {}
                    }
                    continue;
                }

                match state.mode {
                    Mode::Dashboard => match key.code {
                        KeyCode::Char('q') => {
                            if state.quit_confirm {
                                return Ok(());
                            } else {
                                state.quit_confirm = true;
                                state.status_msg = "Press q again to exit (x = instant exit)".into();
                            }
                        }
                        KeyCode::Char('x') => return Ok(()),
                        KeyCode::Char('j') | KeyCode::Down => { state.quit_confirm = false; state.move_cursor(1); }
                        KeyCode::Char('k') | KeyCode::Up => { state.quit_confirm = false; state.move_cursor(-1); }
                        KeyCode::Char('r') => { state.quit_confirm = false; state.refresh(); }
                        KeyCode::Char('s') => {
                            // 's' is the settings entry — most-used so it
                            // gets the lowercase. Capital 'S' kept for the
                            // muscle-memory "start lamu serve" action.
                            state.mode = Mode::Settings;
                            state.status_msg.clear();
                        }
                        KeyCode::Char('S') => {
                            if !state.serve_up {
                                state.status_msg = "Starting `lamu serve` on :8020 in background...".into();
                                spawn_detached(&["lamu", "serve"]);
                            } else {
                                state.status_msg = "lamu serve already running on :8020".into();
                            }
                        }
                        KeyCode::Char('b') => {
                            // back — no parent from dashboard; just clear quit_confirm
                            state.quit_confirm = false;
                        }
                        KeyCode::Char('B') => {
                            state.quit_confirm = false;
                            if !state.bifrost_up {
                                state.status_msg = "Starting Bifrost (just serve-bifrost)...".into();
                                spawn_detached(&["just", "serve-bifrost"]);
                            } else {
                                state.status_msg = "Bifrost already up on :8080".into();
                            }
                        }
                        KeyCode::Char('h') => {
                            state.mode = Mode::Launchers;
                            state.status_msg.clear();
                        }
                        KeyCode::Char('m') => {
                            state.mode = Mode::McpServers;
                            state.mcp_servers = mcp_servers::load_servers();
                            state.status_msg = format!(
                                "{} MCP server(s) configured. [p] probe, [a] probe all.",
                                state.mcp_servers.len()
                            );
                        }
                        KeyCode::Char('*') | KeyCode::Char('f') => {
                            if let Some(name) = state.selected_name() {
                                let added = state.favorites.toggle_model(&name);
                                state.recompute_views();
                                state.status_msg = if added {
                                    format!("★ favorited {}", name)
                                } else {
                                    format!("☆ unfavorited {}", name)
                                };
                            }
                        }
                        KeyCode::Char('L') => {
                            state.source_filter = SourceFilter::LocalOnly;
                            state.recompute_views();
                            state.status_msg = "showing local only".into();
                        }
                        KeyCode::Char('C') => {
                            state.source_filter = SourceFilter::CloudOnly;
                            state.recompute_views();
                            state.status_msg = "showing cloud only".into();
                        }
                        KeyCode::Char('A') => {
                            state.source_filter = SourceFilter::All;
                            state.recompute_views();
                            state.status_msg = "showing all sources".into();
                        }
                        KeyCode::Char('n') => {
                            // Interactive add-cloud-model wizard. Tear down
                            // alt-screen so the prompts render normally,
                            // then append + save + reload.
                            let mut new_entry: Option<CloudModel> = None;
                            run_subprocess_in_tui(terminal, || -> Result<()> {
                                new_entry = cloud_models::add_via_wizard();
                                Ok(())
                            })?;
                            if let Some(entry) = new_entry {
                                let mut all = state.cloud_models.clone();
                                all.push(entry.clone());
                                match cloud_models::save(&all) {
                                    Ok(()) => {
                                        state.cloud_models = all;
                                        state.recompute_views();
                                        state.status_msg = format!(
                                            "✓ added {} → {}",
                                            entry.full_id(),
                                            cloud_models::config_path().display()
                                        );
                                    }
                                    Err(e) => {
                                        state.status_msg = format!("save failed: {e}");
                                    }
                                }
                            } else {
                                state.status_msg = "add cancelled.".into();
                            }
                        }
                        KeyCode::Char('e') => {
                            // Open the cloud-models YAML in $EDITOR so the user
                            // can add models / set api_key_env / flip quota
                            // states. After exit, reload from disk.
                            let path = cloud_models::config_path();
                            let editor = std::env::var("EDITOR")
                                .or_else(|_| std::env::var("VISUAL"))
                                .unwrap_or_else(|_| "vi".into());
                            let path_clone = path.clone();
                            run_subprocess_in_tui(terminal, move || -> Result<()> {
                                println!("\n→ Editing {}\n", path_clone.display());
                                let _ = std::process::Command::new(&editor)
                                    .arg(&path_clone)
                                    .status();
                                Ok(())
                            })?;
                            state.cloud_models = cloud_models::load();
                            state.recompute_views();
                            state.status_msg = format!("Reloaded {}", path.display());
                        }
                        KeyCode::Char('K') => {
                            state.quit_confirm = false;
                            // Show the API-key status for the selected cloud
                            // row. Bifrost actually uses the keys; lamu just
                            // tells the user whether the env var is exported.
                            if let Some(cm) = state.selected_cloud() {
                                state.status_msg = match cm.api_key_env.as_deref() {
                                    None => format!(
                                        "{}: routes via Bifrost provider keys (no per-model env var).",
                                        cm.full_id()
                                    ),
                                    Some(var) => {
                                        if std::env::var(var).is_ok() {
                                            format!("{}: ${} is SET ✓", cm.full_id(), var)
                                        } else {
                                            format!(
                                                "{}: ${} is unset. `export {}=<key>` then [r].",
                                                cm.full_id(), var, var
                                            )
                                        }
                                    }
                                };
                            } else {
                                state.status_msg = "key status — select a [CLOUD] row first.".into();
                            }
                        }
                        KeyCode::Char('a') => {
                            state.quit_confirm = false;
                            if let Some(cm) = state.selected_cloud() {
                                match cm.api_key_env.clone() {
                                    Some(var) => {
                                        let model = cm.full_id();
                                        state.api_key_for = Some((var.clone(), model.clone()));
                                        state.api_key_input.clear();
                                        state.input_mode = InputMode::ApiKey;
                                        state.status_msg = format!("Enter API key for {} ({}): type key, Enter to save, Esc to cancel", model, var);
                                    }
                                    None => {
                                        state.status_msg = "this model has no api_key_env configured.".into();
                                    }
                                }
                            } else {
                                state.status_msg = "select a [CLOUD] model first.".into();
                            }
                        }
                        KeyCode::Char('o') => {
                            state.model_sort = state.model_sort.cycle();
                            state.recompute_views();
                            state.status_msg = format!("sort: {}", state.model_sort.label());
                        }
                        KeyCode::Char('/') => {
                            state.input_mode = InputMode::Filter;
                            state.model_filter.clear();
                            state.status_msg = "filter: type to refine, Enter to apply, Esc to cancel".into();
                        }
                        KeyCode::Enter => {
                            // Branch on local vs cloud. Local goes to the
                            // user's default harness (if set + installed)
                            // or falls back to the built-in lamu repl on
                            // OpenAI compat :8020. Cloud always goes via
                            // the gateway URL.
                            if let Some(local_name) = state.selected_entry().map(|e| e.name.clone()) {
                                // Resolve default harness: lookup by name in
                                // HARNESSES, ensure binary is on $PATH.
                                let default = state
                                    .favorites
                                    .default_harness()
                                    .and_then(|n| HARNESSES.iter().find(|h| h.name == n))
                                    .filter(|h| which_exists(h.bin));
                                if let Some(h) = default {
                                    let argv: Vec<String> = h.launch_argv.iter().map(|s| s.to_string()).collect();
                                    let label = h.name;
                                    let slug = h.slug;
                                    let model_for_env = local_name.clone();
                                    run_subprocess_in_tui(terminal, move || -> Result<()> {
                                        println!("\n→ Launching {label} (default harness, model={})\n", model_for_env);
                                        // Slug present → route through
                                        // `scripts/open-harness.sh` for full env
                                        // injection (lamu URL + per-flavor model
                                        // arg + optional sandbox). No slug →
                                        // direct exec (builtins / GitHub CLI).
                                        let mut cmd = if let Some(slug) = slug {
                                            // Resolve $HOME with a clear failure
                                            // mode — empty HOME would produce
                                            // `/local-llm/scripts/...` and a
                                            // confusing "file not found".
                                            let home = std::env::var("HOME")
                                                .unwrap_or_else(|_| {
                                                    eprintln!(
                                                        "warning: $HOME unset — script path \
                                                         resolution likely to fail; falling back \
                                                         to /home/brianklam"
                                                    );
                                                    "/home/brianklam".to_string()
                                                });
                                            let script = format!("{}/local-llm/scripts/open-harness.sh", home);
                                            let mut c = std::process::Command::new("bash");
                                            c.arg(script).arg(slug);
                                            c
                                        } else {
                                            let mut c = std::process::Command::new(&argv[0]);
                                            c.args(&argv[1..]);
                                            c
                                        };
                                        cmd.env("LAMU_MODEL", &model_for_env);
                                        let _ = cmd.status();
                                        Ok(())
                                    })?;
                                    state.last_harness = Some(label);
                                    state.refresh();
                                    state.status_msg = format!("Returned from {} (default harness).", label);
                                } else {
                                    let cfg = state.config.clone();
                                    let theme = Theme::pick(Some(&cfg.theme));
                                    let name = local_name.clone();
                                    let entry = state.selected_entry().unwrap().clone();
                                    run_subprocess_in_tui(terminal, move || -> Result<()> {
                                        swap_to_model_if_needed(&entry)?;
                                        crate::chat_tui::run(name, theme, cfg)
                                    })?;
                                    state.last_harness = Some("lamu repl");
                                    state.refresh();
                                    state.status_msg = "Returned from chat.".into();
                                }
                            } else if let Some(cloud) = state.selected_cloud().cloned() {
                                let gateway = std::env::var("LAMU_GATEWAY_URL")
                                    .unwrap_or_else(|_| "http://localhost:8080/v1/chat/completions".into());
                                let mut cfg = state.config.clone();
                                // Use the model's own base_url when set (e.g. DeepSeek
                                // direct), otherwise fall back to the Bifrost gateway.
                                cfg.backend_url = cloud.chat_url(&gateway);
                                // Resolve API key from env so chat_tui can auth directly.
                                cfg.api_key = cloud.resolved_api_key();
                                let theme = Theme::pick(Some(&cfg.theme));
                                let model_id = cloud.full_id();
                                run_subprocess_in_tui(terminal, move || -> Result<()> {
                                    crate::chat_tui::run(model_id, theme, cfg)
                                })?;
                                state.last_harness = Some("lamu repl");
                                state.refresh();
                                state.status_msg = "Returned from cloud chat.".into();
                            }
                        }
                        KeyCode::Char('l') => {
                            state.quit_confirm = false;
                            if let Some(e) = state.selected_entry() {
                                state.status_msg = format!(
                                    "Use `lamu start` (MCP) and `load_model('{}')` from Claude Code to load.",
                                    e.name
                                );
                            }
                        }
                        _ => { state.quit_confirm = false; }
                    },
                    Mode::McpServers => match key.code {
                        KeyCode::Char('q') | KeyCode::Char('b') => {
                            state.mode = Mode::Dashboard;
                            state.status_msg.clear();
                        }
                        KeyCode::Char('x') => return Ok(()),
                        KeyCode::Esc | KeyCode::Backspace => {
                            state.mode = Mode::Dashboard;
                            state.status_msg.clear();
                        }
                        KeyCode::Char('j') | KeyCode::Down => state.move_mcp(1),
                        KeyCode::Char('k') | KeyCode::Up => state.move_mcp(-1),
                        KeyCode::Char('r') => {
                            state.mcp_servers = mcp_servers::load_servers();
                            state.status_msg = "Re-loaded ~/.claude.json".into();
                        }
                        KeyCode::Char('p') | KeyCode::Enter => {
                            if let Some(idx) = state.selected_mcp_idx() {
                                if let Some(entry) = state.mcp_servers.get(idx).cloned() {
                                    state.status_msg = format!("Probing {} (≤3s)...", entry.name);
                                    let status = mcp_servers::probe(&entry);
                                    if let Some(e) = state.mcp_servers.get_mut(idx) {
                                        e.status = status.clone();
                                    }
                                    state.status_msg = match &status {
                                        ProbeStatus::Healthy { server_name } =>
                                            format!("✓ {} → server={}", entry.name, server_name),
                                        ProbeStatus::Unreachable { reason } =>
                                            format!("✗ {} — {}", entry.name, reason),
                                        ProbeStatus::Untested =>
                                            format!("? {} — {} not probed", entry.name, entry.typ),
                                    };
                                }
                            }
                        }
                        KeyCode::Char('a') => {
                            state.status_msg = format!("Probing {} server(s)...", state.mcp_servers.len());
                            let mut servers = state.mcp_servers.clone();
                            for s in servers.iter_mut() {
                                s.status = mcp_servers::probe(s);
                            }
                            state.mcp_servers = servers;
                            let healthy = state.mcp_servers.iter().filter(|s| matches!(s.status, ProbeStatus::Healthy{..})).count();
                            state.status_msg = format!(
                                "Probed {}: {} healthy, {} unreachable",
                                state.mcp_servers.len(),
                                healthy,
                                state.mcp_servers.len() - healthy,
                            );
                        }
                        _ => {}
                    },
                    Mode::Settings => match key.code {
                        KeyCode::Char('q') | KeyCode::Char('b') => {
                            state.mode = Mode::Dashboard;
                            state.status_msg.clear();
                        }
                        KeyCode::Char('x') => return Ok(()),
                        KeyCode::Esc | KeyCode::Backspace => {
                            state.mode = Mode::Dashboard;
                            state.status_msg.clear();
                        }
                        KeyCode::Char('j') | KeyCode::Down => state.move_settings(1),
                        KeyCode::Char('k') | KeyCode::Up => state.move_settings(-1),
                        KeyCode::Enter => {
                            let items = state.settings_items();
                            let sel = state.settings_state.selected().unwrap_or(0);
                            if let Some((_, action)) = items.get(sel).cloned() {
                                match action {
                                    SettingAction::CycleBackend => {
                                        state.config.cycle_backend();
                                        let _ = state.config.save();
                                        state.status_msg = format!(
                                            "backend → {} ({})",
                                            state.config.backend_label(),
                                            state.config.backend_url
                                        );
                                    }
                                    SettingAction::EditFile(which) => {
                                        let path = settings_file_path(which);
                                        let editor = pick_editor();
                                        run_subprocess_in_tui(terminal, move || -> Result<()> {
                                            println!("\n→ Editing {}\n", path.display());
                                            let _ = std::process::Command::new(&editor)
                                                .arg(&path).status();
                                            Ok(())
                                        })?;
                                        // Reload anything we know about — some files affect runtime.
                                        state.config = LamuConfig::load();
                                        state.cloud_models = cloud_models::load();
                                        state.favorites = Favorites::load();
                                        state.recompute_views();
                                        state.status_msg = "Settings reloaded.".into();
                                    }
                                    SettingAction::InstallBundledThemes => {
                                        match Theme::install_bundled() {
                                            Ok(n) => state.status_msg = format!(
                                                "Installed {} bundled theme(s) → {}.",
                                                n,
                                                Theme::user_themes_dir()
                                                    .map(|p| p.display().to_string())
                                                    .unwrap_or_else(|| "?".into())
                                            ),
                                            Err(e) => state.status_msg = format!("install failed: {e}"),
                                        }
                                    }
                                    SettingAction::ResetCloudSeed => {
                                        let path = cloud_models::config_path();
                                        match std::fs::remove_file(&path) {
                                            Ok(()) | Err(_) => {
                                                state.cloud_models = cloud_models::load();
                                                state.recompute_views();
                                                state.status_msg = format!(
                                                    "Cloud seed reset → {} entries.",
                                                    state.cloud_models.len()
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    },
                    Mode::Launchers => match key.code {
                        KeyCode::Char('q') | KeyCode::Char('b') => {
                            state.mode = Mode::Dashboard;
                            state.status_msg.clear();
                        }
                        KeyCode::Char('x') => return Ok(()),
                        // Back-to-dashboard: any of Esc, Backspace.
                        KeyCode::Esc | KeyCode::Backspace => {
                            state.mode = Mode::Dashboard;
                            state.status_msg.clear();
                        }
                        KeyCode::Char('j') | KeyCode::Down => state.move_launcher(1),
                        KeyCode::Char('k') | KeyCode::Up => state.move_launcher(-1),
                        KeyCode::Char('r') => state.refresh(),
                        // Set the selected harness as the default one
                        // launched on Dashboard Enter. Toggle off if it's
                        // already the default.
                        KeyCode::Char('D') => {
                            if let Some(h) = state.selected_harness() {
                                let name = h.name.to_string();
                                if state.favorites.default_harness() == Some(&name) {
                                    state.favorites.set_default_harness(None);
                                    state.status_msg = format!(
                                        "default harness cleared (was {})",
                                        name
                                    );
                                } else {
                                    state.favorites.set_default_harness(Some(name.clone()));
                                    state.status_msg = format!(
                                        "default harness = {} — Dashboard Enter now launches it.",
                                        name
                                    );
                                }
                            }
                        }
                        KeyCode::Char('*') | KeyCode::Char('f') => {
                            if let Some(h) = state.selected_harness() {
                                let name = h.name.to_string();
                                let added = state.favorites.toggle_harness(&name);
                                state.recompute_views();
                                state.status_msg = if added {
                                    format!("★ favorited {}", name)
                                } else {
                                    format!("☆ unfavorited {}", name)
                                };
                            }
                        }
                        KeyCode::Char('o') => {
                            state.harness_sort = state.harness_sort.cycle();
                            state.recompute_views();
                            state.status_msg = format!("sort: {}", state.harness_sort.label());
                        }
                        KeyCode::Char('/') => {
                            state.input_mode = InputMode::Filter;
                            state.harness_filter.clear();
                            state.status_msg = "filter: type to refine, Enter to apply, Esc to cancel".into();
                        }
                        KeyCode::Enter => {
                            if let Some(h) = state.selected_harness() {
                                let orig = state.selected_harness_orig_idx().unwrap_or(0);
                                let installed = *state.harness_installed.get(orig).unwrap_or(&false);
                                if installed {
                                    let argv: Vec<String> = h.launch_argv.iter().map(|s| s.to_string()).collect();
                                    let label = h.name;
                                    let slug = h.slug;
                                    // Selected-model pin (from the dashboard's
                                    // model cursor) flows through to the
                                    // harness here too, so Launchers screen
                                    // Enter has the same semantics as Dashboard
                                    // Enter: launches the harness using the
                                    // model the user has highlighted.
                                    let model_for_env =
                                        state.selected_entry().map(|e| e.name.clone());
                                    run_subprocess_in_tui(terminal, move || -> Result<()> {
                                        println!("\n→ Launching {label}\n");
                                        let mut cmd = if let Some(slug) = slug {
                                            // Resolve $HOME with a clear failure
                                            // mode — empty HOME would produce
                                            // `/local-llm/scripts/...` and a
                                            // confusing "file not found".
                                            let home = std::env::var("HOME")
                                                .unwrap_or_else(|_| {
                                                    eprintln!(
                                                        "warning: $HOME unset — script path \
                                                         resolution likely to fail; falling back \
                                                         to /home/brianklam"
                                                    );
                                                    "/home/brianklam".to_string()
                                                });
                                            let script = format!("{}/local-llm/scripts/open-harness.sh", home);
                                            let mut c = std::process::Command::new("bash");
                                            c.arg(script).arg(slug);
                                            c
                                        } else {
                                            let mut c = std::process::Command::new(&argv[0]);
                                            c.args(&argv[1..]);
                                            c
                                        };
                                        if let Some(m) = model_for_env {
                                            cmd.env("LAMU_MODEL", m);
                                        }
                                        let _ = cmd.status();
                                        Ok(())
                                    })?;
                                    state.last_harness = Some(h.name);
                                    state.refresh();
                                    state.status_msg = format!("Returned from {}.", h.name);
                                } else {
                                    state.status_msg = format!(
                                        "Not installed. Run: {}    (then press 'i' to install)",
                                        h.install
                                    );
                                }
                            }
                        }
                        KeyCode::Char('i') => {
                            if let Some(h) = state.selected_harness() {
                                let orig = state.selected_harness_orig_idx().unwrap_or(0);
                                let installed = *state.harness_installed.get(orig).unwrap_or(&false);
                                if installed {
                                    state.status_msg = format!("{} already installed at {}.", h.name, h.bin);
                                } else if h.install.starts_with('(') {
                                    state.status_msg = format!("{}: {}", h.name, h.install);
                                } else {
                                    let cmd = h.install.to_string();
                                    let label = h.name;
                                    run_subprocess_in_tui(terminal, move || -> Result<()> {
                                        println!("\n→ Installing {label}: {cmd}\n");
                                        let status = std::process::Command::new("sh")
                                            .arg("-c").arg(&cmd).status();
                                        match status {
                                            Ok(s) if s.success() => println!("\n✓ Install OK\n"),
                                            Ok(s) => println!("\n✗ exit {s}\n"),
                                            Err(e) => println!("\n✗ failed to exec sh: {e}\n"),
                                        }
                                        Ok(())
                                    })?;
                                    state.refresh();
                                    state.status_msg = format!("Re-checked {} install status.", h.name);
                                }
                            }
                        }
                        _ => {}
                    },
                }
            }
        }

        if state.last_refresh.elapsed() >= Duration::from_millis(REFRESH_MS) {
            state.refresh();
        }
    }
}

/// Tear down the alt-screen, run a closure (which can read stdin and
/// print to stdout normally), then restore the alt-screen. Used by both
/// the chat hand-off and the harness launcher.
fn run_subprocess_in_tui<B, F>(terminal: &mut Terminal<B>, f: F) -> Result<()>
where
    B: ratatui::backend::Backend,
    F: FnOnce() -> Result<()>,
{
    disable_raw_mode().ok();
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture).ok();
    terminal.show_cursor().ok();

    let res = f();

    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    terminal.clear()?;

    res
}

pub(super) fn which_exists(bin: &str) -> bool {
    let status = std::process::Command::new("which")
        .arg(bin)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    matches!(status, Ok(s) if s.success())
}

/// Resolve the on-disk path for a Settings "Edit ..." item.

/// Check which model is loaded on :8020 and swap if it doesn't match
/// `entry`. Kills the existing llama-server, spawns a new one with the
/// optimised flags, health-polls, then warms up one token so cuBLAS is
/// ready before the chat TUI opens.
///
/// Runs inside `run_subprocess_in_tui` so stdout is visible — progress
/// lines print to the terminal while the model loads.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_keeps_short() {
        assert_eq!(truncate("hi", 10), "hi");
    }

    #[test]
    fn truncate_appends_ellipsis() {
        let out = truncate("0123456789abcdef", 8);
        assert_eq!(out.chars().count(), 8);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn truncate_zero_max_returns_empty() {
        assert_eq!(truncate("anything", 0), "");
        assert_eq!(truncate("", 0), "");
    }

    #[test]
    fn format_params_under_10_one_decimal() {
        assert_eq!(format_params(0.8), "0.8");
        assert_eq!(format_params(2.7), "2.7");
    }

    #[test]
    fn format_params_10plus_no_decimal() {
        assert_eq!(format_params(27.6), "28");
        assert_eq!(format_params(35.0), "35");
    }

    #[test]
    fn move_cursor_wraps() {
        let mut s = AppState::new().unwrap();
        s.entries = (0..3).map(|i| dummy_entry(i)).collect();
        s.cloud_models.clear();
        s.recompute_views();
        s.list_state.select(Some(0));
        s.move_cursor(-1);
        assert_eq!(s.list_state.selected(), Some(2));
        s.move_cursor(1);
        assert_eq!(s.list_state.selected(), Some(0));
    }

    #[test]
    fn format_ctx_compact_units() {
        assert_eq!(format_ctx(4096), "4K");
        assert_eq!(format_ctx(8192), "8K");
        assert_eq!(format_ctx(131072), "128K");
        assert_eq!(format_ctx(262144), "256K");
        assert_eq!(format_ctx(1024 * 1024 * 2), "2M");
        assert_eq!(format_ctx(512), "512");
    }

    #[test]
    fn harnesses_table_well_formed() {
        // Sanity check on the static table: every entry has non-empty fields
        // and a non-empty launch_argv.
        for h in HARNESSES {
            assert!(!h.name.is_empty());
            assert!(!h.bin.is_empty());
            assert!(!h.launch_argv.is_empty(), "{} has no launch_argv", h.name);
            assert!(!h.install.is_empty());
        }
    }

    #[test]
    fn move_launcher_wraps() {
        let mut s = AppState::new().unwrap();
        s.launcher_state.select(Some(0));
        let n = HARNESSES.len() as i32;
        if n > 1 {
            s.move_launcher(-1);
            assert_eq!(s.launcher_state.selected(), Some((n - 1) as usize));
            s.move_launcher(1);
            assert_eq!(s.launcher_state.selected(), Some(0));
        }
    }

    #[test]
    fn which_exists_finds_real_binary() {
        // `sh` is on every POSIX box including the CI image.
        assert!(which_exists("sh"));
        assert!(!which_exists("definitely-not-a-real-binary-xyz123"));
    }

    #[test]
    fn sortkey_cycles_through_all() {
        let mut k = SortKey::Default;
        let mut seen: Vec<SortKey> = Vec::new();
        for _ in 0..6 {
            if !seen.contains(&k) {
                seen.push(k);
            }
            k = k.cycle();
        }
        assert_eq!(seen.len(), 5);
    }

    fn local_name_at(s: &AppState, idx: usize) -> &str {
        match s.model_view[idx] {
            ModelRef::Local(i) => s.entries[i].name.as_str(),
            ModelRef::Cloud(_) => panic!("expected local at index {idx}"),
        }
    }

    #[test]
    fn filter_substring_narrows_view() {
        let mut s = AppState::new().unwrap();
        s.entries = vec![
            dummy_entry_named("alpha-7b"),
            dummy_entry_named("beta-13b"),
            dummy_entry_named("gamma-7b"),
        ];
        s.cloud_models.clear(); // narrow to local in tests so cloud rows
                                // don't muddle the assertions
        s.model_filter = "7b".into();
        s.recompute_views();
        assert_eq!(s.model_view.len(), 2);
        assert_eq!(local_name_at(&s, 0), "alpha-7b");
        assert_eq!(local_name_at(&s, 1), "gamma-7b");
    }

    #[test]
    fn sort_by_params_ascending() {
        let mut s = AppState::new().unwrap();
        let mut a = dummy_entry_named("small"); a.params_b = 1.0;
        let mut b = dummy_entry_named("medium"); b.params_b = 7.0;
        let mut c = dummy_entry_named("large"); c.params_b = 70.0;
        s.entries = vec![a, b, c];
        s.cloud_models.clear();
        s.model_sort = SortKey::Params;
        s.recompute_views();
        let names: Vec<_> = (0..s.model_view.len())
            .map(|i| local_name_at(&s, i).to_string())
            .collect();
        assert_eq!(names, vec!["small", "medium", "large"]);
    }

    #[test]
    fn default_sort_local_params_asc_then_vram_asc() {
        let mut s = AppState::new().unwrap();
        let mut a = dummy_entry_named("big-lowvram"); a.params_b = 70.0; a.vram_mb = 8000;
        let mut b = dummy_entry_named("small"); b.params_b = 1.0; b.vram_mb = 1000;
        let mut c = dummy_entry_named("medium"); c.params_b = 7.0; c.vram_mb = 4000;
        s.entries = vec![a, b, c];
        s.cloud_models.clear();
        s.model_sort = SortKey::Default;
        s.recompute_views();
        let names: Vec<_> = (0..s.model_view.len())
            .map(|i| local_name_at(&s, i).to_string())
            .collect();
        assert_eq!(names, vec!["small", "medium", "big-lowvram"]);
    }

    #[test]
    fn favorites_pin_to_top_regardless_of_sort() {
        let mut s = AppState::new().unwrap();
        s.entries = vec![
            dummy_entry_named("alpha"),
            dummy_entry_named("beta"),
            dummy_entry_named("gamma"),
        ];
        s.cloud_models.clear();
        s.favorites.models.insert("gamma".to_string());
        s.recompute_views();
        assert_eq!(local_name_at(&s, 0), "gamma");
        assert_eq!(local_name_at(&s, 1), "alpha");
        assert_eq!(local_name_at(&s, 2), "beta");
    }

    #[test]
    fn source_filter_local_hides_cloud() {
        let mut s = AppState::new().unwrap();
        s.entries = vec![dummy_entry_named("a")];
        // Cloud seed list is non-empty by default — verify filter prunes it.
        let initial_total = s.model_view.len();
        s.source_filter = SourceFilter::LocalOnly;
        s.recompute_views();
        assert_eq!(s.model_view.len(), 1);
        assert!(initial_total > 1, "default seed should provide cloud rows");
    }

    #[test]
    fn source_filter_cloud_hides_local() {
        let mut s = AppState::new().unwrap();
        s.entries = vec![dummy_entry_named("a"), dummy_entry_named("b")];
        s.source_filter = SourceFilter::CloudOnly;
        s.recompute_views();
        for r in &s.model_view {
            assert!(matches!(r, ModelRef::Cloud(_)));
        }
    }

    #[test]
    fn empty_filter_shows_everything() {
        let mut s = AppState::new().unwrap();
        s.entries = (0..5).map(|i| dummy_entry_named(&format!("e{i}"))).collect();
        s.cloud_models.clear();
        s.model_filter.clear();
        s.recompute_views();
        assert_eq!(s.model_view.len(), 5);
    }

    fn dummy_entry_named(name: &str) -> ModelEntry {
        let mut e = dummy_entry(0);
        e.name = name.to_string();
        e
    }

    fn dummy_entry(i: usize) -> ModelEntry {
        use lamu_core::types::{BackendType, Capability, ModelFormat};
        use std::path::PathBuf;
        ModelEntry {
            name: format!("test-{i}"),
            path: PathBuf::from("/tmp/x.gguf"),
            format: ModelFormat::Gguf,
            backend: BackendType::LlamaCpp,
            arch: "qwen35".into(),
            params_b: 1.0,
            quant: "Q4".into(),
            vram_mb: 1000,
            context_max: 8192,
            capabilities: vec![Capability::Chat],
            reasoning_marker: None,
            speculative: None,
            pinned: false,
            main: false,
            notes: String::new(),
            status: lamu_core::types::ModelStatus::default(),
        }
    }
}
