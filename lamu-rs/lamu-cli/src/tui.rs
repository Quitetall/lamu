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
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::block::{Position, Title};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, ListState, Paragraph};
use ratatui::Terminal;
use std::io;
use std::time::{Duration, Instant};

use crate::cloud_models::{self, CloudModel, QuotaState};
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
}

pub const HARNESSES: &[Harness] = &[
    Harness {
        name: "lamu repl",
        bin: "lamu",
        install: "(built-in)",
        launch_argv: &["lamu", "repl"],
        notes: "Built-in REPL talking to `lamu serve`",
    },
    Harness {
        name: "Claude Code",
        bin: "claude",
        install: "npm install -g @anthropic-ai/claude-code",
        launch_argv: &["claude"],
        notes: "Anthropic CLI w/ MCP — best paired with `lamu start`",
    },
    Harness {
        name: "Codex",
        bin: "codex",
        install: "npm install -g @openai/codex",
        launch_argv: &["codex"],
        notes: "OpenAI Codex CLI",
    },
    Harness {
        name: "OpenCode",
        bin: "opencode",
        install: "npm install -g opencode-ai",
        launch_argv: &["opencode"],
        notes: "sst/opencode terminal coding agent",
    },
    Harness {
        name: "Hermes",
        bin: "hermes",
        install: "cargo install hermes-cli  # check the upstream you want",
        launch_argv: &["hermes"],
        notes: "Hermes function-calling CLI (any binary on PATH named `hermes`)",
    },
    Harness {
        name: "Pi",
        bin: "pi",
        install: "npm install -g @inflection-ai/pi-cli  # if/when published",
        launch_argv: &["pi"],
        notes: "Pi assistant CLI",
    },
    Harness {
        name: "GitHub CLI",
        bin: "gh",
        install: "pacman -S github-cli   # or your distro's package",
        launch_argv: &["gh"],
        notes: "GitHub CLI — `gh issue/pr/repo`. Use `gh copilot` for chat.",
    },
    Harness {
        name: "GitHub Copilot",
        bin: "gh",
        install: "gh extension install github/gh-copilot",
        launch_argv: &["gh", "copilot"],
        notes: "Requires `gh` first — Copilot is a gh extension",
    },
];

#[derive(Debug)]
pub struct AppState {
    pub entries: Vec<ModelEntry>,
    pub list_state: ListState,
    pub vram: VramBudget,
    pub gpu_available: bool,
    pub gpu_reason: Option<String>,
    pub last_refresh: Instant,
    pub status_msg: String,
    pub serve_up: bool,
    pub bifrost_up: bool,
    pub mode: Mode,
    pub launcher_state: ListState,
    pub harness_installed: Vec<bool>,
    pub last_harness: Option<&'static str>,
    pub favorites: Favorites,
    pub model_sort: SortKey,
    pub model_filter: String,
    pub harness_sort: SortKey,
    pub harness_filter: String,
    pub input_mode: InputMode,
    pub quit_confirm: bool,
    pub api_key_input: String,
    /// (env_var_name, model_display_name) for the in-progress ApiKey input.
    pub api_key_for: Option<(String, String)>,
    /// Cached sorted+filtered indices into `entries` / `HARNESSES`.
    /// Recomputed on every state change. The `list_state.selected()`
    /// indexes INTO these vecs, not the underlying slices.
    pub harness_view: Vec<usize>,
    pub mcp_servers: Vec<McpServerEntry>,
    pub mcp_state: ListState,
    pub cloud_models: Vec<CloudModel>,
    pub source_filter: SourceFilter,
    /// Unified view: each row references either a local registry entry
    /// or a cloud entry. Replaces the previous `Vec<usize>` so cloud
    /// models can sit alongside local ones in the same scroll buffer.
    pub model_view: Vec<ModelRef>,
    pub config: LamuConfig,
    pub settings_state: ListState,
    /// Live snapshot of nvidia-smi compute-apps. (pid, mem_mb, name).
    /// Refreshed on every tick so the status pane can show what's
    /// actually eating VRAM — including processes lamu didn't spawn.
    pub gpu_procs: Vec<(u32, u32, String)>,
}

impl AppState {
    fn new() -> Result<Self> {
        let entries = load_registry(&registry_path()).unwrap_or_default();
        let scheduler = VramScheduler::new();
        let vram = scheduler.budget();
        let gpu_available = scheduler.gpu_available();
        let gpu_reason = scheduler.gpu_unavailable_reason().map(String::from);

        let mut list_state = ListState::default();
        if !entries.is_empty() {
            list_state.select(Some(0));
        }
        let mut launcher_state = ListState::default();
        launcher_state.select(Some(0));
        let harness_installed = HARNESSES.iter().map(|h| which_exists(h.bin)).collect();
        let favorites = Favorites::load();

        let mcp_servers = mcp_servers::load_servers();
        let mut mcp_state = ListState::default();
        if !mcp_servers.is_empty() {
            mcp_state.select(Some(0));
        }
        let cloud_models = cloud_models::load();
        let config = LamuConfig::load();
        let mut settings_state = ListState::default();
        settings_state.select(Some(0));

        let mut s = Self {
            entries,
            list_state,
            vram,
            gpu_available,
            gpu_reason,
            last_refresh: Instant::now(),
            status_msg: String::new(),
            serve_up: false,
            bifrost_up: false,
            mode: Mode::Dashboard,
            launcher_state,
            harness_installed,
            last_harness: None,
            favorites,
            model_sort: SortKey::Default,
            model_filter: String::new(),
            harness_sort: SortKey::Default,
            harness_filter: String::new(),
            input_mode: InputMode::Normal,
            quit_confirm: false,
            api_key_input: String::new(),
            api_key_for: None,
            harness_view: Vec::new(),
            mcp_servers,
            mcp_state,
            cloud_models,
            source_filter: SourceFilter::All,
            model_view: Vec::new(),
            config,
            settings_state,
            gpu_procs: Vec::new(),
        };
        s.recompute_views();
        s.refresh_gpu_procs();
        Ok(s)
    }

    /// Snapshot the GPU process list. Looks up each PID's command name
    /// from /proc/<pid>/comm so the user can identify the offender.
    fn refresh_gpu_procs(&mut self) {
        let scheduler = VramScheduler::new();
        let pairs = scheduler.query_gpu_pids();
        self.gpu_procs = pairs
            .into_iter()
            .map(|(pid, mb)| {
                let name = std::fs::read_to_string(format!("/proc/{pid}/comm"))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|_| "?".into());
                (pid, mb, name)
            })
            .collect();
    }

    fn settings_items(&self) -> Vec<(String, SettingAction)> {
        vec![
            (
                format!("Backend URL  [{}]  ({})", self.config.backend_label(), self.config.backend_url),
                SettingAction::CycleBackend,
            ),
            ("Edit lamu config (~/.config/lamu/config.toml)".into(), SettingAction::EditFile(SettingFile::LamuConfig)),
            ("Edit cloud models (~/.config/lamu/cloud-models.yaml)".into(), SettingAction::EditFile(SettingFile::CloudModels)),
            ("Edit local models registry (~/local-llm/config/models.yaml)".into(), SettingAction::EditFile(SettingFile::LocalModels)),
            ("Edit MCP servers (~/.claude.json)".into(), SettingAction::EditFile(SettingFile::McpServers)),
            ("Edit favorites (~/.config/lamu/favorites.json)".into(), SettingAction::EditFile(SettingFile::Favorites)),
            ("Open themes directory (~/.config/lamu/themes/)".into(), SettingAction::EditFile(SettingFile::ThemesDir)),
            ("Install bundled themes to user dir".into(), SettingAction::InstallBundledThemes),
            ("Reset cloud-models.yaml to bundled seed".into(), SettingAction::ResetCloudSeed),
        ]
    }

    fn move_settings(&mut self, delta: i32) {
        let n = self.settings_items().len() as i32;
        if n == 0 { return; }
        let cur = self.settings_state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.settings_state.select(Some(next));
    }

    fn move_mcp(&mut self, delta: i32) {
        if self.mcp_servers.is_empty() {
            return;
        }
        let n = self.mcp_servers.len() as i32;
        let cur = self.mcp_state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.mcp_state.select(Some(next));
    }

    fn selected_mcp_idx(&self) -> Option<usize> {
        self.mcp_state.selected()
    }

    /// Test: a model is "deployed" if the scheduler currently lists it
    /// among loaded_models.
    fn model_deployed(&self, name: &str) -> bool {
        self.vram.loaded_models.iter().any(|(n, _)| n == name)
    }

    /// Build sorted+filtered index vectors. Favorites pinned at the top
    /// regardless of sort key. Local models render before cloud (when
    /// source_filter = All) — both blocks individually sorted.
    pub fn recompute_views(&mut self) {
        // Local models filter
        let filter = self.model_filter.to_lowercase();
        let mut local_idx: Vec<usize> = (0..self.entries.len())
            .filter(|i| {
                if filter.is_empty() {
                    return true;
                }
                let e = &self.entries[*i];
                if e.name.to_lowercase().contains(&filter)
                    || e.quant.to_lowercase().contains(&filter)
                {
                    return true;
                }
                e.capabilities.iter().any(|c| {
                    let cs = match c {
                        lamu_core::types::Capability::Chat => "chat",
                        lamu_core::types::Capability::Code => "code",
                        lamu_core::types::Capability::Reasoning => "reasoning",
                        lamu_core::types::Capability::Routing => "routing",
                        lamu_core::types::Capability::Vision => "vision",
                        lamu_core::types::Capability::LongContext => "long",
                    };
                    cs.contains(&filter)
                })
            })
            .collect();

        let mut cloud_idx: Vec<usize> = (0..self.cloud_models.len())
            .filter(|i| {
                if filter.is_empty() {
                    return true;
                }
                let m = &self.cloud_models[*i];
                m.name.to_lowercase().contains(&filter)
                    || m.provider.to_lowercase().contains(&filter)
                    || m.notes.to_lowercase().contains(&filter)
            })
            .collect();

        // Source filter
        match self.source_filter {
            SourceFilter::All => {}
            SourceFilter::LocalOnly => cloud_idx.clear(),
            SourceFilter::CloudOnly => local_idx.clear(),
        }

        let sort = self.model_sort;
        let entries = &self.entries;
        let cloud_models = &self.cloud_models;
        let favs = &self.favorites;

        // Sort local
        local_idx.sort_by(|a, b| {
            let fa = favs.has_model(&entries[*a].name);
            let fb = favs.has_model(&entries[*b].name);
            if fa != fb {
                return fb.cmp(&fa);
            }
            match sort {
                // Default: params asc → vram asc → name (smallest/cheapest first)
                SortKey::Default => {
                    let ea = &entries[*a];
                    let eb = &entries[*b];
                    let params_ord = ea.params_b
                        .partial_cmp(&eb.params_b)
                        .unwrap_or(std::cmp::Ordering::Equal);
                    if params_ord != std::cmp::Ordering::Equal { return params_ord; }
                    let vram_ord = ea.vram_mb.cmp(&eb.vram_mb);
                    if vram_ord != std::cmp::Ordering::Equal { return vram_ord; }
                    ea.name.cmp(&eb.name)
                }
                SortKey::Name => entries[*a].name.cmp(&entries[*b].name),
                SortKey::Params => entries[*a]
                    .params_b
                    .partial_cmp(&entries[*b].params_b)
                    .unwrap_or(std::cmp::Ordering::Equal),
                SortKey::Vram => entries[*a].vram_mb.cmp(&entries[*b].vram_mb),
                SortKey::Ctx => entries[*a].context_max.cmp(&entries[*b].context_max),
            }
        });

        // Sort cloud (ctx asc by default — smaller ctx = lighter/cheaper first)
        cloud_idx.sort_by(|a, b| {
            let fa = favs.has_model(&cloud_models[*a].full_id());
            let fb = favs.has_model(&cloud_models[*b].full_id());
            if fa != fb {
                return fb.cmp(&fa);
            }
            match sort {
                SortKey::Default => cloud_models[*a]
                    .context_max
                    .cmp(&cloud_models[*b].context_max),
                SortKey::Name => cloud_models[*a].name.cmp(&cloud_models[*b].name),
                SortKey::Ctx => cloud_models[*a]
                    .context_max
                    .cmp(&cloud_models[*b].context_max),
                // Cloud has no params/vram — fall back to name.
                _ => cloud_models[*a].name.cmp(&cloud_models[*b].name),
            }
        });

        // Merge: local first, then cloud.
        let mut view: Vec<ModelRef> = Vec::with_capacity(local_idx.len() + cloud_idx.len());
        for i in local_idx {
            view.push(ModelRef::Local(i));
        }
        for i in cloud_idx {
            view.push(ModelRef::Cloud(i));
        }
        self.model_view = view;

        // Harnesses
        let h_filter = self.harness_filter.to_lowercase();
        let mut hidx: Vec<usize> = (0..HARNESSES.len())
            .filter(|i| {
                if h_filter.is_empty() {
                    return true;
                }
                let h = &HARNESSES[*i];
                h.name.to_lowercase().contains(&h_filter)
                    || h.bin.to_lowercase().contains(&h_filter)
            })
            .collect();

        let installed = &self.harness_installed;
        hidx.sort_by(|a, b| {
            let fa = favs.has_harness(HARNESSES[*a].name);
            let fb = favs.has_harness(HARNESSES[*b].name);
            if fa != fb {
                return fb.cmp(&fa);
            }
            // Then installed-before-missing.
            let ia = *installed.get(*a).unwrap_or(&false);
            let ib = *installed.get(*b).unwrap_or(&false);
            if ia != ib {
                return ib.cmp(&ia);
            }
            HARNESSES[*a].name.cmp(HARNESSES[*b].name)
        });
        self.harness_view = hidx;

        // Clamp selection to view length.
        if let Some(sel) = self.list_state.selected() {
            if sel >= self.model_view.len() {
                self.list_state.select(if self.model_view.is_empty() { None } else { Some(0) });
            }
        }
        if let Some(sel) = self.launcher_state.selected() {
            if sel >= self.harness_view.len() {
                self.launcher_state.select(if self.harness_view.is_empty() { None } else { Some(0) });
            }
        }
    }

    fn refresh(&mut self) {
        let scheduler = VramScheduler::new();
        self.vram = scheduler.budget();
        self.gpu_available = scheduler.gpu_available();
        self.gpu_reason = scheduler.gpu_unavailable_reason().map(String::from);
        self.serve_up = probe_port(8020);
        self.bifrost_up = probe_port(8080);
        self.harness_installed = HARNESSES.iter().map(|h| which_exists(h.bin)).collect();
        self.refresh_gpu_procs();
        self.last_refresh = Instant::now();
    }

    fn move_launcher(&mut self, delta: i32) {
        if self.harness_view.is_empty() {
            return;
        }
        let n = self.harness_view.len() as i32;
        let cur = self.launcher_state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.launcher_state.select(Some(next));
    }

    fn selected_harness(&self) -> Option<&'static Harness> {
        self.launcher_state
            .selected()
            .and_then(|i| self.harness_view.get(i).copied())
            .and_then(|orig| HARNESSES.get(orig))
    }

    fn selected_harness_orig_idx(&self) -> Option<usize> {
        self.launcher_state
            .selected()
            .and_then(|i| self.harness_view.get(i).copied())
    }

    fn move_cursor(&mut self, delta: i32) {
        if self.model_view.is_empty() {
            return;
        }
        let n = self.model_view.len() as i32;
        let cur = self.list_state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.list_state.select(Some(next));
    }

    fn selected_ref(&self) -> Option<ModelRef> {
        self.list_state
            .selected()
            .and_then(|i| self.model_view.get(i).copied())
    }

    fn selected_entry(&self) -> Option<&ModelEntry> {
        match self.selected_ref()? {
            ModelRef::Local(i) => self.entries.get(i),
            ModelRef::Cloud(_) => None,
        }
    }

    fn selected_cloud(&self) -> Option<&CloudModel> {
        match self.selected_ref()? {
            ModelRef::Cloud(i) => self.cloud_models.get(i),
            ModelRef::Local(_) => None,
        }
    }

    /// Display name for the selected row (local entry name OR cloud full_id).
    fn selected_name(&self) -> Option<String> {
        match self.selected_ref()? {
            ModelRef::Local(i) => self.entries.get(i).map(|e| e.name.clone()),
            ModelRef::Cloud(i) => self.cloud_models.get(i).map(|m| m.full_id()),
        }
    }
}

fn probe_port(port: u16) -> bool {
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
        terminal.draw(|f| draw(f, state))?;

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
                                    let model_for_env = local_name.clone();
                                    run_subprocess_in_tui(terminal, move || -> Result<()> {
                                        println!("\n→ Launching {label} (default harness, model={})\n", model_for_env);
                                        // Pass the selected model name in env so harnesses
                                        // that read it (claude/codex via env override) pick it up.
                                        let mut cmd = std::process::Command::new(&argv[0]);
                                        cmd.args(&argv[1..]).env("LAMU_MODEL", &model_for_env);
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
                                    run_subprocess_in_tui(terminal, move || -> Result<()> {
                                        println!("\n→ Launching {label}\n");
                                        let mut cmd = std::process::Command::new(&argv[0]);
                                        cmd.args(&argv[1..]);
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

fn which_exists(bin: &str) -> bool {
    let status = std::process::Command::new("which")
        .arg(bin)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    matches!(status, Ok(s) if s.success())
}

fn draw(f: &mut ratatui::Frame, state: &AppState) {
    match state.mode {
        Mode::Dashboard => draw_dashboard(f, state),
        Mode::Launchers => draw_launchers(f, state),
        Mode::McpServers => draw_mcp(f, state),
        Mode::Settings => draw_settings(f, state),
    }
}

/// Resolve the on-disk path for a Settings "Edit ..." item.
fn settings_file_path(which: SettingFile) -> std::path::PathBuf {
    match which {
        SettingFile::LamuConfig => LamuConfig::path(),
        SettingFile::CloudModels => cloud_models::config_path(),
        SettingFile::LocalModels => lamu_core::config::registry_path(),
        SettingFile::McpServers => mcp_servers::config_path(),
        SettingFile::Favorites => Favorites::path(),
        SettingFile::ThemesDir => {
            Theme::user_themes_dir().unwrap_or_else(|| std::path::PathBuf::from("~/.config/lamu/themes"))
        }
    }
}

fn pick_editor() -> String {
    let cfg = LamuConfig::load();
    if !cfg.editor.is_empty() {
        return cfg.editor;
    }
    std::env::var("EDITOR")
        .or_else(|_| std::env::var("VISUAL"))
        .unwrap_or_else(|_| "vi".into())
}

fn draw_settings(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(4),
            Constraint::Length(5),
        ])
        .split(f.area());

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled("SETTINGS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(format!("  config={}  ", LamuConfig::path().display())),
        Span::raw("[d/Esc] back"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    // List
    let items_data = state.settings_items();
    let items: Vec<ListItem> = items_data
        .iter()
        .map(|(label, _)| ListItem::new(label.clone()))
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Configurables"))
        .highlight_style(
            Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("→ ");
    let mut s = state.settings_state.clone();
    f.render_stateful_widget(list, chunks[1], &mut s);

    // Detail
    let mut detail: Vec<Line> = Vec::new();
    detail.push(Line::from(format!(
        "Backend URL: {}  ({})",
        state.config.backend_url, state.config.backend_label()
    )));
    detail.push(Line::from(format!(
        "Theme:       {}",
        if state.config.theme.is_empty() { "lamu (default)".into() } else { state.config.theme.clone() }
    )));
    if !state.status_msg.is_empty() {
        detail.push(Line::from(Span::styled(
            state.status_msg.clone(),
            Style::default().fg(Color::Yellow),
        )));
    }
    f.render_widget(
        Paragraph::new(detail).block(Block::default().borders(Borders::ALL).title("Detail")),
        chunks[2],
    );

    // Footer (3 rows simple → advanced)
    let row_basics = Line::from(vec![
        key_span("j/k"), key_lbl("move"),
        key_span("Enter"), key_lbl("activate selected"),
        key_span("d/Esc/Bksp"), key_lbl("back to dashboard"),
        key_span("q"), key_lbl("quit"),
    ]);
    let row_actions = Line::from(vec![
        Span::styled("Items: ", Style::default().fg(Color::DarkGray)),
        Span::raw("Backend cycles direct↔bifrost. "),
        Span::raw("Edit-* opens $EDITOR. "),
        Span::raw("Reset deletes cloud-models.yaml then reloads seed."),
    ]);
    let row_advanced = Line::from(vec![
        Span::styled(
            "Editor resolves: lamu_config.editor → $EDITOR → $VISUAL → vi",
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    let footer = Paragraph::new(vec![row_basics, row_actions, row_advanced])
        .block(Block::default().borders(Borders::ALL).title("keys — simple → advanced"));
    f.render_widget(footer, chunks[3]);
}

fn draw_mcp(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Min(8),    // list
            Constraint::Length(5), // detail
            Constraint::Length(5), // help footer (3 rows)
        ])
        .split(f.area());

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "MCP SERVERS",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  source="),
        Span::styled(
            mcp_servers::config_path().display().to_string(),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw(format!("  count={}  ", state.mcp_servers.len())),
        Span::raw("[d/Esc] back"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    let label = format!(
        "  {:<20}  {:<6}  {:<8}  {:<24}  {}",
        "NAME", "TYPE", "STATUS", "COMMAND", "ARGS"
    );
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(1)])
        .split(chunks[1]);
    f.render_widget(
        Paragraph::new(label).style(
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        split[0],
    );

    let items: Vec<ListItem> = state
        .mcp_servers
        .iter()
        .map(|s| {
            let (status_label, color) = match &s.status {
                ProbeStatus::Healthy { .. } => ("✓ healthy", Color::Green),
                ProbeStatus::Unreachable { .. } => ("✗ down", Color::Red),
                ProbeStatus::Untested => ("? untested", Color::Yellow),
            };
            let line = format!(
                "{:<20}  {:<6}  {:<10}  {:<24}  {}",
                truncate(&s.name, 20),
                s.typ,
                status_label,
                truncate(&s.command, 24),
                s.args.join(" "),
            );
            ListItem::new(line).style(Style::default().fg(color))
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Configured servers"))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("→ ");
    let mut s = state.mcp_state.clone();
    f.render_stateful_widget(list, split[1], &mut s);

    // Detail pane
    let mut lines: Vec<Line> = Vec::new();
    if let Some(idx) = state.selected_mcp_idx() {
        if let Some(entry) = state.mcp_servers.get(idx) {
            lines.push(Line::from(format!(
                "{} {} {}",
                entry.command,
                entry.args.join(" "),
                entry.cwd.as_deref().map(|c| format!("(cwd={})", c)).unwrap_or_default(),
            )));
            match &entry.status {
                ProbeStatus::Healthy { server_name } => {
                    lines.push(Line::from(Span::styled(
                        format!("✓ initialize → server.name={server_name}"),
                        Style::default().fg(Color::Green),
                    )));
                }
                ProbeStatus::Unreachable { reason } => {
                    lines.push(Line::from(Span::styled(
                        format!("✗ {reason}"),
                        Style::default().fg(Color::Red),
                    )));
                }
                ProbeStatus::Untested => {
                    lines.push(Line::from(Span::styled(
                        "? press [p] or Enter to probe (sends initialize, ≤3s)".to_string(),
                        Style::default().fg(Color::Yellow),
                    )));
                }
            }
        }
    }
    if !state.status_msg.is_empty() {
        lines.push(Line::from(Span::styled(
            state.status_msg.clone(),
            Style::default().fg(Color::Yellow),
        )));
    }
    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Detail")),
        chunks[2],
    );

    let row_basics = Line::from(vec![
        key_span("j/k"), key_lbl("move"),
        key_span("Enter"), key_lbl("probe selected"),
        key_span("d/Esc/Bksp"), key_lbl("back to dashboard"),
        key_span("q"), key_lbl("quit"),
    ]);
    let row_actions = Line::from(vec![
        key_span("p"), key_lbl("re-probe selected"),
        key_span("a"), key_lbl("probe all"),
        key_span("r"), key_lbl("reload ~/.claude.json"),
    ]);
    let row_advanced = Line::from(vec![
        Span::styled(
            "(stdio probe sends initialize JSON-RPC, 3s timeout) ",
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    let footer = Paragraph::new(vec![row_basics, row_actions, row_advanced])
        .block(Block::default().borders(Borders::ALL).title("keys — simple → advanced"));
    f.render_widget(footer, chunks[3]);
}

fn draw_dashboard(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(8),     // models
            Constraint::Length(3),  // vram
            Constraint::Length(6),  // status / health (loaded + GPU procs)
            Constraint::Length(5),  // help footer (3 rows of keys)
        ])
        .split(f.area());

    draw_header(f, chunks[0], state);
    draw_models(f, chunks[1], state);
    draw_vram(f, chunks[2], state);
    draw_status(f, chunks[3], state);
    draw_footer(f, chunks[4]);
}

fn draw_header(f: &mut ratatui::Frame, area: Rect, state: &AppState) {
    let serve = if state.serve_up { "✓" } else { "✗" };
    let bifrost = if state.bifrost_up { "✓" } else { "✗" };
    let header = Paragraph::new(Line::from(vec![
        Span::styled("LAMU", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::raw(format!("registry={} ", state.entries.len())),
        Span::raw(format!("serve={} ", serve)),
        Span::raw(format!("bifrost={}", bifrost)),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, area);
}

fn draw_models(f: &mut ratatui::Frame, area: Rect, state: &AppState) {
    // Split off a 1-row banner above the list for column labels.
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(1)])
        .split(area);

    let header_label = format!(
        "   {:<7} {:<40}  {:>5}  {:<6}  {:>5}  {:>9}  {}",
        "SOURCE", "NAME", "PARAMS", "QUANT", "CTX", "VRAM (MB)", "CAPABILITIES",
    );
    let header_widget = Paragraph::new(header_label).style(
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );
    f.render_widget(header_widget, split[0]);

    let items: Vec<ListItem> = state
        .model_view
        .iter()
        .map(|r| match r {
            ModelRef::Local(i) => {
                let e = &state.entries[*i];
                // Compact capability tags. Drop "chat" (universal),
                // abbreviate the rest. Joined with single space inside
                // brackets: e.g. [code think long].
                let caps = e.capabilities.iter().filter_map(|c| match c {
                    lamu_core::types::Capability::Chat => None,
                    lamu_core::types::Capability::Code => Some("code"),
                    lamu_core::types::Capability::Reasoning => Some("think"),
                    lamu_core::types::Capability::Routing => Some("route"),
                    lamu_core::types::Capability::Vision => Some("vis"),
                    lamu_core::types::Capability::LongContext => Some("long"),
                }).collect::<Vec<_>>().join(" ");
                // Empty caps (chat-only models) → blank, not a stray "[]"
                let caps_field = if caps.is_empty() {
                    String::new()
                } else {
                    format!("[{}]", caps)
                };
                let notes_oneline = first_line(&e.notes);
                let fav = state.favorites.has_model(&e.name);
                let deployed = state.model_deployed(&e.name);
                let glyph = if deployed { "●" } else if fav { "★" } else { " " };
                let line = format!(
                    "{} {:<7} {:<28}  {:>4}B  {:<6}  {:>5}  {:>9}  {:<18}  {}",
                    glyph,
                    "[LOCAL]",
                    truncate(&e.name, 28),
                    format_params(e.params_b),
                    e.quant,
                    format_ctx(e.context_max),
                    e.vram_mb,
                    truncate(&caps_field, 18),
                    notes_oneline,
                );
                let style = if deployed {
                    let s = Style::default().fg(Color::Green);
                    if fav { s.add_modifier(Modifier::BOLD) } else { s }
                } else if fav {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(line).style(style)
            }
            ModelRef::Cloud(i) => {
                let m = &state.cloud_models[*i];
                let id = m.full_id();
                let fav = state.favorites.has_model(&id);
                let key_ok = m.key_present();
                // Single-column glyph so cloud rows stay column-aligned
                // with local rows. 🔒 is 2-col which pushed every
                // subsequent column right by one. `!` = key missing
                // (clearer "this row has a problem" than `?` which
                // reads as "unknown").
                let glyph = if !key_ok { "!" } else if fav { "★" } else { " " };
                // Cloud rows: blank tags column (no Capability vec) and
                // notes flow to the right. Column widths match local
                // rows so the LOCAL/CLOUD blocks line up cleanly.
                let line = format!(
                    "{} {:<7} {:<28}  {:>5}  {:<6}  {:>5}  {:>9}  {:<18}  {}",
                    glyph,
                    "[CLOUD]",
                    truncate(&id, 28),
                    "—",
                    "—",
                    format_ctx(m.context_max),
                    "—",
                    "",
                    first_line(&m.notes),
                );
                // Color rule for cloud rows:
                //   key missing               → red (can't auth at all)
                //   quota Exhausted           → red (used up)
                //   quota Low                 → yellow (running out)
                //   quota Available + key ok  → blue
                //   favorited                 → + bold
                let base = if !key_ok {
                    Color::Red
                } else {
                    match m.quota {
                        QuotaState::Available => Color::Blue,
                        QuotaState::Low => Color::Yellow,
                        QuotaState::Exhausted => Color::Red,
                    }
                };
                let mut style = Style::default().fg(base);
                if fav {
                    style = style.add_modifier(Modifier::BOLD);
                }
                ListItem::new(line).style(style)
            }
        })
        .collect();

    let header = format!("Models  [{}]", state.model_view.len());
    let footer = if state.model_filter.is_empty() {
        format!(
            "source={}  sort={}  L/C/A switch  /filter  *favorite",
            state.source_filter.label(),
            state.model_sort.label()
        )
    } else {
        format!(
            "source={}  filter='{}'  sort={}  Esc clears",
            state.source_filter.label(),
            state.model_filter,
            state.model_sort.label()
        )
    };

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Title::from(header))
                .title(Title::from(footer).position(Position::Bottom))
        )
        .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD))
        .highlight_symbol("→ ");

    let mut s = state.list_state.clone();
    f.render_stateful_widget(list, split[1], &mut s);
}

/// Render a context-window count as a compact human-readable label
/// (e.g. 131072 → "128K", 262144 → "256K", 4096 → "4K").
fn format_ctx(ctx: u32) -> String {
    if ctx >= 1024 * 1024 {
        format!("{}M", ctx / (1024 * 1024))
    } else if ctx >= 1024 {
        format!("{}K", ctx / 1024)
    } else {
        ctx.to_string()
    }
}

fn draw_vram(f: &mut ratatui::Frame, area: Rect, state: &AppState) {
    let total = state.vram.total_mb.max(1) as f64;
    let used = state.vram.used_mb as f64;
    let pct = ((used / total).min(1.0).max(0.0) * 100.0) as u16;
    let label = if state.gpu_available {
        format!(
            "VRAM {} / {} MB · {} MB free · available {} MB",
            state.vram.used_mb, state.vram.total_mb, state.vram.free_mb, state.vram.available_mb
        )
    } else {
        format!("GPU UNAVAILABLE: {}", state.gpu_reason.as_deref().unwrap_or("unknown"))
    };
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("VRAM"))
        .gauge_style(Style::default().fg(if state.gpu_available { Color::Green } else { Color::Red }))
        .percent(if state.gpu_available { pct } else { 0 })
        .label(label);
    f.render_widget(gauge, area);
}

fn draw_status(f: &mut ratatui::Frame, area: Rect, state: &AppState) {
    let mut lines: Vec<Line> = Vec::new();

    let loaded_count = state.vram.loaded_models.len();
    lines.push(Line::from(format!(
        "Loaded models: {}{}",
        loaded_count,
        if loaded_count == 0 { "  (no model in scheduler)" } else { "" }
    )));

    // GPU process snapshot: shows what's eating VRAM, even if lamu's
    // scheduler didn't spawn it. Untracked → orange so user can spot
    // orphans (`just swap`-style externally-launched llama-server, etc.).
    if state.gpu_procs.is_empty() {
        lines.push(Line::from(Span::styled(
            "GPU procs: (none)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for (pid, mb, name) in &state.gpu_procs {
            // Tracked = scheduler.loaded has any model with matching pid (we
            // don't have that mapping here, so approximate: if loaded_models
            // is empty, EVERY proc is untracked).
            let untracked = state.vram.loaded_models.is_empty();
            let line = format!("GPU proc: pid={pid:<7} {name:<24} {mb:>6} MB");
            let style = if untracked {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(line, style)));
        }
    }

    if !state.status_msg.is_empty() {
        lines.push(Line::from(Span::styled(
            state.status_msg.clone(),
            Style::default().fg(Color::Yellow),
        )));
    }

    let p = Paragraph::new(lines).block(
        Block::default().borders(Borders::ALL).title("Status — what's on the GPU"),
    );
    f.render_widget(p, area);
}

/// Cyan `[key]` chip span — used in every footer for visual consistency.
fn key_span(k: &str) -> Span<'static> {
    Span::styled(format!("[{k}]"), Style::default().fg(Color::Cyan))
}
/// Yellow chip — for actions like ★ fav that we want to call out.
fn key_span_warn(k: &str) -> Span<'static> {
    Span::styled(format!("[{k}]"), Style::default().fg(Color::Yellow))
}
/// Inline footer label after a key chip. Renamed from `label` so it
/// doesn't shadow any local `let label = ...` bindings in draw_*.
fn key_lbl(text: &str) -> Span<'static> {
    Span::raw(format!(" {text}  "))
}

fn draw_footer(f: &mut ratatui::Frame, area: Rect) {
    // Three rows, simple → advanced. Same shape on every screen so users
    // build muscle memory for where each tier of action lives.
    let row_basics = Line::from(vec![
        key_span("j/k"), key_lbl("move"),
        key_span("Enter"), key_lbl("chat"),
        key_span("r"), key_lbl("refresh"),
        key_span("q"), key_lbl("quit (×2)"),
        key_span("x"), key_lbl("instant exit"),
    ]);
    let row_list_ops = Line::from(vec![
        key_span_warn("*"), key_lbl("fav"),
        key_span("/"), key_lbl("filter"),
        key_span("o"), key_lbl("sort"),
        key_span("L/C/A"), key_lbl("source local/cloud/all"),
        key_span("a"), key_lbl("set api key"),
    ]);
    let row_advanced = Line::from(vec![
        key_span("h"), key_lbl("harnesses"),
        key_span("m"), key_lbl("mcp servers"),
        key_span("s"), key_lbl("settings"),
        key_span("S"), key_lbl("start lamu serve"),
        key_span("B"), key_lbl("start bifrost"),
        key_span("n"), key_lbl("add cloud"),
        key_span("K"), key_lbl("key status"),
        key_span("e"), key_lbl("edit cloud yaml"),
    ]);
    let footer = Paragraph::new(vec![row_basics, row_list_ops, row_advanced])
        .block(Block::default().borders(Borders::ALL).title("keys — simple → advanced"));
    f.render_widget(footer, area);
}

fn draw_launchers(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header (last harness tag)
            Constraint::Min(8),    // list
            Constraint::Length(4), // detail / status
            Constraint::Length(5), // help footer (3 rows)
        ])
        .split(f.area());

    // Header — show the most recent harness used + return-to-dashboard hint.
    let last = state.last_harness.unwrap_or("(none yet)");
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "HARNESSES",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  last="),
        Span::styled(last, Style::default().fg(Color::Yellow)),
        Span::raw("  press [d] or [Esc] to return to dashboard"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    // List
    let label = format!(
        "  {:<18}  {:<10}  {:<8}  {}",
        "NAME", "BINARY", "STATUS", "NOTES"
    );
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(1)])
        .split(chunks[1]);
    f.render_widget(
        Paragraph::new(label).style(
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        split[0],
    );

    let items: Vec<ListItem> = state
        .harness_view
        .iter()
        .map(|&i| {
            let h = &HARNESSES[i];
            let installed = *state.harness_installed.get(i).unwrap_or(&false);
            let status_label = if installed { "✓ on PATH" } else { "✗ missing" };
            let is_default = state.favorites.default_harness() == Some(h.name);
            let glyph = if is_default {
                "▶"
            } else if state.favorites.has_harness(h.name) {
                "★"
            } else {
                " "
            };
            let line = format!(
                "{} {:<18}  {:<10}  {:<8}  {}",
                glyph,
                h.name,
                h.bin,
                status_label,
                h.notes,
            );
            let mut style = if installed {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Red)
            };
            if state.favorites.has_harness(h.name) {
                style = style.add_modifier(Modifier::BOLD);
            }
            if is_default {
                style = style.add_modifier(Modifier::UNDERLINED);
            }
            ListItem::new(line).style(style)
        })
        .collect();

    let header = format!("Harnesses  [{}]", state.harness_view.len());
    let footer = if state.harness_filter.is_empty() {
        format!("sort={}  /filter  *favorite", state.harness_sort.label())
    } else {
        format!("filter='{}'  Esc clears", state.harness_filter)
    };

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Title::from(header))
                .title(Title::from(footer).position(Position::Bottom))
        )
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("→ ");
    let mut s = state.launcher_state.clone();
    f.render_stateful_widget(list, split[1], &mut s);

    // Detail pane: install hint for selected harness.
    let mut detail_lines: Vec<Line> = Vec::new();
    if let Some(h) = state.selected_harness() {
        let idx = state.launcher_state.selected().unwrap_or(0);
        let installed = *state.harness_installed.get(idx).unwrap_or(&false);
        if installed {
            detail_lines.push(Line::from(format!("[Enter] launches: {}", h.launch_argv.join(" "))));
        } else {
            detail_lines.push(Line::from(Span::styled(
                format!("Install: {}", h.install),
                Style::default().fg(Color::Yellow),
            )));
            detail_lines.push(Line::from("Press [i] to run the install command."));
        }
    }
    if !state.status_msg.is_empty() {
        detail_lines.push(Line::from(Span::styled(
            state.status_msg.clone(),
            Style::default().fg(Color::Yellow),
        )));
    }
    f.render_widget(
        Paragraph::new(detail_lines).block(Block::default().borders(Borders::ALL).title("Detail")),
        chunks[2],
    );

    let row_basics = Line::from(vec![
        key_span("j/k"), key_lbl("move"),
        key_span("Enter"), key_lbl("launch selected"),
        key_span("d/Esc/Bksp"), key_lbl("back to dashboard"),
        key_span("q"), key_lbl("quit"),
    ]);
    let row_list_ops = Line::from(vec![
        key_span_warn("*"), key_lbl("fav"),
        key_span("/"), key_lbl("filter"),
        key_span("o"), key_lbl("sort"),
        key_span("r"), key_lbl("refresh PATH detection"),
    ]);
    let row_advanced = Line::from(vec![
        key_span("i"), key_lbl("install missing harness"),
        key_span("D"), key_lbl("set/unset default harness for Dashboard Enter"),
    ]);
    let footer = Paragraph::new(vec![row_basics, row_list_ops, row_advanced])
        .block(Block::default().borders(Borders::ALL).title("keys — simple → advanced"));
    f.render_widget(footer, chunks[3]);
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}

/// Pull the first line of a multi-line string, trimmed. Empty input → "".
fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").trim().to_string()
}

fn format_params(p: f32) -> String {
    if p >= 10.0 {
        format!("{:.0}", p)
    } else {
        format!("{:.1}", p)
    }
}

fn save_api_key(var_name: &str, key_val: &str) -> std::io::Result<std::path::PathBuf> {
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
    Ok(path)
}

fn spawn_detached(argv: &[&str]) {
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
fn first_run_checks() {
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

fn prompt_yes(question: &str, default_yes: bool) -> bool {
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

fn run_blocking(argv: &[&str]) {
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

/// Check which model is loaded on :8020 and swap if it doesn't match
/// `entry`. Kills the existing llama-server, spawns a new one with the
/// optimised flags, health-polls, then warms up one token so cuBLAS is
/// ready before the chat TUI opens.
///
/// Runs inside `run_subprocess_in_tui` so stdout is visible — progress
/// lines print to the terminal while the model loads.
fn swap_to_model_if_needed(entry: &ModelEntry) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    // Check what's loaded.
    let loaded_id: Option<String> = client
        .get("http://localhost:8020/v1/models")
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|v| {
            v["data"][0]["id"]
                .as_str()
                .map(|s| s.to_lowercase())
        });

    let already_loaded = loaded_id.as_deref().map(|id| {
        id.contains(&entry.name.to_lowercase())
            || entry.name.to_lowercase().contains(id)
    }).unwrap_or(false);

    if already_loaded {
        println!("  ✓ {} already loaded", entry.name);
        return Ok(());
    }

    // Kill existing llama-server on :8020. pkill exit codes:
    //   0 — at least one process matched and was signaled
    //   1 — no matching process (already dead — fine for our case)
    //   2 — syntax error in the pattern (shouldn't happen)
    //   3 — fatal error (e.g. /proc unreadable) — surface this
    println!("\n→ Swapping model → {} ({}B {}, ~{}MB VRAM)", entry.name, entry.params_b, entry.quant, entry.vram_mb);
    match std::process::Command::new("pkill")
        .args(["-f", "llama-server.*--port 8020"])
        .status()
    {
        Ok(status) => match status.code() {
            Some(0) | Some(1) => {} // killed, or nothing to kill — both fine
            Some(2) => anyhow::bail!("pkill syntax error — internal bug, please report"),
            Some(3) => anyhow::bail!("pkill fatal error (cannot read /proc?). Check permissions."),
            // 127 (command not found), 126 (not executable), or any
            // other non-zero/one — refuse to proceed since we have no
            // confirmation the old server actually died. Spawning a new
            // one would 404 on port-bind 60s later with a misleading
            // error.
            Some(code) => anyhow::bail!(
                "pkill exited with unexpected code {} — refusing to spawn new backend. \
                 Kill the existing llama-server manually, then retry.",
                code
            ),
            None => anyhow::bail!("pkill terminated by signal — refusing to proceed"),
        },
        Err(e) => anyhow::bail!(
            "failed to spawn pkill ({}). Install procps or kill the existing llama-server manually before swapping.",
            e
        ),
    }
    // Give it a moment to release GPU mem.
    std::thread::sleep(std::time::Duration::from_secs(2));

    let bin = lamu_core::config::llama_bin();
    if !bin.exists() {
        anyhow::bail!("llama-server not found at {}", bin.display());
    }

    // Phase 4: flag construction shared with lamu-core's Backend::load and
    // lamu-mcp's build_spawn_cmd. Picking up validated LAMU_KV + ngram-mod
    // detection that the local copy here was missing.
    let supports_ngram = lamu_core::backends::llamacpp::detect_ngram_support_blocking(&bin);
    let spawn = lamu_core::backends::llamacpp::build_llama_spawn(entry, 8020, supports_ngram)?;

    let mut cmd = std::process::Command::new(&bin);
    cmd.args(&spawn.args);
    for (k, v) in &spawn.envs {
        cmd.env(k, v);
    }
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;

    // Health poll — print progress every 5s.
    let slow_client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;
    print!("  loading");
    for i in 1..=60u32 {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let healthy = slow_client
            .get("http://localhost:8020/health")
            .send()
            .ok()
            .and_then(|r| r.json::<serde_json::Value>().ok())
            .and_then(|v| v["status"].as_str().map(|s| s == "ok"))
            .unwrap_or(false);
        if healthy {
            println!(" ✓ ({}s)", i);
            // Warmup — fires cuBLAS kernel build so first real prompt is fast.
            let _ = slow_client
                .post("http://localhost:8020/v1/chat/completions")
                .timeout(std::time::Duration::from_secs(30))
                .json(&serde_json::json!({
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1, "stream": false,
                }))
                .send();
            return Ok(());
        }
        if i % 5 == 0 { print!(" {}s", i); }
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    anyhow::bail!("timeout waiting for {} to load", entry.name)
}

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
            notes: String::new(),
            status: String::new(),
        }
    }
}
