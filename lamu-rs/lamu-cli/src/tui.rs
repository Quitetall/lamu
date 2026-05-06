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
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, ListState, Paragraph};
use ratatui::Terminal;
use std::io;
use std::time::{Duration, Instant};

const REFRESH_MS: u64 = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Dashboard,
    Launchers,
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

        Ok(Self {
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
        })
    }

    fn refresh(&mut self) {
        let scheduler = VramScheduler::new();
        self.vram = scheduler.budget();
        self.gpu_available = scheduler.gpu_available();
        self.gpu_reason = scheduler.gpu_unavailable_reason().map(String::from);
        self.serve_up = probe_port(8020);
        self.bifrost_up = probe_port(8080);
        self.harness_installed = HARNESSES.iter().map(|h| which_exists(h.bin)).collect();
        self.last_refresh = Instant::now();
    }

    fn move_launcher(&mut self, delta: i32) {
        if HARNESSES.is_empty() {
            return;
        }
        let n = HARNESSES.len() as i32;
        let cur = self.launcher_state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.launcher_state.select(Some(next));
    }

    fn selected_harness(&self) -> Option<&'static Harness> {
        self.launcher_state.selected().and_then(|i| HARNESSES.get(i))
    }

    fn move_cursor(&mut self, delta: i32) {
        if self.entries.is_empty() {
            return;
        }
        let n = self.entries.len() as i32;
        let cur = self.list_state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.list_state.select(Some(next));
    }

    fn selected_entry(&self) -> Option<&ModelEntry> {
        self.list_state.selected().and_then(|i| self.entries.get(i))
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
                match state.mode {
                    Mode::Dashboard => match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Esc => return Ok(()),
                        KeyCode::Char('j') | KeyCode::Down => state.move_cursor(1),
                        KeyCode::Char('k') | KeyCode::Up => state.move_cursor(-1),
                        KeyCode::Char('r') => state.refresh(),
                        KeyCode::Char('s') => {
                            if !state.serve_up {
                                state.status_msg = "Starting `lamu serve` on :8020 in background...".into();
                                spawn_detached(&["lamu", "serve"]);
                            } else {
                                state.status_msg = "lamu serve already running on :8020".into();
                            }
                        }
                        KeyCode::Char('b') => {
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
                        KeyCode::Char('c') | KeyCode::Enter => {
                            if let Some(model_name) = state.selected_entry().map(|e| e.name.clone()) {
                                run_subprocess_in_tui(
                                    terminal,
                                    || -> Result<()> {
                                        println!("\n→ Chat with {} (/quit returns to dashboard)\n", model_name);
                                        let api_url = "http://localhost:8020/v1/chat/completions".to_string();
                                        crate::repl::run_repl_with_model(api_url, Some(model_name.clone()))
                                    },
                                )?;
                                state.last_harness = Some("lamu repl");
                                state.refresh();
                                state.status_msg = "Returned from chat.".into();
                            }
                        }
                        KeyCode::Char('l') => {
                            if let Some(e) = state.selected_entry() {
                                state.status_msg = format!(
                                    "Use `lamu start` (MCP) and `load_model('{}')` from Claude Code to load.",
                                    e.name
                                );
                            }
                        }
                        _ => {}
                    },
                    Mode::Launchers => match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Esc | KeyCode::Char('d') => {
                            state.mode = Mode::Dashboard;
                            state.status_msg.clear();
                        }
                        KeyCode::Char('j') | KeyCode::Down => state.move_launcher(1),
                        KeyCode::Char('k') | KeyCode::Up => state.move_launcher(-1),
                        KeyCode::Char('r') => state.refresh(),
                        KeyCode::Enter => {
                            if let Some(h) = state.selected_harness() {
                                let idx = state.launcher_state.selected().unwrap_or(0);
                                let installed = *state.harness_installed.get(idx).unwrap_or(&false);
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
                                let idx = state.launcher_state.selected().unwrap_or(0);
                                let installed = *state.harness_installed.get(idx).unwrap_or(&false);
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
    }
}

fn draw_dashboard(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(8),     // models
            Constraint::Length(3),  // vram
            Constraint::Length(4),  // status / health
            Constraint::Length(3),  // help footer
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
        "  {:<48}  {:>5}  {:<6}  {:>5}  {:>9}  {}",
        "NAME", "PARAMS", "QUANT", "CTX", "VRAM (MB)", "CAPABILITIES",
    );
    let header_widget = Paragraph::new(header_label).style(
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );
    f.render_widget(header_widget, split[0]);

    let items: Vec<ListItem> = state
        .entries
        .iter()
        .map(|e| {
            let caps = e.capabilities.iter().map(|c| match c {
                lamu_core::types::Capability::Chat => "chat",
                lamu_core::types::Capability::Code => "code",
                lamu_core::types::Capability::Reasoning => "reason",
                lamu_core::types::Capability::Routing => "route",
                lamu_core::types::Capability::Vision => "vision",
                lamu_core::types::Capability::LongContext => "long",
            }).collect::<Vec<_>>().join(",");
            let line = format!(
                "{:<48}  {:>4}B  {:<6}  {:>5}  {:>9}  [{}]",
                truncate(&e.name, 48),
                format_params(e.params_b),
                e.quant,
                format_ctx(e.context_max),
                e.vram_mb,
                caps,
            );
            ListItem::new(line)
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Models — j/k to move"))
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

    if !state.status_msg.is_empty() {
        lines.push(Line::from(Span::styled(
            state.status_msg.clone(),
            Style::default().fg(Color::Yellow),
        )));
    }

    let p = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(p, area);
}

fn draw_footer(f: &mut ratatui::Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("[j/k]", Style::default().fg(Color::Cyan)),
        Span::raw(" move  "),
        Span::styled("[Enter]", Style::default().fg(Color::Cyan)),
        Span::raw(" chat  "),
        Span::styled("[h]", Style::default().fg(Color::Cyan)),
        Span::raw(" harnesses  "),
        Span::styled("[s]", Style::default().fg(Color::Cyan)),
        Span::raw(" serve  "),
        Span::styled("[b]", Style::default().fg(Color::Cyan)),
        Span::raw(" Bifrost  "),
        Span::styled("[r]", Style::default().fg(Color::Cyan)),
        Span::raw(" refresh  "),
        Span::styled("[q]", Style::default().fg(Color::Cyan)),
        Span::raw(" quit"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, area);
}

fn draw_launchers(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header (last harness tag)
            Constraint::Min(8),    // list
            Constraint::Length(4), // detail / status
            Constraint::Length(3), // help footer
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

    let items: Vec<ListItem> = HARNESSES
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let installed = *state.harness_installed.get(i).unwrap_or(&false);
            let status_label = if installed { "✓ on PATH" } else { "✗ missing" };
            let line = format!(
                "{:<18}  {:<10}  {:<8}  {}",
                h.name,
                h.bin,
                status_label,
                h.notes,
            );
            let style = if installed {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Red)
            };
            ListItem::new(line).style(style)
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Harnesses — j/k move, Enter launch, i install"))
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

    // Footer keybinds
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("[j/k]", Style::default().fg(Color::Cyan)),
        Span::raw(" move  "),
        Span::styled("[Enter]", Style::default().fg(Color::Cyan)),
        Span::raw(" launch  "),
        Span::styled("[i]", Style::default().fg(Color::Cyan)),
        Span::raw(" install  "),
        Span::styled("[r]", Style::default().fg(Color::Cyan)),
        Span::raw(" refresh  "),
        Span::styled("[d/Esc]", Style::default().fg(Color::Cyan)),
        Span::raw(" dashboard  "),
        Span::styled("[q]", Style::default().fg(Color::Cyan)),
        Span::raw(" quit"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[3]);
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

fn format_params(p: f32) -> String {
    if p >= 10.0 {
        format!("{:.0}", p)
    } else {
        format!("{:.1}", p)
    }
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
        // Synthesise three entries so the wrap logic has something to chew on.
        s.entries = (0..3).map(|i| dummy_entry(i)).collect();
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
        }
    }
}
