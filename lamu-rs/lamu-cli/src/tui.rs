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
        })
    }

    fn refresh(&mut self) {
        let scheduler = VramScheduler::new();
        self.vram = scheduler.budget();
        self.gpu_available = scheduler.gpu_available();
        self.gpu_reason = scheduler.gpu_unavailable_reason().map(String::from);
        self.serve_up = probe_port(8020);
        self.bifrost_up = probe_port(8080);
        self.last_refresh = Instant::now();
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
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('j') | KeyCode::Down => state.move_cursor(1),
                    KeyCode::Char('k') | KeyCode::Up => state.move_cursor(-1),
                    KeyCode::Char('r') => state.refresh(),
                    KeyCode::Char('s') => {
                        // Spawn `lamu serve` in detached mode if not up.
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
                    KeyCode::Char('c') | KeyCode::Enter => {
                        // Drop into chat — tear down TUI, run repl bound to
                        // the selected model, then restore the dashboard
                        // when /quit returns. Stage left/right via stdout
                        // directly so the generic backend doesn't need an
                        // io::Write bound.
                        if let Some(model_name) = state.selected_entry().map(|e| e.name.clone()) {
                            disable_raw_mode().ok();
                            let mut out = io::stdout();
                            execute!(out, LeaveAlternateScreen, DisableMouseCapture).ok();
                            terminal.show_cursor().ok();

                            println!("\n→ Chat with {} (/quit returns to dashboard)\n", model_name);
                            let api_url = "http://localhost:8020/v1/chat/completions".to_string();
                            if let Err(e) = crate::repl::run_repl_with_model(api_url, Some(model_name)) {
                                eprintln!("repl error: {e}");
                            }

                            // Restore dashboard
                            enable_raw_mode()?;
                            execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
                            terminal.clear()?;
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
                }
            }
        }

        if state.last_refresh.elapsed() >= Duration::from_millis(REFRESH_MS) {
            state.refresh();
        }
    }
}

fn draw(f: &mut ratatui::Frame, state: &AppState) {
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
                "{:<48}  {:>5}B  {:<6}  {:>5} MB  [{}]",
                truncate(&e.name, 48),
                format_params(e.params_b),
                e.quant,
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
    f.render_stateful_widget(list, area, &mut s);
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
        Span::raw(" select  "),
        Span::styled("[s]", Style::default().fg(Color::Cyan)),
        Span::raw(" lamu serve  "),
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
