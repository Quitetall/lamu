//! Themed ratatui chat. Replaces the rustyline-based REPL.
//!
//! Layout (top → bottom):
//!   ┌ banner (theme.banner_logo) ──────────────────────────────────┐
//!   ├ conversation (auto-scroll, paragraph wrap) ──────────────────┤
//!   ├ input (single multi-line buffer) ────────────────────────────┤
//!   ├ status bar (model · theme · backend · spinner · tokens) ─────┤
//!   └ footer (3 rows: simple → advanced) ──────────────────────────┘
//!
//! Streaming: SSE POST runs in a worker thread; tokens flow back via
//! mpsc. Main loop polls keys with 50ms timeout, drains the channel,
//! redraws.

use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use serde_json::{json, Value};
use std::io::{self, BufRead, BufReader};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use crate::lamu_config::LamuConfig;
use crate::theme::{self, Theme};

const API_KEY: &str = "sk-local";
const SPINNER_TICK_MS: u128 = 90;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug)]
enum StreamEvent {
    Token(String),
    Done,
    Error(String),
}

pub struct ChatTui {
    theme: Theme,
    config: LamuConfig,
    model: String,
    history: Vec<Message>,
    pending: String,
    /// Plain string input + cursor index (byte offset). Multi-line via
    /// embedded `\n`. Shift+Enter inserts newline; Enter sends.
    input: String,
    cursor: usize,
    scroll: u16,
    spinner_frame: usize,
    last_spinner_tick: Instant,
    rx: Option<Receiver<StreamEvent>>,
    show_thinking: bool,
    status_msg: String,
}

impl ChatTui {
    pub fn new(model: String, theme: Theme, config: LamuConfig) -> Self {
        Self {
            theme,
            config,
            model,
            history: Vec::new(),
            pending: String::new(),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            spinner_frame: 0,
            last_spinner_tick: Instant::now(),
            rx: None,
            show_thinking: false,
            status_msg: String::new(),
        }
    }

    fn streaming(&self) -> bool {
        self.rx.is_some()
    }

    fn dispatch_send(&mut self) {
        if self.streaming() { return; }
        let text = self.input.trim().to_string();
        if text.is_empty() { return; }

        if let Some(cmd) = text.strip_prefix('/') {
            self.handle_slash(cmd);
            self.input.clear();
            self.cursor = 0;
            return;
        }

        self.history.push(Message { role: Role::User, content: text.clone() });
        self.input.clear();
        self.cursor = 0;

        let (tx, rx) = mpsc::channel::<StreamEvent>();
        self.rx = Some(rx);
        let url = self.config.backend_url.clone();
        let model = self.model.clone();
        let history: Vec<Value> = self.history.iter().map(|m| {
            let role = match m.role { Role::User => "user", Role::Assistant => "assistant", Role::System => "system" };
            json!({"role": role, "content": m.content})
        }).collect();
        thread::spawn(move || stream_worker(url, model, history, tx));
    }

    fn handle_slash(&mut self, cmd: &str) {
        let mut parts = cmd.splitn(2, char::is_whitespace);
        let head = parts.next().unwrap_or("").to_lowercase();
        let arg = parts.next().unwrap_or("").trim();
        match head.as_str() {
            "quit" | "exit" | "q" => self.status_msg = "(use Esc or Ctrl+C to exit)".into(),
            "clear" => {
                self.history.clear();
                self.pending.clear();
                self.status_msg = "history cleared".into();
            }
            "think" => {
                self.show_thinking = !self.show_thinking;
                self.status_msg = format!(
                    "thinking display: {}",
                    if self.show_thinking { "ON" } else { "OFF" }
                );
            }
            "model" => {
                if arg.is_empty() {
                    self.status_msg = format!("current model: {}", self.model);
                } else {
                    self.model = arg.to_string();
                    self.status_msg = format!("model → {}", self.model);
                }
            }
            "help" => {
                self.status_msg = "/clear  /think  /model [name]  /save FILE  /help  Esc=quit".into();
            }
            "save" => {
                if arg.is_empty() {
                    self.status_msg = "/save needs a path".into();
                } else {
                    let body: String = self.history.iter().map(|m| {
                        let r = match m.role { Role::User => "USER", Role::Assistant => "ASSISTANT", Role::System => "SYSTEM" };
                        format!("─── {r} ───\n{}\n\n", m.content)
                    }).collect();
                    match std::fs::write(arg, body) {
                        Ok(()) => self.status_msg = format!("saved → {arg}"),
                        Err(e) => self.status_msg = format!("save failed: {e}"),
                    }
                }
            }
            other => self.status_msg = format!("unknown command: /{other}"),
        }
    }

    fn drain_stream(&mut self) -> bool {
        let mut changed = false;
        let mut close = false;
        if let Some(rx) = &self.rx {
            loop {
                match rx.try_recv() {
                    Ok(StreamEvent::Token(t)) => {
                        self.pending.push_str(&t);
                        changed = true;
                    }
                    Ok(StreamEvent::Done) => { close = true; break; }
                    Ok(StreamEvent::Error(e)) => {
                        self.pending.push_str(&format!("\n\n[stream error: {e}]"));
                        close = true;
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => { close = true; break; }
                }
            }
        }
        if close {
            let mut content = std::mem::take(&mut self.pending);
            if !self.show_thinking {
                content = strip_think_blocks(&content);
            }
            if !content.trim().is_empty() {
                self.history.push(Message { role: Role::Assistant, content });
            }
            self.rx = None;
            changed = true;
        }
        changed
    }

    fn tick_spinner(&mut self) {
        if !self.streaming() { return; }
        if self.last_spinner_tick.elapsed().as_millis() >= SPINNER_TICK_MS {
            let len = if self.theme.spinner.thinking_faces.is_empty() {
                10
            } else {
                self.theme.spinner.thinking_faces.len()
            };
            self.spinner_frame = (self.spinner_frame + 1) % len;
            self.last_spinner_tick = Instant::now();
        }
    }

    fn spinner_glyph(&self) -> String {
        if self.theme.spinner.thinking_faces.is_empty() {
            const FB: &[&str] = &["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"];
            return FB[self.spinner_frame % FB.len()].to_string();
        }
        let f = &self.theme.spinner.thinking_faces;
        f[self.spinner_frame % f.len()].clone()
    }

    /// Plain-style line render of the conversation. Styled labels at
    /// turn boundaries; body text wrapped by Paragraph.
    fn build_lines(&self) -> Vec<Line<'static>> {
        let mut out: Vec<Line<'static>> = Vec::new();
        let user_color = theme::hex_to_color(&self.theme.colors.ui_label, Color::Cyan);
        let asst_color = theme::hex_to_color(&self.theme.colors.ui_accent, Color::Yellow);
        let asst_label = if self.theme.branding.response_label.trim().is_empty() {
            " assistant ".to_string()
        } else {
            self.theme.branding.response_label.clone()
        };

        for msg in &self.history {
            let (label, color) = match msg.role {
                Role::User => (" you ".to_string(), user_color),
                Role::Assistant => (asst_label.clone(), asst_color),
                Role::System => (" system ".to_string(), Color::Gray),
            };
            out.push(Line::from(Span::styled(
                label,
                Style::default().fg(Color::Black).bg(color).add_modifier(Modifier::BOLD),
            )));
            for body_line in msg.content.split('\n') {
                out.push(Line::from(body_line.to_string()));
            }
            out.push(Line::from(""));
        }

        if !self.pending.is_empty() {
            out.push(Line::from(Span::styled(
                format!("{} streaming", asst_label.trim_end()),
                Style::default().fg(Color::Black).bg(asst_color).add_modifier(Modifier::BOLD),
            )));
            let display = if self.show_thinking {
                self.pending.clone()
            } else {
                strip_think_blocks(&self.pending)
            };
            for body_line in display.split('\n') {
                out.push(Line::from(body_line.to_string()));
            }
        }
        out
    }
}

fn strip_think_blocks(text: &str) -> String {
    let open = "<think>";
    let close = "</think>";
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        match text[i..].find(open) {
            Some(rel) => {
                out.push_str(&text[i..i + rel]);
                let after = i + rel + open.len();
                match text[after..].find(close) {
                    Some(crel) => i = after + crel + close.len(),
                    None => break,
                }
            }
            None => {
                out.push_str(&text[i..]);
                break;
            }
        }
    }
    out.trim().to_string()
}

fn strip_rich_markup(s: &str) -> String {
    // Drop Rich/Textual `[bold #xxxxxx]…[/]` tags so banner art renders
    // as plain text colored uniformly. Real Rich-style render is a
    // future commit.
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some(end) = s[i..].find(']') {
                i += end + 1;
                continue;
            }
        }
        // Walk one char (UTF-8) at a time so braille / box drawing
        // characters don't get cut mid-codepoint.
        let ch_len = utf8_char_len(bytes[i]);
        let end = (i + ch_len).min(bytes.len());
        out.push_str(&s[i..end]);
        i = end;
    }
    out
}

fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 { 1 }
    else if b < 0xC0 { 1 }   // continuation byte — treat alone
    else if b < 0xE0 { 2 }
    else if b < 0xF0 { 3 }
    else { 4 }
}

fn stream_worker(
    url: String,
    model: String,
    messages: Vec<Value>,
    tx: Sender<StreamEvent>,
) {
    let payload = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "max_tokens": 16384,
        "temperature": 0.7,
    });
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(StreamEvent::Error(format!("client init: {e}")));
            return;
        }
    };
    let resp = match client
        .post(&url)
        .bearer_auth(API_KEY)
        .json(&payload)
        .send()
    {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(StreamEvent::Error(format!("connect {url}: {e}")));
            return;
        }
    };
    let reader = BufReader::new(resp);
    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(format!("read: {e}")));
                let _ = tx.send(StreamEvent::Done);
                return;
            }
        };
        let line = line.trim();
        if !line.starts_with("data:") { continue; }
        let data = line[5..].trim();
        if data == "[DONE]" { break; }
        let v: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let token = v.get("choices").and_then(|c| c.get(0))
            .and_then(|c| c.get("delta")).and_then(|d| d.get("content"))
            .and_then(|s| s.as_str()).unwrap_or("").to_string();
        if !token.is_empty() {
            if tx.send(StreamEvent::Token(token)).is_err() { return; }
        }
    }
    let _ = tx.send(StreamEvent::Done);
}

pub fn run(model: String, theme: Theme, config: LamuConfig) -> Result<()> {
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        eprintln!("lamu chat: stdout is not a TTY — falling back to legacy line REPL.");
        return crate::repl::run_repl_with_model(config.backend_url, Some(model));
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = ChatTui::new(model, theme, config);
    let res = run_loop(&mut terminal, &mut state);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    res
}

fn run_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    state: &mut ChatTui,
) -> Result<()> {
    loop {
        let _ = state.drain_stream();
        state.tick_spinner();
        terminal.draw(|f| draw(f, state))?;

        if event::poll(Duration::from_millis(50))? {
            let ev = event::read()?;
            if let Event::Key(key) = ev {
                if key.kind != KeyEventKind::Press { continue; }
                if handle_key(state, key)? { return Ok(()); }
            }
        }
    }
}

fn handle_key(state: &mut ChatTui, key: KeyEvent) -> Result<bool> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    match key.code {
        KeyCode::Esc => {
            if state.streaming() {
                state.status_msg = "(streaming — Esc again to abort)".into();
                return Ok(false);
            }
            return Ok(true);
        }
        KeyCode::Char('c') if ctrl => return Ok(true),
        KeyCode::Char('o') if ctrl => {
            state.show_thinking = !state.show_thinking;
            state.status_msg = format!(
                "thinking display: {}",
                if state.show_thinking { "ON" } else { "OFF" }
            );
            return Ok(false);
        }
        KeyCode::PageUp => {
            state.scroll = state.scroll.saturating_sub(8);
            return Ok(false);
        }
        KeyCode::PageDown => {
            state.scroll = state.scroll.saturating_add(8);
            return Ok(false);
        }
        KeyCode::Char('k') if ctrl => {
            state.scroll = state.scroll.saturating_sub(1);
            return Ok(false);
        }
        KeyCode::Char('j') if ctrl => {
            state.scroll = state.scroll.saturating_add(1);
            return Ok(false);
        }
        KeyCode::Enter => {
            if shift {
                // Shift+Enter inserts a newline.
                state.input.insert(state.cursor, '\n');
                state.cursor += 1;
            } else {
                state.dispatch_send();
            }
            return Ok(false);
        }
        KeyCode::Backspace => {
            if state.cursor > 0 {
                let prev = prev_char_boundary(&state.input, state.cursor);
                state.input.replace_range(prev..state.cursor, "");
                state.cursor = prev;
            }
            return Ok(false);
        }
        KeyCode::Delete => {
            if state.cursor < state.input.len() {
                let next = next_char_boundary(&state.input, state.cursor);
                state.input.replace_range(state.cursor..next, "");
            }
            return Ok(false);
        }
        KeyCode::Left => {
            if state.cursor > 0 {
                state.cursor = prev_char_boundary(&state.input, state.cursor);
            }
            return Ok(false);
        }
        KeyCode::Right => {
            if state.cursor < state.input.len() {
                state.cursor = next_char_boundary(&state.input, state.cursor);
            }
            return Ok(false);
        }
        KeyCode::Home => {
            state.cursor = 0;
            return Ok(false);
        }
        KeyCode::End => {
            state.cursor = state.input.len();
            return Ok(false);
        }
        KeyCode::Char(c) => {
            state.input.insert(state.cursor, c);
            state.cursor += c.len_utf8();
            return Ok(false);
        }
        _ => {}
    }
    Ok(false)
}

fn prev_char_boundary(s: &str, i: usize) -> usize {
    let mut j = i.saturating_sub(1);
    while j > 0 && !s.is_char_boundary(j) {
        j -= 1;
    }
    j
}
fn next_char_boundary(s: &str, i: usize) -> usize {
    let mut j = i + 1;
    while j < s.len() && !s.is_char_boundary(j) {
        j += 1;
    }
    j
}

fn draw(f: &mut ratatui::Frame, state: &mut ChatTui) {
    let banner_h = if state.theme.banner_logo.trim().is_empty() {
        3
    } else {
        (state.theme.banner_logo.lines().count().min(8) as u16) + 2
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(banner_h),
            Constraint::Min(8),
            Constraint::Length(5),  // input
            Constraint::Length(3),  // status
            Constraint::Length(5),  // footer
        ])
        .split(f.area());

    draw_banner(f, chunks[0], state);
    draw_conversation(f, chunks[1], state);
    draw_input(f, chunks[2], state);
    draw_status(f, chunks[3], state);
    draw_footer(f, chunks[4]);
}

fn draw_banner(f: &mut ratatui::Frame, area: Rect, state: &ChatTui) {
    let border = theme::hex_to_color(&state.theme.colors.banner_border, Color::DarkGray);
    let title_color = theme::hex_to_color(&state.theme.colors.banner_title, Color::Cyan);

    let mut lines: Vec<Line> = Vec::new();
    let logo = &state.theme.banner_logo;
    if logo.trim().is_empty() {
        let agent = if state.theme.branding.agent_name.is_empty() {
            "lamu".to_string()
        } else {
            state.theme.branding.agent_name.clone()
        };
        lines.push(Line::from(Span::styled(
            agent,
            Style::default().fg(title_color).add_modifier(Modifier::BOLD),
        )));
    } else {
        for raw in logo.lines().take(8) {
            lines.push(Line::from(Span::styled(
                strip_rich_markup(raw),
                Style::default().fg(title_color),
            )));
        }
    }
    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border)));
    f.render_widget(p, area);
}

fn draw_conversation(f: &mut ratatui::Frame, area: Rect, state: &ChatTui) {
    let response_border = theme::hex_to_color(&state.theme.colors.response_border, Color::Cyan);

    let lines = state.build_lines();
    let content_height = lines.len() as u16;
    let visible = area.height.saturating_sub(2);
    // Auto-scroll to bottom if user hasn't scrolled up.
    let scroll = if state.scroll == 0 {
        content_height.saturating_sub(visible)
    } else {
        state.scroll.min(content_height.saturating_sub(visible))
    };

    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(response_border))
                .title(" conversation "),
        );
    f.render_widget(p, area);
}

fn draw_input(f: &mut ratatui::Frame, area: Rect, state: &ChatTui) {
    let rule = theme::hex_to_color(&state.theme.colors.input_rule, Color::DarkGray);
    let prompt = if state.theme.branding.prompt_symbol.is_empty() {
        "❯ ".to_string()
    } else {
        state.theme.branding.prompt_symbol.clone()
    };
    let title = if state.streaming() {
        " input — locked while streaming ".to_string()
    } else {
        format!(" {} ", prompt.trim())
    };

    // Render input as styled text with cursor marker. Multi-line splits
    // on '\n' and Paragraph wraps the rest.
    let mut lines: Vec<Line> = Vec::new();
    if state.input.is_empty() && !state.streaming() {
        lines.push(Line::from(Span::styled(
            "(type a message — /help for commands)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        // Split input + insert visible cursor mark
        let mut shown = state.input.clone();
        let cur = state.cursor.min(shown.len());
        if !state.streaming() {
            shown.insert_str(cur, "▏");
        }
        for body_line in shown.split('\n') {
            lines.push(Line::from(body_line.to_string()));
        }
    }
    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(rule))
                .title(title),
        );
    f.render_widget(p, area);
}

fn draw_status(f: &mut ratatui::Frame, area: Rect, state: &ChatTui) {
    let bar_bg = theme::hex_to_color(&state.theme.colors.status_bar_bg, Color::Black);
    let bar_text = theme::hex_to_color(&state.theme.colors.status_bar_text, Color::Gray);
    let bar_strong = theme::hex_to_color(&state.theme.colors.status_bar_strong, Color::Cyan);
    let good = theme::hex_to_color(&state.theme.colors.status_bar_good, Color::Green);

    let total_tokens: usize = state.history.iter().map(|m| m.content.len() / 4).sum::<usize>()
        + state.pending.len() / 4;
    let backend = state.config.backend_label();
    let theme_label = state.theme.name.clone();

    let mut spans: Vec<Span> = Vec::new();
    spans.push(Span::styled("MODEL ", Style::default().fg(bar_text)));
    spans.push(Span::styled(state.model.clone(), Style::default().fg(bar_strong).add_modifier(Modifier::BOLD)));
    spans.push(Span::styled(format!("  THEME {theme_label}"), Style::default().fg(bar_text)));
    spans.push(Span::styled(format!("  BACKEND {backend}"), Style::default().fg(bar_text)));
    spans.push(Span::styled(format!("  ~{total_tokens} tok"), Style::default().fg(bar_text)));
    if state.streaming() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(state.spinner_glyph(), Style::default().fg(good).add_modifier(Modifier::BOLD)));
        spans.push(Span::styled(" streaming…", Style::default().fg(good)));
    }
    if state.show_thinking {
        spans.push(Span::styled("  [think:ON]", Style::default().fg(Color::Yellow)));
    }
    if !state.status_msg.is_empty() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(state.status_msg.clone(), Style::default().fg(Color::Yellow)));
    }

    let p = Paragraph::new(Line::from(spans))
        .style(Style::default().bg(bar_bg))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(p, area);
}

fn draw_footer(f: &mut ratatui::Frame, area: Rect) {
    let key = |k: &str| Span::styled(format!("[{k}]"), Style::default().fg(Color::Cyan));
    let lbl = |t: &str| Span::raw(format!(" {t}  "));
    let row1 = Line::from(vec![
        key("Enter"), lbl("send"),
        key("Shift+Enter"), lbl("newline"),
        key("Esc / Ctrl+C"), lbl("quit"),
        key("PgUp/PgDn"), lbl("scroll"),
    ]);
    let row2 = Line::from(vec![
        key("Ctrl+O"), lbl("toggle <think>…</think>"),
        key("Ctrl+J/K"), lbl("scroll one line"),
        key("←/→  Home/End"), lbl("cursor"),
    ]);
    let row3 = Line::from(vec![
        key("/help"), lbl("commands"),
        key("/clear"), lbl("wipe"),
        key("/think"), lbl("toggle"),
        key("/model"), lbl("show or set model"),
        key("/save FILE"), lbl("export transcript"),
    ]);
    let p = Paragraph::new(vec![row1, row2, row3])
        .block(Block::default().borders(Borders::ALL).title("keys — simple → advanced"));
    f.render_widget(p, area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_think_simple() {
        assert_eq!(strip_think_blocks("a<think>x</think>b"), "ab");
        assert_eq!(strip_think_blocks("hello"), "hello");
    }

    #[test]
    fn strip_think_unterminated_drops_tail() {
        assert_eq!(strip_think_blocks("ok<think>never closes"), "ok");
    }

    #[test]
    fn strip_rich_markup_drops_tags() {
        assert_eq!(strip_rich_markup("[bold #FF0000]Hello[/]"), "Hello");
        assert_eq!(strip_rich_markup("plain"), "plain");
    }

    #[test]
    fn slash_help_sets_status() {
        let mut s = ChatTui::new(
            "model".into(),
            Theme::pick(Some("lamu")),
            LamuConfig::default(),
        );
        s.handle_slash("help");
        assert!(s.status_msg.contains("/clear"));
    }

    #[test]
    fn slash_think_toggles() {
        let mut s = ChatTui::new(
            "model".into(),
            Theme::pick(Some("lamu")),
            LamuConfig::default(),
        );
        let before = s.show_thinking;
        s.handle_slash("think");
        assert_ne!(s.show_thinking, before);
    }

    #[test]
    fn slash_clear_wipes_history() {
        let mut s = ChatTui::new(
            "model".into(),
            Theme::pick(Some("lamu")),
            LamuConfig::default(),
        );
        s.history.push(Message { role: Role::User, content: "x".into() });
        s.handle_slash("clear");
        assert!(s.history.is_empty());
    }

    #[test]
    fn input_insert_advances_cursor_utf8() {
        let mut s = ChatTui::new("m".into(), Theme::pick(Some("lamu")), LamuConfig::default());
        s.input.push_str("hi");
        s.cursor = 2;
        s.input.insert(s.cursor, '😀');
        s.cursor += '😀'.len_utf8();
        assert_eq!(s.cursor, 6);
        assert_eq!(s.input, "hi😀");
    }

    #[test]
    fn prev_next_char_boundary_handles_multibyte() {
        let s = "a😀b";
        // bytes: a=1, 😀=4, b=1
        let after_smiley = 5;
        assert_eq!(next_char_boundary(s, 1), 5);
        assert_eq!(prev_char_boundary(s, after_smiley), 1);
    }
}
