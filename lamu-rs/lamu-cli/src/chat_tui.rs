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
    KeyModifiers, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
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
    /// Final-answer token (delta.content).
    Token(String),
    /// Thinking token (delta.reasoning_content). Qwen3-Thinking and
    /// other reasoning models emit these *first*, before any content.
    /// Wrapping them in `<think>…</think>` here lets the existing
    /// strip_think_blocks logic gate them by show_thinking.
    Reason(String),
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
    /// embedded `\n`. Shift+Enter / Alt+Enter inserts newline; Enter sends.
    input: String,
    cursor: usize,
    /// Top-line offset into the wrapped conversation. Capped at
    /// `max_scroll` each draw so it can't run past content.
    scroll: u16,
    /// Last content-height observed during draw. Used by handle_key
    /// to clamp PgUp/PgDn without re-laying-out.
    content_height: u16,
    /// Last conversation-area height (excluding borders).
    visible_height: u16,
    /// When true, scroll snaps to the bottom on every redraw so new
    /// streamed tokens stay in view. Any explicit upward scroll
    /// (PgUp / Ctrl+K) flips this off; End / Ctrl+End / PgDn-at-bottom
    /// flips it back on.
    follow_tail: bool,
    spinner_frame: usize,
    last_spinner_tick: Instant,
    rx: Option<Receiver<StreamEvent>>,
    show_thinking: bool,
    status_msg: String,
    last_save_path: Option<String>,
    /// Conversation has unsaved messages. Flipped true on each assistant
    /// response, false when saved. Drop auto-saves if still true so
    /// unexpected exits preserve the transcript.
    is_dirty: bool,
    /// Set by /quit or any exit key path. run_loop shows the save
    /// prompt instead of returning immediately.
    quit_requested: bool,
    /// True while the "Save transcript? [y/n/Esc]" prompt is active.
    save_prompt: bool,

    // ── streaming timing ───────────────────────────────────────────
    /// Monotonic start of the current request. None when idle.
    req_started: Option<Instant>,
    /// First token arrival. (now - req_started) is TTFT (≈ prompt eval).
    first_token_at: Option<Instant>,
    /// Most recent token arrival — used for live tok/s.
    last_token_at: Option<Instant>,
    /// Tokens received this request. llama.cpp emits one delta per token.
    tokens_this_req: usize,
    /// True when the most recent stream event was reasoning_content
    /// and we're still inside the synthetic <think>…</think> block.
    /// Flipped off when the first content token arrives, or on Done.
    in_think: bool,
    /// Last completed request's TTFT in seconds.
    last_prompt_secs: Option<f32>,
    /// Last completed request's generation tok/s.
    last_gen_tps: Option<f32>,
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
            content_height: 0,
            visible_height: 0,
            follow_tail: true,
            spinner_frame: 0,
            last_spinner_tick: Instant::now(),
            rx: None,
            show_thinking: false,
            status_msg: String::new(),
            last_save_path: None,
            is_dirty: false,
            quit_requested: false,
            save_prompt: false,
            req_started: None,
            first_token_at: None,
            last_token_at: None,
            tokens_this_req: 0,
            in_think: false,
            last_prompt_secs: None,
            last_gen_tps: None,
        }
    }

    fn streaming(&self) -> bool {
        self.rx.is_some()
    }

    fn max_scroll(&self) -> u16 {
        self.content_height.saturating_sub(self.visible_height)
    }

    fn scroll_up(&mut self, n: u16) {
        self.follow_tail = false;
        self.scroll = self.scroll.saturating_sub(n);
    }

    fn scroll_down(&mut self, n: u16) {
        let max = self.max_scroll();
        self.scroll = self.scroll.saturating_add(n).min(max);
        if self.scroll >= max {
            self.follow_tail = true;
        }
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

        // Reset per-request timers. last_* fields stay until the next
        // request finishes so the previous metrics keep showing.
        self.req_started = Some(Instant::now());
        self.first_token_at = None;
        self.last_token_at = None;
        self.tokens_this_req = 0;
        self.in_think = false;

        let (tx, rx) = mpsc::channel::<StreamEvent>();
        self.rx = Some(rx);
        let url = self.config.backend_url.clone();
        let api_key = self.config.api_key.clone()
            .unwrap_or_else(|| API_KEY.to_string());
        let model = self.model.clone();
        let history: Vec<Value> = self.history.iter().map(|m| {
            let role = match m.role { Role::User => "user", Role::Assistant => "assistant", Role::System => "system" };
            json!({"role": role, "content": m.content})
        }).collect();
        thread::spawn(move || stream_worker(url, api_key, model, history, tx));
    }

    fn handle_slash(&mut self, cmd: &str) {
        let mut parts = cmd.splitn(2, char::is_whitespace);
        let head = parts.next().unwrap_or("").to_lowercase();
        let arg = parts.next().unwrap_or("").trim();
        match head.as_str() {
            "quit" | "exit" | "q" => {
                self.quit_requested = true;
            }
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
                self.status_msg = "/quit  /clear  /think  /model [name]  /save FILE  /help  Esc=quit".into();
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
                    Ok(StreamEvent::Reason(t)) => {
                        let now = Instant::now();
                        if self.first_token_at.is_none() {
                            self.first_token_at = Some(now);
                        }
                        self.last_token_at = Some(now);
                        self.tokens_this_req += 1;
                        if !self.in_think {
                            self.pending.push_str("<think>");
                            self.in_think = true;
                        }
                        self.pending.push_str(&t);
                        changed = true;
                    }
                    Ok(StreamEvent::Token(t)) => {
                        let now = Instant::now();
                        if self.first_token_at.is_none() {
                            self.first_token_at = Some(now);
                        }
                        self.last_token_at = Some(now);
                        self.tokens_this_req += 1;
                        if self.in_think {
                            self.pending.push_str("</think>\n");
                            self.in_think = false;
                        }
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
            if self.in_think {
                self.pending.push_str("</think>");
                self.in_think = false;
            }
            // Snapshot timing before we drop req state so the status
            // bar can keep showing the last numbers.
            if let (Some(start), Some(first)) = (self.req_started, self.first_token_at) {
                self.last_prompt_secs = Some(first.duration_since(start).as_secs_f32());
                if let Some(last) = self.last_token_at {
                    let gen_secs = last.duration_since(first).as_secs_f32();
                    let gen_tokens = self.tokens_this_req.saturating_sub(1);
                    if gen_secs > 0.0 && gen_tokens > 0 {
                        self.last_gen_tps = Some(gen_tokens as f32 / gen_secs);
                    } else {
                        self.last_gen_tps = None;
                    }
                }
            }
            let mut content = std::mem::take(&mut self.pending);
            if !self.show_thinking {
                content = strip_think_blocks(&content);
            }
            if !content.trim().is_empty() {
                self.history.push(Message { role: Role::Assistant, content });
                self.is_dirty = true;
            }
            self.rx = None;
            self.req_started = None;
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
            match msg.role {
                Role::Assistant => {
                    out.extend(render_markdown(&msg.content));
                }
                _ => {
                    for body_line in msg.content.split('\n') {
                        out.push(Line::from(body_line.to_string()));
                    }
                }
            }
            out.push(Line::from(""));
        }

        if !self.pending.is_empty() {
            out.push(Line::from(Span::styled(
                format!("{} streaming", asst_label.trim_end()),
                Style::default().fg(Color::Black).bg(asst_color).add_modifier(Modifier::BOLD),
            )));
            if self.show_thinking {
                for body_line in self.pending.split('\n') {
                    out.push(Line::from(body_line.to_string()));
                }
            } else {
                let visible = strip_think_blocks(&self.pending);
                if visible.trim().is_empty() && self.in_think {
                    // Still inside the think block — strip returns nothing
                    // because there's no closing tag yet. Show a dim
                    // placeholder so the user knows the model is working.
                    out.push(Line::from(Span::styled(
                        format!("  [ thinking… {} tokens  Ctrl+O to show ]", self.tokens_this_req),
                        Style::default().fg(Color::DarkGray),
                    )));
                } else {
                    for body_line in visible.split('\n') {
                        out.push(Line::from(body_line.to_string()));
                    }
                }
            }
        }
        out
    }

    fn auto_save(&mut self) {
        if self.history.is_empty() {
            self.status_msg = "nothing to save.".into();
            return;
        }
        let dir = dirs::data_local_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("lamu")
            .join("conversations");
        if std::fs::create_dir_all(&dir).is_err() {
            self.status_msg = "save failed: could not create dir.".into();
            return;
        }
        let ts = chrono_or_timestamp();
        let filename = format!("{}-{}.md", ts, self.model.replace(['/', ':'], "-"));
        let path = dir.join(&filename);
        let body: String = self.history.iter().map(|m| {
            let r = match m.role {
                Role::User => "**You**",
                Role::Assistant => "**Assistant**",
                Role::System => "**System**",
            };
            format!("{}\n\n{}\n\n---\n\n", r, m.content)
        }).collect();
        match std::fs::write(&path, &body) {
            Ok(()) => {
                let p = path.display().to_string();
                self.last_save_path = Some(p.clone());
                self.status_msg = format!("saved → {}", p);
                self.is_dirty = false;
            }
            Err(e) => self.status_msg = format!("save failed: {e}"),
        }
    }

    /// Silent save for Drop / unexpected-exit path — no status_msg update.
    fn auto_save_silent(&mut self) {
        if self.history.is_empty() { return; }
        let dir = match dirs::data_local_dir() {
            Some(d) => d.join("lamu").join("conversations"),
            None => return,
        };
        if std::fs::create_dir_all(&dir).is_err() { return; }
        let ts = chrono_or_timestamp();
        let filename = format!("crash-{}-{}.md", ts, self.model.replace(['/', ':'], "-"));
        let path = dir.join(&filename);
        let body: String = self.history.iter().map(|m| {
            let r = match m.role {
                Role::User => "**You**",
                Role::Assistant => "**Assistant**",
                Role::System => "**System**",
            };
            format!("{}\n\n{}\n\n---\n\n", r, m.content)
        }).collect();
        let _ = std::fs::write(&path, body);
    }
}

impl Drop for ChatTui {
    fn drop(&mut self) {
        if self.is_dirty && !self.history.is_empty() {
            self.auto_save_silent();
        }
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

fn chrono_or_timestamp() -> String {
    // Use std time since we don't depend on chrono.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Format as YYYYMMDD-HHMMSS using manual arithmetic
    let s = secs;
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400;
    // Days since epoch (1970-01-01) to approximate date
    // Close enough for filenames — not calendar-perfect.
    let year = 1970 + days / 365;
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", year, month.min(12), day.min(31), hour, min, sec)
}

fn stream_worker(
    url: String,
    api_key: String,
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
        .bearer_auth(&api_key)
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
        let delta = v.get("choices").and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"));
        // reasoning_content first — Qwen3-Thinking emits these before
        // any content during the prompt-eval-then-think phase.
        let reason = delta.and_then(|d| d.get("reasoning_content"))
            .and_then(|s| s.as_str()).unwrap_or("").to_string();
        if !reason.is_empty() {
            if tx.send(StreamEvent::Reason(reason)).is_err() { return; }
        }
        let token = delta.and_then(|d| d.get("content"))
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
    // Best-effort kitty keyboard protocol — lets us see Shift+Enter,
    // Alt+Enter, Ctrl+Enter as distinct events. Falls back silently on
    // terminals that don't support it (xterm/linux/tmux <3.4).
    let kitty_pushed = execute!(
        stdout,
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
        )
    ).is_ok();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = ChatTui::new(model, theme, config);
    let res = run_loop(&mut terminal, &mut state);

    if kitty_pushed {
        let _ = execute!(terminal.backend_mut(), PopKeyboardEnhancementFlags);
    }
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

        // Intercept quit: show save prompt if history is dirty.
        if state.quit_requested {
            state.quit_requested = false;
            if state.is_dirty && !state.history.is_empty() {
                state.save_prompt = true;
                state.status_msg = "Save transcript? [y] yes  [n] no  [Esc] cancel".into();
            } else {
                return Ok(());
            }
        }

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
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    // ── save prompt intercept ───────────────────────────────────────
    if state.save_prompt {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                state.auto_save();
                state.is_dirty = false;
                state.save_prompt = false;
                return Ok(true); // exit after save
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                state.is_dirty = false; // skip Drop auto-save
                state.save_prompt = false;
                return Ok(true); // exit without save
            }
            KeyCode::Esc => {
                state.save_prompt = false;
                state.status_msg = "exit cancelled.".into();
                return Ok(false);
            }
            _ => return Ok(false), // consume all other keys during prompt
        }
    }

    match key.code {
        KeyCode::Esc => {
            if state.streaming() {
                state.status_msg = "(streaming — Esc again to abort)".into();
                return Ok(false);
            }
            state.quit_requested = true;
            return Ok(false);
        }
        KeyCode::Char('c') if ctrl => {
            state.quit_requested = true;
            return Ok(false);
        }
        KeyCode::Char('o') if ctrl => {
            state.show_thinking = !state.show_thinking;
            state.status_msg = format!(
                "thinking display: {}",
                if state.show_thinking { "ON" } else { "OFF" }
            );
            return Ok(false);
        }
        KeyCode::Char('s') if ctrl => {
            state.auto_save();
            return Ok(false);
        }
        KeyCode::PageUp => {
            state.scroll_up(state.visible_height.max(1));
            return Ok(false);
        }
        KeyCode::PageDown => {
            state.scroll_down(state.visible_height.max(1));
            return Ok(false);
        }
        KeyCode::Char('k') if ctrl => {
            state.scroll_up(1);
            return Ok(false);
        }
        KeyCode::Char('j') if ctrl => {
            state.scroll_down(1);
            return Ok(false);
        }
        KeyCode::Char('g') if ctrl => {
            // Ctrl+G — jump to bottom + resume follow.
            state.follow_tail = true;
            state.scroll = state.max_scroll();
            return Ok(false);
        }
        KeyCode::Char('t') if ctrl => {
            // Ctrl+T — jump to top.
            state.follow_tail = false;
            state.scroll = 0;
            return Ok(false);
        }
        KeyCode::Enter => {
            if shift || alt {
                // Shift+Enter / Alt+Enter inserts a newline. Kitty
                // keyboard protocol delivers Shift+Enter; vt-style
                // terminals usually deliver Alt+Enter (Meta+Enter).
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

fn draw_conversation(f: &mut ratatui::Frame, area: Rect, state: &mut ChatTui) {
    let response_border = theme::hex_to_color(&state.theme.colors.response_border, Color::Cyan);

    let lines = state.build_lines();
    // Estimate wrapped content height. Without re-doing Paragraph's
    // wrapping we approximate: each logical line takes
    // ceil(len / inner_width) visual rows, min 1.
    let inner_w = area.width.saturating_sub(2).max(1) as usize;
    let mut content_h: u16 = 0;
    for line in &lines {
        let chars: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        let rows = (chars / inner_w + if chars % inner_w == 0 { 0 } else { 1 }).max(1);
        content_h = content_h.saturating_add(rows as u16);
    }
    let visible = area.height.saturating_sub(2);
    state.content_height = content_h;
    state.visible_height = visible;

    let max = content_h.saturating_sub(visible);
    let scroll = if state.follow_tail {
        state.scroll = max;
        max
    } else {
        state.scroll.min(max)
    };

    let title = if state.follow_tail || max == 0 {
        " conversation ".to_string()
    } else {
        format!(" conversation  [{}/{}] ", scroll, max)
    };

    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(response_border))
                .title(title),
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

        // Live timing while we wait. Before first token: count up the
        // prompt-eval clock. After first token: show TTFT + live tok/s.
        match (state.req_started, state.first_token_at, state.last_token_at) {
            (Some(start), None, _) => {
                let dt = start.elapsed().as_secs_f32();
                spans.push(Span::styled(
                    format!("  prompt {:.1}s", dt),
                    Style::default().fg(Color::Yellow),
                ));
            }
            (Some(start), Some(first), Some(last)) => {
                let prompt_s = first.duration_since(start).as_secs_f32();
                let gen_s = last.duration_since(first).as_secs_f32();
                let gen_n = state.tokens_this_req.saturating_sub(1);
                let tps = if gen_s > 0.0 && gen_n > 0 { gen_n as f32 / gen_s } else { 0.0 };
                spans.push(Span::styled(
                    format!("  prompt {:.1}s · {:.1} tok/s", prompt_s, tps),
                    Style::default().fg(bar_strong),
                ));
            }
            _ => {}
        }
    } else if let (Some(p), Some(g)) = (state.last_prompt_secs, state.last_gen_tps) {
        spans.push(Span::styled(
            format!("  last: prompt {:.1}s · {:.1} tok/s", p, g),
            Style::default().fg(bar_text),
        ));
    } else if let Some(p) = state.last_prompt_secs {
        spans.push(Span::styled(
            format!("  last: prompt {:.1}s", p),
            Style::default().fg(bar_text),
        ));
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
        key("Shift+Enter / Alt+Enter"), lbl("newline"),
        key("PgUp/PgDn"), lbl("scroll"),
        key("Ctrl+S"), lbl("save"),
        key("Esc / Ctrl+C"), lbl("quit"),
    ]);
    let row2 = Line::from(vec![
        key("Ctrl+O"), lbl("toggle <think>…</think>"),
        key("Ctrl+J/K"), lbl("scroll line"),
        key("Ctrl+G / Ctrl+T"), lbl("jump bottom / top"),
        key("←/→ Home/End"), lbl("cursor"),
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

/// Render a completed message body as styled ratatui Lines.
/// Handles: ``` code blocks, # headers, **bold**, `inline code`, - lists, > blockquotes.
/// Called only for finished messages, not pending (streaming stays plain).
fn render_markdown(text: &str) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code = false;

    for raw in text.split('\n') {
        // ── fenced code block ──
        if raw.trim_start().starts_with("```") {
            if in_code {
                in_code = false;
                lines.push(Line::from(Span::styled(
                    "  ────────────────────────────────".to_string(),
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                in_code = true;
                let lang = raw.trim_start().trim_start_matches('`').trim();
                let label = if lang.is_empty() {
                    "  ── code ──────────────────────────".to_string()
                } else {
                    format!("  ── {} ──────────────────────────────", lang)
                };
                lines.push(Line::from(Span::styled(label, Style::default().fg(Color::DarkGray))));
            }
            continue;
        }
        if in_code {
            lines.push(Line::from(Span::styled(
                format!("  {}", raw),
                Style::default().fg(Color::Green),
            )));
            continue;
        }

        // ── headers ──
        if let Some(rest) = raw.strip_prefix("### ") {
            lines.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        if let Some(rest) = raw.strip_prefix("## ") {
            lines.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )));
            continue;
        }
        if let Some(rest) = raw.strip_prefix("# ") {
            lines.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )));
            continue;
        }

        // ── blockquote ──
        if let Some(rest) = raw.strip_prefix("> ") {
            lines.push(Line::from(Span::styled(
                format!("│ {}", rest),
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }

        // ── bullet list ──
        let bullet_rest = if raw.starts_with("- ") || raw.starts_with("* ") {
            Some(&raw[2..])
        } else if raw.len() > 3 && raw.as_bytes()[0].is_ascii_digit() && raw.as_bytes()[1] == b'.' && raw.as_bytes()[2] == b' ' {
            Some(&raw[3..])
        } else {
            None
        };
        if let Some(rest) = bullet_rest {
            let mut spans = vec![Span::raw("  • ".to_string())];
            spans.extend(parse_inline(rest));
            lines.push(Line::from(spans));
            continue;
        }

        // ── plain line with inline markup ──
        if raw.is_empty() {
            lines.push(Line::from(""));
        } else {
            lines.push(Line::from(parse_inline(raw)));
        }
    }
    // close unclosed code block
    if in_code {
        lines.push(Line::from(Span::styled(
            "  ────────────────────────────────".to_string(),
            Style::default().fg(Color::DarkGray),
        )));
    }
    lines
}

/// Parse inline markdown: **bold**, *italic*, `code`. Returns Vec<Span<'static>>.
fn parse_inline(s: &str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // backtick inline code
        if chars[i] == '`' {
            if !buf.is_empty() {
                spans.push(Span::raw(buf.clone()));
                buf.clear();
            }
            i += 1;
            let mut code = String::new();
            while i < chars.len() && chars[i] != '`' {
                code.push(chars[i]);
                i += 1;
            }
            if i < chars.len() { i += 1; } // skip closing `
            spans.push(Span::styled(code, Style::default().fg(Color::Green)));
            continue;
        }
        // **bold**
        if i + 1 < chars.len() && chars[i] == '*' && chars[i+1] == '*' {
            if !buf.is_empty() {
                spans.push(Span::raw(buf.clone()));
                buf.clear();
            }
            i += 2;
            let mut bold = String::new();
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i+1] == '*') {
                bold.push(chars[i]);
                i += 1;
            }
            if i + 1 < chars.len() { i += 2; }
            spans.push(Span::styled(bold, Style::default().add_modifier(Modifier::BOLD)));
            continue;
        }
        // *italic* (single star, not followed by another star)
        if chars[i] == '*' && (i + 1 >= chars.len() || chars[i+1] != '*') {
            if !buf.is_empty() {
                spans.push(Span::raw(buf.clone()));
                buf.clear();
            }
            i += 1;
            let mut italic = String::new();
            while i < chars.len() && chars[i] != '*' {
                italic.push(chars[i]);
                i += 1;
            }
            if i < chars.len() { i += 1; }
            spans.push(Span::styled(italic, Style::default().add_modifier(Modifier::ITALIC)));
            continue;
        }
        buf.push(chars[i]);
        i += 1;
    }
    if !buf.is_empty() {
        spans.push(Span::raw(buf));
    }
    if spans.is_empty() {
        spans.push(Span::raw(String::new()));
    }
    spans
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
    fn slash_quit_sets_quit_requested() {
        let mut s = ChatTui::new(
            "model".into(),
            Theme::pick(Some("lamu")),
            LamuConfig::default(),
        );
        assert!(!s.quit_requested);
        s.handle_slash("quit");
        assert!(s.quit_requested);

        let mut s2 = ChatTui::new(
            "m".into(),
            Theme::pick(Some("lamu")),
            LamuConfig::default(),
        );
        s2.handle_slash("q");
        assert!(s2.quit_requested);
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
    fn scroll_up_breaks_follow() {
        let mut s = ChatTui::new("m".into(), Theme::pick(Some("lamu")), LamuConfig::default());
        s.content_height = 100;
        s.visible_height = 20;
        s.scroll = 80;
        assert!(s.follow_tail);
        s.scroll_up(8);
        assert!(!s.follow_tail);
        assert_eq!(s.scroll, 72);
    }

    #[test]
    fn scroll_down_to_bottom_resumes_follow() {
        let mut s = ChatTui::new("m".into(), Theme::pick(Some("lamu")), LamuConfig::default());
        s.content_height = 100;
        s.visible_height = 20;
        s.follow_tail = false;
        s.scroll = 70;
        s.scroll_down(20);
        assert_eq!(s.scroll, 80);
        assert!(s.follow_tail);
    }

    #[test]
    fn scroll_down_clamps_at_max() {
        let mut s = ChatTui::new("m".into(), Theme::pick(Some("lamu")), LamuConfig::default());
        s.content_height = 30;
        s.visible_height = 20;
        s.follow_tail = false;
        s.scroll = 5;
        s.scroll_down(99);
        assert_eq!(s.scroll, 10);
        assert!(s.follow_tail);
    }

    #[test]
    fn shift_enter_inserts_newline() {
        let mut s = ChatTui::new("m".into(), Theme::pick(Some("lamu")), LamuConfig::default());
        s.input.push_str("hi");
        s.cursor = 2;
        let key = KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT);
        let _ = handle_key(&mut s, key).unwrap();
        assert_eq!(s.input, "hi\n");
        assert_eq!(s.cursor, 3);
    }

    #[test]
    fn alt_enter_inserts_newline() {
        let mut s = ChatTui::new("m".into(), Theme::pick(Some("lamu")), LamuConfig::default());
        let key = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
        let _ = handle_key(&mut s, key).unwrap();
        assert_eq!(s.input, "\n");
    }

    #[test]
    fn plain_enter_does_not_newline() {
        let mut s = ChatTui::new("m".into(), Theme::pick(Some("lamu")), LamuConfig::default());
        s.input.push_str("hi");
        s.cursor = 2;
        let key = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        let _ = handle_key(&mut s, key).unwrap();
        // dispatch_send tried to POST; no newline got inserted regardless.
        assert!(!s.input.contains('\n'));
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
