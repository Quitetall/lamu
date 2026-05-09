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
    KeyModifiers, KeyboardEnhancementFlags, MouseEvent, MouseEventKind,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
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
use serde_json::Value;
use std::io;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

mod markdown;
mod render;
mod state;

pub use state::ChatTui;
use render::{count_wrapped_rows, draw, next_char_boundary, prev_char_boundary};

use crate::lamu_config::LamuConfig;
use crate::providers::{self, Message, Provider, Role, StreamEvent, ToolCallRef};
use crate::theme::{self, Theme};

pub(super) const API_KEY: &str = "sk-local";
pub(super) const SPINNER_TICK_MS: u128 = 90;

// Unified internal types (Message, Role, ToolCallRef, StreamEvent) and
// the per-provider format adapters live in `crate::providers`.

/// Returns a plain-text summary of the top results for the model.
pub(super) fn web_search(query: &str) -> String {
    if let Ok(key) = std::env::var("BRAVE_SEARCH_API_KEY") {
        return brave_search(query, &key);
    }
    ddg_search(query)
}

pub(super) fn brave_search(query: &str, key: &str) -> String {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build() { Ok(c) => c, Err(e) => return format!("[search error: {e}]") };
    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}&count=5",
        urlenccode(query)
    );
    let resp = match client.get(&url)
        .header("Accept", "application/json")
        .header("X-Subscription-Token", key)
        .send() { Ok(r) => r, Err(e) => return format!("[brave search error: {e}]") };
    let v: Value = match resp.json() { Ok(v) => v, Err(e) => return format!("[parse error: {e}]") };
    let mut out = String::new();
    if let Some(results) = v["web"]["results"].as_array() {
        for r in results.iter().take(5) {
            let title = r["title"].as_str().unwrap_or("");
            let url_s = r["url"].as_str().unwrap_or("");
            let desc = r["description"].as_str().unwrap_or("");
            out.push_str(&format!("**{}**\n{}\n{}\n\n", title, url_s, desc));
        }
    }
    if out.is_empty() { out = "[no results]".into(); }
    out
}

pub(super) fn ddg_search(query: &str) -> String {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("Mozilla/5.0")
        .build() { Ok(c) => c, Err(e) => return format!("[search error: {e}]") };
    let url = format!("https://lite.duckduckgo.com/lite/?q={}", urlenccode(query));
    let html = match client.get(&url).send().and_then(|r| r.text()) {
        Ok(h) => h,
        Err(e) => return format!("[ddg error: {e}]"),
    };
    // Pull out result snippets from the lite HTML — look for <td class="result-snippet">
    // and <a class="result-link"> patterns.
    let mut results: Vec<String> = Vec::new();
    let mut title = String::new();
    let mut link = String::new();
    for line in html.lines() {
        let trimmed = line.trim();
        if trimmed.contains("result-link") {
            // Extract href and text
            if let Some(href_start) = trimmed.find("href=\"") {
                let after = &trimmed[href_start + 6..];
                if let Some(href_end) = after.find('"') {
                    link = after[..href_end].to_string();
                }
            }
            title = strip_html_tags(trimmed);
        } else if trimmed.contains("result-snippet") {
            let snippet = strip_html_tags(trimmed);
            if !title.is_empty() && !snippet.is_empty() {
                results.push(format!("**{}**\n{}\n{}\n", title, link, snippet));
                title.clear(); link.clear();
            }
            if results.len() >= 5 { break; }
        }
    }
    if results.is_empty() {
        return "[no results from DuckDuckGo — try /search off or set BRAVE_SEARCH_API_KEY]".into();
    }
    results.join("\n")
}

pub(super) fn strip_html_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.trim()
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

/// Percent-encode a string for use in a URL query. Multi-byte UTF-8
/// characters are encoded as their byte sequence ("é" → "%C3%A9"),
/// not the Unicode codepoint ("%E9" — wrong; that's Latin-1, not what
/// the receiving server expects).
pub(super) fn urlenccode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            b' ' => out.push('+'),
            _ => {
                use std::fmt::Write;
                let _ = write!(out, "%{:02X}", byte);
            }
        }
    }
    out
}

pub(super) fn strip_think_blocks(text: &str) -> String {
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

pub(super) fn strip_rich_markup(s: &str) -> String {
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

pub(super) fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 { 1 }
    else if b < 0xC0 { 1 }   // continuation byte — treat alone
    else if b < 0xE0 { 2 }
    else if b < 0xF0 { 3 }
    else { 4 }
}

pub(super) fn chrono_or_timestamp() -> String {
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

/// Dispatcher. Looks up the provider by URL, builds the wire payload
/// through the provider's `build_payload`, attaches its auth headers,
/// and hands the SSE response off to its `parse_stream`.
///
/// All format-specific code lives in `providers::*`. Adding a new
/// provider means editing that module, not this function.
pub(super) fn stream_worker(
    provider: &'static (dyn Provider + Sync),
    url: String,
    api_key: String,
    model: String,
    history: Vec<Message>,
    search_enabled: bool,
    tx: Sender<StreamEvent>,
) {
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

    let payload = provider.build_payload(&model, &history, search_enabled);
    let req = client.post(&url).json(&payload);
    let req = provider.auth(req, &api_key);
    let resp = match req.send() {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(StreamEvent::Error(format!("connect {url}: {e}")));
            return;
        }
    };
    provider.parse_stream(resp, tx);
}


pub fn run(model: String, theme: Theme, config: LamuConfig) -> Result<()> {
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        eprintln!("lamu chat: stdout is not a TTY — falling back to legacy line REPL.");
        return crate::repl::run_repl_with_model(config.backend_url, Some(model));
    }

    // Layer 4 — capture a git snapshot at session start so the user
    // can `lamu undo` if the agent makes unwanted changes. Best
    // effort: failures are logged but don't block the chat.
    match crate::sandbox::snap::Snapshot::capture(&model) {
        Ok(snap) => eprintln!("[lamu] session {} snapshotted", snap.session_id),
        Err(e) => eprintln!("[lamu] snapshot failed (non-fatal): {}", e),
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
            match event::read()? {
                Event::Key(key) => {
                    if key.kind != KeyEventKind::Press { continue; }
                    if handle_key(state, key)? { return Ok(()); }
                }
                Event::Mouse(m) => handle_mouse(state, m),
                _ => {}
            }
        }
    }
}

fn handle_mouse(state: &mut ChatTui, m: MouseEvent) {
    match m.kind {
        MouseEventKind::ScrollUp => state.scroll_up(3),
        MouseEventKind::ScrollDown => state.scroll_down(3),
        _ => {}
    }
}

/// Simplified key scheme:
/// - Ctrl+C: only main exit
/// - Ctrl+S: save transcript
/// - Enter: send / Shift+Enter / Alt+Enter: newline
/// - Up/Down/PgUp/PgDn: scroll history (mouse wheel works too)
/// - Left/Right/Home/End/Backspace/Delete: cursor in input
/// - Esc: cancel exit prompt only — no other purpose
/// - All other features via slash commands (/quit /exit /save /think /search /clear /model /help)
fn handle_key(state: &mut ChatTui, key: KeyEvent) -> Result<bool> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    // Save prompt intercept — only y/n/Esc respond.
    if state.save_prompt {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                // auto_save manages is_dirty itself — sets false ONLY
                // on successful write. If save fails, is_dirty stays
                // true and Drop's silent-save catches it. Don't exit
                // on save failure — show the error and let the user
                // try again or pick (n).
                state.auto_save();
                state.save_prompt = false;
                if state.is_dirty {
                    // Save failed — keep the chat alive so the user
                    // can retry or copy out manually.
                    return Ok(false);
                }
                return Ok(true);
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                // Explicit discard — clear dirty so Drop doesn't save.
                state.is_dirty = false;
                state.save_prompt = false;
                return Ok(true);
            }
            KeyCode::Esc => {
                state.save_prompt = false;
                state.status_msg = "exit cancelled.".into();
                return Ok(false);
            }
            _ => return Ok(false),
        }
    }

    match key.code {
        // ── Exit ────────────────────────────────────────────────────
        KeyCode::Char('c') if ctrl => {
            state.quit_requested = true;
            return Ok(false);
        }
        // ── Save ────────────────────────────────────────────────────
        KeyCode::Char('s') if ctrl => {
            state.auto_save();
            return Ok(false);
        }
        // ── Scroll history ──────────────────────────────────────────
        KeyCode::Up => {
            state.scroll_up(1);
            return Ok(false);
        }
        KeyCode::Down => {
            state.scroll_down(1);
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
        // ── Send / newline ──────────────────────────────────────────
        KeyCode::Enter => {
            if shift || alt {
                state.input.insert(state.cursor, '\n');
                state.cursor += 1;
            } else {
                state.dispatch_send();
            }
            return Ok(false);
        }
        // ── Input editing ───────────────────────────────────────────
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

/// Count display rows a string takes when word-wrapped at `width`.
/// Mirrors ratatui's Paragraph wrapping behavior closely enough for
/// scroll-position math. Empty string = 1 row (the line itself).
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
        s.history.push(Message::plain(Role::User, "x".into()));
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
