//! Phase 3.6: ChatTui struct + impl + Drop.
//!
//! Holds chat history, input buffer, scroll state, theme,
//! provider/transport handles, streaming state. Methods cover:
//! input mutation, history buffer build (used by render::draw),
//! slash-command dispatch, send/abort/clear, save/load transcript,
//! token-stat tracking. Free helpers (web_search, stream_worker)
//! stay in mod.rs and are called via super::.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::Value;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::Instant;

use crate::lamu_config::LamuConfig;
use crate::providers::{self, Message, Role, StreamEvent, ToolCallRef};
use crate::theme::{self, Theme};

use super::markdown::render_markdown;
use super::{
    chrono_or_timestamp, stream_worker, strip_think_blocks,
    web_search, API_KEY, SPINNER_TICK_MS,
};

pub struct ChatTui {
    pub(super) theme: Theme,
    pub(super) config: LamuConfig,
    pub(super) model: String,
    pub(super) history: Vec<Message>,
    pub(super) pending: String,
    /// Plain string input + cursor index (byte offset). Multi-line via
    /// embedded `\n`. Shift+Enter / Alt+Enter inserts newline; Enter sends.
    pub(super) input: String,
    pub(super) cursor: usize,
    /// Top-line offset into the wrapped conversation. Capped at
    /// `max_scroll` each draw so it can't run past content.
    pub(super) scroll: u16,
    /// Last content-height observed during draw. Used by handle_key
    /// to clamp PgUp/PgDn without re-laying-out.
    pub(super) content_height: u16,
    /// Last conversation-area height (excluding borders).
    pub(super) visible_height: u16,
    /// When true, scroll snaps to the bottom on every redraw so new
    /// streamed tokens stay in view. Any explicit upward scroll
    /// (PgUp / Ctrl+K) flips this off; End / Ctrl+End / PgDn-at-bottom
    /// flips it back on.
    pub(super) follow_tail: bool,
    pub(super) spinner_frame: usize,
    pub(super) last_spinner_tick: Instant,
    pub(super) rx: Option<Receiver<StreamEvent>>,
    pub(super) show_thinking: bool,
    pub(super) status_msg: String,
    pub(super) last_save_path: Option<String>,
    /// Conversation has unsaved messages. Flipped true on each assistant
    /// response, false when saved. Drop auto-saves if still true so
    /// unexpected exits preserve the transcript.
    pub(super) is_dirty: bool,
    /// Set by /quit or any exit key path. run_loop shows the save
    /// prompt instead of returning immediately.
    pub(super) quit_requested: bool,
    /// True while the "Save transcript? [y/n/Esc]" prompt is active.
    pub(super) save_prompt: bool,

    // ── streaming timing ───────────────────────────────────────────
    /// Monotonic start of the current request. None when idle.
    pub(super) req_started: Option<Instant>,
    /// First token arrival. (now - req_started) is TTFT (≈ prompt eval).
    pub(super) first_token_at: Option<Instant>,
    /// Most recent token arrival — used for live tok/s.
    pub(super) last_token_at: Option<Instant>,
    /// Tokens received this request. llama.cpp emits one delta per token.
    pub(super) tokens_this_req: usize,
    /// True when the most recent stream event was reasoning_content
    /// and we're still inside the synthetic <think>…</think> block.
    /// Flipped off when the first content token arrives, or on Done.
    pub(super) in_think: bool,
    /// Last completed request's TTFT in seconds.
    pub(super) last_prompt_secs: Option<f32>,
    /// Last completed request's generation tok/s.
    pub(super) last_gen_tps: Option<f32>,
    /// Web search via tool calling. Toggled by /search on|off.
    pub(super) search_enabled: bool,
    /// Short message shown while a tool call is executing (e.g. "🔍 searching…").
    pub(super) tool_status: String,
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
            search_enabled: true,
            tool_status: String::new(),
        }
    }

    pub(super) fn streaming(&self) -> bool {
        self.rx.is_some()
    }

    pub(super) fn max_scroll(&self) -> u16 {
        self.content_height.saturating_sub(self.visible_height)
    }

    pub(super) fn scroll_up(&mut self, n: u16) {
        self.follow_tail = false;
        self.scroll = self.scroll.saturating_sub(n);
    }

    pub(super) fn scroll_down(&mut self, n: u16) {
        let max = self.max_scroll();
        self.scroll = self.scroll.saturating_add(n).min(max);
        if self.scroll >= max {
            self.follow_tail = true;
        }
    }

    pub(super) fn dispatch_send(&mut self) {
        if self.streaming() { return; }
        let text = self.input.trim().to_string();
        if text.is_empty() { return; }

        if let Some(cmd) = text.strip_prefix('/') {
            self.handle_slash(cmd);
            self.input.clear();
            self.cursor = 0;
            return;
        }

        self.history.push(Message::plain(Role::User, text.clone()));
        self.input.clear();
        self.cursor = 0;

        // Reset per-request timers. last_* fields stay until the next
        // request finishes so the previous metrics keep showing.
        self.req_started = Some(Instant::now());
        self.first_token_at = None;
        self.last_token_at = None;
        self.tokens_this_req = 0;
        self.in_think = false;

        self.fire_request();
    }

    /// Build and fire the next chat-completions request. Detects the
    /// provider from the backend URL and routes through the appropriate
    /// builder + parser pair.
    pub(super) fn fire_request(&mut self) {
        let (tx, rx) = mpsc::channel::<StreamEvent>();
        self.rx = Some(rx);
        let url = self.config.backend_url.clone();
        let api_key = self.config.api_key.clone()
            .unwrap_or_else(|| API_KEY.to_string());
        let model = self.model.clone();
        let search = self.search_enabled;
        let history = self.history.clone();
        let provider = providers::detect(&url);
        thread::spawn(move || stream_worker(provider, url, api_key, model, history, search, tx));
    }

    pub(super) fn handle_slash(&mut self, cmd: &str) {
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
                self.status_msg = "/quit  /clear  /think  /model [name]  /save FILE  /search [on|off]  /help  Esc=quit".into();
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
            "search" => {
                match arg {
                    "on"  => { self.search_enabled = true;  self.status_msg = "web search: ON".into(); }
                    "off" => { self.search_enabled = false; self.status_msg = "web search: OFF".into(); }
                    _ => {
                        self.search_enabled = !self.search_enabled;
                        self.status_msg = format!("web search: {}", if self.search_enabled { "ON" } else { "OFF" });
                    }
                }
            }
            other => self.status_msg = format!("unknown command: /{other}"),
        }
    }

    pub(super) fn drain_stream(&mut self) -> bool {
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
                    Ok(StreamEvent::ToolCall { id, name, arguments }) => {
                        close = true;
                        // Execute the tool, inject result, continue.
                        if name == "web_search" {
                            let query = serde_json::from_str::<Value>(&arguments)
                                .ok()
                                .and_then(|v| v["query"].as_str().map(String::from))
                                .unwrap_or_else(|| arguments.clone());
                            self.tool_status = format!("🔍 searching: {}", query);
                            // Stash any visible text the assistant emitted
                            // alongside the tool call (often empty for
                            // search-then-answer flows; non-empty when the
                            // model also writes "Let me look that up...").
                            let assistant_text = strip_think_blocks(&std::mem::take(&mut self.pending));
                            // Push the structured assistant tool_call message.
                            self.history.push(Message {
                                role: Role::Assistant,
                                content: assistant_text,
                                tool_call: Some(ToolCallRef {
                                    id: id.clone(),
                                    name: name.clone(),
                                    arguments: arguments.clone(),
                                }),
                                tool_result_for: None,
                            });
                            let results = web_search(&query);
                            // Push the structured tool result. role=User
                            // here is just an internal placeholder — the
                            // per-provider builders translate to "tool"
                            // (OpenAI) or a tool_result content block
                            // inside a user message (Anthropic).
                            self.history.push(Message {
                                role: Role::User,
                                content: results,
                                tool_call: None,
                                tool_result_for: Some(id.clone()),
                            });
                            self.tool_status.clear();
                            self.status_msg = format!("🔍 searched: {}", query);
                            // Re-fire without clearing rx (close handles it).
                            self.rx = None;
                            self.req_started = Some(Instant::now());
                            self.first_token_at = None;
                            self.last_token_at = None;
                            self.tokens_this_req = 0;
                            self.in_think = false;
                            self.fire_request();
                            return true;
                        }
                        break;
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
                self.history.push(Message::plain(Role::Assistant, content));
                self.is_dirty = true;
            }
            self.rx = None;
            self.req_started = None;
            changed = true;
        }
        changed
    }

    pub(super) fn tick_spinner(&mut self) {
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

    pub(super) fn spinner_glyph(&self) -> String {
        if self.theme.spinner.thinking_faces.is_empty() {
            const FB: &[&str] = &["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"];
            return FB[self.spinner_frame % FB.len()].to_string();
        }
        let f = &self.theme.spinner.thinking_faces;
        f[self.spinner_frame % f.len()].clone()
    }

    /// Plain-style line render of the conversation. Styled labels at
    /// turn boundaries; body text wrapped by Paragraph.
    pub(super) fn build_lines(&self) -> Vec<Line<'static>> {
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
                        format!("  [ thinking… {} tokens  /think to show ]", self.tokens_this_req),
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

    pub(super) fn auto_save(&mut self) {
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
    pub(super) fn auto_save_silent(&mut self) {
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
