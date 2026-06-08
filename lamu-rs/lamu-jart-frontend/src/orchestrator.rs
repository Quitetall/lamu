//! Interactive deep-research TUI — the orchestrator pane (ADR 0023 frontend).
//!
//! Type a research question → `deep_research` runs → a cited report + numbered
//! sources scroll into the transcript → follow-up questions go to `research_chat`
//! grounded in the same session. Drives the lamu-jart tools IN-PROCESS via
//! `find_handler` + the server as a `&dyn ToolCtx` (no MCP loop, no self-HTTP).

use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use futures::StreamExt;
use lamu_core::tools_ext::{find_handler, ToolCtx};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Restores the terminal on any exit (return, ?, panic).
struct TermGuard;
impl Drop for TermGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
    }
}

enum Msg {
    Research(Value),
    Chat(Value),
}

struct App {
    model: String,
    input: String,
    status: String,
    lines: Vec<String>,
    session_id: Option<String>,
    busy: bool,
    scroll: u16,
    follow: bool, // stick to the bottom until the user scrolls up
}

impl App {
    fn new(model: String) -> Self {
        App {
            model,
            input: String::new(),
            status: "Type a research question · Enter to search · Ctrl-C to quit".into(),
            lines: vec![
                "lamu research — deep research over HuggingFace / PubMed / bioRxiv /".into(),
                "Semantic Scholar (+ web when keyed). Ask a question to begin.".into(),
            ],
            session_id: None,
            busy: false,
            scroll: 0,
            follow: true,
        }
    }

    fn push(&mut self, line: String) {
        self.lines.push(line);
        self.follow = true;
    }
    fn push_wrapped_block(&mut self, text: &str) {
        self.push(String::new());
        for l in text.lines() {
            self.push(l.to_string());
        }
    }

    fn on_msg(&mut self, msg: Msg) {
        self.busy = false;
        match msg {
            Msg::Research(v) => {
                if let Some(err) = v.get("synthesis_error").and_then(|e| e.as_str()) {
                    self.push_wrapped_block(&format!("(no synthesis: {err})"));
                } else if let Some(report) = v.get("report").and_then(|r| r.as_str()).filter(|s| !s.is_empty()) {
                    self.push_wrapped_block(report);
                } else if let Some(note) = v.get("note").and_then(|n| n.as_str()) {
                    self.push_wrapped_block(note);
                }
                if let Some(corpus) = v.get("corpus").and_then(|c| c.as_array()) {
                    if !corpus.is_empty() {
                        self.push(String::new());
                        self.push("Sources:".into());
                        for p in corpus {
                            let idx = p.get("idx").and_then(|i| i.as_u64()).unwrap_or(0);
                            let title = p.get("title").and_then(|t| t.as_str()).unwrap_or("");
                            let src = p.get("source").and_then(|s| s.as_str()).unwrap_or("");
                            let link = p.get("link").and_then(|l| l.as_str()).unwrap_or("");
                            self.push(format!("  [{idx}] {title} ({src})"));
                            self.push(format!("      {link}"));
                        }
                    }
                }
                self.session_id = v.get("session_id").and_then(|s| s.as_str()).map(String::from);
                self.status = if self.session_id.is_some() {
                    "Ask a follow-up · Enter · Ctrl-C to quit".into()
                } else {
                    "No session — type a new question".into()
                };
            }
            Msg::Chat(v) => {
                let ans = v
                    .get("answer")
                    .and_then(|a| a.as_str())
                    .or_else(|| v.get("error").and_then(|e| e.as_str()))
                    .unwrap_or("(no answer)");
                self.push_wrapped_block(ans);
                if let Some(cites) = v.get("citations").and_then(|c| c.as_array()) {
                    for c in cites {
                        let idx = c.get("idx").and_then(|i| i.as_u64()).unwrap_or(0);
                        let link = c.get("link").and_then(|l| l.as_str()).unwrap_or("");
                        self.push(format!("  [{idx}] {link}"));
                    }
                }
                self.status = "Ask a follow-up · Enter · Ctrl-C to quit".into();
            }
        }
    }
}

/// Run the orchestrator TUI to completion. Restores the terminal on any exit.
pub async fn run_orchestrator_tui<C: ToolCtx + 'static>(ctx: Arc<C>, model: String) -> anyhow::Result<()> {
    let deep = find_handler("deep_research")
        .ok_or_else(|| anyhow::anyhow!("deep_research tool not registered"))?
        .0;
    let chat = find_handler("research_chat")
        .ok_or_else(|| anyhow::anyhow!("research_chat tool not registered"))?
        .0;

    enable_raw_mode()?;
    execute!(std::io::stdout(), EnterAlternateScreen)?;
    let _guard = TermGuard;
    let mut term = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;

    let (tx, mut rx) = mpsc::channel::<Msg>(8);
    let mut app = App::new(model);
    let mut events = EventStream::new();

    let result = loop {
        term.draw(|f| ui(f, &mut app))?;
        tokio::select! {
            ev = events.next() => match ev {
                Some(Ok(Event::Key(k))) if k.kind == KeyEventKind::Press => {
                    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
                    match k.code {
                        KeyCode::Char('c') if ctrl => break Ok(()),
                        KeyCode::Char('q') if ctrl => break Ok(()),
                        KeyCode::Esc => app.input.clear(),
                        KeyCode::Backspace => { app.input.pop(); }
                        KeyCode::Up => { app.follow = false; app.scroll = app.scroll.saturating_sub(1); }
                        KeyCode::Down => { app.scroll = app.scroll.saturating_add(1); }
                        KeyCode::PageUp => { app.follow = false; app.scroll = app.scroll.saturating_sub(10); }
                        KeyCode::PageDown => { app.scroll = app.scroll.saturating_add(10); }
                        KeyCode::Enter => {
                            if !app.busy && !app.input.trim().is_empty() {
                                let text = std::mem::take(&mut app.input);
                                let m = app.model.clone();
                                let txc = tx.clone();
                                let c = ctx.clone();
                                app.busy = true;
                                if let Some(sid) = app.session_id.clone() {
                                    app.push(format!("❯ {text}"));
                                    app.status = "thinking…".into();
                                    tokio::spawn(async move {
                                        let args = json!({"session_id": sid, "message": text, "model": m});
                                        let out = chat(&*c as &dyn ToolCtx, args).await;
                                        let v = serde_json::from_str(&out).unwrap_or_else(|_| json!({"answer": out}));
                                        let _ = txc.send(Msg::Chat(v)).await;
                                    });
                                } else {
                                    app.push(format!("🔎 {text}"));
                                    app.status = "researching… (decompose · search · synthesize)".into();
                                    tokio::spawn(async move {
                                        let args = json!({
                                            "query": text, "sub_questions": 4, "limit_per_source": 5,
                                            "decompose_model": m, "synthesis_model": m
                                        });
                                        let out = deep(&*c as &dyn ToolCtx, args).await;
                                        let v = serde_json::from_str(&out).unwrap_or_else(|_| json!({"synthesis_error": out}));
                                        let _ = txc.send(Msg::Research(v)).await;
                                    });
                                }
                            }
                        }
                        KeyCode::Char(ch) => app.input.push(ch),
                        _ => {}
                    }
                }
                Some(Err(e)) => break Err(e.into()),
                None => break Ok(()),
                _ => {}
            },
            Some(msg) = rx.recv() => app.on_msg(msg),
        }
    };
    result
}

fn ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1), Constraint::Length(3)])
        .split(f.area());

    // Header / status.
    let head = format!("lamu research · {}{}", if app.busy { "⏳ " } else { "" }, app.status);
    f.render_widget(
        Paragraph::new(head).style(Style::default().fg(Color::Black).bg(Color::Cyan)),
        chunks[0],
    );

    // Transcript (auto-stick to bottom unless the user scrolled up).
    let body = chunks[1];
    let text = app.lines.join("\n");
    let inner_h = body.height.saturating_sub(2); // borders
    let total = app.lines.len() as u16;
    let max_scroll = total.saturating_sub(inner_h);
    if app.follow {
        app.scroll = max_scroll;
    } else {
        app.scroll = app.scroll.min(max_scroll);
        if app.scroll >= max_scroll {
            app.follow = true;
        }
    }
    f.render_widget(
        Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title(" research "))
            .wrap(Wrap { trim: false })
            .scroll((app.scroll, 0)),
        body,
    );

    // Input box.
    let prompt = if app.session_id.is_some() { "ask" } else { "research" };
    f.render_widget(
        Paragraph::new(format!("{prompt}❯ {}", app.input))
            .block(Block::default().borders(Borders::ALL).title(" ↑/↓ scroll · enter · esc clear · ctrl-c quit ")),
        chunks[2],
    );
}
