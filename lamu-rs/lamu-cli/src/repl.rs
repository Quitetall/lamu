//! Interactive REPL targeting LAMU daemon's OpenAI-compat endpoint.
//!
//! Direct port of `lamu/cli/repl.py`. Talks HTTP to
//! `localhost:8020/v1/chat/completions` (or whatever URL is passed),
//! streams SSE deltas, hides `<think>...</think>` blocks by default,
//! tracks multi-turn history.
//!
//! Slash commands operate on local REPL state — model switching is
//! purely client-side; daemon-level load/unload still requires `lamu start`
//! on the MCP transport.

use anyhow::Result;
use reqwest::blocking::Client;
use rustyline::DefaultEditor;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::time::Duration;

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const DEFAULT_LIST_URL: &str = "http://localhost:8020/v1/models";
const API_KEY: &str = "sk-local";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    #[allow(dead_code)] // reserved for future system-prompt support
    System,
}

impl Role {
    fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    fn to_json(&self) -> Value {
        json!({"role": self.role.as_str(), "content": self.content})
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashCommand {
    Quit,
    Model,
    Models,
    Load,
    Unload,
    Vram,
    Think,
    Clear,
    Help,
}

impl SlashCommand {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "quit" => Some(Self::Quit),
            "model" => Some(Self::Model),
            "models" => Some(Self::Models),
            "load" => Some(Self::Load),
            "unload" => Some(Self::Unload),
            "vram" => Some(Self::Vram),
            "think" => Some(Self::Think),
            "clear" => Some(Self::Clear),
            "help" => Some(Self::Help),
            _ => None,
        }
    }
}

pub struct ReplState {
    pub api_url: String,
    pub model: String,
    pub history: Vec<Message>,
    pub show_thinking: bool,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl ReplState {
    pub fn new(api_url: String) -> Self {
        Self {
            api_url,
            model: "default".into(),
            history: Vec::new(),
            show_thinking: false,
            max_tokens: 16384,
            temperature: 0.7,
        }
    }
}

/// Parse `/cmd args...` → (Command, rest). Returns None for non-commands.
pub fn parse_command(line: &str) -> Option<(SlashCommand, String)> {
    let stripped = line.strip_prefix('/')?.trim();
    let mut parts = stripped.splitn(2, char::is_whitespace);
    let name = parts.next()?.to_lowercase();
    let rest = parts.next().unwrap_or("").trim().to_string();
    SlashCommand::parse(&name).map(|c| (c, rest))
}

/// Strip every `<think>...</think>` block. Used for the history copy so
/// reasoning doesn't bloat subsequent turns.
pub fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let bytes = text.as_bytes();
    while i < text.len() {
        match text[i..].find(THINK_OPEN) {
            Some(rel_open) => {
                let open = i + rel_open;
                out.push_str(&text[i..open]);
                match text[open..].find(THINK_CLOSE) {
                    Some(rel_close) => {
                        i = open + rel_close + THINK_CLOSE.len();
                    }
                    None => {
                        // Unterminated — drop the rest.
                        i = bytes.len();
                    }
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

/// Streaming variant: filter `<think>...</think>` from a chunk while
/// remembering whether we're mid-block across chunk boundaries.
///
/// Caller maintains an `in_think: bool` across calls. The function
/// updates it via the returned tuple.
pub fn filter_think(chunk: &str, in_think: bool, show_thinking: bool) -> (String, bool) {
    if show_thinking {
        return (chunk.to_string(), in_think);
    }
    let mut out = String::with_capacity(chunk.len());
    let mut i = 0;
    let mut state = in_think;
    while i < chunk.len() {
        if !state {
            match chunk[i..].find(THINK_OPEN) {
                Some(rel) => {
                    out.push_str(&chunk[i..i + rel]);
                    state = true;
                    i += rel + THINK_OPEN.len();
                }
                None => {
                    out.push_str(&chunk[i..]);
                    break;
                }
            }
        } else {
            match chunk[i..].find(THINK_CLOSE) {
                Some(rel) => {
                    state = false;
                    i += rel + THINK_CLOSE.len();
                }
                None => break, // rest is inside think
            }
        }
    }
    (out, state)
}

fn http_get_json(client: &Client, url: &str) -> Option<Value> {
    let resp = client
        .get(url)
        .bearer_auth(API_KEY)
        .timeout(Duration::from_secs(3))
        .send()
        .ok()?;
    resp.json::<Value>().ok()
}

fn print_help() {
    println!("\n  /quit              exit");
    println!("  /model [name]      show or set model");
    println!("  /models            list models from daemon");
    println!("  /load <name>       (MCP only) load a model");
    println!("  /unload <name>     (MCP only) unload a model");
    println!("  /vram              (MCP only) show VRAM");
    println!("  /think             toggle reasoning visibility");
    println!("  /clear             clear conversation history");
    println!("  /help              this list\n");
}

fn handle_command(
    client: &Client,
    cmd: SlashCommand,
    args: &str,
    state: &mut ReplState,
) -> bool {
    match cmd {
        SlashCommand::Quit => return false,
        SlashCommand::Help => print_help(),
        SlashCommand::Model => {
            if args.is_empty() {
                println!("current model: {}", state.model);
            } else {
                state.model = args.to_string();
                println!("model → {}", state.model);
            }
        }
        SlashCommand::Models => match http_get_json(client, DEFAULT_LIST_URL) {
            Some(data) => {
                if let Some(arr) = data.get("data").and_then(|v| v.as_array()) {
                    for m in arr {
                        if let Some(id) = m.get("id").and_then(|v| v.as_str()) {
                            println!("  {}", id);
                        }
                    }
                } else {
                    println!("(no models field in response)");
                }
            }
            None => println!("Could not reach daemon at {}", DEFAULT_LIST_URL),
        },
        SlashCommand::Load | SlashCommand::Unload | SlashCommand::Vram => {
            println!(
                "/{} requires the MCP transport (stdio). Run `lamu start` and \
                 use the MCP tools (e.g. via Claude Code), or check `lamu status`.",
                match cmd {
                    SlashCommand::Load => "load",
                    SlashCommand::Unload => "unload",
                    SlashCommand::Vram => "vram",
                    _ => unreachable!(),
                }
            );
        }
        SlashCommand::Think => {
            state.show_thinking = !state.show_thinking;
            println!(
                "thinking display: {}",
                if state.show_thinking { "ON" } else { "OFF" }
            );
        }
        SlashCommand::Clear => {
            state.history.clear();
            println!("history cleared");
        }
    }
    true
}

fn stream_chat(
    client: &Client,
    state: &mut ReplState,
    user_msg: Message,
) -> Option<Message> {
    let mut messages: Vec<Value> = state.history.iter().map(|m| m.to_json()).collect();
    messages.push(user_msg.to_json());

    let payload = json!({
        "model": state.model,
        "messages": messages,
        "stream": true,
        "max_tokens": state.max_tokens,
        "temperature": state.temperature,
    });

    let resp = match client
        .post(&state.api_url)
        .bearer_auth(API_KEY)
        .json(&payload)
        .timeout(Duration::from_secs(300))
        .send()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Connection error: {}", e);
            return None;
        }
    };

    let mut full = String::new();
    let mut in_think = false;
    let mut think_indicator_shown = false;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();

    // Live MD renderer — receives only the *visible* (post-think-filter)
    // tokens. Each push redraws the in-flight assistant message via
    // termimad; trailing scrollback above stays intact.
    let mut md = crate::md_stream::StreamMdRenderer::new();

    let reader = BufReader::new(resp);
    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(e) => {
                eprintln!("\nstream read error: {}", e);
                break;
            }
        };
        let line = line.trim();
        if !line.starts_with("data:") {
            continue;
        }
        let data = line[5..].trim();
        if data == "[DONE]" {
            break;
        }
        let chunk: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let token = chunk
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();
        if token.is_empty() {
            continue;
        }
        full.push_str(&token);

        if token.contains(THINK_OPEN) {
            in_think = true;
            if !state.show_thinking && !think_indicator_shown {
                let _ = write!(handle, "(thinking…)");
                let _ = handle.flush();
                think_indicator_shown = true;
            }
        }
        if token.contains(THINK_CLOSE) {
            in_think = false;
            if !state.show_thinking && think_indicator_shown {
                // Wipe the (thinking…) line so the MD render starts fresh.
                let _ = write!(handle, "\r\x1b[2K");
                let _ = handle.flush();
                think_indicator_shown = false;
            }
        }

        let (visible, new_state) = filter_think(&token, in_think, state.show_thinking);
        in_think = new_state;
        if !visible.is_empty() {
            let _ = md.push_token(&visible);
        }
    }

    let _ = md.finalize();
    let stripped = strip_think_blocks(&full);
    Some(Message {
        role: Role::Assistant,
        content: stripped,
    })
}

pub fn run_repl_with_model(api_url: String, model: Option<String>) -> Result<()> {
    let mut state = ReplState::new(api_url);
    if let Some(m) = model {
        state.model = m;
    }
    let client = Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;

    println!("LAMU REPL — talking to {}", state.api_url);
    println!("model: {} | /help for commands", state.model);

    let mut rl = DefaultEditor::new()?;
    loop {
        let line = match rl.readline("> ") {
            Ok(l) => l,
            Err(_) => {
                println!();
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(trimmed);

        if let Some((cmd, args)) = parse_command(trimmed) {
            if !handle_command(&client, cmd, &args, &mut state) {
                break;
            }
            continue;
        }

        let user_msg = Message {
            role: Role::User,
            content: trimmed.to_string(),
        };
        if let Some(reply) = stream_chat(&client, &mut state, user_msg.clone()) {
            state.history.push(user_msg);
            state.history.push(reply);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_command_round_trip() {
        assert_eq!(parse_command("hi"), None);
        assert_eq!(parse_command("/quit"), Some((SlashCommand::Quit, "".into())));
        assert_eq!(
            parse_command("/model qwen36"),
            Some((SlashCommand::Model, "qwen36".into()))
        );
        assert_eq!(
            parse_command("/models"),
            Some((SlashCommand::Models, "".into()))
        );
        assert_eq!(parse_command("/garbage"), None);
    }

    #[test]
    fn strip_think_simple() {
        assert_eq!(strip_think_blocks("<think>x</think>hello"), "hello");
        assert_eq!(
            strip_think_blocks("a<think>x</think>b<think>y</think>c"),
            "abc"
        );
    }

    #[test]
    fn strip_think_unterminated_drops_tail() {
        assert_eq!(strip_think_blocks("ok<think>never closes"), "ok");
    }

    #[test]
    fn filter_think_pass_through_when_show_enabled() {
        let (out, state) = filter_think("a<think>b</think>c", false, true);
        assert_eq!(out, "a<think>b</think>c");
        assert!(!state);
    }

    #[test]
    fn filter_think_carries_state_across_chunks() {
        let (out1, s1) = filter_think("hi <think>plan ", false, false);
        assert_eq!(out1, "hi ");
        assert!(s1);

        let (out2, s2) = filter_think("more</think> done", s1, false);
        assert_eq!(out2, " done");
        assert!(!s2);
    }

    #[test]
    fn filter_think_within_single_chunk() {
        let (out, s) = filter_think("before<think>x</think>after", false, false);
        assert_eq!(out, "beforeafter");
        assert!(!s);
    }

    #[test]
    fn replstate_default_values() {
        let s = ReplState::new("http://x".into());
        assert_eq!(s.api_url, "http://x");
        assert_eq!(s.model, "default");
        assert_eq!(s.max_tokens, 16384);
        assert!((s.temperature - 0.7).abs() < 1e-6);
        assert!(!s.show_thinking);
        assert!(s.history.is_empty());
    }
}
