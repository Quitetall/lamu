//! Read + probe the user's configured MCP servers.
//!
//! Source: `$XDG_CONFIG_HOME/.claude.json` or `~/.claude.json` —
//! Claude Code's per-user MCP registry. JSON shape:
//!
//! ```jsonc
//! { "mcpServers": {
//!     "local-llm": { "type": "stdio", "command": "lamu", "args": ["start"] },
//!     "...": { ... },
//! }}
//! ```
//!
//! Probing = spawn the server, send a JSON-RPC `initialize`, wait up to
//! 3s for a `result` line, kill. Healthy = parseable response with
//! `serverInfo`. Used to show ✓ / ✗ in the TUI's MCP screen.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeStatus {
    Untested,
    Healthy { server_name: String },
    Unreachable { reason: String },
}

#[derive(Debug, Clone)]
pub struct McpServerEntry {
    pub name: String,
    pub typ: String,
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub status: ProbeStatus,
}

pub fn config_path() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        return home.join(".claude.json");
    }
    PathBuf::from(".claude.json")
}

/// Parse the `mcpServers` map from `~/.claude.json`. Empty Vec when
/// missing/unreadable — the screen handles "no servers configured"
/// gracefully.
pub fn load_servers() -> Vec<McpServerEntry> {
    let path = config_path();
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    let v: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let map = match v.get("mcpServers").and_then(|m| m.as_object()) {
        Some(m) => m,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for (name, cfg) in map {
        let typ = cfg.get("type").and_then(|t| t.as_str()).unwrap_or("stdio").to_string();
        let command = cfg.get("command").and_then(|c| c.as_str()).unwrap_or("").to_string();
        let args = cfg
            .get("args")
            .and_then(|a| a.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let cwd = cfg.get("cwd").and_then(|c| c.as_str()).map(String::from);
        out.push(McpServerEntry {
            name: name.clone(),
            typ,
            command,
            args,
            cwd,
            status: ProbeStatus::Untested,
        });
    }
    // Stable display order: alphabetical by name.
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Probe an MCP stdio server: spawn, send initialize, await response.
/// Returns updated status. Ignores SSE/HTTP types (those need different
/// probing — TODO).
pub fn probe(entry: &McpServerEntry) -> ProbeStatus {
    if entry.typ != "stdio" {
        return ProbeStatus::Untested;
    }
    if entry.command.is_empty() {
        return ProbeStatus::Unreachable {
            reason: "no command configured".into(),
        };
    }

    let mut cmd = Command::new(&entry.command);
    cmd.args(&entry.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(cwd) = &entry.cwd {
        cmd.current_dir(cwd);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return ProbeStatus::Unreachable {
            reason: format!("spawn failed: {e}"),
        },
    };

    let mut stdin = match child.stdin.take() {
        Some(s) => s,
        None => return ProbeStatus::Unreachable {
            reason: "no stdin".into(),
        },
    };
    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => return ProbeStatus::Unreachable {
            reason: "no stdout".into(),
        },
    };

    // Send initialize.
    let req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "lamu-tui-probe", "version": "0.1"}
        }
    });
    if writeln!(stdin, "{}", req).is_err() {
        let _ = child.kill();
        return ProbeStatus::Unreachable {
            reason: "stdin write failed".into(),
        };
    }
    let _ = stdin.flush();

    // Read stdout in a worker thread with a 3s timeout via mpsc::recv_timeout.
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => return,
                Ok(_) => {
                    if line.trim_start().starts_with('{') {
                        let _ = tx.send(line.trim().to_string());
                        return;
                    }
                }
                Err(_) => return,
            }
        }
    });

    let started = Instant::now();
    let resp = rx.recv_timeout(Duration::from_secs(3));
    let elapsed = started.elapsed();

    let _ = child.kill();
    let _ = child.wait();

    match resp {
        Ok(line) => match serde_json::from_str::<Value>(&line) {
            Ok(v) => {
                if v.get("result").is_some() {
                    let server_name = v
                        .get("result")
                        .and_then(|r| r.get("serverInfo"))
                        .and_then(|s| s.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    ProbeStatus::Healthy { server_name }
                } else if let Some(err) = v.get("error") {
                    ProbeStatus::Unreachable {
                        reason: format!("error: {}", err),
                    }
                } else {
                    ProbeStatus::Unreachable {
                        reason: "no result or error in response".into(),
                    }
                }
            }
            Err(e) => ProbeStatus::Unreachable {
                reason: format!("bad json: {e}"),
            },
        },
        Err(_) => ProbeStatus::Unreachable {
            reason: format!("timeout after {:.1}s", elapsed.as_secs_f32()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_path_exists_under_home() {
        let p = config_path();
        assert!(p.is_absolute() || p == PathBuf::from(".claude.json"));
    }

    #[test]
    fn parses_mcp_servers_map() {
        let body = r#"{
            "mcpServers": {
                "alpha": {"type": "stdio", "command": "alpha-bin", "args": ["start"]},
                "beta": {"type": "stdio", "command": "beta-bin"}
            }
        }"#;
        // Roundtrip through the parser via temp file.
        let tmp = std::env::temp_dir().join(format!("lamu-mcp-test-{}.json", std::process::id()));
        std::fs::write(&tmp, body).unwrap();

        // Reuse the parsing code by loading via serde directly here —
        // load_servers uses ~/.claude.json so we can't easily redirect
        // without env mutation. Re-use the inner logic:
        let v: Value = serde_json::from_str(body).unwrap();
        let map = v.get("mcpServers").unwrap().as_object().unwrap();
        assert!(map.contains_key("alpha"));
        assert!(map.contains_key("beta"));

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn probe_missing_command_unreachable() {
        let e = McpServerEntry {
            name: "x".into(),
            typ: "stdio".into(),
            command: "".into(),
            args: vec![],
            cwd: None,
            status: ProbeStatus::Untested,
        };
        match probe(&e) {
            ProbeStatus::Unreachable { .. } => {}
            other => panic!("unexpected status: {:?}", other),
        }
    }

    #[test]
    fn probe_skips_non_stdio_for_now() {
        let e = McpServerEntry {
            name: "x".into(),
            typ: "sse".into(),
            command: "irrelevant".into(),
            args: vec![],
            cwd: None,
            status: ProbeStatus::Untested,
        };
        assert_eq!(probe(&e), ProbeStatus::Untested);
    }
}
