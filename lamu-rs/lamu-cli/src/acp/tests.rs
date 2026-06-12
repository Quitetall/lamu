//! ACP agent integration tests (ADR 0036) — fully hermetic: the "model"
//! is a scripted SSE TCP stub on a loopback port, the executed `query`
//! tool hits an in-test fake `Backend`, and the ACP wire runs over an
//! in-memory `tokio::io::duplex`. No GPU, no llama-server, no network.
//!
//! Run with an isolated `XDG_DATA_HOME` — the dispatch path touches the
//! data dir (scheduler lock probe, observability journal).

use super::agent_loop;
use super::{AcpConfig, AcpServer};
use lamu_core::backends::{Backend, ChatMessage};
use lamu_core::registry::write_registry;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{BackendType, Capability, ModelEntry, ModelFormat, ModelStatus, Modality};
use lamu_mcp::server::LamuMcpServer;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

// ── Fake local backend (the `query` tool's target) ──────────────────

struct FakeBackend {
    reply: String,
}

#[async_trait::async_trait]
impl Backend for FakeBackend {
    async fn load(&mut self, _entry: &ModelEntry, _port: u16) -> lamu_core::Result<u32> {
        Ok(0)
    }
    async fn unload(&mut self) -> lamu_core::Result<()> {
        Ok(())
    }
    async fn is_healthy(&self) -> bool {
        true
    }
    async fn generate(
        &self,
        _messages: Vec<ChatMessage>,
        _max_tokens: u32,
        _temperature: f32,
    ) -> lamu_core::Result<String> {
        Ok(self.reply.clone())
    }
    async fn stream(
        &self,
        _messages: Vec<ChatMessage>,
        _max_tokens: u32,
        _temperature: f32,
    ) -> lamu_core::Result<
        std::pin::Pin<Box<dyn futures_util::Stream<Item = lamu_core::Result<String>> + Send>>,
    > {
        Err(lamu_core::Error::Backend("fake backend has no stream".into()))
    }
    fn port(&self) -> u16 {
        0
    }
    fn model_name(&self) -> &str {
        "fake-model"
    }
}

fn fake_entry() -> ModelEntry {
    ModelEntry {
        name: "fake-model".into(),
        path: "/nonexistent/fake-model.gguf".into(),
        format: ModelFormat::Gguf,
        backend: BackendType::LlamaCpp,
        backend_kind: None,
        arch: "test".into(),
        params_b: 1.0,
        quant: "Q4_K_M".into(),
        vram_mb: 0,
        context_max: 4096,
        capabilities: vec![Capability::Chat],
        reasoning_marker: None,
        speculative: None,
        sampling: None,
        pinned: false,
        main: true,
        notes: String::new(),
        status: ModelStatus::Unspecified,
        modality: Modality::Llm,
        system_prompt: None,
    }
}

// ── Scripted SSE chat stub (the agent loop's model endpoint) ────────

enum Script {
    /// Serve these `data:` payloads then `[DONE]` and close.
    Sse(Vec<Value>),
    /// Serve these payloads then hold the socket open (never `[DONE]`).
    Stall(Vec<Value>),
}

async fn read_http_request(sock: &mut TcpStream) {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
            let need: usize = headers
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
            if buf.len() - (pos + 4) >= need {
                return;
            }
        }
        match sock.read(&mut tmp).await {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    }
}

/// Serve `scripts` sequentially, one per incoming request. Returns the
/// chat-completions URL.
async fn fake_chat_server(scripts: Vec<Script>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let mut scripts = scripts.into_iter();
        while let Ok((mut sock, _)) = listener.accept().await {
            let Some(script) = scripts.next() else { break };
            read_http_request(&mut sock).await;
            let head =
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
            let _ = sock.write_all(head.as_bytes()).await;
            let (events, stall) = match script {
                Script::Sse(e) => (e, false),
                Script::Stall(e) => (e, true),
            };
            for ev in events {
                let _ = sock.write_all(format!("data: {ev}\n\n").as_bytes()).await;
                let _ = sock.flush().await;
            }
            if stall {
                // Keep the connection open so the client stream blocks —
                // only session/cancel can end the turn.
                tokio::time::sleep(Duration::from_secs(600)).await;
            } else {
                let _ = sock.write_all(b"data: [DONE]\n\n").await;
                let _ = sock.flush().await;
            }
        }
    });
    format!("http://{addr}/v1/chat/completions")
}

// ── SSE payload builders ────────────────────────────────────────────

fn delta(d: Value) -> Value {
    json!({"choices": [{"index": 0, "delta": d, "finish_reason": null}]})
}

fn finish(reason: &str) -> Value {
    json!({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
}

fn tool_call_delta(idx: u64, id: Option<&str>, name: Option<&str>, args: &str) -> Value {
    let mut f = json!({ "arguments": args });
    if let Some(n) = name {
        f["name"] = json!(n);
    }
    let mut tc = json!({ "index": idx, "type": "function", "function": f });
    if let Some(i) = id {
        tc["id"] = json!(i);
    }
    delta(json!({ "tool_calls": [tc] }))
}

// ── Test client over an in-memory duplex ────────────────────────────

struct TestClient {
    tx: tokio::io::WriteHalf<tokio::io::DuplexStream>,
    rx: tokio::io::Lines<BufReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>>,
    // Keep temp dirs alive for the test's duration.
    _models_dir: tempfile::TempDir,
    _registry_dir: tempfile::TempDir,
}

impl TestClient {
    async fn send(&mut self, v: Value) {
        self.tx
            .write_all((v.to_string() + "\n").as_bytes())
            .await
            .expect("client write");
    }

    async fn recv(&mut self) -> Value {
        let line = tokio::time::timeout(Duration::from_secs(20), self.rx.next_line())
            .await
            .expect("recv timed out")
            .expect("client read")
            .expect("server closed pipe");
        serde_json::from_str(&line).expect("server sent non-JSON line")
    }

    /// Collect messages until the response to request `id` arrives.
    /// Returns (everything before it, the response).
    async fn collect_until_response(&mut self, id: u64) -> (Vec<Value>, Value) {
        let mut before = Vec::new();
        loop {
            let m = self.recv().await;
            if m.get("method").is_none() && m.get("id") == Some(&json!(id)) {
                return (before, m);
            }
            before.push(m);
        }
    }

    /// initialize + session/new; returns the sessionId.
    async fn handshake(&mut self, fs_caps: bool, cwd: &str) -> String {
        self.send(json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": { "readTextFile": fs_caps, "writeTextFile": fs_caps }
                }
            }
        }))
        .await;
        let init = self.recv().await;
        assert_eq!(init["result"]["protocolVersion"], 1, "init: {init}");
        assert_eq!(
            init["result"]["agentCapabilities"]["promptCapabilities"]["embeddedContext"],
            true
        );
        self.send(json!({
            "jsonrpc": "2.0", "id": 2, "method": "session/new",
            "params": { "cwd": cwd, "mcpServers": [] }
        }))
        .await;
        let new = self.recv().await;
        let sid = new["result"]["sessionId"].as_str().expect("sessionId").to_string();
        assert!(sid.starts_with("sess-"), "session id shape: {sid}");
        sid
    }

    async fn prompt(&mut self, id: u64, session: &str, text: &str) {
        self.send(json!({
            "jsonrpc": "2.0", "id": id, "method": "session/prompt",
            "params": {
                "sessionId": session,
                "prompt": [ { "type": "text", "text": text } ]
            }
        }))
        .await;
    }
}

/// Boot a full ACP server over a duplex pipe: real LamuMcpServer (temp
/// registry with `fake-model` "loaded" against a FakeBackend), agent
/// loop pointed at `chat_url`.
async fn start_acp(chat_url: String) -> TestClient {
    let models_dir = tempfile::tempdir().unwrap();
    let registry_dir = tempfile::tempdir().unwrap();
    let registry_path = registry_dir.path().join("models.yaml");
    write_registry(&[fake_entry()], &registry_path).unwrap();

    let mcp = LamuMcpServer::new(
        models_dir.path().to_path_buf(),
        registry_path,
        VramScheduler::new(),
    )
    .unwrap();
    {
        let mut st = mcp.state.lock();
        st.scheduler.register_loaded(fake_entry(), None, 1, 0);
        st.backends.insert(
            "fake-model".into(),
            Arc::new(tokio::sync::Mutex::new(
                Box::new(FakeBackend { reply: "canned answer".into() }) as Box<dyn Backend>,
            )),
        );
    }

    let cfg = AcpConfig {
        chat_url: Some(chat_url),
        model: Some("fake-model".into()),
        ..Default::default()
    };
    let acp = Arc::new(AcpServer::new(Arc::new(mcp), cfg).unwrap());

    let (client_io, server_io) = tokio::io::duplex(1 << 20);
    let (sr, sw) = tokio::io::split(server_io);
    tokio::spawn(acp.run(sr, sw));
    let (cr, cw) = tokio::io::split(client_io);
    TestClient {
        tx: cw,
        rx: BufReader::new(cr).lines(),
        _models_dir: models_dir,
        _registry_dir: registry_dir,
    }
}

fn updates_of(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .filter(|m| m.get("method").and_then(|v| v.as_str()) == Some("session/update"))
        .map(|m| m["params"]["update"].clone())
        .collect()
}

fn chunk_text(u: &Value) -> String {
    u["content"]["text"].as_str().unwrap_or("").to_string()
}

// ── Tests ───────────────────────────────────────────────────────────

#[tokio::test]
async fn protocol_round_trip_with_tool_call() {
    let url = fake_chat_server(vec![
        // Turn 1: reasoning + visible text + a `query` tool call whose
        // arguments arrive split across two deltas (ToolAcc accumulation).
        Script::Sse(vec![
            delta(json!({"role": "assistant", "reasoning_content": "thinking about it"})),
            delta(json!({"content": "Let me check."})),
            tool_call_delta(0, Some("call_1"), Some("query"), "{\"prompt\":"),
            tool_call_delta(0, None, None, "\"hi\",\"model\":\"fake-model\"}"),
            finish("tool_calls"),
        ]),
        // Turn 2 (after the tool result): plain stop.
        Script::Sse(vec![delta(json!({"content": "All done."})), finish("stop")]),
    ])
    .await;
    let mut client = start_acp(url).await;
    let sid = client.handshake(true, "/tmp").await;

    client.prompt(3, &sid, "what is up").await;
    let (before, resp) = client.collect_until_response(3).await;

    // Final response: end_turn, AFTER every update (by construction of
    // collect_until_response + the assertions below that the updates are
    // in `before`).
    assert_eq!(resp["result"]["stopReason"], "end_turn", "resp: {resp}");

    let updates = updates_of(&before);
    assert!(!updates.is_empty(), "expected session/update notifications before the response");

    // Thought chunks separated from message chunks.
    let thoughts: Vec<String> = updates
        .iter()
        .filter(|u| u["sessionUpdate"] == "agent_thought_chunk")
        .map(chunk_text)
        .collect();
    let messages: Vec<String> = updates
        .iter()
        .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
        .map(chunk_text)
        .collect();
    assert_eq!(thoughts.join(""), "thinking about it");
    assert_eq!(messages.join(""), "Let me check.All done.");

    // Tool-call lifecycle: pending → in_progress → completed, in order,
    // with the accumulated args surfaced as rawInput and the fake
    // backend's reply as rawOutput.
    let idx = |pred: &dyn Fn(&Value) -> bool| updates.iter().position(|u| pred(u));
    let started = idx(&|u| {
        u["sessionUpdate"] == "tool_call"
            && u["toolCallId"] == "call_1"
            && u["status"] == "pending"
    })
    .expect("tool_call pending update");
    let in_progress = idx(&|u| {
        u["sessionUpdate"] == "tool_call_update"
            && u["toolCallId"] == "call_1"
            && u["status"] == "in_progress"
    })
    .expect("tool_call_update in_progress");
    let completed = idx(&|u| {
        u["sessionUpdate"] == "tool_call_update"
            && u["toolCallId"] == "call_1"
            && u["status"] == "completed"
    })
    .expect("tool_call_update completed");
    assert!(started < in_progress && in_progress < completed);

    let pending = &updates[started];
    assert_eq!(pending["kind"], "think");
    assert_eq!(pending["rawInput"]["prompt"], "hi", "split args must reassemble");
    let done = &updates[completed];
    assert!(
        done["rawOutput"]["output"].as_str().unwrap_or("").contains("canned answer"),
        "rawOutput: {done}"
    );
}

#[tokio::test]
async fn cancellation_mid_prompt_returns_cancelled() {
    // The model stub streams one chunk then stalls forever — only
    // session/cancel can end the turn.
    let url = fake_chat_server(vec![Script::Stall(vec![delta(json!({"content": "partial"}))])])
        .await;
    let mut client = start_acp(url).await;
    let sid = client.handshake(false, "/tmp").await;

    client.prompt(3, &sid, "stall please").await;

    // Wait for the streamed chunk so the cancel provably lands MID-prompt.
    loop {
        let m = client.recv().await;
        if m.get("method").and_then(|v| v.as_str()) == Some("session/update")
            && m["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        {
            break;
        }
    }
    client
        .send(json!({
            "jsonrpc": "2.0", "method": "session/cancel",
            "params": { "sessionId": sid }
        }))
        .await;

    let (_, resp) = client.collect_until_response(3).await;
    assert_eq!(resp["result"]["stopReason"], "cancelled", "resp: {resp}");
}

#[tokio::test]
async fn write_file_permission_reject_once_fails_tool_and_continues() {
    let url = fake_chat_server(vec![
        Script::Sse(vec![
            tool_call_delta(
                0,
                Some("call_w1"),
                Some("write_file"),
                "{\"path\":\"a.txt\",\"content\":\"hello\"}",
            ),
            finish("tool_calls"),
        ]),
        Script::Sse(vec![delta(json!({"content": "Understood."})), finish("stop")]),
    ])
    .await;
    let mut client = start_acp(url).await;
    let sid = client.handshake(true, "/tmp/acp-test-cwd").await;

    client.prompt(3, &sid, "write a file").await;

    let mut perm_requests = 0u32;
    let mut fs_writes = 0u32;
    let mut updates: Vec<Value> = Vec::new();
    let resp = loop {
        let m = client.recv().await;
        match m.get("method").and_then(|v| v.as_str()) {
            Some("session/request_permission") => {
                perm_requests += 1;
                // The request must carry the four standard options.
                let kinds: Vec<&str> = m["params"]["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|o| o["kind"].as_str())
                    .collect();
                assert_eq!(
                    kinds,
                    vec!["allow_once", "allow_always", "reject_once", "reject_always"]
                );
                client
                    .send(json!({
                        "jsonrpc": "2.0", "id": m["id"],
                        "result": { "outcome": { "outcome": "selected", "optionId": "reject_once" } }
                    }))
                    .await;
            }
            Some("fs/write_text_file") => {
                fs_writes += 1;
                client
                    .send(json!({"jsonrpc": "2.0", "id": m["id"], "result": {}}))
                    .await;
            }
            Some("session/update") => updates.push(m["params"]["update"].clone()),
            _ if m.get("id") == Some(&json!(3)) => break m,
            _ => {}
        }
    };

    assert_eq!(perm_requests, 1);
    assert_eq!(fs_writes, 0, "rejected write must not reach the client fs");
    assert!(
        updates.iter().any(|u| u["sessionUpdate"] == "tool_call_update"
            && u["toolCallId"] == "call_w1"
            && u["status"] == "failed"),
        "rejected tool call must surface as failed: {updates:?}"
    );
    // The loop continued: the model got the rejection and ended the turn.
    assert_eq!(resp["result"]["stopReason"], "end_turn");
}

#[tokio::test]
async fn write_file_allow_always_is_cached_and_routes_client_fs() {
    let url = fake_chat_server(vec![
        Script::Sse(vec![
            tool_call_delta(
                0,
                Some("call_w1"),
                Some("write_file"),
                "{\"path\":\"a.txt\",\"content\":\"one\"}",
            ),
            finish("tool_calls"),
        ]),
        Script::Sse(vec![
            tool_call_delta(
                0,
                Some("call_w2"),
                Some("write_file"),
                "{\"path\":\"b.txt\",\"content\":\"two\"}",
            ),
            finish("tool_calls"),
        ]),
        Script::Sse(vec![delta(json!({"content": "Both written."})), finish("stop")]),
    ])
    .await;
    let mut client = start_acp(url).await;
    let cwd = "/tmp/acp-test-cwd";
    let sid = client.handshake(true, cwd).await;

    client.prompt(3, &sid, "write two files").await;

    let mut perm_requests = 0u32;
    let mut fs_paths: Vec<String> = Vec::new();
    let mut updates: Vec<Value> = Vec::new();
    let resp = loop {
        let m = client.recv().await;
        match m.get("method").and_then(|v| v.as_str()) {
            Some("session/request_permission") => {
                perm_requests += 1;
                client
                    .send(json!({
                        "jsonrpc": "2.0", "id": m["id"],
                        "result": { "outcome": { "outcome": "selected", "optionId": "allow_always" } }
                    }))
                    .await;
            }
            Some("fs/write_text_file") => {
                fs_paths.push(m["params"]["path"].as_str().unwrap_or("").to_string());
                assert_eq!(m["params"]["sessionId"], json!(sid));
                client
                    .send(json!({"jsonrpc": "2.0", "id": m["id"], "result": {}}))
                    .await;
            }
            Some("session/update") => updates.push(m["params"]["update"].clone()),
            _ if m.get("id") == Some(&json!(3)) => break m,
            _ => {}
        }
    };

    assert_eq!(perm_requests, 1, "allow_always must be cached — no re-prompt");
    assert_eq!(fs_paths, vec![format!("{cwd}/a.txt"), format!("{cwd}/b.txt")]);
    for call in ["call_w1", "call_w2"] {
        assert!(
            updates.iter().any(|u| u["sessionUpdate"] == "tool_call_update"
                && u["toolCallId"] == call
                && u["status"] == "completed"),
            "{call} must complete: {updates:?}"
        );
    }
    assert_eq!(resp["result"]["stopReason"], "end_turn");
}

// ── Unit tests (no IO) ──────────────────────────────────────────────

#[test]
fn mcp_schema_translates_to_openai_function_shape() {
    let entry = agent_loop::mcp_tool_entry("write_file").expect("built-in write_file");
    let t = agent_loop::mcp_to_openai_tool(&entry);
    assert_eq!(t["type"], "function");
    assert_eq!(t["function"]["name"], "write_file");
    assert!(t["function"]["description"].as_str().is_some());
    let params = &t["function"]["parameters"];
    assert_eq!(params["type"], "object");
    assert!(params["properties"].get("path").is_some());
    assert!(params["properties"].get("content").is_some());
    // The ACP layer injects the session id; the model never picks it.
    assert!(params["properties"].get("session_id").is_none());
    assert!(
        !params["required"].as_array().unwrap().iter().any(|v| v == "session_id"),
        "session_id must be stripped from required"
    );
}

#[test]
fn curated_tools_include_module_tools_once_registered() {
    // The composition root (main) registers lamu-jart; tests must do the
    // same to see its module tools. register() is idempotent.
    lamu_jart::register();
    let names: Vec<String> = agent_loop::curated_openai_tools()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap().to_string())
        .collect();
    for required in ["query", "remember", "recall_memory", "write_file", "web_search", "research"] {
        assert!(names.contains(&required.to_string()), "missing {required} in {names:?}");
    }
}

#[test]
fn prompt_blocks_fold_text_and_embedded_context() {
    let blocks = vec![
        json!({"type": "text", "text": "look at this"}),
        json!({"type": "resource", "resource": {"uri": "file:///x/main.rs", "mimeType": "text/x-rust", "text": "fn main() {}"}}),
        json!({"type": "resource_link", "uri": "file:///x/other.rs", "name": "other.rs"}),
    ];
    let s = agent_loop::prompt_blocks_to_text(&blocks);
    assert!(s.starts_with("look at this"));
    assert!(s.contains("<context uri=\"file:///x/main.rs\">"));
    assert!(s.contains("fn main() {}"));
    assert!(s.contains("[linked resource: file:///x/other.rs]"));
}
