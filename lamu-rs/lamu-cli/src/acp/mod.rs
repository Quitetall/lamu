//! Native ACP (Agent Client Protocol) agent surface — `lamu acp` (ADR 0036).
//!
//! An ACP client (Zed's "external agent" support, or anything speaking
//! agentclientprotocol.com) spawns `lamu acp` and drives it over stdio.
//! The agent loop runs a LOCAL model (resolved through the registry /
//! VRAM scheduler) against a curated subset of the in-process LAMU tool
//! surface, streaming visible text, reasoning, and tool-call lifecycle
//! back as `session/update` notifications.
//!
//! # Protocol-layer decision (for ADR 0036)
//!
//! Evaluated Zed's official `agent-client-protocol` crate (v0.14.0,
//! crates.io) and chose to **hand-roll the wire types** on the lamu-mcp
//! stdio-loop pattern instead. Rationale:
//!
//! - **Dep tree**: the crate pulls ~10 transitive deps that are new to
//!   this workspace (`blocking`'s thread pool, `async-process`,
//!   `futures-concurrency`, `schemars` 1.x, `jsonrpcmsg`, `shell-words`,
//!   `uuid`, a proc-macro derive crate, and an `=`-pinned schema crate).
//!   The protocol subset we need is ~15 serde shapes.
//! - **Architecture fit**: its connection layer is runtime-agnostic
//!   (`futures` + `blocking::Unblock` stdio, `ConnectTo`/`Role` traits)
//!   and inverts control — you implement its handler traits and it owns
//!   the loop. lamu's frontends own their loops over plain tokio stdio
//!   (lamu-mcp `server.rs`); keeping that shape means the ACP surface
//!   reads like the MCP one.
//! - **Stability**: 69 releases in ~11 months, still 0.x, with unstable
//!   feature-gated protocol drafts (v2). A vendored subset of v1 shapes
//!   is a smaller maintenance surface than tracking that churn.
//!
//! What we keep from the official SDK: its serialized shapes are the
//! ground truth — `protocol.rs` documents the exact crate versions the
//! types were pinned against, and the **newline-delimited JSON-RPC 2.0
//! framing** was verified against the crate's `src/stdio.rs` (ACP does
//! NOT use LSP `Content-Length` framing).
//!
//! # Crate-vs-CLI-module decision (for ADR 0036)
//!
//! ACP lives as a **module inside lamu-cli** (`src/acp/`), not a separate
//! `lamu-acp` crate. `lamu acp` is just another CLI mode like `lamu
//! start` (MCP stdio): the CLI is already the ADR 0023 composition root
//! that couples lamu-mcp + the module registry, so composing the same
//! `LamuMcpServer` here adds no new inter-frontend dependency edge — a
//! separate crate would have needed lamu-mcp (a fellow FRONTEND) as a
//! dependency, the exact wrinkle ADR 0023's taxonomy avoids. The module
//! registers no backend kind and no tools; it only DRIVES them.
//!
//! # Concurrency (scoped deviation from ADR 0024)
//!
//! ADR 0024's serial dispatch is **MCP-loop-scoped**. The ACP loop reads
//! serially and handles everything inline EXCEPT `session/prompt`, which
//! runs in a spawned task so the read loop keeps processing — required
//! for `session/cancel` to be honored MID-prompt and for the client to
//! answer `session/request_permission` / `fs/write_text_file` requests
//! that the prompt task itself issues (handling those inline would
//! deadlock). One prompt per session at a time (per-session turn mutex);
//! the expensive resource is still serialized below at the per-model
//! RequestQueue, exactly as ADR 0024 intends.

pub mod agent_loop;
pub mod protocol;

#[cfg(test)]
mod tests;

use anyhow::Result;
use lamu_mcp::server::LamuMcpServer;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot, watch};

/// Knobs + test seams for the ACP agent.
pub struct AcpConfig {
    /// Test seam: when set, the agent loop POSTs chat completions here
    /// and skips registry resolution / ensure-load entirely. `None` in
    /// production — the loop resolves the loaded model's port.
    pub chat_url: Option<String>,
    /// Model override. `None` → registry main-alias resolution (the same
    /// `Router::route` pick `/v1/chat/completions` uses with no model).
    pub model: Option<String>,
    /// Cap on model requests within one prompt turn. Hitting it returns
    /// stop reason `max_turn_requests` (the spec's exact reason for this).
    pub max_turns: usize,
    /// max_tokens per model request.
    pub max_tokens: u32,
}

impl Default for AcpConfig {
    fn default() -> Self {
        Self { chat_url: None, model: None, max_turns: 10, max_tokens: 8192 }
    }
}

/// Cached per-session permission verdict for a tool (`allow_always` /
/// `reject_always` outcomes of `session/request_permission`).
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum PermCache {
    AllowAlways,
    RejectAlways,
}

pub(crate) struct AcpSession {
    pub id: String,
    pub cwd: PathBuf,
    /// Model pinned at `session/new` (registry main-alias resolution).
    pub model: String,
    /// OpenAI-shaped message history (user / assistant(+tool_calls) /
    /// tool). The system prompt is prepended per request, not stored.
    pub history: tokio::sync::Mutex<Vec<Value>>,
    /// One prompt turn at a time per session.
    pub turn: Arc<tokio::sync::Mutex<()>>,
    /// Cancellation for the in-flight turn (None when idle). A fresh
    /// watch channel per turn; `session/cancel` flips it to true.
    pub cancel: StdMutex<Option<watch::Sender<bool>>>,
    /// Per-session+tool permission cache.
    pub perms: StdMutex<HashMap<String, PermCache>>,
}

/// Outcome of the permission gate for one tool call.
#[derive(PartialEq, Debug)]
pub(crate) enum PermissionDecision {
    Allowed,
    Rejected,
    /// The turn was cancelled while waiting on the client.
    CancelledTurn,
}

pub struct AcpServer {
    pub(crate) mcp: Arc<LamuMcpServer>,
    pub(crate) cfg: AcpConfig,
    /// HTTP client for the agent loop's streaming chat POSTs. No total
    /// timeout (turn-length generation is unbounded); cancellation and
    /// stream EOF bound the read instead.
    pub(crate) http: reqwest::Client,
    /// Serialized outgoing wire lines (responses, notifications, agent →
    /// client requests). A single writer task drains this, so spawned
    /// prompt tasks and the read loop never interleave partial lines.
    out: mpsc::UnboundedSender<String>,
    out_rx: StdMutex<Option<mpsc::UnboundedReceiver<String>>>,
    /// Id allocator + response routing for agent → client requests.
    next_id: AtomicI64,
    pending: StdMutex<HashMap<i64, oneshot::Sender<Result<Value, Value>>>>,
    sessions: StdMutex<HashMap<String, Arc<AcpSession>>>,
    pub(crate) client_caps: StdMutex<protocol::ClientCapabilities>,
}

impl AcpServer {
    pub fn new(mcp: Arc<LamuMcpServer>, cfg: AcpConfig) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| anyhow::anyhow!("reqwest client init: {e}"))?;
        Ok(Self {
            mcp,
            cfg,
            http,
            out: tx,
            out_rx: StdMutex::new(Some(rx)),
            next_id: AtomicI64::new(1),
            pending: StdMutex::new(HashMap::new()),
            sessions: StdMutex::new(HashMap::new()),
            client_caps: StdMutex::new(protocol::ClientCapabilities::default()),
        })
    }

    /// Drive the ACP loop over `reader`/`writer` until EOF. Production
    /// passes stdio; tests pass `tokio::io::duplex` halves.
    pub async fn run<R, W>(self: Arc<Self>, reader: R, writer: W) -> Result<()>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let mut rx = self
            .out_rx
            .lock()
            .expect("acp out_rx lock")
            .take()
            .ok_or_else(|| anyhow::anyhow!("AcpServer::run called twice"))?;
        let writer_task = tokio::spawn(async move {
            let mut w = writer;
            while let Some(line) = rx.recv().await {
                if w.write_all(line.as_bytes()).await.is_err() { break; }
                if w.write_all(b"\n").await.is_err() { break; }
                if w.flush().await.is_err() { break; }
            }
        });

        let mut lines = BufReader::new(reader).lines();
        loop {
            let line = match lines.next_line().await {
                Ok(Some(l)) => l,
                Ok(None) => {
                    tracing::info!("acp: stdin EOF — client closed pipe, shutting down");
                    break;
                }
                Err(e) => {
                    tracing::error!("acp: read error: {e}");
                    break;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            let msg: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    self.send_error(Value::Null, -32700, &format!("parse error: {e}"));
                    continue;
                }
            };
            self.handle_message(msg);
        }

        // EOF/teardown: flip every in-flight turn to cancelled so spawned
        // prompt tasks wind down instead of streaming into the void.
        let sessions: Vec<Arc<AcpSession>> =
            self.sessions.lock().expect("acp sessions lock").values().cloned().collect();
        for s in sessions {
            if let Some(tx) = s.cancel.lock().expect("acp cancel lock").as_ref() {
                let _ = tx.send(true);
            }
        }
        writer_task.abort();
        Ok(())
    }

    /// Route one incoming wire message. Everything is handled inline and
    /// non-blocking except `session/prompt`, which spawns (see module
    /// docs — scoped ADR 0024 deviation).
    fn handle_message(self: &Arc<Self>, msg: Value) {
        let Some(method) = msg.get("method").and_then(|v| v.as_str()).map(str::to_string) else {
            // No method → a response to one of OUR agent→client requests.
            self.route_client_response(&msg);
            return;
        };
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));

        match method.as_str() {
            protocol::M_INITIALIZE => {
                let p: protocol::InitializeParams =
                    serde_json::from_value(params).unwrap_or_default();
                *self.client_caps.lock().expect("acp caps lock") = p.client_capabilities.clone();
                self.respond(id, protocol::initialize_result(p.version()));
            }
            protocol::M_AUTHENTICATE => {
                // No auth methods advertised; accept vacuously.
                self.respond(id, json!({}));
            }
            protocol::M_SESSION_NEW => self.on_session_new(params, id),
            protocol::M_SESSION_PROMPT => self.on_session_prompt(params, id),
            protocol::M_SESSION_CANCEL => {
                if let Ok(p) = serde_json::from_value::<protocol::CancelParams>(params) {
                    let session = self
                        .sessions
                        .lock()
                        .expect("acp sessions lock")
                        .get(&p.session_id)
                        .cloned();
                    if let Some(s) = session {
                        if let Some(tx) = s.cancel.lock().expect("acp cancel lock").as_ref() {
                            let _ = tx.send(true);
                        }
                    }
                }
            }
            other => {
                // Unknown notification → ignore; unknown request → -32601.
                if id.is_some() {
                    self.send_error(
                        id.unwrap_or(Value::Null),
                        -32601,
                        &format!("method not found: {other}"),
                    );
                }
            }
        }
    }

    fn on_session_new(self: &Arc<Self>, params: Value, id: Option<Value>) {
        let p: protocol::NewSessionParams = match serde_json::from_value(params) {
            Ok(p) => p,
            Err(e) => {
                self.send_error(id.unwrap_or(Value::Null), -32602, &format!("session/new: {e}"));
                return;
            }
        };
        if !p.mcp_servers.is_empty() {
            tracing::warn!(
                "acp: session/new passed {} mcpServers — ignored in v1 (LAMU exposes its in-process tools)",
                p.mcp_servers.len()
            );
        }

        // Registry main-alias resolution: the same `Router::route` pick a
        // model-less /v1/chat/completions gets (operator `main: true`
        // entry preferred, else best chat-capable candidate).
        let model = match &self.cfg.model {
            Some(m) => m.clone(),
            None => {
                let st = self.mcp.state.lock();
                let d = st.router.route(&st.scheduler, None, None, Some(st.health.all()));
                if d.model_name.is_empty() {
                    if self.cfg.chat_url.is_none() {
                        self.send_error(
                            id.unwrap_or(Value::Null),
                            -32603,
                            &format!("no local model available: {}", d.reason),
                        );
                        return;
                    }
                    // Test seam: a chat_url override doesn't need a registry.
                    "lamu".to_string()
                } else {
                    d.model_name
                }
            }
        };

        let session_id = format!("sess-{}", random_hex(16));
        let session = Arc::new(AcpSession {
            id: session_id.clone(),
            cwd: p.cwd,
            model: model.clone(),
            history: tokio::sync::Mutex::new(Vec::new()),
            turn: Arc::new(tokio::sync::Mutex::new(())),
            cancel: StdMutex::new(None),
            perms: StdMutex::new(HashMap::new()),
        });
        self.sessions
            .lock()
            .expect("acp sessions lock")
            .insert(session_id.clone(), session);

        // Warm-up: ensure-load the session model in the background so the
        // first prompt doesn't eat the whole cold-load latency. Best
        // effort — the agent loop re-ensures (authoritatively) per turn.
        if self.cfg.chat_url.is_none() {
            let mcp = self.mcp.clone();
            let warm_model = model;
            tokio::spawn(async move {
                use lamu_core::tools_ext::ToolCtx;
                if let Err(e) = (&*mcp as &dyn ToolCtx).ensure_loaded(&warm_model).await {
                    tracing::warn!("acp: session warm-up load of '{warm_model}' failed: {e}");
                }
            });
        }

        self.respond(id, json!({ "sessionId": session_id }));
    }

    fn on_session_prompt(self: &Arc<Self>, params: Value, id: Option<Value>) {
        let p: protocol::PromptParams = match serde_json::from_value(params) {
            Ok(p) => p,
            Err(e) => {
                self.send_error(id.unwrap_or(Value::Null), -32602, &format!("session/prompt: {e}"));
                return;
            }
        };
        let session = self
            .sessions
            .lock()
            .expect("acp sessions lock")
            .get(&p.session_id)
            .cloned();
        let Some(session) = session else {
            self.send_error(
                id.unwrap_or(Value::Null),
                -32602,
                &format!("unknown sessionId: {}", p.session_id),
            );
            return;
        };
        let Ok(guard) = session.turn.clone().try_lock_owned() else {
            self.send_error(
                id.unwrap_or(Value::Null),
                -32600,
                "a prompt is already running for this session",
            );
            return;
        };

        let (cancel_tx, cancel_rx) = watch::channel(false);
        *session.cancel.lock().expect("acp cancel lock") = Some(cancel_tx);

        // Scoped ADR 0024 deviation (see module docs): the turn runs in a
        // spawned task so the read loop stays free for session/cancel and
        // for the client's answers to our permission/fs requests.
        let srv = self.clone();
        tokio::spawn(async move {
            let _turn = guard;
            let res = agent_loop::run_prompt_turn(&srv, &session, p.prompt, cancel_rx).await;
            *session.cancel.lock().expect("acp cancel lock") = None;
            match res {
                Ok(stop) => srv.respond(id, protocol::prompt_result(stop)),
                Err(e) => srv.send_error(id.unwrap_or(Value::Null), -32603, &e),
            }
        });
    }

    // ── Outgoing wire helpers ───────────────────────────────────────

    pub(crate) fn respond(&self, id: Option<Value>, result: Value) {
        let _ = self.out.send(
            json!({ "jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "result": result })
                .to_string(),
        );
    }

    pub(crate) fn send_error(&self, id: Value, code: i64, message: &str) {
        let _ = self.out.send(
            json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
                .to_string(),
        );
    }

    /// Fire a `session/update` notification.
    pub(crate) fn send_update(&self, session_id: &str, update: Value) {
        let _ = self
            .out
            .send(protocol::session_update(session_id, update).to_string());
    }

    /// Issue an agent → client request; the read loop routes the response
    /// into the returned receiver. Caller must `select!` against its
    /// cancel token and call [`Self::forget_client_request`] on abandon.
    pub(crate) fn begin_client_request(
        &self,
        method: &str,
        params: Value,
    ) -> (i64, oneshot::Receiver<Result<Value, Value>>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().expect("acp pending lock").insert(id, tx);
        let _ = self.out.send(
            json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }).to_string(),
        );
        (id, rx)
    }

    pub(crate) fn forget_client_request(&self, id: i64) {
        self.pending.lock().expect("acp pending lock").remove(&id);
    }

    fn route_client_response(&self, msg: &Value) {
        let Some(id) = msg.get("id").and_then(|v| v.as_i64()) else {
            return;
        };
        let Some(tx) = self.pending.lock().expect("acp pending lock").remove(&id) else {
            return; // late/unknown response (e.g. turn already cancelled)
        };
        let res = if let Some(err) = msg.get("error") {
            Err(err.clone())
        } else {
            Ok(msg.get("result").cloned().unwrap_or(Value::Null))
        };
        let _ = tx.send(res);
    }

    // ── Permission gate ─────────────────────────────────────────────

    /// `session/request_permission` for a write-effecting tool call,
    /// honoring the per-session `allow_always`/`reject_always` cache and
    /// the turn's cancel token.
    pub(crate) async fn request_permission(
        &self,
        session: &AcpSession,
        tool_name: &str,
        tool_call_id: &str,
        title: &str,
        kind: &str,
        raw_input: &Value,
        cancel: &mut watch::Receiver<bool>,
    ) -> PermissionDecision {
        match session.perms.lock().expect("acp perms lock").get(tool_name) {
            Some(PermCache::AllowAlways) => return PermissionDecision::Allowed,
            Some(PermCache::RejectAlways) => return PermissionDecision::Rejected,
            None => {}
        }

        let params = protocol::request_permission_params(
            &session.id,
            tool_call_id,
            title,
            kind,
            raw_input,
        );
        let (req_id, rx) = self.begin_client_request(protocol::M_REQUEST_PERMISSION, params);

        let res = tokio::select! {
            r = rx => r,
            _ = agent_loop::wait_cancelled(cancel) => {
                self.forget_client_request(req_id);
                return PermissionDecision::CancelledTurn;
            }
        };

        let result = match res {
            Ok(Ok(v)) => v,
            // Client error / dropped channel → fail safe: reject.
            _ => return PermissionDecision::Rejected,
        };
        match protocol::permission_outcome_option(&result).as_deref() {
            Some("allow_once") => PermissionDecision::Allowed,
            Some("allow_always") => {
                session
                    .perms
                    .lock()
                    .expect("acp perms lock")
                    .insert(tool_name.to_string(), PermCache::AllowAlways);
                PermissionDecision::Allowed
            }
            Some("reject_always") => {
                session
                    .perms
                    .lock()
                    .expect("acp perms lock")
                    .insert(tool_name.to_string(), PermCache::RejectAlways);
                PermissionDecision::Rejected
            }
            // reject_once, unknown option ids, and outcome `cancelled`
            // (client reports the turn cancelled) all refuse this call.
            _ => PermissionDecision::Rejected,
        }
    }
}

/// Random lowercase-hex string of `2 * n_bytes` chars (the lamu-api
/// `random_hex` pattern — getrandom is already in lamu-cli's tree for
/// `lamu auth init`).
fn random_hex(n_bytes: usize) -> String {
    let mut buf = vec![0u8; n_bytes];
    if getrandom::getrandom(&mut buf).is_err() {
        // Degraded fallback: timestamp-based, still unique enough for a
        // per-process session id.
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        return format!("{t:032x}");
    }
    buf.iter().map(|b| format!("{b:02x}")).collect()
}
