//! A2A (Agent2Agent) protocol surface for LAMU — `lamu a2a` (ADR 0038).
//!
//! Exposes a JSON-RPC 2.0 over HTTP endpoint plus a `/.well-known/agent-card.json`
//! agent card so any A2A-compliant client can discover and drive the local LAMU
//! model via the same in-process MCP tool surface as ACP.
//!
//! # Architecture (ADR 0023/0029 compliant)
//!
//! A2A lives in lamu-cli (the composition root) as a sibling module of ACP —
//! NOT in lamu-api (which must never depend on lamu-mcp). The agent loop uses
//! the `agent_core::loop::run_prompt_turn` parameterized core (ADR 0038),
//! wired to `A2aSink` (SSE) and `A2aPermissionGate` (DenyWrites).
//!
//! # Auth v1
//!
//! Loopback-default frictionless. `--bind 0.0.0.0` (off-loopback) refused at
//! startup unless `LAMU_A2A_TOKEN` is set. Card routes stay auth-exempt.
//! All other routes require `Authorization: Bearer <token>` when a token is
//! configured. Mirrors the ADR 0005/0012 shape from lamu-api/src/auth.rs —
//! inlined here to avoid importing a fellow frontend.
//!
//! # Tool subset
//!
//! Excludes `write_file` (no human to answer prompts). Curated set:
//! `query, web_search, research, deep_research, recall_memory, remember`.
//! A forged `write_file` call fails closed via `A2aPermissionGate`.

pub mod protocol;
pub mod sink;

#[cfg(test)]
mod tests;

use crate::agent_core::r#loop::{LoopConfig, TurnStop};
use crate::a2a::protocol as proto;
use crate::a2a::sink::{A2aPermissionGate, A2aSink};
use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use lamu_mcp::server::LamuMcpServer;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{mpsc, watch};

// ── A2A tool subset (no write_file) ─────────────────────────────────

/// Curated A2A tool subset — excludes write_file (no human present).
pub const A2A_TOOLS: &[&str] =
    &["query", "web_search", "research", "deep_research", "recall_memory", "remember"];

// ── Per-task state ───────────────────────────────────────────────────

/// Live state for one A2A task. Created at message/send or message/stream.
pub struct A2aTask {
    pub id: String,
    pub context_id: String,
    /// Completed task JSON (None while in-flight).
    pub completed: Option<Value>,
    /// Cancellation channel for the running turn.
    pub cancel: Option<watch::Sender<bool>>,
}

/// Per-context session: persistent history across tasks in one context.
pub struct A2aContext {
    /// OpenAI-shaped message history (shared across tasks in this context).
    pub history: tokio::sync::Mutex<Vec<Value>>,
}

impl A2aContext {
    pub fn new() -> Self {
        Self { history: tokio::sync::Mutex::new(Vec::new()) }
    }
}

// ── Server state ─────────────────────────────────────────────────────

/// Shared A2A server state (injected into axum via `State`).
#[derive(Clone)]
pub struct A2aState {
    inner: Arc<A2aInner>,
}

struct A2aInner {
    mcp: Arc<LamuMcpServer>,
    cfg: A2aConfig,
    http: reqwest::Client,
    /// Live + recently completed tasks (capped at TASK_CAP).
    tasks: StdMutex<HashMap<String, Arc<StdMutex<A2aTask>>>>,
    /// Insertion-order ring for cap enforcement.
    task_order: StdMutex<VecDeque<String>>,
    /// Per-context session history.
    contexts: StdMutex<HashMap<String, Arc<A2aContext>>>,
}

const TASK_CAP: usize = 256;

/// Knobs for the A2A server.
pub struct A2aConfig {
    /// Test seam: when set, skip registry/ensure-load and POST here.
    pub chat_url: Option<String>,
    /// Model to use (resolved from registry if None — always Some in tests).
    pub model: Option<String>,
    /// Cap on inner model-request iterations per turn.
    pub max_turns: usize,
    /// max_tokens per model request.
    pub max_tokens: u32,
    /// Bearer token for off-loopback auth (None → auth off on loopback).
    pub token: Option<String>,
    /// Public base URL advertised in the agent card. Populated by `serve()`
    /// from the effective bind address; None falls back to the loopback
    /// default (tests construct configs without binding).
    pub public_url: Option<String>,
}

impl Default for A2aConfig {
    fn default() -> Self {
        Self {
            chat_url: None,
            model: None,
            max_turns: 10,
            max_tokens: 8192,
            token: std::env::var("LAMU_A2A_TOKEN")
                .ok()
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty()),
            public_url: None,
        }
    }
}

impl A2aState {
    pub fn new(mcp: Arc<LamuMcpServer>, cfg: A2aConfig) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()?;
        Ok(Self {
            inner: Arc::new(A2aInner {
                mcp,
                cfg,
                http,
                tasks: StdMutex::new(HashMap::new()),
                task_order: StdMutex::new(VecDeque::new()),
                contexts: StdMutex::new(HashMap::new()),
            }),
        })
    }

    fn mcp(&self) -> &Arc<LamuMcpServer> {
        &self.inner.mcp
    }

    fn cfg(&self) -> &A2aConfig {
        &self.inner.cfg
    }

    fn http(&self) -> &reqwest::Client {
        &self.inner.http
    }

    /// Resolve model name for a new task.
    fn resolve_model(&self) -> Result<String, String> {
        if let Some(m) = &self.inner.cfg.model {
            return Ok(m.clone());
        }
        let st = self.inner.mcp.state.lock();
        let d = st.router.route(&st.scheduler, None, None, Some(st.health.all()));
        if d.model_name.is_empty() {
            if self.inner.cfg.chat_url.is_none() {
                return Err(format!("no local model available: {}", d.reason));
            }
            Ok("lamu".to_string())
        } else {
            Ok(d.model_name)
        }
    }

    fn get_or_create_context(&self, context_id: &str) -> Arc<A2aContext> {
        let mut ctxs = self.inner.contexts.lock().expect("a2a contexts lock");
        ctxs.entry(context_id.to_string())
            .or_insert_with(|| Arc::new(A2aContext::new()))
            .clone()
    }

    fn register_task(&self, task: Arc<StdMutex<A2aTask>>) {
        let id = task.lock().expect("a2a task lock").id.clone();
        {
            let mut tasks = self.inner.tasks.lock().expect("a2a tasks lock");
            let mut order = self.inner.task_order.lock().expect("a2a order lock");
            tasks.insert(id.clone(), task);
            order.push_back(id);
            // Evict oldest completed tasks when over cap.
            while order.len() > TASK_CAP {
                if let Some(old) = order.pop_front() {
                    tasks.remove(&old);
                }
            }
        }
    }

    fn get_task(&self, id: &str) -> Option<Arc<StdMutex<A2aTask>>> {
        self.inner.tasks.lock().expect("a2a tasks lock").get(id).cloned()
    }

    fn cancel_task(&self, id: &str) -> bool {
        let task = match self.get_task(id) {
            Some(t) => t,
            None => return false,
        };
        let sender = task.lock().expect("a2a task lock").cancel.clone();
        if let Some(tx) = sender {
            let _ = tx.send(true);
            true
        } else {
            false
        }
    }

    fn mark_completed(&self, id: &str, completed_task: Value) {
        if let Some(t) = self.get_task(id) {
            let mut guard = t.lock().expect("a2a task lock");
            guard.completed = Some(completed_task);
            guard.cancel = None;
        }
    }
}

// ── Auth helpers (inlined; not imported from lamu-api) ───────────────

fn check_auth(st: &A2aState, req: &Request<Body>) -> bool {
    let token = match &st.inner.cfg.token {
        Some(t) => t,
        None => return true, // no token configured → pass
    };
    // Parse `Authorization: Bearer <token>` leniently (RFC 7235).
    let presented: Option<String> = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
        .map(|(_, tok)| tok.trim().to_string())
        .filter(|t| !t.is_empty());
    match presented.as_deref() {
        Some(t) => {
            use subtle::ConstantTimeEq;
            t.as_bytes().ct_eq(token.as_bytes()).into()
        }
        None => false,
    }
}

fn unauthorized_response() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        Json(json!({"error": {"code": -32001, "message": "unauthorized"}})),
    )
        .into_response()
}

// ── Axum router ──────────────────────────────────────────────────────

pub fn router(state: A2aState) -> Router {
    Router::new()
        .route(proto::PATH_AGENT_CARD, get(handle_agent_card))
        .route(proto::PATH_AGENT_CARD_WK_COMPAT, get(handle_agent_card))
        .route(proto::PATH_AGENT_CARD_COMPAT, get(handle_agent_card))
        .route(proto::PATH_RPC, post(handle_rpc))
        .with_state(state)
}

/// `GET /.well-known/agent-card.json` (+ aliases) — auth exempt.
async fn handle_agent_card(State(st): State<A2aState>) -> impl IntoResponse {
    // Advertise the effective bind URL (set by serve()); fall back to the
    // loopback default when unset (e.g. router-only test construction).
    let url = st
        .cfg()
        .public_url
        .clone()
        .unwrap_or_else(|| "http://127.0.0.1:8022".to_string());
    let tool_names = A2A_TOOLS
        .iter()
        .filter(|&&name| {
            crate::acp::agent_loop::mcp_tool_entry(name).is_some()
        })
        .copied()
        .collect::<Vec<_>>();
    let skills = proto::build_skills(&tool_names);
    let card = proto::agent_card(&url, &skills);
    Json(card)
}

/// `POST /` — JSON-RPC 2.0 dispatch.
async fn handle_rpc(
    State(st): State<A2aState>,
    req: Request<Body>,
) -> Response {
    // Auth check (token-gated when configured).
    if !check_auth(&st, &req) {
        return unauthorized_response();
    }
    let body_bytes = match axum::body::to_bytes(req.into_body(), 4 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(proto::rpc_error(&json!(null), -32700, "failed to read body")),
            )
                .into_response();
        }
    };
    let msg: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(proto::rpc_error(&json!(null), -32700, &format!("parse error: {e}"))),
            )
                .into_response();
        }
    };
    let id = msg.get("id").cloned().unwrap_or(Value::Null);
    let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));

    match method {
        proto::M_MESSAGE_SEND => handle_message_send(st, id, params).await,
        proto::M_MESSAGE_STREAM => handle_message_stream(st, id, params).await,
        proto::M_TASKS_GET => handle_tasks_get(st, id, params).await,
        proto::M_TASKS_CANCEL => handle_tasks_cancel(st, id, params).await,
        other => (
            StatusCode::OK,
            Json(proto::rpc_error(&id, -32601, &format!("method not found: {other}"))),
        )
            .into_response(),
    }
}

// ── message/send ─────────────────────────────────────────────────────

async fn handle_message_send(st: A2aState, id: Value, params: Value) -> Response {
    let (context_id, text, _) = match proto::parse_send_params(&params) {
        Some(t) => t,
        None => {
            return (
                StatusCode::OK,
                Json(proto::rpc_error(&id, -32602, "message/send: missing message or text part")),
            )
                .into_response();
        }
    };
    let model = match st.resolve_model() {
        Ok(m) => m,
        Err(e) => {
            return (
                StatusCode::OK,
                Json(proto::rpc_error(&id, -32603, &e)),
            )
                .into_response();
        }
    };

    let task_id = proto::pseudo_id("task-");
    let ctx = st.get_or_create_context(&context_id);
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let task_arc = Arc::new(StdMutex::new(A2aTask {
        id: task_id.clone(),
        context_id: context_id.clone(),
        completed: None,
        cancel: Some(cancel_tx),
    }));
    st.register_task(task_arc.clone());

    // Collect all events into a null sink (sync run — no SSE).
    let null_sink = NullSink;
    let gate = A2aPermissionGate;
    let cfg = LoopConfig {
        chat_url: st.cfg().chat_url.clone(),
        model: model.clone(),
        max_turns: st.cfg().max_turns,
        max_tokens: st.cfg().max_tokens,
    };
    let prompt = vec![json!({ "type": "text", "text": text })];

    let stop = crate::agent_core::r#loop::run_prompt_turn(
        st.mcp(),
        st.http(),
        &cfg,
        &ctx.history,
        prompt,
        A2A_TOOLS,
        &null_sink,
        &gate,
        cancel_rx,
    )
    .await;

    // Assemble completed task from history.
    let history_snapshot = ctx.history.lock().await.clone();
    let last_assistant = history_snapshot
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"))
        .and_then(|m| m.get("content").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();

    let (state_str, err_msg) = match stop {
        Ok(TurnStop::EndTurn) | Ok(TurnStop::MaxTurnRequests) => (proto::STATE_COMPLETED, None),
        Ok(TurnStop::Cancelled) => (proto::STATE_CANCELED, None),
        Err(ref e) => (proto::STATE_FAILED, Some(e.clone())),
    };

    let completed = if state_str == proto::STATE_COMPLETED {
        proto::task_completed(&task_id, &context_id, &last_assistant, to_a2a_history(&history_snapshot))
    } else {
        let msg = err_msg.map(|e| proto::agent_message(&e));
        proto::task(&task_id, &context_id, state_str, msg, vec![], to_a2a_history(&history_snapshot))
    };

    st.mark_completed(&task_id, completed.clone());
    (StatusCode::OK, Json(proto::rpc_result(&id, completed))).into_response()
}

// ── message/stream ────────────────────────────────────────────────────

async fn handle_message_stream(st: A2aState, id: Value, params: Value) -> Response {
    let (context_id, text, _) = match proto::parse_send_params(&params) {
        Some(t) => t,
        None => {
            return (
                StatusCode::OK,
                Json(proto::rpc_error(&id, -32602, "message/stream: missing message or text part")),
            )
                .into_response();
        }
    };
    let model = match st.resolve_model() {
        Ok(m) => m,
        Err(e) => {
            return (
                StatusCode::OK,
                Json(proto::rpc_error(&id, -32603, &e)),
            )
                .into_response();
        }
    };

    let task_id = proto::pseudo_id("task-");
    let ctx = st.get_or_create_context(&context_id);
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let task_arc = Arc::new(StdMutex::new(A2aTask {
        id: task_id.clone(),
        context_id: context_id.clone(),
        completed: None,
        cancel: Some(cancel_tx),
    }));
    st.register_task(task_arc.clone());

    // SSE channel: agent loop pushes data lines, the response body drains them.
    let (sse_tx, mut sse_rx) = mpsc::unbounded_channel::<String>();
    let sink = A2aSink { task_id: task_id.clone(), tx: sse_tx.clone() };
    let gate = A2aPermissionGate;
    let cfg = LoopConfig {
        chat_url: st.cfg().chat_url.clone(),
        model: model.clone(),
        max_turns: st.cfg().max_turns,
        max_tokens: st.cfg().max_tokens,
    };
    let prompt = vec![json!({ "type": "text", "text": text })];
    let task_id_clone = task_id.clone();
    let context_id_clone = context_id.clone();
    let st_clone = st.clone();

    // Emit initial "submitted" status.
    let _ = sse_tx.send(proto::sse_line(&proto::status_update_event(
        &task_id,
        proto::STATE_SUBMITTED,
        None,
    )));
    // Emit "working" status.
    let _ = sse_tx.send(proto::sse_line(&proto::status_update_event(
        &task_id,
        proto::STATE_WORKING,
        None,
    )));

    // Run the agent loop in a separate task so we can stream concurrently.
    let mcp = st.mcp().clone();
    let http = st.http().clone();
    let ctx_clone = ctx.clone();
    let sse_tx_done = sse_tx.clone();
    tokio::spawn(async move {
        let stop = crate::agent_core::r#loop::run_prompt_turn(
            &mcp,
            &http,
            &cfg,
            &ctx_clone.history,
            prompt,
            A2A_TOOLS,
            &sink,
            &gate,
            cancel_rx,
        )
        .await;

        // Assemble and stream the final Task.
        let history_snapshot = ctx_clone.history.lock().await.clone();
        let last_assistant = history_snapshot
            .iter()
            .rev()
            .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"))
            .and_then(|m| m.get("content").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string();

        let (state_str, err_msg) = match stop {
            Ok(TurnStop::EndTurn) | Ok(TurnStop::MaxTurnRequests) => {
                (proto::STATE_COMPLETED, None)
            }
            Ok(TurnStop::Cancelled) => (proto::STATE_CANCELED, None),
            Err(ref e) => (proto::STATE_FAILED, Some(e.clone())),
        };

        let completed = if state_str == proto::STATE_COMPLETED {
            proto::task_completed(
                &task_id_clone,
                &context_id_clone,
                &last_assistant,
                to_a2a_history(&history_snapshot),
            )
        } else {
            let msg = err_msg.map(|e| proto::agent_message(&e));
            proto::task(
                &task_id_clone,
                &context_id_clone,
                state_str,
                msg,
                vec![],
                to_a2a_history(&history_snapshot),
            )
        };

        st_clone.mark_completed(&task_id_clone, completed.clone());

        // Final SSE: the completed Task.
        let final_event = json!({ "task": completed });
        let _ = sse_tx_done.send(proto::sse_line(&final_event));
    });

    // Stream the SSE channel as the response body.
    let stream = async_stream::stream! {
        while let Some(line) = sse_rx.recv().await {
            yield Ok::<_, std::io::Error>(line);
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("X-Accel-Buffering", "no")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

// ── tasks/get ────────────────────────────────────────────────────────

async fn handle_tasks_get(st: A2aState, id: Value, params: Value) -> Response {
    let task_id = match proto::parse_get_params(&params) {
        Some(t) => t,
        None => {
            return (
                StatusCode::OK,
                Json(proto::rpc_error(&id, -32602, "tasks/get: missing id")),
            )
                .into_response();
        }
    };
    let task = match st.get_task(&task_id) {
        Some(t) => t,
        None => {
            return (
                StatusCode::OK,
                Json(proto::rpc_error(&id, -32001, &format!("task not found: {task_id}"))),
            )
                .into_response();
        }
    };
    let guard = task.lock().expect("a2a task lock");
    let result = if let Some(completed) = &guard.completed {
        completed.clone()
    } else {
        // In-flight: return a working stub.
        proto::task(&guard.id, &guard.context_id, proto::STATE_WORKING, None, vec![], vec![])
    };
    (StatusCode::OK, Json(proto::rpc_result(&id, result))).into_response()
}

// ── tasks/cancel ─────────────────────────────────────────────────────

async fn handle_tasks_cancel(st: A2aState, id: Value, params: Value) -> Response {
    let task_id = match proto::parse_cancel_params(&params) {
        Some(t) => t,
        None => {
            return (
                StatusCode::OK,
                Json(proto::rpc_error(&id, -32602, "tasks/cancel: missing id")),
            )
                .into_response();
        }
    };
    let found = st.cancel_task(&task_id);
    if found {
        // Return a canceled stub immediately; the task will update itself.
        let result = proto::task(&task_id, "", proto::STATE_CANCELED, None, vec![], vec![]);
        (StatusCode::OK, Json(proto::rpc_result(&id, result))).into_response()
    } else {
        (
            StatusCode::OK,
            Json(proto::rpc_error(&id, -32001, &format!("task not found: {task_id}"))),
        )
            .into_response()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Convert OpenAI-shaped history to A2A Message array (best-effort).
fn to_a2a_history(history: &[Value]) -> Vec<Value> {
    history
        .iter()
        .filter_map(|m| {
            let role = m.get("role").and_then(|v| v.as_str())?;
            let a2a_role = match role {
                "user" => "user",
                "assistant" => "agent",
                _ => return None, // skip tool messages
            };
            let text = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
            Some(json!({
                "messageId": proto::pseudo_id("msg-"),
                "role": a2a_role,
                "parts": [{ "kind": "text", "text": text }],
            }))
        })
        .collect()
}

/// Null UpdateSink for synchronous message/send (no streaming).
struct NullSink;

impl crate::agent_core::UpdateSink for NullSink {
    fn emit(&self, _ev: crate::agent_core::LoopEvent) -> anyhow::Result<()> {
        Ok(())
    }
}

// ── Startup ──────────────────────────────────────────────────────────

/// Start the A2A HTTP server. Refuses off-loopback bind without a token.
pub async fn serve(
    mcp: Arc<LamuMcpServer>,
    mut cfg: A2aConfig,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    let is_loopback = addr.ip().is_loopback();
    if !is_loopback && cfg.token.is_none() {
        anyhow::bail!(
            "A2A bind address {} is off-loopback but LAMU_A2A_TOKEN is not set. \
             Set the env var to a strong secret token, then retry. \
             To expose only to localhost, use --bind 127.0.0.1 (the default).",
            addr
        );
    }
    // Advertise the real bind address in the agent card (unless overridden).
    if cfg.public_url.is_none() {
        cfg.public_url = Some(format!("http://{addr}"));
    }
    let state = A2aState::new(mcp, cfg)?;
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("LAMU A2A agent ready on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
