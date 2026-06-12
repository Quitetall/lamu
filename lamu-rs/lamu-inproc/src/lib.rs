//! lamu-inproc — in-process engines behind the port-proxy seam (ADR 0033).
//!
//! lamu's loading architecture drops the `Box<dyn Backend>` right after a
//! successful `load()` and from then on talks to the backend ONLY via its
//! port (`lamu-core/src/loader.rs`; `lamu-api`'s `/v1/embeddings` proxies
//! `POST http://localhost:{port}/v1/embeddings` verbatim). A subprocess
//! backend satisfies that naturally — it IS a server on a port. An
//! IN-PROCESS engine (ort, candle, …) must do the same thing: bind a real
//! TCP port and serve a llama-server-compatible surface from a tokio task.
//! This crate is that shim, shared by every in-process engine module
//! (`lamu-onnx` today; `lamu-hf` is the next consumer).
//!
//! Two servers, one engine each:
//!
//! Embed surface ([`spawn_embed_server`], ADR 0034):
//! - `GET /health` AND `GET /v1/health` → `{"status":"ok"}`. Both paths
//!   carry the same body because the two health conventions in the tree
//!   differ: llama.cpp probes `/health` and requires `status == "ok"`
//!   (lamu-core llamacpp.rs `is_healthy`), while fish-speech polls
//!   `/v1/health` for any 2xx. Serving both keeps any existing prober
//!   happy.
//! - `POST /v1/embeddings` → OpenAI embeddings shape.
//! - `POST /tokenize` → 404 with a JSON error. Embed stub: embedding
//!   engines don't expose llama-server's tokenize surface, and the only
//!   caller (`tokenize_count`, ADR 0021 context occupancy) already treats
//!   "unsupported" as a clean error. The route exists (instead of the
//!   implicit fallback) purely so the 404 body says WHY.
//!
//! Chat surface ([`spawn_chat_server`], ADR 0035 — `lamu-hf` is the first
//! consumer):
//! - `GET /health` + `GET /v1/health` → same `{"status":"ok"}` contract.
//! - `POST /v1/chat/completions` → llama-server-shaped OpenAI responses,
//!   stream and non-stream, including `usage` token counts. The wire
//!   shapes mirror llama-server's because lamu-api's bridges parse them
//!   (`parse_upstream_line` extracts `delta.content` / `finish_reason` /
//!   `usage` from `data:` lines and stops on `data: [DONE]`). Raw
//!   `<think>` tags flow through as content — the ADR 0037 reasoning
//!   splitter downstream owns that classification.
//! - `POST /tokenize` `{content}` → `{"tokens":[...]}` — the real
//!   llama-server surface this time, backed by the engine's tokenizer
//!   (ADR 0021 engine-truth occupancy counts the array length).

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;

/// A synchronous, thread-safe embedding engine. Implementations are CPU
/// (or otherwise blocking) compute; the HTTP handler runs `embed` inside
/// `tokio::task::spawn_blocking`, so a sync `fn` here is correct — do not
/// make this async to "fix" blocking, the server already handles it.
pub trait EmbedEngine: Send + Sync {
    /// Stable identifier (usually the registry entry name / model name).
    fn id(&self) -> &str;
    /// Embedding dimensionality (discovered at load).
    fn dims(&self) -> usize;
    /// Embed a batch of texts. Must return one vector per input, each of
    /// `dims()` length.
    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
}

#[derive(Clone)]
struct ServerState {
    model_name: String,
    engine: Arc<dyn EmbedEngine>,
}

/// Spawn the in-process embed server on `port` (binds `127.0.0.1:{port}`).
///
/// Binding happens INSIDE the spawned task (the signature returns a plain
/// `JoinHandle`, there is nowhere to surface a bind error), so a failed
/// bind logs at error level and the task exits — callers confirm liveness
/// the same way they would for a subprocess backend: poll `GET /health`
/// until it answers (see `lamu-onnx`'s `Backend::load`). Tests that need
/// a deterministic port use [`spawn_embed_server_on`] with a pre-bound
/// listener instead.
pub fn spawn_embed_server(
    port: u16,
    model_name: String,
    engine: Arc<dyn EmbedEngine>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(port, model = %model_name, "inproc embed server bind failed: {e}");
                return;
            }
        };
        serve(listener, model_name, engine).await;
    })
}

/// Like [`spawn_embed_server`] but on an already-bound listener. This is
/// the seam tests use: bind `127.0.0.1:0`, read the real port off the
/// listener, then serve on it — axum needs a concrete socket either way.
pub fn spawn_embed_server_on(
    listener: tokio::net::TcpListener,
    model_name: String,
    engine: Arc<dyn EmbedEngine>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(serve(listener, model_name, engine))
}

async fn serve(
    listener: tokio::net::TcpListener,
    model_name: String,
    engine: Arc<dyn EmbedEngine>,
) {
    let state = ServerState { model_name, engine };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/embeddings", post(embeddings))
        .route("/tokenize", post(tokenize_stub))
        .with_state(state);
    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("inproc embed server exited: {e}");
    }
}

/// `{"status":"ok"}` — byte-shape llama-server uses, which the llamacpp
/// `is_healthy` parses for `status == "ok"`. Served on `/health` and
/// `/v1/health` (see module docs).
async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

/// v1 stub — see module docs. 404 (not 501) so probers that treat "route
/// exists" as capability detection don't think tokenize works here.
async fn tokenize_stub() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": {
            "message": "tokenize is not supported by the in-process embed server (v1 is embeddings-only)"
        }})),
    )
        .into_response()
}

/// OpenAI-compat `POST /v1/embeddings`. Accepts `{"input": string | [string], "model"?: string}`.
async fn embeddings(State(state): State<ServerState>, Json(body): Json<Value>) -> Response {
    let texts: Vec<String> = match body.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match it.as_str() {
                    Some(s) => out.push(s.to_string()),
                    None => return error_response(
                        StatusCode::BAD_REQUEST,
                        "embeddings: 'input' array items must be strings (token-id arrays are not supported by the in-process server)",
                    ),
                }
            }
            out
        }
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "embeddings: 'input' must be a string or an array of strings",
            )
        }
    };

    // ~4 chars/token heuristic — the in-process surface has no shared
    // tokenizer to count with at this layer, and the only consumer of
    // `usage` here is quota metering, which is best-effort by design.
    let approx_tokens: u64 = texts.iter().map(|t| (t.len() as u64) / 4).sum();

    let engine = state.engine.clone();
    let embedded = tokio::task::spawn_blocking(move || engine.embed(&texts)).await;
    let vectors = match embedded {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("embedding engine failed: {e:#}"),
            )
        }
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("embedding task panicked: {e}"),
            )
        }
    };

    let data: Vec<Value> = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({
                "object": "embedding",
                "index": index,
                "embedding": embedding,
            })
        })
        .collect();

    Json(json!({
        "object": "list",
        "data": data,
        "model": state.model_name,
        "usage": {
            "prompt_tokens": approx_tokens,
            "total_tokens": approx_tokens,
        }
    }))
    .into_response()
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({"error": {"message": message}}))).into_response()
}

// ──────────────────────── chat half (ADR 0035) ────────────────────────

/// One parsed `/v1/chat/completions` request. Unknown body fields are
/// accepted and ignored (matching llama-server), so bridge-added extras
/// like `cache_prompt` / `stream_options` never 400.
#[derive(Debug, Clone)]
pub struct ChatRequestIn {
    /// `(role, content)` pairs in request order.
    pub messages: Vec<(String, String)>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stream: bool,
}

/// Absent `max_tokens` → a bounded default rather than llama-server's
/// unbounded `-1`: an in-process engine shares lamu's CPU/GPU, so a stray
/// curl without limits must not pin a core indefinitely. (lamu-api's
/// bridges always send an explicit max_tokens; this only covers direct
/// callers.)
const DEFAULT_MAX_TOKENS: u32 = 512;
const DEFAULT_TEMPERATURE: f32 = 0.7;

/// A text-generation engine served by [`spawn_chat_server`]. `generate`
/// is async (implementations wrap their compute in `spawn_blocking` and
/// feed fragments through `tx`); `tokenize_count` is sync and cheap —
/// the handler still calls it inside `spawn_blocking`.
#[async_trait::async_trait]
pub trait ChatEngine: Send + Sync {
    /// Stable identifier (usually the registry entry name / model name).
    fn model(&self) -> &str;
    /// Engine-tokenizer token count for `text` (ADR 0021 engine-truth).
    fn tokenize_count(&self, text: &str) -> anyhow::Result<usize>;
    /// Generate, sending each text fragment to `tx`; return
    /// `(prompt_tokens, completion_tokens, finish_reason)`. Fragments are
    /// raw model text — `<think>` tags included; reasoning classification
    /// is downstream's job (ADR 0037).
    async fn generate(
        &self,
        req: ChatRequestIn,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> anyhow::Result<(usize, usize, String)>;
}

#[derive(Clone)]
struct ChatServerState {
    model_name: String,
    engine: Arc<dyn ChatEngine>,
}

/// Spawn the in-process chat server on `port` (binds `127.0.0.1:{port}`).
/// Same liveness contract as [`spawn_embed_server`]: the bind happens
/// inside the task, callers poll `GET /health`.
pub fn spawn_chat_server(
    port: u16,
    model_name: String,
    engine: Arc<dyn ChatEngine>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(port, model = %model_name, "inproc chat server bind failed: {e}");
                return;
            }
        };
        serve_chat(listener, model_name, engine).await;
    })
}

/// Like [`spawn_chat_server`] but on an already-bound listener (the test
/// seam — bind `127.0.0.1:0`, read the real port, serve).
pub fn spawn_chat_server_on(
    listener: tokio::net::TcpListener,
    model_name: String,
    engine: Arc<dyn ChatEngine>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(serve_chat(listener, model_name, engine))
}

async fn serve_chat(
    listener: tokio::net::TcpListener,
    model_name: String,
    engine: Arc<dyn ChatEngine>,
) {
    let state = ChatServerState { model_name, engine };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/tokenize", post(chat_tokenize))
        .with_state(state);
    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("inproc chat server exited: {e}");
    }
}

/// Build a `ChatRequestIn` from a raw JSON body — strict on the fields we
/// honor, silent on the rest. Errors are client errors (400).
fn parse_chat_request(body: &Value) -> std::result::Result<ChatRequestIn, String> {
    let raw = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or("chat: 'messages' must be an array")?;
    let mut messages = Vec::with_capacity(raw.len());
    for (i, m) in raw.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or_else(|| format!("chat: messages[{i}].role must be a string"))?;
        let content = match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            // OpenAI content-parts array — concatenate the text parts.
            // Non-text parts (images, …) are out of scope and skipped.
            Some(Value::Array(parts)) => {
                let mut out = String::new();
                for p in parts {
                    if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                        out.push_str(t);
                    }
                }
                out
            }
            Some(Value::Null) | None => String::new(),
            _ => {
                return Err(format!(
                    "chat: messages[{i}].content must be a string or content-part array"
                ))
            }
        };
        messages.push((role.to_string(), content));
    }
    if messages.is_empty() {
        return Err("chat: 'messages' must not be empty".to_string());
    }
    Ok(ChatRequestIn {
        messages,
        max_tokens: body
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v.min(u32::MAX as u64) as u32)
            .unwrap_or(DEFAULT_MAX_TOKENS),
        temperature: body
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(DEFAULT_TEMPERATURE),
        top_p: body.get("top_p").and_then(|v| v.as_f64()).map(|v| v as f32),
        top_k: body
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v.min(u32::MAX as u64) as u32),
        stream: body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false),
    })
}

/// Process-unique completion id. Monotonic counter + pid keeps ids unique
/// across in-process servers without a uuid dependency.
fn completion_id() -> String {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("chatcmpl-{}-{}", std::process::id(), n)
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// One OpenAI `chat.completion.chunk` — the byte shape llama-server emits
/// and lamu-api's `parse_upstream_line` consumes (`choices[0].delta` /
/// `choices[0].finish_reason`).
fn chat_chunk(id: &str, created: u64, model: &str, delta: Value, finish: Value) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
}

/// The include_usage-style trailing usage chunk (`choices: []` + `usage`).
/// lamu-api's bridges gate their ADR 0021 occupancy block on its
/// `prompt_tokens`.
fn usage_chunk(id: &str, created: u64, model: &str, prompt: usize, completion: usize) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    })
}

/// `POST /v1/chat/completions` — stream + non-stream.
///
/// Error seam (what lamu-api's bridge understands, see its
/// `send_upstream`/`backend_error_message`):
/// - Failures BEFORE any output → plain non-2xx `{"error":{"message"}}`.
///   For the streaming path this works because the handler waits for the
///   FIRST fragment before committing to a 200/SSE response.
/// - Failures AFTER streaming started → one in-stream
///   `data: {"error":{...}}` line, then the connection closes WITHOUT
///   `[DONE]` (the bridge flushes whatever content arrived and closes; an
///   output-less stream trips its `backend_returned_empty` gate).
async fn chat_completions(State(state): State<ChatServerState>, Json(body): Json<Value>) -> Response {
    let req = match parse_chat_request(&body) {
        Ok(r) => r,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let stream_requested = req.stream;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
    let engine = state.engine.clone();
    let gen_task = tokio::spawn(async move { engine.generate(req, tx).await });

    let id = completion_id();
    let created = unix_now();

    if !stream_requested {
        let mut content = String::new();
        while let Some(frag) = rx.recv().await {
            content.push_str(&frag);
        }
        return match gen_task.await {
            Err(e) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("generation task panicked: {e}"),
            ),
            Ok(Err(e)) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("generation failed: {e:#}"),
            ),
            Ok(Ok((prompt_tokens, completion_tokens, finish_reason))) => Json(json!({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": state.model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }))
            .into_response(),
        };
    }

    // Streaming. Hold the response until the first event so pre-output
    // failures stay a clean non-2xx (see handler docs).
    let first = rx.recv().await;
    // Option dance: the early branch consumes the JoinHandle; the stream
    // below takes it only when the early branch didn't.
    let mut gen_task = Some(gen_task);
    let early_result = if first.is_none() {
        // Channel closed with zero fragments: generation already finished
        // (legitimately-empty completion) or failed before any output.
        match gen_task.take().expect("handle untouched before this").await {
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("generation task panicked: {e}"),
                )
            }
            Ok(Err(e)) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("generation failed: {e:#}"),
                )
            }
            Ok(Ok(triple)) => Some(triple),
        }
    } else {
        None
    };

    let model_name = state.model_name.clone();
    let s = async_stream::stream! {
        if let Some((p, c, fr)) = early_result {
            // Legitimately-empty completion: finish + usage + DONE, no
            // content chunks. The finish_reason keeps lamu-api's
            // backend_returned_empty gate from misfiring.
            yield Ok::<_, Infallible>(Event::default()
                .data(chat_chunk(&id, created, &model_name, json!({}), json!(fr)).to_string()));
            yield Ok(Event::default().data(usage_chunk(&id, created, &model_name, p, c).to_string()));
            yield Ok(Event::default().data("[DONE]"));
            return;
        }
        // First fragment carries the role, OpenAI-style.
        let frag = first.expect("first is Some when early_result is None");
        yield Ok(Event::default().data(
            chat_chunk(&id, created, &model_name, json!({"role": "assistant", "content": frag}), Value::Null)
                .to_string(),
        ));
        while let Some(frag) = rx.recv().await {
            yield Ok(Event::default().data(
                chat_chunk(&id, created, &model_name, json!({"content": frag}), Value::Null).to_string(),
            ));
        }
        match gen_task.take().expect("present when early_result is None").await {
            Err(e) => {
                yield Ok(Event::default().data(
                    json!({"error": {"type": "backend_error", "message": format!("generation task panicked: {e}")}})
                        .to_string(),
                ));
                // Deliberately no [DONE]: a mid-stream failure must not
                // look like a clean close.
            }
            Ok(Err(e)) => {
                yield Ok(Event::default().data(
                    json!({"error": {"type": "backend_error", "message": format!("generation failed: {e:#}")}})
                        .to_string(),
                ));
            }
            Ok(Ok((p, c, fr))) => {
                yield Ok(Event::default()
                    .data(chat_chunk(&id, created, &model_name, json!({}), json!(fr)).to_string()));
                yield Ok(Event::default().data(usage_chunk(&id, created, &model_name, p, c).to_string()));
                yield Ok(Event::default().data("[DONE]"));
            }
        }
    };
    Sse::new(Box::pin(s)).into_response()
}

/// `POST /tokenize` `{content}` → `{"tokens":[...]}` — llama-server's wire
/// shape, backed by the engine tokenizer (ADR 0021).
///
/// The ids are SEQUENTIAL PLACEHOLDERS (`0..count`), not real vocab ids:
/// the `ChatEngine` seam deliberately carries only the count, because its
/// one in-tree consumer (`lamu_core::backends::llamacpp::tokenize_count_at`,
/// shared by `Backend::tokenize_count` and MCP `context_status`) measures
/// the ARRAY LENGTH. If a future caller needs real ids, widen `ChatEngine`
/// — don't fabricate meaning here.
async fn chat_tokenize(State(state): State<ChatServerState>, Json(body): Json<Value>) -> Response {
    let Some(content) = body.get("content").and_then(|c| c.as_str()).map(String::from) else {
        return error_response(StatusCode::BAD_REQUEST, "tokenize: 'content' must be a string");
    };
    let engine = state.engine.clone();
    match tokio::task::spawn_blocking(move || engine.tokenize_count(&content)).await {
        Ok(Ok(count)) => {
            Json(json!({"tokens": (0..count as u64).collect::<Vec<u64>>()})).into_response()
        }
        Ok(Err(e)) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("tokenize failed: {e:#}"),
        ),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("tokenize task panicked: {e}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeEngine {
        dims: usize,
        fail: bool,
    }

    impl EmbedEngine for FakeEngine {
        fn id(&self) -> &str {
            "fake"
        }
        fn dims(&self) -> usize {
            self.dims
        }
        fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            if self.fail {
                anyhow::bail!("synthetic engine failure");
            }
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32; self.dims])
                .collect())
        }
    }

    async fn spawn_test_server(fail: bool) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = spawn_embed_server_on(
            listener,
            "test-embed".to_string(),
            Arc::new(FakeEngine { dims: 4, fail }),
        );
        (port, handle)
    }

    #[tokio::test]
    async fn health_served_on_both_paths_with_ok_status() {
        let (port, handle) = spawn_test_server(false).await;
        for path in ["/health", "/v1/health"] {
            let resp = reqwest::get(format!("http://127.0.0.1:{port}{path}"))
                .await
                .unwrap();
            assert!(resp.status().is_success(), "{path} must be 2xx");
            let body: Value = resp.json().await.unwrap();
            assert_eq!(
                body.get("status").and_then(|v| v.as_str()),
                Some("ok"),
                "{path} must carry status:ok (llamacpp is_healthy contract)"
            );
        }
        handle.abort();
    }

    #[tokio::test]
    async fn embeddings_single_string_input() {
        let (port, handle) = spawn_test_server(false).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/v1/embeddings"))
            .json(&json!({"input": "hello world!", "model": "ignored"}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["object"], "list");
        assert_eq!(body["model"], "test-embed");
        let data = body["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["object"], "embedding");
        assert_eq!(data[0]["index"], 0);
        assert_eq!(data[0]["embedding"].as_array().unwrap().len(), 4);
        // "hello world!" is 12 chars → 3 approx tokens.
        assert_eq!(body["usage"]["prompt_tokens"], 3);
        assert_eq!(body["usage"]["total_tokens"], 3);
        handle.abort();
    }

    #[tokio::test]
    async fn embeddings_array_input_indexed_in_order() {
        let (port, handle) = spawn_test_server(false).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/v1/embeddings"))
            .json(&json!({"input": ["a", "bbbb"]}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let body: Value = resp.json().await.unwrap();
        let data = body["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["index"], 0);
        assert_eq!(data[1]["index"], 1);
        // FakeEngine encodes text length — proves order is preserved.
        assert_eq!(data[0]["embedding"][0], 1.0);
        assert_eq!(data[1]["embedding"][0], 4.0);
        handle.abort();
    }

    #[tokio::test]
    async fn embeddings_engine_failure_is_500_with_error_shape() {
        let (port, handle) = spawn_test_server(true).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/v1/embeddings"))
            .json(&json!({"input": "boom"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 500);
        let body: Value = resp.json().await.unwrap();
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("synthetic engine failure"), "got: {msg}");
        handle.abort();
    }

    #[tokio::test]
    async fn embeddings_bad_input_is_400() {
        let (port, handle) = spawn_test_server(false).await;
        let client = reqwest::Client::new();
        for bad in [json!({"input": 42}), json!({}), json!({"input": [1, 2]})] {
            let resp = client
                .post(format!("http://127.0.0.1:{port}/v1/embeddings"))
                .json(&bad)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status().as_u16(), 400, "payload: {bad}");
        }
        handle.abort();
    }

    #[tokio::test]
    async fn tokenize_is_a_documented_404() {
        let (port, handle) = spawn_test_server(false).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/tokenize"))
            .json(&json!({"content": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
        let body: Value = resp.json().await.unwrap();
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("embeddings-only"));
        handle.abort();
    }

    // ───────────────────── chat server (ADR 0035) ─────────────────────

    /// Scriptable fake: emits `fragments` then returns `result`.
    struct FakeChatEngine {
        fragments: Vec<&'static str>,
        result: anyhow::Result<(usize, usize, String)>,
        tokens_per_char: bool,
    }

    impl FakeChatEngine {
        fn ok(fragments: Vec<&'static str>, prompt: usize, completion: usize) -> Self {
            Self {
                fragments,
                result: Ok((prompt, completion, "stop".to_string())),
                tokens_per_char: false,
            }
        }
        fn failing(fragments: Vec<&'static str>, msg: &str) -> Self {
            Self {
                fragments,
                result: Err(anyhow::anyhow!("{}", msg.to_string())),
                tokens_per_char: false,
            }
        }
    }

    #[async_trait::async_trait]
    impl ChatEngine for FakeChatEngine {
        fn model(&self) -> &str {
            "fake-chat"
        }
        fn tokenize_count(&self, text: &str) -> anyhow::Result<usize> {
            if self.tokens_per_char {
                Ok(text.chars().count())
            } else {
                anyhow::bail!("tokenizer broken (synthetic)")
            }
        }
        async fn generate(
            &self,
            _req: ChatRequestIn,
            tx: tokio::sync::mpsc::Sender<String>,
        ) -> anyhow::Result<(usize, usize, String)> {
            for f in &self.fragments {
                if tx.send(f.to_string()).await.is_err() {
                    break;
                }
            }
            match &self.result {
                Ok(t) => Ok(t.clone()),
                Err(e) => Err(anyhow::anyhow!("{e}")),
            }
        }
    }

    async fn spawn_chat_test_server(engine: FakeChatEngine) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = spawn_chat_server_on(listener, "test-chat".to_string(), Arc::new(engine));
        (port, handle)
    }

    fn chat_body(stream: bool) -> Value {
        json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "temperature": 0.2,
            "stream": stream,
            // Unknown fields must be accepted-and-ignored (bridge extras).
            "cache_prompt": true,
            "stream_options": {"include_usage": true},
        })
    }

    /// Split an SSE body into its `data: ` payloads — the same line
    /// extraction lamu-api's bridge applies upstream.
    fn data_lines(body: &str) -> Vec<String> {
        body.lines()
            .filter_map(|l| l.trim().strip_prefix("data: ").map(String::from))
            .collect()
    }

    #[tokio::test]
    async fn chat_health_served_on_both_paths() {
        let (port, handle) = spawn_chat_test_server(FakeChatEngine::ok(vec!["x"], 1, 1)).await;
        for path in ["/health", "/v1/health"] {
            let resp = reqwest::get(format!("http://127.0.0.1:{port}{path}")).await.unwrap();
            assert!(resp.status().is_success());
            let body: Value = resp.json().await.unwrap();
            assert_eq!(body["status"], "ok", "{path}");
        }
        handle.abort();
    }

    #[tokio::test]
    async fn chat_non_stream_is_llama_server_shaped() {
        let (port, handle) =
            spawn_chat_test_server(FakeChatEngine::ok(vec!["Hel", "lo"], 7, 2)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&chat_body(false))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "test-chat");
        assert_eq!(body["choices"][0]["index"], 0);
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["prompt_tokens"], 7);
        assert_eq!(body["usage"]["completion_tokens"], 2);
        assert_eq!(body["usage"]["total_tokens"], 9);
        handle.abort();
    }

    #[tokio::test]
    async fn chat_stream_sse_framing_matches_bridge_expectations() {
        let (port, handle) =
            spawn_chat_test_server(FakeChatEngine::ok(vec!["He", "y"], 5, 2)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&chat_body(true))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let body = resp.text().await.unwrap();
        let lines = data_lines(&body);
        assert_eq!(lines.last().map(|s| s.as_str()), Some("[DONE]"));

        // Replay the exact extraction lamu-api's parse_upstream_line does:
        // delta.content tokens, finish_reason, usage counts.
        let mut tokens = String::new();
        let mut finish = String::new();
        let mut usage: Option<(u64, u64)> = None;
        for line in &lines {
            if line == "[DONE]" {
                break;
            }
            let v: Value = serde_json::from_str(line).expect("every data line is JSON");
            assert_eq!(v["object"], "chat.completion.chunk");
            if let Some(fr) = v["choices"][0]["finish_reason"].as_str() {
                finish = fr.to_string();
            }
            if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                usage = Some((
                    u["prompt_tokens"].as_u64().unwrap(),
                    u["completion_tokens"].as_u64().unwrap(),
                ));
            }
            if let Some(tok) = v["choices"][0]["delta"]["content"].as_str() {
                tokens.push_str(tok);
            }
        }
        assert_eq!(tokens, "Hey");
        assert_eq!(finish, "stop");
        assert_eq!(usage, Some((5, 2)));
        // First content chunk carries the role.
        let first: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
        handle.abort();
    }

    #[tokio::test]
    async fn chat_stream_empty_completion_still_closes_cleanly() {
        // No fragments, but a legitimate finish: stream must carry the
        // finish chunk + usage + [DONE] so the bridge's
        // backend_returned_empty gate sees a real finish_reason.
        let (port, handle) = spawn_chat_test_server(FakeChatEngine::ok(vec![], 3, 0)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&chat_body(true))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let body = resp.text().await.unwrap();
        let lines = data_lines(&body);
        assert_eq!(lines.last().map(|s| s.as_str()), Some("[DONE]"));
        let finish_seen = lines.iter().filter(|l| *l != "[DONE]").any(|l| {
            serde_json::from_str::<Value>(l)
                .ok()
                .and_then(|v| v["choices"][0]["finish_reason"].as_str().map(|s| s == "stop"))
                .unwrap_or(false)
        });
        assert!(finish_seen, "got lines: {lines:?}");
        handle.abort();
    }

    #[tokio::test]
    async fn chat_pre_stream_failure_is_non_2xx_json_error() {
        // Engine fails before ANY fragment → the handler must NOT have
        // committed to SSE; lamu-api's send_upstream decodes this body.
        let (port, handle) =
            spawn_chat_test_server(FakeChatEngine::failing(vec![], "tokenizer exploded")).await;
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&chat_body(true))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 500);
        let body: Value = resp.json().await.unwrap();
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("tokenizer exploded"), "got: {msg}");
        handle.abort();
    }

    #[tokio::test]
    async fn chat_mid_stream_failure_emits_error_line_and_no_done() {
        let (port, handle) =
            spawn_chat_test_server(FakeChatEngine::failing(vec!["par", "tial"], "cuda oom")).await;
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&chat_body(true))
            .send()
            .await
            .unwrap();
        // Already committed to the stream when the error hit.
        assert!(resp.status().is_success());
        let body = resp.text().await.unwrap();
        let lines = data_lines(&body);
        assert!(!lines.iter().any(|l| l == "[DONE]"), "mid-stream failure must not [DONE]: {lines:?}");
        let last: Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert!(
            last["error"]["message"].as_str().unwrap().contains("cuda oom"),
            "last line must be the error envelope: {last}"
        );
        handle.abort();
    }

    #[tokio::test]
    async fn chat_non_stream_failure_is_500() {
        let (port, handle) =
            spawn_chat_test_server(FakeChatEngine::failing(vec!["x"], "exploded late")).await;
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&chat_body(false))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 500);
        let body: Value = resp.json().await.unwrap();
        assert!(body["error"]["message"].as_str().unwrap().contains("exploded late"));
        handle.abort();
    }

    #[tokio::test]
    async fn chat_bad_request_is_400() {
        let (port, handle) = spawn_chat_test_server(FakeChatEngine::ok(vec!["x"], 1, 1)).await;
        let client = reqwest::Client::new();
        for bad in [
            json!({}),                                    // no messages
            json!({"messages": "nope"}),                  // wrong type
            json!({"messages": []}),                      // empty
            json!({"messages": [{"content": "hi"}]}),     // missing role
            json!({"messages": [{"role": "user", "content": 42}]}), // bad content
        ] {
            let resp = client
                .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
                .json(&bad)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status().as_u16(), 400, "payload: {bad}");
        }
        handle.abort();
    }

    #[tokio::test]
    async fn chat_tokenize_returns_engine_count() {
        let engine = FakeChatEngine {
            fragments: vec![],
            result: Ok((0, 0, "stop".into())),
            tokens_per_char: true,
        };
        let (port, handle) = spawn_chat_test_server(engine).await;
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/tokenize"))
            .json(&json!({"content": "hello"}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let body: Value = resp.json().await.unwrap();
        // ADR 0021 consumers count the array length.
        assert_eq!(body["tokens"].as_array().unwrap().len(), 5);
        handle.abort();
    }

    #[tokio::test]
    async fn chat_tokenize_engine_failure_is_500_and_bad_body_400() {
        let (port, handle) = spawn_chat_test_server(FakeChatEngine::ok(vec![], 0, 0)).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/tokenize"))
            .json(&json!({"content": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 500, "engine tokenizer error must 500");
        let resp = client
            .post(format!("http://127.0.0.1:{port}/tokenize"))
            .json(&json!({"nope": 1}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400, "missing content must 400");
        handle.abort();
    }

    #[test]
    fn parse_chat_request_defaults_and_content_parts() {
        let req = parse_chat_request(&json!({
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "assistant"},
            ],
        }))
        .unwrap();
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[1], ("user".to_string(), "ab".to_string()));
        assert_eq!(req.messages[2].1, "", "absent content is empty, not an error");
        assert_eq!(req.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(req.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(req.top_p, None);
        assert_eq!(req.top_k, None);
        assert!(!req.stream);

        let req = parse_chat_request(&json!({
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 9, "temperature": 0.5, "top_p": 0.9, "top_k": 40, "stream": true,
        }))
        .unwrap();
        assert_eq!(req.max_tokens, 9);
        assert_eq!(req.temperature, 0.5);
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.top_k, Some(40));
        assert!(req.stream);
    }

    #[tokio::test]
    async fn port_taking_wrapper_binds_and_serves() {
        // Find a free port the OS way, release it, then hand it to the
        // port-taking wrapper (small race window — acceptable in a test).
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);
        let handle = spawn_embed_server(
            port,
            "wrapper".to_string(),
            Arc::new(FakeEngine { dims: 2, fail: false }),
        );
        // Poll health like a real loader would — bind happens inside the task.
        let mut ok = false;
        for _ in 0..50 {
            if let Ok(resp) = reqwest::get(format!("http://127.0.0.1:{port}/health")).await {
                if resp.status().is_success() {
                    ok = true;
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(ok, "port-taking wrapper must come up healthy on {port}");
        handle.abort();
    }
}
