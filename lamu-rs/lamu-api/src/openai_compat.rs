//! OpenAI-compatible HTTP layer.
//! Direct port of `lamu/api/openai_compat.py`.

use crate::metrics::LamuMetrics;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router as AxumRouter;
use futures_util::stream::Stream;
use lamu_core::health::HealthRegistry;
use lamu_core::reasoning::get_extractor;
use lamu_core::registry::load_registry;
use lamu_core::router::Router;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{Capability, ModelEntry, ReasoningMarker};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Toggle Qwen3.6 / Qwen3.5 reasoning mode. When None, defaults to
    /// the backend's chat template default (typically thinking ON for
    /// Qwen3.6). Set false to skip the `<think>` block entirely and
    /// shave wall time on simple queries.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

fn default_max_tokens() -> u32 { 16384 }
fn default_temperature() -> f32 { 0.7 }

#[derive(Clone)]
pub struct AppState {
    pub scheduler: Arc<Mutex<VramScheduler>>,
    pub router: Arc<Mutex<Router>>,
    pub entries: Arc<HashMap<String, ModelEntry>>,
    pub client: reqwest::Client,
    /// Shared with the daemon when the v3 wire-up matures. Today the
    /// HTTP layer creates its own — DEAD/QUARANTINED filtering still
    /// works, just per-surface state.
    pub health: Arc<Mutex<HealthRegistry>>,
    pub metrics: Arc<LamuMetrics>,
}

pub fn build_app(state: AppState) -> AxumRouter {
    AxumRouter::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        // Anthropic Messages API shim for Claude Code et al. Translates
        // /v1/messages payload → OpenAI ChatRequest, delegates to
        // chat_completions, then maps the response back.
        .route("/v1/messages", post(anthropic_messages))
        .with_state(state)
}

async fn metrics_endpoint(State(state): State<AppState>) -> Response {
    state.metrics.scrapes_total.inc();
    {
        let scheduler = state.scheduler.lock();
        let health = state.health.lock();
        state.metrics.refresh(&scheduler, &health, None);
    }
    let (body, ctype) = state.metrics.render();
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, ctype)],
        body,
    )
        .into_response()
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    let n = state.scheduler.lock().loaded_models().len();
    Json(json!({"status": "ok", "models_loaded": n}))
}

async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let mut data = Vec::new();
    let scheduler = state.scheduler.lock();
    for (name, entry) in state.entries.iter() {
        let loaded = scheduler.is_loaded(name);
        let caps: Vec<&str> = entry.capabilities.iter().map(|c| match c {
            Capability::Chat => "chat",
            Capability::Code => "code",
            Capability::Reasoning => "reasoning",
            Capability::Routing => "routing",
            Capability::Vision => "vision",
            Capability::LongContext => "long_context",
        }).collect();
        data.push(json!({
            "id": name,
            "object": "model",
            "owned_by": "local",
            "loaded": loaded,
            "params_b": entry.params_b,
            "vram_mb": entry.vram_mb,
            "capabilities": caps,
        }));
    }
    Json(json!({"data": data, "object": "list"}))
}

#[derive(Serialize)]
struct ErrorResponse<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    message: String,
    #[serde(rename = "type")]
    typ: &'a str,
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let completion_id = format!("chatcmpl-{}", random_hex(12));
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let t_start = Instant::now();

    // Refuse before VRAM allocation if `lamu-train` (or another
    // exclusive holder) owns the GPU. Check before taking the
    // scheduler/router locks so a held GPU returns fast without
    // contending on internal state.
    if let Err(e) = lamu_core::scheduler_lock::check_unlocked() {
        state.metrics.requests_total
            .with_label_values(&[req.model.as_deref().unwrap_or("unknown"), "gpu_locked"])
            .inc();
        let body = ErrorResponse {
            error: ErrorBody {
                message: format!("{e}"),
                typ: "gpu_locked",
            },
        };
        let body_json = serde_json::to_value(&body)
            .unwrap_or_else(|err| json!({"error": {
                "message": format!("internal serialization: {}", err),
                "type": "serialization_error",
            }}));
        return (StatusCode::SERVICE_UNAVAILABLE, Json(body_json)).into_response();
    }

    let (port, model_name, marker) = {
        let scheduler = state.scheduler.lock();
        let router = state.router.lock();
        let health = state.health.lock();
        // health_map filters DEAD/QUARANTINED. health is populated as the
        // OpenAI-compat layer sees failures from backends — see the error
        // arms below.
        let decision = router.route(
            &scheduler,
            req.model.as_deref(),
            None,
            Some(health.all()),
        );

        if decision.model_name.is_empty() || !decision.loaded {
            state.metrics.requests_total
                .with_label_values(&[req.model.as_deref().unwrap_or("unknown"), "no_backend"])
                .inc();
            let body = ErrorResponse {
                error: ErrorBody {
                    message: format!("No loaded model available: {}", decision.reason),
                    typ: "model_not_available",
                },
            };
            let body_json = serde_json::to_value(&body)
                .unwrap_or_else(|e| json!({"error": {
                    "message": format!("internal serialization: {}", e),
                    "type": "internal"
                }}));
            return (StatusCode::SERVICE_UNAVAILABLE, Json(body_json)).into_response();
        }

        let Some(loaded) = scheduler.get_loaded(&decision.model_name) else {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {"message": "internal: lost loaded model"}}))).into_response();
        };
        let port = loaded.port;
        let entry = state.entries.get(&decision.model_name).cloned();
        let marker = entry.as_ref().and_then(|e| e.reasoning_marker.clone());
        (port, decision.model_name, marker)
    };

    {
        let mut scheduler = state.scheduler.lock();
        scheduler.mark_used(&model_name);
    }

    let messages_json: Vec<Value> = req.messages.iter()
        .map(|m| json!({"role": m.role, "content": m.content}))
        .collect();

    let mut payload = json!({
        "messages": messages_json,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    });
    if let Some(k) = req.top_k { payload["top_k"] = json!(k); }
    if let Some(p) = req.top_p { payload["top_p"] = json!(p); }
    // Thread enable_thinking through to backend's chat template kwargs.
    // bee llama-server consumes chat_template_kwargs.enable_thinking
    // (Qwen3.6 / Qwen3.5 chat templates honor it).
    if let Some(et) = req.enable_thinking {
        payload["chat_template_kwargs"] = json!({ "enable_thinking": et });
    }

    // Routing: by default, forward straight to the loaded backend on its
    // local port. If LAMU_GATEWAY_URL is set (e.g. a Bifrost endpoint),
    // forward through that instead — Bifrost then dispatches to the
    // backend keyed by `model`. Bifrost expects `provider/model` ids
    // (e.g. `qwen/qwen3.6-27b-uncensored`); we override the payload's
    // `model` field with whatever the user sent so client-side mappings
    // pass through unchanged.
    let backend_url = if let Ok(gw) = std::env::var("LAMU_GATEWAY_URL") {
        // Validate the env var is a well-formed http(s) URL with no
        // userinfo/fragment — refuses redirects to file://, weird
        // schemes, or credentialed URLs that could leak through.
        let parsed = match reqwest::Url::parse(&gw) {
            Ok(u) => u,
            Err(_) => {
                // Don't echo the bad URL back — could leak misconfigured
                // hostnames or credentials embedded by mistake.
                eprintln!("openai_compat: LAMU_GATEWAY_URL parse failed");
                return (StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": {
                        "message": "LAMU_GATEWAY_URL is not a valid HTTP URL",
                        "type": "config"}}))).into_response();
            }
        };
        match parsed.scheme() {
            "http" | "https" => {}
            other => {
                return (StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": {
                        "message": format!("LAMU_GATEWAY_URL scheme '{}' not allowed", other),
                        "type": "config"}}))).into_response();
            }
        }
        if parsed.username() != "" || parsed.password().is_some() {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {
                    "message": "LAMU_GATEWAY_URL must not contain userinfo",
                    "type": "config"}}))).into_response();
        }
        let trimmed = gw.trim_end_matches('/');
        if let Some(model) = req.model.as_deref() {
            payload["model"] = json!(model);
        }
        format!("{}/chat/completions", trimmed)
    } else {
        format!("http://localhost:{}/v1/chat/completions", port)
    };

    if req.stream {
        return stream_response(state.client.clone(), backend_url, payload,
                               completion_id, created, model_name, marker).await
            .into_response();
    }

    // Non-streaming
    let resp = match state.client.post(&backend_url).json(&payload).send().await {
        Ok(r) => r,
        Err(e) => {
            state.health.lock().get_or_create(&model_name).record_error(format!("{e}"));
            state.metrics.requests_total
                .with_label_values(&[&model_name, "backend_error"])
                .inc();
            return (StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": format!("Backend unreachable: {}", e)}}))).into_response();
        }
    };

    let data: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            state.health.lock().get_or_create(&model_name).record_error(format!("{e}"));
            state.metrics.requests_total
                .with_label_values(&[&model_name, "backend_error"])
                .inc();
            return (StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": format!("Bad JSON from backend: {}", e)}}))).into_response();
        }
    };

    let msg = data.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message"));
    let raw_content = msg.and_then(|m| m.get("content")).and_then(|v| v.as_str()).unwrap_or("");
    let reasoning_content = msg.and_then(|m| m.get("reasoning_content")).and_then(|v| v.as_str()).unwrap_or("");

    let extractor = get_extractor(marker);
    let (reasoning, content) = if !reasoning_content.is_empty() {
        (reasoning_content.to_string(), raw_content.to_string())
    } else {
        extractor.split(raw_content)
    };

    let finish = data.get("choices").and_then(|c| c.get(0))
        .and_then(|c| c.get("finish_reason")).and_then(|v| v.as_str())
        .unwrap_or("stop");

    let usage = data.get("usage").cloned().unwrap_or(Value::Null);
    let timings = data.get("timings").cloned();

    let mut message_obj = json!({
        "role": "assistant",
        "content": content,
    });
    if !reasoning.is_empty() {
        message_obj["reasoning_content"] = Value::String(reasoning);
    }

    let mut response = json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": message_obj,
            "finish_reason": finish,
        }],
        "usage": usage.clone(),
    });
    if let Some(t) = timings {
        response["timings"] = t;
    }

    // Metrics: success path — match Python lamu/api/openai_compat.py.
    state.health.lock().get_or_create(&model_name).record_success();
    state.metrics.requests_total
        .with_label_values(&[&model_name, "ok"])
        .inc();
    state.metrics.request_duration_seconds
        .with_label_values(&[&model_name, "total"])
        .observe(t_start.elapsed().as_secs_f64());
    let completion_tokens = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    if completion_tokens > 0 {
        state.metrics.tokens_generated_total
            .with_label_values(&[&model_name, "content"])
            .inc_by(completion_tokens);
    }
    if !response["choices"][0]["message"]["reasoning_content"].is_null() {
        let r = response["choices"][0]["message"]["reasoning_content"]
            .as_str().map(|s| s.len() as u64 / 4).unwrap_or(0);
        if r > 0 {
            state.metrics.tokens_generated_total
                .with_label_values(&[&model_name, "reasoning"])
                .inc_by(r);
        }
    }

    Json(response).into_response()
}

async fn stream_response(
    client: reqwest::Client,
    backend_url: String,
    payload: Value,
    completion_id: String,
    created: u64,
    model_name: String,
    marker: Option<ReasoningMarker>,
) -> Sse<Pin<Box<dyn Stream<Item = std::result::Result<Event, Infallible>> + Send>>> {
    let s = async_stream::stream! {
        let resp = match client.post(&backend_url).json(&payload).send().await {
            Ok(r) => r,
            Err(e) => {
                let chunk = json!({"error": format!("backend: {}", e)});
                yield Ok(Event::default().data(chunk.to_string()));
                yield Ok(Event::default().data("[DONE]"));
                return;
            }
        };

        let mut byte_stream = resp.bytes_stream();
        let mut buf = String::new();

        let open_tag = marker.as_ref().map(|m| m.open_tag.clone()).unwrap_or_else(|| "<think>".to_string());
        let close_tag = marker.as_ref().map(|m| m.close_tag.clone()).unwrap_or_else(|| "</think>".to_string());

        let mut pending = String::new();
        let mut in_reasoning = false;
        let mut reasoning_done = false;

        use futures_util::stream::StreamExt;
        while let Some(chunk_res) = byte_stream.next().await {
            let Ok(bytes) = chunk_res else { break };
            buf.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(nl) = buf.find('\n') {
                let line: String = buf.drain(..=nl).collect();
                let line = line.trim();
                let Some(rest) = line.strip_prefix("data: ") else { continue };
                if rest == "[DONE]" {
                    if !pending.trim().is_empty() && reasoning_done {
                        let chunk = make_chunk(&completion_id, created, &model_name, &pending);
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                    let done_chunk = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    });
                    yield Ok(Event::default().data(done_chunk.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                let Ok(val) = serde_json::from_str::<Value>(rest) else { continue };
                let Some(token) = val.get("choices").and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                else { continue };
                if token.is_empty() { continue; }

                pending.push_str(token);

                if !in_reasoning && !reasoning_done {
                    if let Some(idx) = pending.find(open_tag.as_str()) {
                        in_reasoning = true;
                        let pre = pending[..idx].to_string();
                        let after = pending[idx + open_tag.len()..].to_string();
                        pending = after;
                        if !pre.trim().is_empty() {
                            let chunk = make_chunk(&completion_id, created, &model_name, &pre);
                            yield Ok(Event::default().data(chunk.to_string()));
                        }
                    } else if pending.len() > open_tag.len() * 3 {
                        reasoning_done = true;
                        let chunk = make_chunk(&completion_id, created, &model_name, &pending);
                        yield Ok(Event::default().data(chunk.to_string()));
                        pending.clear();
                    }
                } else if in_reasoning && !reasoning_done {
                    if let Some(idx) = pending.find(close_tag.as_str()) {
                        reasoning_done = true;
                        in_reasoning = false;
                        pending = pending[idx + close_tag.len()..].to_string();
                        if !pending.trim().is_empty() {
                            let chunk = make_chunk(&completion_id, created, &model_name, &pending);
                            yield Ok(Event::default().data(chunk.to_string()));
                            pending.clear();
                        }
                    } else {
                        pending.clear();
                    }
                } else if reasoning_done {
                    let chunk = make_chunk(&completion_id, created, &model_name, token);
                    yield Ok(Event::default().data(chunk.to_string()));
                    pending.clear();
                }
            }
        }

        let done_chunk = json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        });
        yield Ok(Event::default().data(done_chunk.to_string()));
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(Box::pin(s))
}

fn make_chunk(id: &str, created: u64, model: &str, content: &str) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]
    })
}

fn random_hex(len: usize) -> String {
    use std::time::SystemTime;
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let mut s = format!("{:x}", nanos);
    s.truncate(len);
    s
}

/// Auto-register running models found on standard ports.
///
/// Probes `/v1/models` (not `/health`) so we learn which model is actually
/// serving on each port, then match against the registry by bidirectional
/// substring. Avoids registering the wrong entry just because it's first
/// in iteration order.
pub async fn auto_register(state: &AppState) {
    let probes = [lamu_core::config::PORT_MAIN, lamu_core::config::PORT_SIDECAR];
    for port in probes {
        let url = format!("http://localhost:{}/v1/models", port);
        let Ok(resp) = state.client.get(&url)
            .timeout(Duration::from_secs(2))
            .send().await
        else { continue };
        let Ok(j) = resp.json::<Value>().await else { continue };
        let Some(model_id) = j.get("data").and_then(|d| d.get(0))
            .and_then(|m| m.get("id")).and_then(|v| v.as_str())
        else { continue };
        let model_id = model_id.to_lowercase();

        let mut scheduler = state.scheduler.lock();
        for entry in state.entries.values() {
            let ename = entry.name.to_lowercase();
            if ename.contains(&model_id) || model_id.contains(&ename) {
                if !scheduler.is_loaded(&entry.name) {
                    scheduler.register_loaded(entry.clone(), None, port, entry.vram_mb);
                }
                break;
            }
        }
    }
}

/// Load registry from default path + build app state.
pub fn build_state(registry_path: &Path) -> anyhow::Result<AppState> {
    let entries_vec = load_registry(registry_path)?;
    let scheduler = VramScheduler::new();
    let router = Router::new(&scheduler, entries_vec.clone());
    let entries: HashMap<String, ModelEntry> = entries_vec.into_iter()
        .map(|e| (e.name.clone(), e)).collect();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;
    let metrics = LamuMetrics::new()
        .map_err(|e| anyhow::anyhow!("prometheus init: {e}"))?;
    Ok(AppState {
        scheduler: Arc::new(Mutex::new(scheduler)),
        router: Arc::new(Mutex::new(router)),
        entries: Arc::new(entries),
        client,
        health: Arc::new(Mutex::new(HealthRegistry::new())),
        metrics: Arc::new(metrics),
    })
}

// ── Anthropic Messages API shim ─────────────────────────────────────
//
// Claude Code, Continue (anthropic mode), and other tools call this
// surface. We translate to our internal ChatRequest, reuse the OpenAI
// pipeline, then map the response back into Anthropic's envelope.
// Streaming SSE is not yet implemented — non-streaming only.

#[derive(Debug, Clone, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: serde_json::Value, // string OR vec of content blocks
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    system: Option<serde_json::Value>, // string OR vec of content blocks
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    top_p: Option<f32>,
}

fn anthropic_content_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            // Concatenate text blocks; ignore tool_use / image blocks for now.
            arr.iter()
                .filter_map(|b| {
                    let ty = b.get("type").and_then(|x| x.as_str()).unwrap_or("");
                    if ty == "text" {
                        b.get("text").and_then(|x| x.as_str()).map(String::from)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => String::new(),
    }
}

async fn anthropic_messages(
    State(state): State<AppState>,
    Json(req): Json<AnthropicRequest>,
) -> impl IntoResponse {
    if req.stream {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "type": "error",
                "error": {"type": "not_implemented",
                          "message": "streaming on /v1/messages not yet supported by lamu"}
            })),
        )
            .into_response();
    }

    // Translate to OpenAI-style ChatRequest.
    let mut messages: Vec<Message> = Vec::new();
    if let Some(sys) = req.system.as_ref() {
        let sys_text = anthropic_content_to_string(sys);
        if !sys_text.is_empty() {
            messages.push(Message {
                role: "system".to_string(),
                content: sys_text,
            });
        }
    }
    for m in &req.messages {
        messages.push(Message {
            role: m.role.clone(),
            content: anthropic_content_to_string(&m.content),
        });
    }

    let oai_req = ChatRequest {
        model: req.model.clone(),
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stream: false,
        top_k: req.top_k,
        top_p: req.top_p,
        enable_thinking: None,
    };

    let resp = chat_completions(State(state), Json(oai_req)).await.into_response();
    let (parts, body) = resp.into_parts();
    if parts.status != StatusCode::OK {
        // Pass error envelope through (already JSON).
        return (parts.status, body).into_response();
    }
    let body_bytes = match axum::body::to_bytes(body, 4 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "type": "error",
                    "error": {"type": "internal", "message": format!("body read: {e}")}
                })),
            )
                .into_response();
        }
    };
    let oai_resp: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({
                    "type": "error",
                    "error": {"type": "bad_response", "message": format!("json: {e}")}
                })),
            )
                .into_response();
        }
    };

    // Map OAI response → Anthropic envelope.
    let oai_msg = oai_resp
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"));
    let oai_content = oai_msg
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let oai_model = oai_resp
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("lamu");
    let usage = oai_resp.get("usage");
    let in_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let out_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let anthro = json!({
        "id": format!("msg_{}", random_hex(12)),
        "type": "message",
        "role": "assistant",
        "model": oai_model,
        "content": [{
            "type": "text",
            "text": oai_content,
        }],
        "stop_reason": "end_turn",
        "stop_sequence": serde_json::Value::Null,
        "usage": {
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        },
    });
    (StatusCode::OK, Json(anthro)).into_response()
}
