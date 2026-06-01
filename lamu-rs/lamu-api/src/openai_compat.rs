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
use lamu_core::types::{Capability, ModelEntry, ReasoningMarker, SamplingProfile};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

// Custom deserializer for `Message.content` — pi (and other harnesses
// modeled on OpenAI's Vision spec) send `content` as an array of
// content parts: `[{type: "text", text: "..."}]`. lamu and bee's
// chat templates want a plain string. Flatten on the way in by
// concatenating any text parts and ignoring image / tool blocks
// (downstream will reject them with a clearer error if applicable).
impl<'de> Deserialize<'de> for Message {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            role: String,
            content: Value,
        }
        let r = Raw::deserialize(d)?;
        let content = match r.content {
            Value::String(s) => s,
            Value::Array(arr) => {
                let mut buf = String::new();
                for part in arr {
                    if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                        if !buf.is_empty() {
                            buf.push('\n');
                        }
                        buf.push_str(t);
                    }
                }
                buf
            }
            Value::Null => String::new(),
            // Anything else — bool, number, object — render as JSON so
            // we don't silently drop it. Edge case; production callers
            // shouldn't hit it.
            other => other.to_string(),
        };
        Ok(Message { role: r.role, content })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    // `max_tokens` / `temperature` are `Option` (NOT plain-with-default)
    // so omission is detectable — required to merge a per-model sampling
    // profile without clobbering legitimate client values. The builtin
    // defaults (16384 / 0.7) are applied at the payload-build site as the
    // final merge fallback, preserving prior effective behavior when no
    // profile and no request value are present.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    /// Toggle Qwen3.6 / Qwen3.5 reasoning mode. When None, defaults to
    /// the backend's chat template default (typically thinking ON for
    /// Qwen3.6). Set false to skip the `<think>` block entirely and
    /// shave wall time on simple queries.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// OpenAI tools array. Forwarded verbatim to bee. Anthropic
    /// translates its own tool schema → this on entry.
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
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
    /// The port this lamu serve listens on. Passed to
    /// `lamu_core::loader::ensure_loaded` so spawned backends don't try
    /// to bind the same port we already own.
    pub http_port: u16,
    /// Optional bearer token (ADR 0012). `None` → auth off (frictionless
    /// loopback). `Some` → every route except /health + /metrics requires
    /// `Authorization: Bearer <token>`. Resolved once at `build_state`.
    pub auth_token: Arc<Option<String>>,
}

pub fn build_app(state: AppState) -> AxumRouter {
    AxumRouter::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        // OpenAI-compat embeddings — backs RAG front-ends (odysseus
        // ChromaDB, etc.). Proxies to a local llama-server `--embedding`.
        .route("/v1/embeddings", post(embeddings))
        // Anthropic Messages API shim for Claude Code et al. Translates
        // /v1/messages payload → OpenAI ChatRequest, delegates to
        // chat_completions, then maps the response back.
        .route("/v1/messages", post(anthropic_messages))
        // Ollama-compat shim for AnythingLLM, Open WebUI (Ollama mode),
        // and other tools that hardcode the Ollama API surface.
        .route("/api/tags", get(ollama_tags))
        .route("/api/chat", post(ollama_chat))
        // Bearer auth (ADR 0012). No-op when no token is configured; /health +
        // /metrics are exempt inside the middleware.
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            crate::auth::require_bearer,
        ))
        // CORS OUTERMOST (added last = wraps first): browser frontends'
        // preflight OPTIONS is answered here before auth, and CORS headers
        // land on every response incl. 401. Permissive (any origin/method/
        // header, no credentials) — correct for a local backend whose auth
        // is a Bearer header, not a cookie (ADR 0016).
        .layer(tower_http::cors::CorsLayer::permissive())
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

/// Route the request to a loaded model, spawning the backend on demand
/// if the router says "will load". Returns
/// `(port, model_name, marker, sampling_profile)` on success or an
/// `IntoResponse` error envelope (503/500) on failure. The sampling
/// profile (if the resolved entry carries one) is merged into the
/// downstream payload by each caller.
///
/// Single place where `lamu serve` decides whether to spawn a backend.
/// Used by the OpenAI, Anthropic-stream, and Ollama-stream entry points.
/// OpenAI-compat `/v1/embeddings`. Resolves an embedding model (the
/// request's `model` if it's an embedding entry, else the first registry
/// entry with `Capability::Embedding`), ensure-loads it (spawns
/// llama-server `--embedding`), and proxies the body to the backend's
/// `/v1/embeddings`, passing the response through verbatim. Lets RAG
/// front-ends point `EMBEDDING_URL` at `lamu serve`.
async fn embeddings(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    let is_embed = |e: &ModelEntry| {
        e.capabilities.contains(&lamu_core::types::Capability::Embedding)
    };
    let req_model = body.get("model").and_then(|m| m.as_str());
    let name = req_model
        .filter(|m| state.entries.get(*m).map(is_embed).unwrap_or(false))
        .map(|m| m.to_string())
        .or_else(|| state.entries.values().find(|e| is_embed(e)).map(|e| e.name.clone()));
    let Some(name) = name else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {
                "message": "no embedding model in registry — add one with capability 'embedding' (llama-server --embedding)",
                "type": "model_not_available",
            }})),
        )
            .into_response();
    };

    let port = {
        let already = state.scheduler.lock().get_loaded(&name).and_then(|m| {
            if m.port != 0 { Some(m.port) } else { None }
        });
        match already {
            Some(p) => p,
            None => match lamu_core::loader::ensure_loaded(
                &name,
                state.entries.as_ref(),
                &state.scheduler,
                &state.health,
                Some(state.http_port),
            )
            .await
            {
                Ok(lm) => lm.port,
                Err(e) => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(json!({"error": {
                            "message": format!("failed to load embedding model '{name}': {e}"),
                            "type": "spawn_failed",
                        }})),
                    )
                        .into_response();
                }
            },
        }
    };

    let url = format!("http://localhost:{port}/v1/embeddings");
    match state.client.post(&url).json(&body).send().await {
        Ok(r) => {
            let status =
                StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            // A read failure here previously became `200 OK` + empty body
            // via unwrap_or_default → silent RAG breakage downstream. Surface
            // it as 502 instead.
            let bytes = match r.bytes().await {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    return (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({"error": {"message": format!("embeddings backend read failed: {e}")}})),
                    )
                        .into_response();
                }
            };
            let mut resp = (status, bytes).into_response();
            resp.headers_mut().insert(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            );
            resp
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": {"message": format!("embeddings backend: {e}")}})),
        )
            .into_response(),
    }
}

async fn resolve_and_ensure_loaded(
    state: &AppState,
    model_req: Option<&str>,
) -> std::result::Result<(u16, String, Option<ReasoningMarker>, Option<SamplingProfile>), Response> {
    let decision = {
        let scheduler = state.scheduler.lock();
        let router = state.router.lock();
        let health = state.health.lock();
        router.route(&scheduler, model_req, None, Some(health.all()))
    };

    if decision.model_name.is_empty() {
        state.metrics.requests_total
            .with_label_values(&[model_req.unwrap_or("unknown"), "no_candidate"])
            .inc();
        return Err((StatusCode::SERVICE_UNAVAILABLE, Json(json!({
            "error": {
                "message": format!("No model: {}", decision.reason),
                "type": "model_not_available",
            }
        }))).into_response());
    }

    if !decision.loaded {
        match lamu_core::loader::ensure_loaded(
            &decision.model_name,
            state.entries.as_ref(),
            &state.scheduler,
            &state.health,
            Some(state.http_port),
        ).await {
            Ok(_lm) => {}
            Err(e) => {
                state.metrics.requests_total
                    .with_label_values(&[&decision.model_name, "spawn_failed"])
                    .inc();
                return Err((StatusCode::SERVICE_UNAVAILABLE, Json(json!({
                    "error": {
                        "message": format!("Failed to load '{}': {}", decision.model_name, e),
                        "type": "spawn_failed",
                    }
                }))).into_response());
            }
        }
    }

    let scheduler = state.scheduler.lock();
    let Some(loaded) = scheduler.get_loaded(&decision.model_name) else {
        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({
            "error": {"message": "internal: lost loaded model after spawn"}
        }))).into_response());
    };
    let port = loaded.port;
    let entry = state.entries.get(&decision.model_name).cloned();
    let marker = entry.as_ref().and_then(|e| e.reasoning_marker.clone());
    let sampling = entry.as_ref().and_then(|e| e.sampling.clone());
    Ok((port, decision.model_name, marker, sampling))
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
            Capability::Embedding => "embedding",
        }).collect();
        data.push(json!({
            "id": name,
            "object": "model",
            "created": 0, // OpenAI-shape field; LAMU doesn't track model ctime
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

    let (port, model_name, marker, sampling) = match resolve_and_ensure_loaded(
        &state,
        req.model.as_deref(),
    ).await {
        Ok(t) => t,
        Err(resp) => return resp,
    };

    {
        let mut scheduler = state.scheduler.lock();
        scheduler.mark_used(&model_name);
    }

    let messages_json: Vec<Value> = req.messages.iter()
        .map(|m| json!({"role": m.role, "content": m.content}))
        .collect();

    // Merge the per-model sampling profile (if any) with the request's
    // sampler values. Precedence: locked profile field > request value >
    // unlocked profile field > builtin default. temperature/max_tokens
    // collapse to a concrete value (builtin default 0.7 / 16384) since
    // they're always present in the OpenAI payload; the others stay
    // Option so we only emit them when actually set (no nulls).
    let s = lamu_core::types::resolve_samplers(
        sampling.as_ref(),
        req.temperature, req.top_p, req.top_k, req.min_p, req.repeat_penalty,
        req.max_tokens,
    );
    let eff_temperature = s.temperature.unwrap_or_else(default_temperature);
    let eff_max_tokens = s.max_tokens.unwrap_or_else(default_max_tokens);

    let mut payload = json!({
        "messages": messages_json,
        "max_tokens": eff_max_tokens,
        "temperature": eff_temperature,
        "stream": req.stream,
    });
    if let Some(k) = s.top_k { payload["top_k"] = json!(k); }
    if let Some(p) = s.top_p { payload["top_p"] = json!(p); }
    if let Some(v) = s.min_p { payload["min_p"] = json!(v); }
    if let Some(v) = s.repeat_penalty { payload["repeat_penalty"] = json!(v); }
    // Thread enable_thinking through to backend's chat template kwargs.
    // bee llama-server consumes chat_template_kwargs.enable_thinking
    // (Qwen3.6 / Qwen3.5 chat templates honor it).
    if let Some(et) = req.enable_thinking {
        payload["chat_template_kwargs"] = json!({ "enable_thinking": et });
    }
    if let Some(tools) = req.tools.as_ref() {
        payload["tools"] = json!(tools);
    }
    if let Some(tc) = req.tool_choice.as_ref() {
        payload["tool_choice"] = tc.clone();
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
    let has_tool_calls = msg.and_then(|m| m.get("tool_calls"))
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    let raw_finish = data.get("choices").and_then(|c| c.get(0))
        .and_then(|c| c.get("finish_reason")).and_then(|v| v.as_str()).unwrap_or("");

    // 502 when the backend gave us a structurally-empty response: no
    // content, no reasoning, no tool calls, AND no recognized finish
    // reason. Distinguishes silent backend failure from a legitimate
    // empty completion (which always carries a finish_reason).
    if msg.is_none()
        || (raw_content.is_empty() && reasoning_content.is_empty() && !has_tool_calls
            && !matches!(raw_finish, "stop" | "length" | "tool_calls" | "content_filter"))
    {
        state.metrics.requests_total
            .with_label_values(&[&model_name, "backend_empty"])
            .inc();
        return (StatusCode::BAD_GATEWAY, Json(json!({
            "error": {
                "type": "backend_returned_empty",
                "message": format!(
                    "backend on :{} returned no usable content (finish_reason='{}')",
                    port, raw_finish
                ),
            }
        }))).into_response();
    }

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
        let mut buf: Vec<u8> = Vec::new();

        let open_tag = marker.as_ref().map(|m| m.open_tag.clone()).unwrap_or_else(|| "<think>".to_string());
        let close_tag = marker.as_ref().map(|m| m.close_tag.clone()).unwrap_or_else(|| "</think>".to_string());

        let mut pending = String::new();
        let mut in_reasoning = false;
        let mut reasoning_done = false;
        // Empty-backend gate state: did the backend yield ANY non-empty
        // token, and what finish_reason (if any) did it report? Mirrors
        // the non-streaming 502 backend_returned_empty gate.
        let mut any_content = false;
        let mut finish_reason = String::new();

        use futures_util::stream::StreamExt;
        while let Some(chunk_res) = byte_stream.next().await {
            let Ok(bytes) = chunk_res else { break };
            // Byte-buffer, decode whole lines only: from_utf8_lossy on a raw
            // chunk corrupts a multibyte char split across chunk boundaries.
            buf.extend_from_slice(&bytes);

            while let Some(line) = lamu_core::sse::next_sse_line(&mut buf) {
                let line = line.trim();
                let Some(rest) = line.strip_prefix("data: ") else { continue };
                if rest == "[DONE]" {
                    // Flush whatever's buffered. The reasoning-tag scan
                    // buffers tokens until either (a) it spots `<think>`
                    // and routes to reasoning, or (b) ~24 chars arrive
                    // without one and we declare reasoning_done. Short
                    // outputs (e.g. "ok\n") never hit either branch, so
                    // we have to flush at end-of-stream. Buffered content
                    // is only safe to drop if we're MID-reasoning — in
                    // that case the close_tag never arrived and emitting
                    // the partial think-block would leak it to the user.
                    if !pending.trim().is_empty() && !in_reasoning {
                        let chunk = make_chunk(&completion_id, created, &model_name, &pending);
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                    if streaming_backend_empty(any_content, &finish_reason) {
                        let err = json!({"error": {"type":"backend_returned_empty","message":"backend produced no content and no legitimate finish reason"}});
                        yield Ok(Event::default().data(err.to_string()));
                        yield Ok(Event::default().data("[DONE]"));
                        return;
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
                if let Some(fr) = finish_reason_of(&val) {
                    finish_reason = fr.to_string();
                }
                let Some(token) = val.get("choices").and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                else { continue };
                if token.is_empty() { continue; }
                any_content = true;

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
                        // close_tag may be split across tokens (`</`,`think`,`>`).
                        // Keep only the trailing close_tag.len()-1 bytes so a
                        // split tag matches on the next token; the rest is
                        // reasoning content we intentionally discard. The old
                        // `pending.clear()` here meant a split close tag NEVER
                        // matched → in_reasoning stuck → every post-</think>
                        // answer token dropped → empty completion to the client.
                        let keep = close_tag.len().saturating_sub(1);
                        if pending.len() > keep {
                            let mut cut = pending.len() - keep;
                            while cut < pending.len() && !pending.is_char_boundary(cut) {
                                cut += 1;
                            }
                            pending.drain(..cut);
                        }
                    }
                } else if reasoning_done {
                    let chunk = make_chunk(&completion_id, created, &model_name, token);
                    yield Ok(Event::default().data(chunk.to_string()));
                    pending.clear();
                }
            }
        }

        // Stream ended without an explicit [DONE] line (some backends
        // just close the connection). Flush the same way the [DONE]
        // branch does: emit pending if not still mid-reasoning, then
        // emit the synthetic close envelope so clients see a proper
        // finish_reason.
        if !pending.trim().is_empty() && !in_reasoning {
            let chunk = make_chunk(&completion_id, created, &model_name, &pending);
            yield Ok(Event::default().data(chunk.to_string()));
        }
        if streaming_backend_empty(any_content, &finish_reason) {
            let err = json!({"error": {"type":"backend_returned_empty","message":"backend produced no content and no legitimate finish reason"}});
            yield Ok(Event::default().data(err.to_string()));
            yield Ok(Event::default().data("[DONE]"));
            return;
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

/// Extract `choices[0].finish_reason` as a borrowed str when present and
/// non-null. All three streaming bridges read the same OpenAI-format
/// upstream (lamu's llama-server at :PORT/v1/chat/completions — the
/// Ollama/Anthropic shapes are client-facing only), so the extraction is
/// shared to keep the three gate paths from drifting.
fn finish_reason_of(val: &Value) -> Option<&str> {
    val.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str())
}

/// Empty-backend gate for the streaming paths. True when the backend
/// yielded no usable output (`any_output == false`) AND reported no
/// legitimate finish reason — i.e. it silently failed (crashed or
/// dropped the socket) rather than producing a legitimately-empty
/// completion. Mirrors the non-streaming 502 `backend_returned_empty`
/// gate's finish-reason allowlist so all four surfaces agree.
fn streaming_backend_empty(any_output: bool, finish_reason: &str) -> bool {
    !any_output
        && !matches!(finish_reason, "stop" | "length" | "tool_calls" | "content_filter")
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
pub fn build_state(registry_path: &Path, http_port: u16) -> anyhow::Result<AppState> {
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
        http_port,
        auth_token: Arc::new(crate::auth::resolve_token()),
    })
}

// ── Anthropic Messages API shim ─────────────────────────────────────
//
// Claude Code, Continue (anthropic mode), and other tools call this
// surface. We translate to our internal ChatRequest, reuse the OpenAI
// pipeline, then map the response back into Anthropic's envelope.
// Both non-streaming and streaming (SSE) are supported: when the request
// sets `stream: true`, `anthropic_messages` dispatches to
// `stream_response_anthropic`, which emits Anthropic-shaped SSE events.

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
    // `Option` (not plain-with-default) so the per-model sampling profile
    // merge can distinguish omitted-vs-default. Anthropic's spec makes
    // `max_tokens` required, but we keep it lenient + Option here and let
    // the merge apply the builtin default (16384) when absent.
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    /// Anthropic tool spec. Translated to OpenAI tools array.
    /// Each tool: {name, description, input_schema}.
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    /// Qwen3.6 / Qwen3.5 reasoning toggle. Not part of Anthropic's spec —
    /// extension that lamu honors so harnesses on this surface can opt
    /// out of the `<think>` block for latency-sensitive calls.
    #[serde(default)]
    enable_thinking: Option<bool>,
}

fn anthropic_tools_to_openai(tools: &[Value]) -> Vec<Value> {
    tools.iter().map(|t| {
        let name = t.get("name").cloned().unwrap_or_else(|| Value::String("tool".into()));
        let desc = t.get("description").cloned().unwrap_or_else(|| Value::String("".into()));
        let schema = t.get("input_schema").cloned()
            .unwrap_or_else(|| json!({"type":"object","properties":{}}));
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": schema,
            }
        })
    }).collect()
}

fn anthropic_tool_choice_to_openai(tc: &Value) -> Value {
    // Anthropic: {type:"auto"|"any"|"tool"|"none", name?:"..."}
    // OpenAI:   "auto"|"none"|"required" | {type:"function",function:{name:"..."}}
    match tc {
        Value::Object(o) => {
            let ty = o.get("type").and_then(|v| v.as_str()).unwrap_or("auto");
            match ty {
                "any" => Value::String("required".into()),
                "tool" => json!({
                    "type": "function",
                    "function": {"name": o.get("name").cloned().unwrap_or(Value::Null)},
                }),
                "none" => Value::String("none".into()),
                _ => Value::String("auto".into()),
            }
        }
        _ => Value::String("auto".into()),
    }
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

/// Expand an Anthropic message's content into one-or-more OpenAI-style
/// `Message`s. Handles text + tool_use + tool_result blocks so multi-turn
/// tool flows survive the bridge. Our `Message` is string-only; tool_use
/// is encoded as readable JSON for the model to parse from context.
fn anthropic_message_to_openai(
    role: &str,
    content: &Value,
    out: &mut Vec<Message>,
) {
    let blocks: Vec<Value> = match content {
        Value::String(s) => {
            out.push(Message {
                role: role.to_string(),
                content: s.clone(),
            });
            return;
        }
        Value::Array(arr) => arr.clone(),
        _ => return,
    };

    let mut text_buf = String::new();
    for b in &blocks {
        let ty = b.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match ty {
            "text" => {
                if let Some(t) = b.get("text").and_then(|v| v.as_str()) {
                    if !text_buf.is_empty() {
                        text_buf.push('\n');
                    }
                    text_buf.push_str(t);
                }
            }
            "tool_use" => {
                // Flush any pending text first under the message's role.
                if !text_buf.is_empty() {
                    out.push(Message {
                        role: role.to_string(),
                        content: std::mem::take(&mut text_buf),
                    });
                }
                let name = b.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                let input = b.get("input").cloned().unwrap_or(json!({}));
                let id = b.get("id").and_then(|v| v.as_str()).unwrap_or("");
                // Render the tool call as readable JSON. bee's llama-server
                // sees this as plain text — won't preserve tool_call_id
                // wiring, but conversation context survives.
                out.push(Message {
                    role: "assistant".to_string(),
                    content: format!(
                        "[Tool call id={id}] {}({})",
                        name,
                        serde_json::to_string(&input).unwrap_or_default()
                    ),
                });
            }
            "tool_result" => {
                if !text_buf.is_empty() {
                    out.push(Message {
                        role: role.to_string(),
                        content: std::mem::take(&mut text_buf),
                    });
                }
                let id = b.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("");
                let body = b.get("content").cloned().unwrap_or(Value::String("".into()));
                let body_str = match &body {
                    Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };
                out.push(Message {
                    // OpenAI uses role:"tool" for these. bee llama-server
                    // accepts it via its chat template. tool_call_id link
                    // is lost (we'd need a richer Message struct).
                    role: "tool".to_string(),
                    content: format!("[Tool result id={id}]\n{body_str}"),
                });
            }
            _ => {} // ignore image / unknown blocks
        }
    }
    if !text_buf.is_empty() {
        out.push(Message {
            role: role.to_string(),
            content: text_buf,
        });
    }
}

async fn anthropic_messages(
    State(state): State<AppState>,
    Json(req): Json<AnthropicRequest>,
) -> Response {
    // Translate to OpenAI-style messages. For multi-turn tool flows,
    // expand each Anthropic message's content blocks into one or more
    // OpenAI messages — preserves tool_use / tool_result semantics so
    // Claude Code's tool agent loop works.
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
        anthropic_message_to_openai(&m.role, &m.content, &mut messages);
    }

    // Translate tools/tool_choice up front so BOTH the streaming and
    // non-streaming paths forward them. Previously this ran AFTER the
    // stream branch returned, so streaming /v1/messages dropped tools
    // entirely — fatal for Claude Code, which streams by default.
    let oai_tools = req.tools.as_ref().map(|t| anthropic_tools_to_openai(t));
    let oai_tool_choice = req.tool_choice.as_ref().map(anthropic_tool_choice_to_openai);

    if req.stream {
        return stream_response_anthropic(state, messages, req.model.clone(),
                                         req.max_tokens, req.temperature,
                                         req.top_k, req.top_p,
                                         req.min_p, req.repeat_penalty,
                                         req.enable_thinking,
                                         oai_tools, oai_tool_choice).await;
    }

    let oai_req = ChatRequest {
        model: req.model.clone(),
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stream: false,
        top_k: req.top_k,
        top_p: req.top_p,
        min_p: req.min_p,
        repeat_penalty: req.repeat_penalty,
        enable_thinking: req.enable_thinking,
        tools: oai_tools,
        tool_choice: oai_tool_choice,
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

    // Build content blocks. Text block first (if any), then tool_use blocks
    // for any tool_calls the model emitted. Mirrors Anthropic's native
    // response shape so Claude Code can drive multi-turn tool flows.
    let mut content_blocks: Vec<Value> = Vec::new();
    if !oai_content.is_empty() {
        content_blocks.push(json!({"type":"text","text": oai_content}));
    }
    let mut had_tool_use = false;
    if let Some(tcs) = oai_msg.and_then(|m| m.get("tool_calls")).and_then(|v| v.as_array()) {
        for tc in tcs {
            let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let fname = tc.get("function").and_then(|f| f.get("name"))
                .and_then(|v| v.as_str()).unwrap_or("").to_string();
            let args_raw = tc.get("function").and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str()).unwrap_or("{}");
            let args_val: Value = serde_json::from_str(args_raw).unwrap_or(json!({}));
            content_blocks.push(json!({
                "type":"tool_use",
                "id": id,
                "name": fname,
                "input": args_val,
            }));
            had_tool_use = true;
        }
    }
    if content_blocks.is_empty() {
        // chat_completions already filters structurally-empty backend
        // responses to 502 (see backend_returned_empty gate), so reaching
        // this is unexpected. Surface a 502 here too instead of silently
        // returning an empty text block — clients can retry.
        return (StatusCode::BAD_GATEWAY, Json(json!({
            "type": "error",
            "error": {
                "type": "backend_returned_empty",
                "message": "backend produced neither text nor tool_use blocks",
            }
        }))).into_response();
    }
    let stop_reason = if had_tool_use { "tool_use" } else { "end_turn" };

    let anthro = json!({
        "id": format!("msg_{}", random_hex(12)),
        "type": "message",
        "role": "assistant",
        "model": oai_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": serde_json::Value::Null,
        "usage": {
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        },
    });
    (StatusCode::OK, Json(anthro)).into_response()
}

/// Accumulates one streamed OpenAI tool_call across SSE chunks. OpenAI
/// delivers a tool call incrementally: the first chunk carries `id` +
/// `function.name`, later chunks append `function.arguments` fragments.
#[derive(Default)]
struct ToolAcc {
    id: String,
    name: String,
    args: String,
}

// Anthropic streaming: bridge OpenAI SSE → Anthropic event-typed SSE.
// Re-uses chat_completions routing/lock acquisition by calling backend
// directly via the routing decision pulled from state.
#[allow(clippy::too_many_arguments)]
async fn stream_response_anthropic(
    state: AppState,
    messages: Vec<Message>,
    model_req: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    enable_thinking: Option<bool>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
) -> Response {
    let (port, model_name, _marker, sampling) = match resolve_and_ensure_loaded(
        &state,
        model_req.as_deref(),
    ).await {
        Ok(t) => t,
        Err(resp) => return resp,
    };

    // Merge the per-model sampling profile with the request values.
    let s = lamu_core::types::resolve_samplers(
        sampling.as_ref(), temperature, top_p, top_k, min_p, repeat_penalty, max_tokens,
    );
    let eff_temperature = s.temperature.unwrap_or_else(default_temperature);
    let eff_max_tokens = s.max_tokens.unwrap_or_else(default_max_tokens);

    let backend_url = format!("http://localhost:{}/v1/chat/completions", port);
    let mut payload = json!({
        "messages": messages,
        "max_tokens": eff_max_tokens,
        "temperature": eff_temperature,
        "stream": true,
    });
    // Only emit samplers when actually set (no nulls — fixes the prior
    // quirk where an omitted top_k/top_p serialized to JSON `null`).
    if let Some(k) = s.top_k { payload["top_k"] = json!(k); }
    if let Some(p) = s.top_p { payload["top_p"] = json!(p); }
    if let Some(v) = s.min_p { payload["min_p"] = json!(v); }
    if let Some(v) = s.repeat_penalty { payload["repeat_penalty"] = json!(v); }
    if let Some(et) = enable_thinking {
        payload["chat_template_kwargs"] = json!({ "enable_thinking": et });
    }
    // Forward tool schemas + choice so the backend can emit tool calls
    // on the streaming path (see anthropic_messages for the why).
    if let Some(t) = &tools {
        payload["tools"] = json!(t);
    }
    if let Some(tc) = &tool_choice {
        payload["tool_choice"] = tc.clone();
    }

    let msg_id = format!("msg_{}", random_hex(12));
    let client = state.client.clone();

    let s = async_stream::stream! {
        // message_start
        let start = json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [],
                "stop_reason": serde_json::Value::Null,
                "stop_sequence": serde_json::Value::Null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        });
        yield Ok::<_, Infallible>(Event::default().event("message_start").data(start.to_string()));

        // content_block_start (single text block)
        let cb_start = json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        });
        yield Ok(Event::default().event("content_block_start").data(cb_start.to_string()));

        let resp = match client.post(&backend_url).json(&payload).send().await {
            Ok(r) => r,
            Err(e) => {
                let err = json!({
                    "type": "error",
                    "error": {"type":"backend_error","message": format!("{e}")}
                });
                yield Ok(Event::default().event("error").data(err.to_string()));
                return;
            }
        };

        let mut byte_stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut out_tokens: u64 = 0;
        // Tool calls accumulated by their OpenAI delta index. BTreeMap so
        // the emitted tool_use blocks keep the backend's call order.
        let mut tool_acc: std::collections::BTreeMap<usize, ToolAcc> = std::collections::BTreeMap::new();
        // Last finish_reason seen in the stream — drives the empty-backend
        // gate at close. Empty = stream closed without one.
        let mut finish_reason = String::new();

        use futures_util::stream::StreamExt;
        'read: while let Some(chunk_res) = byte_stream.next().await {
            let Ok(bytes) = chunk_res else { break 'read };
            // Byte-buffer, decode whole lines only: from_utf8_lossy on a raw
            // chunk corrupts a multibyte char split across chunk boundaries.
            buf.extend_from_slice(&bytes);
            while let Some(line) = lamu_core::sse::next_sse_line(&mut buf) {
                let line = line.trim();
                let Some(rest) = line.strip_prefix("data: ") else { continue };
                if rest == "[DONE]" { break 'read; }
                let Ok(val) = serde_json::from_str::<Value>(rest) else { continue };
                let choice = val.get("choices").and_then(|c| c.get(0));
                if let Some(fr) = finish_reason_of(&val) {
                    finish_reason = fr.to_string();
                }
                let delta = choice.and_then(|c| c.get("delta"));

                // Text token → text_delta on the index-0 text block.
                if let Some(token) = delta.and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                    if !token.is_empty() {
                        out_tokens += 1;
                        let d = json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type":"text_delta","text": token}
                        });
                        yield Ok(Event::default().event("content_block_delta").data(d.to_string()));
                    }
                }

                // Tool-call deltas → accumulate by index; emitted as Anthropic
                // tool_use blocks at close once fully assembled.
                if let Some(tcs) = delta.and_then(|d| d.get("tool_calls")).and_then(|v| v.as_array()) {
                    for tc in tcs {
                        let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let slot = tool_acc.entry(idx).or_default();
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                            if !id.is_empty() { slot.id = id.to_string(); }
                        }
                        if let Some(name) = tc.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()) {
                            if !name.is_empty() { slot.name = name.to_string(); }
                        }
                        if let Some(a) = tc.get("function").and_then(|f| f.get("arguments")).and_then(|v| v.as_str()) {
                            slot.args.push_str(a);
                        }
                    }
                }
            }
        }

        // ── Close ────────────────────────────────────────────────────
        // Empty-backend gate (mirrors the non-streaming 502): zero text,
        // zero tool calls, and no legitimate finish reason ⇒ the backend
        // silently failed. Emit an Anthropic `error` event instead of
        // reporting a clean (but empty) completion.
        if streaming_backend_empty(out_tokens > 0 || !tool_acc.is_empty(), &finish_reason) {
            // Close the index-0 text block (opened pre-loop) before the
            // error so block-lifecycle-tracking clients see valid SSE.
            let cb_stop = json!({"type":"content_block_stop","index":0});
            yield Ok(Event::default().event("content_block_stop").data(cb_stop.to_string()));
            let err = json!({
                "type": "error",
                "error": {
                    "type": "backend_returned_empty",
                    "message": "backend produced no content and no legitimate finish reason"
                }
            });
            yield Ok(Event::default().event("error").data(err.to_string()));
            let m_stop = json!({"type":"message_stop"});
            yield Ok(Event::default().event("message_stop").data(m_stop.to_string()));
            return;
        }

        // Close the index-0 text block.
        let cb_stop = json!({"type":"content_block_stop","index":0});
        yield Ok(Event::default().event("content_block_stop").data(cb_stop.to_string()));

        // Emit one tool_use content block per accumulated call (start +
        // a single input_json_delta carrying the full args + stop).
        // stop_reason tracks blocks ACTUALLY emitted, so a malformed call
        // (empty name) that we skip doesn't yield a phantom tool_use verdict.
        let mut emitted_tools = false;
        let mut next_index = 1;
        for (_oai_idx, acc) in tool_acc {
            if acc.name.is_empty() {
                tracing::warn!("anthropic stream: dropping tool_call with empty function name");
                continue;
            }
            // Anthropic requires a non-empty tool_use id; synthesize one if
            // the backend never sent it (some OpenAI-compat servers omit it).
            let id = if acc.id.is_empty() {
                format!("toolu_{}", random_hex(12))
            } else {
                acc.id.clone()
            };
            let cbs = json!({
                "type": "content_block_start",
                "index": next_index,
                "content_block": {"type":"tool_use","id": id, "name": acc.name, "input": {}}
            });
            yield Ok(Event::default().event("content_block_start").data(cbs.to_string()));
            // Forward the assembled args as a single input_json_delta. Empty
            // → "{}" (valid no-arg call). Non-empty but unparseable → forward
            // RAW (don't substitute "{}", which would silently erase a
            // truncated-but-meaningful call) and warn so operators can spot
            // a misbehaving backend.
            let partial_json = if acc.args.trim().is_empty() {
                "{}".to_string()
            } else {
                if serde_json::from_str::<Value>(&acc.args).is_err() {
                    tracing::warn!(args = %acc.args, "anthropic stream: tool_call args are not valid JSON; forwarding raw");
                }
                acc.args.clone()
            };
            let cbd = json!({
                "type": "content_block_delta",
                "index": next_index,
                "delta": {"type":"input_json_delta","partial_json": partial_json}
            });
            yield Ok(Event::default().event("content_block_delta").data(cbd.to_string()));
            let cbstop = json!({"type":"content_block_stop","index": next_index});
            yield Ok(Event::default().event("content_block_stop").data(cbstop.to_string()));
            next_index += 1;
            emitted_tools = true;
        }

        let stop_reason = if emitted_tools { "tool_use" } else { "end_turn" };
        let m_delta = json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": serde_json::Value::Null},
            "usage": {"output_tokens": out_tokens}
        });
        yield Ok(Event::default().event("message_delta").data(m_delta.to_string()));
        let m_stop = json!({"type":"message_stop"});
        yield Ok(Event::default().event("message_stop").data(m_stop.to_string()));
    };

    Sse::new(Box::pin(s)).into_response()
}

// ── Ollama-compat shim ───────────────────────────────────────────────
//
// Supports /api/tags (list models) and /api/chat (chat with stream).
// AnythingLLM, Open WebUI in Ollama mode, and a few other tools
// expect this surface. Body format is line-delimited JSON, not SSE.

#[derive(Debug, Clone, Deserialize)]
struct OllamaChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<OllamaMessage>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    options: Option<OllamaOptions>,
    /// Qwen3.6 / Qwen3.5 reasoning toggle. Not in Ollama's spec —
    /// extension that lamu honors so harnesses on this surface can opt
    /// out of the `<think>` block. Accepted at top level for symmetry
    /// with /v1/chat/completions.
    #[serde(default)]
    enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct OllamaOptions {
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    num_predict: Option<u32>,
}

async fn ollama_tags(State(state): State<AppState>) -> impl IntoResponse {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let models: Vec<Value> = state.entries.iter().map(|(name, e)| {
        json!({
            "name": name,
            "model": name,
            "modified_at": now,
            "size": (e.vram_mb as u64) * 1024 * 1024,
            "details": {
                "family": e.arch,
                "parameter_size": format!("{}B", e.params_b),
                "quantization_level": e.quant,
            }
        })
    }).collect();
    (StatusCode::OK, Json(json!({"models": models}))).into_response()
}

async fn ollama_chat(
    State(state): State<AppState>,
    Json(req): Json<OllamaChatRequest>,
) -> Response {
    let stream_on = req.stream.unwrap_or(true);
    let opts = req.options.unwrap_or_default();
    // Keep as Option (don't collapse to defaults here) so the per-model
    // sampling profile merge downstream can distinguish omitted-vs-set.
    // The builtin default is applied as the final merge fallback in
    // chat_completions / stream_response_ollama.
    let max_tokens = opts.num_predict;
    let temperature = opts.temperature;

    let messages: Vec<Message> = req.messages.iter().map(|m| Message {
        role: m.role.clone(),
        content: m.content.clone(),
    }).collect();

    if stream_on {
        return stream_response_ollama(state, messages, req.model.clone(),
                                      max_tokens, temperature,
                                      opts.top_k, opts.top_p,
                                      opts.min_p, opts.repeat_penalty,
                                      req.enable_thinking).await;
    }

    let oai_req = ChatRequest {
        model: req.model.clone(),
        messages,
        max_tokens,
        temperature,
        stream: false,
        top_k: opts.top_k,
        top_p: opts.top_p,
        min_p: opts.min_p,
        repeat_penalty: opts.repeat_penalty,
        enable_thinking: req.enable_thinking,
        tools: None,
        tool_choice: None,
    };
    let resp = chat_completions(State(state), Json(oai_req)).await.into_response();
    let (parts, body) = resp.into_parts();
    if parts.status != StatusCode::OK {
        return (parts.status, body).into_response();
    }
    let body_bytes = match axum::body::to_bytes(body, 4 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("body read: {e}")}))).into_response();
        }
    };
    let oai: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return (StatusCode::BAD_GATEWAY,
                Json(json!({"error": format!("json: {e}")}))).into_response();
        }
    };
    let content = oai.get("choices").and_then(|c| c.get(0))
        .and_then(|c| c.get("message")).and_then(|m| m.get("content"))
        .and_then(|v| v.as_str()).unwrap_or("");
    let model = oai.get("model").and_then(|v| v.as_str()).unwrap_or("lamu");
    let usage = oai.get("usage");
    let in_tok = usage.and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);
    let out_tok = usage.and_then(|u| u.get("completion_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);

    let ollama = json!({
        "model": model,
        "created_at": rfc3339_now(),
        "message": {"role":"assistant","content": content},
        "done_reason": "stop",
        "done": true,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": in_tok,
        "eval_count": out_tok,
        "eval_duration": 0,
    });
    (StatusCode::OK, Json(ollama)).into_response()
}

fn rfc3339_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // Coarse-grained RFC3339; nanos zeroed. Good enough for clients
    // that only check the field exists.
    let (y, mo, d, h, mi, s) = epoch_to_civil(secs);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, mi, s)
}

fn epoch_to_civil(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    // Howard Hinnant's algorithm
    let z = (secs / 86400) as i64;
    let s = secs % 86400;
    let h = (s / 3600) as u32;
    let mi = ((s % 3600) / 60) as u32;
    let sc = (s % 60) as u32;
    let z2 = z + 719468;
    let era = if z2 >= 0 { z2 } else { z2 - 146096 } / 146097;
    let doe = (z2 - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let mo = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let yr = if mo <= 2 { y + 1 } else { y } as u32;
    (yr, mo, d, h, mi, sc)
}

#[allow(clippy::too_many_arguments)]
async fn stream_response_ollama(
    state: AppState,
    messages: Vec<Message>,
    model_req: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    enable_thinking: Option<bool>,
) -> Response {
    let (port, model_name, _marker, sampling) = match resolve_and_ensure_loaded(
        &state,
        model_req.as_deref(),
    ).await {
        Ok(t) => t,
        Err(resp) => return resp,
    };

    // Merge the per-model sampling profile with the request values.
    let s = lamu_core::types::resolve_samplers(
        sampling.as_ref(), temperature, top_p, top_k, min_p, repeat_penalty, max_tokens,
    );
    let eff_temperature = s.temperature.unwrap_or_else(default_temperature);
    let eff_max_tokens = s.max_tokens.unwrap_or_else(default_max_tokens);

    let backend_url = format!("http://localhost:{}/v1/chat/completions", port);
    let mut payload = json!({
        "messages": messages,
        "max_tokens": eff_max_tokens,
        "temperature": eff_temperature,
        "stream": true,
    });
    if let Some(k) = s.top_k { payload["top_k"] = json!(k); }
    if let Some(p) = s.top_p { payload["top_p"] = json!(p); }
    if let Some(v) = s.min_p { payload["min_p"] = json!(v); }
    if let Some(v) = s.repeat_penalty { payload["repeat_penalty"] = json!(v); }
    if let Some(et) = enable_thinking {
        payload["chat_template_kwargs"] = json!({ "enable_thinking": et });
    }

    let client = state.client.clone();
    let body_stream = async_stream::stream! {
        let resp = match client.post(&backend_url).json(&payload).send().await {
            Ok(r) => r,
            Err(e) => {
                let err = json!({"error": format!("backend: {e}")});
                yield Ok::<_, std::io::Error>(format!("{}\n", err).into_bytes());
                return;
            }
        };
        let mut byte_stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut out_tokens: u64 = 0;
        // Empty-backend gate state (mirrors the non-streaming 502).
        let mut finish_reason = String::new();

        use futures_util::stream::StreamExt;
        'read: while let Some(chunk_res) = byte_stream.next().await {
            let Ok(bytes) = chunk_res else { break 'read };
            // Byte-buffer, decode whole lines only: from_utf8_lossy on a raw
            // chunk corrupts a multibyte char split across chunk boundaries.
            buf.extend_from_slice(&bytes);
            while let Some(line) = lamu_core::sse::next_sse_line(&mut buf) {
                let line = line.trim();
                let Some(rest) = line.strip_prefix("data: ") else { continue };
                if rest == "[DONE]" { break 'read; }
                let Ok(val) = serde_json::from_str::<Value>(rest) else { continue };
                if let Some(fr) = finish_reason_of(&val) {
                    finish_reason = fr.to_string();
                }
                let Some(token) = val.get("choices").and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                else { continue };
                if token.is_empty() { continue; }
                out_tokens += 1;
                let chunk = json!({
                    "model": model_name,
                    "created_at": rfc3339_now(),
                    "message": {"role":"assistant","content": token},
                    "done": false,
                });
                yield Ok(format!("{}\n", chunk).into_bytes());
            }
        }

        // Single close path for both [DONE] and bare socket close (the
        // latter previously emitted NOTHING — clients hung waiting for a
        // done marker). Empty-backend gate: zero tokens + no legitimate
        // finish reason ⇒ emit an error line instead of a clean done.
        if streaming_backend_empty(out_tokens > 0, &finish_reason) {
            let err = json!({
                "model": model_name,
                "created_at": rfc3339_now(),
                "error": "backend_returned_empty: backend produced no content and no legitimate finish reason",
                "done": true,
            });
            yield Ok(format!("{}\n", err).into_bytes());
            return;
        }
        let final_obj = json!({
            "model": model_name,
            "created_at": rfc3339_now(),
            "message": {"role":"assistant","content":""},
            "done_reason":"stop",
            "done": true,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "eval_count": out_tokens,
            "eval_duration": 0,
        });
        yield Ok(format!("{}\n", final_obj).into_bytes());
    };

    let stream_body = axum::body::Body::from_stream(body_stream);
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/x-ndjson")
        .body(stream_body)
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "stream build failed").into_response())
}

#[cfg(test)]
mod compat_tests {
    use super::*;
    use proptest::prelude::*;

    // ── enable_thinking serde plumbing ──────────────────────────────

    #[test]
    fn anthropic_request_accepts_enable_thinking_true() {
        let body = r#"{
            "model":"lamu","max_tokens":10,
            "messages":[{"role":"user","content":"hi"}],
            "enable_thinking": true
        }"#;
        let req: AnthropicRequest = serde_json::from_str(body).expect("parse");
        assert_eq!(req.enable_thinking, Some(true));
    }

    #[test]
    fn anthropic_request_accepts_enable_thinking_false() {
        let body = r#"{
            "model":"lamu","max_tokens":10,
            "messages":[{"role":"user","content":"hi"}],
            "enable_thinking": false
        }"#;
        let req: AnthropicRequest = serde_json::from_str(body).expect("parse");
        assert_eq!(req.enable_thinking, Some(false));
    }

    #[test]
    fn anthropic_request_defaults_enable_thinking_to_none() {
        let body = r#"{
            "model":"lamu","max_tokens":10,
            "messages":[{"role":"user","content":"hi"}]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(body).expect("parse");
        assert_eq!(req.enable_thinking, None);
    }

    #[test]
    fn ollama_chat_request_accepts_enable_thinking() {
        let body = r#"{
            "model":"lamu","stream":false,
            "messages":[{"role":"user","content":"hi"}],
            "enable_thinking": false
        }"#;
        let req: OllamaChatRequest = serde_json::from_str(body).expect("parse");
        assert_eq!(req.enable_thinking, Some(false));
    }

    // ── Multipart content lenient deserializer ──────────────────────

    #[test]
    fn message_accepts_plain_string_content() {
        let raw = r#"{"role":"user","content":"hello"}"#;
        let m: Message = serde_json::from_str(raw).unwrap();
        assert_eq!(m.role, "user");
        assert_eq!(m.content, "hello");
    }

    #[test]
    fn message_accepts_openai_vision_array_content() {
        // pi (Earendil) sends content as an array of parts per the
        // OpenAI Vision spec. Lamu flattens text parts; non-text parts
        // (image_url, etc) drop silently.
        let raw = r#"{
            "role":"user",
            "content":[
                {"type":"text","text":"part one"},
                {"type":"image_url","image_url":{"url":"x"}},
                {"type":"text","text":"part two"}
            ]
        }"#;
        let m: Message = serde_json::from_str(raw).unwrap();
        assert_eq!(m.role, "user");
        assert_eq!(m.content, "part one\npart two");
    }

    #[test]
    fn message_accepts_empty_content_array() {
        let raw = r#"{"role":"user","content":[]}"#;
        let m: Message = serde_json::from_str(raw).unwrap();
        assert_eq!(m.content, "");
    }

    #[test]
    fn message_accepts_null_content() {
        // Some OpenAI tool-call responses set content: null when the
        // assistant only emits tool_calls. Treat as empty string.
        let raw = r#"{"role":"assistant","content":null}"#;
        let m: Message = serde_json::from_str(raw).unwrap();
        assert_eq!(m.content, "");
    }

    // ── stream_response reasoning-tag scanner flush ─────────────────
    //
    // Regression for the bug pi hit: short outputs (`ok\n`) buffered in
    // `pending` waiting for either a `<think>` open_tag or enough chars
    // to trip the reasoning_done threshold. End-of-stream then dropped
    // the buffer because the [DONE] flush required `reasoning_done`.
    // The fix flushes pending whenever we're NOT mid-reasoning at EOS.
    //
    // This test mirrors the production gate logic on a synthetic state.
    // We can't easily drive the full stream_response async_stream from
    // a unit test (axum SSE machinery), so check the predicate the
    // production code uses: "flush iff content present AND not still
    // mid-reasoning". Plus a sanity test that the threshold-based
    // mid-stream flush still fires for long outputs without tags.

    fn should_flush_at_eos(pending: &str, in_reasoning: bool) -> bool {
        !pending.trim().is_empty() && !in_reasoning
    }

    #[test]
    fn eos_flush_short_response_without_tags() {
        // 27B with enable_thinking=false: model emits "ok" with no
        // <think> tag and finishes in <24 chars. Must flush at EOS.
        assert!(should_flush_at_eos("ok", false));
    }

    #[test]
    fn eos_flush_drops_partial_think_block() {
        // Model started thinking but stream cut off before </think>.
        // Must NOT leak partial reasoning to the client.
        assert!(!should_flush_at_eos("I should consider...", true));
    }

    #[test]
    fn eos_flush_empty_pending_no_op() {
        // Stream already drained mid-flight (long enough output that the
        // threshold path flushed). Nothing left to do at EOS.
        assert!(!should_flush_at_eos("", false));
        assert!(!should_flush_at_eos("   ", false));  // whitespace-only
    }

    #[test]
    fn eos_flush_drops_partial_when_mid_reasoning_even_if_pending_present() {
        // Both conditions: pending has content AND mid-reasoning. The
        // content is reasoning that wasn't terminated; drop it.
        assert!(!should_flush_at_eos("partial thought", true));
    }

    // Threshold-flush predicate: production code declares
    // reasoning_done = true once pending.len() > open_tag.len() * 3
    // without spotting `<think>`. Mirrors that gate so a future edit
    // shifting the multiplier gets caught.
    fn passes_threshold_flush(pending: &str, open_tag: &str) -> bool {
        pending.len() > open_tag.len() * 3
    }

    #[test]
    fn threshold_flush_fires_on_long_output_without_tags() {
        let open_tag = "<think>";  // 7 chars; threshold = 21
        // 22 chars of plain output without a `<think>` tag — must trip.
        assert!(passes_threshold_flush("0123456789012345678901", open_tag));
    }

    #[test]
    fn threshold_flush_holds_for_short_output() {
        // Short outputs must NOT trip the threshold prematurely —
        // could mistake real `<think>` prefix for non-reasoning.
        let open_tag = "<think>";  // 7 chars; threshold = 21
        assert!(!passes_threshold_flush("ok", open_tag));
        assert!(!passes_threshold_flush("12345678901234567890", open_tag)); // 20 chars, just under
    }

    #[test]
    fn ollama_chat_request_defaults_enable_thinking() {
        let body = r#"{
            "model":"lamu","stream":false,
            "messages":[{"role":"user","content":"hi"}]
        }"#;
        let req: OllamaChatRequest = serde_json::from_str(body).expect("parse");
        assert_eq!(req.enable_thinking, None);
    }

    // ── per-model sampling: request serde (Option-ization) ──────────

    #[test]
    fn chat_request_omitted_samplers_are_none() {
        // temperature/max_tokens were Option-ized (were plain-with-default)
        // so omission is detectable → the per-model profile can fill, and
        // the builtin 0.7/16384 fallback is applied at the payload site.
        // A regression that reverts them to plain-with-default would break
        // profile fill silently; this pins None-on-omission.
        let req: ChatRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        ).unwrap();
        assert_eq!(req.temperature, None);
        assert_eq!(req.max_tokens, None);
        assert_eq!(req.top_k, None);
        assert_eq!(req.top_p, None);
        assert_eq!(req.min_p, None);
        assert_eq!(req.repeat_penalty, None);
    }

    #[test]
    fn chat_request_parses_full_sampler_set() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],
                "temperature":0.3,"max_tokens":256,"top_k":40,
                "top_p":0.9,"min_p":0.05,"repeat_penalty":1.1}"#,
        ).unwrap();
        assert_eq!(req.temperature, Some(0.3));
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.min_p, Some(0.05));
        assert_eq!(req.repeat_penalty, Some(1.1));
    }

    // ── streaming empty-backend gate ────────────────────────────────

    #[test]
    fn streaming_gate_fires_only_on_silent_failure() {
        // Silent failure: no output AND no legitimate finish reason.
        assert!(streaming_backend_empty(false, ""));
        assert!(streaming_backend_empty(false, "error"));
        assert!(streaming_backend_empty(false, "null"));
        // Legitimately-empty completion (model chose to stop) → NOT a gate.
        assert!(!streaming_backend_empty(false, "stop"));
        assert!(!streaming_backend_empty(false, "length"));
        assert!(!streaming_backend_empty(false, "tool_calls"));
        assert!(!streaming_backend_empty(false, "content_filter"));
        // Any output → never a gate, regardless of finish reason.
        assert!(!streaming_backend_empty(true, ""));
        assert!(!streaming_backend_empty(true, "stop"));
    }

    // ── Anthropic tool translator ───────────────────────────────────

    #[test]
    fn anthropic_tools_translator_preserves_name_and_schema() {
        let anthro_tools = vec![serde_json::json!({
            "name": "get_weather",
            "description": "Return the weather for a location.",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        })];
        let oai = anthropic_tools_to_openai(&anthro_tools);
        assert_eq!(oai.len(), 1);
        let t = &oai[0];
        assert_eq!(t["type"], "function");
        assert_eq!(t["function"]["name"], "get_weather");
        assert_eq!(t["function"]["parameters"]["properties"]["location"]["type"], "string");
    }

    #[test]
    fn anthropic_tool_choice_translator_handles_known_shapes() {
        // Exact OAI values per anthropic_tool_choice_to_openai contract:
        //   {type:"auto"} → "auto"
        //   {type:"any"}  → "required" (forces tool use)
        //   {type:"none"} → "none"
        //   {type:"tool",name:N} → {type:"function",function:{name:N}}
        // Tight equality so a translator regression that silently
        // collapses "any" to "auto" (or any other semantic drift)
        // fails this test immediately.
        let auto = anthropic_tool_choice_to_openai(&serde_json::json!({"type": "auto"}));
        assert_eq!(auto, serde_json::json!("auto"));

        let any = anthropic_tool_choice_to_openai(&serde_json::json!({"type": "any"}));
        assert_eq!(any, serde_json::json!("required"),
            "any must map to forced 'required', NOT 'auto' (semantic mismatch)");

        let none = anthropic_tool_choice_to_openai(&serde_json::json!({"type": "none"}));
        assert_eq!(none, serde_json::json!("none"));

        let named = anthropic_tool_choice_to_openai(&serde_json::json!({
            "type": "tool", "name": "get_weather"
        }));
        assert_eq!(named, serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather"}
        }));

        // Non-object input falls through to "auto".
        let str_only = anthropic_tool_choice_to_openai(&serde_json::json!("anything"));
        assert_eq!(str_only, serde_json::json!("auto"));
    }

    proptest! {
        // For any reasonable Anthropic tool spec, translating to OpenAI
        // preserves every field downstream OAI consumers depend on:
        // name, description, and schema properties. Translator is one-way
        // (Anthropic → OAI), so we check field-by-field equality, not
        // round-trip equivalence.
        #[test]
        fn anthropic_tool_translation_preserves_all_fields(
            name in "[a-zA-Z][a-zA-Z0-9_]{0,30}",
            desc in ".{0,80}",
            field in "[a-z][a-z0-9_]{0,10}",
        ) {
            let anthro = vec![serde_json::json!({
                "name": name.clone(),
                "description": desc.clone(),
                "input_schema": {
                    "type": "object",
                    "properties": { &field: { "type": "string" } },
                    "required": [&field]
                }
            })];
            let oai = anthropic_tools_to_openai(&anthro);
            prop_assert_eq!(oai.len(), 1);
            prop_assert_eq!(&oai[0]["type"], &serde_json::Value::String("function".to_string()));
            prop_assert_eq!(&oai[0]["function"]["name"], &serde_json::Value::String(name.clone()));
            prop_assert_eq!(&oai[0]["function"]["description"], &serde_json::Value::String(desc.clone()));
            prop_assert_eq!(
                &oai[0]["function"]["parameters"]["properties"][&field]["type"],
                &serde_json::Value::String("string".to_string())
            );
            prop_assert_eq!(
                &oai[0]["function"]["parameters"]["required"][0],
                &serde_json::Value::String(field.clone())
            );
        }
    }

    // ── 502 backend_returned_empty gate ─────────────────────────────
    //
    // The gate fires when content + reasoning_content + tool_calls are
    // all empty AND finish_reason isn't a recognized terminator.
    //
    // KEEP IN SYNC WITH `chat_completions` in this file (search for
    // "backend_returned_empty"). The helper below is a verbatim mirror
    // of the production predicate; changing one without the other
    // leaves the test passing while the real gate diverges. Until the
    // predicate gets extracted into a shared `pub(crate)` function,
    // the discipline is: every edit to the production expression
    // must edit this mirror too.

    fn is_empty_backend_response(data: &Value) -> bool {
        let msg = data.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message"));
        let raw_content = msg.and_then(|m| m.get("content")).and_then(|v| v.as_str()).unwrap_or("");
        let reasoning = msg.and_then(|m| m.get("reasoning_content")).and_then(|v| v.as_str()).unwrap_or("");
        let has_tool_calls = msg.and_then(|m| m.get("tool_calls"))
            .and_then(|v| v.as_array()).is_some_and(|a| !a.is_empty());
        let raw_finish = data.get("choices").and_then(|c| c.get(0))
            .and_then(|c| c.get("finish_reason")).and_then(|v| v.as_str()).unwrap_or("");
        msg.is_none()
            || (raw_content.is_empty() && reasoning.is_empty() && !has_tool_calls
                && !matches!(raw_finish, "stop" | "length" | "tool_calls" | "content_filter"))
    }

    #[test]
    fn gate_fires_when_choices_missing() {
        assert!(is_empty_backend_response(&serde_json::json!({})));
    }

    #[test]
    fn gate_fires_when_message_empty_and_no_finish_reason() {
        assert!(is_empty_backend_response(&serde_json::json!({
            "choices": [{"message": {}}]
        })));
    }

    #[test]
    fn gate_passes_legitimate_stop() {
        assert!(!is_empty_backend_response(&serde_json::json!({
            "choices": [{
                "message": {"content": "hello"},
                "finish_reason": "stop"
            }]
        })));
    }

    #[test]
    fn gate_passes_empty_content_with_valid_finish_reason() {
        // Some backends legitimately produce empty content with a recognized
        // finish_reason (e.g. content_filter). Gate must NOT fire.
        for fr in ["stop", "length", "tool_calls", "content_filter"] {
            let v = serde_json::json!({
                "choices": [{"message": {"content": ""}, "finish_reason": fr}]
            });
            assert!(!is_empty_backend_response(&v),
                "gate must not fire on legitimate finish_reason '{fr}': {v}");
        }
    }

    #[test]
    fn gate_passes_tool_calls_only() {
        // Tool-only completion: content empty, tool_calls non-empty.
        assert!(!is_empty_backend_response(&serde_json::json!({
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls"
            }]
        })));
    }

    #[test]
    fn gate_passes_reasoning_only() {
        // Some Qwen3.6 paths return content="" but populate reasoning_content.
        assert!(!is_empty_backend_response(&serde_json::json!({
            "choices": [{
                "message": {"content": "", "reasoning_content": "thinking..."},
                "finish_reason": "stop"
            }]
        })));
    }

    #[test]
    fn gate_fires_on_unknown_finish_reason_with_empty_content() {
        assert!(is_empty_backend_response(&serde_json::json!({
            "choices": [{"message": {"content": ""}, "finish_reason": "weird_new_reason"}]
        })));
    }
}
