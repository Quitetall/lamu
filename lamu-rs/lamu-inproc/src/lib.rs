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
//! Served surface (v1, embeddings-first):
//! - `GET /health` AND `GET /v1/health` → `{"status":"ok"}`. Both paths
//!   carry the same body because the two health conventions in the tree
//!   differ: llama.cpp probes `/health` and requires `status == "ok"`
//!   (lamu-core llamacpp.rs `is_healthy`), while fish-speech polls
//!   `/v1/health` for any 2xx. Serving both keeps any existing prober
//!   happy.
//! - `POST /v1/embeddings` → OpenAI embeddings shape.
//! - `POST /tokenize` → 404 with a JSON error. v1 stub: in-process
//!   engines don't expose llama-server's tokenize surface yet, and the
//!   only caller (`tokenize_count`, ADR 0021 context occupancy) already
//!   treats "unsupported" as a clean error. The route exists (instead of
//!   the implicit fallback) purely so the 404 body says WHY.
//!
//! `ChatEngine` is deliberately NOT in scope here — W5 (`lamu-hf`, ADR
//! 0035) adds the chat/completions surface when the first in-process
//! text-generation engine lands. Keep this crate embeddings-only until
//! then.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
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
