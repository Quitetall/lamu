//! HTTP route tests for lamu-api. Uses tower::ServiceExt::oneshot to drive
//! axum without binding a real port.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use lamu_api::metrics::LamuMetrics;
use lamu_api::openai_compat::{build_app, AppState, AuthMode};
use lamu_core::health::HealthRegistry;
use lamu_core::router::Router;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{
    BackendType, Capability, ModelEntry, ModelFormat,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tower::util::ServiceExt;

fn sample_entry(name: &str) -> ModelEntry {
    ModelEntry {
        name: name.to_string(),
        path: PathBuf::from(format!("/tmp/{name}.gguf")),
        format: ModelFormat::Gguf,
        backend: BackendType::LlamaCpp,
        arch: "qwen35".into(),
        params_b: 27.0,
        quant: "Q5_K_M".into(),
        vram_mb: 18000,
        context_max: 131072,
        capabilities: vec![Capability::Chat, Capability::Code],
        reasoning_marker: None,
        speculative: None,
        sampling: None,
        pinned: false,
        main: false,
        notes: String::new(),
        status: lamu_core::types::ModelStatus::default(),
        modality: lamu_core::types::Modality::Llm,
    }
}

fn make_state() -> AppState {
    let entries = vec![sample_entry("qwen35-27b"), sample_entry("qwen35-0.8b")];
    let entries_map: HashMap<String, ModelEntry> = entries.iter()
        .map(|e| (e.name.clone(), e.clone()))
        .collect();
    let scheduler = VramScheduler::new();
    let router = Router::new(&scheduler, entries.clone());
    let client = reqwest::Client::new();
    AppState {
        scheduler: Arc::new(Mutex::new(scheduler)),
        router: Arc::new(Mutex::new(router)),
        entries: Arc::new(entries_map),
        client,
        health: Arc::new(Mutex::new(HealthRegistry::new())),
        metrics: Arc::new(LamuMetrics::new().unwrap()),
        http_port: 8020,
        auth: Arc::new(AuthMode::Off), // auth off by default in route tests
        quota: Arc::new(lamu_api::quota::QuotaManager::new()),
    }
}

#[tokio::test]
async fn health_returns_ok() {
    let app = build_app(make_state());
    let resp = app.oneshot(
        Request::builder().uri("/health").body(Body::empty()).unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn list_models_returns_registered_models() {
    let app = build_app(make_state());
    let resp = app.oneshot(
        Request::builder().uri("/v1/models").body(Body::empty()).unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["object"], "list");
    let names: Vec<_> = body["data"].as_array().unwrap().iter()
        .map(|m| m["id"].as_str().unwrap().to_string()).collect();
    assert!(names.contains(&"qwen35-27b".into()));
    assert!(names.contains(&"qwen35-0.8b".into()));
}

#[tokio::test]
async fn chat_completions_503_when_no_loaded_model() {
    let app = build_app(make_state());
    let req_body = r#"{"model":"qwen35-27b","messages":[{"role":"user","content":"hi"}]}"#;
    let resp = app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(req_body.to_string()))
            .unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn metrics_endpoint_serves_prometheus_text() {
    let app = build_app(make_state());
    let resp = app.oneshot(
        Request::builder().uri("/metrics").body(Body::empty()).unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap().to_string();
    assert!(ct.starts_with("text/plain"), "ct = {ct}");
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    // Series that always render (Gauge + non-vec Counter):
    //   lamu_vram_total_mb, lamu_metrics_scrapes_total.
    // Vector series only render once they have a sample — covered by
    // metrics_counts_503 below.
    assert!(text.contains("lamu_vram_total_mb"), "no vram_total: {text}");
    assert!(text.contains("lamu_metrics_scrapes_total"), "no scrapes: {text}");
}

#[tokio::test]
async fn metrics_counts_503() {
    let app = build_app(make_state());
    // Trigger a 503 — router says "will load", loader attempts spawn,
    // the fake gguf path doesn't exist → spawn_failed counter increments.
    // Either label is valid for the "503 path bumps a counter" invariant,
    // so we just check the model label appears on a non-success status.
    let req_body = r#"{"model":"qwen35-27b","messages":[{"role":"user","content":"hi"}]}"#;
    let _ = app.clone().oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(req_body.to_string()))
            .unwrap(),
    ).await.unwrap();
    let resp = app.oneshot(
        Request::builder().uri("/metrics").body(Body::empty()).unwrap(),
    ).await.unwrap();
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let bumped =
        text.contains(r#"lamu_requests_total{model="qwen35-27b",status="spawn_failed",user="anon"} 1"#)
        || text.contains(r#"lamu_requests_total{model="qwen35-27b",status="no_candidate",user="anon"} 1"#)
        || text.contains(r#"lamu_requests_total{model="qwen35-27b",status="no_backend",user="anon"} 1"#);
    assert!(bumped, "metrics body: {text}");
}

#[tokio::test]
async fn chat_completions_validation_error() {
    let app = build_app(make_state());
    let req_body = r#"{"messages":"not-a-list"}"#;
    let resp = app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(req_body.to_string()))
            .unwrap(),
    ).await.unwrap();
    assert!(resp.status().is_client_error() || resp.status() == StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_no_deadlock() {
    // Scale-test P1 (ADR 0017 roadmap): 300 concurrent requests across surfaces
    // hammer the shared AppState (parking_lot scheduler/router/health locks).
    // Asserts no deadlock (whole batch inside a timeout) + every response is a
    // sane status — the safety net for the multi-user/multi-GPU concurrency work.
    // Read-path endpoints only — they all contend the shared scheduler /
    // health / metrics locks, which is what we're stress-testing. The chat
    // spawn path is the e2e test's job (spec_e2e), not a lock test.
    let app = build_app(make_state());
    let mut handles = Vec::new();
    for i in 0..300 {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            let uri = match i % 3 {
                0 => "/health",
                1 => "/v1/models",
                _ => "/metrics",
            };
            app.oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap()
                .status()
        }));
    }
    let results = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        futures_util::future::join_all(handles),
    )
    .await
    .expect("concurrent HTTP batch deadlocked / timed out");
    for r in results {
        let st = r.expect("request task panicked");
        assert!(st.is_success(), "read endpoint failed under concurrency: {st}");
    }
}

#[tokio::test]
async fn unknown_route_404() {
    let app = build_app(make_state());
    let resp = app.oneshot(
        Request::builder().uri("/nonexistent").body(Body::empty()).unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── bearer auth middleware (ADR 0012) ───────────────────────────────

fn state_with_token(tok: &str) -> AppState {
    let mut st = make_state();
    st.auth = Arc::new(AuthMode::StaticToken(tok.to_string()));
    st
}

#[tokio::test]
async fn auth_health_exempt_even_with_token() {
    let app = build_app(state_with_token("lamu_secret"));
    let resp = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_401_without_bearer_when_token_set() {
    let app = build_app(state_with_token("lamu_secret"));
    let resp = app
        .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_401_with_wrong_bearer() {
    let app = build_app(state_with_token("lamu_secret"));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "Bearer lamu_wrong")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_200_with_right_bearer() {
    let app = build_app(state_with_token("lamu_secret"));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "Bearer lamu_secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_metrics_exempt_even_with_token() {
    let app = build_app(state_with_token("lamu_secret"));
    let resp = app
        .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_case_insensitive_scheme_accepted() {
    let app = build_app(state_with_token("lamu_secret"));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "bearer lamu_secret") // lowercase scheme
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn ollama_delegated_503_uses_flat_error() {
    // /api/chat (stream:false) with no model → 503 → flat Ollama {"error":"..."}.
    let app = build_app(make_state());
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"qwen35-27b","stream":false,"messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v["error"].as_str().is_some_and(|m| !m.is_empty())); // flat string, not nested
}

#[tokio::test]
async fn messages_delegated_503_uses_anthropic_envelope() {
    // No model loaded → /v1/messages delegates to chat_completions (503), and
    // the delegated OpenAI-shaped error must be translated to the Anthropic
    // envelope (not passed through). Auth off so we exercise the 503 path.
    let app = build_app(make_state());
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"qwen35-27b","messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["type"], "error"); // Anthropic envelope, not OpenAI {error:{...}}
    assert!(v["error"]["type"].is_string());
    assert!(v["error"]["message"].as_str().is_some_and(|m| !m.is_empty()));
}

#[tokio::test]
async fn auth_401_uses_anthropic_envelope_on_messages() {
    let app = build_app(state_with_token("lamu_secret"));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn cors_preflight_allowed_pre_auth() {
    // CORS is outermost: a browser preflight OPTIONS is answered before the
    // bearer middleware, so browser frontends work even with a token set.
    let app = build_app(state_with_token("lamu_secret"));
    let resp = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("origin", "http://localhost:3000")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(resp.status().is_success());
    assert!(resp.headers().contains_key("access-control-allow-origin"));
}

#[tokio::test]
async fn auth_off_passes_without_bearer() {
    // make_state() has auth_token None → middleware is a no-op.
    let app = build_app(make_state());
    let resp = app
        .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ── per-user quota (ADR 0018 P2) ────────────────────────────────────

use lamu_api::keys::KeyStore;

/// Build state with a KeyStore auth backend holding a single key for `user`
/// with `quota` daily tokens. Returns (state, plaintext_token).
fn state_with_keystore(user: &str, quota: Option<u32>) -> (AppState, String) {
    let path = std::env::temp_dir().join(format!(
        "lamu-http-keys-{}-{}.db",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let _ = std::fs::remove_file(&path);
    let ks = KeyStore::open(&path).unwrap();
    let token = ks.issue_with(user, 0, quota).unwrap();
    let mut st = make_state();
    st.auth = Arc::new(AuthMode::KeyStore(Arc::new(ks)));
    (st, token)
}

#[tokio::test]
async fn quota_unlimited_key_is_not_rate_limited() {
    // None quota → unlimited. With no loaded model the handler still reaches
    // the 503 (no candidate) path, NOT a 429 — proving the gate admitted it.
    let (st, token) = state_with_keystore("alice", None);
    let app = build_app(st);
    let resp = app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"qwen35-27b","messages":[{"role":"user","content":"hi"}]}"#))
            .unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn quota_zero_key_is_429_openai_shape() {
    // Zero daily quota → hard 429 before any routing, OpenAI envelope.
    let (st, token) = state_with_keystore("bob", Some(0));
    let app = build_app(st);
    let resp = app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"qwen35-27b","messages":[{"role":"user","content":"hi"}]}"#))
            .unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(resp.headers().get("retry-after").unwrap(), "3600");
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["type"], "rate_limit_exceeded");
}

#[tokio::test]
async fn quota_zero_key_429_anthropic_shape() {
    let (st, token) = state_with_keystore("carol", Some(0));
    let app = build_app(st);
    let resp = app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"qwen35-27b","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#))
            .unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "rate_limit_error");
}

#[tokio::test]
async fn quota_zero_key_429_ollama_flat_shape() {
    let (st, token) = state_with_keystore("dave", Some(0));
    let app = build_app(st);
    let resp = app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"qwen35-27b","stream":false,"messages":[{"role":"user","content":"hi"}]}"#))
            .unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v["error"].is_string(), "ollama error must be a flat string");
}

#[tokio::test]
async fn static_token_path_never_429s_on_quota() {
    // StaticToken carries NO Principal → unlimited. A right-bearer request
    // with no loaded model returns 503 (no candidate), never 429.
    let app = build_app(state_with_token("lamu_secret"));
    let resp = app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer lamu_secret")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"qwen35-27b","messages":[{"role":"user","content":"hi"}]}"#))
            .unwrap(),
    ).await.unwrap();
    assert_ne!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn streaming_reserves_max_tokens_then_429s_next_request() {
    // Finding 1 (ADR 0018 review): a streaming request must DEPLETE the bucket
    // — otherwise `stream: true` (Ollama's default) bypasses the quota. The
    // ollama/anthropic streaming handlers reserve `max_tokens` up-front, before
    // delegating to the SSE generator, so the charge lands even though the
    // stream body is never polled here.
    let (st, token) = state_with_keystore("frank", Some(50));
    let app = build_app(st);
    // Streaming /api/chat with num_predict=100 reserves 100 against the 50-token
    // bucket → drains it negative. The response is a 200 NDJSON stream; the
    // reserve already charged synchronously in the handler.
    let r1 = app.clone().oneshot(
        Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"qwen35-27b","stream":true,"options":{"num_predict":100},"messages":[{"role":"user","content":"hi"}]}"#))
            .unwrap(),
    ).await.unwrap();
    // r1 is admitted (not 429); its exact status (200 stream, or 503 when the
    // fake gguf can't load) is irrelevant — the reserve charged in the handler
    // before the SSE generator's own resolve step ran.
    assert_ne!(r1.status(), StatusCode::TOO_MANY_REQUESTS, "first stream must be admitted");
    // Any subsequent request for frank now hits the drained bucket → 429.
    let r2 = app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"qwen35-27b","messages":[{"role":"user","content":"hi"}]}"#))
            .unwrap(),
    ).await.unwrap();
    assert_eq!(r2.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn keystore_unknown_token_still_401() {
    // Quota wiring must not weaken auth: an unknown bearer is 401, not 429.
    let (st, _token) = state_with_keystore("eve", Some(100));
    let app = build_app(st);
    let resp = app.oneshot(
        Request::builder()
            .uri("/v1/models")
            .header("authorization", "Bearer lamu_unknown")
            .body(Body::empty())
            .unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}
