//! HTTP route tests for lamu-api. Uses tower::ServiceExt::oneshot to drive
//! axum without binding a real port.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use lamu_api::metrics::LamuMetrics;
use lamu_api::openai_compat::{build_app, AppState};
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
        text.contains(r#"lamu_requests_total{model="qwen35-27b",status="spawn_failed"} 1"#)
        || text.contains(r#"lamu_requests_total{model="qwen35-27b",status="no_candidate"} 1"#)
        || text.contains(r#"lamu_requests_total{model="qwen35-27b",status="no_backend"} 1"#);
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

#[tokio::test]
async fn unknown_route_404() {
    let app = build_app(make_state());
    let resp = app.oneshot(
        Request::builder().uri("/nonexistent").body(Body::empty()).unwrap(),
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
