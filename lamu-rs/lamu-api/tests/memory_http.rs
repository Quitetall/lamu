//! HTTP route tests for the ADR 0032 memory API (`/v1/memory/*`).
//! tower::ServiceExt::oneshot drives axum without binding a port, same
//! pattern as `http.rs`.
//!
//! PROCESS-GLOBAL SETUP: lamu-memory's store singleton pins to
//! `$LAMU_DB` on FIRST touch, and the embedder chain is process-wide.
//! `init()` (std::sync::Once, called at the top of every test) claims
//! both BEFORE any test body runs: `LAMU_DB` → a per-process temp dir,
//! the chain → a deterministic constant-vector fake (also shielding the
//! suite from a real `OPENAI_API_KEY` in the environment). All tests in
//! this binary share that one DB — each uses its own owners + unique
//! fact tokens so they can run on parallel threads without interfering.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use lamu_api::keys::KeyStore;
use lamu_api::metrics::LamuMetrics;
use lamu_api::openai_compat::{build_app, AppState, AuthMode};
use lamu_core::health::HealthRegistry;
use lamu_core::router::Router;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{BackendType, Capability, ModelEntry, ModelFormat};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Once};
use tower::util::ServiceExt;

// ── process-global setup ────────────────────────────────────────────

/// Constant-vector embedder: every text maps to [1.0, 0.0]. Exercises
/// the vector leg deterministically — under owner scoping, EVERY fact
/// is a perfect cosine match for every query, so the ONLY thing that
/// can keep another owner's fact out of a recall is the owner filter.
struct ConstEmbedder;

#[async_trait::async_trait]
impl lamu_memory::embedder::Embedder for ConstEmbedder {
    fn identity(&self) -> lamu_memory::embedder::EmbedderId {
        lamu_memory::embedder::EmbedderId { model: "memhttp-fake".into(), dims: 2 }
    }
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![1.0, 0.0]; texts.len()])
    }
}

fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let dir = std::env::temp_dir().join(format!(
            "lamu-memhttp-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create test data dir");
        // SAFETY: Once runs to completion before any caller proceeds, so
        // every env mutation happens-before every test body.
        unsafe {
            std::env::set_var("LAMU_DB", dir.join("lamu.db"));
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("LAMU_EMBED_PROVIDER");
        }
        lamu_memory::embedder::set_global(Arc::new(ConstEmbedder));
    });
}

// ── state fixtures (mirrors http.rs) ────────────────────────────────

fn sample_entry(name: &str) -> ModelEntry {
    ModelEntry {
        name: name.to_string(),
        path: PathBuf::from(format!("/tmp/{name}.gguf")),
        format: ModelFormat::Gguf,
        backend: BackendType::LlamaCpp,
        backend_kind: None,
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
        system_prompt: None,
    }
}

fn make_state() -> AppState {
    let entries = vec![sample_entry("qwen35-27b")];
    let entries_map: HashMap<String, ModelEntry> =
        entries.iter().map(|e| (e.name.clone(), e.clone())).collect();
    let scheduler = VramScheduler::new();
    let router = Router::new(&scheduler, entries.clone());
    AppState {
        scheduler: Arc::new(Mutex::new(scheduler)),
        router: Arc::new(Mutex::new(router)),
        entries: Arc::new(entries_map),
        client: reqwest::Client::new(),
        health: Arc::new(Mutex::new(HealthRegistry::new())),
        metrics: Arc::new(LamuMetrics::new().unwrap()),
        http_port: 8020,
        auth: Arc::new(AuthMode::Off),
        quota: Arc::new(lamu_api::quota::QuotaManager::new()),
        priority_queue: None,
    }
}

/// KeyStore-auth state with one key per user. Returns the state plus
/// the plaintext tokens in `users` order.
fn state_with_keys(users: &[&str]) -> (AppState, Vec<String>) {
    let path = std::env::temp_dir().join(format!(
        "lamu-memhttp-keys-{}-{}.db",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_file(&path);
    let ks = KeyStore::open(&path).unwrap();
    let tokens = users.iter().map(|u| ks.issue_with(u, 0, None).unwrap()).collect();
    let mut st = make_state();
    st.auth = Arc::new(AuthMode::KeyStore(Arc::new(ks)));
    (st, tokens)
}

// ── request helpers ─────────────────────────────────────────────────

fn post(uri: &str, bearer: Option<&str>, body: &str) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(tok) = bearer {
        b = b.header("authorization", format!("Bearer {tok}"));
    }
    b.body(Body::from(body.to_string())).unwrap()
}

async fn json_of(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

// ── tests ───────────────────────────────────────────────────────────

#[tokio::test]
async fn memory_routes_401_without_bearer_when_auth_on() {
    init();
    let (st, _tokens) = state_with_keys(&["authcheck"]);
    let app = build_app(st);
    for uri in [
        "/v1/memory/remember",
        "/v1/memory/recall",
        "/v1/memory/forget",
        "/v1/memory/supersede",
    ] {
        let resp = app
            .clone()
            .oneshot(post(uri, None, r#"{"text":"x","query":"x","id":1,"old_id":1}"#))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED, "{uri} must sit inside the bearer layer");
    }
}

#[tokio::test]
async fn keystore_two_key_isolation_end_to_end() {
    init();
    let (st, tokens) = state_with_keys(&["alice-iso", "bob-iso"]);
    let (tok_a, tok_b) = (&tokens[0], &tokens[1]);
    let app = build_app(st);

    // remember as A.
    let resp = app
        .clone()
        .oneshot(post(
            "/v1/memory/remember",
            Some(tok_a),
            r#"{"text":"isofact alpha keeps her notes in zoxide"}"#,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let id = json_of(resp).await["id"].as_i64().expect("remember returns id");

    // recall as B → empty (vector leg matches everything via the const
    // embedder, FTS matches "isofact" — only the owner filter hides it).
    let resp = app
        .clone()
        .oneshot(post("/v1/memory/recall", Some(tok_b), r#"{"query":"isofact"}"#))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let hits = json_of(resp).await["hits"].as_array().unwrap().clone();
    assert!(hits.is_empty(), "B must not see A's facts: {hits:?}");

    // recall as A → the fact comes back.
    let resp = app
        .clone()
        .oneshot(post("/v1/memory/recall", Some(tok_a), r#"{"query":"isofact"}"#))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let hits = json_of(resp).await["hits"].as_array().unwrap().clone();
    assert!(
        hits.iter().any(|h| h["id"].as_i64() == Some(id)),
        "A must recall her own fact: {hits:?}"
    );

    // forget as B against A's id → forgotten:false (same as missing id).
    let resp = app
        .clone()
        .oneshot(post("/v1/memory/forget", Some(tok_b), &format!(r#"{{"id":{id}}}"#)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(json_of(resp).await["forgotten"], false);

    // supersede as B against A's id → 404 not_found envelope.
    let resp = app
        .clone()
        .oneshot(post(
            "/v1/memory/supersede",
            Some(tok_b),
            &format!(r#"{{"old_id":{id},"text":"bob takeover"}}"#),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v = json_of(resp).await;
    assert_eq!(v["error"]["type"], "not_found");

    // supersede as A works and the old id drops out of default recall.
    let resp = app
        .clone()
        .oneshot(post(
            "/v1/memory/supersede",
            Some(tok_a),
            &format!(r#"{{"old_id":{id},"text":"isofact alpha moved her notes to zellij"}}"#),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let new_id = json_of(resp).await["id"].as_i64().expect("supersede returns id");
    assert_ne!(new_id, id);

    // forget as A on the (still current) new fact → forgotten:true.
    let resp = app
        .clone()
        .oneshot(post("/v1/memory/forget", Some(tok_a), &format!(r#"{{"id":{new_id}}}"#)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(json_of(resp).await["forgotten"], true);
}

#[tokio::test]
async fn off_mode_shares_owner_local_with_mcp_writes() {
    init();
    // A fact written through lamu_memory directly with owner "local" —
    // exactly what every MCP tool handler does.
    lamu_memory::lifetime_memory::remember(
        "offmode zebrafact lives in the shared local owner",
        "fact",
        "manual",
        lamu_memory::lifetime_memory::LOCAL_OWNER,
    )
    .await
    .unwrap();

    // AuthMode::Off → no Principal → the HTTP caller IS owner "local".
    let app = build_app(make_state());
    let resp = app
        .oneshot(post("/v1/memory/recall", None, r#"{"query":"zebrafact"}"#))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = json_of(resp).await;
    let hits = v["hits"].as_array().unwrap();
    assert!(
        hits.iter().any(|h| h["text"].as_str().unwrap_or("").contains("zebrafact")),
        "Off-mode recall must see MCP-written local facts: {hits:?}"
    );
    // Response shape: every hit carries the documented fields.
    for h in hits {
        for field in ["id", "text", "kind", "ts"] {
            assert!(!h[field].is_null(), "hit missing {field}: {h}");
        }
        assert!(h.as_object().unwrap().contains_key("score"));
        assert!(h.as_object().unwrap().contains_key("valid_until"));
    }
}

#[tokio::test]
async fn malformed_body_is_400_with_envelope() {
    init();
    let app = build_app(make_state());
    // Broken JSON.
    let resp = app
        .clone()
        .oneshot(post("/v1/memory/remember", None, r#"{"text": unquoted}"#))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = json_of(resp).await;
    assert_eq!(v["error"]["type"], "invalid_request_error");
    // Wrong shape (missing required field).
    let resp = app
        .clone()
        .oneshot(post("/v1/memory/recall", None, r#"{"k": 3}"#))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    // Present-but-empty text.
    let resp = app
        .oneshot(post("/v1/memory/remember", None, r#"{"text": "   "}"#))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = json_of(resp).await;
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn recall_clamps_k_instead_of_erroring() {
    init();
    let app = build_app(make_state());
    for body in [r#"{"query":"clampcheck","k":0}"#, r#"{"query":"clampcheck","k":100000}"#] {
        let resp = app
            .clone()
            .oneshot(post("/v1/memory/recall", None, body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "k out of range must clamp, not error");
    }
}
