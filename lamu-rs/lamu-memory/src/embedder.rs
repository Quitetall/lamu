//! Embedder chain — local-first embedding resolution (ADR 0030).
//!
//! Pre-0030, every embedding in this crate was a hardcoded OpenAI call
//! (`text-embedding-3-small`), gated on `OPENAI_API_KEY`. ADR 0030
//! inverts the default: a LOCAL embedder (lamu-mcp's registry-resolved
//! embedding model, lamu-api's `/v1/embeddings` path, or the ONNX
//! backend behind either) is preferred, with OpenAI kept as an explicit
//! escape hatch and a keyed fallback.
//!
//! ## Why a PROCESS-GLOBAL registration
//!
//! The memory MCP tools are dispatched `HandlerKind::Free` — the
//! handlers never receive a server reference — and the detached tasks
//! (autocapture, reconcile) can't reach one either. Threading a context
//! through every storage fn would change frozen handler signatures, so
//! the composition root (lamu-mcp's `LamuMcpServer::new`, lamu-api's
//! `build_state`, the CLI's reembed command) registers ONE process-wide
//! [`Embedder`] via [`set_global`], and every storage fn resolves it
//! through [`resolve`].
//!
//! ## Resolution order ([`resolve`])
//!
//! 1. `LAMU_EMBED_PROVIDER=openai` → the OpenAI impl (escape hatch) if
//!    `OPENAI_API_KEY` is set, else `None` (warned once). The override
//!    never falls through — it pins the provider.
//! 2. The globally registered local embedder ([`set_global`]).
//! 3. [`OpenAiEmbedder`] iff `OPENAI_API_KEY` is set.
//! 4. `None` — writers store `embedding = NULL`; recall degrades to
//!    FTS + recency (see `crate::hybrid`).
//!
//! The chain is STATIC per process: the composition root registers the
//! local adapter only when the registry has an embedding-capable model
//! at startup. Adding one to the registry later requires a restart to
//! be picked up.
//!
//! ## Sync vs async (judgment call, documented)
//!
//! The trait is **async**. The existing storage fns that embed
//! (`remember`, `recall_memory`, `remember_if_novel`, `supersede`,
//! `index_repo`, `semantic_search`, `recall_ranked`) were already
//! `async` and `.await`ed async-reqwest OpenAI calls, so an async trait
//! keeps every public signature (and the MCP wire) unchanged. A sync
//! trait would have forced the local adapters to `block_on` async work
//! (ensure-load + HTTP) from inside a tokio worker — a panic on a
//! current-thread runtime and a deadlock risk elsewhere.

use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use serde_json::{json, Value};
use std::sync::Arc;

/// The identity an embedder stamps onto every row it embeds: the model
/// name (persisted per-row in `embedding_model`) and its vector width.
/// `dims` may be 0 for adapters that probe the backend lazily — the
/// write path records dims from the actual vectors, never from here.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbedderId {
    pub model: String,
    pub dims: usize,
}

/// One embedding backend. Implementations: [`OpenAiEmbedder`] (cloud,
/// keyed), [`HttpServeEmbedder`] (a running `lamu serve`), and the
/// frontends' composition-root adapters (lamu-mcp's server embed path,
/// lamu-api's in-process resolution).
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    fn identity(&self) -> EmbedderId;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

// ── Process-global registration ─────────────────────────────────────

static GLOBAL: RwLock<Option<Arc<dyn Embedder>>> = RwLock::new(None);

/// Register the process-wide local embedder. Composition-root only.
/// Idempotent: a second call replaces the first (last registration
/// wins — e.g. a CLI process that builds both an MCP server and an API
/// state keeps whichever registered last; both resolve the same
/// registry, so the identity is the same in practice).
pub fn set_global(e: Arc<dyn Embedder>) {
    *GLOBAL.write() = Some(e);
}

/// Drop the process-wide registration. Test seam (and a composition
/// root that wants to force the keyed fallback).
pub fn clear_global() {
    *GLOBAL.write() = None;
}

/// Resolve the embedder chain. See module docs for the order. `None`
/// means "no embedding capability in this process" — writers must
/// store `embedding = NULL` and recall degrades to FTS + recency.
pub fn resolve() -> Option<Arc<dyn Embedder>> {
    // 1. Env escape hatch — pins the provider, never falls through.
    match provider_override().as_deref() {
        Some("openai") => {
            return match openai_key() {
                Some(key) => Some(Arc::new(OpenAiEmbedder::new(key))),
                None => {
                    warn_once_no_key();
                    None
                }
            };
        }
        Some(other) => {
            warn_once_bad_provider(other);
            // Unknown value: ignore the override, continue the chain.
        }
        None => {}
    }
    // 2. Globally registered local embedder (the ADR 0030 default).
    if let Some(e) = GLOBAL.read().clone() {
        return Some(e);
    }
    // 3. Keyed OpenAI fallback.
    openai_key().map(|key| Arc::new(OpenAiEmbedder::new(key)) as Arc<dyn Embedder>)
}

/// The CLI's chain (used by `lamu memory reembed`, which runs in a
/// process with no MCP server or API state to register a local
/// adapter): env override → a RUNNING `lamu serve` (probed via
/// `/health`, then a one-item embed to learn model + dims) → OpenAI.
pub async fn resolve_for_cli(serve_base_url: &str) -> Option<Arc<dyn Embedder>> {
    match provider_override().as_deref() {
        Some("openai") => {
            return match openai_key() {
                Some(key) => Some(Arc::new(OpenAiEmbedder::new(key))),
                None => {
                    warn_once_no_key();
                    None
                }
            };
        }
        Some(other) => warn_once_bad_provider(other),
        None => {}
    }
    if let Some(h) = HttpServeEmbedder::probe(serve_base_url).await {
        return Some(Arc::new(h));
    }
    openai_key().map(|key| Arc::new(OpenAiEmbedder::new(key)) as Arc<dyn Embedder>)
}

fn provider_override() -> Option<String> {
    std::env::var("LAMU_EMBED_PROVIDER")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
}

fn warn_once_no_key() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            "LAMU_EMBED_PROVIDER=openai but OPENAI_API_KEY is unset — no embedder; \
             memories store without embeddings, recall degrades to FTS + recency"
        )
    });
}

fn warn_once_bad_provider(got: &str) {
    // Per-VALUE gating (not a single Once): if the operator changes a bad
    // override to a different bad value mid-debug, the new value still logs.
    static WARNED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    let seen = WARNED.get_or_init(Default::default);
    if seen.lock().expect("warn set poisoned").insert(got.to_string()) {
        tracing::warn!(
            "LAMU_EMBED_PROVIDER='{got}' is not a known provider (expected 'openai') — \
             ignoring the override and continuing the chain"
        );
    }
}

// ── OpenAI impl (the pre-0030 plumbing, moved here from rag.rs) ─────

/// OpenAI's embedding model + dimension. The escape-hatch / fallback
/// provider; also the model every pre-0030 row was embedded with (the
/// legacy import backfills `embedding_model` with this).
pub const OPENAI_EMBED_MODEL: &str = "text-embedding-3-small";
pub const OPENAI_EMBED_DIMS: usize = 1536;

/// Resolve the OpenAI API key. If unset, the OpenAI leg of the chain is
/// unavailable.
pub(crate) fn openai_key() -> Option<String> {
    std::env::var("OPENAI_API_KEY").ok().filter(|s| !s.is_empty())
}

/// Pooled embeddings client (no default timeout — set per request). A
/// fresh Client per embed meant a new TLS handshake to api.openai.com on
/// every remember/recall/supersede — this reuses one connection pool.
pub(crate) fn embed_client() -> &'static reqwest::Client {
    static EMBED_CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    EMBED_CLIENT.get_or_init(reqwest::Client::new)
}

/// Cloud OpenAI embeddings — `text-embedding-3-small`, 1536 dims.
/// Chain position: env-override target (1) and keyed fallback (3).
pub struct OpenAiEmbedder {
    key: String,
}

impl OpenAiEmbedder {
    pub fn new(key: String) -> Self {
        Self { key }
    }
}

#[async_trait::async_trait]
impl Embedder for OpenAiEmbedder {
    fn identity(&self) -> EmbedderId {
        EmbedderId {
            model: OPENAI_EMBED_MODEL.to_string(),
            dims: OPENAI_EMBED_DIMS,
        }
    }

    /// Batch-embed. OpenAI accepts an `input` array up to 2048 items per
    /// call; we cap at 96 for safety + smaller payloads (unchanged from
    /// the pre-0030 `rag::embed_batch`).
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(96) {
            let body = json!({
                "model": OPENAI_EMBED_MODEL,
                "input": chunk,
            });
            let resp = embed_client()
                .post("https://api.openai.com/v1/embeddings")
                .timeout(std::time::Duration::from_secs(120))
                .bearer_auth(&self.key)
                .json(&body)
                .send()
                .await?;
            let v: Value = resp.json().await?;
            all.extend(parse_embeddings_data(&v)?);
        }
        Ok(all)
    }
}

/// Parse the OpenAI-shaped `{"data":[{"embedding":[..]},..]}` envelope
/// every backend in the chain speaks (OpenAI itself, llama-server
/// `--embedding`, lamu-api's proxy, the ONNX backend).
pub fn parse_embeddings_data(v: &Value) -> Result<Vec<Vec<f32>>> {
    let arr = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("embeddings response missing data[]"))?;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let emb = entry
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| anyhow!("embeddings item missing embedding[]"))?;
        out.push(
            emb.iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect(),
        );
    }
    Ok(out)
}

// ── HTTP `lamu serve` impl (CLI chain leg) ──────────────────────────

/// Embeds against a RUNNING `lamu serve`'s `/v1/embeddings` (which
/// resolves + ensure-loads its registry's embedding-capable model).
/// Constructed only via [`HttpServeEmbedder::probe`], which confirms
/// `/health` answers and runs a one-item embed to learn the model name
/// and dims up front (the CLI's dry-run needs the identity before any
/// bulk embed).
pub struct HttpServeEmbedder {
    base: String,
    id: EmbedderId,
}

impl HttpServeEmbedder {
    /// Probe `base_url` (e.g. `http://127.0.0.1:8020`). Returns `None`
    /// when the server is unreachable, unhealthy, or has no embedding
    /// model to serve.
    pub async fn probe(base_url: &str) -> Option<Self> {
        let base = base_url.trim_end_matches('/').to_string();
        let client = embed_client();
        let health = client
            .get(format!("{base}/health"))
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .ok()?;
        if !health.status().is_success() {
            return None;
        }
        // One-item embed: learns the served model's name + dims, and
        // proves the embedding path actually works (a serve without an
        // embedding-capable registry entry 503s here → None).
        let resp = client
            .post(format!("{base}/v1/embeddings"))
            .timeout(std::time::Duration::from_secs(120))
            .json(&json!({ "input": ["lamu embedder identity probe"] }))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let v: Value = resp.json().await.ok()?;
        let vecs = parse_embeddings_data(&v).ok()?;
        let dims = vecs.first().map(|e| e.len()).filter(|&d| d > 0)?;
        let model = v
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("lamu-serve-embedding")
            .to_string();
        Some(Self {
            base,
            id: EmbedderId { model, dims },
        })
    }
}

#[async_trait::async_trait]
impl Embedder for HttpServeEmbedder {
    fn identity(&self) -> EmbedderId {
        self.id.clone()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(64) {
            let resp = embed_client()
                .post(format!("{}/v1/embeddings", self.base))
                .timeout(std::time::Duration::from_secs(300))
                .json(&json!({ "model": self.id.model, "input": chunk }))
                .send()
                .await?;
            if !resp.status().is_success() {
                return Err(anyhow!(
                    "lamu serve embeddings {}: {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                ));
            }
            let v: Value = resp.json().await?;
            all.extend(parse_embeddings_data(&v)?);
        }
        if all.len() != texts.len() {
            return Err(anyhow!(
                "lamu serve embed count mismatch: requested {}, got {}",
                texts.len(),
                all.len()
            ));
        }
        Ok(all)
    }
}

// ── Test support ────────────────────────────────────────────────────

/// Deterministic in-process embedder for tests: a text→vector lookup
/// with a default. Lives behind cfg(test); never compiled into release.
#[cfg(test)]
pub(crate) mod testutil {
    use super::*;
    use std::collections::HashMap;

    pub(crate) struct FakeEmbedder {
        pub model: String,
        pub dims: usize,
        pub map: HashMap<String, Vec<f32>>,
        pub default: Vec<f32>,
    }

    impl FakeEmbedder {
        pub(crate) fn new(model: &str, default: Vec<f32>) -> Self {
            let dims = default.len();
            Self {
                model: model.to_string(),
                dims,
                map: HashMap::new(),
                default,
            }
        }

        pub(crate) fn with(mut self, text: &str, v: Vec<f32>) -> Self {
            self.map.insert(text.to_string(), v);
            self
        }
    }

    #[async_trait::async_trait]
    impl Embedder for FakeEmbedder {
        fn identity(&self) -> EmbedderId {
            EmbedderId {
                model: self.model.clone(),
                dims: self.dims,
            }
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| self.map.get(t).cloned().unwrap_or_else(|| self.default.clone()))
                .collect())
        }
    }

    /// Serializes tests that mutate the process-global embedder or the
    /// chain's env vars (`LAMU_EMBED_PROVIDER`, `OPENAI_API_KEY`). Every
    /// such test MUST hold this guard for its full body.
    pub(crate) fn chain_lock() -> parking_lot::MutexGuard<'static, ()> {
        static LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
        LOCK.lock()
    }

    /// Clear the chain's inputs to a known state (no override, no key,
    /// no global). Caller must hold [`chain_lock`].
    pub(crate) fn reset_chain() {
        clear_global();
        // SAFETY: serialized by chain_lock; no other thread reads these
        // env vars concurrently in the test binary.
        unsafe {
            std::env::remove_var("LAMU_EMBED_PROVIDER");
            std::env::remove_var("OPENAI_API_KEY");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::*;
    use super::*;

    #[test]
    fn no_global_no_key_resolves_none() {
        let _g = chain_lock();
        reset_chain();
        assert!(resolve().is_none());
    }

    #[test]
    fn global_registration_wins_and_is_replaceable() {
        let _g = chain_lock();
        reset_chain();
        set_global(Arc::new(FakeEmbedder::new("fake-a", vec![1.0, 0.0])));
        let e = resolve().expect("registered embedder resolves");
        assert_eq!(e.identity().model, "fake-a");
        assert_eq!(e.identity().dims, 2);
        // Idempotent registration: a second set_global replaces.
        set_global(Arc::new(FakeEmbedder::new("fake-b", vec![0.0, 1.0, 0.0])));
        assert_eq!(resolve().unwrap().identity().model, "fake-b");
        reset_chain();
    }

    #[test]
    fn global_beats_openai_key() {
        let _g = chain_lock();
        reset_chain();
        unsafe { std::env::set_var("OPENAI_API_KEY", "test-key-never-used") };
        set_global(Arc::new(FakeEmbedder::new("local-fake", vec![1.0])));
        // SELECTION only — embed() is never called, no network.
        assert_eq!(resolve().unwrap().identity().model, "local-fake");
        reset_chain();
    }

    #[test]
    fn env_override_forces_openai_over_global() {
        let _g = chain_lock();
        reset_chain();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test-key-never-used");
            std::env::set_var("LAMU_EMBED_PROVIDER", "openai");
        }
        set_global(Arc::new(FakeEmbedder::new("local-fake", vec![1.0])));
        let e = resolve().expect("override selects OpenAI");
        assert_eq!(e.identity().model, OPENAI_EMBED_MODEL);
        assert_eq!(e.identity().dims, OPENAI_EMBED_DIMS);
        reset_chain();
    }

    #[test]
    fn env_override_without_key_is_none_not_fallthrough() {
        let _g = chain_lock();
        reset_chain();
        unsafe { std::env::set_var("LAMU_EMBED_PROVIDER", "openai") };
        // A global IS registered, but the override pins the provider —
        // it must NOT fall through to the local embedder.
        set_global(Arc::new(FakeEmbedder::new("local-fake", vec![1.0])));
        assert!(resolve().is_none());
        reset_chain();
    }

    #[test]
    fn unknown_override_falls_through_the_chain() {
        let _g = chain_lock();
        reset_chain();
        unsafe { std::env::set_var("LAMU_EMBED_PROVIDER", "frobnicator") };
        set_global(Arc::new(FakeEmbedder::new("local-fake", vec![1.0])));
        assert_eq!(resolve().unwrap().identity().model, "local-fake");
        reset_chain();
    }

    #[test]
    fn key_only_resolves_openai() {
        let _g = chain_lock();
        reset_chain();
        unsafe { std::env::set_var("OPENAI_API_KEY", "test-key-never-used") };
        assert_eq!(resolve().unwrap().identity().model, OPENAI_EMBED_MODEL);
        reset_chain();
    }

    #[test]
    fn parse_embeddings_data_shapes() {
        let v = json!({"data": [{"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]}]});
        let out = parse_embeddings_data(&v).unwrap();
        assert_eq!(out, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(parse_embeddings_data(&json!({})).is_err());
        assert!(parse_embeddings_data(&json!({"data": [{"no_embedding": 1}]})).is_err());
    }
}
