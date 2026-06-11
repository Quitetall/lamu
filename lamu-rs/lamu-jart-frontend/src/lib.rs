//! lamu-jart-frontend — FRONTEND (ADR 0023): drives the lamu-jart research
//! capability.
//!
//! R0: a LAMU-backed [`Summarizer`] that generates IN-PROCESS via
//! [`ToolCtx::generate`] (no self-HTTP to `:8020`), so jart's TUI/web frontends
//! summarize through LAMU's scheduler + routing. The `lamu research` TUI and the
//! bundled web SPA come in later phases.

mod orchestrator;
pub use orchestrator::run_orchestrator_tui;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use jart::core::ai::{build_grounded_content, Summarizer};
use jart::core::cache::Cache;
use jart::core::config::Config;
use jart::core::ratelimit::Pacer;
use jart::server::{router, AppState};
use jart::tui::{self, TuiConfig};
use lamu_core::tools_ext::ToolCtx;
use std::path::PathBuf;
use std::sync::Arc;

/// jart's Python scrapers dir. `JART_SCRAPERS_DIR` overrides; else the jart
/// checkout's `scrapers/`. (lamu-jart-frontend can't read jart's
/// `CARGO_MANIFEST_DIR`, so it resolves the standalone checkout.)
fn scrapers_dir() -> PathBuf {
    jart_path("JART_SCRAPERS_DIR", "scrapers")
}
/// jart's built web SPA dir. `JART_DIST_DIR` overrides; else the checkout's
/// `frontend/dist/`.
fn dist_dir() -> PathBuf {
    jart_path("JART_DIST_DIR", "frontend/dist")
}
fn jart_path(env_key: &str, rel: &str) -> PathBuf {
    if let Some(p) = std::env::var_os(env_key).filter(|s| !s.is_empty()) {
        return PathBuf::from(p);
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join("Desktop/jart")
        .join(rel)
}

/// Run jart's GRAPHICAL frontend (the ratatui TUI, or the bundled web SPA when
/// `web`) wired to a LAMU-backed [`LamuSummarizer`] — so the "Summarize with AI"
/// feature generates IN-PROCESS via `ctx.generate` instead of a self-HTTP round
/// trip. Mirrors `jart::cli::run`'s setup but injects the LAMU summarizer at
/// jart's `Arc<dyn Summarizer>` seam. `model` is the summary model (local or
/// cloud; routed by `ctx.generate`).
pub async fn run_graphical<C: ToolCtx + 'static>(
    ctx: Arc<C>,
    model: String,
    web: bool,
) -> Result<()> {
    let cfg = Config::load(None);
    let ai: Arc<dyn Summarizer> = Arc::new(LamuSummarizer::new(ctx, model));
    let scrapers = scrapers_dir();
    let dist = dist_dir();
    let cache = Arc::new(Cache::new());
    let pacer = Arc::new(Pacer::new());
    let topics = cfg.topics();
    let addr = format!("127.0.0.1:{}", cfg.web_port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let web_url = format!("http://{addr}");

    // Web server in the background for both modes (TUI's `w` key + `--web`).
    let state = AppState {
        scrapers_dir: scrapers.clone(),
        topics: topics.clone(),
        ai: ai.clone(),
        dist_dir: dist,
        cache: cache.clone(),
        pacer: pacer.clone(),
    };
    let server = router(state);
    tokio::spawn(async move {
        let _ = axum::serve(listener, server).await;
    });

    if web {
        let _ = std::process::Command::new("xdg-open").arg(&web_url).spawn();
        println!("jart web GUI (LAMU-backed) on {web_url}  (Ctrl-C to stop)");
        tokio::signal::ctrl_c().await?;
        return Ok(());
    }
    tui::run(TuiConfig { scrapers_dir: scrapers, topics, ai, web_url, cache, pacer }).await
}

/// jart [`Summarizer`] backed by an in-process [`ToolCtx`] (a running
/// `LamuMcpServer`). Generic over the ctx so it is unit-testable with a fake and
/// doesn't hard-bind to `lamu-mcp` at the type level. `model` is a local
/// registry id OR a cloud id — `ctx.generate` routes either.
pub struct LamuSummarizer<C: ToolCtx> {
    ctx: Arc<C>,
    model: String,
}

impl<C: ToolCtx> LamuSummarizer<C> {
    pub fn new(ctx: Arc<C>, model: impl Into<String>) -> Self {
        Self { ctx, model: model.into() }
    }
}

#[async_trait]
impl<C: ToolCtx + 'static> Summarizer for LamuSummarizer<C> {
    async fn summarize(&self, prompt: &str, items: &[String]) -> Result<String> {
        // Same prompt-injection fence + positional citation indexing as the HTTP
        // path (item N -> <source id="N"> -> citation [N]).
        let content = build_grounded_content(prompt, items);

        // A LOCAL registry model must be loaded before generate — handle_query
        // does NOT auto-load a cold model. `model_modality` is Some only for
        // local models; cloud models are routed by `generate` and need no load.
        if self.ctx.model_modality(&self.model).is_some() {
            if let Err(e) = self.ctx.ensure_loaded(&self.model).await {
                return Err(anyhow!("load model '{}': {e}", self.model));
            }
        }

        // ADR 0027: the seam is typed — Err carries the failure message.
        self.ctx
            .generate(&self.model, &content)
            .await
            .map_err(|e| anyhow!("{e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamu_core::tools_ext::ToolCtxError;
    use lamu_core::types::Modality;

    /// Fake ToolCtx: toggle local-vs-cloud, load success, and the generate result.
    struct FakeCtx {
        local: bool,
        load_ok: bool,
        gen_out: std::result::Result<String, String>,
    }

    #[async_trait]
    impl ToolCtx for FakeCtx {
        fn model_modality(&self, _m: &str) -> Option<Modality> {
            self.local.then_some(Modality::Llm)
        }
        async fn ensure_loaded(&self, _m: &str) -> std::result::Result<String, ToolCtxError> {
            if self.load_ok {
                Ok("loaded".into())
            } else {
                Err(ToolCtxError::Load("out of VRAM".into()))
            }
        }
        fn loaded_port(&self, _m: &str) -> Option<u16> {
            None
        }
        async fn generate(
            &self,
            _m: &str,
            _p: &str,
        ) -> std::result::Result<String, ToolCtxError> {
            self.gen_out.clone().map_err(ToolCtxError::Generate)
        }
        async fn embed(
            &self,
            texts: &[String],
        ) -> std::result::Result<Vec<Vec<f32>>, ToolCtxError> {
            Ok(texts.iter().map(|_| vec![0.0_f32; 4]).collect())
        }
    }

    #[tokio::test]
    async fn cloud_model_skips_load_and_returns_text() {
        // Cloud model: model_modality None -> no ensure_loaded (load_ok=false
        // would error if it were called).
        let ctx = Arc::new(FakeCtx { local: false, load_ok: false, gen_out: Ok("a summary".into()) });
        let s = LamuSummarizer::new(ctx, "claude-opus-4-7");
        assert_eq!(s.summarize("sum", &["t".into()]).await.unwrap(), "a summary");
    }

    #[tokio::test]
    async fn local_cold_load_failure_surfaces_as_err() {
        let ctx = Arc::new(FakeCtx { local: true, load_ok: false, gen_out: Ok("x".into()) });
        let s = LamuSummarizer::new(ctx, "qwen3.6-27b");
        let err = s.summarize("sum", &["t".into()]).await.unwrap_err();
        assert!(err.to_string().contains("load model"), "got: {err}");
    }

    #[tokio::test]
    async fn generate_err_propagates_as_err() {
        let ctx = Arc::new(FakeCtx { local: false, load_ok: true, gen_out: Err("boom".into()) });
        let s = LamuSummarizer::new(ctx, "mimo-v2.5");
        let err = s.summarize("sum", &["t".into()]).await.unwrap_err();
        assert!(err.to_string().contains("boom"), "got: {err}");
    }

    #[tokio::test]
    async fn local_loaded_returns_text() {
        let ctx = Arc::new(FakeCtx { local: true, load_ok: true, gen_out: Ok("ok summary".into()) });
        let s = LamuSummarizer::new(ctx, "qwen3.6-27b");
        assert_eq!(s.summarize("sum", &["t".into()]).await.unwrap(), "ok summary");
    }
}
