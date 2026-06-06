//! lamu-jart-frontend — FRONTEND (ADR 0023): drives the lamu-jart research
//! capability.
//!
//! R0: a LAMU-backed [`Summarizer`] that generates IN-PROCESS via
//! [`ToolCtx::generate`] (no self-HTTP to `:8020`), so jart's TUI/web frontends
//! summarize through LAMU's scheduler + routing. The `lamu research` TUI and the
//! bundled web SPA come in later phases.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use jart::core::ai::{build_grounded_content, Summarizer};
use lamu_core::tools_ext::ToolCtx;
use std::sync::Arc;

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
        // does NOT auto-load a cold model (it returns "error: model ... not
        // loaded"). `model_modality` is Some only for local models; cloud models
        // are routed by `generate` and need no load.
        if self.ctx.model_modality(&self.model).is_some() {
            let status = self.ctx.ensure_loaded(&self.model).await;
            // handle_load_model returns "error: ..." (with colon) on failure;
            // key on the colon to match generate's convention.
            if status.trim_start().to_lowercase().starts_with("error:") {
                return Err(anyhow!("load model '{}': {status}", self.model));
            }
        }

        let out = self.ctx.generate(&self.model, &content).await;
        // ToolCtx convention: an "error:"-prefixed string is a failure (matches
        // the server's is_error check, which keys on the colon).
        if out.trim_start().to_lowercase().starts_with("error:") {
            return Err(anyhow!(out));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamu_core::types::Modality;

    /// Fake ToolCtx: toggle local-vs-cloud, load success, and the generate output.
    struct FakeCtx {
        local: bool,
        load_ok: bool,
        gen_out: String,
    }

    #[async_trait]
    impl ToolCtx for FakeCtx {
        fn model_modality(&self, _m: &str) -> Option<Modality> {
            self.local.then_some(Modality::Llm)
        }
        async fn ensure_loaded(&self, _m: &str) -> String {
            if self.load_ok { "loaded".into() } else { "error: out of VRAM".into() }
        }
        fn loaded_port(&self, _m: &str) -> Option<u16> {
            None
        }
        async fn generate(&self, _m: &str, _p: &str) -> String {
            self.gen_out.clone()
        }
    }

    #[tokio::test]
    async fn cloud_model_skips_load_and_returns_text() {
        // Cloud model: model_modality None -> no ensure_loaded (load_ok=false
        // would error if it were called).
        let ctx = Arc::new(FakeCtx { local: false, load_ok: false, gen_out: "a summary".into() });
        let s = LamuSummarizer::new(ctx, "claude-opus-4-7");
        assert_eq!(s.summarize("sum", &["t".into()]).await.unwrap(), "a summary");
    }

    #[tokio::test]
    async fn local_cold_load_failure_surfaces_as_err() {
        let ctx = Arc::new(FakeCtx { local: true, load_ok: false, gen_out: "x".into() });
        let s = LamuSummarizer::new(ctx, "qwen3.6-27b");
        let err = s.summarize("sum", &["t".into()]).await.unwrap_err();
        assert!(err.to_string().contains("load model"), "got: {err}");
    }

    #[tokio::test]
    async fn error_prefixed_generate_is_err() {
        let ctx = Arc::new(FakeCtx { local: false, load_ok: true, gen_out: "error: boom".into() });
        let s = LamuSummarizer::new(ctx, "mimo-v2.5");
        assert!(s.summarize("sum", &["t".into()]).await.is_err());
    }

    #[tokio::test]
    async fn local_loaded_returns_text() {
        let ctx = Arc::new(FakeCtx { local: true, load_ok: true, gen_out: "ok summary".into() });
        let s = LamuSummarizer::new(ctx, "qwen3.6-27b");
        assert_eq!(s.summarize("sum", &["t".into()]).await.unwrap(), "ok summary");
    }
}
