//! Test doubles for the ToolCtx seam (ADR 0023/0027).
//!
//! `FakeCtx` is a scripted [`ToolCtx`]: queue responses up front, run the
//! agentic flow, assert on the output — no model, no GPU, no network.
//! Gated behind the `test-support` feature so production binaries never
//! compile it; consumer crates enable it in `[dev-dependencies]`:
//!
//! ```toml
//! lamu-core = { workspace = true, features = ["test-support"] }
//! ```
//!
//! Determinism: each method pops its FIFO queue. An empty queue yields a
//! loud, recognizable default (`Ok("loaded")` for ensure_loaded — the
//! common don't-care — and `Err("FakeCtx: no scripted … response")` for
//! generate/embed) so a flow that calls more than the test scripted fails
//! visibly instead of silently succeeding.

use crate::tools_ext::{ToolCtx, ToolCtxError};
use crate::types::Modality;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

type GenResult = Result<String, ToolCtxError>;
type EmbedResult = Result<Vec<Vec<f32>>, ToolCtxError>;

/// A recorded `generate` call: (model, prompt, max_tokens, temperature).
pub type GenerateCall = (String, String, Option<u32>, Option<f32>);

#[derive(Default)]
pub struct FakeCtx {
    modalities: HashMap<String, Modality>,
    ports: HashMap<String, u16>,
    ensure_loaded_q: Mutex<VecDeque<GenResult>>,
    generate_q: Mutex<VecDeque<GenResult>>,
    embed_q: Mutex<VecDeque<EmbedResult>>,
    /// Every `generate` call, in order — assert prompts/budgets reached the seam.
    pub generate_calls: Mutex<Vec<GenerateCall>>,
}

impl FakeCtx {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `model` as a LOCAL model of the given modality (None = cloud).
    pub fn with_modality(mut self, model: &str, m: Modality) -> Self {
        self.modalities.insert(model.to_string(), m);
        self
    }

    pub fn with_port(mut self, model: &str, port: u16) -> Self {
        self.ports.insert(model.to_string(), port);
        self
    }

    pub fn enqueue_ensure_loaded(self, r: GenResult) -> Self {
        self.ensure_loaded_q.lock().unwrap().push_back(r);
        self
    }

    pub fn enqueue_generate(self, r: GenResult) -> Self {
        self.generate_q.lock().unwrap().push_back(r);
        self
    }

    pub fn enqueue_embed(self, r: EmbedResult) -> Self {
        self.embed_q.lock().unwrap().push_back(r);
        self
    }

    /// Convenience: queue a successful generate returning `s`.
    pub fn gen_ok(self, s: &str) -> Self {
        self.enqueue_generate(Ok(s.to_string()))
    }

    /// Convenience: queue a failed generate with message `m`.
    pub fn gen_err(self, m: &str) -> Self {
        self.enqueue_generate(Err(ToolCtxError::Generate(m.to_string())))
    }
}

#[async_trait::async_trait]
impl ToolCtx for FakeCtx {
    fn model_modality(&self, model: &str) -> Option<Modality> {
        self.modalities.get(model).copied()
    }

    async fn ensure_loaded(&self, _model: &str) -> Result<String, ToolCtxError> {
        self.ensure_loaded_q
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Ok("loaded".into()))
    }

    fn loaded_port(&self, model: &str) -> Option<u16> {
        self.ports.get(model).copied()
    }

    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String, ToolCtxError> {
        self.generate_calls.lock().unwrap().push((
            model.to_string(),
            prompt.to_string(),
            max_tokens,
            temperature,
        ));
        self.generate_q.lock().unwrap().pop_front().unwrap_or_else(|| {
            Err(ToolCtxError::Generate(
                "FakeCtx: no scripted generate response left".into(),
            ))
        })
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, ToolCtxError> {
        self.embed_q.lock().unwrap().pop_front().unwrap_or_else(|| {
            Err(ToolCtxError::Embed(
                "FakeCtx: no scripted embed response left".into(),
            ))
        })
    }
}
