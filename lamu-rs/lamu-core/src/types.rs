//! Core types — direct ports of `lamu/core/types.py`.
//!
//! All types should be `Debug + Clone + Serialize + Deserialize` where
//! possible. Use `#[non_exhaustive]` only for stable API guarantees.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

/// Stable capability vocabulary. Resist expanding without clear use case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Chat,
    Code,
    Reasoning,
    Routing,
    Vision,
    LongContext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    Safetensors,
    Onnx,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    LlamaCpp,
    Megakernel,
    Dflash,
    DflashLucebox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    Unloaded,
    Loading,
    Loaded,
    Error,
}

/// How a model family marks reasoning content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningMarker {
    pub open_tag: String,
    pub close_tag: String,
    pub family: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    pub draft_path: PathBuf,
    pub method: String,
    #[serde(default = "default_draft_max")]
    pub draft_max: u32,
}

fn default_draft_max() -> u32 { 8 }

/// A discovered model on disk. Immutable after scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub format: ModelFormat,
    pub backend: BackendType,
    pub arch: String,
    pub params_b: f32,
    pub quant: String,
    pub vram_mb: u32,
    pub context_max: u32,
    pub capabilities: Vec<Capability>,
    #[serde(default)]
    pub reasoning_marker: Option<ReasoningMarker>,
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    pub pinned: bool,
}

/// Runtime state for a currently loaded model.
#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub entry: ModelEntry,
    pub state: ModelState,
    pub pid: Option<u32>,
    pub port: u16,
    pub vram_actual_mb: u32,
    pub last_used: Instant,
}

/// Result of router's model selection. Used by plan_query dry-run.
#[derive(Debug, Clone, Serialize)]
pub struct RouteDecision {
    pub model_name: String,
    pub reason: String,
    pub loaded: bool,
    pub would_evict: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub enum StreamChunk {
    Reasoning(String),
    Content(String),
}

/// Performance stats for a query.
#[derive(Debug, Clone, Serialize)]
pub struct QueryStats {
    pub latency_ms: f64,
    pub time_to_first_token_ms: f64,
    pub tokens_generated: u32,
    pub tokens_per_second: f64,
    pub prompt_tokens: u32,
    pub retries: u32,
    pub stream_chunks: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub content: String,
    pub reasoning: Option<String>,
    pub model_used: String,
    pub stats: QueryStats,
    pub finish_reason: String,
}

/// Snapshot of VRAM allocation.
#[derive(Debug, Clone, Serialize)]
pub struct VramBudget {
    pub total_mb: u32,
    pub used_mb: u32,
    pub free_mb: u32,
    pub loaded_models: Vec<(String, u32)>,
    pub available_mb: u32,
}
