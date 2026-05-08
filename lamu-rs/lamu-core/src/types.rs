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
    /// Free-text description shown in `mcp__local-llm__list_models`
    /// output. Used by humans (and orchestrating agents) to pick the
    /// right model for a task. (TUI dashboard rendering is a TODO —
    /// currently MCP-only.)
    #[serde(default)]
    pub notes: String,
    /// One of: "recommended", "utility", "deprecated", or "" (default).
    /// Renders as a glyph in `list_models`: ★ ⚙ ⊘ respectively.
    #[serde(default)]
    pub status: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_round_trips_through_serde() {
        let cap = Capability::LongContext;
        let s = serde_json::to_string(&cap).unwrap();
        assert_eq!(s, "\"long_context\"");
        let back: Capability = serde_json::from_str(&s).unwrap();
        assert_eq!(back, Capability::LongContext);
    }

    #[test]
    fn backend_type_serde_snake_case() {
        let b = BackendType::DflashLucebox;
        assert_eq!(serde_json::to_string(&b).unwrap(), "\"dflash_lucebox\"");
    }

    #[test]
    fn model_format_serde_lowercase() {
        let f = ModelFormat::Gguf;
        assert_eq!(serde_json::to_string(&f).unwrap(), "\"gguf\"");
    }

    #[test]
    fn model_state_distinct() {
        // distinct variants — guards against accidental dedup
        let states = [
            ModelState::Unloaded, ModelState::Loading,
            ModelState::Loaded, ModelState::Error,
        ];
        for (i, a) in states.iter().enumerate() {
            for (j, b) in states.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn speculative_config_default_draft_max() {
        let yaml = r#"
draft_path: /tmp/d.gguf
method: dflash
"#;
        let cfg: SpeculativeConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.draft_max, 8);
    }

    #[test]
    fn route_decision_serializable() {
        let d = RouteDecision {
            model_name: "x".into(),
            reason: "r".into(),
            loaded: true,
            would_evict: vec!["a".into()],
        };
        let j = serde_json::to_value(&d).unwrap();
        assert_eq!(j["loaded"], true);
        assert_eq!(j["would_evict"][0], "a");
    }
}
