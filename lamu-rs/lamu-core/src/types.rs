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
    /// Operator-curated tag. Renders as a glyph in `list_models`:
    /// ★ for Recommended, ⚙ for Utility, ⊘ for Deprecated; nothing
    /// when Unspecified.
    #[serde(default, deserialize_with = "ModelStatus::deserialize_lenient")]
    pub status: ModelStatus,
}

/// Operator-curated status tag for a model. Stored in the YAML registry
/// as a lowercase string — `recommended` / `utility` / `deprecated` —
/// or omitted entirely when unspecified. The custom deserializer also
/// accepts an empty string as `Unspecified` for backward compat with
/// pre-Phase-5.2 YAMLs that wrote `status: ""`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    #[default]
    #[serde(rename = "")]
    Unspecified,
    Recommended,
    Utility,
    Deprecated,
}

impl ModelStatus {
    pub fn is_unspecified(&self) -> bool {
        matches!(self, ModelStatus::Unspecified)
    }

    /// Glyph for the dashboard column. Empty string when unspecified
    /// so unflagged models don't print a stray double space.
    pub fn glyph(&self) -> &'static str {
        match self {
            ModelStatus::Recommended => "★ ",
            ModelStatus::Utility => "⚙ ",
            ModelStatus::Deprecated => "⊘ ",
            ModelStatus::Unspecified => "",
        }
    }

    /// Lenient deserializer: accepts the canonical lowercase strings
    /// ("recommended", "utility", "deprecated"), the empty string, or
    /// a missing field. Unknown values produce a clear error so typos
    /// like "recomended" surface visibly instead of silently mapping
    /// to Unspecified.
    pub fn deserialize_lenient<'de, D>(d: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw: Option<String> = Option::deserialize(d)?;
        match raw.as_deref() {
            None | Some("") => Ok(ModelStatus::Unspecified),
            Some("recommended") => Ok(ModelStatus::Recommended),
            Some("utility") => Ok(ModelStatus::Utility),
            Some("deprecated") => Ok(ModelStatus::Deprecated),
            Some(other) => Err(serde::de::Error::custom(format!(
                "invalid status '{}' — expected one of: recommended, utility, deprecated (or omit the field)",
                other
            ))),
        }
    }
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

    // Anchor: status field deserialization. The lenient deserializer
    // must accept missing field, empty string, and the three canonical
    // values, and reject typos visibly.
    #[derive(Debug, serde::Deserialize)]
    struct StatusOnly {
        #[serde(default, deserialize_with = "ModelStatus::deserialize_lenient")]
        status: ModelStatus,
    }

    #[test]
    fn status_missing_field_is_unspecified() {
        let s: StatusOnly = serde_yaml::from_str("{}").unwrap();
        assert_eq!(s.status, ModelStatus::Unspecified);
    }

    #[test]
    fn status_empty_string_is_unspecified() {
        let s: StatusOnly = serde_yaml::from_str("status: ''").unwrap();
        assert_eq!(s.status, ModelStatus::Unspecified);
    }

    #[test]
    fn status_canonical_variants_parse() {
        let r: StatusOnly = serde_yaml::from_str("status: recommended").unwrap();
        assert_eq!(r.status, ModelStatus::Recommended);
        let u: StatusOnly = serde_yaml::from_str("status: utility").unwrap();
        assert_eq!(u.status, ModelStatus::Utility);
        let d: StatusOnly = serde_yaml::from_str("status: deprecated").unwrap();
        assert_eq!(d.status, ModelStatus::Deprecated);
    }

    #[test]
    fn status_typo_rejected_visibly() {
        let r = serde_yaml::from_str::<StatusOnly>("status: recomended");
        assert!(r.is_err(), "typo should fail to parse");
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("invalid status"), "got: {msg}");
        assert!(msg.contains("recomended"), "got: {msg}");
    }

    #[test]
    fn status_glyph_table() {
        assert_eq!(ModelStatus::Recommended.glyph(), "★ ");
        assert_eq!(ModelStatus::Utility.glyph(), "⚙ ");
        assert_eq!(ModelStatus::Deprecated.glyph(), "⊘ ");
        assert_eq!(ModelStatus::Unspecified.glyph(), "");
    }
}
