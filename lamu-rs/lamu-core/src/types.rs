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
    /// Embedding model (served via llama-server `--embedding`). Carries
    /// only this cap (no Chat) so it's never chat-routed; backs the
    /// `/v1/embeddings` endpoint.
    Embedding,
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
    /// Local fish-speech (OpenAudio S2-Pro) TTS server. Non-LLM modality.
    FishSpeech,
    /// Local ComfyUI image-generation server. Non-LLM modality.
    #[serde(rename = "comfyui")]
    ComfyUI,
}

impl BackendType {
    /// Canonical string key for this backend kind — MUST stay identical to
    /// the serde wire name (ADR 0026). `make_backend` dispatches on this
    /// string, so the enum and the string-keyed module registry share one
    /// namespace. A unit test pins serde-name ↔ this-method agreement;
    /// adding a variant without updating both is a compile/test failure,
    /// not silent drift.
    pub fn as_kind_str(&self) -> &'static str {
        match self {
            BackendType::LlamaCpp => "llama_cpp",
            BackendType::Megakernel => "megakernel",
            BackendType::Dflash => "dflash",
            BackendType::DflashLucebox => "dflash_lucebox",
            BackendType::FishSpeech => "fish_speech",
            BackendType::ComfyUI => "comfyui",
        }
    }
}

/// Model modality. `Default == Llm` keeps every existing models.yaml valid
/// (an entry with no `modality:` key deserializes to `Llm`). Drives
/// modality-aware VRAM eviction (evict image/tts before LLMs) and
/// `text_to_speech` local-vs-cloud routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    #[default]
    Llm,
    Image,
    Tts,
}

impl Modality {
    pub fn is_llm(&self) -> bool {
        matches!(self, Modality::Llm)
    }
}

/// Where a loaded model physically lives across the managed GPU pool
/// (ADR 0017 P2). `Single(idx)` pins the whole model to one NVML device
/// index — the only variant the backends thread through today. `Sharded`
/// reserves the wire format for tensor/layer-split across several devices;
/// P2 records it but backends treat it as Single-on-first-index (TODO:
/// real multi-device split needs `--tensor-split` / `--split-mode`).
///
/// `Default == Single(0)` so every existing `LoadedModel` / serialized
/// blob with no `device:` key deserializes to "device 0", byte-identical
/// to the pre-P2 single-GPU path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevicePlacement {
    Single(u32),
    Sharded(Vec<u32>),
}

impl Default for DevicePlacement {
    fn default() -> Self {
        DevicePlacement::Single(0)
    }
}

impl DevicePlacement {
    /// The NVML index a single-device spawn should target. For `Single`
    /// it's the index; for `Sharded` it's the first member (P2 placeholder
    /// until real sharding lands). Empty `Sharded` → 0.
    pub fn primary_index(&self) -> u32 {
        match self {
            DevicePlacement::Single(i) => *i,
            DevicePlacement::Sharded(v) => v.first().copied().unwrap_or(0),
        }
    }
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

/// Optional per-model sampling overrides. Any omitted field falls back
/// to the caller-supplied value (or the builtin backend default) at
/// request time — see the `effective_*` merge helpers. A profile with
/// no fields set serializes to an empty map (every `Option` field is
/// `skip_serializing_if` when `None`, and `lock` is omitted when false),
/// so absent fields round-trip cleanly and existing registries that
/// carry no `sampling:` key deserialize to `None`.
///
/// `lock` flips the precedence: when set, a profile field that is `Some`
/// OVERRIDES the caller's request value instead of merely filling in for
/// an omitted one. This lets an operator pin, e.g., a deterministic
/// `temperature: 0.0` that clients cannot override.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// When true, any field set on this profile overrides the caller's
    /// request value (operator-pinned sampling). When false (default),
    /// the profile only fills fields the caller omitted.
    #[serde(default, skip_serializing_if = "is_false")]
    pub lock: bool,
}

fn is_false(b: &bool) -> bool { !*b }

/// Total, side-effect-free precedence resolver shared by every typed
/// `effective_*` helper:
///   1. profile.lock AND profile field is Some  => profile value (pin)
///   2. else request value if Some               => caller wins
///   3. else profile value if Some               => profile fills gap
///   4. else builtin default
fn merge_sampler<T: Copy>(lock: bool, profile: Option<T>, req: Option<T>, default: T) -> T {
    if lock {
        if let Some(p) = profile {
            return p;
        }
    }
    req.or(profile).unwrap_or(default)
}

impl SamplingProfile {
    pub fn temperature(&self, req: Option<f32>, default: f32) -> f32 {
        merge_sampler(self.lock, self.temperature, req, default)
    }

    pub fn top_p(&self, req: Option<f32>, default: f32) -> f32 {
        merge_sampler(self.lock, self.top_p, req, default)
    }

    pub fn top_k(&self, req: Option<u32>, default: u32) -> u32 {
        merge_sampler(self.lock, self.top_k, req, default)
    }

    pub fn min_p(&self, req: Option<f32>, default: f32) -> f32 {
        merge_sampler(self.lock, self.min_p, req, default)
    }

    pub fn repeat_penalty(&self, req: Option<f32>, default: f32) -> f32 {
        merge_sampler(self.lock, self.repeat_penalty, req, default)
    }

    pub fn max_tokens(&self, req: Option<u32>, default: u32) -> u32 {
        merge_sampler(self.lock, self.max_tokens, req, default)
    }

    /// Resolve a single optional sampler field WITHOUT collapsing to a
    /// builtin default — used by request-build sites that only want to
    /// emit a field downstream when the merged result is actually set
    /// (profile or caller supplied it). Honors the same lock precedence.
    pub fn resolve_temperature(&self, req: Option<f32>) -> Option<f32> {
        merge_opt(self.lock, self.temperature, req)
    }
    pub fn resolve_top_p(&self, req: Option<f32>) -> Option<f32> {
        merge_opt(self.lock, self.top_p, req)
    }
    pub fn resolve_top_k(&self, req: Option<u32>) -> Option<u32> {
        merge_opt(self.lock, self.top_k, req)
    }
    pub fn resolve_min_p(&self, req: Option<f32>) -> Option<f32> {
        merge_opt(self.lock, self.min_p, req)
    }
    pub fn resolve_repeat_penalty(&self, req: Option<f32>) -> Option<f32> {
        merge_opt(self.lock, self.repeat_penalty, req)
    }
    pub fn resolve_max_tokens(&self, req: Option<u32>) -> Option<u32> {
        merge_opt(self.lock, self.max_tokens, req)
    }
}

/// Like `merge_sampler` but yields `None` when neither profile nor
/// request supplies a value, so the caller can decide the fallback
/// (or omit the field downstream). Same lock precedence.
fn merge_opt<T: Copy>(lock: bool, profile: Option<T>, req: Option<T>) -> Option<T> {
    if lock && profile.is_some() {
        return profile;
    }
    req.or(profile)
}

/// Free-function convenience: resolve the four sampler fields against an
/// optional profile, returning `Option`s suitable for conditional
/// payload injection. `None` profile behaves as "no overrides".
pub fn resolve_samplers(
    profile: Option<&SamplingProfile>,
    req_temperature: Option<f32>,
    req_top_p: Option<f32>,
    req_top_k: Option<u32>,
    req_min_p: Option<f32>,
    req_repeat_penalty: Option<f32>,
    req_max_tokens: Option<u32>,
) -> ResolvedSamplers {
    match profile {
        Some(p) => ResolvedSamplers {
            temperature: p.resolve_temperature(req_temperature),
            top_p: p.resolve_top_p(req_top_p),
            top_k: p.resolve_top_k(req_top_k),
            min_p: p.resolve_min_p(req_min_p),
            repeat_penalty: p.resolve_repeat_penalty(req_repeat_penalty),
            max_tokens: p.resolve_max_tokens(req_max_tokens),
        },
        None => ResolvedSamplers {
            temperature: req_temperature,
            top_p: req_top_p,
            top_k: req_top_k,
            min_p: req_min_p,
            repeat_penalty: req_repeat_penalty,
            max_tokens: req_max_tokens,
        },
    }
}

/// Result of merging a request's sampler values against a per-model
/// profile. Each field is `None` when neither side supplied it; the
/// caller applies its own builtin default for those.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedSamplers {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
}

/// A discovered model on disk. Immutable after scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub format: ModelFormat,
    pub backend: BackendType,
    /// Optional string dispatch key (ADR 0026). When set, `make_backend`
    /// routes by THIS string (module-registry kinds included) and the
    /// `backend` enum is ignored for dispatch — operator-curated entries
    /// can target module backends core never names. Absent (every
    /// pre-0026 registry) → enum dispatch, byte-identical behavior.
    /// Preserved across `lamu scan` like other curated fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_kind: Option<String>,
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
    /// Optional per-model sampling overrides (temperature/top_p/top_k/
    /// min_p/repeat_penalty/max_tokens + a `lock` flag). Merged into
    /// requests at the API/MCP layer. Absent in existing registries →
    /// `None` (no overrides).
    #[serde(default)]
    pub sampling: Option<SamplingProfile>,
    #[serde(default)]
    pub pinned: bool,
    /// Designate this entry as the default model when external harnesses
    /// (Claude Code, Codex, Cursor, etc.) call /v1/chat/completions
    /// without a `model` field, or with the alias `default`/`main`/`lamu`.
    /// Exactly one entry should set `main: true`. First match wins.
    #[serde(default)]
    pub main: bool,
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
    /// Model modality (llm/image/tts). Absent in existing registries → Llm.
    /// Drives VRAM eviction tiering + text_to_speech routing.
    #[serde(default)]
    pub modality: Modality,
    /// Optional per-model system prompt. Precedence on the chat path:
    /// a request's own system message wins outright; absent that, this
    /// value (when set) replaces the global default
    /// (~/.config/lamu/system_prompt.txt / built-in grounding prompt).
    /// A blank/whitespace value explicitly disables ANY default prompt
    /// for this model (mirrors the global file's blank-disables rule).
    /// Curated: preserved across `lamu scan` like sampling/notes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
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
    /// GPU placement the scheduler chose (ADR 0017 P2) — NVML index
    /// via `DevicePlacement`. `Default` is `Single(0)`, so the
    /// single-GPU path is unchanged. Read by the loader to call
    /// `Backend::set_device` before spawn.
    pub device: DevicePlacement,
    /// The booted context window (ADR 0021): `effective_ctx_size(context_max)`
    /// captured at load time, i.e. the `--ctx-size` the backend actually
    /// spawned with. The occupancy denominator reads THIS instead of
    /// re-deriving from `LAMU_DEFAULT_CTX` per request, so it can't drift if
    /// the env changes after spawn.
    pub booted_ctx: u32,
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

/// Per-GPU VRAM snapshot (multi-GPU, ADR 0017). `available_mb` already nets
/// out the per-device reserve + the larger of registered/NVML-actual usage.
#[derive(Debug, Clone, Serialize)]
pub struct DeviceVram {
    pub index: u32,
    pub name: String,
    pub total_mb: u32,
    pub used_mb: u32,
    pub available_mb: u32,
}

/// Snapshot of VRAM allocation. Scalar fields are aggregates across all
/// devices (single-GPU → the one device); `per_device` is the breakdown.
#[derive(Debug, Clone, Serialize)]
pub struct VramBudget {
    pub total_mb: u32,
    pub used_mb: u32,
    pub free_mb: u32,
    pub loaded_models: Vec<(String, u32)>,
    pub available_mb: u32,
    /// Per-GPU breakdown (ADR 0017). Empty on the pre-multi-GPU path.
    #[serde(default)]
    pub per_device: Vec<DeviceVram>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_placement_default_is_single_zero() {
        assert_eq!(DevicePlacement::default(), DevicePlacement::Single(0));
    }

    #[test]
    fn device_placement_primary_index() {
        assert_eq!(DevicePlacement::Single(3).primary_index(), 3);
        assert_eq!(DevicePlacement::Sharded(vec![2, 5, 7]).primary_index(), 2);
        assert_eq!(DevicePlacement::Sharded(vec![]).primary_index(), 0);
    }

    #[test]
    fn device_placement_serde_round_trips() {
        let s = DevicePlacement::Single(1);
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(j, r#"{"single":1}"#);
        assert_eq!(serde_json::from_str::<DevicePlacement>(&j).unwrap(), s);
        let sh = DevicePlacement::Sharded(vec![0, 1]);
        let jj = serde_json::to_string(&sh).unwrap();
        assert_eq!(serde_json::from_str::<DevicePlacement>(&jj).unwrap(), sh);
    }

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

    // ── SamplingProfile: serde + merge precedence + lock matrix ─────

    fn empty_profile() -> SamplingProfile {
        SamplingProfile {
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            max_tokens: None,
            lock: false,
        }
    }

    #[test]
    fn sampling_profile_empty_yaml_parses() {
        // A profile with no fields → all None, lock false.
        let p: SamplingProfile = serde_yaml::from_str("{}").unwrap();
        assert_eq!(p, empty_profile());
    }

    #[test]
    fn sampling_profile_partial_yaml_round_trips() {
        // Only temperature set; the rest stay None and `lock` defaults
        // false. Re-serializing must omit every absent field.
        let yaml = "temperature: 0.2\n";
        let p: SamplingProfile = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(p.temperature, Some(0.2));
        assert_eq!(p.top_p, None);
        assert!(!p.lock);
        let back = serde_yaml::to_string(&p).unwrap();
        assert_eq!(back.trim(), "temperature: 0.2");
    }

    #[test]
    fn sampling_profile_empty_serializes_empty() {
        let back = serde_yaml::to_string(&empty_profile()).unwrap();
        assert_eq!(back.trim(), "{}");
    }

    #[test]
    fn resolve_samplers_unlocked_profile_fills_and_yields() {
        // UNLOCKED profile (lock=false) — the previously-untested branch:
        //   row 2 (request wins over an unlocked profile field), and
        //   row 3 (profile fills a caller-omitted field).
        let p = SamplingProfile {
            temperature: Some(0.2),   // caller overrides this below
            top_k: Some(40),          // caller omits → profile fills
            top_p: None,
            min_p: None,
            repeat_penalty: None,
            max_tokens: None,
            lock: false,
        };
        let r = resolve_samplers(
            Some(&p),
            Some(0.9), // req temperature — must win (unlocked)
            None,      // req top_p
            None,      // req top_k — profile must fill
            None, None, None,
        );
        assert_eq!(r.temperature, Some(0.9), "unlocked: caller value wins");
        assert_eq!(r.top_k, Some(40), "profile fills caller-omitted field");
        assert_eq!(r.top_p, None, "neither side set → stays None (no builtin)");
        assert_eq!(r.max_tokens, None);
    }

    #[test]
    fn sampling_profile_lock_serializes_when_true() {
        let mut p = empty_profile();
        p.lock = true;
        p.temperature = Some(0.0);
        let back = serde_yaml::to_string(&p).unwrap();
        assert!(back.contains("lock: true"), "got: {back}");
        assert!(back.contains("temperature: 0.0"), "got: {back}");
    }

    #[test]
    fn model_entry_no_sampling_key_is_none() {
        // Backward compat: a registry-proxy-shaped YAML with no
        // `sampling:` key deserializes the field to None.
        let yaml = r#"
name: m
path: /tmp/m.gguf
format: gguf
backend: llama_cpp
arch: qwen3
params_b: 7.0
quant: Q4_K_M
vram_mb: 8000
context_max: 32768
capabilities: [chat]
"#;
        let e: ModelEntry = serde_yaml::from_str(yaml).unwrap();
        assert!(e.sampling.is_none());
    }

    #[test]
    fn modality_defaults_to_llm_when_absent() {
        // Back-compat: a YAML with no `modality:` key → Llm.
        let yaml = r#"
name: m
path: /tmp/m.gguf
format: gguf
backend: llama_cpp
arch: qwen3
params_b: 7.0
quant: Q4_K_M
vram_mb: 8000
context_max: 32768
capabilities: [chat]
"#;
        let e: ModelEntry = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(e.modality, Modality::Llm);
        assert!(e.modality.is_llm());
    }

    #[test]
    fn modality_tts_parses_and_round_trips() {
        let yaml = r#"
name: t
path: /tmp/s2-pro
format: gguf
backend: llama_cpp
arch: fish
params_b: 0.5
quant: fp16
vram_mb: 16000
context_max: 0
capabilities: []
modality: tts
"#;
        let e: ModelEntry = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(e.modality, Modality::Tts);
        assert!(!e.modality.is_llm());
        let s = serde_yaml::to_string(&e).unwrap();
        assert!(s.contains("modality: tts"), "serialized: {s}");
    }

    #[test]
    fn model_entry_with_sampling_round_trips() {
        let yaml = r#"
name: m
path: /tmp/m.gguf
format: gguf
backend: llama_cpp
arch: qwen3
params_b: 7.0
quant: Q4_K_M
vram_mb: 8000
context_max: 32768
capabilities: [chat]
sampling:
  temperature: 0.3
  top_p: 0.9
  lock: true
"#;
        let e: ModelEntry = serde_yaml::from_str(yaml).unwrap();
        let s = e.sampling.expect("sampling present");
        assert_eq!(s.temperature, Some(0.3));
        assert_eq!(s.top_p, Some(0.9));
        assert_eq!(s.top_k, None);
        assert!(s.lock);
    }

    #[test]
    fn merge_unlocked_request_wins_over_profile() {
        let mut p = empty_profile();
        p.temperature = Some(0.1);
        // caller supplied 0.9 → caller wins (unlocked).
        assert_eq!(p.temperature(Some(0.9), 0.7), 0.9);
    }

    #[test]
    fn merge_unlocked_profile_fills_omitted() {
        let mut p = empty_profile();
        p.temperature = Some(0.1);
        // caller omitted → profile fills.
        assert_eq!(p.temperature(None, 0.7), 0.1);
    }

    #[test]
    fn merge_unlocked_default_when_neither() {
        let p = empty_profile();
        // neither profile nor caller → builtin default.
        assert_eq!(p.temperature(None, 0.7), 0.7);
        assert_eq!(p.max_tokens(None, 16384), 16384);
    }

    #[test]
    fn merge_locked_profile_overrides_request() {
        let mut p = empty_profile();
        p.lock = true;
        p.temperature = Some(0.0);
        // locked + profile Some → profile overrides caller's 0.9.
        assert_eq!(p.temperature(Some(0.9), 0.7), 0.0);
    }

    #[test]
    fn merge_locked_but_field_unset_falls_back_to_request() {
        let mut p = empty_profile();
        p.lock = true;
        // lock is on but THIS field unset → caller still wins.
        assert_eq!(p.top_p(Some(0.5), 1.0), 0.5);
        // and falls to default when caller also omits.
        assert_eq!(p.top_p(None, 1.0), 1.0);
    }

    #[test]
    fn resolve_opt_keeps_none_when_neither_supplies() {
        let p = empty_profile();
        assert_eq!(p.resolve_top_k(None), None);
        assert_eq!(p.resolve_temperature(Some(0.4)), Some(0.4));
    }

    #[test]
    fn resolve_samplers_none_profile_passthrough() {
        let r = resolve_samplers(None, Some(0.5), None, Some(40), None, None, Some(256));
        assert_eq!(r.temperature, Some(0.5));
        assert_eq!(r.top_k, Some(40));
        assert_eq!(r.max_tokens, Some(256));
        assert_eq!(r.top_p, None);
    }

    #[test]
    fn resolve_samplers_locked_profile_overrides() {
        let mut p = empty_profile();
        p.lock = true;
        p.temperature = Some(0.0);
        p.max_tokens = Some(4096);
        let r = resolve_samplers(Some(&p), Some(0.9), None, None, None, None, Some(99));
        assert_eq!(r.temperature, Some(0.0)); // locked override
        assert_eq!(r.max_tokens, Some(4096)); // locked override
        assert_eq!(r.top_k, None); // unset everywhere
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
