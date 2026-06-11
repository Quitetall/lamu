//! Module-tool extension seam (ADR 0023). Lets in-repo MODULES (lamu-image,
//! lamu-tts, …) contribute MCP tools WITHOUT depending on lamu-mcp, and keeps
//! lamu-core ignorant of what those tools do.
//!
//! A module ships a [`ModuleTool`] (name + description + schema + an async
//! handler over a `&dyn ToolCtx`) and registers it at the composition root. The
//! MCP server impls [`ToolCtx`] for its server type and, when a tool name isn't
//! a built-in, dispatches here with `self as &dyn ToolCtx`.

use crate::types::Modality;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Mutex, OnceLock};

/// Typed failure at the ToolCtx seam (ADR 0027). Replaces the legacy
/// stringly-typed channel where `ensure_loaded`/`generate` returned plain
/// `String`s and failures were signalled by an `"error:"` prefix — which
/// ~65 call sites had to sniff (inconsistently: lamu-image/tts checked
/// `"error"` without the colon and could false-match prose).
///
/// The variant names WHICH seam operation failed; the payload is the
/// human-readable message WITHOUT any `"error:"` prefix. `Display` prints
/// the bare message — tool handlers compose their own wire string
/// (`format!("error: <step>: {e}")`), so the MCP wire contract (one text
/// string, `isError` inferred from the prefix) is unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCtxError {
    /// `ensure_loaded` failed (scheduler refusal, spawn failure, GPU lock…).
    Load(String),
    /// `generate` failed (routing gate, backend error, empty completion…).
    Generate(String),
    /// `embed` failed (no embedding model, backend/HTTP failure…).
    Embed(String),
}

impl ToolCtxError {
    /// The bare message (no prefix, no variant tag).
    pub fn message(&self) -> &str {
        match self {
            ToolCtxError::Load(m) | ToolCtxError::Generate(m) | ToolCtxError::Embed(m) => m,
        }
    }

    /// Strip the legacy `"error:"` wire prefix (case-insensitive, tolerant
    /// of leading whitespace) from a handler string. Bridge for impls that
    /// wrap legacy handlers still returning prefixed strings.
    pub fn strip_wire_prefix(s: &str) -> &str {
        let t = s.trim_start();
        // 6 == len("error:"); ASCII, so the byte slice is char-safe here.
        if t.len() >= 6 && t[..6].eq_ignore_ascii_case("error:") {
            t[6..].trim_start()
        } else {
            s
        }
    }
}

impl std::fmt::Display for ToolCtxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for ToolCtxError {}

/// True when a legacy handler string signals failure on the MCP wire
/// (`"error:"` prefix, tolerant of leading whitespace + case). The single
/// shared sniffer — impls bridging legacy handlers use this instead of
/// hand-rolled (and historically inconsistent) `starts_with` checks.
pub fn is_wire_error(s: &str) -> bool {
    // get(..6) is None on short input or a non-char boundary — both "no".
    s.trim_start()
        .get(..6)
        .is_some_and(|h| h.eq_ignore_ascii_case("error:"))
}

/// Defaults a `ToolCtx::generate` impl MUST apply when the caller passes
/// `None` — sized for short summaries. One source of truth so a second
/// impl can't drift from the trait docs.
pub const GENERATE_DEFAULT_MAX_TOKENS: u32 = 1200;
pub const GENERATE_DEFAULT_TEMPERATURE: f32 = 0.3;

/// What a module tool needs from the running server, abstracted so the tool
/// (e.g. `generate_image`) lives in its module crate rather than lamu-mcp.
#[async_trait::async_trait]
pub trait ToolCtx: Send + Sync {
    /// Modality of a registered model, if known.
    fn model_modality(&self, model: &str) -> Option<Modality>;
    /// Ensure `model` is loaded (spawn + evict via the scheduler). `Ok` is
    /// the handler's status string; `Err(ToolCtxError::Load)` carries the
    /// failure message (ADR 0027 — no more "error:" sniffing).
    async fn ensure_loaded(&self, model: &str) -> Result<String, ToolCtxError>;
    /// Bound port of `model` if currently loaded with a live port.
    fn loaded_port(&self, model: &str) -> Option<u16>;
    /// Generate a completion from `model` (a local registry model OR a cloud
    /// model), honoring routing mode + the scheduler. `Ok` is the completion
    /// text; `Err(ToolCtxError::Generate)` carries the failure message
    /// (ADR 0027). Lets a module (e.g. lamu-jart) summarize/generate
    /// in-process instead of a self-HTTP round-trip.
    ///
    /// `max_tokens`/`temperature`: `None` keeps the summarization defaults
    /// ([`GENERATE_DEFAULT_MAX_TOKENS`] / [`GENERATE_DEFAULT_TEMPERATURE`]).
    /// Long-form callers (deep_research synthesis, grounded answers) pass
    /// `Some` so syntheses stop truncating at the short-summary budget.
    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String, ToolCtxError>;
    /// Embed `texts` via the registry's embedding-capable model (e.g. nomic,
    /// 768-dim). Ensure-loads it and returns one vector per input (same
    /// order). `Err(ToolCtxError::Embed)` on no embedding model / backend
    /// failure. Lets a module (e.g. lamu-jart RAG) rank/retrieve without a
    /// self-HTTP round-trip.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ToolCtxError>;
}

/// Async handler for a module tool. Borrows the ctx for the future's lifetime
/// (mirrors lamu-mcp's `HandlerKind::Stateful`, but over the trait object).
pub type ModuleToolHandler =
    for<'a> fn(&'a dyn ToolCtx, Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>>;

/// A tool contributed by a module (ADR 0023).
pub struct ModuleTool {
    pub name: &'static str,
    pub description: &'static str,
    pub schema_fn: fn() -> Value,
    pub handler: ModuleToolHandler,
    /// True if the tool can reach a cloud provider — so a frontend can apply the
    /// same `local-only` routing gate built-in cloud tools get. A purely-local
    /// tool (e.g. `generate_image` → managed ComfyUI) sets `false`.
    pub cloud: bool,
}

static MODULE_TOOLS: OnceLock<Mutex<Vec<ModuleTool>>> = OnceLock::new();
fn registry() -> &'static Mutex<Vec<ModuleTool>> {
    MODULE_TOOLS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Register a module tool (from a module's `register()` at the composition
/// root). Idempotent by name — a re-register replaces, so repeated
/// composition-root calls never duplicate a tool in `tools/list`.
pub fn register_tool(tool: ModuleTool) {
    let mut t = registry().lock().expect("module-tool registry poisoned");
    if let Some(slot) = t.iter_mut().find(|x| x.name == tool.name) {
        *slot = tool;
    } else {
        t.push(tool);
    }
}

/// Look up a module tool's handler + cloud flag by name (both copied out — fn
/// ptrs and `bool` are `Copy`, so nothing is borrowed across the dispatch
/// `.await`). The `cloud` flag lets the frontend apply its `local-only` gate.
pub fn find_handler(name: &str) -> Option<(ModuleToolHandler, bool)> {
    registry()
        .lock()
        .expect("module-tool registry poisoned")
        .iter()
        .find(|t| t.name == name)
        .map(|t| (t.handler, t.cloud))
}

/// MCP `tools/list` entries for every registered module tool.
pub fn list_entries() -> Vec<Value> {
    registry()
        .lock()
        .expect("module-tool registry poisoned")
        .iter()
        .map(|t| json!({"name": t.name, "description": t.description, "inputSchema": (t.schema_fn)()}))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_wire_error_requires_colon() {
        assert!(is_wire_error("error: boom"));
        assert!(is_wire_error("  ERROR: boom"));
        // No colon → prose, not a failure signal (the lamu-image/tts bug
        // this type retires: "Error bars on the chart" is not an error).
        assert!(!is_wire_error("error bars on the chart"));
        assert!(!is_wire_error("loaded ok"));
        assert!(!is_wire_error(""));
    }

    #[test]
    fn strip_wire_prefix_strips_only_the_prefix() {
        assert_eq!(ToolCtxError::strip_wire_prefix("error: boom"), "boom");
        assert_eq!(ToolCtxError::strip_wire_prefix("  Error:  boom"), "boom");
        assert_eq!(ToolCtxError::strip_wire_prefix("no prefix"), "no prefix");
        // Inner occurrences stay (only the leading prefix is wire framing).
        assert_eq!(
            ToolCtxError::strip_wire_prefix("error: load: error: oom"),
            "load: error: oom"
        );
    }

    #[test]
    fn display_is_bare_message() {
        let e = ToolCtxError::Generate("model down".into());
        assert_eq!(e.to_string(), "model down");
        assert_eq!(format!("error: decide step: {e}"), "error: decide step: model down");
        assert_eq!(e.message(), "model down");
    }
}
