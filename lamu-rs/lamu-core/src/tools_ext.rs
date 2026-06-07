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

/// What a module tool needs from the running server, abstracted so the tool
/// (e.g. `generate_image`) lives in its module crate rather than lamu-mcp.
#[async_trait::async_trait]
pub trait ToolCtx: Send + Sync {
    /// Modality of a registered model, if known.
    fn model_modality(&self, model: &str) -> Option<Modality>;
    /// Ensure `model` is loaded (spawn + evict via the scheduler). Returns a
    /// status string; an `"error"`-prefixed string on failure (mirrors the MCP
    /// `load_model` handler).
    async fn ensure_loaded(&self, model: &str) -> String;
    /// Bound port of `model` if currently loaded with a live port.
    fn loaded_port(&self, model: &str) -> Option<u16>;
    /// Generate a completion from `model` (a local registry model OR a cloud
    /// model), honoring routing mode + the scheduler. Returns the completion
    /// text, or an `"error"`-prefixed string on failure (mirrors the MCP
    /// handlers' convention). Lets a module (e.g. lamu-jart) summarize/generate
    /// in-process instead of a self-HTTP round-trip.
    async fn generate(&self, model: &str, prompt: &str) -> String;
    /// Embed `texts` via the registry's embedding-capable model (e.g. nomic,
    /// 768-dim). Ensure-loads it and returns one vector per input (same order).
    /// `Err(msg)` on no embedding model / backend failure. Lets a module
    /// (e.g. lamu-jart RAG) rank/retrieve without a self-HTTP round-trip.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
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
