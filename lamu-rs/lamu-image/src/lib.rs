//! lamu-image — image-generation MODULE (ADR 0023).
//!
//! Expands the backend with image generation. Owns the ComfyUI backend
//! (currently `lamu-core/src/backends/comfyui.rs`) and the `generate_image`
//! tool (currently `lamu-mcp/src/image.rs`), which move here once the
//! `BackendRegistry` + `ToolCtx` seams exist (ADR 0023 steps 2–3). Depends on
//! `lamu-core`; `lamu-core` does NOT depend on this crate.
//!
//! At the composition root each frontend will call `lamu_image::register(&mut
//! reg)` to install this module's backend factory (kind `comfyui`, eviction
//! tier = media) and its tool definition.
//!
//! STATUS: scaffold (ADR 0023 step 0 — seams first). No backend moved yet so
//! the workspace build + `lamu serve` stay green.

/// Placeholder so the crate has a symbol until the ComfyUI backend lands here.
/// Replaced by `pub fn register(reg: &mut lamu_core::registry::LamuRegistry)`.
pub const MODULE_NAME: &str = "lamu-image";
