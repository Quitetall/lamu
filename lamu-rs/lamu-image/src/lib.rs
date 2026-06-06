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
mod comfyui;
pub use comfyui::ComfyUIBackend;

/// Register this module's backend(s) into lamu-core (ADR 0023). Call ONCE at the
/// composition root (binary startup) before serving, so `make_backend` can
/// resolve `backend_kind = "comfyui"`. Idempotent (re-registering overwrites).
pub fn register() {
    // `entry` is ignored: ComfyUI is one sidecar process; model/checkpoint
    // selection happens at the workflow level in the generate_image tool, not at
    // backend construction. A future per-model variant would read `entry` here.
    lamu_core::backends::register_backend("comfyui", |_entry| {
        Ok(Box::new(comfyui::ComfyUIBackend::new()?) as Box<dyn lamu_core::backends::Backend>)
    });
}
