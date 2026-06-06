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
mod image;
pub use comfyui::ComfyUIBackend;

/// Register this module's backend + MCP tool into lamu-core (ADR 0023). Call
/// ONCE at the composition root (binary startup) before serving. Idempotent.
pub fn register() {
    // Backend: `entry` is ignored — ComfyUI is one sidecar; checkpoint selection
    // happens at the workflow level in generate_image, not at construction.
    lamu_core::backends::register_backend("comfyui", |_entry| {
        Ok(Box::new(comfyui::ComfyUIBackend::new()?) as Box<dyn lamu_core::backends::Backend>)
    });
    // Tool: generate_image now lives here, dispatched over &dyn ToolCtx.
    lamu_core::tools_ext::register_tool(lamu_core::tools_ext::ModuleTool {
        name: "generate_image",
        description: "Generate an image via the local ComfyUI backend (a registry model with modality: image). Spawns/evicts ComfyUI through the scheduler, runs a txt2img workflow, writes a PNG to a confined dir.",
        schema_fn: image::schema_generate_image,
        handler: image::dispatch_generate_image,
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn register_installs_generate_image_tool() {
        super::register();
        assert!(
            lamu_core::tools_ext::find_handler("generate_image").is_some(),
            "register() must install the generate_image tool"
        );
        assert!(
            lamu_core::tools_ext::list_entries().iter().any(|e| e["name"] == "generate_image"),
            "generate_image must appear in tools/list"
        );
    }
}
