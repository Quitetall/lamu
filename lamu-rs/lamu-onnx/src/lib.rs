//! lamu-onnx — embeddings-first ONNX backend (ADR 0034).
//!
//! Runs sentence-embedding ONNX models (BGE, MiniLM, GTE, …) on the ort
//! CPU execution provider, IN-PROCESS, served through lamu-inproc's port
//! shim (ADR 0033) so the existing port-proxy loading architecture works
//! unchanged. v1 is embeddings-only and CPU-only (vram_mb 0 — invisible
//! to the VRAM scheduler).
//!
//! At the composition root the binary calls `lamu_onnx::register()`
//! (feature-gated: lamu-cli's `onnx` feature) to install backend_kind
//! "onnx" (ADR 0023/0026).

mod backend;
mod engine;

pub use backend::OnnxBackend;
pub use engine::OnnxEmbedEngine;

/// Register this module's backend into lamu-core (ADR 0023). Call ONCE at
/// the composition root before serving. Idempotent.
pub fn register() {
    lamu_core::backends::register_backend("onnx", |entry| {
        Ok(Box::new(backend::OnnxBackend::new(entry)) as Box<dyn lamu_core::backends::Backend>)
    });
}

#[cfg(test)]
mod tests {
    use lamu_core::types::BackendType;

    #[test]
    fn backend_type_onnx_serde_roundtrip() {
        assert_eq!(serde_json::to_string(&BackendType::Onnx).unwrap(), "\"onnx\"");
        assert_eq!(
            serde_json::from_str::<BackendType>("\"onnx\"").unwrap(),
            BackendType::Onnx
        );
    }
}
