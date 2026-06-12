//! lamu-hf — in-process HuggingFace-candle runtime (ADR 0035).
//!
//! Serves safetensors checkpoints of the Llama / Mistral / Qwen2 families
//! through candle, IN-PROCESS, behind lamu-inproc's port shim (ADR 0033)
//! so the existing port-proxy loading architecture works unchanged.
//! Milestones 5a (load/unload/non-stream generate) and 5b (streaming +
//! chat templates + sampling) live here; 5c (BERT-embed archs through
//! candle) and the `hf_py` python escape hatch for exotic archs are
//! follow-ups — see ADR 0035.
//!
//! At the composition root the binary calls `lamu_hf::register()`
//! (feature-gated: lamu-cli's `hf-candle` feature) to install
//! backend_kind "hf_candle" (ADR 0023/0026). GPU inference additionally
//! needs the `cuda` cargo feature (lamu-cli: `hf-candle-cuda`); the
//! default is CPU candle so tests and `--features full` build without
//! nvcc.

mod backend;
mod engine;

pub use backend::HfCandleBackend;
pub use engine::{chatml_prompt, render_chat_template, CandleEngine, SUPPORTED_MODEL_TYPES};

/// Register this module's backend into lamu-core (ADR 0023). Call ONCE at
/// the composition root before serving. Idempotent.
pub fn register() {
    lamu_core::backends::register_backend("hf_candle", |entry| {
        Ok(Box::new(backend::HfCandleBackend::new(entry)) as Box<dyn lamu_core::backends::Backend>)
    });
}

#[cfg(test)]
mod tests {
    use lamu_core::types::BackendType;

    /// Serde-agreement pin (ADR 0026): wire name and kind string both
    /// "hf_candle".
    #[test]
    fn backend_type_hf_candle_serde_roundtrip() {
        assert_eq!(
            serde_json::to_string(&BackendType::HfCandle).unwrap(),
            "\"hf_candle\""
        );
        assert_eq!(
            serde_json::from_str::<BackendType>("\"hf_candle\"").unwrap(),
            BackendType::HfCandle
        );
        assert_eq!(BackendType::HfCandle.as_kind_str(), "hf_candle");
    }

    #[test]
    fn register_resolves_hf_candle_kind() {
        super::register();
        let entry = lamu_core::types::ModelEntry {
            name: "t".into(),
            path: "/m/dir".into(),
            format: lamu_core::types::ModelFormat::Safetensors,
            backend: BackendType::HfCandle,
            backend_kind: None,
            arch: "llama".into(),
            params_b: 7.0,
            quant: "bf16".into(),
            vram_mb: 16000,
            context_max: 8192,
            capabilities: vec![],
            reasoning_marker: None,
            speculative: None,
            sampling: None,
            pinned: false,
            main: false,
            notes: String::new(),
            status: Default::default(),
            modality: Default::default(),
            system_prompt: None,
        };
        assert!(
            lamu_core::backends::make_backend(&entry).is_ok(),
            "hf_candle kind must resolve after register()"
        );
    }
}
