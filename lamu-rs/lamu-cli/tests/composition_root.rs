//! Composition-root drift test (ADR 0026).
//!
//! The ONLY mechanism preventing "added a `BackendType` variant / module
//! backend but forgot to register it at the composition root" from becoming
//! a runtime 'not registered' error in production. Mirrors main()'s
//! register() calls, then proves every backend kind the registry can name
//! actually resolves through `make_backend` — i.e. it never dies with the
//! ADR 0023 "not registered" error. (Construction may still fail for
//! environment reasons — missing python checkpoints etc. — which is fine;
//! the drift error is the one thing asserted against.)

use lamu_core::backends::make_backend;
use lamu_core::types::{BackendType, ModelEntry, ModelFormat};

fn entry_for(kind: &str) -> ModelEntry {
    ModelEntry {
        name: format!("drift-{kind}"),
        path: "/tmp/drift-test.gguf".into(),
        format: ModelFormat::Gguf,
        backend: BackendType::LlamaCpp, // ignored: backend_kind wins dispatch
        backend_kind: Some(kind.to_string()),
        arch: "qwen3".into(),
        params_b: 7.0,
        quant: "Q4_K_M".into(),
        vram_mb: 8000,
        context_max: 32768,
        capabilities: vec![],
        reasoning_marker: None,
        speculative: None,
        sampling: None,
        pinned: false,
        main: false,
        notes: String::new(),
        status: Default::default(),
        system_prompt: None,
        modality: Default::default(),
    }
}

#[test]
fn every_backend_kind_resolves_at_the_composition_root() {
    // Same registrations as lamu-cli main() — THE composition root. If a
    // register() call is added in main() but not here (or vice versa), this
    // test's coverage drifts; keep the two lists in lockstep.
    // Coverage boundary: this iterates the BackendType variants. lamu-jart
    // registers module TOOLS only (no register_backend call); if a module
    // ever registers a kind with no enum variant, add its string here.
    lamu_image::register();
    lamu_tts::register();
    lamu_jart::register();
    // Same cfgs as main(): feature-gated modules exist only when compiled.
    #[cfg(feature = "onnx")]
    lamu_onnx::register();
    #[cfg(feature = "hf-candle")]
    lamu_hf::register();

    let mut kinds = vec![
        BackendType::LlamaCpp,
        BackendType::Megakernel,
        BackendType::Dflash,
        BackendType::DflashLucebox,
        BackendType::FishSpeech,
        BackendType::ComfyUI,
    ];
    // BackendType::Onnx / BackendType::HfCandle resolve only when their
    // feature compiled the module in; the feature-OFF expectations are
    // pinned by the `*_not_registered_without_feature` tests below.
    #[cfg(feature = "onnx")]
    kinds.push(BackendType::Onnx);
    #[cfg(feature = "hf-candle")]
    kinds.push(BackendType::HfCandle);
    for bt in kinds {
        let kind = bt.as_kind_str();
        match make_backend(&entry_for(kind)) {
            Ok(_) => {}
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    !msg.contains("not registered"),
                    "backend kind '{kind}' did not resolve at the composition root: {msg}"
                );
            }
        }
    }
}

#[test]
fn unknown_kind_still_fails_loudly_after_registration() {
    lamu_image::register();
    lamu_tts::register();
    match make_backend(&entry_for("nope_not_real")) {
        Ok(_) => panic!("unknown kind must error"),
        Err(e) => assert!(format!("{e}").contains("not registered")),
    }
}

/// Feature-OFF half of the onnx drift contract (ADR 0034): a build without
/// `--features onnx` never registers the module, so an "onnx" entry must
/// die with the ADR 0023 "not registered" error — and that error must point
/// the operator at the feature flag instead of leaving them guessing.
#[cfg(not(feature = "onnx"))]
#[test]
fn onnx_kind_not_registered_without_feature() {
    lamu_image::register();
    lamu_tts::register();
    lamu_jart::register();
    match make_backend(&entry_for(BackendType::Onnx.as_kind_str())) {
        Ok(_) => panic!("'onnx' must not resolve when the feature is off"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("not registered"), "got: {msg}");
            assert!(
                msg.contains("--features"),
                "error must hint at the cargo feature gate: {msg}"
            );
        }
    }
}

/// Feature-OFF half of the hf-candle drift contract (ADR 0035) — same
/// shape as the onnx one above.
#[cfg(not(feature = "hf-candle"))]
#[test]
fn hf_candle_kind_not_registered_without_feature() {
    lamu_image::register();
    lamu_tts::register();
    lamu_jart::register();
    match make_backend(&entry_for(BackendType::HfCandle.as_kind_str())) {
        Ok(_) => panic!("'hf_candle' must not resolve when the feature is off"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("not registered"), "got: {msg}");
            assert!(
                msg.contains("--features"),
                "error must hint at the cargo feature gate: {msg}"
            );
        }
    }
}
