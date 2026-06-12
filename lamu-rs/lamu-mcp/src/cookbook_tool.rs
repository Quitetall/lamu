//! `cookbook` MCP tool — hardware-aware model-fit ranking as JSON, so the
//! outer agent can pick a LOCAL model by predicted throughput + VRAM fit
//! instead of guessing. Read-only; same scorer as `lamu cookbook` (ADR 0015).

use lamu_core::cookbook::{self, ModelSpec};
use lamu_core::types::{Capability, Modality, ModelEntry};
use serde_json::{json, Value};

/// Use-case bucket from an entry's modality + capabilities (scoring lens).
fn infer_use_case(e: &ModelEntry) -> String {
    if e.modality == Modality::Tts {
        return "tts".to_string();
    }
    let has = |c: Capability| e.capabilities.contains(&c);
    if has(Capability::Embedding) {
        "embedding".to_string()
    } else if has(Capability::Vision) {
        "multimodal".to_string()
    } else if has(Capability::Code) {
        "coding".to_string()
    } else if has(Capability::Reasoning) {
        "reasoning".to_string()
    } else if has(Capability::Chat) {
        "chat".to_string()
    } else {
        "general".to_string()
    }
}

pub async fn handle_cookbook(args: Value) -> String {
    // Cross-vendor detection (ADR 0015): NVIDIA → AMD → Apple → Intel → CPU.
    let mut hw = cookbook::detect_hardware();
    // simulate_vram (MB) overrides the detected card's VRAM. Zero the RAM-offload
    // budget too, so the simulation is a pure GPU budget (no double-counting on a
    // CPU box where avail_ram was populated).
    if let Some(v) = args["simulate_vram"].as_u64() {
        hw.set_total_vram_gb(v as f32 / 1024.0); // keeps the device list in sync
        hw.avail_ram_gb = 0.0;
    }
    let ctx_override = args["ctx"].as_u64().map(|c| c as u32);

    let entries =
        lamu_core::registry::load_registry(&lamu_core::config::registry_path()).unwrap_or_default();
    let specs: Vec<ModelSpec> = entries
        .iter()
        .filter(|e| e.modality.is_llm() && e.params_b > 0.0)
        .map(|e| {
            // MoE fidelity: A<N>B name marker or *moe* arch → sparse; active
            // params drive roofline + KV, TOTAL params drive VRAM.
            let active = cookbook::active_params_from_name(&e.name);
            let is_moe = active.is_some() || e.arch.to_ascii_lowercase().contains("moe");
            ModelSpec {
                name: e.name.clone(),
                params_b: e.params_b,
                active_params_b: active.unwrap_or(e.params_b),
                is_moe,
                quant: e.quant.clone(),
                context_max: ctx_override.unwrap_or(e.context_max),
                use_case: infer_use_case(e),
            }
        })
        .collect();

    let mut ranked = cookbook::rank(&specs, &hw, args["use_case"].as_str(), args["quant"].as_str());
    if let Some(n) = args["top"].as_u64() {
        ranked.truncate(n as usize);
    }

    let body = json!({
        "gpu": hw.gpu_name,
        "backend": hw.backend.tag(),
        "vram_gb": hw.gpu_vram_gb,
        "avail_ram_gb": hw.avail_ram_gb,
        "models": serde_json::to_value(&ranked).unwrap_or(Value::Null),
    });
    serde_json::to_string_pretty(&body).unwrap_or_else(|e| format!("error: serialize cookbook: {e}"))
}
