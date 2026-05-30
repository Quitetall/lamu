use lamu_core::registry::scan_directory;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{BackendType, Modality, ModelEntry, ModelFormat, ModelStatus};
use std::path::PathBuf;

fn mk_entry(name: &str, vram: u32, modality: Modality) -> ModelEntry {
    ModelEntry {
        name: name.to_string(),
        path: PathBuf::from(format!("/tmp/{name}")),
        format: ModelFormat::Gguf,
        backend: BackendType::LlamaCpp,
        arch: "test".into(),
        params_b: 1.0,
        quant: "Q4".into(),
        vram_mb: vram,
        context_max: 0,
        capabilities: vec![],
        reasoning_marker: None,
        speculative: None,
        sampling: None,
        pinned: false,
        main: false,
        notes: String::new(),
        status: ModelStatus::Unspecified,
        modality,
    }
}

#[test]
fn eviction_prefers_tts_over_llm() {
    let mut s = VramScheduler::new();
    let llm = mk_entry("chat-llm", 8000, Modality::Llm);
    let tts = mk_entry("local-tts", 16000, Modality::Tts);
    // LLM registered first (older last_used); TTS second; then bump TTS to
    // most-recently-used so pure-LRU would evict the LLM, not the TTS.
    s.register_loaded(llm.clone(), Some(1), 8001, llm.vram_mb);
    s.register_loaded(tts.clone(), Some(2), 8002, tts.vram_mb);
    s.mark_used("local-tts");

    // Freeing 8000MB: tiered eviction must drop the (newer, non-LLM) TTS
    // before the older LLM, and the TTS alone suffices so the LLM stays.
    let victims = s.plan_eviction(8000);
    assert_eq!(
        victims.first().map(String::as_str),
        Some("local-tts"),
        "non-LLM modality must evict before the LLM regardless of LRU; got {victims:?}"
    );
    assert!(
        !victims.contains(&"chat-llm".to_string()),
        "the LLM must not be evicted when the TTS alone frees enough; got {victims:?}"
    );
}

#[test]
fn vram_query() {
    let s = VramScheduler::new();
    let total = s.total_mb();
    if total == 0 {
        eprintln!("skip: no NVIDIA GPU");
        return;
    }
    eprintln!("Total VRAM: {} MB", total);
    let (used, total) = s.query_vram();
    eprintln!("Used: {} / Total: {} MB", used, total);
    assert!(total > 0);
}

#[test]
fn register_and_evict() {
    let models_dir = PathBuf::from(std::env::var("HOME").unwrap()).join("models");
    if !models_dir.exists() {
        return;
    }
    let entries = scan_directory(&models_dir).unwrap();
    if entries.len() < 2 {
        return;
    }
    let mut s = VramScheduler::new();
    if s.total_mb() == 0 {
        return;
    }

    let big = entries.iter().max_by_key(|e| e.vram_mb).unwrap();
    let small = entries.iter().filter(|e| e.vram_mb < 5000).next();

    s.register_loaded(big.clone(), Some(1234), 8020, big.vram_mb);
    assert!(s.is_loaded(&big.name));

    if let Some(small) = small {
        let (can, evict) = s.plan_load(small);
        eprintln!("Plan {}: can={} evict={:?}", small.name, can, evict);
    }
}
