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
        backend_kind: None,
        system_prompt: None,
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
fn multi_device_best_fit_placement() {
    // Two synthetic GPUs (24 GB + 48 GB). ADR 0017 P1: best-fit places a model
    // on the device with the most available VRAM that fits it.
    let mut s = VramScheduler::new();
    s.set_devices_for_tests(&[(24000, "small"), (48000, "big")]);
    assert_eq!(s.device_count_for_tests(), 2);

    // m1 fits both → most-available device (big, index 1).
    let m1 = mk_entry("m1", 10000, Modality::Llm);
    s.register_loaded(m1.clone(), Some(1), 8001, m1.vram_mb);
    assert_eq!(s.device_of_for_tests("m1"), Some(1), "best-fit picks the roomier card");

    // m2 (30 GB) now fits only the big card (small free=22500, big free=36500).
    let m2 = mk_entry("m2", 30000, Modality::Llm);
    assert!(s.can_fit(&m2));
    s.register_loaded(m2.clone(), Some(2), 8002, m2.vram_mb);
    assert_eq!(s.device_of_for_tests("m2"), Some(1));

    // Aggregate available = small(24000-1500) + big(48000-1500-10000-30000).
    assert_eq!(s.available_mb(), 22500 + 6500);
    assert_eq!(s.budget().per_device.len(), 2, "per-device breakdown present");
}

#[test]
fn multi_device_can_fit_is_per_device_not_summed() {
    let mut s = VramScheduler::new();
    s.set_devices_for_tests(&[(24000, "a"), (24000, "b")]);
    // 40 GB model: summed free (2×22500=45000) ≥ 40000, but NO single device
    // fits — can_fit must be false (a non-shardable model needs one card).
    let huge = mk_entry("huge", 40000, Modality::Llm);
    assert!(!s.can_fit(&huge), "must not claim fit by summing across devices");
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

// ── device placement (ADR 0017 P2) ──────────────────────────────────

#[test]
fn placement_of_records_device_index_multi_gpu() {
    use lamu_core::types::DevicePlacement;
    let mut s = VramScheduler::new();
    s.set_devices_for_tests(&[(24000, "small"), (48000, "big")]);
    // best-fit puts a 10GB model on the roomier card (index 1).
    let m1 = mk_entry("m1", 10000, Modality::Llm);
    s.register_loaded(m1.clone(), Some(1), 8001, m1.vram_mb);
    assert_eq!(s.device_of_for_tests("m1"), Some(1));
    assert_eq!(
        s.placement_of("m1"),
        Some(DevicePlacement::Single(1)),
        "placement_of must report the same NVML index the scheduler placed on"
    );
    // a smaller model — assert against actual placement, not a guess.
    let m2 = mk_entry("m2", 1000, Modality::Llm);
    s.register_loaded(m2.clone(), Some(2), 8002, m2.vram_mb);
    let want = DevicePlacement::Single(s.device_of_for_tests("m2").unwrap());
    assert_eq!(s.placement_of("m2"), Some(want));
}

#[test]
fn placement_of_single_gpu_is_index_zero() {
    use lamu_core::types::DevicePlacement;
    let mut s = VramScheduler::new();
    s.set_total_mb_for_tests(24000); // one synthetic device, index 0
    let m = mk_entry("solo", 8000, Modality::Llm);
    s.register_loaded(m.clone(), Some(1), 8020, m.vram_mb);
    assert_eq!(
        s.placement_of("solo"),
        Some(DevicePlacement::Single(0)),
        "single-GPU placement must be Single(0) — byte-identical to pre-P2"
    );
}

#[test]
fn mark_loading_records_placement_before_confirm() {
    use lamu_core::types::DevicePlacement;
    let mut s = VramScheduler::new();
    s.set_devices_for_tests(&[(24000, "a"), (48000, "b")]);
    let e = mk_entry("pending", 12000, Modality::Llm);
    let slot_gen = s.mark_loading(e.clone());
    // placement is fixed at mark_loading time (loader reads it pre-spawn).
    assert_eq!(s.placement_of("pending"), Some(DevicePlacement::Single(1)));
    // confirm_loaded must NOT move the placement.
    s.confirm_loaded("pending", slot_gen, 4242, 8001, 12000).unwrap();
    assert_eq!(
        s.placement_of("pending"),
        Some(DevicePlacement::Single(1)),
        "confirm_loaded must preserve the device chosen at mark_loading"
    );
}

#[test]
fn placement_of_absent_is_none() {
    let s = VramScheduler::new();
    assert!(s.placement_of("never-loaded").is_none());
}

// ── Generation-token tests (ADR 0040) ─────────────────────────────────────

/// mark_loading returns strictly increasing generation tokens.
#[test]
fn mark_loading_returns_strictly_increasing_gens() {
    let mut s = VramScheduler::new();
    s.set_total_mb_for_tests(48000);
    let a = mk_entry("alpha", 1000, Modality::Llm);
    let b = mk_entry("beta", 1000, Modality::Llm);
    let g1 = s.mark_loading(a.clone());
    s.mark_unloaded("alpha");
    let g2 = s.mark_loading(b.clone());
    s.mark_unloaded("beta");
    let g3 = s.mark_loading(a.clone());
    assert!(g1 < g2, "second mark_loading must get a higher gen than the first");
    assert!(g2 < g3, "third mark_loading must get a higher gen than the second");
}

/// confirm_loaded with the correct generation succeeds and transitions to Loaded.
#[test]
fn confirm_loaded_correct_gen_succeeds() {
    let mut s = VramScheduler::new();
    s.set_total_mb_for_tests(48000);
    let e = mk_entry("model", 1000, Modality::Llm);
    let slot_gen = s.mark_loading(e);
    assert!(
        s.confirm_loaded("model", slot_gen, 1234, 8001, 1000).is_ok(),
        "confirm with current gen must succeed"
    );
    let m = s.get_loaded("model").expect("entry must still exist after confirm");
    assert_eq!(m.state, lamu_core::types::ModelState::Loaded, "state must be Loaded");
}

/// confirm_loaded with a STALE generation returns Err; the entry (newer gen) survives.
#[test]
fn confirm_loaded_stale_gen_returns_err_entry_untouched() {
    let mut s = VramScheduler::new();
    s.set_total_mb_for_tests(48000);
    let e = mk_entry("model", 1000, Modality::Llm);
    let stale_gen = s.mark_loading(e.clone());
    // Supersede the first reservation with a second mark_loading.
    let current_gen = s.mark_loading(e);

    // Stale confirm must fail.
    let result = s.confirm_loaded("model", stale_gen, 1, 8001, 1000);
    assert!(result.is_err(), "stale gen confirm must return Err");

    // The entry must still exist and still be Loading (the new gen).
    let m = s.get_loaded("model").expect("entry must survive stale confirm");
    assert_eq!(m.state, lamu_core::types::ModelState::Loading, "entry must still be Loading");
    assert_eq!(m.generation, current_gen, "entry must hold the newer generation");
}

/// mark_unloaded_gen with a stale gen is a no-op; newer entry survives.
#[test]
fn mark_unloaded_gen_stale_is_noop() {
    let mut s = VramScheduler::new();
    s.set_total_mb_for_tests(48000);
    let e = mk_entry("model", 1000, Modality::Llm);
    let stale_gen = s.mark_loading(e.clone());
    let current_gen = s.mark_loading(e); // supersedes stale

    // Stale gen-gated removal: must NOT remove the newer entry.
    s.mark_unloaded_gen("model", stale_gen);
    assert!(
        s.get_loaded("model").is_some(),
        "stale mark_unloaded_gen must leave the newer entry intact"
    );
    assert_eq!(
        s.get_loaded("model").unwrap().generation,
        current_gen,
        "entry must still hold the current generation after stale cleanup"
    );
}

/// mark_unloaded_gen with the current gen removes the entry.
#[test]
fn mark_unloaded_gen_current_removes_entry() {
    let mut s = VramScheduler::new();
    s.set_total_mb_for_tests(48000);
    let e = mk_entry("model", 1000, Modality::Llm);
    let slot_gen = s.mark_loading(e);

    s.mark_unloaded_gen("model", slot_gen);
    assert!(
        s.get_loaded("model").is_none(),
        "current-gen mark_unloaded_gen must remove the entry"
    );
}

/// Unconditional mark_unloaded still removes regardless of generation (operator path).
#[test]
fn mark_unloaded_unconditional_removes() {
    let mut s = VramScheduler::new();
    s.set_total_mb_for_tests(48000);
    let e = mk_entry("model", 1000, Modality::Llm);
    let _slot_gen = s.mark_loading(e);

    s.mark_unloaded("model");
    assert!(
        s.get_loaded("model").is_none(),
        "unconditional mark_unloaded must always remove the entry"
    );
}

/// Full race regression: gen-1 load drains → gen-2 load starts →
/// stale gen-1 mark_unloaded_gen and confirm_loaded are both no-ops.
#[test]
fn race_regression_stale_gen1_cannot_clobber_gen2() {
    let mut s = VramScheduler::new();
    s.set_total_mb_for_tests(48000);
    let e = mk_entry("racemodel", 2000, Modality::Llm);

    // Gen-1: start a load.
    let gen1 = s.mark_loading(e.clone());

    // Simulate mid-spawn drain (e.g. eviction scan removes the slot).
    s.mark_unloaded("racemodel");
    assert!(s.get_loaded("racemodel").is_none(), "after drain, model must not be loaded");

    // Gen-2: a fresh load starts (e.g. user retries).
    let gen2 = s.mark_loading(e);
    assert!(gen2 > gen1, "gen2 must be strictly greater than gen1");

    // Gen-1's stale self-cleanup fires: must be a NO-OP.
    s.mark_unloaded_gen("racemodel", gen1);
    assert!(
        s.get_loaded("racemodel").is_some(),
        "stale gen-1 mark_unloaded_gen must not clobber gen-2 entry"
    );

    // Gen-1's stale confirm fires: must return Err and not transition gen-2.
    let result = s.confirm_loaded("racemodel", gen1, 99, 8099, 2000);
    assert!(result.is_err(), "stale gen-1 confirm must return Err");

    // Gen-2 entry is intact and still Loading.
    let m = s.get_loaded("racemodel").expect("gen-2 entry must survive");
    assert_eq!(m.state, lamu_core::types::ModelState::Loading, "gen-2 must still be Loading");
    assert_eq!(m.generation, gen2, "gen-2 entry must hold gen2 token");
}

#[test]
fn plan_load_uses_per_device_deficit_not_aggregate() {
    // M9: two 24GB cards each holding a 20GB model → 4GB free each, 8GB
    // aggregate. A new 6GB model fits on NEITHER card as-is, but evicting one
    // card's model makes room. The old aggregate deficit (6 - 8 = 0) evicted
    // nothing and wrongly refused; the per-device deficit (6 - 4 = 2) evicts.
    let mut s = VramScheduler::new();
    s.set_devices_for_tests(&[(24000, "a"), (24000, "b")]);
    let big1 = mk_entry("big1", 20000, Modality::Llm);
    let big2 = mk_entry("big2", 20000, Modality::Llm);
    s.register_loaded(big1.clone(), Some(1), 8001, 20000);
    s.register_loaded(big2.clone(), Some(2), 8002, 20000);

    let newcomer = mk_entry("newcomer", 6000, Modality::Llm);
    let (can_load, to_evict) = s.plan_load(&newcomer);
    assert!(can_load, "must be loadable by evicting one card's model (M9)");
    assert!(!to_evict.is_empty(), "must plan an eviction, not refuse with empty");
    assert_eq!(to_evict.len(), 1, "evicting one 20GB model frees enough on the target");
}
