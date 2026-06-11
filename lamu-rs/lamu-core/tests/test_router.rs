use lamu_core::health::{BackendHealth, HealthState};
use lamu_core::registry::scan_directory;
use lamu_core::router::Router;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{
    BackendType, Capability, ModelEntry, ModelFormat, ModelStatus,
};
use std::collections::HashMap;
use std::path::PathBuf;

fn sample_entry(name: &str, main: bool) -> ModelEntry {
    ModelEntry {
        name: name.to_string(),
        path: PathBuf::from(format!("/tmp/{name}.gguf")),
        format: ModelFormat::Gguf,
        backend: BackendType::LlamaCpp,
        arch: "qwen35".into(),
        params_b: 7.0,
        quant: "Q4_K_M".into(),
        vram_mb: 8000,
        context_max: 32768,
        capabilities: vec![Capability::Chat],
        reasoning_marker: None,
        speculative: None,
        sampling: None,
        pinned: false,
        main,
        notes: String::new(),
        status: ModelStatus::Unspecified,
        modality: lamu_core::types::Modality::Llm,
        backend_kind: None,
        system_prompt: None,
    }
}

#[test]
fn route_loaded() {
    let models_dir = PathBuf::from(std::env::var("HOME").unwrap()).join("models");
    if !models_dir.exists() {
        return;
    }
    let entries = scan_directory(&models_dir).unwrap();
    let mut sched = VramScheduler::new();
    if sched.total_mb() == 0 {
        return;
    }

    let heretic = entries.iter()
        .find(|e| e.name.contains("heretic") && e.name.contains("q4_k_m"))
        .expect("heretic GGUF present");
    sched.register_loaded(heretic.clone(), Some(0), 8020, 18500);

    let router = Router::new(&sched, entries.clone());

    let d = router.route(&sched, None, None, None);
    eprintln!("default: {} loaded={}", d.model_name, d.loaded);
    assert_eq!(d.model_name, heretic.name);
    assert!(d.loaded);

    let d = router.route(&sched, None, Some(&[Capability::Code]), None);
    assert_eq!(d.model_name, heretic.name);
    assert!(d.loaded);

    let d = router.route(&sched, Some("gpt2"), None, None);
    eprintln!("gpt2: {} loaded={} reason={}", d.model_name, d.loaded, d.reason);
    assert!(d.model_name.contains("gpt2"));

    // health_map: explicit unhealthy backend is refused with reason
    let mut hm: HashMap<String, BackendHealth> = HashMap::new();
    let mut h = BackendHealth::new(&heretic.name);
    h.force_quarantine("test");
    assert_eq!(h.state, HealthState::Quarantined);
    hm.insert(heretic.name.clone(), h);

    let d = router.route(&sched, Some(&heretic.name), None, Some(&hm));
    assert!(d.reason.contains("unhealthy"), "reason={}", d.reason);
    assert!(!d.loaded);

    // Capability route falls back when the only loaded match is quarantined.
    let d = router.route(&sched, None, None, Some(&hm));
    // Must NOT pick heretic since it's quarantined.
    assert_ne!(d.model_name, heretic.name);
}

#[test]
fn route_excludes_non_llm_modality_from_chat() {
    use lamu_core::types::Modality;
    // A tts entry with EMPTY capabilities would match a no-filter chat
    // request (empty `required` ⊆ everything) if the modality guard didn't
    // exclude it — the exact regression the guard prevents.
    let mut tts = sample_entry("local-tts", false);
    tts.modality = Modality::Tts;
    tts.capabilities = vec![];
    let llm = sample_entry("chat-llm", false); // default caps = [Chat]
    let (sched, router) = alias_router(vec![tts, llm]);
    let d = router.route(&sched, None, None, None);
    assert_eq!(
        d.model_name, "chat-llm",
        "chat must route to the LLM, never the tts entry; got {:?} ({})",
        d.model_name, d.reason
    );

    // A registry with ONLY a tts entry yields no chat candidate at all.
    let mut only = sample_entry("only-tts", false);
    only.modality = Modality::Tts;
    only.capabilities = vec![];
    let (sched2, router2) = alias_router(vec![only]);
    let d2 = router2.route(&sched2, None, None, None);
    assert_ne!(d2.model_name, "only-tts", "tts entry must never be chat-routable");
    assert!(
        d2.model_name.is_empty() || d2.reason.contains("no model"),
        "expected no chat candidate; got {:?} ({})",
        d2.model_name, d2.reason
    );
}

// ── Alias resolution ─────────────────────────────────────────────────
//
// These tests don't depend on a real GGUF on disk — they exercise the
// router's `main: true` alias logic against synthetic entries.

fn alias_router(entries: Vec<ModelEntry>) -> (VramScheduler, Router) {
    let mut sched = VramScheduler::new();
    sched.set_total_mb_for_tests(24_000);  // also nulls NVML for deterministic plan_load
    let router = Router::new(&sched, entries);
    (sched, router)
}

#[test]
fn alias_lamu_resolves_to_main() {
    let entries = vec![
        sample_entry("small", false),
        sample_entry("big-main", true),
        sample_entry("other", false),
    ];
    let (sched, router) = alias_router(entries);
    for alias in ["lamu", "main", "default", "LAMU", "Default"] {
        let d = router.route(&sched, Some(alias), None, None);
        assert_eq!(d.model_name, "big-main", "alias '{alias}' must resolve to main entry");
    }
}

#[test]
fn alias_falls_through_when_no_main() {
    let entries = vec![
        sample_entry("a", false),
        sample_entry("b", false),
    ];
    let (sched, router) = alias_router(entries);
    let d = router.route(&sched, Some("lamu"), None, None);
    // "lamu" not a real model name → not found. router.find_model substring
    // is intentionally strict; this surfaces a misconfiguration clearly.
    assert!(d.model_name.contains("lamu") || d.reason.contains("not found"),
        "with no main set, alias must produce a clear miss; got {:?} ({})",
        d.model_name, d.reason);
}

#[test]
fn alias_first_main_wins_on_duplicate() {
    // Two entries flagged main — current contract is first-wins on
    // HashMap iteration order. The test asserts deterministic behavior
    // (the alias resolves to ONE of them and the same one across calls),
    // not the specific entry, because HashMap order isn't guaranteed.
    let entries = vec![
        sample_entry("main-a", true),
        sample_entry("main-b", true),
        sample_entry("other", false),
    ];
    let (sched, router) = alias_router(entries);
    let d1 = router.route(&sched, Some("lamu"), None, None);
    let d2 = router.route(&sched, Some("lamu"), None, None);
    assert!(d1.model_name == "main-a" || d1.model_name == "main-b");
    assert_eq!(d1.model_name, d2.model_name, "alias must resolve deterministically across calls");
}

#[test]
fn no_model_prefers_loaded_main_over_capability_match() {
    // When model is None AND the main entry is loaded + healthy, it
    // wins regardless of capability ranking.
    let main = sample_entry("main-model", true);
    let other = sample_entry("other", false);
    let mut entries_vec = vec![main.clone(), other];
    let (mut sched, _) = alias_router(entries_vec.clone());
    sched.register_loaded(main.clone(), Some(0), 8020, 8000);
    // Re-create router with the now-loaded scheduler so plan_load sees it.
    entries_vec[0] = main.clone();
    let router = Router::new(&sched, entries_vec);
    let d = router.route(&sched, None, None, None);
    assert_eq!(d.model_name, "main-model");
    assert!(d.loaded);
    assert!(d.reason.contains("main"));
}

#[test]
fn route_main_preference_respects_requested_capabilities() {
    // M10: model=None + an explicit capability the loaded main LACKS must not be
    // satisfied by the chat-only main — it falls through to capability routing.
    let main = sample_entry("main-chat", true);
    let entries = vec![main.clone()];
    let mut sched = VramScheduler::new();
    sched.set_total_mb_for_tests(24_000);
    sched.register_loaded(main.clone(), Some(0), 8020, 8000); // main is LOADED
    let router = Router::new(&sched, entries);

    // No capability ask → prefer the loaded main.
    let d = router.route(&sched, None, None, None);
    assert_eq!(d.model_name, "main-chat");
    assert!(d.loaded);

    // Vision requested → the chat-only main must NOT be returned.
    let d = router.route(&sched, None, Some(&[Capability::Vision]), None);
    assert_ne!(d.model_name, "main-chat",
        "a vision request must not be answered by a chat-only main (M10)");
    assert!(d.model_name.is_empty(),
        "no vision-capable model in registry → no candidate");
}
