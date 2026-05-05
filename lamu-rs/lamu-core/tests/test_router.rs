use lamu_core::registry::scan_directory;
use lamu_core::router::Router;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::Capability;
use std::path::PathBuf;

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

    let d = router.route(&sched, None, None);
    eprintln!("default: {} loaded={}", d.model_name, d.loaded);
    assert_eq!(d.model_name, heretic.name);
    assert!(d.loaded);

    let d = router.route(&sched, None, Some(&[Capability::Code]));
    assert_eq!(d.model_name, heretic.name);
    assert!(d.loaded);

    let d = router.route(&sched, Some("gpt2"), None);
    eprintln!("gpt2: {} loaded={} reason={}", d.model_name, d.loaded, d.reason);
    assert!(d.model_name.contains("gpt2"));
}
