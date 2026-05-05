use lamu_core::registry::scan_directory;
use lamu_core::scheduler::VramScheduler;
use std::path::PathBuf;

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
