use lamu_core::registry::{load_registry, scan_directory, write_registry};
use std::path::PathBuf;

#[test]
fn scan_models_dir() {
    let models_dir = PathBuf::from(std::env::var("HOME").unwrap()).join("models");
    if !models_dir.exists() {
        eprintln!("skip: no ~/models");
        return;
    }
    let entries = scan_directory(&models_dir).expect("scan failed");
    assert!(!entries.is_empty(), "expected at least one GGUF");
    eprintln!("Discovered {} models:", entries.len());
    for e in &entries {
        eprintln!(
            "  {}: {}B {} ({} MB) [{}] caps={:?}",
            e.name, e.params_b, e.quant, e.vram_mb, e.arch, e.capabilities
        );
    }
}

#[test]
fn yaml_roundtrip() {
    let models_dir = PathBuf::from(std::env::var("HOME").unwrap()).join("models");
    if !models_dir.exists() {
        return;
    }
    let entries = scan_directory(&models_dir).unwrap();
    let tmp = std::env::temp_dir().join("lamu-test-registry.yaml");
    write_registry(&entries, &tmp).unwrap();
    let loaded = load_registry(&tmp).unwrap();
    assert_eq!(entries.len(), loaded.len());
}
