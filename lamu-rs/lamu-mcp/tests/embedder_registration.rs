//! ADR 0030 — composition-root registration of the local embedder.
//!
//! `LamuMcpServer::new` must register the process-global embedder iff
//! the registry has an embedding-capable model at startup. Both cases
//! run in ONE test fn: the global chain + the env vars it reads are
//! process-wide, and integration tests in this binary would otherwise
//! race each other on them.

use lamu_core::scheduler::VramScheduler;
use lamu_mcp::server::LamuMcpServer;
use tempfile::tempdir;

#[tokio::test]
async fn server_new_registers_embedder_only_with_embedding_capable_registry() {
    // Known chain state: no override, no key, no global.
    // SAFETY: this is the only test in this binary touching the env.
    unsafe {
        std::env::remove_var("LAMU_EMBED_PROVIDER");
        std::env::remove_var("OPENAI_API_KEY");
    }
    lamu_memory::embedder::clear_global();

    // 1) Registry WITHOUT an embedding-capable model → no registration,
    //    and (with no key) the chain resolves to None.
    let dir = tempdir().unwrap();
    let registry = dir.path().join("registry.yaml");
    std::fs::write(
        &registry,
        "models:\n  \
         chat-model:\n    \
         path: /tmp/nonexistent.gguf\n    \
         format: gguf\n    \
         backend: llama_cpp\n    \
         arch: qwen3\n    \
         params_b: 1.0\n    \
         quant: Q4_K_M\n    \
         vram_mb: 1000\n    \
         context_max: 4096\n    \
         capabilities: [chat]\n",
    )
    .unwrap();
    let _srv = LamuMcpServer::new(dir.path().to_path_buf(), registry, VramScheduler::new())
        .expect("server new (no embedding model)");
    assert!(
        lamu_memory::embedder::resolve().is_none(),
        "no embedding-capable entry → nothing registered"
    );

    // 2) Registry WITH an embedding-capable model → the adapter is
    //    registered and resolve() carries the entry's name as identity.
    let dir2 = tempdir().unwrap();
    let registry2 = dir2.path().join("registry.yaml");
    std::fs::write(
        &registry2,
        "models:\n  \
         bge-small:\n    \
         path: /tmp/nonexistent-embed.gguf\n    \
         format: gguf\n    \
         backend: llama_cpp\n    \
         arch: bert\n    \
         params_b: 0.03\n    \
         quant: Q8_0\n    \
         vram_mb: 200\n    \
         context_max: 512\n    \
         capabilities: [embedding]\n",
    )
    .unwrap();
    let _srv2 = LamuMcpServer::new(dir2.path().to_path_buf(), registry2, VramScheduler::new())
        .expect("server new (embedding model)");
    let e = lamu_memory::embedder::resolve().expect("adapter registered");
    let id = e.identity();
    assert_eq!(id.model, "bge-small");
    assert_eq!(id.dims, 0, "dims probe is lazy — 0 before the first embed");

    lamu_memory::embedder::clear_global();
}
