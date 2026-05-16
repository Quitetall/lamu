//! Phase 3.4: model swap on the launcher path.
//!
//! `swap_to_model_if_needed` is the function bound to the
//! "launch chat" key in the launchers tab. It checks what's loaded
//! on :8020, kills the existing llama-server if a different model
//! lives there, spawns a new one with the canonical
//! lamu-core::backends::llamacpp::build_llama_spawn argv (Phase 4
//! unified all 3 spawn paths), and warms up one token.

use anyhow::Result;
use lamu_core::types::ModelEntry;

pub(super) fn swap_to_model_if_needed(entry: &ModelEntry) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    // Check what's loaded.
    let loaded_id: Option<String> = client
        .get("http://localhost:8020/v1/models")
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|v| {
            v["data"][0]["id"]
                .as_str()
                .map(|s| s.to_lowercase())
        });

    let already_loaded = loaded_id.as_deref().map(|id| {
        id.contains(&entry.name.to_lowercase())
            || entry.name.to_lowercase().contains(id)
    }).unwrap_or(false);

    if already_loaded {
        println!("  ✓ {} already loaded", entry.name);
        return Ok(());
    }

    // Kill existing llama-server on :8020. pkill exit codes:
    //   0 — at least one process matched and was signaled
    //   1 — no matching process (already dead — fine for our case)
    //   2 — syntax error in the pattern (shouldn't happen)
    //   3 — fatal error (e.g. /proc unreadable) — surface this
    println!("\n→ Swapping model → {} ({}B {}, ~{}MB VRAM)", entry.name, entry.params_b, entry.quant, entry.vram_mb);
    match std::process::Command::new("pkill")
        .args(["-f", "llama-server.*--port 8020"])
        .status()
    {
        Ok(status) => match status.code() {
            Some(0) | Some(1) => {} // killed, or nothing to kill — both fine
            Some(2) => anyhow::bail!("pkill syntax error — internal bug, please report"),
            Some(3) => anyhow::bail!("pkill fatal error (cannot read /proc?). Check permissions."),
            // 127 (command not found), 126 (not executable), or any
            // other non-zero/one — refuse to proceed since we have no
            // confirmation the old server actually died. Spawning a new
            // one would 404 on port-bind 60s later with a misleading
            // error.
            Some(code) => anyhow::bail!(
                "pkill exited with unexpected code {} — refusing to spawn new backend. \
                 Kill the existing llama-server manually, then retry.",
                code
            ),
            None => anyhow::bail!("pkill terminated by signal — refusing to proceed"),
        },
        Err(e) => anyhow::bail!(
            "failed to spawn pkill ({}). Install procps or kill the existing llama-server manually before swapping.",
            e
        ),
    }
    // Give it a moment to release GPU mem.
    std::thread::sleep(std::time::Duration::from_secs(2));

    let bin = lamu_core::config::llama_bin();
    if !bin.exists() {
        anyhow::bail!("llama-server not found at {}", bin.display());
    }

    // Phase 4: flag construction shared with lamu-core's Backend::load and
    // lamu-mcp's build_spawn_cmd. Picking up validated LAMU_KV + ngram-mod
    // detection that the local copy here was missing.
    let supports_ngram = lamu_core::backends::llamacpp::detect_ngram_support_blocking(&bin);
    let spawn = lamu_core::backends::llamacpp::build_llama_spawn(entry, 8020, supports_ngram, &bin)?;

    let mut cmd = std::process::Command::new(&bin);
    cmd.args(&spawn.args);
    for (k, v) in &spawn.envs {
        cmd.env(k, v);
    }
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;

    // Health poll — print progress every 5s.
    let slow_client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;
    print!("  loading");
    for i in 1..=60u32 {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let healthy = slow_client
            .get("http://localhost:8020/health")
            .send()
            .ok()
            .and_then(|r| r.json::<serde_json::Value>().ok())
            .and_then(|v| v["status"].as_str().map(|s| s == "ok"))
            .unwrap_or(false);
        if healthy {
            println!(" ✓ ({}s)", i);
            // Warmup — fires cuBLAS kernel build so first real prompt is fast.
            let _ = slow_client
                .post("http://localhost:8020/v1/chat/completions")
                .timeout(std::time::Duration::from_secs(30))
                .json(&serde_json::json!({
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1, "stream": false,
                }))
                .send();
            return Ok(());
        }
        if i % 5 == 0 { print!(" {}s", i); }
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    anyhow::bail!("timeout waiting for {} to load", entry.name)
}
