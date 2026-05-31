//! llama.cpp backend — manages llama-server subprocess.
//! Direct port of `lamu/backends/llamacpp.py`.

use crate::backends::{Backend, ChatMessage};
use crate::config::llama_bin;
use crate::scheduler::VramScheduler;
use crate::types::ModelEntry;
use crate::{Error, Result};
use async_trait::async_trait;
use futures_util::stream::{Stream, StreamExt};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::time::sleep;

/// Argv + env for spawning a `llama-server` configured for one
/// `ModelEntry`. Pure data — caller picks tokio or std::process,
/// caller spawns + health-polls + warms up. The flag set is the
/// authoritative one; lamu-mcp and lamu-cli's swap path consume from
/// here so the three spawn paths can never drift again.
#[derive(Debug, Clone)]
pub struct LlamaSpawn {
    pub args: Vec<String>,
    pub envs: Vec<(String, String)>,
}

/// Build the canonical llama-server argv + env for `entry` on `port`.
///
/// `bin` is the llama-server binary that will run the args. When `bin`
/// is the BeeLlama fork (`is_bee_binary(bin) == true`) AND the entry
/// has a usable DFlash speculative config, this function emits the
/// matching `--spec-dflash-default` + `-md <draft>` flags and switches
/// the KV cache to `turbo3_tcq` (bee's TurboQuant TCQ encoding).
///
/// `supports_ngram` is the result of probing `llama-server --help` for
/// the full ngram-mod sub-flag set. Pass `false` if the binary doesn't
/// support speculative ngram-mod (older builds) — the ngram flags
/// will be omitted instead of failing the spawn. Ignored when the
/// DFlash path is taken (DFlash > ngram on the bee binary).
///
/// Env knobs read here:
/// - `LAMU_DEFAULT_CTX` — caps the context window (default: full).
/// - `LAMU_KV` — KV cache type. Validated against the set llama.cpp
///   actually accepts; an unknown value is rejected up front rather
///   than crashing the server at startup or silently falling back to
///   f16. Default: `q8_0` (speed/VRAM sweet spot) — auto-upgraded to
///   `turbo3_tcq` when the bee binary + DFlash spec path is active,
///   unless `LAMU_KV` is explicitly set.
/// - `LAMU_BIND_HOST` — bind address (default: `127.0.0.1`). Set to
///   `0.0.0.0` to opt in to remote exposure.
pub fn build_llama_spawn(
    entry: &ModelEntry,
    port: u16,
    supports_ngram: bool,
    bin: &std::path::Path,
) -> Result<LlamaSpawn> {
    // Set-but-unparseable warns (was silently u32::MAX = unbounded ctx).
    let ctx_cap = crate::config::parse_env_or("LAMU_DEFAULT_CTX", u32::MAX);
    let ctx = entry.context_max.min(ctx_cap);

    // Embedding models serve via llama-server `--embedding` with mean
    // pooling (OpenAI-compat /v1/embeddings + /embedding) — none of the
    // chat KV/spec/flash-attn machinery applies. Short branch, early return.
    if entry.capabilities.contains(&crate::types::Capability::Embedding) {
        let host = std::env::var("LAMU_BIND_HOST").unwrap_or_else(|_| "127.0.0.1".into());
        return Ok(LlamaSpawn {
            args: vec![
                "-m".into(),
                entry.path.display().to_string(),
                "--host".into(),
                host,
                "--port".into(),
                port.to_string(),
                "--embedding".into(),
                "--pooling".into(),
                "mean".into(),
                "-ngl".into(),
                "99".into(),
                "--ctx-size".into(),
                ctx.clamp(512, 8192).to_string(),
            ],
            envs: vec![("CUDA_VISIBLE_DEVICES".into(), "0".into())],
        });
    }

    // Decide whether the DFlash-on-bee path applies. Both the binary-
    // selection logic in `new_for_entry` and this arg-emission path
    // gate on `entry_has_usable_dflash(entry)`; if those diverge a
    // bee binary would spawn without `-md`, defeating the speedup.
    let dflash_spec = if is_bee_binary(bin) && entry_has_usable_dflash(entry) {
        entry.speculative.as_ref()
    } else {
        None
    };
    let bee_dflash_active = dflash_spec.is_some();

    let kv_type = match std::env::var("LAMU_KV").as_deref() {
        Ok(v @ ("q8_0" | "q4_0" | "q4_1" | "q5_0" | "q5_1"
            | "f16" | "bf16" | "f32"
            | "turbo2" | "turbo3" | "turbo3_tcq" | "turbo4")) => v.to_string(),
        Ok(other) => {
            return Err(Error::Backend(format!(
                "LAMU_KV='{}' invalid — expected one of: q8_0, q4_0, q4_1, q5_0, q5_1, f16, bf16, f32, turbo2, turbo3, turbo3_tcq, turbo4",
                other
            )));
        }
        Err(_) if bee_dflash_active => "turbo3_tcq".to_string(),
        Err(_) => "q8_0".to_string(),
    };

    let host = std::env::var("LAMU_BIND_HOST").unwrap_or_else(|_| "127.0.0.1".into());

    let mut args: Vec<String> = vec![
        "-m".into(), entry.path.display().to_string(),
        "--host".into(), host,
        "--port".into(), port.to_string(),
        "--ctx-size".into(), ctx.to_string(),
        "-ngl".into(), "99".into(),
        "--flash-attn".into(), "on".into(),
        "--cache-type-k".into(), kv_type.clone(),
        "--cache-type-v".into(), kv_type,
        "--parallel".into(), "1".into(),
        // Larger prompt-eval batches = fewer kernel launches per turn.
        // 4096/512 keeps VRAM stable on a 24GB card.
        "--batch-size".into(), "4096".into(),
        "--ubatch-size".into(), "512".into(),
        // Reuse shared prefixes across multi-turn chat — the next
        // turn's KV starts where the last one ended, so re-eval skips
        // the system prompt + history.
        "--cache-reuse".into(), "256".into(),
    ];

    if let Some(sc) = dflash_spec {
        // DFlash drafter pairing — biggest single t/s win on bee.
        // ~82-101 t/s on Qwen3.6-27B vs ~44 t/s vanilla. Uses
        // `--spec-dflash-default` (convenience flag setting spec-type
        // dflash + sensible cross-ctx / max-slots defaults), matching
        // the validated `scripts/serve-qwen36-bee.sh` config.
        args.extend([
            "-md".into(), sc.draft_path.display().to_string(),
            "-ngld".into(), "99".into(),
            "--spec-dflash-default".into(),
        ]);
    } else if supports_ngram && (entry.arch == "qwen35" || entry.arch == "qwen3") {
        // Fallback for binaries without DFlash but with ngram-mod (~10-15%
        // over baseline on warm runs). Only fires when DFlash didn't.
        args.extend([
            "--spec-type".into(), "ngram-mod".into(),
            "--spec-ngram-mod-n-match".into(), "24".into(),
            "--spec-ngram-mod-n-min".into(), "12".into(),
            "--spec-ngram-mod-n-max".into(), "48".into(),
        ]);
    }

    Ok(LlamaSpawn {
        args,
        envs: vec![("CUDA_VISIBLE_DEVICES".into(), "0".into())],
    })
}

/// Resolve the BeeLlama llama-server binary, if available.
///
/// Resolution order:
///   1. `LAMU_BEE_BIN` env var — explicit override. The path must be
///      under a `beellama.cpp` directory component so a hostile env
///      can't redirect to an attacker-controlled binary outside the
///      bee tree (substring `contains("beellama")` wasn't enough —
///      e.g. `/tmp/notbeellama.cpp/x` matched). If the env var is set
///      but invalid (missing / non-executable / wrong directory),
///      a warning is logged and the default is tried.
///   2. `~/local-llm/beellama.cpp/build/bin/llama-server` — default
///      build layout matching `scripts/serve-qwen36-bee.sh`.
///
/// Returns `None` when neither path exists, so callers can fall back
/// to the generic `llama_bin()`.
pub fn bee_bin() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("LAMU_BEE_BIN") {
        let pb = std::path::PathBuf::from(&p);
        if !pb.exists() {
            tracing::warn!(
                "LAMU_BEE_BIN points to non-existent path '{}'; falling back to default",
                p
            );
        } else if !is_bee_binary(&pb) {
            tracing::warn!(
                "LAMU_BEE_BIN='{}' not under a `beellama.cpp` directory — refusing to \
                 spawn an out-of-tree binary as the bee fork. Place the binary under a \
                 `beellama.cpp` path or rebuild bee at the default location.",
                p
            );
        } else if !is_file_executable(&pb) {
            tracing::warn!(
                "LAMU_BEE_BIN='{}' is not an executable file; falling back to default",
                p
            );
        } else {
            return Some(pb);
        }
    }
    let home = std::env::var("HOME").ok().map(std::path::PathBuf::from)
        .or_else(dirs::home_dir)?;
    let default = home
        .join("local-llm")
        .join("beellama.cpp")
        .join("build")
        .join("bin")
        .join("llama-server");
    // Same executability gate the env-var path uses — if the default
    // location is e.g. a stale directory or non-executable file we
    // refuse rather than handing back a path that will fail at
    // `Command::new` time with a confusing OS error.
    (default.exists() && is_file_executable(&default)).then_some(default)
}

/// True if `bin` resides under a `beellama.cpp` directory component.
/// Tighter than a substring check: paths like
/// `/tmp/notbeellama.cpp/llama-server` no longer match, closing a
/// LAMU_BEE_BIN injection vector.
pub fn is_bee_binary(bin: &std::path::Path) -> bool {
    bin.components()
        .any(|c| c.as_os_str() == std::ffi::OsStr::new("beellama.cpp"))
}

#[cfg(unix)]
fn is_file_executable(path: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.is_file() && (m.permissions().mode() & 0o111) != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_file_executable(path: &std::path::Path) -> bool {
    // No portable executable-bit on Windows; defer to file existence.
    // Lamu's target platform is Linux per CLAUDE.md — this stub exists
    // only so the crate keeps compiling under cross / cargo check on
    // non-Unix targets.
    path.is_file()
}

/// Returns true when `entry` carries a usable DFlash speculative
/// config — single source of truth for both binary selection
/// (`LlamaCppBackend::new_for_entry`) and arg emission
/// (`build_llama_spawn`). Keeping these in lockstep prevents the
/// pick-bee-but-emit-no-DFlash drift V4 Pro flagged.
fn entry_has_usable_dflash(entry: &ModelEntry) -> bool {
    entry.speculative.as_ref().is_some_and(|sc| {
        sc.method.eq_ignore_ascii_case("dflash")
            && sc.draft_path.is_file()
    })
}

/// Async helper: probe `llama-server --help` for ngram-mod support.
///
/// Probes for ALL THREE flags `build_llama_spawn` emits (`-n-match`,
/// `-n-min`, `-n-max`) — older `llama-server` builds (the Lucebox
/// `dflash-pr` branch in particular) advertise `--spec-type ngram-mod`
/// in the generic enum but lack the supporting sub-flags; passing only
/// `--spec-type ngram-mod` without `-n-min`/`-n-max` would let the
/// server start with a broken speculator config.
pub async fn detect_ngram_support(bin: &std::path::Path) -> bool {
    match Command::new(bin).arg("--help").output().await {
        Ok(o) => has_full_ngram_mod_flags(&String::from_utf8_lossy(&o.stdout)),
        Err(_) => false,
    }
}

/// Sync helper for blocking callers (the TUI swap path).
pub fn detect_ngram_support_blocking(bin: &std::path::Path) -> bool {
    match std::process::Command::new(bin).arg("--help").output() {
        Ok(o) => has_full_ngram_mod_flags(&String::from_utf8_lossy(&o.stdout)),
        Err(_) => false,
    }
}

/// Returns true when the help text contains all three ngram-mod
/// sub-flags `build_llama_spawn` emits. Factored out so the async and
/// blocking probes can share the predicate (and so it's
/// straight-forward to unit-test against captured help output).
fn has_full_ngram_mod_flags(help: &str) -> bool {
    help.contains("--spec-ngram-mod-n-match")
        && help.contains("--spec-ngram-mod-n-min")
        && help.contains("--spec-ngram-mod-n-max")
}

pub struct LlamaCppBackend {
    bin_path: PathBuf,
    proc: Option<Child>,
    port: u16,
    model_name: String,
    client: reqwest::Client,
}

impl LlamaCppBackend {
    pub fn new(bin_path: Option<PathBuf>) -> Result<Self> {
        // reqwest::Client::build can fail (e.g. invalid TLS config). Phase C:
        // propagate as Error::Http instead of panicking.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| Error::Http(format!("reqwest client init: {}", e)))?;
        Ok(Self {
            bin_path: bin_path.unwrap_or_else(llama_bin),
            proc: None,
            port: 0,
            model_name: String::new(),
            client,
        })
    }

    /// Entry-aware constructor. Tries the BeeLlama fork binary first
    /// when the entry carries a usable DFlash spec; falls back to the
    /// generic `llama_bin()` when bee isn't available OR the entry
    /// has no DFlash config. `build_llama_spawn` checks the same
    /// predicate (`entry_has_usable_dflash`) before emitting DFlash
    /// args, so the selection here can never mismatch the args.
    pub fn new_for_entry(entry: &ModelEntry) -> Result<Self> {
        let bin_path = if entry_has_usable_dflash(entry) {
            bee_bin().unwrap_or_else(llama_bin)
        } else {
            llama_bin()
        };
        Self::new(Some(bin_path))
    }
}

#[async_trait]
impl Backend for LlamaCppBackend {
    async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32> {
        self.port = port;
        self.model_name = entry.name.clone();

        if !self.bin_path.exists() {
            return Err(Error::Backend(format!(
                "llama-server not found at {}",
                self.bin_path.display()
            )));
        }

        let supports_ngram = detect_ngram_support(&self.bin_path).await;
        let spawn = build_llama_spawn(entry, port, supports_ngram, &self.bin_path)?;

        let mut cmd = Command::new(&self.bin_path);
        cmd.args(&spawn.args);
        for (k, v) in &spawn.envs {
            cmd.env(k, v);
        }
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
        crate::backends::harden_child_command(&mut cmd);

        let child = cmd.spawn()
            .map_err(|e| Error::Backend(format!("spawn failed: {}", e)))?;

        let pid = child.id().unwrap_or(0);
        self.proc = Some(child);

        // Health poll
        for _ in 0..60 {
            sleep(Duration::from_secs(1)).await;
            // Detect a child that exited during startup (CUDA OOM, bad
            // args, port already bound, missing .so) — otherwise the loop
            // can't tell "crashed" from "still warming up" and runs the
            // full timeout while holding the global spawn gate. (#16)
            if let Some(p) = self.proc.as_mut() {
                if let Ok(Some(status)) = p.try_wait() {
                    return Err(Error::Backend(format!(
                        "llama-server exited during startup (port {port}, {status})"
                    )));
                }
            }
            if self.is_healthy().await {
                // Warmup: one-token completion so cuBLAS / cuDNN
                // handles + kv slot are JIT-compiled before the user's
                // first real prompt. Otherwise TTFT for the first
                // message bakes in 2–4s of kernel build cost.
                let warm_url = format!("http://localhost:{}/v1/chat/completions", port);
                let _ = self.client
                    .post(&warm_url)
                    .timeout(Duration::from_secs(30))
                    .json(&json!({
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "stream": false,
                    }))
                    .send()
                    .await;
                return Ok(pid);
            }
        }

        let _ = self.unload().await;
        Err(Error::Backend(format!("llama-server health timeout (port {})", port)))
    }

    async fn unload(&mut self) -> Result<()> {
        if let Some(mut p) = self.proc.take() {
            crate::backends::graceful_kill(&mut p).await;
        }
        self.model_name.clear();
        Ok(())
    }

    async fn is_healthy(&self) -> bool {
        let url = format!("http://localhost:{}/health", self.port);
        let Ok(resp) = self.client.get(&url)
            .timeout(Duration::from_secs(2))
            .send().await
        else { return false; };
        let Ok(json) = resp.json::<Value>().await else { return false; };
        json.get("status").and_then(|v| v.as_str()) == Some("ok")
    }

    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String> {
        self.generate_with_opts(messages, max_tokens, temperature,
                                crate::backends::GenerateOpts::default()).await
    }

    async fn generate_with_opts(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        opts: crate::backends::GenerateOpts,
    ) -> Result<String> {
        let mut payload = json!({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": false,
        });
        // bee llama-server honors chat_template_kwargs.enable_thinking for
        // Qwen3.6 / Qwen3.5. None = leave default (model decides).
        if let Some(et) = opts.enable_thinking {
            payload["chat_template_kwargs"] = json!({ "enable_thinking": et });
        }
        // Extra samplers — llama-server's OpenAI-compat endpoint accepts
        // these as top-level fields. Only send when Some (no nulls).
        if let Some(v) = opts.top_p { payload["top_p"] = json!(v); }
        if let Some(v) = opts.top_k { payload["top_k"] = json!(v); }
        if let Some(v) = opts.min_p { payload["min_p"] = json!(v); }
        if let Some(v) = opts.repeat_penalty { payload["repeat_penalty"] = json!(v); }
        let url = format!("http://localhost:{}/v1/chat/completions", self.port);
        let resp = self.client.post(&url).json(&payload).send().await
            .map_err(|e| Error::Backend(format!("http: {}", e)))?;
        let data: Value = resp.json().await
            .map_err(|e| Error::Backend(format!("json: {}", e)))?;
        let msg = data.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message"))
            .ok_or_else(|| Error::Backend("missing choices[0].message".into()))?;
        let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let reasoning = msg.get("reasoning_content").and_then(|v| v.as_str()).unwrap_or("");
        if !reasoning.is_empty() {
            Ok(format!("<think>\n{}\n</think>\n{}", reasoning, content))
        } else {
            Ok(content.to_string())
        }
    }

    async fn stream(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        let payload = json!({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": true,
        });
        let url = format!("http://localhost:{}/v1/chat/completions", self.port);
        let resp = self.client.post(&url).json(&payload).send().await
            .map_err(|e| Error::Backend(format!("http: {}", e)))?;

        let byte_stream = resp.bytes_stream();
        let line_stream = byte_stream
            .map(|res| res.map_err(|e| Error::Backend(format!("stream: {}", e))));

        let decoded = async_stream::try_stream! {
            let mut buf: Vec<u8> = Vec::new();
            let mut s = std::pin::pin!(line_stream);
            while let Some(chunk) = s.next().await {
                let chunk = chunk?;
                // Byte-buffer, decode whole lines only: from_utf8_lossy on a
                // raw chunk corrupts a multibyte char split across chunks.
                buf.extend_from_slice(&chunk);
                while let Some(line) = crate::sse::next_sse_line(&mut buf) {
                    let line = line.trim();
                    if let Some(rest) = line.strip_prefix("data: ") {
                        if rest == "[DONE]" { return; }
                        let Ok(val) = serde_json::from_str::<Value>(rest) else { continue };
                        let Some(token) = val.get("choices")
                            .and_then(|c| c.get(0))
                            .and_then(|c| c.get("delta"))
                            .and_then(|d| d.get("content"))
                            .and_then(|c| c.as_str())
                        else { continue };
                        if !token.is_empty() {
                            yield token.to_string();
                        }
                    }
                }
            }
        };

        Ok(Box::pin(decoded))
    }

    fn port(&self) -> u16 { self.port }
    fn model_name(&self) -> &str { &self.model_name }
}

impl LlamaCppBackend {
    /// Query actual VRAM usage via NVML PID lookup.
    pub fn get_vram_mb(&self, scheduler: &VramScheduler) -> u32 {
        let Some(p) = self.proc.as_ref() else { return 0 };
        let Some(my_pid) = p.id() else { return 0 };
        for (pid, mb) in scheduler.query_gpu_pids() {
            if pid == my_pid {
                return mb;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BackendType, Capability, ModelFormat};
    use std::sync::Mutex;

    // Tests in this mod read/write LAMU_* env vars. They are
    // process-local but thread-shared, so cargo test's default
    // multi-thread runner would race. Serialize via this Mutex.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_env() {
        // SAFETY: ENV_LOCK held by the calling test for the duration of
        // the test body. No concurrent reads from another LAMU env user
        // exist within the lamu-core test binary.
        unsafe {
            std::env::remove_var("LAMU_BIND_HOST");
            std::env::remove_var("LAMU_KV");
            std::env::remove_var("LAMU_DEFAULT_CTX");
        }
    }

    fn dummy_entry(arch: &str, ctx_max: u32) -> ModelEntry {
        ModelEntry {
            name: "test".into(),
            path: PathBuf::from("/tmp/test.gguf"),
            format: ModelFormat::Gguf,
            backend: BackendType::LlamaCpp,
            arch: arch.into(),
            params_b: 4.0,
            quant: "Q4_K_M".into(),
            vram_mb: 4096,
            context_max: ctx_max,
            capabilities: vec![Capability::Code],
            reasoning_marker: None,
            speculative: None,
            sampling: None,
            pinned: false,
            main: false,
            notes: String::new(),
            status: crate::types::ModelStatus::default(),
            modality: crate::types::Modality::Llm,
        }
    }

    #[test]
    fn build_llama_spawn_embedding_mode() {
        let mut e = dummy_entry("nomic-bert", 8192);
        e.capabilities = vec![Capability::Embedding];
        let spawn =
            build_llama_spawn(&e, 8030, false, std::path::Path::new("/usr/bin/llama-server"))
                .unwrap();
        assert!(spawn.args.iter().any(|a| a == "--embedding"), "args: {:?}", spawn.args);
        assert!(spawn.args.windows(2).any(|w| w[0] == "--pooling" && w[1] == "mean"));
        // None of the chat-only machinery in embedding mode.
        assert!(!spawn.args.iter().any(|a| a == "--flash-attn"));
        assert!(!spawn.args.iter().any(|a| a == "--cache-reuse"));
    }

    #[test]
    fn build_llama_spawn_defaults_localhost_q8_0() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let entry = dummy_entry("llama", 8192);
        let s = build_llama_spawn(&entry, 8020, false, std::path::Path::new("/usr/bin/llama-server")).unwrap();
        let joined = s.args.join(" ");
        assert!(joined.contains("--host 127.0.0.1"), "{joined}");
        assert!(joined.contains("--cache-type-k q8_0"), "{joined}");
        assert!(joined.contains("--cache-type-v q8_0"), "{joined}");
        assert!(joined.contains("--port 8020"), "{joined}");
        assert!(joined.contains("--ctx-size 8192"), "{joined}");
        assert!(joined.contains("--batch-size 4096"), "{joined}");
        assert!(joined.contains("--ubatch-size 512"), "{joined}");
        assert!(joined.contains("--cache-reuse 256"), "{joined}");
        assert!(!joined.contains("--spec-type"), "{joined}");
        assert_eq!(s.envs, vec![("CUDA_VISIBLE_DEVICES".into(), "0".into())]);
    }

    #[test]
    fn build_llama_spawn_rejects_bad_kv() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { std::env::set_var("LAMU_KV", "garbage"); }
        let entry = dummy_entry("llama", 8192);
        let r = build_llama_spawn(&entry, 8020, false, std::path::Path::new("/usr/bin/llama-server"));
        clear_env();
        assert!(matches!(r, Err(Error::Backend(_))), "expected Backend err, got {:?}", r);
    }

    #[test]
    fn build_llama_spawn_caps_ctx_via_env() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { std::env::set_var("LAMU_DEFAULT_CTX", "4096"); }
        let entry = dummy_entry("llama", 131072);
        let s = build_llama_spawn(&entry, 8020, false, std::path::Path::new("/usr/bin/llama-server")).unwrap();
        clear_env();
        assert!(s.args.join(" ").contains("--ctx-size 4096"));
    }

    #[test]
    fn build_llama_spawn_emits_ngram_for_qwen3_when_supported() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let entry = dummy_entry("qwen3", 32768);
        let s = build_llama_spawn(&entry, 8020, true, std::path::Path::new("/usr/bin/llama-server")).unwrap();
        let joined = s.args.join(" ");
        assert!(joined.contains("--spec-type ngram-mod"));
        assert!(joined.contains("--spec-ngram-mod-n-match 24"));
    }

    #[test]
    fn build_llama_spawn_omits_ngram_for_non_qwen() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let entry = dummy_entry("llama", 8192);
        let s = build_llama_spawn(&entry, 8020, true, std::path::Path::new("/usr/bin/llama-server")).unwrap();
        assert!(!s.args.join(" ").contains("--spec-type"));
    }

    // ── has_full_ngram_mod_flags ────────────────────────────────────

    #[test]
    fn ngram_probe_requires_all_three_subflags() {
        // Bee v0.1.2 help text contains all three.
        let bee_like = "\
            --spec-ngram-mod-n-min N    minimum number of ngram tokens\n\
            --spec-ngram-mod-n-max N    maximum number of ngram tokens\n\
            --spec-ngram-mod-n-match N  ngram-mod lookup length\n";
        assert!(has_full_ngram_mod_flags(bee_like));
    }

    #[test]
    fn ngram_probe_rejects_dflash_pr_style() {
        // dflash-pr branch advertises `ngram-mod` in --spec-type enum but
        // ships none of the supporting -n-* sub-flags. Probe must NOT
        // greenlight ngram-mod here — passing --spec-type ngram-mod
        // without -n-min/-n-max would start a broken speculator.
        let dflash_pr_like = "\
            --spec-type [none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]\n\
            --spec-ngram-size-n N       ngram size N for ngram-simple\n\
            --spec-ngram-size-m N       ngram size M for ngram-simple\n";
        assert!(!has_full_ngram_mod_flags(dflash_pr_like));
    }

    #[test]
    fn ngram_probe_rejects_partial_subflags() {
        // A future build that ships -n-min but not -n-match should also
        // be refused — we emit all three, so all three must be supported.
        let partial = "--spec-ngram-mod-n-min N    minimum\n\
                       --spec-ngram-mod-n-max N    maximum\n";
        assert!(!has_full_ngram_mod_flags(partial));
    }

    // ── BeeLlama / DFlash detection + spawn args ────────────────────

    #[test]
    fn is_bee_binary_matches_beellama_path() {
        let bee = std::path::Path::new("/home/u/local-llm/beellama.cpp/build/bin/llama-server");
        assert!(is_bee_binary(bee));
    }

    #[test]
    fn is_bee_binary_rejects_vanilla() {
        let vanilla = std::path::Path::new("/usr/bin/llama-server");
        assert!(!is_bee_binary(vanilla));
        let dflash_pr = std::path::Path::new("/home/u/llama.cpp/build/bin/llama-server");
        assert!(!is_bee_binary(dflash_pr));
    }

    #[test]
    fn is_bee_binary_rejects_substring_decoys() {
        // V4 Pro flagged: substring match `contains("beellama")` matched
        // attacker-controlled paths like `/tmp/notbeellama.cpp/x`.
        // Component match closes that hole — only a *directory named*
        // `beellama.cpp` in the path qualifies.
        let decoy = std::path::Path::new("/tmp/notbeellama.cpp/build/bin/llama-server");
        assert!(!is_bee_binary(decoy));
        let backup = std::path::Path::new("/home/u/beellama.cpp.backup/bin/llama-server");
        assert!(!is_bee_binary(backup));
        let prefix = std::path::Path::new("/home/u/beellama-fork/bin/llama-server");
        assert!(!is_bee_binary(prefix));
    }

    #[test]
    fn entry_has_usable_dflash_predicate() {
        let real = std::env::current_exe().expect("current_exe");
        let with_real = dummy_entry_with_spec("qwen35", real);
        assert!(entry_has_usable_dflash(&with_real));

        let with_missing = dummy_entry_with_spec("qwen35", PathBuf::from("/tmp/nope.gguf"));
        assert!(!entry_has_usable_dflash(&with_missing));

        let mut wrong_method = dummy_entry_with_spec("qwen35", std::env::current_exe().unwrap());
        wrong_method.speculative.as_mut().unwrap().method = "ngram".into();
        assert!(!entry_has_usable_dflash(&wrong_method));

        let no_spec = dummy_entry("qwen35", 32768);
        assert!(!entry_has_usable_dflash(&no_spec));
    }

    #[test]
    fn is_file_executable_rejects_dirs_and_regular_files() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // A directory is never executable for our purposes (we'd
        // `Command::new` it and the kernel would refuse).
        assert!(!is_file_executable(tmp.path()));
        // A regular file with no exec bit is rejected.
        let regular = tmp.path().join("plain.txt");
        std::fs::write(&regular, b"hi").unwrap();
        assert!(!is_file_executable(&regular));
        // Same file with exec bit set on Unix → accepted.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = std::fs::metadata(&regular).unwrap().permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(&regular, perm).unwrap();
            assert!(is_file_executable(&regular));
        }
    }

    fn dummy_entry_with_spec(arch: &str, draft: PathBuf) -> ModelEntry {
        let mut e = dummy_entry(arch, 32768);
        e.speculative = Some(crate::types::SpeculativeConfig {
            draft_path: draft,
            method: "dflash".into(),
            draft_max: 8,
        });
        e
    }

    #[test]
    fn build_llama_spawn_emits_dflash_on_bee_when_draft_exists() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        // Use the test binary itself as a stand-in for a real GGUF —
        // `Path::exists()` is what build_llama_spawn checks.
        let real_path = std::env::current_exe().expect("current_exe");
        let entry = dummy_entry_with_spec("qwen35", real_path.clone());
        let bee_path = PathBuf::from(
            "/home/u/local-llm/beellama.cpp/build/bin/llama-server"
        );
        let s = build_llama_spawn(&entry, 8020, true, &bee_path).unwrap();
        let joined = s.args.join(" ");
        assert!(joined.contains("--spec-dflash-default"), "{joined}");
        assert!(joined.contains(&format!("-md {}", real_path.display())), "{joined}");
        assert!(joined.contains("-ngld 99"), "{joined}");
        // DFlash > ngram on bee — must not also emit ngram-mod.
        assert!(!joined.contains("--spec-type ngram-mod"), "{joined}");
        // KV auto-upgrade to turbo3_tcq when LAMU_KV unset.
        assert!(joined.contains("--cache-type-k turbo3_tcq"), "{joined}");
        assert!(joined.contains("--cache-type-v turbo3_tcq"), "{joined}");
    }

    #[test]
    fn build_llama_spawn_skips_dflash_on_vanilla_binary() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let real_path = std::env::current_exe().expect("current_exe");
        let entry = dummy_entry_with_spec("qwen35", real_path);
        // Not under the beellama.cpp tree → not bee.
        let vanilla = PathBuf::from("/usr/bin/llama-server");
        let s = build_llama_spawn(&entry, 8020, true, &vanilla).unwrap();
        let joined = s.args.join(" ");
        assert!(!joined.contains("--spec-dflash-default"), "{joined}");
        // Default KV stays at q8_0 (no bee → no auto-turbo).
        assert!(joined.contains("--cache-type-k q8_0"), "{joined}");
    }

    #[test]
    fn build_llama_spawn_skips_dflash_when_draft_missing() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let entry = dummy_entry_with_spec("qwen35", PathBuf::from("/tmp/definitely-not-here.gguf"));
        let bee_path = PathBuf::from(
            "/home/u/local-llm/beellama.cpp/build/bin/llama-server"
        );
        let s = build_llama_spawn(&entry, 8020, false, &bee_path).unwrap();
        let joined = s.args.join(" ");
        assert!(!joined.contains("--spec-dflash-default"), "{joined}");
        // No DFlash → KV stays at the default q8_0.
        assert!(joined.contains("--cache-type-k q8_0"), "{joined}");
    }

    #[test]
    fn build_llama_spawn_honors_explicit_lamu_kv_over_turbo() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { std::env::set_var("LAMU_KV", "f16"); }
        let real_path = std::env::current_exe().expect("current_exe");
        let entry = dummy_entry_with_spec("qwen35", real_path);
        let bee_path = PathBuf::from(
            "/home/u/local-llm/beellama.cpp/build/bin/llama-server"
        );
        let s = build_llama_spawn(&entry, 8020, true, &bee_path).unwrap();
        clear_env();
        let joined = s.args.join(" ");
        // LAMU_KV=f16 must win even when bee+DFlash would auto-upgrade.
        assert!(joined.contains("--cache-type-k f16"), "{joined}");
        assert!(joined.contains("--cache-type-v f16"), "{joined}");
        // Still emits DFlash args.
        assert!(joined.contains("--spec-dflash-default"), "{joined}");
    }

    #[test]
    fn build_llama_spawn_accepts_turbo3_tcq_via_env() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { std::env::set_var("LAMU_KV", "turbo3_tcq"); }
        let entry = dummy_entry("llama", 8192);
        let s = build_llama_spawn(&entry, 8020, false, std::path::Path::new("/usr/bin/llama-server")).unwrap();
        clear_env();
        let joined = s.args.join(" ");
        assert!(joined.contains("--cache-type-k turbo3_tcq"), "{joined}");
    }
}
