//! Backends — model lifecycle management.
//! Direct port of `lamu/backends/`.

pub mod dflash;
pub mod fish_speech;
pub mod llamacpp;
pub mod megakernel;

use crate::types::{BackendType, DevicePlacement, ModelEntry};
use crate::{Error, Result};
use async_trait::async_trait;
use futures_util::stream::Stream;
use serde_json::{json, Value};
use std::pin::Pin;

/// Construct the right backend impl for the entry's declared type.
///
/// For `LlamaCpp`, uses `new_for_entry` so registry entries carrying a
/// `speculative: { method: dflash, ... }` config transparently spawn
/// the BeeLlama fork binary with DFlash drafter + `turbo3_tcq` KV.
/// Entries without spec config get the generic `llama_bin()`.
pub fn make_backend(entry: &ModelEntry) -> Result<Box<dyn Backend>> {
    match entry.backend {
        BackendType::LlamaCpp => Ok(Box::new(llamacpp::LlamaCppBackend::new_for_entry(entry)?)),
        BackendType::Megakernel => Ok(Box::new(megakernel::MegakernelBackend::new()?)),
        BackendType::Dflash | BackendType::DflashLucebox => {
            Ok(Box::new(dflash::DflashBackend::new()?))
        }
        BackendType::FishSpeech => Ok(Box::new(fish_speech::FishSpeechBackend::new()?)),
        // ADR 0023: ComfyUI moved to the `lamu-image` module. Core no longer
        // names it — it resolves via the backend registry that the module
        // populates at the composition root. As lamu-tts/lamu-jart land, their
        // BackendType variants route here the same way (-> make_registered(kind)).
        BackendType::ComfyUI => make_registered("comfyui", entry),
    }
}

/// Factory for a module-provided backend kind (ADR 0023).
pub type BackendFactory = fn(&ModelEntry) -> Result<Box<dyn Backend>>;

static BACKEND_REGISTRY: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, BackendFactory>>,
> = std::sync::OnceLock::new();

fn backend_registry(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, BackendFactory>> {
    BACKEND_REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Register a backend kind from a module (ADR 0023). Each module's `register()`
/// calls this once at the binary's startup (the composition root); `make_backend`
/// then resolves that `backend_kind` here instead of core naming the module.
pub fn register_backend(kind: &str, factory: BackendFactory) {
    backend_registry()
        .lock()
        .expect("backend registry poisoned")
        .insert(kind.to_string(), factory);
}

fn make_registered(kind: &str, entry: &ModelEntry) -> Result<Box<dyn Backend>> {
    let f = backend_registry()
        .lock()
        .expect("backend registry poisoned")
        .get(kind)
        .copied();
    match f {
        Some(factory) => factory(entry),
        None => Err(crate::Error::Backend(format!(
            "backend kind '{kind}' is not registered — its module was not loaded at startup (ADR 0023)"
        ))),
    }
}

/// Per-call backend options. Backends without a corresponding feature
/// silently ignore unknown fields.
#[derive(Debug, Clone, Default)]
pub struct GenerateOpts {
    /// Qwen3.6 / Qwen3.5 reasoning toggle. `Some(false)` disables the
    /// `<think>` block via `chat_template_kwargs.enable_thinking`. `None`
    /// leaves the model's default behaviour (thinking on).
    pub enable_thinking: Option<bool>,
    /// Nucleus sampling cutoff. `None` = leave the backend/server default.
    pub top_p: Option<f32>,
    /// Top-k truncation (integer count). `None` = server default.
    pub top_k: Option<u32>,
    /// Min-p sampling cutoff. `None` = server default.
    pub min_p: Option<f32>,
    /// Repetition penalty. `None` = server default.
    pub repeat_penalty: Option<f32>,
}

#[async_trait]
pub trait Backend: Send + Sync {
    /// Load model. Returns PID.
    async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32>;

    /// Pin this backend's next spawn to a specific GPU placement (ADR 0017
    /// P2). Backends that spawn a CUDA child override this to set
    /// `CUDA_VISIBLE_DEVICES` to the placement's primary NVML index. The
    /// loader calls it AFTER the scheduler picks a device and BEFORE
    /// `load`, so the child process lands on the placed card. Default is a
    /// no-op for CPU/proxy backends. `Sharded` is treated as its primary
    /// index for now (single-device spawn); true multi-device split is a
    /// later phase.
    fn set_device(&mut self, _placement: DevicePlacement) {}

    /// Stop process and free VRAM.
    async fn unload(&mut self) -> Result<()>;

    /// Health check.
    async fn is_healthy(&self) -> bool;

    /// Generate non-streaming. Returns raw text (think blocks included).
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String>;

    /// Generate non-streaming with extended options. Default impl ignores
    /// `opts` and forwards to `generate()`. Backends override to honor
    /// per-call params like `enable_thinking`.
    async fn generate_with_opts(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        _opts: GenerateOpts,
    ) -> Result<String> {
        self.generate(messages, max_tokens, temperature).await
    }

    /// Generate streaming. Yields tokens.
    async fn stream(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>;

    fn port(&self) -> u16;
    fn model_name(&self) -> &str;

    /// Count tokens in `text` using the BACKEND's own tokenizer — the engine's
    /// out-of-band accounting, which the generating model cannot fabricate
    /// (ADR 0021). Default: unsupported (proxy/CPU/TTS/image backends).
    /// `LlamaCppBackend` implements it via `POST /tokenize`. Returns the exact
    /// token count; callers divide by the model's real `n_ctx_train` for an
    /// un-fakeable context-occupancy ratio.
    async fn tokenize_count(&self, _text: &str) -> Result<u32> {
        Err(crate::Error::Backend(
            "tokenize_count unsupported by this backend".into(),
        ))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Harden a spawned backend child so it can NEVER outlive lamu and leak
/// its VRAM — WITHOUT killing it when the owning handle is merely dropped.
///
/// Mechanism: `PR_SET_PDEATHSIG(SIGKILL)` via `pre_exec`. The kernel
/// SIGKILLs the child the instant lamu (its parent) dies by ANY means —
/// graceful SIGTERM/EOF, a hard crash, `SIGKILL`, or the orphan-
/// watchdog's `std::process::exit(0)` (none of which need to run
/// destructors). That closes the VRAM-leak-on-lamu-death hole.
///
/// We deliberately DO NOT set `kill_on_drop(true)`: lamu-api (`lamu
/// serve`) intentionally drops the `Box<dyn Backend>` right after load
/// and then proxies to the still-running llama-server by its port (see
/// loader.rs). kill_on_drop would SIGKILL that server on the drop,
/// leaving a phantom "loaded" entry pointing at a dead port. PDEATHSIG
/// is independent of the handle — it's a property of the child process,
/// so dropping the handle never clears it; the child still dies with
/// lamu. lamu-mcp's explicit unload/shutdown-drain handles in-session
/// teardown, and PDEATHSIG backstops every abnormal exit.
///
/// This is UNCONDITIONAL — unlike lamu's own PDEATHSIG
/// (lamu-core/src/lifecycle.rs), which respects a `nohup` SIGHUP=SIG_IGN
/// "survive my parent" marker. A backend must never survive its lamu.
///
/// Non-Linux (incl. macOS): no-op — there is no portable
/// `PR_SET_PDEATHSIG` equivalent. lamu's backends are CUDA/Linux-only in
/// practice, so the gap is theoretical.
pub fn harden_child_command(cmd: &mut tokio::process::Command) {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `pre_exec` runs in the forked child between fork and
        // exec, where only async-signal-safe calls are permitted. We use
        // only `prctl` and `getppid`, both async-signal-safe. The
        // `getppid() == 1` re-check closes the classic fork-vs-parent-
        // death race: if lamu already died before this ran, the
        // PDEATHSIG would be relative to init (pid 1) and never fire, so
        // we fail the exec instead of spawning an immortal orphan.
        unsafe {
            cmd.pre_exec(|| {
                // New session → the child leads its own process group. That
                // makes `unload` able to signal the WHOLE group (negative pid)
                // so a python server's forked workers / CUDA helper children
                // die with it instead of leaking as VRAM-holding orphans. Only
                // a group LEADER can be group-signalled safely; without setsid
                // the child shares lamu's group and a group signal would hit
                // lamu. setsid() is async-signal-safe. It fails (EPERM) only if
                // the caller already leads a group — never true for a fresh
                // fork — so a failure here is genuinely unexpected: surface it.
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong, 0, 0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() == 1 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "lamu (parent) died before PDEATHSIG armed",
                    ));
                }
                Ok(())
            });
        }
    }
    // Reference cmd on non-Linux so the param isn't flagged unused.
    #[cfg(not(target_os = "linux"))]
    let _ = cmd;
}

/// True when `pid` leads its own process group (i.e. it was spawned with
/// `setsid` via `harden_child_command`). ONLY a group leader may be safely
/// signalled as a group (negative pid); a non-leader shares lamu's group, so
/// a group signal there would take down lamu itself.
#[cfg(unix)]
fn is_group_leader(pid: u32) -> bool {
    // getpgid(pid) == pid  ⟺  pid is the leader of its own group.
    unsafe { libc::getpgid(pid as i32) == pid as i32 }
}

/// Send `sig` to the backend. When the child leads its own group (the normal
/// case — `harden_child_command` setsids it) signal the WHOLE group via a
/// negative pid so python workers / CUDA helper children die with it.
/// Otherwise fall back to the single pid — NEVER signal a shared group.
#[cfg(unix)]
fn signal_backend(pid: u32, sig: nix::sys::signal::Signal) {
    let target = if is_group_leader(pid) {
        -(pid as i32) // whole process group
    } else {
        pid as i32 // single process — shared group, must not broadcast
    };
    if let Err(e) = nix::sys::signal::kill(nix::unistd::Pid::from_raw(target), sig) {
        // ESRCH (process already gone) is the common, harmless case — the
        // subsequent child.wait() is the authority. Log at trace so a genuine
        // EPERM stays visible without spamming routine teardown.
        tracing::trace!(pid, ?sig, "signal_backend: kill returned {e}");
    }
}

/// True once nothing accepts a TCP connection on `127.0.0.1:port` — proof the
/// backend's listening socket is released and no surviving child still holds
/// it. Listening sockets don't linger in TIME_WAIT, so a dead server refuses
/// connections within milliseconds; we poll briefly to absorb teardown lag.
///
/// Scope: this confirms the LISTENING SOCKET is gone. The preceding group
/// SIGKILL is what actually reaps grandchildren — this is a secondary, end-to-
/// end confirmation. It cannot see a process holding VRAM without a socket
/// (already killed by the group signal) and conservatively treats an unrelated
/// process that grabbed the same port in the teardown window as "still bound"
/// (caller retries) rather than a false all-clear.
#[cfg(unix)]
async fn port_released(port: u16) -> bool {
    use std::time::Duration;
    for _ in 0..15 {
        match tokio::time::timeout(
            Duration::from_millis(200),
            tokio::net::TcpStream::connect(("127.0.0.1", port)),
        )
        .await
        {
            // Handshake completed → something is still listening. Wait, retry.
            Ok(Ok(_stream)) => tokio::time::sleep(Duration::from_millis(200)).await,
            // Connection refused (Ok(Err)) or connect timed out (Err) → the
            // listener is gone. On loopback a refusal is immediate; a timeout
            // means unreachable, which for our purposes is "not bound".
            _ => return true,
        }
    }
    false
}

/// Graceful, VERIFIED child teardown: SIGTERM the backend's process group,
/// wait up to 10s for the direct child to exit (reaping it — no zombie),
/// escalate to a group SIGKILL + 5s wait if it ignores TERM, then confirm
/// the port is released so a surviving grandchild can't keep holding VRAM.
///
/// Returns `Err` if the process won't die or the port stays bound — the
/// caller MUST treat that as "still loaded" and NOT flip scheduler state,
/// so we never report a model dead while its backend is alive. This is the
/// load-bearing half of "model lifecycle is impossible to get wrong".
///
/// Why 10s: llama-server exits in <1s; megakernel/dflash/fish python servers
/// can take 2-5s to flush. 10s leaves margin without hanging the caller when
/// a child genuinely refuses to die.
#[cfg(unix)]
pub async fn graceful_kill(child: &mut tokio::process::Child, port: u16) -> Result<()> {
    use std::time::Duration;
    // We hold the unreaped Child for the whole function, so the kernel cannot
    // recycle `pid` before our wait() below — the getpgid/kill in
    // signal_backend are race-free with respect to PID reuse.
    let pid = child.id();
    if let Some(p) = pid {
        signal_backend(p, nix::sys::signal::Signal::SIGTERM);
    }
    // child.wait() reaps the direct child; its return is authoritative proof
    // that process exited (no PID-reuse ambiguity).
    let reaped = match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(_) => true,
        Err(_) => {
            tracing::warn!("graceful_kill: 10s SIGTERM timeout, escalating to SIGKILL");
            if let Some(p) = pid {
                signal_backend(p, nix::sys::signal::Signal::SIGKILL);
            }
            tokio::time::timeout(Duration::from_secs(5), child.wait())
                .await
                .is_ok()
        }
    };
    if !reaped {
        return Err(Error::Backend(format!(
            "backend pid {pid:?} survived SIGTERM+SIGKILL (15s) — refusing to report it dead"
        )));
    }
    // Direct child reaped. Confirm no surviving grandchild still holds the
    // port before the caller is allowed to mark the model unloaded.
    if port != 0 && !port_released(port).await {
        return Err(Error::Backend(format!(
            "backend pid {pid:?} exited but port {port} is still bound — a child process survived the kill"
        )));
    }
    Ok(())
}

#[cfg(not(unix))]
pub async fn graceful_kill(child: &mut tokio::process::Child, _port: u16) -> Result<()> {
    // No SIGTERM/process groups on non-Unix; best-effort kill + reap.
    child
        .kill()
        .await
        .map_err(|e| Error::Backend(format!("kill: {e}")))
}

/// Kill a backend we only know by `(pid, port)` — the HTTP serve path drops
/// the `Box<dyn Backend>` after spawn (loader.rs), so when an operator later
/// unloads such a model via MCP there is no `Child` handle to `wait()` on.
///
/// The PORT is the identity anchor, which makes this safe against PID reuse:
/// we signal `pid` ONLY while the port is still bound (i.e. the backend is
/// provably alive, so `pid` is still the backend — not a recycled, unrelated
/// process). The instant the port goes silent we stop and report success; we
/// never escalate to SIGKILL against a pid whose backend already died.
///
/// Returns Err if the port stays bound after SIGTERM+SIGKILL — caller MUST
/// treat that as "still loaded".
///
/// Residual TOCTOU: between `port_released` returning false and the signal,
/// the backend could die and `pid` be recycled. The window is microseconds
/// AND, while the port is still bound, the backend is alive, so `pid` is the
/// backend; a stray group-signal in that exact sliver lands on a not-yet-
/// recycled pid and is harmless. Bounded but not zero — the handle path
/// (graceful_kill, which wait()s an owned Child) is preferred when available.
#[cfg(unix)]
pub async fn kill_pid_and_verify(pid: u32, port: u16) -> Result<()> {
    if port == 0 {
        // No port to anchor on (a Loaded model always has a real port). We
        // can't VERIFY death without it, so fail closed rather than report a
        // kill we didn't confirm — the caller keeps the model marked loaded.
        return Err(Error::Backend(
            "cannot verify backend death without a port — refusing to report it killed".into(),
        ));
    }
    // Already dead? Port silent ⇒ nothing to kill.
    if port_released(port).await {
        return Ok(());
    }
    // Port bound ⇒ backend alive ⇒ `pid` is the backend. SIGTERM the group.
    signal_backend(pid, nix::sys::signal::Signal::SIGTERM);
    if port_released(port).await {
        return Ok(());
    }
    // Still bound ⇒ still alive ⇒ `pid` still valid. Escalate to group SIGKILL.
    signal_backend(pid, nix::sys::signal::Signal::SIGKILL);
    if port_released(port).await {
        return Ok(());
    }
    Err(Error::Backend(format!(
        "pid {pid}: port {port} still bound after SIGTERM+SIGKILL — backend would not die"
    )))
}

#[cfg(not(unix))]
pub async fn kill_pid_and_verify(_pid: u32, _port: u16) -> Result<()> {
    Err(Error::Backend(
        "kill_pid_and_verify unsupported on non-unix".into(),
    ))
}

/// Last ~2 KiB of a captured backend log — surfaces WHY a python-server
/// spawn failed (CUDA OOM, missing dep, bad checkpoint) in the error.
/// Empty if unreadable. Shared by the fish_speech + comfyui backends.
pub fn read_log_tail(path: &std::path::Path) -> String {
    match std::fs::read(path) {
        Ok(bytes) => {
            let start = bytes.len().saturating_sub(2048);
            String::from_utf8_lossy(&bytes[start..]).trim().to_string()
        }
        Err(_) => String::new(),
    }
}

/// OpenAI-compat chat payload + optional sampler overrides (only emitted
/// when `Some`, no nulls). Unknown fields are ignored server-side, so it's
/// a safe no-op where unsupported. Shared by the dflash + megakernel
/// custom-server backends.
pub fn build_payload(
    messages: &[ChatMessage],
    max_tokens: u32,
    temperature: f32,
    stream: bool,
    opts: &GenerateOpts,
) -> Value {
    let mut payload = json!({
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    });
    if let Some(v) = opts.top_p { payload["top_p"] = json!(v); }
    if let Some(v) = opts.top_k { payload["top_k"] = json!(v); }
    if let Some(v) = opts.min_p { payload["min_p"] = json!(v); }
    if let Some(v) = opts.repeat_penalty { payload["repeat_penalty"] = json!(v); }
    payload
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_opts_default_all_none() {
        let o = GenerateOpts::default();
        assert_eq!(o.enable_thinking, None);
        assert_eq!(o.top_p, None);
        assert_eq!(o.top_k, None);
        assert_eq!(o.min_p, None);
        assert_eq!(o.repeat_penalty, None);
    }

    #[test]
    fn generate_opts_carries_new_sampler_fields() {
        let o = GenerateOpts {
            enable_thinking: Some(false),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            repeat_penalty: Some(1.1),
        };
        assert_eq!(o.top_p, Some(0.9));
        assert_eq!(o.top_k, Some(40));
        assert_eq!(o.min_p, Some(0.05));
        assert_eq!(o.repeat_penalty, Some(1.1));
        assert_eq!(o.enable_thinking, Some(false));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn harden_child_command_setsids_into_own_group() {
        use std::time::Duration;
        let mut cmd = tokio::process::Command::new("sleep");
        cmd.arg("30");
        harden_child_command(&mut cmd);
        let mut child = cmd.spawn().expect("spawn sleep");
        let pid = child.id().expect("pid");
        // spawn() returns after fork; the child may still be in pre_exec
        // running setsid, so poll briefly rather than racing it.
        let mut leader = false;
        for _ in 0..100 {
            if is_group_leader(pid) {
                leader = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(leader, "harden_child_command must setsid the child into its own process group");
        let _ = graceful_kill(&mut child, 0).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn graceful_kill_reaps_sigterm_respecting_child() {
        // `sleep` exits on SIGTERM, so graceful_kill should reap it well
        // inside the 10s window and report Ok (port 0 → skip port check).
        let mut cmd = tokio::process::Command::new("sleep");
        cmd.arg("60");
        harden_child_command(&mut cmd);
        let mut child = cmd.spawn().expect("spawn sleep");
        let r = graceful_kill(&mut child, 0).await;
        assert!(r.is_ok(), "graceful_kill must reap a SIGTERM-respecting child: {r:?}");
        // Reaped: the handle no longer refers to a live process.
        assert!(matches!(child.try_wait(), Ok(Some(_)) | Err(_)));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn port_released_true_when_nothing_listening() {
        // Bind then drop → the listening socket is gone → connections refused.
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        drop(l);
        assert!(port_released(port).await, "a port with no listener must read as released");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn kill_pid_and_verify_ok_when_port_already_silent() {
        // Backend already gone (port silent): the port-anchor early return
        // reports success WITHOUT signalling any pid — so a recycled/bogus pid
        // is never touched.
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        drop(l);
        let r = kill_pid_and_verify(999_999_999, port).await;
        assert!(r.is_ok(), "silent port ⇒ already dead ⇒ Ok: {r:?}");
    }
}
