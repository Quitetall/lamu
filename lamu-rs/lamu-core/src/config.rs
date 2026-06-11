//! Configuration constants. Port of `lamu/core/config.py`.

use std::path::PathBuf;

use crate::error::{Error, Result};

/// Parse env var `key` into `T`, falling back to `default`. Unset → default
/// silently; set-but-unparseable → default + a `warn!` (so a typo'd
/// `LAMU_DEFAULT_CTX=abc` doesn't silently choose a surprising value). Trims
/// surrounding whitespace.
pub fn parse_env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    match std::env::var(key) {
        Ok(v) => match v.trim().parse::<T>() {
            Ok(parsed) => parsed,
            Err(_) => {
                tracing::warn!(
                    "{key}='{v}' is not a valid {}; using default",
                    std::any::type_name::<T>()
                );
                default
            }
        },
        Err(_) => default,
    }
}

pub fn lamu_root() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join("local-llm")
}

pub fn models_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join("models")
}

/// Live model registry: `~/.local/share/lamu/models.yaml` — in the user data
/// dir, OUTSIDE the git work tree (ADR 0025), because every scan and load
/// status flip mutates it at runtime and kept dirtying the tree. Override
/// with `$LAMU_REGISTRY` (tests, sandboxes). Seeded on first run by
/// [`ensure_registry`].
pub fn registry_path() -> PathBuf {
    if let Ok(p) = std::env::var("LAMU_REGISTRY") {
        let t = p.trim();
        if !t.is_empty() {
            return PathBuf::from(t);
        }
    }
    // Explicit ~/.local/share fallback: a None data_dir must not collapse
    // to a *relative* path that scan would then write into the CWD.
    dirs::data_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".local").join("share")))
        .unwrap_or_default()
        .join("lamu")
        .join("models.yaml")
}

/// Tracked seed registry shipped in the repo: `config/models_default.yaml`.
/// Read-only from LAMU's perspective — the live copy is [`registry_path`].
pub fn registry_default_path() -> PathBuf {
    lamu_root().join("config").join("models_default.yaml")
}

/// Where the live registry lived before ADR 0025 (inside the work tree).
/// Only consulted as a migration source by [`ensure_registry`].
fn registry_legacy_path() -> PathBuf {
    lamu_root().join("config").join("models.yaml")
}

/// First-run seed / migration for the live registry. No-op when the live
/// file already exists. Otherwise copies, in preference order: the legacy
/// in-tree registry (preserves curated main/speculative/sampling/notes from
/// pre-ADR-0025 setups), else the tracked seed. Neither existing is fine —
/// the first `lamu scan` creates the file.
pub fn ensure_registry() -> Result<()> {
    seed_registry(
        &registry_path(),
        &[
            (registry_legacy_path(), "legacy in-tree registry"),
            (registry_default_path(), "tracked seed"),
        ],
    )
}

/// Pure-path core of [`ensure_registry`] (testable with tempdirs). Copies
/// the first existing source to `live` via a pid-suffixed temp file +
/// rename, so a crash or a concurrent first-run never leaves a half-written
/// registry at the live path.
fn seed_registry(live: &std::path::Path, sources: &[(PathBuf, &str)]) -> Result<()> {
    if live.exists() {
        return Ok(());
    }
    if let Some(parent) = live.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::Config(format!("create {}: {e}", parent.display())))?;
    }
    for (src, what) in sources {
        if !src.exists() {
            continue;
        }
        // Benign TOCTOU: two concurrent first-runs both pass the exists()
        // check, both copy the same source, and the atomic rename means the
        // loser overwrites the winner byte-for-byte. PID suffix keeps their
        // temp files from interleaving.
        let tmp = live.with_extension(format!("yaml.tmp.{}", std::process::id()));
        std::fs::copy(src, &tmp).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            Error::Config(format!("seed registry: copy {} failed: {e}", src.display()))
        })?;
        std::fs::rename(&tmp, live).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            Error::Config(format!("seed registry: rename to {} failed: {e}", live.display()))
        })?;
        tracing::info!(
            "seeded live registry {} from {} ({what})",
            live.display(),
            src.display()
        );
        return Ok(());
    }
    tracing::info!(
        "no registry seed found; {} will be created by the first scan",
        live.display()
    );
    Ok(())
}

/// Built-in grounding system prompt. A local model has strong reasoning but
/// limited / possibly-stale world knowledge, so by default it should LOOK THINGS
/// UP (the `web_search` / `research` tools) rather than answer factual questions
/// from parametric memory. Applied to a chat request that carries no system
/// message of its own; a request with its own system message is left untouched.
pub const GROUNDING_SYSTEM_PROMPT: &str = "You are a capable reasoning assistant running locally. Your built-in world knowledge is limited and may be outdated or wrong, so do NOT answer factual questions from memory — names, dates, numbers, current events, prices, library/API details, who-said-or-did-what. Instead use the available search/research tools to look them up, and ground your answer in what you actually find, with sources. If you cannot look something up and are not sure, say so plainly rather than guessing. Your reasoning, math, coding, and analysis of information you've been given are strong — use them freely.";

fn system_prompt_override_path() -> Option<PathBuf> {
    dirs::config_dir().map(|c| c.join("lamu").join("system_prompt.txt"))
}

/// Pure resolver (testable). Precedence: `LAMU_DISABLE_GROUNDING_PROMPT` set →
/// `None` (clean context, for eval/structured-output callers); else an override
/// file at `path` — its trimmed contents, or `None` if blank (explicit disable);
/// else the built-in [`GROUNDING_SYSTEM_PROMPT`].
fn resolve_system_prompt(path: Option<&std::path::Path>) -> Option<String> {
    if std::env::var_os("LAMU_DISABLE_GROUNDING_PROMPT").is_some() {
        return None;
    }
    if let Some(p) = path {
        if let Ok(s) = std::fs::read_to_string(p) {
            let t = s.trim();
            return if t.is_empty() { None } else { Some(t.to_string()) };
        }
    }
    Some(GROUNDING_SYSTEM_PROMPT.to_string())
}

/// The default system prompt applied to a chat request that omits one. Override
/// the text in `~/.config/lamu/system_prompt.txt`; make that file empty (or set
/// `LAMU_DISABLE_GROUNDING_PROMPT`) to disable. Cached on first call — the
/// override file is read once, so restart to change it (it's config, not a hot
/// value), keeping the per-request serve path allocation-free.
pub fn default_system_prompt() -> Option<String> {
    static CACHE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| resolve_system_prompt(system_prompt_override_path().as_deref()))
        .clone()
}

/// Root of the user's `llama.cpp` checkout. Resolution order:
///
///   1. `$LAMU_LLAMACPP_DIR` env var (explicit override).
///   2. `~/llama.cpp` (default for users who built from source in $HOME).
///
/// The path is returned regardless of whether it exists — callers
/// invoke `llama_tool` to actually resolve a binary inside it, and
/// `llama_tool` is the function that produces a clear error if the
/// dir is missing.
pub fn llamacpp_dir() -> PathBuf {
    if let Ok(p) = std::env::var("LAMU_LLAMACPP_DIR") {
        return PathBuf::from(p);
    }
    dirs::home_dir().unwrap_or_default().join("llama.cpp")
}

/// Locate a llama.cpp tool by name. Tries, in order:
///
///   1. `<llamacpp_dir>/build/bin/<name>` — standard cmake build layout
///   2. `<llamacpp_dir>/<name>` — flat layout / older builds
///   3. `<name>` on `$PATH` (via `which`)
///
/// Returns `Error::Config` with a self-explanatory message naming the
/// env var if all three miss — gives the user one sentence to fix it.
pub fn llama_tool(name: &str) -> Result<PathBuf> {
    let base = llamacpp_dir();
    let candidates = [base.join("build").join("bin").join(name), base.join(name)];
    for c in candidates {
        if c.exists() {
            return Ok(c);
        }
    }
    if let Ok(p) = which::which(name) {
        return Ok(p);
    }
    Err(Error::Config(format!(
        "llama.cpp tool '{name}' not found in {} or on $PATH. \
         Set $LAMU_LLAMACPP_DIR to your llama.cpp checkout.",
        base.display()
    )))
}

/// Back-compat: the original API. Returns the resolved llama-server
/// path, falling back to the historical hardcoded location if
/// `llama_tool` can't find it. Spawning a non-existent path produces
/// a clear OS error from the backend layer, so the fallback is fine.
pub fn llama_bin() -> PathBuf {
    llama_tool("llama-server").unwrap_or_else(|_| {
        dirs::home_dir()
            .unwrap_or_default()
            .join("llama.cpp")
            .join("build")
            .join("bin")
            .join("llama-server")
    })
}

pub const PORT_MAIN: u16 = 8020;
pub const PORT_SIDECAR: u16 = 8001;
pub const PORT_DFLASH: u16 = 8000;

pub const VRAM_RESERVED_MB: u32 = 1500;
pub const DEFAULT_MAX_TOKENS: u32 = 16384;

/// Which CUDA device LAMU uses (NVML monitoring + backend `CUDA_VISIBLE_DEVICES`).
/// `LAMU_GPU_INDEX`, default 0. The single-card zero-config path is unchanged
/// (0 → device 0 → `CUDA_VISIBLE_DEVICES=0`, byte-identical to before). This is
/// the ADR-0014-named seam and multi-GPU P0 (ADR 0017); full per-device
/// placement builds on top of it. Pin a specific card with e.g.
/// `LAMU_GPU_INDEX=1`.
pub fn gpu_index() -> u32 {
    parse_env_or("LAMU_GPU_INDEX", 0)
}

/// Host the spawned inference backends bind to (M8). Defaults to loopback
/// `127.0.0.1` — matching lamu-api's bind default + the llama-server backend —
/// so a backend is never reachable off-box unless the operator explicitly opts
/// in via `LAMU_BIND_HOST=0.0.0.0` (the same env lamu-api's off-loopback auth
/// gate keys on). Without this the Python servers (dflash/megakernel) defaulted
/// to `0.0.0.0` with no auth.
pub fn backend_bind_host() -> String {
    std::env::var("LAMU_BIND_HOST")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

/// Whether the HTTP serve path may auto-evict its OWN resident models to make
/// room for an on-demand load. OFF by default (ADR 0006): on a shared GPU a
/// request must never surprise-kill a model another client uses, and lamu must
/// never touch VRAM it didn't allocate (e.g. a training job). A single-user
/// desktop (Odysseus) opts in with `LAMU_HTTP_AUTOEVICT=1` so selecting an
/// inactive model just loads it instead of returning 503. Accepts 1/true/yes/on
/// (case-insensitive); anything else (incl. unset) is false.
pub fn http_autoevict() -> bool {
    env_truthy(&std::env::var("LAMU_HTTP_AUTOEVICT").unwrap_or_default())
}

/// Parse a boolean-ish env value: 1/true/yes/on (case-insensitive) → true,
/// everything else (incl. empty) → false. Pure so it's testable without
/// mutating process-global env (which races across the test runner).
fn env_truthy(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod autoevict_tests {
    use super::env_truthy;

    #[test]
    fn env_truthy_accepts_canonical_true_values() {
        for v in ["1", "true", "TRUE", "yes", "On", "  on  "] {
            assert!(env_truthy(v), "{v:?} should be truthy");
        }
    }

    #[test]
    fn env_truthy_rejects_everything_else() {
        for v in ["", "0", "false", "no", "off", "2", "enabled", "y"] {
            assert!(!env_truthy(v), "{v:?} should be falsey");
        }
    }
}
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_CTX_SIZE: u32 = 131072;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serializes env-var mutations across tests in this module so
    // parallel test execution can't race on $LAMU_LLAMACPP_DIR.
    // Other test binaries in the workspace are separate processes,
    // so the lock only needs to cover this module.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn grounding_resolver_precedence() {
        let _g = ENV_LOCK.lock().unwrap(); // resolver reads LAMU_DISABLE_GROUNDING_PROMPT
        // Absent override file -> the built-in grounding default.
        let p = resolve_system_prompt(None).expect("absent file -> built-in default");
        assert_eq!(p, GROUNDING_SYSTEM_PROMPT);
        assert!(p.contains("look") || p.contains("search"), "must nudge lookups");
        assert!(p.to_lowercase().contains("memory") || p.to_lowercase().contains("knowledge"));

        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("system_prompt.txt");
        // Custom file -> its trimmed contents.
        std::fs::write(&f, "  custom house rules  ").unwrap();
        assert_eq!(resolve_system_prompt(Some(&f)).as_deref(), Some("custom house rules"));
        // Blank file -> disabled (None).
        std::fs::write(&f, "   \n").unwrap();
        assert_eq!(resolve_system_prompt(Some(&f)), None);
    }

    #[test]
    fn ports_distinct() {
        assert_ne!(PORT_MAIN, PORT_SIDECAR);
        assert_ne!(PORT_MAIN, PORT_DFLASH);
        assert_ne!(PORT_SIDECAR, PORT_DFLASH);
    }

    #[test]
    fn ports_in_user_range() {
        for p in [PORT_MAIN, PORT_SIDECAR, PORT_DFLASH] {
            assert!(p >= 1024 && p < u16::MAX);
        }
    }

    #[test]
    fn registry_path_in_data_dir_not_work_tree() {
        let _g = ENV_LOCK.lock().unwrap(); // registry_path reads LAMU_REGISTRY
        let prev = std::env::var("LAMU_REGISTRY").ok();
        unsafe { std::env::remove_var("LAMU_REGISTRY") };
        let reg = registry_path();
        // ADR 0025: the live registry must NOT resolve into the git work tree.
        assert!(!reg.starts_with(lamu_root()), "live registry in work tree: {reg:?}");
        assert!(reg.ends_with("lamu/models.yaml"), "unexpected layout: {reg:?}");
        // The override wins verbatim.
        unsafe { std::env::set_var("LAMU_REGISTRY", "/tmp/override.yaml") };
        assert_eq!(registry_path(), PathBuf::from("/tmp/override.yaml"));
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_REGISTRY", v),
                None => std::env::remove_var("LAMU_REGISTRY"),
            }
        }
    }

    #[test]
    fn seed_registry_prefers_legacy_then_default_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("data").join("models.yaml");
        let legacy = dir.path().join("legacy.yaml");
        let seed = dir.path().join("default.yaml");
        std::fs::write(&seed, "from-seed").unwrap();

        // Only the tracked seed exists -> seeded from it (parent dir created).
        seed_registry(&live, &[(legacy.clone(), "legacy"), (seed.clone(), "seed")]).unwrap();
        assert_eq!(std::fs::read_to_string(&live).unwrap(), "from-seed");

        // Live exists -> no-op even though legacy now appears.
        std::fs::write(&legacy, "from-legacy").unwrap();
        seed_registry(&live, &[(legacy.clone(), "legacy"), (seed.clone(), "seed")]).unwrap();
        assert_eq!(std::fs::read_to_string(&live).unwrap(), "from-seed");

        // Fresh live path with both present -> legacy (curated entries) wins.
        let live2 = dir.path().join("data2").join("models.yaml");
        seed_registry(&live2, &[(legacy, "legacy"), (seed, "seed")]).unwrap();
        assert_eq!(std::fs::read_to_string(&live2).unwrap(), "from-legacy");
    }

    #[test]
    fn seed_registry_no_sources_is_ok_and_creates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("models.yaml");
        let ghost = dir.path().join("ghost.yaml");
        seed_registry(&live, &[(ghost, "ghost")]).unwrap();
        assert!(!live.exists(), "no source must mean no live file");
    }

    #[test]
    fn defaults_sane() {
        assert!(DEFAULT_MAX_TOKENS > 0);
        assert!(DEFAULT_TEMPERATURE > 0.0 && DEFAULT_TEMPERATURE <= 2.0);
        assert!(DEFAULT_CTX_SIZE >= 4096);
        assert!(VRAM_RESERVED_MB > 0);
    }

    #[test]
    fn paths_are_pathbufs() {
        let _ = lamu_root();
        let _ = models_dir();
        let _ = registry_path();
        let _ = llama_bin();
    }

    #[test]
    fn llamacpp_dir_respects_env() {
        let _g = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("LAMU_LLAMACPP_DIR").ok();
        // SAFETY: tests in this module serialize on ENV_LOCK; other
        // test binaries are separate processes.
        unsafe { std::env::set_var("LAMU_LLAMACPP_DIR", "/opt/custom/llama.cpp") };
        assert_eq!(llamacpp_dir(), PathBuf::from("/opt/custom/llama.cpp"));
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_LLAMACPP_DIR", v),
                None => std::env::remove_var("LAMU_LLAMACPP_DIR"),
            }
        }
    }

    #[test]
    fn llamacpp_dir_defaults_to_home() {
        let _g = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("LAMU_LLAMACPP_DIR").ok();
        unsafe { std::env::remove_var("LAMU_LLAMACPP_DIR") };
        let p = llamacpp_dir();
        assert!(p.ends_with("llama.cpp"));
        unsafe {
            if let Some(v) = prev {
                std::env::set_var("LAMU_LLAMACPP_DIR", v);
            }
        }
    }

    #[test]
    fn llama_tool_finds_via_build_bin() {
        let _g = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("LAMU_LLAMACPP_DIR").ok();
        let dir = tempfile::tempdir().expect("tempdir");
        let bin = dir.path().join("build").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let tool = bin.join("fake-tool");
        std::fs::write(&tool, b"#!/bin/sh\nexit 0\n").unwrap();
        unsafe {
            std::env::set_var("LAMU_LLAMACPP_DIR", dir.path());
        }
        let resolved = llama_tool("fake-tool").expect("must resolve");
        assert_eq!(resolved, tool);
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_LLAMACPP_DIR", v),
                None => std::env::remove_var("LAMU_LLAMACPP_DIR"),
            }
        }
    }

    #[test]
    fn llama_tool_finds_flat_layout() {
        let _g = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("LAMU_LLAMACPP_DIR").ok();
        let dir = tempfile::tempdir().expect("tempdir");
        let tool = dir.path().join("flat-tool");
        std::fs::write(&tool, b"x").unwrap();
        unsafe {
            std::env::set_var("LAMU_LLAMACPP_DIR", dir.path());
        }
        let resolved = llama_tool("flat-tool").expect("must resolve");
        assert_eq!(resolved, tool);
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_LLAMACPP_DIR", v),
                None => std::env::remove_var("LAMU_LLAMACPP_DIR"),
            }
        }
    }

    #[test]
    fn llama_tool_errors_with_helpful_message() {
        let _g = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("LAMU_LLAMACPP_DIR").ok();
        let dir = tempfile::tempdir().expect("tempdir");
        unsafe {
            std::env::set_var("LAMU_LLAMACPP_DIR", dir.path());
        }
        let err = llama_tool("definitely-not-a-real-tool").expect_err("must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("LAMU_LLAMACPP_DIR"),
            "error must name the env var: {msg}"
        );
        assert!(
            msg.contains("definitely-not-a-real-tool"),
            "error must name the missing tool: {msg}"
        );
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_LLAMACPP_DIR", v),
                None => std::env::remove_var("LAMU_LLAMACPP_DIR"),
            }
        }
    }
}
