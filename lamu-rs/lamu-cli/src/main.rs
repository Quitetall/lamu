//! LAMU CLI entry point. Port of `lamu/daemon.py`.

mod chat_tui;
mod cloud_models;
mod favorites;
mod lamu_config;
mod mcp_servers;
mod md_stream;
mod providers;
mod repl;
mod sandbox;
mod theme;
mod tui;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lamu_core::config::{models_dir, registry_path, PORT_MAIN, PORT_SIDECAR};
use lamu_core::registry::{load_registry, scan_directory, write_registry};
use lamu_core::scheduler::VramScheduler;
use lamu_mcp::server::LamuMcpServer;
use serde_json::Value;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "lamu", version, about = "LAMU — MCP-first model management")]
struct Cli {
    /// Subcommand. Bare `lamu` (no subcommand) drops into the TUI dashboard.
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Discover models on disk, write registry
    Scan,
    /// Show running models + VRAM
    Status,
    /// Boot MCP server (stdio)
    Start,
    /// Boot OpenAI-compat HTTP server
    Serve {
        #[arg(short, long, default_value_t = 8020)]
        port: u16,
    },
    /// Interactive chat REPL talking to a running `lamu serve`.
    Repl {
        /// OpenAI-compat URL. Defaults to localhost:8020/v1/chat/completions.
        #[arg(default_value = "http://localhost:8020/v1/chat/completions")]
        api_url: String,
    },
    /// Load + chat with a model in one shot (Ollama-shaped).
    Run {
        /// Model name or substring. Resolved against the local registry.
        model: String,
    },
    /// Download a model from HuggingFace into ~/models/.
    Pull {
        /// Shorthand id (e.g. `qwen36-27b`) or `org/repo`.
        model: String,
        /// Quant suffix when the shorthand resolves to a multi-quant repo.
        #[arg(short, long, default_value = "Q4_K_M")]
        quant: String,
    },
    /// Show one model's registry entry as YAML.
    Show {
        /// Model name or substring.
        model: String,
    },
    /// Remove a model from the registry and delete its file on disk.
    Rm {
        /// Model name or substring.
        model: String,
        /// Skip the confirmation prompt.
        #[arg(short, long)]
        yes: bool,
    },
    /// List recent chat sessions (git snapshots).
    Sessions,
    /// Restore a session's git snapshot — undoes any changes since
    /// that session started.
    Undo {
        /// Session id (from `lamu sessions`). Defaults to the most recent.
        session_id: Option<String>,
        /// Skip the confirmation prompt.
        #[arg(short, long)]
        yes: bool,
    },
    /// Replay an agent's filesystem journal in reverse, restoring
    /// each modified file to its pre-session bytes.
    Rollback {
        /// Session id (from `lamu sessions`).
        session_id: String,
    },
    /// Run a command inside the lamu sandbox (bubblewrap/firejail).
    /// Strict bind mounts, no network, no $HOME access.
    Agent {
        /// Allow outbound network access (default: no network).
        #[arg(long)]
        net: bool,
        /// Command to run. Use `--` before flags meant for the command.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        cmd: Vec<String>,
    },
    /// List active per-agent worktrees with last-checkpoint and stats.
    Agents,
    /// Squash an agent worktree's branch into a single commit on the
    /// current branch — folds the agent's hard-earned work into main
    /// without losing it.
    Preserve {
        /// Session id (from `lamu agents`).
        session_id: String,
    },
    /// Selectively pull files matching a glob from an agent worktree
    /// onto the current branch. Use to keep only the parts you want.
    CherryPick {
        session_id: String,
        glob: String,
    },
    /// Drop an agent worktree + branch. Use after `preserve` (or when
    /// the agent's work is unwanted). Main was never touched, so this
    /// is non-destructive to the main branch.
    DropAgent {
        session_id: String,
    },
    /// Store cloud-provider API keys in ~/.config/lamu/api-keys.env (sourced
    /// into the env at startup by `lamu`, `serve`, and the MCP server). Bare
    /// `lamu login` prompts for every provider key in cloud-models.yaml that
    /// isn't already set; `lamu login <name>` does just one.
    Login {
        /// Optional provider / env-var filter, e.g. `deepseek` or
        /// `DEEPSEEK_API_KEY`. Omit to be prompted for all unset keys.
        provider: Option<String>,
    },
    /// Hardware-aware model fit: rank registry LLMs by predicted throughput +
    /// VRAM fit on the detected GPU (roofline scorer, ADR 0015).
    Cookbook {
        /// Score every model through one use-case lens
        /// (general|coding|reasoning|chat|multimodal|embedding). Default:
        /// each model's own inferred use-case.
        #[arg(long)]
        use_case: Option<String>,
        /// Evaluate at this quant (e.g. Q4_K_M) instead of each model's native.
        #[arg(long)]
        quant: Option<String>,
        /// Context length to score at. Default: each model's context_max.
        #[arg(long)]
        ctx: Option<u32>,
        /// Simulate a different VRAM budget (MB) — "what could I run on a 48 GB card?".
        #[arg(long)]
        simulate_vram: Option<u32>,
        /// Show only the top N results.
        #[arg(long)]
        top: Option<usize>,
        /// Emit JSON (FitResult array) instead of the table.
        #[arg(long)]
        json: bool,
        /// Also score a curated set of models you could `lamu pull`.
        #[arg(long)]
        suggest: bool,
    },
    /// Manage the optional HTTP API bearer token (ADR 0012). Auth is off on a
    /// loopback bind; a token is required to bind off-loopback.
    Auth {
        #[command(subcommand)]
        action: AuthAction,
    },
}

#[derive(clap::Subcommand, Debug)]
enum AuthAction {
    /// Mint a new API token, write it to ~/.config/lamu/api-token (chmod
    /// 0600), and print it once. Enables bearer auth on `lamu serve`.
    Init,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Default to `warn` when RUST_LOG is unset so operationally-relevant
    // warnings (zombie children, dropped thinking blocks, etc.) are
    // visible without the user having to opt in. RUST_LOG=info or debug
    // for higher verbosity; RUST_LOG=error to silence warns.
    let env_filter = match tracing_subscriber::EnvFilter::try_from_default_env() {
        Ok(f) => f,
        Err(e) => {
            // Distinguish "RUST_LOG unset" (silent fallback to warn) from
            // "RUST_LOG set but malformed" (eprintln so the user notices
            // the typo). Subscriber isn't initialized yet — eprintln is
            // the only available output.
            if std::env::var_os("RUST_LOG").is_some() {
                eprintln!("RUST_LOG ignored ({}); defaulting to 'warn'", e);
            }
            tracing_subscriber::EnvFilter::new("warn")
        }
    };
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .init();

    // Orphan cleanup: PDEATHSIG signals on parent death; watchdog
    // catches reparent-to-init when PDEATHSIG silently fails (observed
    // with `lamu start` zombies surviving terminal close).
    lamu_core::lifecycle::install_parent_death_signal();
    lamu_core::lifecycle::spawn_orphan_watchdog();

    load_api_keys_env();

    let cli = Cli::parse();
    match cli.command {
        None => {
            // Bare `lamu` → ratatui dashboard. Run on a blocking thread so
            // tokio's runtime stays free for any IO the TUI's status pings
            // might kick off in the background.
            tokio::task::spawn_blocking(tui::run).await??;
            Ok(())
        }
        Some(Command::Scan) => cmd_scan().await,
        Some(Command::Status) => cmd_status().await,
        Some(Command::Start) => cmd_start().await,
        Some(Command::Serve { port }) => cmd_serve(port).await,
        Some(Command::Repl { api_url }) => {
            // Themed ratatui chat. chat_tui::run falls back to the
            // legacy line REPL when stdout is not a TTY.
            tokio::task::spawn_blocking(move || -> Result<()> {
                let mut config = lamu_config::LamuConfig::load();
                config.backend_url = api_url;
                let theme = theme::Theme::pick(Some(&config.theme));
                chat_tui::run("default".into(), theme, config)
            }).await??;
            Ok(())
        }
        Some(Command::Run { model }) => cmd_run(model).await,
        Some(Command::Pull { model, quant }) => cmd_pull(&model, &quant).await,
        Some(Command::Show { model }) => cmd_show(&model),
        Some(Command::Rm { model, yes }) => cmd_rm(&model, yes),
        Some(Command::Sessions) => cmd_sessions(),
        Some(Command::Undo { session_id, yes }) => cmd_undo(session_id, yes),
        Some(Command::Rollback { session_id }) => cmd_rollback(&session_id),
        Some(Command::Agent { net, cmd }) => cmd_agent(net, cmd),
        Some(Command::Agents) => cmd_agents(),
        Some(Command::Preserve { session_id }) => cmd_preserve(&session_id),
        Some(Command::CherryPick { session_id, glob }) => cmd_cherry_pick(&session_id, &glob),
        Some(Command::DropAgent { session_id }) => cmd_drop_agent(&session_id),
        Some(Command::Login { provider }) => cmd_login(provider),
        Some(Command::Cookbook {
            use_case,
            quant,
            ctx,
            simulate_vram,
            top,
            json,
            suggest,
        }) => cmd_cookbook(CookbookOpts {
            use_case,
            quant,
            ctx,
            simulate_vram,
            top,
            json,
            suggest,
        }),
        Some(Command::Auth { action }) => cmd_auth(action),
    }
}

/// Infer a cookbook use-case bucket from a registry entry's modality +
/// capabilities (drives the scoring weights + speed/context targets).
fn infer_use_case(e: &lamu_core::types::ModelEntry) -> String {
    use lamu_core::types::{Capability, Modality};
    if e.modality == Modality::Tts {
        return "tts".to_string();
    }
    let has = |c: Capability| e.capabilities.contains(&c);
    if has(Capability::Embedding) {
        "embedding".to_string()
    } else if has(Capability::Vision) {
        "multimodal".to_string()
    } else if has(Capability::Code) {
        "coding".to_string()
    } else if has(Capability::Reasoning) {
        "reasoning".to_string()
    } else if has(Capability::Chat) {
        "chat".to_string()
    } else {
        "general".to_string()
    }
}

struct CookbookOpts {
    use_case: Option<String>,
    quant: Option<String>,
    ctx: Option<u32>,
    simulate_vram: Option<u32>,
    top: Option<usize>,
    json: bool,
    suggest: bool,
}

/// Build a cookbook ModelSpec from a registry entry. MoE fidelity: an `A<N>B`
/// name marker (qwen3.6-35b-a3b → 3B active) or a `*moe*` arch flags a sparse
/// model — active params drive the roofline + KV; TOTAL params still drive VRAM.
fn entry_to_spec(e: &lamu_core::types::ModelEntry, ctx_override: Option<u32>) -> lamu_core::cookbook::ModelSpec {
    let active = lamu_core::cookbook::active_params_from_name(&e.name);
    let is_moe = active.is_some() || e.arch.to_ascii_lowercase().contains("moe");
    lamu_core::cookbook::ModelSpec {
        name: e.name.clone(),
        params_b: e.params_b,
        active_params_b: active.unwrap_or(e.params_b),
        is_moe,
        quant: e.quant.clone(),
        context_max: ctx_override.unwrap_or(e.context_max),
        use_case: infer_use_case(e),
    }
}

/// Curated "you could pull this" set for `--suggest` (name, total B, quant,
/// ctx, is_moe, active B, repo). Tiny + hand-picked — not an HF mirror.
const CURATED: &[(&str, f32, &str, u32, bool, f32, &str)] = &[
    ("qwen3-4b", 4.0, "Q4_K_M", 32768, false, 4.0, "Qwen/Qwen3-4B-Instruct-GGUF"),
    ("mistral-small-24b", 24.0, "Q4_K_M", 32768, false, 24.0, "bartowski/Mistral-Small-24B-Instruct-2501-GGUF"),
    ("qwen3-30b-a3b", 30.0, "Q4_K_M", 32768, true, 3.0, "Qwen/Qwen3-30B-A3B-GGUF"),
    ("llama-3.3-70b", 70.0, "Q4_K_M", 8192, false, 70.0, "bartowski/Llama-3.3-70B-Instruct-GGUF"),
    ("qwen3-235b-a22b", 235.0, "Q4_K_M", 32768, true, 22.0, "Qwen/Qwen3-235B-A22B-GGUF"),
];

fn cmd_cookbook(opts: CookbookOpts) -> Result<()> {
    use lamu_core::cookbook::{self, Backend, FitLevel, Hardware, ModelSpec};

    let sched = lamu_core::scheduler::VramScheduler::new();
    let (used, total) = sched.query_vram();
    let gpu_name = sched.gpu_name();

    // --simulate-vram overrides the detected card's budget.
    let vram_mb = opts.simulate_vram.unwrap_or(total);
    let hw = Hardware {
        gpu_name: gpu_name.clone(),
        gpu_vram_gb: vram_mb as f32 / 1024.0,
        avail_ram_gb: 0.0, // GPU-only budget
        backend: Backend::Cuda,
    };

    let entries =
        lamu_core::registry::load_registry(&lamu_core::config::registry_path()).unwrap_or_default();
    let mut specs: Vec<ModelSpec> = entries
        .iter()
        .filter(|e| e.modality.is_llm() && e.params_b > 0.0)
        .map(|e| entry_to_spec(e, opts.ctx))
        .collect();

    if opts.suggest {
        for (name, pb, q, ctx, is_moe, active, _repo) in CURATED {
            specs.push(ModelSpec {
                name: format!("{name}  (pull)"),
                params_b: *pb,
                active_params_b: *active,
                is_moe: *is_moe,
                quant: opts.quant.clone().unwrap_or_else(|| q.to_string()),
                context_max: opts.ctx.unwrap_or(*ctx),
                use_case: "general".to_string(),
            });
        }
    }

    let mut ranked = cookbook::rank(&specs, &hw, opts.use_case.as_deref(), opts.quant.as_deref());
    if let Some(n) = opts.top {
        ranked.truncate(n);
    }

    if opts.json {
        println!("{}", serde_json::to_string_pretty(&ranked)?);
        return Ok(());
    }

    if let Some(sim) = opts.simulate_vram {
        println!("Simulating {:.1} GB VRAM\n", sim as f32 / 1024.0);
    } else if total == 0 {
        println!("GPU: no NVIDIA GPU detected via NVML — throughput uses the CPU fallback.\n");
    } else {
        println!(
            "GPU: {} — {:.1} GB VRAM ({:.1} GB free)\n",
            gpu_name.as_deref().unwrap_or("unknown"),
            total as f32 / 1024.0,
            total.saturating_sub(used) as f32 / 1024.0,
        );
    }

    if ranked.is_empty() {
        println!("No LLM models — run `lamu scan`, `lamu pull <repo>`, or pass --suggest.");
        return Ok(());
    }

    println!("  {:<30} {:>16} {:>9} {:>8} {:>6}", "model", "quant@ctx", "VRAM", "tok/s~", "score");
    for r in &ranked {
        let glyph = match r.fit_level {
            FitLevel::Perfect | FitLevel::Good => "🟢",
            FitLevel::Marginal => "🟡",
            FitLevel::TooTight => "🔴",
        };
        let qc = format!("{}@{}", r.quant, r.context);
        println!(
            "  {} {:<28} {:>16} {:>6.1} GB {:>8.0} {:>6.1}",
            glyph, r.name, qc, r.required_gb, r.tps_est, r.score
        );
    }
    println!("\n🟢 fits comfortably · 🟡 marginal · 🔴 won't fit");
    Ok(())
}

fn cmd_auth(action: AuthAction) -> Result<()> {
    match action {
        AuthAction::Init => cmd_auth_init(),
    }
}

/// Mint a `lamu_<64-hex>` API token, write it to ~/.config/lamu/api-token at
/// mode 0600, and print it once (ADR 0012). Refuses to clobber an existing
/// token (delete it to rotate). The token enables bearer auth on `lamu serve`
/// and is required to bind off-loopback.
fn cmd_auth_init() -> Result<()> {
    use std::io::Write as _;
    let dir = dirs::config_dir()
        .map(|d| d.join("lamu"))
        .ok_or_else(|| anyhow::anyhow!("cannot resolve config dir (~/.config)"))?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("api-token");

    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).map_err(|e| anyhow::anyhow!("getrandom: {e}"))?;
    let tok = format!(
        "lamu_{}",
        bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
    );

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true); // refuse to overwrite an existing token
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = match opts.open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            anyhow::bail!(
                "a token already exists at {} — delete it to rotate, or set LAMU_API_TOKEN to override",
                path.display()
            );
        }
        Err(e) => return Err(e.into()),
    };
    f.write_all(tok.as_bytes())?;
    f.write_all(b"\n")?;

    println!("API token written to {} (chmod 0600).", path.display());
    println!("\n    {tok}\n");
    println!("Shown ONCE. Clients send:  Authorization: Bearer {tok}");
    println!("Auth is OFF on a loopback bind; this token is required to bind off-loopback");
    println!("(LAMU_BIND_HOST=0.0.0.0). Override per-process with LAMU_API_TOKEN.");
    Ok(())
}

/// Interactive key entry → ~/.config/lamu/api-keys.env. Prompts for every
/// distinct `api_key_env` in cloud-models.yaml (plus OPENAI/FISH_AUDIO for
/// embeddings + cloud TTS) that isn't already set; `provider` narrows it.
fn cmd_login(provider: Option<String>) -> Result<()> {
    use std::io::Write;
    let mut vars: Vec<String> = lamu_providers::cloud_config::load_or_empty()
        .iter()
        .filter_map(|m| m.api_key_env.clone())
        .collect();
    vars.push("OPENAI_API_KEY".to_string()); // memory/rag embeddings
    vars.push("FISH_AUDIO_API_KEY".to_string()); // cloud TTS
    vars.sort();
    vars.dedup();
    if let Some(p) = &provider {
        let pl = p.to_lowercase();
        let filtered: Vec<String> =
            vars.iter().filter(|v| v.to_lowercase().contains(&pl)).cloned().collect();
        vars = if filtered.is_empty() { vec![p.clone()] } else { filtered };
    }

    let mut wrote = 0usize;
    for var in vars {
        let set = std::env::var(&var).ok().map(|v| !v.trim().is_empty()).unwrap_or(false);
        print!("{var} [{}] — paste key (blank to skip): ", if set { "set" } else { "unset" });
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        let key = line.trim();
        if key.is_empty() {
            println!("  skipped");
            continue;
        }
        let path = lamu_providers::cloud_config::save_api_key_env(&var, key)?;
        // SAFETY: single-threaded CLI path, before any async/threads spawn.
        unsafe { std::env::set_var(&var, key); }
        println!("  ✓ saved to {}", path.display());
        wrote += 1;
    }
    if wrote > 0 {
        println!(
            "\n{wrote} key(s) saved. They load automatically on `lamu` / `lamu serve` / MCP startup."
        );
    } else {
        println!("No keys entered.");
    }
    Ok(())
}

fn cmd_agents() -> Result<()> {
    let trees = sandbox::preserve::list_agent_worktrees()?;
    if trees.is_empty() {
        println!("(no active agent worktrees — `lamu agent` to create one)");
        return Ok(());
    }
    println!("Active agent worktrees ({}):\n", trees.len());
    for w in &trees {
        let last = w.last_checkpoint_secs.map(|s| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs()).unwrap_or(0);
            let age = now.saturating_sub(s);
            if age < 60 { format!("{}s ago", age) }
            else if age < 3600 { format!("{}m ago", age / 60) }
            else { format!("{}h ago", age / 3600) }
        }).unwrap_or_else(|| "(no checkpoint yet)".into());
        println!(
            "  {}\n    branch={}\n    path={}\n    last_checkpoint={}\n    files={}, loc_delta={:+}",
            w.session_id, w.branch, w.path.display(), last, w.files_changed, w.loc_delta
        );
    }
    Ok(())
}

fn cmd_preserve(session_id: &str) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let sha = sandbox::preserve::preserve_session(session_id, &cwd)?;
    println!("✓ preserved agent/{} → {}", session_id, sha);
    println!("  Drop the worktree with `lamu drop-agent {}` once you're done.", session_id);
    Ok(())
}

fn cmd_cherry_pick(session_id: &str, glob: &str) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let summary = sandbox::preserve::cherry_pick_files(session_id, glob, &cwd)?;
    println!("{}", summary);
    Ok(())
}

fn cmd_drop_agent(session_id: &str) -> Result<()> {
    let cwd = std::env::current_dir()?;
    sandbox::preserve::drop_session(session_id, &cwd)?;
    println!("✓ dropped agent/{} (worktree + branch removed)", session_id);
    Ok(())
}

fn cmd_sessions() -> Result<()> {
    let snaps = sandbox::snap::Snapshot::list()?;
    if snaps.is_empty() {
        println!("(no sessions yet — start a chat to capture one)");
        return Ok(());
    }
    println!("Recent chat sessions ({} total):\n", snaps.len());
    for s in snaps.iter().take(20) {
        println!("  {}", s.pretty_summary());
    }
    Ok(())
}

fn cmd_undo(session_id: Option<String>, yes: bool) -> Result<()> {
    let snap = match session_id {
        Some(id) => sandbox::snap::Snapshot::load(&id)?,
        None => {
            let all = sandbox::snap::Snapshot::list()?;
            all.into_iter().next()
                .ok_or_else(|| anyhow::anyhow!("no sessions captured yet"))?
        }
    };
    println!("About to restore session:\n  {}", snap.pretty_summary());
    if snap.restored {
        eprintln!("⚠️  this session was already restored once.");
    }
    if !yes {
        eprint!("Proceed? [y/N] ");
        use std::io::Write;
        std::io::stderr().flush().ok();
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf)?;
        if !buf.trim().to_lowercase().starts_with('y') {
            println!("Cancelled.");
            return Ok(());
        }
    }
    snap.restore()?;
    println!("✓ restored {}", snap.session_id);
    Ok(())
}

fn cmd_rollback(session_id: &str) -> Result<()> {
    let (restored, skipped) = sandbox::journal::rollback(session_id)?;
    println!("Journal rollback complete: {} restored, {} skipped.", restored, skipped);
    Ok(())
}

fn git_repo_root(cwd: &std::path::Path) -> Option<std::path::PathBuf> {
    let out = std::process::Command::new("git")
        .arg("-C").arg(cwd)
        .args(["rev-parse", "--show-toplevel"])
        .output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then(|| std::path::PathBuf::from(s))
}

fn cmd_agent(allow_net: bool, cmd: Vec<String>) -> Result<()> {
    if cmd.is_empty() {
        anyhow::bail!("usage: lamu agent [--net] -- <command...>");
    }

    let cwd = std::env::current_dir()?;

    // Opt-in filesystem isolation: LAMU_AGENT_WORKTREE=1 runs the agent in
    // a dedicated git worktree (branch agent/<id>) instead of binding the
    // live cwd, so concurrent `lamu agent` runs don't collide on the same
    // working tree. OFF by default — most callers want edits to land in
    // the cwd they launched from; auto-creating a worktree would silently
    // redirect their writes. The bubblewrap namespace (pid/net/user) is
    // isolated regardless; this adds *filesystem* isolation when asked.
    // create_worktree failure propagates BEFORE the snapshot below, so a
    // failed run never leaves an orphaned snapshot.
    let worktree = if std::env::var("LAMU_AGENT_WORKTREE").as_deref() == Ok("1") {
        match git_repo_root(&cwd) {
            Some(repo) => {
                let session = sandbox::new_session_id();
                let wt = sandbox::preserve::create_worktree(&session, &repo)?;
                eprintln!(
                    "[sandbox] agent worktree: {} (branch agent/{session})\n\
                     [sandbox]   merge your changes back with `git worktree`/`git cherry-pick`, \
                     then `git worktree remove` to clean up.",
                    wt.display()
                );
                Some(wt)
            }
            None => {
                eprintln!("[sandbox] LAMU_AGENT_WORKTREE=1 but cwd is not a git repo — using cwd");
                None
            }
        }
    } else {
        None
    };

    // Rollback coverage: capture a git snapshot so `lamu undo` / `lamu
    // agents` cover sandboxed agent runs (previously only the chat-TUI
    // path did). Only needed when the agent writes into the live cwd — in
    // worktree mode the changes live on an isolated branch, rolled back
    // with `git worktree remove`. Best-effort: failure must not block.
    let workdir = match worktree {
        Some(wt) => wt,
        None => {
            match sandbox::snap::Snapshot::capture("agent") {
                Ok(s) => eprintln!("[sandbox] snapshot {} captured — `lamu undo` to restore", s.session_id),
                Err(e) => eprintln!("[sandbox] warning: snapshot capture failed: {e}"),
            }
            cwd
        }
    };

    let mut opts = sandbox::launcher::SandboxOpts::new(workdir);
    if allow_net { opts = opts.with_net(); }
    let status = sandbox::launcher::run(&opts, &cmd)?;
    if !status.success() {
        anyhow::bail!("sandboxed command exited {}", status);
    }
    Ok(())
}

async fn cmd_scan() -> Result<()> {
    let dir = models_dir();
    let path = registry_path();
    let entries = scan_directory(&dir)?;
    write_registry(&entries, &path)?;
    println!("Discovered {} models → {}", entries.len(), path.display());
    for e in &entries {
        let caps: Vec<&str> = e.capabilities.iter().map(|c| match c {
            lamu_core::types::Capability::Chat => "chat",
            lamu_core::types::Capability::Code => "code",
            lamu_core::types::Capability::Reasoning => "reasoning",
            lamu_core::types::Capability::Routing => "routing",
            lamu_core::types::Capability::Vision => "vision",
            lamu_core::types::Capability::LongContext => "long_context",
            lamu_core::types::Capability::Embedding => "embedding",
        }).collect();
        println!(
            "  {}: {}B {} ({}MB) [{}]",
            e.name, e.params_b, e.quant, e.vram_mb, caps.join(", ")
        );
    }
    Ok(())
}

async fn cmd_status() -> Result<()> {
    let entries = load_registry(&registry_path())?;
    let scheduler = VramScheduler::new();
    let (used, total) = scheduler.query_vram();
    println!("VRAM: {}/{} MB ({} MB free)", used, total, total.saturating_sub(used));
    println!("Models in registry: {}", entries.len());
    println!();

    let client = reqwest::Client::builder().timeout(Duration::from_secs(1)).build()?;
    for port in [PORT_MAIN, PORT_SIDECAR, 8000u16] {
        let health = fetch_json(&client, port, "/health").await;
        let models = if health
            .as_ref()
            .and_then(|h| h.get("models_loaded"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            > 0
        {
            fetch_json(&client, port, "/v1/models").await
        } else {
            None
        };
        println!("{}", format_status_line(port, health.as_ref(), models.as_ref()));
    }
    Ok(())
}

async fn fetch_json(client: &reqwest::Client, port: u16, path: &str) -> Option<Value> {
    let url = format!("http://localhost:{}{}", port, path);
    let r = client.get(&url).send().await.ok()?;
    r.json::<Value>().await.ok()
}

/// Render the status line for one probed port. Pure function over the
/// probe results so it can be exhaustively unit-tested:
///
/// - `health: None` → "⚪ :{port} — not running" (HTTP layer not reachable)
/// - `health.status != "ok"` → same as above
/// - `health.models_loaded == 0` → "🟡 :{port} — http up, no model loaded"
/// - `health.models_loaded > 0` + `models` has loaded entries → green line listing names
/// - `health.models_loaded > 0` + `models` missing → green count fallback
fn format_status_line(port: u16, health: Option<&Value>, models: Option<&Value>) -> String {
    let Some(h) = health else {
        return format!("  ⚪ :{} — not running", port);
    };
    if h.get("status").and_then(|v| v.as_str()) != Some("ok") {
        return format!("  ⚪ :{} — not running", port);
    }
    let n_loaded = h.get("models_loaded").and_then(|v| v.as_u64()).unwrap_or(0);
    if n_loaded == 0 {
        return format!("  🟡 :{} — http up, no model loaded", port);
    }
    let mut loaded_names: Vec<String> = Vec::new();
    if let Some(arr) = models.and_then(|m| m.get("data")).and_then(|d| d.as_array()) {
        for m in arr {
            if m.get("loaded").and_then(|v| v.as_bool()).unwrap_or(false) {
                if let Some(id) = m.get("id").and_then(|v| v.as_str()) {
                    loaded_names.push(id.to_string());
                }
            }
        }
    }
    if loaded_names.is_empty() {
        // /health says loaded > 0 but /v1/models didn't surface them.
        // Don't fabricate a name; report a count.
        format!("  🟢 :{} — {} model(s) loaded (names unavailable)", port, n_loaded)
    } else {
        format!("  🟢 :{} — {}", port, loaded_names.join(", "))
    }
}

async fn cmd_start() -> Result<()> {
    let dir = models_dir();
    let path = registry_path();
    let mut scheduler = VramScheduler::new();

    // Auto-register running models
    let client = reqwest::Client::builder().timeout(Duration::from_secs(2)).build()?;
    let entries_vec = load_registry(&path)?;
    for port in [PORT_MAIN, PORT_SIDECAR] {
        let url = format!("http://localhost:{}/v1/models", port);
        if let Ok(r) = client.get(&url).send().await {
            if let Ok(j) = r.json::<Value>().await {
                if let Some(model_id) = j.get("data").and_then(|d| d.get(0))
                    .and_then(|m| m.get("id")).and_then(|v| v.as_str())
                {
                    let model_id = model_id.to_lowercase();
                    for entry in &entries_vec {
                        if entry.name.contains(&model_id) || model_id.contains(entry.name.as_str()) {
                            // Use the registry's declared footprint for THIS
                            // model. The llama-server was started outside this
                            // lamu process, so we don't have its PID to attribute
                            // a specific GPU process to it. The prior
                            // `query_gpu_pids().max()` stamped every
                            // auto-registered model with the single largest GPU
                            // process globally — double-counting when both ports
                            // are live, and mis-estimating eviction freed-math.
                            // available_mb() still clamps the aggregate to NVML
                            // truth, so the declared per-model size is the honest
                            // estimate for budget + eviction.
                            scheduler.register_loaded(entry.clone(), None, port, entry.vram_mb);
                            break;
                        }
                    }
                }
            }
        }
    }

    let server = LamuMcpServer::new(dir, path, scheduler)?;
    server.run().await
}

async fn cmd_serve(port: u16) -> Result<()> {
    lamu_api::serve(port).await
}

// ── Ollama-shaped subcommands ──────────────────────────────────────────────

/// Resolve a name fragment to a single registry entry. Substring match,
/// both directions, case-insensitive. Errors when zero or multiple matches.
fn resolve_entry(query: &str) -> Result<lamu_core::types::ModelEntry> {
    let entries = load_registry(&registry_path())?;
    let q = query.to_lowercase();
    let matches: Vec<_> = entries.iter()
        .filter(|e| {
            let n = e.name.to_lowercase();
            n.contains(&q) || q.contains(&n)
        })
        .cloned()
        .collect();
    match matches.len() {
        0 => anyhow::bail!(
            "no model in registry matches '{}'. Run `lamu scan` or `lamu pull {}`.",
            query, query
        ),
        1 => Ok(matches.into_iter().next().unwrap()),
        n => {
            let names: Vec<String> = matches.iter().map(|e| e.name.clone()).collect();
            anyhow::bail!("'{}' is ambiguous ({} matches): {:?}", query, n, names)
        }
    }
}

async fn cmd_run(query: String) -> Result<()> {
    let entry = resolve_entry(&query)?;
    println!("→ Resolved to: {}", entry.name);

    // Make sure the OpenAI-compat layer is up. If not, spawn `lamu serve`
    // in a detached child so the chat session can talk to it.
    let client = reqwest::Client::builder().timeout(Duration::from_secs(1)).build()?;
    let serve_up = client.get("http://localhost:8020/health").send().await
        .map(|r| r.status().is_success()).unwrap_or(false);
    if !serve_up {
        eprintln!(
            "  lamu serve not on :8020. Start it in another terminal with \
             `lamu serve`, then re-run."
        );
        anyhow::bail!("daemon not running");
    }

    // The MCP-style load_model lives behind stdio. For the run shortcut
    // we only need the backend's HTTP port to answer; if the model isn't
    // loaded yet, the OpenAI compat will return 503 and prompt the user
    // to load via Claude Code or `lamu start`. (Future: wire up a proper
    // load over the MCP transport from here.)
    let model_name = entry.name.clone();
    println!("  Dropping into chat (model={}). Esc/Ctrl+C to exit.\n", entry.name);
    tokio::task::spawn_blocking(move || -> Result<()> {
        let config = lamu_config::LamuConfig::load();
        let theme = theme::Theme::pick(Some(&config.theme));
        chat_tui::run(model_name, theme, config)
    }).await??;
    Ok(())
}

/// Map a shorthand to (hf_repo, filename_pattern).
fn pull_spec(shorthand: &str, quant: &str) -> Option<(String, String)> {
    match shorthand {
        "qwen36-27b" | "qwen3.6-27b" | "heretic" => Some((
            "llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF".into(),
            format!("Qwen3.6-27B-uncensored-heretic-v2-{}.gguf", quant),
        )),
        "qwen36-35b" | "qwen3.6-35b" => Some((
            "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF".into(),
            format!("Qwen3.6-35B-A3B-uncensored-heretic-{}.gguf", quant),
        )),
        "qwen35-4b" | "qwen3.5-4b" => Some((
            "ggml-org/Qwen3.5-4B-Q4_K_M-GGUF".into(),
            format!("Qwen3.5-4B-{}.gguf", quant),
        )),
        s if s.contains('/') => Some((s.to_string(), String::new())),
        _ => None,
    }
}

async fn cmd_pull(shorthand: &str, quant: &str) -> Result<()> {
    let (repo, file) = match pull_spec(shorthand, quant) {
        Some(x) => x,
        None => anyhow::bail!(
            "unknown shorthand '{}'. Try `qwen36-27b`, `qwen36-35b`, `qwen35-4b`, or `org/repo`.",
            shorthand
        ),
    };

    let dir = models_dir().join(shorthand.replace('/', "-"));
    std::fs::create_dir_all(&dir)?;

    println!("Pulling {} → {}", repo, dir.display());

    let mut cmd = std::process::Command::new("hf");
    cmd.arg("download").arg(&repo);
    if !file.is_empty() {
        cmd.arg(&file);
    }
    cmd.arg("--local-dir").arg(&dir);

    let status = cmd.status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => anyhow::bail!("hf download exited with {}", s),
        Err(e) => anyhow::bail!("failed to invoke `hf` (install with `pip install huggingface-hub`): {}", e),
    }

    // Re-scan so the new GGUF lands in the registry.
    cmd_scan().await?;
    Ok(())
}

fn cmd_show(query: &str) -> Result<()> {
    let entry = resolve_entry(query)?;
    let yaml = serde_yaml::to_string(&entry)
        .map_err(|e| anyhow::anyhow!("yaml render: {e}"))?;
    print!("{}", yaml);
    Ok(())
}

fn cmd_rm(query: &str, yes: bool) -> Result<()> {
    let entry = resolve_entry(query)?;
    let path = &entry.path;

    // Containment: the registry can be edited / corrupted, so we never
    // trust ModelEntry.path blindly. Refuse to delete anything that
    // doesn't canonicalize to a real path under the configured
    // models_dir(). Defense against tampered registry pointing at
    // /etc/passwd or ~/.ssh/id_rsa.
    let dir = models_dir();
    let dir_canonical = dir.canonicalize()
        .with_context(|| format!("canonicalize models dir {}", dir.display()))?;
    let path_canonical = match path.canonicalize() {
        Ok(p) => Some(p),
        Err(_) => None, // file may already be gone — that's fine
    };
    if let Some(p) = &path_canonical {
        if !p.starts_with(&dir_canonical) {
            anyhow::bail!(
                "refusing to delete: registry path {} is outside models dir {}",
                p.display(), dir_canonical.display()
            );
        }
    }

    println!("Will remove from registry: {}", entry.name);
    if let Some(p) = &path_canonical {
        let size_mb = std::fs::metadata(p)
            .map(|m| m.len() / (1024 * 1024))
            .unwrap_or(0);
        println!("Will delete file: {} ({} MB)", p.display(), size_mb);
    } else {
        println!("(file already gone: {})", path.display());
    }

    if !yes {
        eprint!("Proceed? [y/N] ");
        use std::io::Write;
        let _ = std::io::stderr().flush();
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf)?;
        if !buf.trim().to_lowercase().starts_with('y') {
            println!("Cancelled.");
            return Ok(());
        }
    }

    // Delete the canonicalized path — never the original (would be a
    // TOCTOU race with symlink swap if we used `path` here).
    if let Some(p) = path_canonical {
        match std::fs::remove_file(&p) {
            Ok(()) => println!("  deleted file"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("  (already gone)");
            }
            Err(e) => return Err(e).with_context(|| format!("remove {}", p.display())),
        }
    }

    // Re-scan to drop the entry from registry.
    let dir = models_dir();
    let entries = scan_directory(&dir)?;
    write_registry(&entries, &registry_path())?;
    println!("  registry refreshed ({} models remaining)", entries.len());
    Ok(())
}

/// Load `~/.config/lamu/api-keys.env` into the process environment.
/// Parses `export VAR=value` or `VAR=value` lines.
/// Existing env vars are NOT overwritten — shell exports take priority.
fn load_api_keys_env() {
    let path = if let Some(d) = dirs::config_dir() {
        d.join("lamu").join("api-keys.env")
    } else {
        return;
    };
    let Ok(contents) = std::fs::read_to_string(&path) else { return };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let line = line.strip_prefix("export ").unwrap_or(line);
        if let Some((k, v)) = line.split_once('=') {
            let k = k.trim();
            let v = v.trim().trim_matches('"').trim_matches('\'');
            if !k.is_empty() && std::env::var(k).is_err() {
                // SAFETY: single-threaded before tokio runtime starts.
                unsafe { std::env::set_var(k, v); }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn status_line_down_when_health_missing() {
        let line = format_status_line(8020, None, None);
        assert_eq!(line, "  ⚪ :8020 — not running");
    }

    #[test]
    fn status_line_down_when_status_not_ok() {
        let health = json!({"status": "degraded", "models_loaded": 1});
        let line = format_status_line(8020, Some(&health), None);
        assert_eq!(line, "  ⚪ :8020 — not running");
    }

    #[test]
    fn status_line_yellow_when_zero_models() {
        let health = json!({"status": "ok", "models_loaded": 0});
        let line = format_status_line(8020, Some(&health), None);
        assert_eq!(line, "  🟡 :8020 — http up, no model loaded");
    }

    #[test]
    fn status_line_green_with_loaded_names() {
        let health = json!({"status": "ok", "models_loaded": 2});
        let models = json!({
            "data": [
                {"id": "qwen3.6-27b", "loaded": true},
                {"id": "draft", "loaded": false},
                {"id": "qwen3.6-35b-a3b", "loaded": true},
            ]
        });
        let line = format_status_line(8020, Some(&health), Some(&models));
        assert_eq!(line, "  🟢 :8020 — qwen3.6-27b, qwen3.6-35b-a3b");
    }

    #[test]
    fn status_line_green_count_fallback_when_models_missing() {
        let health = json!({"status": "ok", "models_loaded": 3});
        let line = format_status_line(8020, Some(&health), None);
        assert_eq!(line, "  🟢 :8020 — 3 model(s) loaded (names unavailable)");
    }

    #[test]
    fn status_line_green_count_fallback_when_data_is_not_array() {
        // /v1/models returned something but `data` was missing or malformed.
        // Fall back to the count rather than printing nothing or panicking.
        let health = json!({"status": "ok", "models_loaded": 1});
        let models = json!({"unexpected": "shape"});
        let line = format_status_line(8020, Some(&health), Some(&models));
        assert_eq!(line, "  🟢 :8020 — 1 model(s) loaded (names unavailable)");
    }

    #[test]
    fn status_line_green_skips_loaded_false_entries() {
        // All entries `loaded: false` → no names → count fallback.
        let health = json!({"status": "ok", "models_loaded": 1});
        let models = json!({
            "data": [
                {"id": "a", "loaded": false},
                {"id": "b", "loaded": false},
            ]
        });
        let line = format_status_line(8020, Some(&health), Some(&models));
        assert_eq!(line, "  🟢 :8020 — 1 model(s) loaded (names unavailable)");
    }
}
