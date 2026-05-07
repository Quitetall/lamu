//! LAMU CLI entry point. Port of `lamu/daemon.py`.

mod chat_tui;
mod cloud_models;
mod favorites;
mod lamu_config;
mod mcp_servers;
mod md_stream;
mod providers;
mod repl;
mod theme;
mod tui;

use anyhow::Result;
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
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

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
    }
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
        let url = format!("http://localhost:{}/health", port);
        match client.get(&url).send().await {
            Ok(r) => {
                if let Ok(j) = r.json::<Value>().await {
                    if j.get("status").and_then(|v| v.as_str()) == Some("ok") {
                        let url2 = format!("http://localhost:{}/v1/models", port);
                        let model = match client.get(&url2).send().await {
                            Ok(r) => match r.json::<Value>().await {
                                Ok(jj) => jj.get("data").and_then(|d| d.get(0))
                                    .and_then(|m| m.get("id")).and_then(|v| v.as_str())
                                    .map(String::from).unwrap_or_else(|| "unknown".into()),
                                Err(_) => "unknown".into(),
                            },
                            Err(_) => "unknown".into(),
                        };
                        println!("  🟢 :{} — {}", port, model);
                        continue;
                    }
                }
                println!("  ⚪ :{} — not running", port);
            }
            Err(_) => println!("  ⚪ :{} — not running", port),
        }
    }
    Ok(())
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
                            let pids = scheduler.query_gpu_pids();
                            let vram = pids.iter().map(|(_, m)| *m).max().unwrap_or(entry.vram_mb);
                            scheduler.register_loaded(entry.clone(), None, port, vram);
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
    println!("Will remove from registry: {}", entry.name);
    if path.exists() {
        let size_mb = std::fs::metadata(path)
            .map(|m| m.len() / (1024 * 1024))
            .unwrap_or(0);
        println!("Will delete file: {} ({} MB)", path.display(), size_mb);
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

    if path.exists() {
        std::fs::remove_file(path)?;
        println!("  deleted file");
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
