//! LAMU CLI entry point. Port of `lamu/daemon.py`.

mod repl;
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
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

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
            tokio::task::spawn_blocking(move || repl::run_repl(api_url)).await??;
            Ok(())
        }
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
