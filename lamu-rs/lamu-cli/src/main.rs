//! LAMU CLI entry point. Port of `lamu/daemon.py` + `lamu/__main__.py`.

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "lamu", version, about = "LAMU — MCP-first model management")]
struct Cli {
    #[command(subcommand)]
    command: Command,
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Scan => cmd_scan().await,
        Command::Status => cmd_status().await,
        Command::Start => cmd_start().await,
        Command::Serve { port } => cmd_serve(port).await,
    }
}

async fn cmd_scan() -> anyhow::Result<()> {
    todo!("port lamu/daemon.py::cmd_scan")
}

async fn cmd_status() -> anyhow::Result<()> {
    todo!("port lamu/daemon.py::cmd_status")
}

async fn cmd_start() -> anyhow::Result<()> {
    todo!("port lamu/daemon.py::cmd_start — boot MCP server")
}

async fn cmd_serve(_port: u16) -> anyhow::Result<()> {
    lamu_api::serve(_port).await
}
