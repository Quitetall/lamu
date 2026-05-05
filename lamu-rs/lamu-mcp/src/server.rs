//! MCP server lifecycle. Port of `lamu/mcp/server.py::LamuMcpServer`.

use lamu_core::scheduler::VramScheduler;
use std::path::PathBuf;

pub struct LamuMcpServer {
    _models_dir: PathBuf,
    _registry_path: PathBuf,
    _scheduler: VramScheduler,
}

impl LamuMcpServer {
    pub fn new(
        models_dir: PathBuf,
        registry_path: PathBuf,
        scheduler: VramScheduler,
    ) -> Self {
        Self {
            _models_dir: models_dir,
            _registry_path: registry_path,
            _scheduler: scheduler,
        }
    }

    pub async fn run(self) -> anyhow::Result<()> {
        todo!("port stdio server loop — JSON-RPC handshake + tool dispatch")
    }
}
