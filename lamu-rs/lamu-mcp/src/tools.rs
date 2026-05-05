//! MCP tool handlers. One function per tool, mirrors Python `_handle_*`.

use anyhow::Result;
use serde_json::Value;

pub async fn query(_args: Value) -> Result<String> {
    todo!("port _handle_query — route + generate + reasoning extract")
}

pub async fn plan_query(_args: Value) -> Result<String> {
    todo!("port _handle_plan_query — dry-run routing")
}

pub async fn list_models(_args: Value) -> Result<String> {
    todo!("port _handle_list_models")
}

pub async fn load_model(_args: Value) -> Result<String> {
    todo!("port _handle_load_model — subprocess + health poll + scheduler register")
}

pub async fn unload_model(_args: Value) -> Result<String> {
    todo!("port _handle_unload_model — kill PID + scheduler unregister")
}

pub async fn vram_status(_args: Value) -> Result<String> {
    todo!("port _handle_vram_status")
}

pub async fn scan_models(_args: Value) -> Result<String> {
    todo!("port _handle_scan — re-scan + write registry + update router")
}
