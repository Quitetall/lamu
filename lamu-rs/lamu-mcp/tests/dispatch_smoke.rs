//! Phase 6.2 — integration smoke tests for the MCP request dispatcher.
//!
//! These tests bypass stdio and call `LamuMcpServer::handle` directly
//! so we exercise the full JSON-RPC envelope (initialize → tools/list →
//! tools/call) without spawning a subprocess. Network is not touched
//! by design: cloud_query / review_commit are tested via their
//! validation paths (no API key set, bad refs) where the error string
//! is reproducible without a live cloud endpoint.

use lamu_core::scheduler::VramScheduler;
use lamu_mcp::server::LamuMcpServer;
use serde_json::{json, Value};
use tempfile::tempdir;

fn fresh_server() -> LamuMcpServer {
    // Empty registry + empty models dir keeps the constructor cheap;
    // we don't load any model in these tests.
    let dir = tempdir().expect("tempdir");
    let registry = dir.path().join("registry.yaml");
    std::fs::write(&registry, "models: {}\n").unwrap();
    LamuMcpServer::new(
        dir.path().to_path_buf(),
        registry,
        VramScheduler::new(),
    )
    .expect("server new")
}

async fn call_tool(srv: &LamuMcpServer, name: &str, args: Value) -> Value {
    let params = json!({"name": name, "arguments": args});
    srv.handle("tools/call", params, Some(json!(1)))
        .await
        .expect("response")
}

#[tokio::test]
async fn initialize_and_tools_list_round_trip() {
    let srv = fresh_server();

    let init = srv
        .handle("initialize", json!({}), Some(json!(1)))
        .await
        .expect("initialize response");
    assert_eq!(init["jsonrpc"], "2.0");
    assert!(init["result"].is_object(), "initialize must return result");

    let list = srv
        .handle("tools/list", json!({}), Some(json!(2)))
        .await
        .expect("tools/list response");
    let tools = list["result"]["tools"].as_array().unwrap();
    assert!(tools.len() >= 16, "tools list shrank: {}", tools.len());
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    for required in [
        "query", "cloud_query", "review_commit", "review_diff", "write_file",
        "list_models", "list_cloud_models", "parallel_query", "set_routing_mode",
        "recall_conversation",
    ] {
        assert!(names.contains(&required), "tools/list missing {required}");
    }
}

#[tokio::test]
async fn unknown_tool_returns_is_error() {
    let srv = fresh_server();
    let resp = call_tool(&srv, "definitely-not-a-real-tool", json!({})).await;
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["result"]["isError"], true);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.to_lowercase().contains("unknown tool"),
        "expected 'Unknown tool:' marker, got: {text}"
    );
}

#[tokio::test]
async fn cloud_query_missing_prompt_errors() {
    let srv = fresh_server();
    let resp = call_tool(&srv, "cloud_query", json!({})).await;
    assert_eq!(resp["result"]["isError"], true);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("error: prompt"), "got: {text}");
}

#[tokio::test]
async fn cloud_query_unknown_model_errors() {
    let srv = fresh_server();
    let resp = call_tool(
        &srv,
        "cloud_query",
        json!({"prompt": "hi", "model": "nope-not-a-real-model"}),
    )
    .await;
    assert_eq!(resp["result"]["isError"], true);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("not in cloud-models.yaml"), "got: {text}");
}

#[tokio::test]
async fn review_commit_rejects_unsafe_ref() {
    let srv = fresh_server();
    let resp = call_tool(
        &srv,
        "review_commit",
        json!({"commit": "--upload-pack=evil", "repo": "."}),
    )
    .await;
    assert_eq!(resp["result"]["isError"], true);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("rejected"), "got: {text}");
}

#[tokio::test]
async fn review_diff_requires_diff_field() {
    let srv = fresh_server();
    let resp = call_tool(&srv, "review_diff", json!({})).await;
    assert_eq!(resp["result"]["isError"], true);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("'diff' is required"), "got: {text}");
}

#[tokio::test]
async fn parallel_query_rejects_empty_tasks_array() {
    let srv = fresh_server();
    let resp = call_tool(&srv, "parallel_query", json!({"tasks": []})).await;
    // Empty fan-out is rejected with a clear error message.
    assert_eq!(resp["result"]["isError"], true);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("non-empty array"), "got: {text}");
}

#[tokio::test]
async fn list_cloud_models_round_trip() {
    let srv = fresh_server();
    let resp = call_tool(&srv, "list_cloud_models", json!({})).await;
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    // Either populated (cloud-models.yaml present) or the explicit
    // "no cloud models" message — both shapes are acceptable, just
    // never empty.
    assert!(!text.is_empty(), "list_cloud_models returned empty");
}

#[tokio::test]
async fn write_file_round_trip_uses_journal() {
    use std::env;
    let srv = fresh_server();
    let cwd = tempdir().unwrap();
    let prev = env::current_dir().unwrap();
    env::set_current_dir(cwd.path()).unwrap();
    let resp = call_tool(
        &srv,
        "write_file",
        json!({
            "path": "smoke.txt",
            "content": "integration",
            "session_id": "test-dispatch-smoke"
        }),
    )
    .await;
    let read_back = std::fs::read(cwd.path().join("smoke.txt"));
    env::set_current_dir(prev).unwrap();

    assert_eq!(resp["result"]["isError"], false);
    let bytes = read_back.expect("file should exist");
    assert_eq!(bytes, b"integration");
}

// Phase 6 step 6 — context layer args reach the dispatch path. We
// verify the args parse + route correctly without hitting the cloud
// (request errors at the cloud-models lookup, not at parsing).

#[tokio::test]
async fn cloud_query_accepts_context_and_plan_file_args() {
    let srv = fresh_server();
    let resp = call_tool(
        &srv,
        "cloud_query",
        json!({
            "prompt": "hi",
            "model": "definitely-not-a-real-model",
            "context": "tactical context blob",
            "plan_file": "/tmp/nonexistent-plan.md"
        }),
    )
    .await;
    // Errors at "model not in cloud-models.yaml" — proves args parsed
    // and routed past the new context-layer plumbing.
    assert_eq!(resp["result"]["isError"], true);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("not in cloud-models.yaml"), "got: {text}");
}

#[tokio::test]
async fn review_diff_accepts_plan_file_arg() {
    // Empty `diff` is rejected up front by review_diff before any
    // cloud call — keeps this test from depending on whether
    // DEEPSEEK_API_KEY is set in the test env.
    let srv = fresh_server();
    let resp = call_tool(
        &srv,
        "review_diff",
        json!({
            "diff": "",
            "context": "tactical context",
            "plan_file": "/tmp/nonexistent-plan.md"
        }),
    )
    .await;
    assert_eq!(resp["result"]["isError"], true);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("'diff' is required"), "got: {text}");
}

#[tokio::test]
async fn review_commit_accepts_plan_file_and_context_args() {
    let srv = fresh_server();
    let resp = call_tool(
        &srv,
        "review_commit",
        json!({
            "commit": "HEAD",
            "repo": "/tmp/definitely-not-a-git-repo",
            "context": "tactical context",
            "plan_file": "/tmp/nonexistent-plan.md"
        }),
    )
    .await;
    // git show fails since the path isn't a repo. Confirms args route
    // through the new plumbing without panicking.
    assert_eq!(resp["result"]["isError"], true);
}
