//! MCP tool `train_from_conversations` — fine-tune a local model on
//! the user's recent conversation history.
//!
//! Architectural boundary: lamu-mcp does NOT link against
//! lamu-train. This module shells out to the `lamu-train` binary
//! via `tokio::process::Command`. The MCP server stays
//! plug-and-play for any harness; the training subsystem is a
//! separate program that LAMU's data feeds into.
//!
//! Two phases:
//!
//!   1. Confirmation gate — first call without `confirm: true`
//!      returns a dataset estimate (conversation count + turn
//!      count) computed via `Memory::recall_since`. No subprocess
//!      spawned; user reads the estimate and decides.
//!
//!   2. Spawn — second call with `confirm: true` shells out to
//!      `lamu-train --from-conversations --background <name>`
//!      with a fully detached child (no zombie reaping, no
//!      kill-on-drop). Returns immediately with a hint to check
//!      `lamu-train jobs` for the running job.
//!
//! The job_id is NOT captured at spawn time — that would require
//! either piping stderr (defeats detach) or coordinating an id
//! between the two binaries (couples them more than we want for
//! one feature). The user lists jobs to find the new entry.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde_json::Value;

use crate::memory;

/// Locate the `lamu-train` binary. Resolution:
///   1. `$LAMU_TRAIN_BIN` env (explicit override; useful for tests
///      and bespoke installs)
///   2. `lamu-train` on `$PATH` via `which`
///
/// Errors with the env var name in the message so users have one
/// sentence to fix.
fn resolve_train_binary() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("LAMU_TRAIN_BIN") {
        return Ok(PathBuf::from(p));
    }
    which::which("lamu-train")
        .map_err(|e| format!(
            "lamu-train binary not found on $PATH: {e}. \
             Install via `cargo install --path lamu-rs/lamu-train` \
             or set $LAMU_TRAIN_BIN."
        ))
}

/// Parse the `since` arg. Accepts humantime durations (`30d`,
/// `7d`, `12h`, etc.). Empty / missing → default 30 days.
fn parse_since(s: Option<&str>) -> Result<Duration, String> {
    let raw = s.unwrap_or("30d").trim();
    if raw.is_empty() {
        return Ok(Duration::from_secs(30 * 24 * 60 * 60));
    }
    humantime::parse_duration(raw).map_err(|e| format!("--since '{raw}': {e}"))
}

/// Group `recall_since` rows by conversation id, applying the same
/// filter rules as `lamu-train::conversations::dump_to_jsonl` so
/// the estimate matches what the trainer will actually see.
fn estimate_dataset(cutoff_unix_secs: i64) -> Result<(usize, usize), String> {
    const MIN_TURNS: usize = 4;
    let mem = memory::shared().map_err(|e| format!("open memory: {e}"))?;
    let rows = mem
        .recall_since(cutoff_unix_secs)
        .map_err(|e| format!("recall_since: {e}"))?;

    let mut by_conv: std::collections::BTreeMap<String, Vec<()>> =
        std::collections::BTreeMap::new();
    let mut error_rows = 0usize;
    let mut oversize_rows = 0usize;
    for (conv, turn) in rows {
        if turn.content.starts_with("error:") {
            error_rows += 1;
            continue;
        }
        if turn.content.len() > 200 * 1024 {
            oversize_rows += 1;
            continue;
        }
        by_conv.entry(conv).or_default().push(());
    }
    let mut n_convs = 0usize;
    let mut n_turns = 0usize;
    for (_, turns) in by_conv {
        if turns.len() >= MIN_TURNS {
            n_convs += 1;
            n_turns += turns.len();
        }
    }
    let _ = (error_rows, oversize_rows); // counts available if we
                                          // ever want to surface them
                                          // in the estimate
    Ok((n_convs, n_turns))
}

/// Validate the output_name. Must be a safe registry name —
/// mirrors `lamu-train::spec::is_safe_registry_name`. Duplicating
/// the rule here avoids the dep; if the rule diverges the failure
/// surfaces as a clear validation error in lamu-train.
fn is_safe_output_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && !name.starts_with('-')
        && !name.contains("..")
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// Validate the base_model arg as `org/name` HF repo id — mirrors
/// `lamu-train::spec::is_safe_hf_repo_id`. Same rationale as
/// is_safe_output_name.
fn is_safe_hf_repo_id(repo: &str) -> bool {
    let trimmed = repo.trim();
    if trimmed.is_empty() || trimmed.starts_with('/') {
        return false;
    }
    if trimmed.contains('\\') || trimmed.contains('\0') {
        return false;
    }
    let parts: Vec<&str> = trimmed.split('/').collect();
    if parts.len() != 2 {
        return false;
    }
    parts.iter().all(|p| {
        !p.is_empty()
            && *p != "."
            && !p.contains("..")
            && p.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    })
}

pub async fn handle_train_from_conversations(args: Value) -> String {
    let output_name = match args.get("output_name").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => return "error: 'output_name' is required".into(),
    };
    if !is_safe_output_name(&output_name) {
        return format!(
            "error: output_name '{output_name}' must match [A-Za-z0-9_.-]+ \
             with no leading '.' or '-' and no '..' substring"
        );
    }

    let base_model = args
        .get("base_model")
        .and_then(|v| v.as_str())
        .unwrap_or("Qwen/Qwen3-7B")
        .to_string();
    if !is_safe_hf_repo_id(&base_model) {
        return format!(
            "error: base_model '{base_model}' must be a HuggingFace repo id (org/name)"
        );
    }

    let method = args
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("qlora")
        .to_string();
    if !matches!(method.as_str(), "qlora" | "lora" | "full") {
        return format!("error: method '{method}' must be one of qlora|lora|full");
    }

    let since_str = args.get("since").and_then(|v| v.as_str());
    let since = match parse_since(since_str) {
        Ok(d) => d,
        Err(e) => return format!("error: {e}"),
    };

    let confirm = args
        .get("confirm")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Confirmation gate. Compute estimate from in-process memory
    // — no subprocess yet.
    let cutoff = std::time::SystemTime::now()
        .checked_sub(since)
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (n_convs, n_turns) = match estimate_dataset(cutoff) {
        Ok(c) => c,
        Err(e) => return format!("error: estimate dataset: {e}"),
    };

    if !confirm {
        return format!(
            "error: training is expensive (30 min – 4 h, locks the GPU). \
             Pass confirm=true to proceed.\n\
             Estimated dataset over the last {}: {} conversations, {} turns.\n\
             Output model would land in the registry as '{}'.\n\
             GPU will be unavailable to inference (`query`, HTTP server, \
             `lamu run`) until the run completes; clients return a clear \
             error and can pass --allow-evict to wait.",
            humantime::format_duration(since),
            n_convs,
            n_turns,
            output_name
        );
    }

    if n_convs == 0 {
        return format!(
            "error: no usable conversations in the last {}. \
             A usable conversation has at least 4 non-error, non-oversize turns.",
            humantime::format_duration(since)
        );
    }

    // Confirmed. Shell out to lamu-train.
    let bin = match resolve_train_binary() {
        Ok(p) => p,
        Err(e) => return format!("error: {e}"),
    };

    let since_arg = humantime::format_duration(since).to_string();
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.arg(&output_name)
        .arg("--from-conversations")
        .arg("--since")
        .arg(&since_arg)
        .arg("--base")
        .arg(&base_model)
        .arg("--method")
        .arg(&method)
        .arg("--background")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // Detach: do NOT kill on drop, do NOT wait on the child. The
    // training process must outlive the MCP request. We fire-and-
    // forget. lamu-train manages its own job dir + state file;
    // the user lists `lamu-train jobs` to find the new entry.
    cmd.kill_on_drop(false);

    match cmd.spawn() {
        Ok(child) => {
            let pid = child.id().unwrap_or(0);
            // Drop the Child handle in a background task that
            // wait()s on it. Without this, the child becomes a
            // zombie until the lamu-mcp daemon exits. The task
            // outlives this request — fine, since our daemon is
            // long-lived and one task per training run is cheap.
            let mut child = child;
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            format!(
                "training started: pid={pid}, output_name='{output_name}'.\n\
                 Run `lamu-train jobs` to find the job id, \
                 `lamu-train log <id>` for live progress."
            )
        }
        Err(e) => format!("error: spawn lamu-train: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_since_default_30d() {
        let d = parse_since(None).unwrap();
        assert_eq!(d.as_secs(), 30 * 24 * 3600);
    }

    #[test]
    fn parse_since_accepts_humantime() {
        assert_eq!(parse_since(Some("7d")).unwrap().as_secs(), 7 * 24 * 3600);
        assert_eq!(parse_since(Some("12h")).unwrap().as_secs(), 12 * 3600);
    }

    #[test]
    fn parse_since_empty_treated_as_default() {
        let d = parse_since(Some("")).unwrap();
        assert_eq!(d.as_secs(), 30 * 24 * 3600);
    }

    #[test]
    fn parse_since_invalid_errors() {
        assert!(parse_since(Some("nonsense")).is_err());
    }

    #[test]
    fn is_safe_output_name_basic() {
        for ok in ["alpha", "test-7b", "qwen3.6_v2"] {
            assert!(is_safe_output_name(ok), "{ok} should be safe");
        }
        for bad in ["", ".hidden", "-leading", "a..b", "a/b", "name space"] {
            assert!(!is_safe_output_name(bad), "{bad} should reject");
        }
    }

    #[test]
    fn is_safe_hf_repo_id_basic() {
        assert!(is_safe_hf_repo_id("Qwen/Qwen3-7B"));
        assert!(is_safe_hf_repo_id("org/repo_v2.0"));
        assert!(!is_safe_hf_repo_id(""));
        assert!(!is_safe_hf_repo_id("no-slash"));
        assert!(!is_safe_hf_repo_id("/abs/path"));
        assert!(!is_safe_hf_repo_id("../etc/passwd"));
        assert!(!is_safe_hf_repo_id("a/b/c"));
        assert!(!is_safe_hf_repo_id("a\\b"));
    }

    #[tokio::test]
    async fn missing_output_name_errors() {
        let r = handle_train_from_conversations(json!({})).await;
        assert!(r.starts_with("error:"));
        assert!(r.contains("output_name"));
    }

    #[tokio::test]
    async fn unsafe_output_name_errors() {
        let r = handle_train_from_conversations(json!({
            "output_name": "../escape",
        }))
        .await;
        assert!(r.starts_with("error:"));
        assert!(r.contains("output_name"));
    }

    #[tokio::test]
    async fn unsafe_base_model_errors() {
        let r = handle_train_from_conversations(json!({
            "output_name": "ok",
            "base_model": "/abs/path",
        }))
        .await;
        assert!(r.starts_with("error:"));
        assert!(r.contains("base_model"));
    }

    #[tokio::test]
    async fn invalid_method_errors() {
        let r = handle_train_from_conversations(json!({
            "output_name": "ok",
            "method": "rlhf",
        }))
        .await;
        assert!(r.starts_with("error:"));
        assert!(r.contains("method"));
    }

    #[tokio::test]
    async fn invalid_since_errors() {
        let r = handle_train_from_conversations(json!({
            "output_name": "ok",
            "since": "yesterday",
        }))
        .await;
        assert!(r.starts_with("error:"));
    }

    #[tokio::test]
    async fn missing_confirm_returns_estimate() {
        // No confirm flag → estimate-and-refuse path. The estimate
        // call will read whatever memory.db is in scope; in CI
        // that's a fresh db (no rows) so n_convs = 0, but the gate
        // text fires before the empty-dataset check.
        let r = handle_train_from_conversations(json!({
            "output_name": "ok",
        }))
        .await;
        assert!(r.starts_with("error:"));
        assert!(
            r.contains("confirm=true"),
            "missing confirmation hint: {r}"
        );
        assert!(r.contains("Estimated dataset"));
        assert!(r.contains("--allow-evict"));
    }
}
