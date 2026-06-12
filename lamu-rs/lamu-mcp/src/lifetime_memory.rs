//! Lifetime cross-session memory — MCP frontend: cloud-judged
//! orchestration + tool handlers.
//!
//! The storage core (schema + temporal migration, `remember` /
//! `recall_memory` / `supersede` / `forget`, novelty dedup, graphify
//! corpus export) moved to `lamu_memory::lifetime_memory` (ADR 0029:
//! memory is a shared capability used by multiple frontends — lamu-mcp
//! tools today, lamu-api HTTP memory routes next — and a frontend must
//! not depend on another frontend, so storage lives in a module crate).
//! The glob re-export below keeps every existing
//! `crate::lifetime_memory::X` call site compiling unchanged.
//!
//! What stays here is genuinely frontend-shaped:
//! - the MiMo-judged orchestration ([`consolidate`],
//!   [`extract_from_exchange`], [`reconcile_memory`] /
//!   [`maybe_spawn_reconcile`]) — it calls `crate::cloud` and fences
//!   recalled content via `crate::untrusted` (ADR 0011);
//! - the five MCP tool handlers (`handle_remember`,
//!   `handle_recall_memory`, `handle_consolidate_memory`,
//!   `handle_forget_memory`, `handle_export_memory_graph`). Wire
//!   contracts frozen.

use anyhow::{anyhow, Result};
use std::path::PathBuf;

pub use lamu_memory::lifetime_memory::*;

/// System prompt for fact extraction. Instructs MiMo to keep only
/// durable, user-specific facts worth remembering across sessions.
const EXTRACTION_PROMPT: &str = "\
You extract DURABLE, user-specific facts worth remembering across future \
sessions from the conversation transcript below. Keep only stable, \
re-usable facts: the user's identity, preferences, project facts, \
tooling/environment, and decisions they have made. Drop ephemeral \
chit-chat, one-off questions, transient state, and anything that will \
not matter next session.\n\
\n\
Output ONE fact per line, with no preamble, no numbering, no bullets, and \
no commentary. Each line must be a self-contained statement that reads \
correctly with no surrounding context. If nothing in the transcript is \
worth keeping, output exactly NONE and nothing else.";

// ── Consolidation (fact extraction) ────────────────────────────────

/// Same caller-supplied-id allowlist `write_file` / `memory.rs` enforce
/// — fail fast with a clear message rather than relying on the inner
/// `recall` validation, so the error surfaces at the tool boundary.
fn validate_conversation_id(id: &str) -> Result<()> {
    if id.is_empty() {
        return Err(anyhow!("conversation_id is empty"));
    }
    if id.starts_with('.') {
        return Err(anyhow!("conversation_id cannot start with '.': {id}"));
    }
    if id.contains("..") {
        return Err(anyhow!("conversation_id contains '..': {id}"));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(anyhow!(
            "conversation_id contains forbidden character — allowed: [A-Za-z0-9_-.]: {id}"
        ));
    }
    Ok(())
}

/// Extract durable facts from a conversation via MiMo and store each as
/// a fact memory keyed by the conversation id. Returns the count stored.
///
/// `recall(id, 0)` uses `limit = 0` to mean "no cap" (full transcript).
/// NOTE: the transcript is sent to MiMo with the spec-mandated
/// `max_tokens: 1024` for the *output*; an extremely long input
/// conversation could exceed the model's context window. Input
/// truncation is a deliberate follow-up (see commit message).
pub async fn consolidate(conversation_id: &str) -> Result<usize> {
    validate_conversation_id(conversation_id)?;
    let mem = crate::memory::shared()?;
    let turns = mem.recall(conversation_id, 0)?;
    if turns.is_empty() {
        return Ok(0);
    }
    // Compact "role: content" transcript — same shape memory.rs embeds.
    let transcript = turns
        .iter()
        .map(|t| format!("{}: {}", t.role, t.content))
        .collect::<Vec<_>>()
        .join("\n");

    let args = serde_json::json!({
        "model": "mimo-v2.5",
        "system": EXTRACTION_PROMPT,
        "prompt": transcript,
        "max_tokens": 1024,
        "temperature": 0.2,
        "include_reasoning": false,
    });
    let resp = crate::cloud::handle_cloud_query(args).await;
    if resp.starts_with("error:") {
        return Err(anyhow!("fact extraction failed: {resp}"));
    }

    let facts = parse_extracted_facts(&resp);
    // Best-effort: a per-fact embed/insert hiccup (network/rate-limit on
    // the Nth fact) must NOT abort the rest or hide the facts already
    // stored. Log-and-continue, return the count actually persisted, so a
    // transient failure on one fact doesn't both lose the others and make
    // the caller think nothing was stored.
    let mut stored = 0usize;
    for fact in facts {
        match remember(&fact, "fact", conversation_id).await {
            Ok(_) => stored += 1,
            Err(e) => tracing::warn!("consolidate({conversation_id}): store fact failed: {e}"),
        }
    }
    Ok(stored)
}

// ── Autocapture (single-exchange extraction) ────────────────────────

/// Extract durable facts from a SINGLE user/assistant exchange via MiMo.
///
/// This is `consolidate()`'s extraction step applied to one turn-pair
/// instead of a whole conversation: build a compact transcript, send it
/// to MiMo with the shared [`EXTRACTION_PROMPT`], and parse the reply
/// with [`parse_extracted_facts`]. Deliberately OMITS `conversation_id`
/// from the cloud args so the recall-and-prepend branch never engages —
/// it stays a stateless one-shot and cannot recurse into autocapture.
pub async fn extract_from_exchange(user: &str, assistant: &str) -> Result<Vec<String>> {
    let transcript = format!("User: {user}\n\nAssistant: {assistant}");
    let args = serde_json::json!({
        "model": "mimo-v2.5",
        "system": EXTRACTION_PROMPT,
        "prompt": transcript,
        "max_tokens": 512,
        "temperature": 0.2,
        "include_reasoning": false,
    });
    // LOAD-BEARING: `args` has NO `conversation_id`, so handle_cloud_query's
    // autocapture gate (`!conv_id.is_empty()`) is false for THIS call — the
    // extraction request cannot itself trigger autocapture and recurse.
    let resp = crate::cloud::handle_cloud_query(args).await;
    if resp.starts_with("error:") {
        return Err(anyhow!("fact extraction failed: {resp}"));
    }
    Ok(parse_extracted_facts(&resp))
}

// ── Auto-contradiction (retire facts a new one supersedes) ──────────

const CONTRADICTION_PROMPT: &str = "\
You compare a NEW fact against EXISTING stored facts and report which \
EXISTING facts the new fact makes OUTDATED — same subject with a \
conflicting value (e.g. NEW 'lives in SF' vs EXISTING 'lives in NYC', or \
NEW 'uses Rust' vs EXISTING 'uses Go'). Do NOT flag facts that are merely \
related, additional, or compatible — only direct contradictions/updates. \
Each EXISTING fact is annotated with its source (e.g. user-stated, \
extracted, tool-ingested). Be MORE conservative about marking a \
user-stated fact outdated than a tool-ingested or model-derived one — \
require a clear, direct conflict, since the human's own statement \
outranks an inferred one. \
Reply with ONLY a JSON object {\"outdated\": [<id>, ...]} listing the ids \
of the EXISTING facts the new fact supersedes; empty list if none.";

/// Parse the judge's reply into the ids of existing facts to retire.
/// Accepts `{"outdated":[..]}`, a bare `[..]`, optionally wrapped in code
/// fences or prose. Returns `[]` on anything unparseable.
pub(crate) fn parse_contradiction_ids(reply: &str) -> Vec<i64> {
    let trimmed = reply
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(arr) = v.get("outdated").and_then(|x| x.as_array()) {
            return arr.iter().filter_map(|x| x.as_i64()).collect();
        }
        if let Some(arr) = v.as_array() {
            return arr.iter().filter_map(|x| x.as_i64()).collect();
        }
    }
    // Fallback: first bracketed int list anywhere in the reply.
    if let (Some(s), Some(e)) = (reply.find('['), reply.rfind(']')) {
        if e > s {
            if let Ok(arr) = serde_json::from_str::<Vec<i64>>(&reply[s..=e]) {
                return arr;
            }
        }
    }
    Vec::new()
}

fn autocontradict_enabled() -> bool {
    matches!(
        std::env::var("LAMU_AUTOCONTRADICT")
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1") | Some("true")
    )
}

/// Find current facts that the freshly-stored `new_text` (rowid `new_id`)
/// makes outdated and expire them via [`forget`]. MiMo judges against the
/// new fact's nearest current neighbors; only ids that were actually in the
/// candidate set are acted on (guards against hallucinated ids). Near-
/// identical neighbors (cosine ≥ NOVELTY_THRESHOLD) are skipped — those are
/// duplicates, not contradictions. Returns the count expired. Needs an
/// OPENAI key (neighbor search) + a reachable cloud model.
pub async fn reconcile_memory(new_id: i64, new_text: &str) -> Result<usize> {
    let neighbors = recall_memory(new_text, 8, false).await?;
    let candidates: Vec<&MemoryHit> = neighbors
        .iter()
        .filter(|h| h.id != new_id)
        .filter(|h| h.score.map(|s| s < NOVELTY_THRESHOLD).unwrap_or(true))
        .take(6)
        .collect();
    if candidates.is_empty() {
        return Ok(0);
    }
    let listing = candidates
        .iter()
        .map(|h| format!("[{}] (source: {}) {}", h.id, h.source.as_deref().unwrap_or("unknown"), h.text))
        .collect::<Vec<_>>()
        .join("\n");
    // The recalled neighbors are the attack vector (a poisoned fact could tell
    // the judge to expire a real one); fence them. `new_text` is the just-stored
    // user fact and stays trusted. The id-validation below is orthogonal — it
    // guards against acting on hallucinated ids; the fence guards the verdict.
    let args = serde_json::json!({
        "model": "mimo-v2.5",
        "system": format!("{}\n\n---\n\n{}", crate::untrusted::UNTRUSTED_POLICY, CONTRADICTION_PROMPT),
        "prompt": format!(
            "NEW fact:\n{new_text}\n\nEXISTING facts:\n{}",
            crate::untrusted::wrap_untrusted("recalled facts", &listing)
        ),
        "max_tokens": 256,
        "temperature": 0.1,
        "include_reasoning": false,
    });
    let resp = crate::cloud::handle_cloud_query(args).await;
    if resp.starts_with("error:") {
        return Err(anyhow!("contradiction judge failed: {resp}"));
    }
    let valid: std::collections::HashSet<i64> = candidates.iter().map(|h| h.id).collect();
    let mut expired = 0usize;
    for id in parse_contradiction_ids(&resp) {
        if id == new_id || !valid.contains(&id) {
            continue; // self, or a hallucinated id outside the candidate set
        }
        match forget(id) {
            Ok(true) => expired += 1,
            Ok(false) => {}
            Err(e) => tracing::warn!("reconcile: forget({id}) failed: {e}"),
        }
    }
    Ok(expired)
}

/// Opt-in (`LAMU_AUTOCONTRADICT=1`) fire-and-forget reconcile after a
/// remember. Detached on the current tokio runtime; only owned data crosses
/// the boundary. No-op when the flag is off or no runtime is present (tests).
pub fn maybe_spawn_reconcile(new_id: i64, new_text: &str) {
    if !autocontradict_enabled() || tokio::runtime::Handle::try_current().is_err() {
        return;
    }
    let txt = new_text.to_string();
    tokio::spawn(async move {
        match reconcile_memory(new_id, &txt).await {
            Ok(n) if n > 0 => {
                tracing::info!("auto-contradiction: retired {n} outdated fact(s) superseded by #{new_id}")
            }
            Ok(_) => {}
            Err(e) => tracing::debug!("auto-contradiction(#{new_id}): {e}"),
        }
    });
}

// ── MCP tool handlers ──────────────────────────────────────────────

/// `remember` tool handler. Local store, embedding optional.
/// Args: `text` (required), `kind` (default "fact"), `source` (default "manual").
pub(crate) async fn handle_remember(args: serde_json::Value) -> String {
    let text = args["text"].as_str().unwrap_or("").trim();
    if text.is_empty() {
        return "error: text is required".to_string();
    }
    let kind = match args["kind"].as_str() {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => "fact",
    };
    let source = match args["source"].as_str() {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => "manual",
    };
    match remember(text, kind, source).await {
        Ok(id) => {
            // Opt-in: retire any current fact this one contradicts (detached).
            maybe_spawn_reconcile(id, text);
            format!("remembered memory #{id} (kind={kind}, source={source})")
        }
        Err(e) => format!("error: {e}"),
    }
}

/// `recall_memory` tool handler. Local read; degrades to recency
/// without a key. Args: `query` (required), `k` (default 8).
pub(crate) async fn handle_recall_memory(args: serde_json::Value) -> String {
    let query = args["query"].as_str().unwrap_or("").trim();
    if query.is_empty() {
        return "error: query is required".to_string();
    }
    // Cap k so a pathological request can't ask for an unbounded result
    // set (the underlying scan is already bounded; this bounds output).
    let k = (args["k"].as_u64().unwrap_or(8) as usize).min(100);
    // Default false: hide superseded/forgotten facts. true → historical
    // recall over the full timeline.
    let include_expired = args["include_expired"].as_bool().unwrap_or(false);
    match recall_memory(query, k, include_expired).await {
        Ok(hits) if hits.is_empty() => "(no memories matched)".to_string(),
        Ok(hits) => {
            let mut out = String::new();
            for h in hits {
                let score = match h.score {
                    Some(s) => format!("{s:.3}"),
                    None => "recency".to_string(),
                };
                let source = h.source.as_deref().unwrap_or("?");
                out.push_str(&format!(
                    "#{} [{}] (source={}, score={}) {}\n",
                    h.id, h.kind, source, score, h.text
                ));
            }
            // This tool result reaches the outer agent (Claude Code) verbatim;
            // a poisoned memory could carry an injection. Fence it as data.
            crate::untrusted::wrap_untrusted("recalled memory", &out)
        }
        Err(e) => format!("error: {e}"),
    }
}

/// `consolidate_memory` tool handler. Cloud (requires MiMo extraction)
/// — gated under local-only by the dispatcher. Args: `conversation_id`
/// (required).
pub(crate) async fn handle_consolidate_memory(args: serde_json::Value) -> String {
    let conversation_id = args["conversation_id"].as_str().unwrap_or("").trim();
    if conversation_id.is_empty() {
        return "error: conversation_id is required".to_string();
    }
    match consolidate(conversation_id).await {
        Ok(n) => format!("stored {n} memories from {conversation_id}"),
        Err(e) => format!("error: {e}"),
    }
}

/// `forget_memory` tool handler. Soft-deletes a fact (sets `valid_until`)
/// — local store op, no network. Args: `id` (required integer).
pub(crate) async fn handle_forget_memory(args: serde_json::Value) -> String {
    let id = match args["id"].as_i64() {
        Some(id) => id,
        None => return "error: id is required (integer)".to_string(),
    };
    match forget(id) {
        Ok(true) => format!("forgot memory {id}"),
        Ok(false) => format!("no current memory with id {id}"),
        Err(e) => format!("error: {e}"),
    }
}

/// `export_memory_graph` tool handler. Writes the graphify corpus — local
/// filesystem op, no network. Args: `dir` (default
/// `<data_dir>/lamu/memory-corpus`), `include_expired` (default false).
pub(crate) async fn handle_export_memory_graph(args: serde_json::Value) -> String {
    let dir: PathBuf = match args["dir"].as_str() {
        Some(s) if !s.trim().is_empty() => PathBuf::from(s.trim()),
        _ => {
            let base = match dirs::data_local_dir() {
                Some(d) => d,
                None => return "error: no data_local_dir for default corpus path".to_string(),
            };
            base.join("lamu").join("memory-corpus")
        }
    };
    // Defense-in-depth: allow absolute paths (the documented
    // `/graphify <abs-dir>` workflow needs them) but reject `..`
    // traversal so a controlled `dir` arg can't escape upward into a
    // surprising tree. Mirrors write_file's `..` refusal.
    if dir.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return "error: '..' is not allowed in the export dir path".to_string();
    }
    let include_expired = args["include_expired"].as_bool().unwrap_or(false);
    match export_graph_corpus(&dir, include_expired) {
        Ok(n) => format!(
            "wrote {n} memories to {} — run `/graphify {}` (or `graphify {}`) to build the \
             entity/hypergraph/community graph; it has an MCP server (graphify.serve) for \
             live querying.",
            dir.display(),
            dir.display(),
            dir.display()
        ),
        Err(e) => format!("error: {e}"),
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_conversation_id_allowlist() {
        assert!(validate_conversation_id("sess-1_2.3").is_ok());
        assert!(validate_conversation_id("a.b").is_ok()); // non-leading dot allowed
        assert!(validate_conversation_id("").is_err());
        assert!(validate_conversation_id(".").is_err()); // bare dot → leading-dot reject
        assert!(validate_conversation_id(".hidden").is_err());
        assert!(validate_conversation_id("a..b").is_err());
        assert!(validate_conversation_id("a/b").is_err());
        assert!(validate_conversation_id("a b").is_err());
    }

    #[test]
    fn parse_contradiction_ids_handles_shapes() {
        assert_eq!(parse_contradiction_ids(r#"{"outdated":[3,7]}"#), vec![3, 7]);
        assert_eq!(parse_contradiction_ids("```json\n{\"outdated\":[]}\n```"), Vec::<i64>::new());
        assert_eq!(parse_contradiction_ids("[1, 2]"), vec![1, 2]);
        assert_eq!(
            parse_contradiction_ids("facts 5 and 9 conflict: [5, 9]."),
            vec![5, 9]
        );
        assert_eq!(parse_contradiction_ids("none of them"), Vec::<i64>::new());
        assert_eq!(parse_contradiction_ids(""), Vec::<i64>::new());
    }

    #[tokio::test]
    async fn export_memory_graph_rejects_parent_dir_traversal() {
        // The `..` guard fires BEFORE any db/fs work, so this is hermetic.
        let r = handle_export_memory_graph(serde_json::json!({"dir": "../escape"})).await;
        assert!(r.starts_with("error:"), "got: {r}");
        assert!(r.contains(".."), "got: {r}");
        let r2 = handle_export_memory_graph(serde_json::json!({"dir": "ok/../../up"})).await;
        assert!(r2.starts_with("error:"), "got: {r2}");
    }
}
