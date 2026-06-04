//! ADR 0021 C4/C5: `compact_context` — tools to compact a conversation.
//!
//! Stateless by default (C4): the caller passes a `messages` array, we preserve
//! the leading system turns + the last `keep_recent` turns verbatim, summarize
//! only the stale middle via the cloud model, and return the shrunk list. The
//! opt-in persist path (C5, `conversation_id` + `persist:true`) rewrites the
//! stored cloud_query conversation with append-only supersede markers.

use crate::server::LamuMcpServer;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

/// In-flight persist compactions keyed by conversation_id — at most one per id,
/// so two concurrent compactions can't both summarize-and-supersede the same
/// conversation (ADR 0021 C5).
static COMPACTING: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// RAII release of the per-conversation compaction lock (drops on any return,
/// including the early-return error paths and a panic).
struct CompactGuard(String);
impl Drop for CompactGuard {
    fn drop(&mut self) {
        if let Ok(mut g) = COMPACTING.lock() {
            g.remove(&self.0);
        }
    }
}

/// Compaction-specific summary prompt. Deliberately NOT the cross-session
/// `EXTRACTION_PROMPT` (which drops transient/in-flight state) — a
/// mid-conversation compaction MUST keep the working state needed to continue.
const SUMMARIZATION_PROMPT: &str = "\
You are compacting the middle of an ongoing work conversation to save context. \
Summarize the excerpt below into a dense briefing that lets the work continue \
with NO loss of load-bearing detail. PRESERVE: decisions made and their \
rationale, exact file paths, function/type/API names, command invocations, \
open questions, unresolved errors, and the current task state. DROP: \
pleasantries, acknowledgements, and redundant restatements. Output only the \
summary, with no preamble.";

/// char/4 token ESTIMATE. lamu has no general tokenizer (only a loaded model's
/// /tokenize, via context_status); this is labeled `approx_tokens` everywhere
/// so it is never mistaken for the engine-truth count.
fn estimate_tokens(messages: &[Value]) -> u64 {
    let chars: usize = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .map(|s| s.len())
        .sum();
    (chars / 4) as u64
}

/// Partition into (head = leading system turns, middle = stale, tail = last
/// `keep_recent` non-head turns). Preserve-first: head + tail are returned
/// verbatim; only the middle is ever summarized.
fn partition_messages(messages: &[Value], keep_recent: usize) -> (Vec<Value>, Vec<Value>, Vec<Value>) {
    let head_len = messages
        .iter()
        .take_while(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .count();
    let head = messages[..head_len].to_vec();
    let rest = &messages[head_len..];
    let tail_len = keep_recent.min(rest.len());
    let split = rest.len() - tail_len;
    (head, rest[..split].to_vec(), rest[split..].to_vec())
}

fn summary_message(n: usize, summary: &str) -> Value {
    json!({
        "role": "system",
        "content": format!("[compacted summary of {n} earlier turn(s)]\n{summary}"),
    })
}

fn render_turns(turns: &[Value]) -> String {
    turns
        .iter()
        .map(|m| {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            // Non-string content (multimodal arrays/objects) is JSON-serialized
            // rather than dropped, so its text survives into the summary.
            let content = m
                .get("content")
                .map(|c| c.as_str().map(String::from).unwrap_or_else(|| c.to_string()))
                .unwrap_or_default();
            format!("{role}: {content}")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

impl LamuMcpServer {
    /// ADR 0021 C4: stateless conversation compaction. Returns a dry-run plan
    /// unless `confirm:true`. Preserves head (system) + tail (last keep_recent)
    /// verbatim, summarizes the middle via the cloud model, returns the shrunk
    /// `messages`. Never mutates anything (the persist path is C5).
    pub(crate) async fn handle_compact_context(&self, args: Value) -> String {
        // C5 persist path: a conversation_id targets the stored cloud_query log
        // (rewrite-in-place via append-only supersede markers). Stateless
        // message-list mode is the default below.
        if let Some(cid) = args.get("conversation_id").and_then(|v| v.as_str()) {
            return self.compact_persist(cid.to_string(), args.clone()).await;
        }

        let messages: Vec<Value> = match args.get("messages").and_then(|m| m.as_array()) {
            Some(a) if !a.is_empty() => a.clone(),
            _ => {
                return json!({
                    "error": "compact_context requires a non-empty `messages` array (stateless mode)"
                })
                .to_string()
            }
        };
        let keep_recent = args.get("keep_recent").and_then(|v| v.as_u64()).unwrap_or(6) as usize;
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("mimo-v2.5")
            .to_string();
        let confirm = args.get("confirm").and_then(|v| v.as_bool()).unwrap_or(false);
        let max_summary_tokens = args
            .get("max_summary_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024);

        let (head, middle, tail) = partition_messages(&messages, keep_recent);
        let before_tokens = estimate_tokens(&messages);

        if middle.is_empty() {
            return json!({
                "compacted": false,
                "reason": "nothing to compact — middle is empty (turns <= head + keep_recent)",
                "before": {"turns": messages.len(), "approx_tokens": before_tokens},
            })
            .to_string();
        }

        if !confirm {
            // Rough post-compaction estimate: preserved head+tail verbatim plus
            // the summary capped at max_summary_tokens. The savings is the whole
            // point of a dry-run, so surface it.
            let est_after = estimate_tokens(&head) + estimate_tokens(&tail) + max_summary_tokens;
            return json!({
                "dry_run": true,
                "before": {"turns": messages.len(), "approx_tokens": before_tokens},
                "plan": {
                    "preserve_head_system_turns": head.len(),
                    "summarize_middle_turns": middle.len(),
                    "preserve_tail_turns": tail.len(),
                    "estimated_after_approx_tokens": est_after,
                    "estimated_savings_approx_tokens": before_tokens.saturating_sub(est_after),
                },
                "note": "estimates are char/4 approximations; call again with confirm:true to perform the compaction",
            })
            .to_string();
        }

        // Self-enforce the routing gate. The confirm path makes a cloud call,
        // but the tool is `cloud:false` (so dry-run stays usable in local-only),
        // which means the dispatcher's local-only refusal doesn't fire here —
        // mirror parallel_query (handlers.rs) and refuse the cloud step itself.
        // Don't hold the lock across the .await below.
        if self.routing_mode.lock().await.as_str() == "local-only" {
            return json!({
                "compacted": false,
                "error": "routing mode is 'local-only' — compact_context's summary needs the cloud model. Dry-run (confirm:false) still works; or set_routing_mode(mode='auto').",
            })
            .to_string();
        }

        // NOTE: middle turns are untrusted conversation content fed to the
        // summarizer. The call is ephemeral and tool-less, so an injected
        // instruction can at worst degrade the summary — there is no data-exfil
        // path. Callers should only compact conversations they trust.
        let summary = crate::cloud::handle_cloud_query(json!({
            "model": model,
            "system": SUMMARIZATION_PROMPT,
            "prompt": render_turns(&middle),
            "max_tokens": max_summary_tokens,
            "temperature": 0.3,
            "ephemeral": true,
        }))
        .await;
        // handle_cloud_query signals every failure with an "error:" prefix (the
        // stringly contract every caller relies on). Also reject an empty
        // summary so a blank is never spliced into the messages.
        if summary.starts_with("error:") || summary.trim().is_empty() {
            return json!({"compacted": false, "error": format!("summarization failed: {summary}")})
                .to_string();
        }

        let mut compacted = head.clone();
        compacted.push(summary_message(middle.len(), &summary));
        compacted.extend(tail.clone());
        let after_tokens = estimate_tokens(&compacted);

        json!({
            "compacted": true,
            "messages": compacted,
            "before": {"turns": messages.len(), "approx_tokens": before_tokens},
            "after": {"turns": compacted.len(), "approx_tokens": after_tokens},
            "summarized_middle_turns": middle.len(),
            "preserved": {"head_system": head.len(), "tail_recent": tail.len()},
            "note": "approx_tokens is a char/4 estimate; call context_status with the returned messages for the engine-truth count",
        })
        .to_string()
    }

    /// ADR 0021 C5: persist-mode compaction of the stored cloud_query
    /// conversation. Append-only — originals are never deleted; a supersede
    /// marker hides the summarized range on recall. Serialized per
    /// conversation_id; the lock releases on every return path.
    pub(crate) async fn compact_persist(&self, cid: String, args: Value) -> String {
        let _guard = {
            let mut g = match COMPACTING.lock() {
                Ok(g) => g,
                Err(_) => {
                    return json!({"compacted": false, "error": "compaction lock poisoned"}).to_string()
                }
            };
            if !g.insert(cid.clone()) {
                return json!({
                    "compacted": false,
                    "error": "a compaction is already in progress for this conversation"
                })
                .to_string();
            }
            CompactGuard(cid.clone())
        };
        self.compact_persist_inner(&cid, &args).await
    }

    async fn compact_persist_inner(&self, cid: &str, args: &Value) -> String {
        let keep_recent = args.get("keep_recent").and_then(|v| v.as_u64()).unwrap_or(6) as usize;
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("mimo-v2.5")
            .to_string();
        let confirm = args.get("confirm").and_then(|v| v.as_bool()).unwrap_or(false);
        let max_summary_tokens = args
            .get("max_summary_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024);

        let mem = match crate::memory::shared() {
            Ok(m) => m,
            Err(e) => {
                return json!({"compacted": false, "error": format!("memory unavailable: {e}")})
                    .to_string()
            }
        };
        // Fresh read (already supersede-filtered). BOTH dry-run and confirm read
        // here, so confirm computes the range from a CURRENT view: any turn
        // appended after a dry-run gets a higher idx, lands in the preserved
        // tail, and is never superseded (TOCTOU-safe without a long-held lock).
        let turns = match mem.recall(cid, 0) {
            Ok(t) => t,
            Err(e) => {
                return json!({"compacted": false, "error": format!("recall failed: {e}")})
                    .to_string()
            }
        };
        if turns.len() <= keep_recent {
            return json!({
                "compacted": false, "conversation_id": cid,
                "reason": "nothing to compact — conversation has <= keep_recent turns",
                "turns": turns.len(),
            })
            .to_string();
        }
        let split = turns.len() - keep_recent;
        let middle = &turns[..split];
        let lo = middle.first().map(|t| t.idx).unwrap_or(0);
        let hi = middle.last().map(|t| t.idx).unwrap_or(0);
        let before_tokens = (turns.iter().map(|t| t.content.len()).sum::<usize>() / 4) as u64;

        if !confirm {
            return json!({
                "dry_run": true, "conversation_id": cid,
                "before": {"turns": turns.len(), "approx_tokens": before_tokens},
                "plan": {
                    "summarize_middle_turns": middle.len(),
                    "preserve_tail_turns": keep_recent,
                    "supersede_idx_range": [lo, hi],
                },
                "note": "append-only persist (originals kept on disk); call again with confirm:true",
            })
            .to_string();
        }

        if self.routing_mode.lock().await.as_str() == "local-only" {
            return json!({
                "compacted": false,
                "error": "routing mode is 'local-only' — compaction summary needs the cloud model. Dry-run still works; or set_routing_mode(mode='auto').",
            })
            .to_string();
        }

        let middle_text = middle
            .iter()
            .map(|t| format!("{}: {}", t.role, t.content))
            .collect::<Vec<_>>()
            .join("\n\n");
        let summary = crate::cloud::handle_cloud_query(json!({
            "model": model,
            "system": SUMMARIZATION_PROMPT,
            "prompt": middle_text,
            "max_tokens": max_summary_tokens,
            "temperature": 0.3,
            "ephemeral": true,
        }))
        .await;
        if summary.starts_with("error:") || summary.trim().is_empty() {
            return json!({"compacted": false, "error": format!("summarization failed: {summary}")})
                .to_string();
        }

        // Append-only: write the summary as a new turn carrying the supersede
        // marker. recall now hides idx [lo,hi]; the originals stay on disk for
        // audit/undo. Nothing is deleted or overwritten.
        let md = json!({"kind": "compaction_summary", "supersedes": [lo, hi]}).to_string();
        let content = format!("[compacted summary of turns {lo}-{hi}]\n{summary}");
        if let Err(e) = mem.append_turn(cid, "system", &content, Some(&md)) {
            return json!({"compacted": false, "error": format!("append failed: {e}")}).to_string();
        }
        let after_turns = mem.recall(cid, 0).map(|t| t.len()).unwrap_or(0);
        json!({
            "compacted": true, "persisted": true, "conversation_id": cid,
            "superseded_idx_range": [lo, hi],
            "summarized_turns": middle.len(),
            "before_turns": turns.len(), "after_turns": after_turns,
            "note": "append-only — originals remain on disk; recall now hides the superseded range. Nothing was deleted.",
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> Value {
        json!({"role": role, "content": content})
    }

    #[test]
    fn partition_keeps_head_and_tail_verbatim() {
        let msgs = vec![
            msg("system", "sys"),
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("user", "u2"),
            msg("assistant", "a2"),
            msg("user", "u3"),
        ];
        let (head, middle, tail) = partition_messages(&msgs, 2);
        assert_eq!(head.len(), 1);
        assert_eq!(head[0]["content"], "sys");
        // middle = u1, a1, u2 (everything after head except last 2)
        assert_eq!(middle.len(), 3);
        // tail = a2, u3 (last 2, includes the latest user turn)
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[1]["content"], "u3");
    }

    #[test]
    fn partition_empty_middle_when_short() {
        // head(1) + keep_recent(6) covers all 4 turns → nothing to summarize.
        let msgs = vec![msg("system", "s"), msg("user", "u"), msg("assistant", "a"), msg("user", "u2")];
        let (_h, middle, _t) = partition_messages(&msgs, 6);
        assert!(middle.is_empty());
    }

    #[test]
    fn partition_no_system_head() {
        let msgs = vec![msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2")];
        let (head, middle, tail) = partition_messages(&msgs, 1);
        assert!(head.is_empty());
        assert_eq!(middle.len(), 2);
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0]["content"], "u2");
    }

    #[test]
    fn estimate_tokens_char_over_4() {
        let msgs = vec![msg("user", "abcd"), msg("assistant", "abcdefgh")];
        assert_eq!(estimate_tokens(&msgs), (4 + 8) / 4);
    }

    #[test]
    fn summary_message_labels_count() {
        let m = summary_message(5, "the gist");
        assert_eq!(m["role"], "system");
        assert!(m["content"].as_str().unwrap().starts_with("[compacted summary of 5 earlier turn(s)]"));
    }
}
