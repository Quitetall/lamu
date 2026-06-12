//! Conversation memory — MCP frontend shim.
//!
//! The storage (append-only per-`conversation_id` SQLite log at
//! `~/.local/share/lamu/conversations.db`) moved to
//! `lamu_memory::memory` (ADR 0029: memory is a shared capability used
//! by multiple frontends, so storage lives in a module crate that
//! depends only on external crates). Everything is re-exported so
//! existing `crate::memory::X` call sites compile unchanged.
//!
//! The one frontend-shaped piece kept here is [`render_for_context`]:
//! the untrusted-content fencing (ADR 0011) is a wire concern of THIS
//! frontend, so the wrap stays on this side of the seam, layered over
//! the pure `lamu_memory::memory::render_turns` body.

pub use lamu_memory::memory::*;

/// Render a recalled transcript as a Markdown string suitable for
/// dropping into the Tactical tier of the context layer.
pub fn render_for_context(turns: &[Turn]) -> String {
    let body = lamu_memory::memory::render_turns(turns);
    if body.is_empty() {
        return body;
    }
    // Prior turns are attacker-influenceable (a poisoned earlier message could
    // carry an injection); fence them as data so a downstream prompt won't act
    // on embedded instructions (ADR 0011).
    crate::untrusted::wrap_untrusted("prior conversation turns", body.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_for_context_empty_returns_empty_string() {
        assert!(render_for_context(&[]).is_empty());
    }

    #[test]
    fn render_for_context_includes_role_and_content() {
        let turns = vec![Turn {
            idx: 0,
            role: "user".into(),
            content: "hello".into(),
            ts: 0,
            metadata: None,
        }];
        let s = render_for_context(&turns);
        assert!(s.contains("user"));
        assert!(s.contains("hello"));
        // Recalled turns are fenced as untrusted data (ADR 0011).
        assert!(s.contains("<<<LAMU_UNTRUSTED"));
        assert!(s.contains("<<<END_LAMU_UNTRUSTED"));
    }
}
