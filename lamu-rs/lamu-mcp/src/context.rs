//! 3-tier context-injection layer for MCP tools (cloud_query,
//! review_commit, review_diff, parallel_query).
//!
//! Step 4 ships the **Central** tier only: a bundled `&'static str`
//! review policy + V4 Pro false-positive list, prepended to the
//! reviewer's system prompt. Steps 5–7 add the Plan tier (file-driven,
//! opt-in) and Tactical tier (caller-supplied).
//!
//! ## Cache-friendly layout
//!
//! DeepSeek's prompt cache keys on contiguous bytes from the start of
//! the system prompt. We prepend the most-stable tier first so the
//! prefix stays bit-identical across calls:
//!
//! ```text
//! <central — &'static str, byte-stable forever>
//! ---
//! <plan — file-stable per sprint>     (step 5)
//! ---
//! <tactical — caller-supplied, varies>(step 6)
//! ---
//! <original system prompt>
//! ```

use std::path::Path;

const CENTRAL_DEFAULT: &str = include_str!("../assets/central_review_policy.md");

/// Separator between context tiers / role prompt. Plain `---` is
/// unambiguous across markdown + diffs and stays bit-stable so cache
/// hits aren't broken by formatting drift.
pub(crate) const TIER_SEP: &str = "\n\n---\n\n";

/// Caller-supplied configuration for one assemble() call.
#[derive(Debug, Default)]
pub struct ContextConfig<'a> {
    /// Always-on. Set false to opt out per-call (rare — reviewer
    /// callers should always engage central).
    pub central: bool,
    /// Caller-supplied plan path; None → auto-detect (step 5).
    pub plan: Option<&'a str>,
    /// Verbatim caller-supplied tactical context (step 6).
    pub tactical: &'a str,
    /// For repo-local plan auto-detect (step 5).
    pub repo: Option<&'a Path>,
}

/// Where the plan tier resolved from. None when no plan engaged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlanSource {
    #[default]
    None,
    Arg,
    EnvVar,
    RepoLocal,
    HomeDir,
    OverrideEmpty,
}

#[derive(Debug, Default)]
pub struct ContextStats {
    pub central_bytes: usize,
    pub plan_bytes: usize,
    pub tactical_bytes: usize,
    pub plan_source: PlanSource,
    pub plan_truncated: bool,
}

fn resolve_central() -> &'static str {
    // Step 7 will add an XDG override here. For now the bundled
    // default is the only source.
    CENTRAL_DEFAULT
}

/// Resolve all enabled tiers and concatenate them into a single prefix
/// suitable for prepending to a model's system prompt.
///
/// Returns (prefix, stats). Empty prefix when nothing engaged — the
/// caller's `format!()` should branch on `prefix.is_empty()` so
/// backward-compat is bit-identical to pre-Step-4.
pub fn assemble(cfg: ContextConfig) -> (String, ContextStats) {
    let mut stats = ContextStats::default();
    let mut parts: Vec<&str> = Vec::with_capacity(3);

    let central = if cfg.central { resolve_central() } else { "" };
    if !central.is_empty() {
        parts.push(central);
        stats.central_bytes = central.len();
    }

    // Plan tier (step 5) — placeholder for now; resolves to empty
    // until step 5 lands its source resolution.
    let plan_text: String = String::new();
    let _ = cfg.plan;
    let _ = cfg.repo;
    if !plan_text.is_empty() {
        parts.push(&plan_text);
        stats.plan_bytes = plan_text.len();
    }

    if !cfg.tactical.is_empty() {
        parts.push(cfg.tactical);
        stats.tactical_bytes = cfg.tactical.len();
    }

    let prefix = parts.join(TIER_SEP);
    (prefix, stats)
}

/// Convenience wrapper: returns just the prefix string, separator
/// included between prefix and the caller's system prompt only when
/// non-empty. Designed for the common pattern in cloud.rs:
///
/// ```ignore
/// let system = match prepend_to_system(cfg, REVIEW_SYSTEM_PROMPT) {
///     (s, _) => s,
/// };
/// ```
pub fn prepend_to_system(cfg: ContextConfig, role_prompt: &str) -> (String, ContextStats) {
    let (prefix, stats) = assemble(cfg);
    if prefix.is_empty() {
        return (role_prompt.to_string(), stats);
    }
    (format!("{}{}{}", prefix, TIER_SEP, role_prompt), stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn central_default_loaded_from_bundled_asset() {
        // include_str! at build time means the bundled asset must
        // contain the load-bearing FP marker text. If someone trims
        // the asset and breaks the contract, this test catches it.
        assert!(CENTRAL_DEFAULT.contains("Verify-before-flag"));
        assert!(CENTRAL_DEFAULT.contains("serde_json"));
        assert!(CENTRAL_DEFAULT.contains("bwrap"));
        assert!(CENTRAL_DEFAULT.contains("GGUF"));
        assert!(CENTRAL_DEFAULT.contains("env"));
    }

    #[test]
    fn assemble_central_only() {
        let (s, stats) = assemble(ContextConfig {
            central: true,
            ..Default::default()
        });
        assert_eq!(stats.central_bytes, CENTRAL_DEFAULT.len());
        assert_eq!(stats.plan_bytes, 0);
        assert_eq!(stats.tactical_bytes, 0);
        assert!(s.starts_with(CENTRAL_DEFAULT));
    }

    #[test]
    fn assemble_central_off_returns_empty() {
        let (s, stats) = assemble(ContextConfig::default());
        assert!(s.is_empty());
        assert_eq!(stats.central_bytes, 0);
    }

    #[test]
    fn assemble_central_plus_tactical() {
        let (s, stats) = assemble(ContextConfig {
            central: true,
            tactical: "TACTICAL_PROBE_BLOB",
            ..Default::default()
        });
        assert_eq!(stats.tactical_bytes, "TACTICAL_PROBE_BLOB".len());
        // Order: central, then separator, then tactical.
        assert!(s.contains(CENTRAL_DEFAULT));
        assert!(s.contains(TIER_SEP));
        assert!(s.ends_with("TACTICAL_PROBE_BLOB"));
    }

    #[test]
    fn prepend_to_system_skips_when_empty() {
        let (s, _) = prepend_to_system(ContextConfig::default(), "ROLE_PROMPT");
        assert_eq!(s, "ROLE_PROMPT");
    }

    #[test]
    fn prepend_to_system_appends_role_after_separator() {
        let (s, _) = prepend_to_system(
            ContextConfig {
                central: true,
                ..Default::default()
            },
            "ROLE_PROMPT",
        );
        assert!(s.starts_with(CENTRAL_DEFAULT));
        assert!(s.ends_with("ROLE_PROMPT"));
        // Separator must appear exactly between central and the role
        // prompt — count occurrences to confirm.
        assert_eq!(s.matches(TIER_SEP).count(), 1);
    }
}
