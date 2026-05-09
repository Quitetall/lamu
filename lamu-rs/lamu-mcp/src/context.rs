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

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const CENTRAL_DEFAULT: &str = include_str!("../assets/central_review_policy.md");

/// XDG override path: `~/.config/lamu/context/central.md`. When
/// present + ≤ CENTRAL_MAX_OVERRIDE_BYTES, replaces the bundled
/// default. Read once per process via OnceLock — leaked to keep
/// `&'static str` identity for byte-prefix cache stability.
const CENTRAL_MAX_OVERRIDE_BYTES: usize = 8 * 1024;

/// Plan tier hard cap (~8K tokens). Truncate-from-front keeps the
/// tail = newest plan content, since plan files grow downward.
pub(crate) const PLAN_MAX_BYTES: usize = 32 * 1024;

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
    // Read the XDG override once per process and leak the String to a
    // &'static so the prompt-cache prefix stays bit-identical across
    // every call. If the override is missing / unreadable / oversized,
    // fall through to the bundled default.
    static OVERRIDE: OnceLock<Option<&'static str>> = OnceLock::new();
    let override_text = OVERRIDE.get_or_init(|| {
        let path = dirs::config_dir()?.join("lamu").join("context").join("central.md");
        load_central_override_from(&path)
    });
    override_text.unwrap_or(CENTRAL_DEFAULT)
}

/// Read a candidate central-override file at `path`. Returns None when
/// missing, unreadable, or oversized. Factored out of `resolve_central`
/// so unit tests can exercise it without poisoning the OnceLock.
fn load_central_override_from(path: &Path) -> Option<&'static str> {
    let body = std::fs::read_to_string(path).ok()?;
    // Empty / whitespace-only override → fall through to bundled
    // default. A user who `touch`-es the file shouldn't accidentally
    // suppress the entire central tier.
    if body.trim().is_empty() {
        tracing::debug!(
            "context: central override at {} is empty; using bundled default",
            path.display()
        );
        return None;
    }
    if body.len() > CENTRAL_MAX_OVERRIDE_BYTES {
        tracing::warn!(
            "context: central override at {} is {} bytes (> {} limit); using bundled default",
            path.display(),
            body.len(),
            CENTRAL_MAX_OVERRIDE_BYTES
        );
        return None;
    }
    tracing::debug!(
        "context: loaded central override from {} ({} bytes)",
        path.display(),
        body.len()
    );
    // Leak so the &'static lifetime is honored. Read-once-per-process,
    // so a single allocation persists for the lifetime of the binary —
    // not a real leak, just a static-lifetime conversion.
    Some(Box::leak(body.into_boxed_str()))
}

/// Resolve the plan tier source path in priority order:
/// 1. Caller-supplied `cfg.plan` arg.
/// 2. `LAMU_PLAN` env var.
/// 3. Repo-local `<repo>/.claude/plans/active.md`.
/// 4. Home `~/.claude/plans/active.md`.
///
/// Returns (path, source) when something resolved, None otherwise.
fn resolve_plan_path(arg: Option<&str>, repo: Option<&Path>) -> Option<(PathBuf, PlanSource)> {
    if let Some(p) = arg {
        if !p.is_empty() {
            return Some((PathBuf::from(p), PlanSource::Arg));
        }
    }
    if let Ok(p) = std::env::var("LAMU_PLAN") {
        if !p.is_empty() {
            return Some((PathBuf::from(p), PlanSource::EnvVar));
        }
    }
    if let Some(r) = repo {
        let candidate = r.join(".claude").join("plans").join("active.md");
        if candidate.is_file() {
            return Some((candidate, PlanSource::RepoLocal));
        }
    }
    if let Some(home) = dirs::home_dir() {
        let candidate = home.join(".claude").join("plans").join("active.md");
        if candidate.is_file() {
            return Some((candidate, PlanSource::HomeDir));
        }
    }
    None
}

/// Recent-activity header for the plan tier. Last 50 commits, oneline.
/// Best-effort: empty string on any git failure. Cached per-process via
/// OnceLock keyed on repo path string so the cache prefix stays stable
/// within a session.
fn recent_activity_header(repo: &Path) -> String {
    use std::collections::HashMap;
    static CACHE: OnceLock<parking_lot::Mutex<HashMap<String, String>>> = OnceLock::new();
    let map = CACHE.get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
    let key = repo.to_string_lossy().to_string();
    {
        let m = map.lock();
        if let Some(cached) = m.get(&key) {
            return cached.clone();
        }
    }
    let out = std::process::Command::new("git")
        .current_dir(repo)
        .args(["log", "--oneline", "-50"])
        .output();
    let body = match out {
        Ok(o) if o.status.success() => {
            let log = String::from_utf8_lossy(&o.stdout);
            format!(
                "## Recent activity (last 50 commits)\n\nUse this to ground the plan against what's already shipped — don't second-guess items that match a recent commit.\n\n```\n{}```",
                log
            )
        }
        _ => String::new(),
    };
    map.lock().insert(key, body.clone());
    body
}

/// V5 improvement B: strip plan headings whose `Phase X.Y` /
/// `Step N` / `Improvement Z` token also appears in the recent
/// activity log. The reviewer keeps the plan's strategic context
/// without the per-phase TODO list re-raising items already shipped.
///
/// Conservative: only strips when a heading contains a phase token
/// AND the same token appears in the activity. Keeps headings that
/// don't carry recognized tokens. False-positive on this stripper
/// just means the plan stays intact (no harm); false-negative means
/// stale items survive (no worse than V4).
fn strip_shipped_phases(plan_body: &str, activity: &str) -> String {
    if activity.is_empty() {
        return plan_body.to_string();
    }
    use std::collections::HashSet;
    let phase_re_chars: &[char] = &['.', ':'];

    // Tokens we recognize in activity log: "Phase X.Y", "Step N",
    // "Improvement Z". Walk activity once, build a set.
    let mut shipped: HashSet<String> = HashSet::new();
    for line in activity.lines() {
        for word in line.split_whitespace() {
            let word = word.trim_matches(phase_re_chars);
            if let Some(rest) = word.strip_prefix("Phase") {
                if !rest.is_empty() && rest.chars().next().map_or(false, |c| c.is_ascii_digit() || c.is_whitespace()) {
                    // Multi-token form: "Phase 6.3" — peek next word in activity
                    // is too involved; skip. Catch single-token "Phase6.3" if
                    // ever used. For multi-token we look for "Phase" anchor +
                    // next word, handled below.
                }
            }
        }
        // Multi-token capture: walk word pairs, find "Phase X.Y" /
        // "Step N" / "Improvement Z".
        let words: Vec<&str> = line.split_whitespace().collect();
        for i in 0..words.len().saturating_sub(1) {
            let anchor = words[i];
            let next = words[i + 1].trim_matches(phase_re_chars).trim_end_matches(',');
            if (anchor == "Phase" || anchor == "Step" || anchor == "Improvement")
                && !next.is_empty()
            {
                shipped.insert(format!("{} {}", anchor, next));
            }
        }
    }

    if shipped.is_empty() {
        return plan_body.to_string();
    }

    // Walk plan body line-by-line. When a heading line ("##" / "###")
    // contains a shipped token, mark a "strip until next heading at
    // <= same depth" window. Append a one-line "(shipped: …)" stub
    // so reviewer sees the heading was elided + which commit context.
    let mut out = String::with_capacity(plan_body.len());
    let mut strip_depth: Option<usize> = None;
    for line in plan_body.lines() {
        let depth = line.chars().take_while(|c| *c == '#').count();
        let is_heading = depth > 0 && line.chars().nth(depth).map_or(false, |c| c == ' ');
        if is_heading {
            if let Some(d) = strip_depth {
                if depth <= d {
                    strip_depth = None;
                }
            }
            if strip_depth.is_none() {
                let shipped_match = shipped.iter().find(|tok| line.contains(tok.as_str()));
                if let Some(tok) = shipped_match {
                    out.push_str(&format!(
                        "{} _(shipped — {} appears in recent activity; details elided)_\n",
                        line, tok
                    ));
                    strip_depth = Some(depth);
                    continue;
                }
            }
        }
        if strip_depth.is_none() {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Read the plan file with a max-size guard. Truncates from the front
/// (drops oldest content) so the tail — typically the active sprint
/// section — stays in the prompt.
fn load_plan(path: &Path) -> Option<(String, bool)> {
    let body = std::fs::read_to_string(path).ok()?;
    if body.len() <= PLAN_MAX_BYTES {
        return Some((body, false));
    }
    // Truncate-from-front: keep the last PLAN_MAX_BYTES, snap back to
    // a UTF-8 char boundary so we never split a multi-byte codepoint.
    let mut start = body.len() - PLAN_MAX_BYTES;
    while start < body.len() && !body.is_char_boundary(start) {
        start += 1;
    }
    let mut out = String::with_capacity(PLAN_MAX_BYTES + 64);
    out.push_str(&format!(
        "[…truncated — plan exceeded {} bytes; showing tail]\n\n",
        PLAN_MAX_BYTES
    ));
    out.push_str(&body[start..]);
    Some((out, true))
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

    let plan_text = if let Some((path, source)) = resolve_plan_path(cfg.plan, cfg.repo) {
        match load_plan(&path) {
            Some((body, truncated)) => {
                stats.plan_source = source;
                stats.plan_truncated = truncated;
                // V4 staleness fix: prepend recent activity from the
                // repo so the reviewer knows what's already shipped.
                // Plans rot fast — adding a "what's done" header
                // keeps reviewer from second-guessing plan items
                // that landed weeks ago.
                if let Some(repo) = cfg.repo {
                    let header = recent_activity_header(repo);
                    if !header.is_empty() {
                        let stripped = strip_shipped_phases(&body, &header);
                        format!("{}\n\n---\n\n{}", header, stripped)
                    } else {
                        body
                    }
                } else {
                    body
                }
            }
            None => {
                tracing::debug!("context: plan file at {} unreadable", path.display());
                String::new()
            }
        }
    } else {
        String::new()
    };
    if !plan_text.is_empty() {
        stats.plan_bytes = plan_text.len();
        parts.push(plan_text.as_str());
    }

    if !cfg.tactical.is_empty() {
        parts.push(cfg.tactical);
        stats.tactical_bytes = cfg.tactical.len();
    }

    // Need to outlive `parts` references — keep plan_text alive until
    // the join completes.
    let prefix = parts.join(TIER_SEP);
    drop(plan_text);
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
        with_clean_plan_env(|| {
            let (s, stats) = assemble(ContextConfig {
                central: true,
                ..Default::default()
            });
            assert_eq!(stats.central_bytes, CENTRAL_DEFAULT.len());
            assert_eq!(stats.plan_bytes, 0);
            assert_eq!(stats.tactical_bytes, 0);
            assert!(s.starts_with(CENTRAL_DEFAULT));
        });
    }

    #[test]
    fn assemble_central_off_returns_empty() {
        with_clean_plan_env(|| {
            let (s, stats) = assemble(ContextConfig::default());
            assert!(s.is_empty());
            assert_eq!(stats.central_bytes, 0);
        });
    }

    #[test]
    fn assemble_central_plus_tactical() {
        with_clean_plan_env(|| {
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
        });
    }

    #[test]
    fn prepend_to_system_skips_when_empty() {
        with_clean_plan_env(|| {
            let (s, _) = prepend_to_system(ContextConfig::default(), "ROLE_PROMPT");
            assert_eq!(s, "ROLE_PROMPT");
        });
    }

    #[test]
    fn prepend_to_system_appends_role_after_separator() {
        with_clean_plan_env(|| {
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
        });
    }

    // Plan tier tests. Env reads/writes serialized via PLAN_ENV_LOCK
    // since LAMU_PLAN is process-global; tempfiles+tempdirs scope each
    // test's filesystem state.
    use std::sync::Mutex;
    static PLAN_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_clean_plan_env<F: FnOnce() -> R, R>(f: F) -> R {
        let _g = PLAN_ENV_LOCK.lock().unwrap();
        // SAFETY: PLAN_ENV_LOCK serializes test access. No other thread
        // reads LAMU_PLAN within the lamu-mcp test binary.
        unsafe {
            std::env::remove_var("LAMU_PLAN");
        }
        f()
    }

    #[test]
    fn plan_arg_overrides_env() {
        with_clean_plan_env(|| {
            let arg_dir = tempfile::tempdir().unwrap();
            let arg_path = arg_dir.path().join("arg.md");
            std::fs::write(&arg_path, "ARG_PLAN_BODY").unwrap();

            let env_dir = tempfile::tempdir().unwrap();
            let env_path = env_dir.path().join("env.md");
            std::fs::write(&env_path, "ENV_PLAN_BODY").unwrap();
            unsafe {
                std::env::set_var("LAMU_PLAN", env_path.to_str().unwrap());
            }

            let (s, stats) = assemble(ContextConfig {
                central: false, // isolate plan tier
                plan: arg_path.to_str(),
                ..Default::default()
            });
            unsafe { std::env::remove_var("LAMU_PLAN"); }
            assert_eq!(stats.plan_source, PlanSource::Arg);
            assert!(s.contains("ARG_PLAN_BODY"));
            assert!(!s.contains("ENV_PLAN_BODY"));
        });
    }

    #[test]
    fn plan_env_used_when_no_arg() {
        with_clean_plan_env(|| {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("env.md");
            std::fs::write(&path, "ENV_ONLY_PLAN").unwrap();
            unsafe {
                std::env::set_var("LAMU_PLAN", path.to_str().unwrap());
            }

            let (s, stats) = assemble(ContextConfig {
                central: false,
                ..Default::default()
            });
            unsafe { std::env::remove_var("LAMU_PLAN"); }
            assert_eq!(stats.plan_source, PlanSource::EnvVar);
            assert!(s.contains("ENV_ONLY_PLAN"));
        });
    }

    #[test]
    fn plan_repo_local_beats_home() {
        with_clean_plan_env(|| {
            let repo = tempfile::tempdir().unwrap();
            let plans_dir = repo.path().join(".claude").join("plans");
            std::fs::create_dir_all(&plans_dir).unwrap();
            std::fs::write(plans_dir.join("active.md"), "REPO_LOCAL_PLAN").unwrap();

            let (s, stats) = assemble(ContextConfig {
                central: false,
                repo: Some(repo.path()),
                ..Default::default()
            });
            assert_eq!(stats.plan_source, PlanSource::RepoLocal);
            assert!(s.contains("REPO_LOCAL_PLAN"));
        });
    }

    #[test]
    fn plan_missing_returns_empty() {
        with_clean_plan_env(|| {
            // No arg, no env, no repo path → no plan tier engages.
            let (s, stats) = assemble(ContextConfig {
                central: false,
                ..Default::default()
            });
            assert!(s.is_empty());
            assert_eq!(stats.plan_source, PlanSource::None);
            assert_eq!(stats.plan_bytes, 0);
        });
    }

    #[test]
    fn strip_shipped_phases_drops_matching_heading_block() {
        let plan = "## Plan\n\n### Phase 1: setup\nSetup details.\n\n### Phase 2.3: feature X\nDetails about phase 2.3.\n\n### Phase 9: future\nNot yet shipped.\n";
        let activity = "abc1234 Phase 2.3: shipped feature X\ndef5678 unrelated work\n";
        let out = strip_shipped_phases(plan, activity);
        assert!(out.contains("Phase 1: setup"));
        assert!(out.contains("Phase 9: future"));
        assert!(out.contains("shipped"));
        assert!(out.contains("Phase 2.3"));
        assert!(!out.contains("Details about phase 2.3"));
    }

    #[test]
    fn strip_shipped_phases_unchanged_when_activity_empty() {
        let plan = "## Phase 1\nbody\n";
        assert_eq!(strip_shipped_phases(plan, ""), plan);
    }

    #[test]
    fn strip_shipped_phases_preserves_non_phase_headings() {
        let plan = "## Architecture\n\nbody about arch.\n\n## Phase 7\nshipped body.\n";
        let activity = "abc Phase 7: done\n";
        let out = strip_shipped_phases(plan, activity);
        assert!(out.contains("Architecture"));
        assert!(out.contains("body about arch"));
        assert!(!out.contains("shipped body"));
    }

    #[test]
    fn central_override_loads_when_present_and_within_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("central.md");
        std::fs::write(&path, "OVERRIDE_BODY_MARK").unwrap();
        let s = super::load_central_override_from(&path).expect("override should load");
        assert_eq!(s, "OVERRIDE_BODY_MARK");
    }

    #[test]
    fn central_override_skipped_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("definitely-not-here.md");
        assert!(super::load_central_override_from(&path).is_none());
    }

    #[test]
    fn central_override_skipped_when_empty_or_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("empty.md");
        std::fs::write(&empty, "").unwrap();
        assert!(super::load_central_override_from(&empty).is_none());

        let ws = dir.path().join("whitespace.md");
        std::fs::write(&ws, "   \n\t  \n").unwrap();
        assert!(super::load_central_override_from(&ws).is_none());
    }

    #[test]
    fn central_override_rejected_when_oversized() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.md");
        let body: String = "X".repeat(CENTRAL_MAX_OVERRIDE_BYTES + 1);
        std::fs::write(&path, &body).unwrap();
        assert!(super::load_central_override_from(&path).is_none());
    }

    #[test]
    fn plan_truncated_when_oversized() {
        with_clean_plan_env(|| {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("big.md");
            // 50 KiB > PLAN_MAX_BYTES (32 KiB).
            let body: String = "X".repeat(50 * 1024) + "TAIL_MARKER_FOR_TEST";
            std::fs::write(&path, &body).unwrap();
            unsafe {
                std::env::set_var("LAMU_PLAN", path.to_str().unwrap());
            }

            let (s, stats) = assemble(ContextConfig {
                central: false,
                ..Default::default()
            });
            unsafe { std::env::remove_var("LAMU_PLAN"); }
            assert!(stats.plan_truncated);
            // Truncate-from-front keeps the tail.
            assert!(s.contains("TAIL_MARKER_FOR_TEST"));
            assert!(s.contains("truncated"));
            assert!(stats.plan_bytes <= PLAN_MAX_BYTES + 256);
        });
    }
}
