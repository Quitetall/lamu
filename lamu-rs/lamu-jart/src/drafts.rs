//! Persist research drafts. Every `deep_research` / `answer` result is
//! auto-saved as a markdown file (report + numbered sources + metadata) under
//! the research dir, so no draft is ever lost — the user's drafts all flow
//! through here.
//!
//! Default dir: `<data_dir>/lamu/research` (`~/.local/share/lamu/research`);
//! `LAMU_RESEARCH_DIR` overrides. Best-effort: a save failure never fails the
//! tool (the JSON result is still returned). Disable with `LAMU_NO_SAVE_DRAFTS`.

use serde_json::Value;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Where drafts are written. `LAMU_RESEARCH_DIR` overrides the default.
fn research_dir() -> PathBuf {
    if let Ok(p) = std::env::var("LAMU_RESEARCH_DIR") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    dirs::data_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("lamu")
        .join("research")
}

/// Save a draft as markdown; returns the written path (display string).
/// `None` when saving is disabled (`LAMU_NO_SAVE_DRAFTS`) or on any IO error —
/// callers treat the path as optional so a save failure never breaks the tool.
/// `sources` items carry `idx`, `title`, and a `url` OR `link` field.
pub(crate) fn save_draft(
    kind: &str,
    query: &str,
    report: &str,
    sources: &[Value],
    meta: &[(&str, String)],
) -> Option<String> {
    if std::env::var_os("LAMU_NO_SAVE_DRAFTS").is_some() {
        return None;
    }
    let dir = research_dir();
    std::fs::create_dir_all(&dir).ok()?;

    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (stamp, date) = stamp_and_date(secs);
    let path = dir.join(format!("{stamp}-{}.md", slugify(query, 48)));

    let md = build_markdown(kind, query, report, sources, &date, meta);
    std::fs::write(&path, md).ok()?;
    Some(path.display().to_string())
}

fn build_markdown(
    kind: &str,
    query: &str,
    report: &str,
    sources: &[Value],
    date: &str,
    meta: &[(&str, String)],
) -> String {
    let mut s = String::new();
    s.push_str(&format!("# {}\n\n", query.trim()));
    if report.trim().is_empty() {
        s.push_str("_(no synthesis — sources only)_\n\n");
    } else {
        s.push_str(report.trim());
        s.push_str("\n\n");
    }
    if !sources.is_empty() {
        s.push_str("## Sources\n\n");
        for (i, src) in sources.iter().enumerate() {
            let idx = src["idx"].as_u64().unwrap_or((i + 1) as u64);
            let title = src["title"].as_str().unwrap_or("(untitled)");
            // deep_research corpus uses `link`; the answer tool uses `url`.
            let url = src["url"].as_str().or_else(|| src["link"].as_str()).unwrap_or("");
            // `<url>` = CommonMark autolink → renders clickable, and a `)`/`]`
            // in the URL can't break the line.
            if url.is_empty() {
                s.push_str(&format!("[{idx}] {title}\n"));
            } else {
                s.push_str(&format!("[{idx}] {title} — <{url}>\n"));
            }
        }
        s.push('\n');
    }
    s.push_str("---\n");
    s.push_str(&format!("- kind: {kind}\n- date: {date}\n"));
    for (k, v) in meta {
        s.push_str(&format!("- {k}: {v}\n"));
    }
    s
}

/// Lowercase ascii-alphanumeric slug, non-alnum → `-`, collapsed, trimmed, capped.
fn slugify(s: &str, max: usize) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
        if out.chars().count() >= max {
            break;
        }
    }
    let t = out.trim_matches('-').to_string();
    if t.is_empty() { "draft".to_string() } else { t }
}

/// (filename stamp `YYYYMMDD-HHMMSS`, human `YYYY-MM-DD HH:MM:SS UTC`) from unix
/// seconds. Hinnant civil-from-days — no date-lib dependency.
fn stamp_and_date(secs: u64) -> (String, String) {
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as i64; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (
        format!("{y:04}{m:02}{d:02}-{hh:02}{mm:02}{ss:02}"),
        format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}:{ss:02} UTC"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn slugify_basics() {
        assert_eq!(slugify("EEG signal compression!", 48), "eeg-signal-compression");
        assert_eq!(slugify("   ???   ", 48), "draft");
        assert!(slugify(&"x".repeat(100), 10).chars().count() <= 10);
    }

    #[test]
    fn stamp_known_epoch() {
        // 2021-09-01 00:00:00 UTC = 1630454400.
        let (stamp, date) = stamp_and_date(1630454400);
        assert_eq!(stamp, "20210901-000000");
        assert_eq!(date, "2021-09-01 00:00:00 UTC");
    }

    #[test]
    fn markdown_has_report_sources_meta() {
        let sources = vec![json!({"idx": 1, "title": "Paper A", "link": "https://a"})];
        let md = build_markdown(
            "deep_research",
            "EEG compression",
            "Answer [1].",
            &sources,
            "2026-06-09 00:00:00 UTC",
            &[("model", "tongyi".to_string())],
        );
        assert!(md.starts_with("# EEG compression"));
        assert!(md.contains("Answer [1]."));
        assert!(md.contains("[1] Paper A — <https://a>"));
        assert!(md.contains("- kind: deep_research"));
        assert!(md.contains("- model: tongyi"));
    }

    #[test]
    fn save_draft_disabled_returns_none() {
        // SAFETY: set/remove a process env var. Safe here because this is the
        // only test in the crate that touches LAMU_NO_SAVE_DRAFTS, so a parallel
        // test can't observe the transient set.
        unsafe { std::env::set_var("LAMU_NO_SAVE_DRAFTS", "1") };
        let out = save_draft("answer", "q", "r", &[], &[]);
        unsafe { std::env::remove_var("LAMU_NO_SAVE_DRAFTS") };
        assert!(out.is_none());
    }
}
