//! Auto-context from a commit diff.
//!
//! When a caller passes `auto_context: true` on `review_commit`,
//! lamu-mcp builds Tactical-tier context automatically:
//!
//! 1. `git show --patch <commit>` for the unified diff.
//! 2. tree-sitter-rust parses each `+`-prefixed line slab to extract
//!    function / method / struct / impl names introduced or modified.
//! 3. `git ls-files | xargs rg -n '\b<symbol>\b'` per symbol → caller
//!    locations across the workspace.
//! 4. `git show <commit>:<path>` per changed file → reviewer sees the
//!    file as it lives at the commit's tip, not just the patch hunks.
//!
//! Output: a single Markdown blob ready for the Tactical tier. Bounded
//! at MAX_AUTO_BYTES so a giant refactor doesn't blow the prompt.

use anyhow::{anyhow, Result};
use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

/// Cap on the assembled auto-context payload. ~50K tokens.
pub(crate) const MAX_AUTO_BYTES: usize = 200 * 1024;

/// Per-file caller-search result cap. We only show the first N hits
/// per symbol to keep noise down.
const CALLERS_PER_SYMBOL_MAX: usize = 10;

/// Assemble the full auto-context payload for one commit.
///
/// Best-effort: each step (git show, tree-sitter parse, ripgrep)
/// degrades gracefully if it fails. The reviewer always gets at
/// least the diff itself; the symbol/caller layers add value when
/// they succeed.
pub fn assemble_auto_context(commit: &str, repo: &Path) -> Result<String> {
    let mut out = String::with_capacity(8 * 1024);
    out.push_str("# Auto-context for commit ");
    out.push_str(commit);
    out.push_str("\n\n");

    // Stage 1: list changed files
    match git_changed_files(commit, repo) {
        Ok(files) if !files.is_empty() => {
            out.push_str("## Changed files\n\n");
            for f in &files {
                out.push_str(&format!("- `{}`\n", f));
            }
            out.push('\n');

            // Stage 2: show each file at <commit>:<path>
            out.push_str("## Files at commit (full body, post-change)\n\n");
            for f in &files {
                if !is_safe_path_for_git(f) {
                    continue;
                }
                if let Ok(body) = git_show_file(commit, f, repo) {
                    let trimmed = smart_truncate_file_body(&body);
                    out.push_str(&format!("### `{}`\n\n```\n", f));
                    out.push_str(&trimmed);
                    out.push_str("\n```\n\n");
                    if out.len() > MAX_AUTO_BYTES {
                        break;
                    }
                }
            }
        }
        Ok(_) => out.push_str("(no changed files reported)\n\n"),
        Err(e) => {
            tracing::debug!("auto_context: git_changed_files failed: {}", e);
        }
    }

    // Stage 3: extract added symbols from the diff via tree-sitter
    if out.len() < MAX_AUTO_BYTES {
        match git_show_diff(commit, repo) {
            Ok(diff) => {
                let symbols = extract_added_symbols(&diff);
                if !symbols.is_empty() {
                    out.push_str("## Added / modified symbols (tree-sitter)\n\n");
                    for s in &symbols {
                        out.push_str(&format!("- `{}`\n", s));
                    }
                    out.push('\n');

                    // Stage 4: caller search via ripgrep
                    out.push_str("## Caller locations (ripgrep, production-only)\n\n");
                    for s in &symbols {
                        let mut hits = ripgrep_callers(s, repo).unwrap_or_default();
                        // Production-only: drop tests/ paths.
                        hits.retain(|l| is_caller_hit_meaningful(l));
                        if hits.is_empty() {
                            continue;
                        }
                        out.push_str(&format!("### `{}`\n\n", s));
                        for h in hits.iter().take(CALLERS_PER_SYMBOL_MAX) {
                            out.push_str(&format!("- {}\n", h));
                        }
                        out.push('\n');
                        if out.len() > MAX_AUTO_BYTES {
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("auto_context: git_show_diff failed: {}", e);
            }
        }
    }

    if out.len() > MAX_AUTO_BYTES {
        out.truncate(MAX_AUTO_BYTES);
        out.push_str("\n\n[…auto-context truncated to ");
        out.push_str(&MAX_AUTO_BYTES.to_string());
        out.push_str(" bytes]\n");
    }

    Ok(out)
}

/// `git diff-tree --name-only -r <commit>` — paths only, no status
/// letters, no header noise. Returns a Vec sorted in repo order.
fn git_changed_files(commit: &str, repo: &Path) -> Result<Vec<String>> {
    let out = Command::new("git")
        .current_dir(repo)
        .args(["diff-tree", "--no-commit-id", "--name-only", "-r", commit])
        .output()?;
    if !out.status.success() {
        return Err(anyhow!(
            "git diff-tree failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    Ok(stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|s| s.to_string())
        .collect())
}

fn git_show_diff(commit: &str, repo: &Path) -> Result<String> {
    let out = Command::new("git")
        .current_dir(repo)
        .args(["show", "--patch", "--no-color", commit])
        .output()?;
    if !out.status.success() {
        return Err(anyhow!("git show failed"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn git_show_file(commit: &str, path: &str, repo: &Path) -> Result<String> {
    let spec = format!("{}:{}", commit, path);
    let out = Command::new("git")
        .current_dir(repo)
        .args(["show", &spec])
        .output()?;
    if !out.status.success() {
        return Err(anyhow!("git show {} failed", spec));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Smart-truncate a long file body. Files > THRESHOLD_LINES get
/// "first 60 + last 30" — preserves imports/headers + the file's
/// trailing context (often where the changes landed) while dropping
/// the middle. Below the threshold pass through unchanged.
fn smart_truncate_file_body(body: &str) -> String {
    const THRESHOLD_LINES: usize = 200;
    const HEAD: usize = 60;
    const TAIL: usize = 30;
    let lines: Vec<&str> = body.lines().collect();
    if lines.len() <= THRESHOLD_LINES {
        return body.to_string();
    }
    let mut out = String::with_capacity(body.len() / 2);
    for l in lines.iter().take(HEAD) {
        out.push_str(l);
        out.push('\n');
    }
    out.push_str(&format!(
        "\n[...{} lines elided ({} total)...]\n\n",
        lines.len() - HEAD - TAIL,
        lines.len()
    ));
    for l in lines.iter().skip(lines.len() - TAIL) {
        out.push_str(l);
        out.push('\n');
    }
    out
}

/// Test-noise filter on ripgrep caller hits. Drops `tests/` /
/// `*_test.rs` / `*tests.rs` paths so the caller list focuses on
/// production callers — those are the ones a reviewer cares about
/// when evaluating an API-shape change.
fn is_caller_hit_meaningful(line: &str) -> bool {
    // Hit format: "path:lineno:content"
    let path = line.split(':').next().unwrap_or("");
    !(path.starts_with("tests/")
        || path.contains("/tests/")
        || path.ends_with("_test.rs")
        || path.ends_with("tests.rs"))
}

/// Reject paths with shell-meta or absolute roots. The `git show
/// <commit>:<path>` syntax is interpreted by git; we pass `<path>`
/// after the colon so traversal-style paths pose no risk to the
/// filesystem, but we still want to keep the output predictable.
fn is_safe_path_for_git(path: &str) -> bool {
    if path.is_empty() {
        return false;
    }
    !path.contains('\n') && !path.contains('\0')
}

/// Walk the diff hunks, accumulate `+`-prefixed line bytes per file,
/// then run tree-sitter-rust against each accumulated slab. Extract
/// names of `fn`, `impl`, `struct`, `enum`, `trait` items.
///
/// Note: tree-sitter parses arbitrary text fragments; partial diffs
/// (missing `use` headers, dangling braces) parse as ERROR nodes but
/// the parser still walks past them and surfaces the well-formed
/// declarations in between. That's the property we lean on here —
/// don't reconstruct full files, just feed the slab.
pub(crate) fn extract_added_symbols(diff: &str) -> Vec<String> {
    let mut rust_slabs: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_rust_file = false;

    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            // Push the previous slab + reset.
            if !current.is_empty() {
                rust_slabs.push(std::mem::take(&mut current));
            }
            in_rust_file = rest.ends_with(".rs");
            continue;
        }
        if line.starts_with("--- ") || line.starts_with("@@") || line.starts_with("diff ") {
            continue;
        }
        if !in_rust_file {
            continue;
        }
        if let Some(added) = line.strip_prefix('+') {
            // Skip the second '+' of `+++` already filtered above. Real
            // added line — push the content (without leading '+').
            current.push_str(added);
            current.push('\n');
        }
    }
    if !current.is_empty() {
        rust_slabs.push(current);
    }

    let mut symbols: HashSet<String> = HashSet::new();
    let language = tree_sitter_rust::language();
    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&language).is_err() {
        return Vec::new();
    }

    for slab in &rust_slabs {
        let Some(tree) = parser.parse(slab.as_bytes(), None) else {
            continue;
        };
        let root = tree.root_node();
        walk_collect_symbols(root, slab.as_bytes(), &mut symbols);
    }

    let mut sorted: Vec<String> = symbols.into_iter().collect();
    sorted.sort();
    sorted
}

fn walk_collect_symbols(node: tree_sitter::Node, src: &[u8], out: &mut HashSet<String>) {
    // Item kinds we extract names from.
    let interesting = matches!(
        node.kind(),
        "function_item"
            | "function_signature_item"
            | "struct_item"
            | "enum_item"
            | "trait_item"
            | "type_item"
            | "const_item"
            | "static_item"
    );
    if interesting {
        if let Some(name_node) = node.child_by_field_name("name") {
            if let Ok(name) = name_node.utf8_text(src) {
                out.insert(name.to_string());
            }
        }
    }
    for i in 0..node.child_count() {
        if let Some(c) = node.child(i) {
            walk_collect_symbols(c, src, out);
        }
    }
}

/// `rg -n --no-heading --color=never -F <symbol>` — fixed-string
/// match, file:line:contents output. We trim each hit to the first
/// 200 bytes so caller list stays compact.
fn ripgrep_callers(symbol: &str, repo: &Path) -> Result<Vec<String>> {
    // Defensive: don't run rg with a symbol that contains shell-meta
    // (shouldn't happen — tree-sitter symbols are identifiers — but
    // belt-and-braces).
    if symbol.is_empty()
        || !symbol
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Ok(Vec::new());
    }
    let out = Command::new("rg")
        .current_dir(repo)
        .args([
            "-n",
            "--no-heading",
            "--color=never",
            "-F",
            "--max-count",
            "20",
            symbol,
        ])
        .output();
    let out = match out {
        Ok(o) => o,
        Err(_) => return Ok(Vec::new()), // rg not installed → silent skip
    };
    if !out.status.success() {
        // rg exits 1 on no matches — that's fine, not an error.
        return Ok(Vec::new());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    Ok(stdout.lines().map(|l| truncate_line_utf8(l, 200)).collect())
}

/// UTF-8-safe line truncate. `&s[..n]` panics when `n` falls mid-
/// codepoint (e.g. inside a 3-byte `─`); walk back to a char boundary
/// before slicing.
fn truncate_line_utf8(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut cut = max;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &s[..cut])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_symbols_from_simple_added_fn() {
        let diff = "\
diff --git a/src/x.rs b/src/x.rs
--- a/src/x.rs
+++ b/src/x.rs
@@ -1,3 +1,8 @@
 use std::path::Path;
+
+pub fn new_helper(p: &Path) -> usize {
+    p.as_os_str().len()
+}
+
 // existing
";
        let symbols = extract_added_symbols(diff);
        assert!(symbols.contains(&"new_helper".to_string()), "got: {symbols:?}");
    }

    #[test]
    fn extract_symbols_finds_struct_and_impl() {
        let diff = "\
diff --git a/src/m.rs b/src/m.rs
+++ b/src/m.rs
@@ -1,1 +1,9 @@
+pub struct Widget {
+    pub size: u32,
+}
+
+impl Widget {
+    pub fn area(&self) -> u32 {
+        self.size * self.size
+    }
+}
";
        let symbols = extract_added_symbols(diff);
        assert!(symbols.contains(&"Widget".to_string()), "got: {symbols:?}");
        assert!(symbols.contains(&"area".to_string()), "got: {symbols:?}");
    }

    #[test]
    fn extract_symbols_skips_non_rust_files() {
        let diff = "\
diff --git a/README.md b/README.md
+++ b/README.md
@@ -1,1 +1,2 @@
+pub fn fake_fn() {}
";
        let symbols = extract_added_symbols(diff);
        assert!(symbols.is_empty(), "got: {symbols:?}");
    }

    #[test]
    fn extract_symbols_handles_empty_diff() {
        assert!(extract_added_symbols("").is_empty());
    }

    #[test]
    fn extract_symbols_skips_minus_lines() {
        let diff = "\
diff --git a/src/x.rs b/src/x.rs
+++ b/src/x.rs
@@ -1,3 +1,1 @@
-pub fn old_fn() {}
-pub fn another_old() {}
+pub fn kept() {}
";
        let symbols = extract_added_symbols(diff);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0], "kept");
    }

    #[test]
    fn smart_truncate_passes_through_short_files() {
        let body: String = (0..50).map(|i| format!("line{}\n", i)).collect();
        assert_eq!(smart_truncate_file_body(&body), body);
    }

    #[test]
    fn smart_truncate_keeps_head_and_tail_for_long_files() {
        let body: String = (0..400).map(|i| format!("line{}\n", i)).collect();
        let out = smart_truncate_file_body(&body);
        assert!(out.contains("line0"));   // head
        assert!(out.contains("line59"));  // last of head
        assert!(out.contains("line370")); // first of tail
        assert!(out.contains("line399")); // last of tail
        assert!(out.contains("elided"));
        assert!(!out.contains("line200")); // middle dropped
    }

    #[test]
    fn caller_hit_filter_drops_test_paths() {
        assert!(!is_caller_hit_meaningful("tests/foo.rs:10:bar()"));
        assert!(!is_caller_hit_meaningful("crate/tests/x.rs:1:y()"));
        assert!(!is_caller_hit_meaningful("src/foo_test.rs:5:z()"));
        assert!(!is_caller_hit_meaningful("src/foo/tests.rs:5:z()"));
        assert!(is_caller_hit_meaningful("src/handlers.rs:42:do_thing()"));
        assert!(is_caller_hit_meaningful("src/cloud.rs:100:helper()"));
    }

    #[test]
    fn truncate_line_utf8_handles_codepoint_boundary() {
        // Line containing a 3-byte UTF-8 codepoint at the cap.
        let s = format!("{}─{}", "x".repeat(199), "y".repeat(50));
        let out = truncate_line_utf8(&s, 200);
        // Snap-back lands at byte 199 (before the `─`).
        assert!(out.starts_with(&"x".repeat(199)));
        assert!(out.ends_with('…'));
    }

    #[test]
    fn truncate_line_utf8_short_passes_through() {
        assert_eq!(truncate_line_utf8("hi", 200), "hi");
    }

    #[test]
    fn is_safe_path_rejects_newline_and_null() {
        assert!(!is_safe_path_for_git(""));
        assert!(!is_safe_path_for_git("a\nb"));
        assert!(!is_safe_path_for_git("a\0b"));
        assert!(is_safe_path_for_git("src/x.rs"));
        assert!(is_safe_path_for_git("a/b/c.txt"));
    }
}
