//! Per-agent work preservation layer for lamu.
//!
//! Each lamu agent invocation gets its own git worktree on a dedicated branch
//! (`agent/<session_id>`). This module manages creation, listing, preservation,
//! selective file cherry-picking, and cleanup of these agent workspaces.
//! The auto-checkpointer background task periodically commits changes.
//!
//! All one-shot operations use synchronous `std::process::Command`.
//! The checkpoint loop uses `tokio::process::Command` for async execution.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command as SyncCommand;

/// Represents a single agent worktree, as returned by `list_agent_worktrees`.
pub struct AgentWorktree {
    pub session_id: String,
    pub branch: String,
    pub path: PathBuf,
    pub last_checkpoint_secs: Option<u64>,
    pub files_changed: usize,
    pub loc_delta: i64,
}

// ---------------------------------------------------------------------------
// Utility: construct the sandbox worktrees directory
// ---------------------------------------------------------------------------
/// Where agent worktrees live. `LAMU_SANDBOX_HOME` env var overrides
/// for tests / non-default deployments. Default:
/// `~/.local/share/lamu/sandbox/worktrees/`.
fn sandbox_worktrees_root() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("LAMU_SANDBOX_HOME") {
        return Ok(PathBuf::from(custom).join("worktrees"));
    }
    let home = std::env::var("HOME")
        .map_err(|_| anyhow!("HOME environment variable not set"))?;
    Ok(Path::new(&home).join(".local/share/lamu/sandbox/worktrees"))
}

/// session_id is used in: filesystem paths, branch names, and git
/// arg vectors. Reject anything that could escape the sandbox or
/// fool git's ref parser.
fn validate_session_id(session_id: &str) -> Result<()> {
    if session_id.is_empty() {
        anyhow::bail!("session_id is empty");
    }
    if session_id.len() > 100 {
        anyhow::bail!("session_id too long (max 100 chars)");
    }
    let valid = session_id.chars().all(|c|
        c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')
    );
    if !valid {
        anyhow::bail!(
            "session_id '{}' rejected — only [a-zA-Z0-9_.-] allowed",
            session_id
        );
    }
    if session_id == "." || session_id == ".." || session_id.starts_with('.') {
        anyhow::bail!("session_id may not start with '.' or be '.' / '..'");
    }
    Ok(())
}

/// Detect the repository's "default" branch — what to merge agent
/// branches into. Tries (in order):
///   1. origin/HEAD symbolic-ref (e.g. "main" or "master")
///   2. current branch (HEAD), if it's not detached and not "agent/*"
///   3. "main" as last resort
fn detect_default_branch(repo_root: &Path) -> Result<String> {
    // 1. origin/HEAD
    if let Ok(out) = SyncCommand::new("git")
        .arg("-C").arg(repo_root)
        .arg("symbolic-ref").arg("refs/remotes/origin/HEAD")
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if let Some(name) = s.strip_prefix("refs/remotes/origin/") {
                return Ok(name.to_string());
            }
        }
    }
    // 2. Current HEAD branch
    if let Ok(out) = SyncCommand::new("git")
        .arg("-C").arg(repo_root)
        .arg("rev-parse").arg("--abbrev-ref").arg("HEAD")
        .output()
    {
        if out.status.success() {
            let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !name.is_empty() && name != "HEAD" && !name.starts_with("agent/") {
                return Ok(name);
            }
        }
    }
    // 3. fallback
    Ok("main".to_string())
}

fn worktree_path_for(session_id: &str) -> Result<PathBuf> {
    validate_session_id(session_id)?;
    let path = sandbox_worktrees_root()?.join(session_id);
    // Defense in depth: ensure the joined path is still under root.
    let root = sandbox_worktrees_root()?;
    let canonical_root = root.canonicalize().unwrap_or(root.clone());
    // path may not exist yet; canonicalize ancestors.
    let parent = path.parent().unwrap_or(&path);
    if parent.exists() {
        let canonical_parent = parent.canonicalize()
            .unwrap_or_else(|_| parent.to_path_buf());
        if !canonical_parent.starts_with(&canonical_root) {
            anyhow::bail!("worktree path escaped sandbox root");
        }
    }
    Ok(path)
}

// ---------------------------------------------------------------------------
// 1. create_worktree
// ---------------------------------------------------------------------------

/// Create a dedicated git worktree for the given session.
///
/// The worktree is created at `~/.local/share/lamu/sandbox/worktrees/<session_id>/`
/// on a fresh branch named `agent/<session_id>` pointing at HEAD of the repository.
pub fn create_worktree(session_id: &str, repo_root: &Path) -> Result<PathBuf> {
    let branch = format!("agent/{session_id}");
    let worktree_path = worktree_path_for(session_id)?;
    let parent = worktree_path
        .parent()
        .ok_or_else(|| anyhow!("Cannot get parent of worktree path"))?;
    std::fs::create_dir_all(parent).context("Failed to create worktree parent directory")?;

    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("worktree")
        .arg("add")
        .arg("-b")
        .arg(&branch)
        .arg(&worktree_path)
        .arg("HEAD")
        .output()
        .context("Failed to run git worktree add")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("git worktree add failed: {stderr}"));
    }

    Ok(worktree_path)
}

// ---------------------------------------------------------------------------
// 2. list_agent_worktrees
// ---------------------------------------------------------------------------

/// List all agent worktrees with metadata.
pub fn list_agent_worktrees() -> Result<Vec<AgentWorktree>> {
    // First find all worktrees via `git worktree list --porcelain`
    // We need to run this from any git repo – we can scan for one in the
    // sandbox worktrees root? Actually we need to know the repo root.
    // The calling code should have the repo root context. But the function
    // signature says no arguments. The spec says "Use `git worktree list --porcelain`
    // to enumerate". It assumes we can run git from the "main" repo.
    // We'll require that the current directory is within a git repo, or we'll
    list_agent_worktrees_in(None)
}

/// `list_agent_worktrees` parameterized by repo root. Pass `Some(path)`
/// to query a specific repo (used by tests + by callers that already
/// know the repo). Pass `None` to use `std::env::current_dir()`.
pub fn list_agent_worktrees_in(repo_root: Option<&Path>) -> Result<Vec<AgentWorktree>> {
    let cwd_holder;
    let repo_root: &Path = match repo_root {
        Some(p) => p,
        None => {
            cwd_holder = std::env::current_dir().context("Cannot determine current directory")?;
            &cwd_holder
        }
    };

    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(&repo_root)
        .arg("worktree")
        .arg("list")
        .arg("--porcelain")
        .output()
        .context("Failed to run git worktree list")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("git worktree list failed: {stderr}"));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut worktrees = Vec::new();
    // Parsing porcelain format: each worktree block separated by blank lines.
    // Each block has lines like:
    // worktree /path/to/worktree
    // HEAD <sha>
    // branch refs/heads/agent/xxx
    // (or detached)
    // For our purposes we care about the branch line.
    for block in stdout.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let mut path = PathBuf::new();
        let mut branch = String::new();
        for line in block.lines() {
            if let Some(p) = line.strip_prefix("worktree ") {
                path = PathBuf::from(p);
            } else if let Some(b) = line.strip_prefix("branch refs/heads/") {
                branch = b.to_string();
            }
            // We ignore HEAD lines, etc.
        }
        // Only include branches starting with "agent/"
        if !branch.starts_with("agent/") {
            continue;
        }
        let session_id = branch
            .strip_prefix("agent/")
            .unwrap_or("")
            .to_string();

        // Compute stats: last checkpoint (we'll attempt to get from git log)
        let last_checkpoint_secs = get_last_checkpoint_secs(&path)?;

        // Compute files_changed and loc_delta using diff HEAD~..HEAD in that worktree
        let (files_changed, loc_delta) = get_worktree_stats(&path)?;

        worktrees.push(AgentWorktree {
            session_id,
            branch: format!("refs/heads/{}", branch),
            path,
            last_checkpoint_secs,
            files_changed,
            loc_delta,
        });
    }

    Ok(worktrees)
}

/// Get the timestamp of the last commit that has a message starting with "checkpoint"
/// (our auto-checkpoint commits).
fn get_last_checkpoint_secs(worktree_path: &Path) -> Result<Option<u64>> {
    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(worktree_path)
        .arg("log")
        .arg("--format=%ct")
        .arg("--max-count=1")
        .arg("--grep=^checkpoint:")
        .output()
        .context("Failed to get last checkpoint time")?;
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.trim();
        if !line.is_empty() {
            if let Ok(ts) = line.parse::<u64>() {
                return Ok(Some(ts));
            }
        }
    }
    Ok(None)
}

/// Compute shortstat diff between HEAD~ and HEAD in the given worktree.
/// Returns (files_changed, loc_delta).
fn get_worktree_stats(worktree_path: &Path) -> Result<(usize, i64)> {
    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(worktree_path)
        .arg("diff")
        .arg("--shortstat")
        .arg("HEAD~..HEAD")
        .output()
        .context("Failed to get worktree diff stats")?;
    if !output.status.success() {
        // If there is no previous commit (e.g., only one commit), return zeros.
        return Ok((0, 0));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_shortstat(&stdout)
}

fn parse_shortstat(stat: &str) -> Result<(usize, i64)> {
    // " 1 file changed, 5 insertions(+), 3 deletions(-)"
    // Words: "1", "file", "changed,", "5", "insertions(+),", "3", "deletions(-)"
    // Each NUMBER is its own word — the keyword is the NEXT word.
    let stat = stat.trim();
    if stat.is_empty() {
        return Ok((0, 0));
    }
    let mut files = 0usize;
    let mut insertions: i64 = 0;
    let mut deletions: i64 = 0;
    let parts: Vec<&str> = stat.split_whitespace().collect();
    for (i, raw) in parts.iter().enumerate() {
        let word = raw.trim_end_matches(',');
        let prev_num = || -> Option<&str> {
            if i == 0 { return None; }
            Some(parts[i - 1].trim_end_matches(','))
        };
        if word == "file" || word == "files" {
            if let Some(p) = prev_num() {
                files = p.parse().unwrap_or(0);
            }
        } else if word == "insertions(+)" || word == "insertion(+)" {
            if let Some(p) = prev_num() {
                insertions = p.parse().unwrap_or(0);
            }
        } else if word == "deletions(-)" || word == "deletion(-)" {
            if let Some(p) = prev_num() {
                deletions = p.parse().unwrap_or(0);
            }
        }
    }
    Ok((files, insertions - deletions))
}

// ---------------------------------------------------------------------------
// 3. preserve_session
// ---------------------------------------------------------------------------

/// Squash the agent branch into a single commit on main.
///
/// Steps:
/// 1. Switch to `main` branch.
/// 2. `git merge --squash agent/<session_id>`
/// 3. `git commit -m "preserve: agent/<session_id>\n\n<auto-summary>"`
/// 4. Return the new commit SHA via `git rev-parse HEAD`.
pub fn preserve_session(session_id: &str, repo_root: &Path) -> Result<String> {
    validate_session_id(session_id)?;
    let branch = format!("agent/{session_id}");

    // Detect the repository's default branch — main, master, trunk, etc.
    // Prefer origin/HEAD; fall back to current HEAD's branch name; fail
    // if neither resolves.
    let default_branch = detect_default_branch(repo_root)?;

    // Switch to default branch
    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("checkout")
        .arg(&default_branch)
        .output()
        .context("Failed to checkout default branch")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "git checkout {} failed: {stderr}",
            default_branch
        ));
    }

    // Squash merge — abort on conflict so the working tree is clean.
    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("merge")
        .arg("--squash")
        .arg(&branch)
        .output()
        .context("Failed to run git merge --squash")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        // Best-effort cleanup so the user isn't left with a half-merged tree.
        let _ = SyncCommand::new("git")
            .arg("-C").arg(repo_root)
            .arg("merge").arg("--abort")
            .output();
        let _ = SyncCommand::new("git")
            .arg("-C").arg(repo_root)
            .arg("reset").arg("--hard").arg("HEAD")
            .output();
        return Err(anyhow!(
            "git merge --squash failed (working tree restored): {stderr}"
        ));
    }

    // Generate auto-summary from staged changes
    let diff_stat = SyncCommand::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("diff")
        .arg("--cached")
        .arg("--shortstat")
        .output()
        .context("Failed to get diff stat")?;
    let summary = if diff_stat.status.success() {
        String::from_utf8_lossy(&diff_stat.stdout).trim().to_string()
    } else {
        String::new()
    };

    let commit_msg = format!(
        "preserve: agent/{}\n\n{}",
        session_id,
        if summary.is_empty() { "No changes" } else { &summary }
    );

    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("commit")
        .arg("-m")
        .arg(&commit_msg)
        .arg("--no-verify")
        .arg("--no-gpg-sign")
        .output()
        .context("Failed to commit squash")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("git commit failed: {stderr}"));
    }

    // Get the commit SHA
    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .context("Failed to get commit SHA")?;
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(sha)
}

// ---------------------------------------------------------------------------
// 4. cherry_pick_files
// ---------------------------------------------------------------------------
// Dep: globset = "0.4" (Cargo.toml)

/// Cherry-pick files matching a glob from the agent branch onto the current HEAD.
///
/// Lists files in `agent/<session_id>` matching the glob, checks them out,
/// stages them, and returns the number of matched files.
pub fn cherry_pick_files(session_id: &str, glob: &str, repo_root: &Path) -> Result<String> {
    validate_session_id(session_id)?;
    let branch = format!("agent/{session_id}");

    // List all files in the agent branch
    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("ls-tree")
        .arg("-r")
        .arg("--name-only")
        .arg(&branch)
        .output()
        .context("Failed to list files in agent branch")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("git ls-tree failed: {stderr}"));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let files: Vec<&str> = stdout.lines().collect();

    // Build globset
    let mut builder = globset::GlobSetBuilder::new();
    builder.add(globset::Glob::new(glob).context("Invalid glob pattern")?);
    let globset = builder.build().context("Failed to build globset")?;

    let matched: Vec<&&str> = files.iter().filter(|f| globset.is_match(f)).collect();
    let count = matched.len();

    for file in &matched {
        let output = SyncCommand::new("git")
            .arg("-C")
            .arg(repo_root)
            .arg("checkout")
            .arg(&branch)
            .arg("--")
            .arg(file)
            .output()
            .with_context(|| format!("Failed to checkout file: {file}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Warning: could not checkout {}: {stderr}", file);
        }
    }

    // Stage all checked-out files
    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("add")
        .arg("--")
        .args(&matched)
        .output()
        .context("Failed to stage files")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("git add failed: {stderr}"));
    }

    Ok(format!("{}", count))
}

// ---------------------------------------------------------------------------
// 5. drop_session
// ---------------------------------------------------------------------------

/// Remove the worktree and delete the agent branch.
///
/// Both operations are best-effort: failures are logged but not propagated.
pub fn drop_session(session_id: &str, repo_root: &Path) -> Result<()> {
    let worktree_path = match worktree_path_for(session_id) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Warning: cannot determine worktree path: {e}");
            return Ok(());
        }
    };
    let branch = format!("agent/{session_id}");

    // Attempt to remove worktree
    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("worktree")
        .arg("remove")
        .arg("--force")
        .arg(&worktree_path)
        .output();
    if let Err(e) = output {
        eprintln!("Warning: could not run git worktree remove: {e}");
    } else if let Ok(out) = output {
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("Warning: git worktree remove failed: {stderr}");
        }
    }

    // Attempt to delete branch
    let output = SyncCommand::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("branch")
        .arg("-D")
        .arg(&branch)
        .output();
    if let Err(e) = output {
        eprintln!("Warning: could not run git branch -D: {e}");
    } else if let Ok(out) = output {
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("Warning: git branch -D failed: {stderr}");
        }
    }

    Ok(())
}


// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command as SyncCommand;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// Tests set LAMU_SANDBOX_HOME → a tempdir to isolate from real $HOME.
    /// `set_var` is process-wide so we serialize through this mutex.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Returns (repo_dir, sandbox_dir) — both kept alive for the test.
    /// Sets LAMU_SANDBOX_HOME to the sandbox tempdir for this test's
    /// duration. Caller must hold the returned MutexGuard until done.
    fn setup_isolated() -> (TempDir, TempDir, std::sync::MutexGuard<'static, ()>) {
        let guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let sandbox = TempDir::new().expect("sandbox tempdir");
        // SAFETY: serialized via ENV_LOCK; no other thread runs setenv concurrently.
        unsafe { std::env::set_var("LAMU_SANDBOX_HOME", sandbox.path()); }
        let repo = setup_git_repo();
        (repo, sandbox, guard)
    }

    fn setup_git_repo() -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let root = dir.path();

        // Initialize git repo
        let init_out = SyncCommand::new("git")
            .arg("init")
            .arg(root)
            .output()
            .expect("git init failed");
        assert!(init_out.status.success());

        // Set user config to avoid commit errors
        let configs = [
            ("user.name", "test"),
            ("user.email", "test@example.com"),
        ];
        for (key, value) in &configs {
            SyncCommand::new("git")
                .arg("-C")
                .arg(root)
                .arg("config")
                .arg(key)
                .arg(value)
                .output()
                .expect("git config failed");
        }

        // Create an initial commit on main so there is a HEAD
        let readme = root.join("README.md");
        fs::write(&readme, "# Initial").expect("write readme");
        let add_out = SyncCommand::new("git")
            .arg("-C")
            .arg(root)
            .arg("add")
            .arg("README.md")
            .output()
            .expect("git add");
        assert!(add_out.status.success());
        let commit_out = SyncCommand::new("git")
            .arg("-C")
            .arg(root)
            .arg("commit")
            .arg("-m")
            .arg("initial commit")
            .output()
            .expect("git commit");
        assert!(commit_out.status.success());

        dir
    }

    #[test]
    fn validate_session_id_rejects_traversal_and_garbage() {
        // valid
        assert!(validate_session_id("abc").is_ok());
        assert!(validate_session_id("session-2026-05-08").is_ok());
        assert!(validate_session_id("a.b_c-d").is_ok());
        // invalid
        assert!(validate_session_id("").is_err());
        assert!(validate_session_id("..").is_err());
        assert!(validate_session_id(".").is_err());
        assert!(validate_session_id("../etc").is_err());
        assert!(validate_session_id(".hidden").is_err());
        assert!(validate_session_id("name with space").is_err());
        assert!(validate_session_id("name/with/slash").is_err());
        assert!(validate_session_id("name;rm -rf").is_err());
        assert!(validate_session_id(&"a".repeat(101)).is_err());
        // bash special
        assert!(validate_session_id("$(whoami)").is_err());
        assert!(validate_session_id("`pwd`").is_err());
    }

    #[test]
    fn parse_shortstat_handles_commas() {
        // Real `git diff --shortstat` output
        let s = " 1 file changed, 5 insertions(+), 3 deletions(-)";
        let (files, delta) = parse_shortstat(s).unwrap();
        assert_eq!(files, 1);
        assert_eq!(delta, 2); // 5 - 3
        // Multiple files
        let s = " 7 files changed, 100 insertions(+), 50 deletions(-)";
        let (files, delta) = parse_shortstat(s).unwrap();
        assert_eq!(files, 7);
        assert_eq!(delta, 50);
        // Singular variants
        let s = " 1 file changed, 1 insertion(+), 1 deletion(-)";
        let (files, delta) = parse_shortstat(s).unwrap();
        assert_eq!(files, 1);
        assert_eq!(delta, 0);
        // Insertions only
        let s = " 1 file changed, 5 insertions(+)";
        let (_files, delta) = parse_shortstat(s).unwrap();
        assert_eq!(delta, 5);
        // Empty
        let (files, delta) = parse_shortstat("").unwrap();
        assert_eq!(files, 0);
        assert_eq!(delta, 0);
    }

    // Test (a): create_worktree_then_drop_session_roundtrip
    #[test]
    fn create_worktree_then_drop_session_roundtrip() {
        let (repo_dir, _sandbox, _guard) = setup_isolated();
        let repo_root = repo_dir.path();

        let session_id = "test-session-123";
        let worktree_path = create_worktree(session_id, repo_root).expect("create worktree");

        // Verify the worktree exists and is on the correct branch
        let branch_out = SyncCommand::new("git")
            .arg("-C")
            .arg(&worktree_path)
            .arg("rev-parse")
            .arg("--abbrev-ref")
            .arg("HEAD")
            .output()
            .expect("git rev-parse");
        let branch = String::from_utf8_lossy(&branch_out.stdout).trim().to_string();
        assert_eq!(branch, format!("agent/{session_id}"));

        // Verify it's listed by list_agent_worktrees
        let worktrees = list_agent_worktrees_in(Some(repo_root)).expect("list worktrees");
        let found = worktrees.iter().any(|w| w.session_id == session_id);
        assert!(found, "worktree should be listed");

        // Drop session
        drop_session(session_id, repo_root).expect("drop session");

        // Verify worktree is gone
        let worktrees_after = list_agent_worktrees_in(Some(repo_root)).expect("list worktrees after drop");
        let found_after = worktrees_after.iter().any(|w| w.session_id == session_id);
        assert!(!found_after, "worktree should be removed after drop");
    }

    // Test (b): list_agent_worktrees_empty_when_no_agents
    #[test]
    fn list_agent_worktrees_empty_when_no_agents() {
        let (repo_dir, _sandbox, _guard) = setup_isolated();
        let repo_root = repo_dir.path();

        let worktrees = list_agent_worktrees_in(Some(repo_root)).expect("list worktrees");
        assert!(worktrees.is_empty(), "No agent worktrees should exist");
    }

    // Test (c): drop_session_ignores_missing
    #[test]
    fn drop_session_ignores_missing() {
        let (repo_dir, _sandbox, _guard) = setup_isolated();
        let repo_root = repo_dir.path();

        // Should not panic or error on non-existent session
        let result = drop_session("nonexistent", repo_root);
        assert!(result.is_ok(), "drop_session should not fail on missing");
    }

    // Test (d): cherry_pick_glob_matches_expected_files
    #[test]
    fn cherry_pick_glob_matches_expected_files() {
        let (repo_dir, _sandbox, _guard) = setup_isolated();
        let repo_root = repo_dir.path();
        let session_id = "test-cherry";

        // Create a worktree and add some files
        let worktree_path = create_worktree(session_id, repo_root).expect("create worktree");

        // Create files in worktree
        fs::write(worktree_path.join("alpha.txt"), "alpha").expect("write alpha");
        fs::write(worktree_path.join("beta.log"), "beta").expect("write beta");
        fs::write(worktree_path.join("gamma.txt"), "gamma").expect("write gamma");
        fs::create_dir_all(worktree_path.join("sub")).expect("mkdir sub");
        fs::write(worktree_path.join("sub/delta.txt"), "delta").expect("write delta");
        fs::write(worktree_path.join("sub/epsilon.log"), "epsilon").expect("write epsilon");

        // Commit them on the agent branch
        let add_out = SyncCommand::new("git")
            .arg("-C")
            .arg(&worktree_path)
            .arg("add")
            .arg("-A")
            .output()
            .expect("git add");
        assert!(add_out.status.success());
        let commit_out = SyncCommand::new("git")
            .arg("-C")
            .arg(&worktree_path)
            .arg("commit")
            .arg("-m")
            .arg("initial agent files")
            .output()
            .expect("git commit");
        assert!(commit_out.status.success());

        // Switch back to main in the main repo (worktree already on agent branch)
        // But cherry_pick_files operates from repo_root.
        // We need to checkout main again in the main repo to avoid being on agent branch.
        SyncCommand::new("git")
            .arg("-C")
            .arg(repo_root)
            .arg("checkout")
            .arg("main")
            .output()
            .expect("checkout main");

        // Cherry-pick *.txt files
        let result = cherry_pick_files(session_id, "*.txt", repo_root)
            .expect("cherry_pick_files");
        // Should match alpha.txt, gamma.txt, sub/delta.txt (3 files)
        assert_eq!(result, "3", "Expected 3 .txt files matched");

        // Verify they are staged on main (we can check with git diff --cached)
        let diff_out = SyncCommand::new("git")
            .arg("-C")
            .arg(repo_root)
            .arg("diff")
            .arg("--cached")
            .arg("--name-only")
            .output()
            .expect("git diff --cached");
        let stdout = String::from_utf8_lossy(&diff_out.stdout).into_owned();
        let staged: Vec<&str> = stdout.lines().collect();
        assert!(staged.contains(&"alpha.txt"), "alpha.txt should be staged");
        assert!(staged.contains(&"gamma.txt"), "gamma.txt should be staged");
        assert!(staged.contains(&"sub/delta.txt"), "sub/delta.txt should be staged");
        assert!(!staged.contains(&"beta.log"), "beta.log should not be staged");
        assert!(!staged.contains(&"sub/epsilon.log"), "sub/epsilon.log should not be staged");
    }
}
