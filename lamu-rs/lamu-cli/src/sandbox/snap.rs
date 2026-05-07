//! Layer 4 — Git snapshot at session start.
//!
//! When a chat session opens in a git repo, capture:
//! - HEAD commit SHA
//! - `git stash create` blob (tracked changes, doesn't disturb working tree)
//! - tar.zst of untracked files
//!
//! On `lamu undo`, the user picks a session and we restore: hard-reset
//! to HEAD, apply the stash, restore untracked files from the archive.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

use super::{new_session_id, sandbox_root};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub session_id: String,
    pub timestamp: u64,
    pub model: String,
    pub cwd: PathBuf,
    /// Git repo root (cwd or a parent). None if cwd isn't in a git repo.
    pub git_repo: Option<PathBuf>,
    /// HEAD commit SHA at session start.
    pub git_head: Option<String>,
    /// `git stash create` SHA. None if working tree was clean.
    pub git_stash: Option<String>,
    /// Path to tar.zst of untracked files. None if no untracked files.
    pub untracked_archive: Option<PathBuf>,
    /// True once restored — Snapshot::restore sets this so `lamu undo`
    /// can warn before re-applying.
    #[serde(default)]
    pub restored: bool,
}

impl Snapshot {
    /// Capture current state. Cheap when not in a git repo (just records
    /// metadata).
    pub fn capture(model: &str) -> Result<Self> {
        let cwd = std::env::current_dir().context("getting cwd")?;
        let git_repo = find_git_root(&cwd);
        let session_id = new_session_id();

        let (git_head, git_stash, untracked_archive) = if let Some(repo) = &git_repo {
            let head = git_str(repo, &["rev-parse", "HEAD"]).ok();
            // git stash create captures tracked changes WITHOUT modifying
            // working tree or pushing onto stash list.
            let stash = git_str(repo, &["stash", "create"])
                .ok()
                .filter(|s| !s.trim().is_empty());
            let untracked_archive = archive_untracked(repo, &session_id)?;
            (head, stash, untracked_archive)
        } else {
            (None, None, None)
        };

        let snap = Self {
            session_id: session_id.clone(),
            timestamp: now_secs(),
            model: model.to_string(),
            cwd,
            git_repo,
            git_head,
            git_stash,
            untracked_archive,
            restored: false,
        };
        snap.save()?;
        Ok(snap)
    }

    pub fn save(&self) -> Result<()> {
        let dir = sessions_dir();
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.toml", self.session_id));
        let body = toml::to_string_pretty(self).context("serialize snapshot")?;
        std::fs::write(&path, body)?;
        Ok(())
    }

    pub fn load(session_id: &str) -> Result<Self> {
        let path = sessions_dir().join(format!("{}.toml", session_id));
        let body = std::fs::read_to_string(&path)
            .with_context(|| format!("read session {}", session_id))?;
        let snap: Self = toml::from_str(&body).context("parse snapshot toml")?;
        Ok(snap)
    }

    pub fn list() -> Result<Vec<Self>> {
        let dir = sessions_dir();
        let mut out: Vec<Self> = Vec::new();
        let Ok(entries) = std::fs::read_dir(&dir) else { return Ok(out); };
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some("toml") { continue; }
            if let Ok(body) = std::fs::read_to_string(&path) {
                if let Ok(s) = toml::from_str::<Self>(&body) {
                    out.push(s);
                }
            }
        }
        out.sort_by_key(|s| std::cmp::Reverse(s.timestamp));
        Ok(out)
    }

    /// Restore the working tree to its pre-session state. Hard reset to
    /// HEAD, apply tracked-changes stash, restore untracked files.
    pub fn restore(&self) -> Result<()> {
        let Some(repo) = &self.git_repo else {
            anyhow::bail!("session {} was not in a git repo — nothing to restore", self.session_id);
        };
        let Some(head) = &self.git_head else {
            anyhow::bail!("session {} captured no HEAD — refusing to reset", self.session_id);
        };

        // Reset tracked files to HEAD at session start.
        git_str(repo, &["reset", "--hard", head])
            .with_context(|| format!("git reset --hard {head}"))?;

        // Re-apply the tracked-changes stash, if any.
        if let Some(stash) = &self.git_stash {
            // `git stash apply <sha>` applies the stash blob.
            git_str(repo, &["stash", "apply", stash])
                .with_context(|| format!("git stash apply {stash}"))?;
        }

        // Restore untracked files from the archive.
        if let Some(archive) = &self.untracked_archive {
            if archive.exists() {
                let status = Command::new("tar")
                    .arg("-xf")
                    .arg(archive)
                    .arg("-C")
                    .arg(repo)
                    .status()
                    .context("untar untracked archive")?;
                if !status.success() {
                    anyhow::bail!("tar exited {} restoring {}", status, archive.display());
                }
            }
        }

        // Mark restored so a second undo warns.
        let mut snap = self.clone();
        snap.restored = true;
        let _ = snap.save();

        Ok(())
    }

    pub fn pretty_summary(&self) -> String {
        let when = format_ts(self.timestamp);
        let where_ = self.git_repo.as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| format!("(no git) {}", self.cwd.display()));
        let head_short = self.git_head.as_deref()
            .map(|s| s.chars().take(8).collect::<String>())
            .unwrap_or_else(|| "—".into());
        let dirty = if self.git_stash.is_some() { "+stash" } else { "" };
        let untracked = if self.untracked_archive.is_some() { "+untracked" } else { "" };
        let status = if self.restored { " [RESTORED]" } else { "" };
        format!(
            "{}  {}  model={}  HEAD={}{}{}{}  {}",
            self.session_id, when, self.model, head_short, dirty, untracked, status, where_
        )
    }
}

fn sessions_dir() -> PathBuf {
    sandbox_root().join("sessions")
}

fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut cur = start.to_path_buf();
    loop {
        if cur.join(".git").exists() { return Some(cur); }
        if !cur.pop() { return None; }
    }
}

fn git_str(repo: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git").current_dir(repo).args(args).output()
        .with_context(|| format!("spawn git {:?}", args))?;
    if !out.status.success() {
        anyhow::bail!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn archive_untracked(repo: &Path, session_id: &str) -> Result<Option<PathBuf>> {
    let listing = git_str(repo, &["ls-files", "--others", "--exclude-standard"])?;
    if listing.trim().is_empty() {
        return Ok(None);
    }
    let dir = sandbox_root().join("untracked");
    std::fs::create_dir_all(&dir)?;
    let archive = dir.join(format!("{}.tar.zst", session_id));

    // Pipe the listing as -T file list to tar.
    let list_path = dir.join(format!("{}.list", session_id));
    std::fs::write(&list_path, &listing)?;

    let status = Command::new("tar")
        .arg("--zstd")
        .arg("-cf")
        .arg(&archive)
        .arg("-C")
        .arg(repo)
        .arg("-T")
        .arg(&list_path)
        .status()
        .context("spawn tar")?;
    let _ = std::fs::remove_file(&list_path);
    if !status.success() {
        anyhow::bail!("tar exited {} archiving untracked", status);
    }
    Ok(Some(archive))
}

fn now_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn format_ts(secs: u64) -> String {
    let s = secs;
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400;
    let year = 1970 + days / 365;
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month.min(12), day.min(31), hour, min, sec
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretty_summary_no_git() {
        let s = Snapshot {
            session_id: "20260507-094500-12345".into(),
            timestamp: 0,
            model: "test".into(),
            cwd: PathBuf::from("/tmp/x"),
            git_repo: None,
            git_head: None,
            git_stash: None,
            untracked_archive: None,
            restored: false,
        };
        let line = s.pretty_summary();
        assert!(line.contains("20260507-094500-12345"));
        assert!(line.contains("(no git)"));
        assert!(line.contains("HEAD=—"));
    }

    #[test]
    fn restored_marker_in_summary() {
        let s = Snapshot {
            session_id: "x".into(),
            timestamp: 0,
            model: "m".into(),
            cwd: PathBuf::from("/tmp"),
            git_repo: Some(PathBuf::from("/tmp")),
            git_head: Some("a".repeat(40)),
            git_stash: None,
            untracked_archive: None,
            restored: true,
        };
        assert!(s.pretty_summary().contains("[RESTORED]"));
    }
}
