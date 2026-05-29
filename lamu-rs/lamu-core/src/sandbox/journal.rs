//! Layer 1 — Filesystem journal.
//!
//! Every fs mutation an agent makes is recorded with the original
//! bytes so we can replay-in-reverse. The journal is per-session at
//! `~/.local/share/lamu/sandbox/journal/<session_id>.jsonl`.
//!
//! Use `safe_write`, `safe_delete`, `safe_create_dir` instead of the
//! raw `std::fs::*` calls when accepting agent-driven paths. They:
//! 1. Stat the path before mutation (capture original bytes / type).
//! 2. Write a journal entry to disk (durable BEFORE the mutation
//!    happens, so a crash mid-op still leaves a recoverable record).
//! 3. Apply the mutation.
//!
//! `lamu rollback <session_id>` reads the journal in reverse and
//! restores each entry's pre-state.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use super::sandbox_root;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum JournalEntry {
    /// Record an existing file's bytes before overwrite/delete. Empty
    /// string when the path didn't exist (creation case).
    Write {
        path: PathBuf,
        /// Pre-state: None when path didn't exist before the op.
        before: Option<EncodedBlob>,
        ts: u64,
    },
    Delete {
        path: PathBuf,
        before: Option<EncodedBlob>,
        ts: u64,
    },
    Mkdir {
        path: PathBuf,
        existed: bool,
        ts: u64,
    },
}

/// Base64-encoded file contents. Plain JSON keeps the journal
/// human-inspectable for small files; binaries get base64.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedBlob {
    pub size: u64,
    pub b64: String,
}

impl EncodedBlob {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            size: bytes.len() as u64,
            b64: base64_encode(bytes),
        }
    }
    /// Decode the journaled bytes. Returns an error when the blob
    /// payload is malformed — callers can decide to skip the rollback
    /// step rather than silently restoring a file to empty bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        base64_decode(&self.b64)
    }
}

/// Per-session journal handle. Cheap to construct.
pub struct Journal {
    pub session_id: String,
    pub path: PathBuf,
}

/// Validate a session id before joining it into a filesystem path.
/// Allows ASCII letters, digits, underscore, dash, and dot, and
/// rejects empty, leading-dot, or `..`-containing values to block
/// path traversal. The on-disk format `<session_id>.jsonl` means a
/// session id with `/` could escape the journal directory entirely;
/// MCP and CLI both pass user-controllable strings here so the
/// allowlist must be narrow.
fn validate_session_id(session_id: &str) -> Result<()> {
    if session_id.is_empty() {
        anyhow::bail!("session id is empty");
    }
    if session_id.starts_with('.') {
        anyhow::bail!("session id cannot start with '.': {session_id}");
    }
    if session_id.contains("..") {
        anyhow::bail!("session id contains '..': {session_id}");
    }
    if !session_id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) {
        anyhow::bail!(
            "session id contains forbidden character — allowed: [A-Za-z0-9_-.]: {session_id}"
        );
    }
    Ok(())
}

impl Journal {
    pub fn open(session_id: &str) -> Result<Self> {
        validate_session_id(session_id)?;
        let dir = sandbox_root().join("journal");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.jsonl", session_id));
        Ok(Self { session_id: session_id.to_string(), path })
    }

    fn append(&self, entry: &JournalEntry) -> Result<()> {
        let line = serde_json::to_string(entry)? + "\n";
        let mut f = OpenOptions::new()
            .create(true).append(true).open(&self.path)
            .with_context(|| format!("open journal {}", self.path.display()))?;
        f.write_all(line.as_bytes())?;
        f.sync_data()?;
        Ok(())
    }

    pub fn read_all(&self) -> Result<Vec<JournalEntry>> {
        let mut out = Vec::new();
        let Ok(f) = std::fs::File::open(&self.path) else { return Ok(out); };
        for line_res in BufReader::new(f).lines() {
            let line = line_res?;
            if line.trim().is_empty() { continue; }
            match serde_json::from_str::<JournalEntry>(&line) {
                Ok(e) => out.push(e),
                Err(e) => eprintln!("journal: skipping bad line: {}", e),
            }
        }
        Ok(out)
    }
}

// ── safe_* mutation helpers ──────────────────────────────────────────

pub fn safe_write(journal: &Journal, path: &Path, bytes: &[u8]) -> Result<()> {
    // Refuse to journal-then-write over a directory: std::fs::write
    // would fail and leave the journal claiming a write that never
    // happened. Catch it before recording.
    if path.is_dir() {
        anyhow::bail!(
            "safe_write target is a directory, not a file: {}",
            path.display()
        );
    }
    // Refuse to follow symlinks at the leaf. symlink_metadata does NOT
    // dereference; it tells us whether the path itself is a symlink.
    // The metadata error is intentionally swallowed: NotFound means
    // the leaf doesn't exist yet (the common create case); a perm
    // error cascades into the open below failing anyway, so the net
    // effect is the same — we never silently succeed at a bypass.
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            anyhow::bail!(
                "safe_write refuses to follow leaf symlink: {}",
                path.display()
            );
        }
    }
    let before = read_blob(path)?; // aborts if the target exists but is unreadable (#27)
    journal.append(&JournalEntry::Write {
        path: path.to_path_buf(),
        before,
        ts: now_secs(),
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_no_follow(path, bytes)?;
    Ok(())
}

/// Write `bytes` to `path` with O_NOFOLLOW on Unix so an attacker who
/// races between symlink_metadata and the open call can't swap in a
/// symlink and have us write through it. `OpenOptions::truncate(true)`
/// keeps the existing file behavior (overwrite-not-append) consistent
/// with `std::fs::write`. On non-Unix platforms we fall back to
/// `std::fs::write`; the symlink_metadata pre-check above is the only
/// guard there.
#[cfg(unix)]
fn write_no_follow(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    // O_NOFOLLOW on Linux = 0x20000, macOS = 0x100, FreeBSD = 0x100.
    // libc isn't a direct dep here, so go through cfg-specific values.
    #[cfg(target_os = "linux")]
    const O_NOFOLLOW: i32 = 0x20000;
    #[cfg(any(target_os = "macos", target_os = "freebsd", target_os = "ios"))]
    const O_NOFOLLOW: i32 = 0x100;
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "freebsd", target_os = "ios")))]
    const O_NOFOLLOW: i32 = 0; // unknown unix — soft fallback (no-op flag)

    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(O_NOFOLLOW)
        .open(path)?;
    f.write_all(bytes)?;
    Ok(())
}

#[cfg(not(unix))]
fn write_no_follow(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).map_err(Into::into)
}

pub fn safe_delete(journal: &Journal, path: &Path) -> Result<()> {
    // symlink_metadata does NOT follow the leaf, so we classify the path
    // itself, never its target.
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()), // idempotent
        Err(e) => return Err(e).with_context(|| format!("stat {}", path.display())),
    };
    let ft = meta.file_type();

    if ft.is_symlink() {
        // Unlink the LINK only — never deref it. The old code's read_blob
        // followed the symlink and copied the TARGET's bytes into the
        // journal (info-leak), and rollback would then materialize a
        // regular file with that content at the link's path (type change).
        // The journal has no Symlink entry, so a deleted symlink is not
        // restorable; record before: None. (#18)
        journal.append(&JournalEntry::Delete {
            path: path.to_path_buf(),
            before: None,
            ts: now_secs(),
        })?;
        std::fs::remove_file(path)?; // remove_file on a symlink unlinks the link
        return Ok(());
    }

    if ft.is_dir() {
        // Journal each contained FILE as a restorable Delete (rollback's
        // create_dir_all recreates parent dirs), THEN wipe the tree. The
        // old code journaled before: None for the whole directory, so
        // remove_dir_all destroyed the subtree unrecoverably. Empty
        // sub-directories are not separately restored — files + their
        // paths are. (#17) If any file can't be read (read_blob errs), the
        // whole delete aborts BEFORE remove_dir_all — we never wipe a tree
        // we couldn't fully journal.
        let mut files = Vec::new();
        collect_files_no_follow(path, &mut files)?;
        for f in &files {
            let before = read_blob(f)?;
            journal.append(&JournalEntry::Delete {
                path: f.clone(),
                before,
                ts: now_secs(),
            })?;
        }
        std::fs::remove_dir_all(path)?;
        return Ok(());
    }

    // Regular file.
    let before = read_blob(path)?;
    journal.append(&JournalEntry::Delete {
        path: path.to_path_buf(),
        before,
        ts: now_secs(),
    })?;
    std::fs::remove_file(path)?;
    Ok(())
}

pub fn safe_create_dir(journal: &Journal, path: &Path) -> Result<()> {
    let existed = path.exists();
    journal.append(&JournalEntry::Mkdir {
        path: path.to_path_buf(),
        existed,
        ts: now_secs(),
    })?;
    std::fs::create_dir_all(path)?;
    Ok(())
}

// ── rollback ─────────────────────────────────────────────────────────

/// Walk journal entries in reverse, restoring each entry's pre-state.
/// Returns count of ops restored / skipped.
pub fn rollback(session_id: &str) -> Result<(usize, usize)> {
    let journal = Journal::open(session_id)?;
    let entries = journal.read_all()?;
    let mut restored = 0;
    let mut skipped = 0;
    for entry in entries.into_iter().rev() {
        if rollback_one(&entry).is_ok() {
            restored += 1;
        } else {
            skipped += 1;
        }
    }
    Ok((restored, skipped))
}

pub fn rollback_one(entry: &JournalEntry) -> Result<()> {
    match entry {
        JournalEntry::Write { path, before, .. } => match before {
            Some(blob) => {
                let bytes = blob.to_bytes()
                    .with_context(|| format!("decode journaled blob for {}", path.display()))?;
                if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
                std::fs::write(path, bytes)?;
            }
            None => {
                // Path didn't exist before — delete it now if it does.
                if path.exists() && path.is_file() {
                    std::fs::remove_file(path)?;
                }
            }
        },
        JournalEntry::Delete { path, before, .. } => {
            if let Some(blob) = before {
                let bytes = blob.to_bytes()
                    .with_context(|| format!("decode journaled blob for {}", path.display()))?;
                if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
                std::fs::write(path, bytes)?;
            }
        }
        JournalEntry::Mkdir { path, existed, .. } => {
            if !existed && path.exists() && path.is_dir() {
                // Only remove empty dirs we created — refuse to clobber
                // files the agent put inside.
                let is_empty = std::fs::read_dir(path).map(|mut it| it.next().is_none()).unwrap_or(false);
                if is_empty {
                    std::fs::remove_dir(path)?;
                }
            }
        }
    }
    Ok(())
}

// ── helpers ──────────────────────────────────────────────────────────

/// Capture a leaf's bytes for the journal. Returns:
/// - `Ok(None)` when the path doesn't exist, is a directory, or is a
///   symlink (none of which carry restorable leaf bytes), OR
/// - `Ok(Some(blob))` for a readable regular file, OR
/// - `Err(..)` when the path IS a regular file but reading it fails
///   (EACCES/EIO). The old `std::fs::read(path).ok()` collapsed that last
///   case to `None`, so `safe_write` journaled `before: None` and a later
///   rollback DELETED the (unreadable but existing) file. Propagating the
///   error makes the caller abort instead. (#27)
fn read_blob(path: &Path) -> Result<Option<EncodedBlob>> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("stat {}", path.display())),
    };
    if meta.is_dir() || meta.file_type().is_symlink() {
        return Ok(None);
    }
    match std::fs::read(path) {
        Ok(b) => Ok(Some(EncodedBlob::from_bytes(&b))),
        Err(e) => Err(e).with_context(|| format!("read {} for journal", path.display())),
    }
}

/// Recursively collect every non-directory leaf (files AND symlinks) under
/// `dir`, WITHOUT following symlinks (`DirEntry::file_type` reads the entry
/// type, not the target). Used by `safe_delete` to journal each file before
/// wiping a tree.
fn collect_files_no_follow(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        let p = entry.path();
        if ft.is_dir() {
            collect_files_no_follow(&p, out)?;
        } else {
            out.push(p);
        }
    }
    Ok(())
}

fn now_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// Tiny base64 implementation (we don't pull in a crate just for this).
const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        out.push(B64[(b0 >> 2) as usize] as char);
        out.push(B64[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn base64_decode(input: &str) -> Result<Vec<u8>> {
    // Build a lookup table where invalid byte = 0xFF sentinel. This
    // lets us reject anything outside the alphabet — without it, a
    // tampered or truncated journal silently restores garbage bytes.
    let mut lookup = [0xFFu8; 256];
    for (i, b) in B64.iter().enumerate() { lookup[*b as usize] = i as u8; }

    let bytes: Vec<u8> = input.bytes().filter(|&b| b != b'\n' && b != b'\r').collect();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);

    let valid_byte = |b: u8| -> Result<u8> {
        let v = lookup[b as usize];
        if v == 0xFF {
            anyhow::bail!("invalid base64 byte 0x{:02x}", b);
        }
        Ok(v)
    };

    let mut chunks = bytes.chunks(4).peekable();
    while let Some(chunk) = chunks.next() {
        if chunk.len() < 2 {
            anyhow::bail!("base64 truncated: trailing chunk has {} bytes", chunk.len());
        }
        let v0 = valid_byte(chunk[0])? as u32;
        let v1 = valid_byte(chunk[1])? as u32;

        // Padding rules: once we see '=' the rest of the chunk and
        // everything after the chunk must also be '='. Reject `X==Y`
        // (mid-chunk pad followed by data) and reject pad in non-final
        // chunk.
        let c2_pad = chunk.len() > 2 && chunk[2] == b'=';
        let c3_pad = chunk.len() > 3 && chunk[3] == b'=';
        if c2_pad && chunk.len() > 3 && !c3_pad {
            anyhow::bail!("base64 invalid padding: '=' followed by non-pad in same chunk");
        }
        if (c2_pad || c3_pad) && chunks.peek().is_some() {
            anyhow::bail!("base64 invalid padding: '=' before final chunk");
        }

        let v2 = if chunk.len() > 2 && !c2_pad {
            valid_byte(chunk[2])? as u32
        } else { 0 };
        let v3 = if chunk.len() > 3 && !c3_pad {
            valid_byte(chunk[3])? as u32
        } else { 0 };
        out.push(((v0 << 2) | (v1 >> 4)) as u8);
        if chunk.len() > 2 && !c2_pad {
            out.push(((v1 << 4) | (v2 >> 2)) as u8);
        }
        if chunk.len() > 3 && !c3_pad {
            out.push(((v2 << 6) | v3) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn base64_rejects_invalid() {
        // Garbage outside the alphabet
        assert!(base64_decode("X#YZ").is_err());
        // Padding in middle then data (X==Y)
        assert!(base64_decode("X==Y").is_err());
        // Truncated
        assert!(base64_decode("X").is_err());
        // Padding in non-final chunk
        assert!(base64_decode("X===abcd").is_err());
    }

    #[test]
    fn base64_roundtrip() {
        let cases = [&b""[..], b"a", b"ab", b"abc", b"abcd", b"hello world"];
        for c in cases {
            let enc = base64_encode(c);
            let dec = base64_decode(&enc).unwrap();
            assert_eq!(dec, c);
        }
    }

    #[test]
    fn safe_write_records_create() {
        let tmp = tempdir().unwrap();
        let session = format!("test-{}", std::process::id());
        let j = Journal { session_id: session.clone(), path: tmp.path().join("journal.jsonl") };
        let target = tmp.path().join("new.txt");
        safe_write(&j, &target, b"hello").unwrap();
        let entries = j.read_all().unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            JournalEntry::Write { before, .. } => assert!(before.is_none()),
            _ => panic!("wrong entry type"),
        }
        assert_eq!(std::fs::read(&target).unwrap(), b"hello");
    }

    #[test]
    fn safe_write_records_overwrite_and_rollback_restores() {
        let tmp = tempdir().unwrap();
        let target = tmp.path().join("f.txt");
        std::fs::write(&target, b"original").unwrap();

        let j = Journal {
            session_id: "test-overwrite".into(),
            path: tmp.path().join("journal.jsonl"),
        };
        safe_write(&j, &target, b"clobbered").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"clobbered");

        // Manually walk-back via the helper.
        let entries = j.read_all().unwrap();
        for e in entries.iter().rev() {
            super::rollback_one(e).unwrap();
        }
        assert_eq!(std::fs::read(&target).unwrap(), b"original");
    }

    #[test]
    fn safe_delete_records_and_rollback_restores() {
        let tmp = tempdir().unwrap();
        let target = tmp.path().join("doomed.txt");
        std::fs::write(&target, b"keepme").unwrap();

        let j = Journal {
            session_id: "test-delete".into(),
            path: tmp.path().join("journal.jsonl"),
        };
        safe_delete(&j, &target).unwrap();
        assert!(!target.exists());

        let entries = j.read_all().unwrap();
        for e in entries.iter().rev() {
            super::rollback_one(e).unwrap();
        }
        assert_eq!(std::fs::read(&target).unwrap(), b"keepme");
    }

    #[test]
    fn safe_delete_dir_recurses_and_rollback_restores_files() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path().join("tree");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.txt"), b"alpha").unwrap();
        std::fs::write(dir.join("sub/b.txt"), b"bravo").unwrap();

        let j = Journal {
            session_id: "test-delete-dir".into(),
            path: tmp.path().join("journal.jsonl"),
        };
        safe_delete(&j, &dir).unwrap();
        assert!(!dir.exists(), "tree wiped");

        for e in j.read_all().unwrap().iter().rev() {
            super::rollback_one(e).unwrap();
        }
        // Files (and their parent dirs) restored — the old before:None on
        // the whole dir made this unrecoverable.
        assert_eq!(std::fs::read(dir.join("a.txt")).unwrap(), b"alpha");
        assert_eq!(std::fs::read(dir.join("sub/b.txt")).unwrap(), b"bravo");
    }

    #[cfg(unix)]
    #[test]
    fn safe_delete_symlink_unlinks_link_not_target() {
        let tmp = tempdir().unwrap();
        let target = tmp.path().join("secret.txt");
        std::fs::write(&target, b"do-not-touch").unwrap();
        let link = tmp.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let j = Journal {
            session_id: "test-delete-symlink".into(),
            path: tmp.path().join("journal.jsonl"),
        };
        safe_delete(&j, &link).unwrap();

        assert!(!link.exists(), "link removed");
        // The TARGET and its bytes are untouched (no deref-and-delete).
        assert_eq!(std::fs::read(&target).unwrap(), b"do-not-touch");
        // The journal recorded before: None for the link — no target bytes
        // leaked into it.
        let entries = j.read_all().unwrap();
        let leaked = entries.iter().any(|e| matches!(
            e,
            JournalEntry::Delete { before: Some(b), .. } if b.to_bytes().ok() == Some(b"do-not-touch".to_vec())
        ));
        assert!(!leaked, "target bytes must not be journaled");
    }

    #[test]
    fn rollback_skips_dir_if_agent_added_files() {
        let tmp = tempdir().unwrap();
        let new_dir = tmp.path().join("agent_dir");
        let j = Journal {
            session_id: "test-mkdir".into(),
            path: tmp.path().join("journal.jsonl"),
        };
        safe_create_dir(&j, &new_dir).unwrap();
        std::fs::write(new_dir.join("user_added.txt"), b"data").unwrap();

        let entries = j.read_all().unwrap();
        for e in entries.iter().rev() {
            super::rollback_one(e).unwrap();
        }
        // Dir should still exist because it's not empty — refusing to
        // clobber files the user added.
        assert!(new_dir.exists());
    }

    #[test]
    fn validate_session_id_rejects_path_traversal() {
        // Direct slash escape.
        assert!(validate_session_id("../etc/passwd").is_err());
        // Double-dot fragment anywhere.
        assert!(validate_session_id("a..b").is_err());
        // Leading dot — could shadow `.something.jsonl`.
        assert!(validate_session_id(".hidden").is_err());
        // Backslashes still rejected (Windows path-traversal style).
        assert!(validate_session_id("a\\b").is_err());
        // Empty.
        assert!(validate_session_id("").is_err());
        // Canonical accepted shapes.
        assert!(validate_session_id("20260509-035410-12345").is_ok());
        assert!(validate_session_id("test-rollback").is_ok());
        assert!(validate_session_id("agent_42").is_ok());
    }

    #[test]
    fn safe_write_rejects_directory_target() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path().join("a_dir");
        std::fs::create_dir(&dir).unwrap();
        let j = Journal {
            session_id: "test-dir-target".into(),
            path: tmp.path().join("journal.jsonl"),
        };
        let r = safe_write(&j, &dir, b"hello");
        assert!(r.is_err(), "should reject dir target before journaling");
        // Journal must NOT have appended a phantom Write entry.
        assert!(!j.path.exists(), "journal should not exist when safe_write refuses up front");
    }

    #[test]
    fn rollback_surfaces_decode_error_instead_of_silent_truncate() {
        let entry = JournalEntry::Write {
            path: PathBuf::from("/tmp/lamu-test-rollback-decode"),
            before: Some(EncodedBlob {
                size: 3,
                b64: "@@@".into(),
            }),
            ts: 0,
        };
        let r = super::rollback_one(&entry);
        assert!(r.is_err(), "malformed b64 should surface as rollback error");
    }

    #[cfg(unix)]
    #[test]
    fn safe_write_o_nofollow_blocks_swap_race() {
        // Simulate the TOCTOU race: pre-create the leaf as a symlink
        // pointing outside (the prior symlink_metadata check would
        // catch this on the first call, but we want to prove
        // O_NOFOLLOW also fails to follow when the open is reached).
        let tmp = tempdir().unwrap();
        let outside = tmp.path().join("outside.txt");
        std::fs::write(&outside, b"original").unwrap();
        let inside = tmp.path().join("inside.txt");
        std::os::unix::fs::symlink(&outside, &inside).unwrap();
        let r = write_no_follow(&inside, b"clobber");
        assert!(r.is_err(), "O_NOFOLLOW should refuse to open symlink");
        // outside.txt is unchanged.
        assert_eq!(std::fs::read(&outside).unwrap(), b"original");
    }

    #[cfg(unix)]
    #[test]
    fn safe_write_o_nofollow_creates_new_file() {
        // Sanity: O_NOFOLLOW does NOT block creating a new regular
        // file at a path that doesn't exist yet (the common case).
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("fresh.txt");
        write_no_follow(&path, b"hello").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello");
    }
}
