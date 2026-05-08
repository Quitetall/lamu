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
    pub fn to_bytes(&self) -> Vec<u8> {
        base64_decode(&self.b64).unwrap_or_default()
    }
}

/// Per-session journal handle. Cheap to construct.
pub struct Journal {
    pub session_id: String,
    pub path: PathBuf,
}

impl Journal {
    pub fn open(session_id: &str) -> Result<Self> {
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
    let before = read_blob(path);
    journal.append(&JournalEntry::Write {
        path: path.to_path_buf(),
        before,
        ts: now_secs(),
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn safe_delete(journal: &Journal, path: &Path) -> Result<()> {
    let before = read_blob(path);
    journal.append(&JournalEntry::Delete {
        path: path.to_path_buf(),
        before,
        ts: now_secs(),
    })?;
    if path.is_dir() {
        std::fs::remove_dir_all(path)?;
    } else if path.exists() {
        std::fs::remove_file(path)?;
    }
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

fn rollback_one(entry: &JournalEntry) -> Result<()> {
    match entry {
        JournalEntry::Write { path, before, .. } => match before {
            Some(blob) => {
                if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
                std::fs::write(path, blob.to_bytes())?;
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
                if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
                std::fs::write(path, blob.to_bytes())?;
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

fn read_blob(path: &Path) -> Option<EncodedBlob> {
    if !path.exists() || path.is_dir() { return None; }
    std::fs::read(path).ok().map(|b| EncodedBlob::from_bytes(&b))
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
}
