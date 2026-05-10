//! Stage output cache.
//!
//! Skips re-execution of a stage when the inputs + args + stage
//! identity match a previous run's cached output. Per-job by
//! default (lives at `<job_dir>/_cache/<key:hex>/output.json`); the
//! `--shared-cache` flag (commit 5) flips lookup to the global
//! cache at `~/.local/share/lamu/train-cache/` first, then job-local.
//!
//! Cache key formula:
//!
//! ```text
//! sha256(
//!   b"blut.cache.v1" ‖
//!   stage_name (as bytes) ‖
//!   stage_schema (LE u32) ‖
//!   input_content_hash (32 bytes) ‖
//!   canonical(args_json)
//! )
//! ```
//!
//! `canonical(args_json)` = serde_json with object keys sorted
//! lexicographically. Field reorder doesn't invalidate; rename
//! does (semantic change). Test-covered.
//!
//! What lives in `<key:hex>/`:
//!
//! - `output.json` — the `ErasedArtifact` JSON. Cheap to read.
//! - The artifact's payload files DO NOT live here. They live
//!   wherever the producing stage put them (typically
//!   `<job_dir>/stages/<idx>-<name>/`). Cache hit means "I know
//!   the output of this stage; here's the metadata"; the on-disk
//!   payload is content-addressed via the artifact's primary path
//!   so it's findable even across jobs.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::framework::artifact::ContentHash;
use crate::framework::stage::ErasedArtifact;

/// Per-job + (commit-5) global cache handle.
#[derive(Clone, Debug)]
pub struct CacheHandle {
    pub job_local: PathBuf,
    pub global: Option<PathBuf>,
}

impl CacheHandle {
    /// Construct a per-job cache handle. Commit 5 adds
    /// `with_global` for the `--shared-cache` flag.
    pub fn job_local(path: PathBuf) -> Self {
        Self {
            job_local: path,
            global: None,
        }
    }

    /// Compute the cache key for a stage invocation.
    pub fn key_for(
        stage_name: &str,
        stage_schema: u32,
        input_hash: ContentHash,
        args: &serde_json::Value,
    ) -> ContentHash {
        const VERSION_TAG: &[u8] = b"blut.cache.v1";
        let mut hasher = Sha256::new();
        hasher.update(VERSION_TAG);
        hasher.update([0u8]);
        hasher.update(stage_name.as_bytes());
        hasher.update([0u8]);
        hasher.update(stage_schema.to_le_bytes());
        hasher.update(input_hash.0);
        // Canonical form: sorted object keys. serde_json's default
        // is insertion-order; we re-emit through a sorted writer
        // so reordered Args fields don't invalidate the cache.
        let canon = canonical_json(args);
        hasher.update(canon.as_bytes());
        let arr: [u8; 32] = hasher.finalize().into();
        ContentHash(arr)
    }

    /// Look up a cached output. Returns the parsed
    /// `ErasedArtifact` if present, `None` if absent. I/O errors
    /// other than "file not found" are downgraded to None with a
    /// `tracing::warn` — a corrupt cache entry shouldn't break the
    /// run, just trigger a re-execution.
    pub fn lookup(&self, key: ContentHash) -> Option<CacheHit> {
        for base in self.search_order() {
            let path = base.join(key.to_hex()).join("output.json");
            match std::fs::read(&path) {
                Ok(body) => match serde_json::from_slice::<ErasedArtifact>(&body) {
                    Ok(art) => {
                        return Some(CacheHit {
                            artifact: art,
                            from_path: path,
                        });
                    }
                    Err(e) => {
                        tracing::warn!(
                            "cache: corrupt entry at {}: {e}; treating as miss",
                            path.display()
                        );
                    }
                },
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    tracing::warn!(
                        "cache: read {}: {e}; treating as miss",
                        path.display()
                    );
                }
            }
        }
        None
    }

    /// Insert an output for the given key. Atomic: writes to a
    /// sibling `.tmp.<pid>.<nanos>` and renames into place.
    pub fn insert(&self, key: ContentHash, output: &ErasedArtifact) -> std::io::Result<()> {
        let dir = self.write_target().join(key.to_hex());
        std::fs::create_dir_all(&dir)?;
        let dest = dir.join("output.json");
        let body = serde_json::to_vec_pretty(output).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("serialize cache entry: {e}"))
        })?;
        write_atomic(&dest, &body)
    }

    /// Search order for lookups: global first when `--shared-cache`
    /// promoted it, then job-local. Writes always go to
    /// `write_target` (job-local unless `--shared-cache`).
    fn search_order(&self) -> Vec<&Path> {
        let mut v = Vec::with_capacity(2);
        if let Some(g) = &self.global {
            v.push(g.as_path());
        }
        v.push(self.job_local.as_path());
        v
    }

    fn write_target(&self) -> &Path {
        // Commit 5 adds the --shared-cache promotion path. For now
        // writes always go to the job-local cache.
        &self.job_local
    }
}

#[derive(Debug)]
pub struct CacheHit {
    pub artifact: ErasedArtifact,
    pub from_path: PathBuf,
}

/// Produce a canonical JSON form: object keys sorted
/// lexicographically, recursively. Used as part of the cache key
/// so two args dicts with the same fields in different orders
/// hash identically.
fn canonical_json(value: &serde_json::Value) -> String {
    let canon = canonical_value(value);
    serde_json::to_string(&canon).unwrap_or_default()
}

fn canonical_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            // BTreeMap collects entries in lex-sorted key order.
            let mut sorted: std::collections::BTreeMap<String, serde_json::Value> =
                std::collections::BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), canonical_value(v));
            }
            // Round-trip back to serde_json::Map preserving the
            // sorted order (serde_json::Map is insertion-ordered;
            // the BTreeMap traversal gives us lex order).
            let mut out = serde_json::Map::new();
            for (k, v) in sorted {
                out.insert(k, v);
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(a) => {
            serde_json::Value::Array(a.iter().map(canonical_value).collect())
        }
        other => other.clone(),
    }
}

/// Serializable record used by the executor when writing the
/// cache. Currently identical to `ErasedArtifact`, but kept as a
/// distinct alias so commit 5's lru-prune metadata can extend
/// without touching every call site.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheRecord {
    pub artifact: ErasedArtifact,
}

fn write_atomic(dest: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stem = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "tmp".into());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = dest.with_file_name(format!(
        ".{stem}.tmp.{}.{nanos}",
        std::process::id()
    ));
    let mut f = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&tmp)?;
    f.write_all(bytes)?;
    let _ = f.sync_all();
    drop(f);
    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_erased(payload: serde_json::Value) -> ErasedArtifact {
        ErasedArtifact {
            kind: "test.kind".into(),
            schema: 1,
            payload,
        }
    }

    #[test]
    fn key_changes_on_stage_name_change() {
        let h = ContentHash::of_bytes(b"x");
        let a = serde_json::json!({});
        let k1 = CacheHandle::key_for("alpha", 1, h, &a);
        let k2 = CacheHandle::key_for("beta", 1, h, &a);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_schema_bump() {
        let h = ContentHash::of_bytes(b"x");
        let a = serde_json::json!({});
        let k1 = CacheHandle::key_for("s", 1, h, &a);
        let k2 = CacheHandle::key_for("s", 2, h, &a);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_input_hash_change() {
        let a = serde_json::json!({});
        let k1 = CacheHandle::key_for("s", 1, ContentHash::of_bytes(b"a"), &a);
        let k2 = CacheHandle::key_for("s", 1, ContentHash::of_bytes(b"b"), &a);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_invariant_under_args_field_order() {
        // Same fields, different order → same cache key. Critical
        // property: users shouldn't have to keep arg structs in
        // a specific order to hit the cache.
        let h = ContentHash::of_bytes(b"x");
        let a1 = serde_json::json!({"alpha": 1, "beta": 2});
        let a2 = serde_json::json!({"beta": 2, "alpha": 1});
        let k1 = CacheHandle::key_for("s", 1, h, &a1);
        let k2 = CacheHandle::key_for("s", 1, h, &a2);
        assert_eq!(k1, k2);
    }

    #[test]
    fn key_changes_on_args_value_change() {
        let h = ContentHash::of_bytes(b"x");
        let a1 = serde_json::json!({"alpha": 1});
        let a2 = serde_json::json!({"alpha": 2});
        let k1 = CacheHandle::key_for("s", 1, h, &a1);
        let k2 = CacheHandle::key_for("s", 1, h, &a2);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_handles_nested_object_canonical_order() {
        let h = ContentHash::of_bytes(b"x");
        let a1 = serde_json::json!({"outer": {"a": 1, "b": 2}});
        let a2 = serde_json::json!({"outer": {"b": 2, "a": 1}});
        let k1 = CacheHandle::key_for("s", 1, h, &a1);
        let k2 = CacheHandle::key_for("s", 1, h, &a2);
        assert_eq!(k1, k2);
    }

    #[test]
    fn lookup_returns_none_when_empty() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"missing");
        assert!(h.lookup(key).is_none());
    }

    #[test]
    fn insert_then_lookup_round_trip() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"k");
        let art = fake_erased(serde_json::json!({"n": 7}));
        h.insert(key, &art).unwrap();
        let hit = h.lookup(key).expect("should hit");
        assert_eq!(hit.artifact.kind, "test.kind");
        assert_eq!(hit.artifact.payload, serde_json::json!({"n": 7}));
    }

    #[test]
    fn lookup_returns_none_on_corrupt_entry() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"k");
        let dir = td.path().join(key.to_hex());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("output.json"), b"{not valid json").unwrap();
        assert!(h.lookup(key).is_none());
    }

    #[test]
    fn insert_creates_dir_atomically_no_tmp_remnants() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"k");
        h.insert(key, &fake_erased(serde_json::json!({}))).unwrap();
        let entries: Vec<_> = std::fs::read_dir(td.path().join(key.to_hex()))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .contains(".tmp.")
            })
            .collect();
        assert!(entries.is_empty(), "no tmp files should survive successful insert");
    }
}
