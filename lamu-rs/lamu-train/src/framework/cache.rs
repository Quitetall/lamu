//! Cache stub — full implementation lands commit 3.
//!
//! StageContext holds an `Arc<CacheHandle>` so stage authors can
//! consult / write the cache from inside `Stage::run`. v2 commit 2
//! ships only the type so the rest of the framework compiles; the
//! actual lookup + insert logic + `--shared-cache` flag arrive with
//! the executor in commit 3.

use std::path::PathBuf;

use crate::framework::artifact::ContentHash;

/// Per-job cache. Lives at `<job_dir>/_cache/<key:hex>/`. v2-3 adds
/// global cache lookup at `~/.local/share/lamu/train-cache/` driven
/// by the `--shared-cache` flag.
#[derive(Clone, Debug)]
pub struct CacheHandle {
    pub job_local: PathBuf,
    pub global: Option<PathBuf>,
}

impl CacheHandle {
    /// Construct a per-job cache handle. v2-3 adds `with_global`.
    pub fn job_local(path: PathBuf) -> Self {
        Self {
            job_local: path,
            global: None,
        }
    }

    /// Lookup is unimplemented at this point; commit 3 wires it.
    /// Returns `None` so callers in the meantime treat every key
    /// as a miss and rerun the stage.
    pub fn lookup(&self, _key: ContentHash) -> Option<PathBuf> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_local_constructs() {
        let h = CacheHandle::job_local(PathBuf::from("/tmp/job/_cache"));
        assert_eq!(h.job_local, PathBuf::from("/tmp/job/_cache"));
        assert!(h.global.is_none());
    }

    #[test]
    fn lookup_returns_none_pre_commit_3() {
        let h = CacheHandle::job_local(PathBuf::from("/tmp/_cache"));
        let key = ContentHash::of_bytes(b"any");
        assert!(h.lookup(key).is_none());
    }
}
