//! `lamu clean` — retention for LAMU's accumulating artifacts.
//!
//! Categories (all under `~/.local/share/lamu/`): research drafts, sandbox
//! sessions (snapshot + untracked archive deleted as a PAIR), chat
//! transcripts, media outputs (images + tts), backend startup logs, and —
//! explicit-only, NOT part of `--all` — the legacy pre-ADR-0028 SQLite
//! stores (`--legacy-dbs`: `conversations.db` / `memory.db` /
//! `embeddings.db` + their `-wal`/`-shm` sidecars). Those are one-time
//! import sources left in place by the unified-`lamu.db` migration; the
//! category is offered ONLY when `lamu.db` exists (i.e. the import
//! happened), so the data they hold is already in the live store.
//!
//! Hard exclusions — never candidates, by construction (the walker only
//! enters the category subdirs / the explicit legacy filenames): the live
//! registry (`models.yaml`, ADR 0025), `scheduler.lock`, the LIVE SQLite
//! store (`lamu.db*`) and the persistent vector indexes (`index/`, ADR
//! 0031) — live state, not retention fodder; their sizes are shown in the
//! report — `train-data/`/`train-jobs/`, and everything in
//! `~/.config/lamu/`.
//!
//! Safety model: DRY-RUN BY DEFAULT — without `--yes` nothing is deleted,
//! you get the would-delete report. Symlinks are never followed (the entry
//! itself is skipped), and every candidate must canonicalize to a path
//! inside the category dir (the `cmd_rm` confinement pattern).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone)]
pub struct CleanOpts {
    pub drafts: bool,
    pub sessions: bool,
    pub conversations: bool,
    pub media: bool,
    pub logs: bool,
    /// Legacy pre-ADR-0028 SQLite stores (conversations.db / memory.db /
    /// embeddings.db + -wal/-shm sidecars). EXPLICIT-ONLY: deliberately
    /// not included in `--all` (deleting databases deserves its own
    /// flag), and only offered when the unified `lamu.db` exists.
    pub legacy_dbs: bool,
    pub all: bool,
    /// Delete files older than this many days (0 = age alone deletes nothing).
    pub keep_days: u64,
    /// Always keep the newest N per category (0 = no count floor).
    pub keep_count: usize,
    /// Per-category size budget; oldest deleted until under it (0 = unlimited).
    pub max_size_mb: u64,
    /// Actually delete. Without it: dry-run report only.
    pub yes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Category {
    Drafts,
    Sessions,
    Conversations,
    Media,
    Logs,
    LegacyDbs,
}

impl Category {
    fn label(self) -> &'static str {
        match self {
            Category::Drafts => "drafts",
            Category::Sessions => "sessions",
            Category::Conversations => "conversations",
            Category::Media => "media",
            Category::Logs => "logs",
            Category::LegacyDbs => "legacy-dbs",
        }
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    path: PathBuf,
    /// Companions deleted with `path` (a session's untracked archive; a
    /// legacy db's -wal/-shm sidecars).
    twins: Vec<PathBuf>,
    mtime: SystemTime,
    size: u64,
}

/// LAMU's data root (`~/.local/share/lamu`). The clean targets the DEFAULT
/// locations only — e.g. a custom `$LAMU_RESEARCH_DIR` is deliberately out
/// of scope (the operator pointed drafts elsewhere on purpose).
fn data_root() -> anyhow::Result<PathBuf> {
    dirs::data_dir()
        .map(|d| d.join("lamu"))
        .ok_or_else(|| anyhow::anyhow!("no user data dir resolvable — refusing to clean"))
}

pub fn cmd_clean(opts: CleanOpts) -> anyhow::Result<()> {
    let root = data_root()?;
    let mut cats: Vec<Category> = Vec::new();
    let pick = |on: bool, c: Category, v: &mut Vec<Category>| {
        if on || opts.all {
            v.push(c)
        }
    };
    pick(opts.drafts, Category::Drafts, &mut cats);
    pick(opts.sessions, Category::Sessions, &mut cats);
    pick(opts.conversations, Category::Conversations, &mut cats);
    pick(opts.media, Category::Media, &mut cats);
    pick(opts.logs, Category::Logs, &mut cats);
    // EXPLICIT-ONLY: legacy DBs are never swept up by --all — deleting
    // databases (even import-source leftovers) deserves its own flag.
    if opts.legacy_dbs {
        cats.push(Category::LegacyDbs);
    }
    if cats.is_empty() {
        anyhow::bail!(
            "nothing selected — pass one or more of --drafts --sessions --conversations \
             --media --logs --legacy-dbs, or --all"
        );
    }

    let mode = if opts.yes { "DELETING" } else { "dry-run (pass --yes to delete)" };
    println!("lamu clean — {mode}");
    println!("root: {}", root.display());
    println!(
        "retention: keep-days={} keep-count={} max-size-mb={}",
        opts.keep_days, opts.keep_count, opts.max_size_mb
    );

    let mut grand_files = 0usize;
    let mut grand_bytes = 0u64;
    for cat in cats {
        let cands = collect(&root, cat);
        let doomed = apply_retention(cands, opts.keep_days, opts.keep_count, opts.max_size_mb);
        let bytes: u64 = doomed.iter().map(|c| c.size).sum();
        let cat_files: usize = doomed.iter().map(|c| 1 + c.twins.len()).sum();
        println!(
            "  {:<14} {:>5} file(s)  {:>9.2} MiB",
            cat.label(),
            cat_files,
            bytes as f64 / (1024.0 * 1024.0)
        );
        for c in &doomed {
            println!("    {}", c.path.display());
            for t in &c.twins {
                println!("    {}", t.display());
            }
            if opts.yes {
                remove_confined(&root, &c.path);
                for t in &c.twins {
                    remove_confined(&root, t);
                }
            }
        }
        grand_files += cat_files;
        grand_bytes += bytes;
    }

    println!(
        "{}: {} file(s), {:.2} MiB",
        if opts.yes { "deleted" } else { "would delete" },
        grand_files,
        grand_bytes as f64 / (1024.0 * 1024.0)
    );

    // Live state the clean never touches — report sizes so the operator
    // knows where the rest of the disk went. The legacy DBs are listed
    // here too (when not selected) since they predate --legacy-dbs.
    for db in ["conversations.db", "memory.db"] {
        if let Ok(m) = fs::metadata(root.join(db)) {
            println!(
                "  (not managed: {} — {:.2} MiB of live state)",
                db,
                m.len() as f64 / (1024.0 * 1024.0)
            );
        }
    }
    // ADR 0028/0031 artifacts: the unified lamu.db and the persistent
    // vector indexes under index/. NEVER deletable here — live state —
    // but their sizes belong in the report.
    if let Ok(m) = fs::metadata(root.join("lamu.db")) {
        println!(
            "  (not managed: lamu.db — {:.2} MiB of live state)",
            m.len() as f64 / (1024.0 * 1024.0)
        );
    }
    let index_bytes = dir_size(&root.join("index"));
    if index_bytes > 0 {
        println!(
            "  (not managed: index/ — {:.2} MiB of live state)",
            index_bytes as f64 / (1024.0 * 1024.0)
        );
    }
    Ok(())
}

/// Total size of the regular files directly inside `dir` (the index/
/// layout is flat: <store>.tv / .ids / .meta.json). Symlinks skipped;
/// missing dir → 0.
fn dir_size(dir: &Path) -> u64 {
    let Ok(rd) = fs::read_dir(dir) else { return 0 };
    rd.flatten()
        .filter_map(|e| fs::symlink_metadata(e.path()).ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// Enumerate a category's deletable files. Only regular files inside the
/// category's own subdir(s); symlinks skipped.
fn collect(root: &Path, cat: Category) -> Vec<Candidate> {
    match cat {
        Category::Drafts => files_in(&root.join("research"), &["md"]),
        Category::Conversations => files_in(&root.join("conversations"), &["md"]),
        Category::Media => {
            let mut v = files_in(&root.join("images"), &["png", "jpg", "jpeg", "webp"]);
            v.extend(files_in(&root.join("tts"), &["mp3", "wav", "pcm", "opus"]));
            v
        }
        Category::Logs => {
            // Backend startup logs are CO-LOCATED with their outputs by the
            // backends themselves (lamu-image writes comfyui-{port}.log into
            // images/, lamu-tts writes fish-speech-{port}.log into tts/) —
            // there is no separate logs/ dir.
            let mut v = files_in(&root.join("images"), &["log"]);
            v.extend(files_in(&root.join("tts"), &["log"]));
            v
        }
        Category::Sessions => {
            // Snapshot toml + its untracked archive form one unit: `lamu
            // undo` resolves archives via the snapshot, so they live and
            // die together. The archive path is constructed directly from
            // the snapshot stem (deliberately NOT a second files_in pass —
            // an archive is only ever deleted via its snapshot).
            let sessions = root.join("sandbox").join("sessions");
            let untracked = root.join("sandbox").join("untracked");
            files_in(&sessions, &["toml"])
                .into_iter()
                .map(|mut c| {
                    if let Some(stem) = c.path.file_stem().and_then(|s| s.to_str()) {
                        let arc = untracked.join(format!("{stem}.tar.zst"));
                        if arc.is_file() {
                            c.size += fs::metadata(&arc).map(|m| m.len()).unwrap_or(0);
                            c.twins.push(arc);
                        }
                    }
                    c
                })
                .collect()
        }
        Category::LegacyDbs => collect_legacy_dbs(root),
    }
}

/// The pre-ADR-0028 standalone stores, deletable ONLY once the unified
/// `lamu.db` exists (i.e. their one-time import has run — before that
/// they ARE the live data and must never be candidates). Each db is
/// paired with whatever `-wal`/`-shm` sidecars exist, deleted as a unit
/// (a dangling sidecar would resurrect stale pages on a re-open).
fn collect_legacy_dbs(root: &Path) -> Vec<Candidate> {
    if !root.join("lamu.db").is_file() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for db in ["conversations.db", "memory.db", "embeddings.db"] {
        let path = root.join(db);
        // symlink_metadata (not metadata): never follow a planted link.
        let Ok(meta) = fs::symlink_metadata(&path) else { continue };
        if !meta.is_file() {
            continue;
        }
        let mut size = meta.len();
        let mut twins = Vec::new();
        for suffix in ["-wal", "-shm"] {
            let side = root.join(format!("{db}{suffix}"));
            if let Ok(m) = fs::symlink_metadata(&side) {
                if m.is_file() {
                    size += m.len();
                    twins.push(side);
                }
            }
        }
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        out.push(Candidate { path, twins, mtime, size });
    }
    out
}

fn files_in(dir: &Path, exts: &[&str]) -> Vec<Candidate> {
    let Ok(rd) = fs::read_dir(dir) else { return Vec::new() };
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        // Never follow symlinks — a planted link must not reach outside.
        let Ok(meta) = fs::symlink_metadata(&path) else { continue };
        if !meta.is_file() {
            continue;
        }
        let ext_ok = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| exts.iter().any(|x| e.eq_ignore_ascii_case(x)));
        if !ext_ok {
            continue;
        }
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        out.push(Candidate { path, twins: Vec::new(), mtime, size: meta.len() });
    }
    out
}

/// Decide what to delete. Newest-first ordering; then:
///   1. The newest `keep_count` are protected outright.
///   2. Unprotected files older than `keep_days` are deleted (0 = skip).
///   3. If the survivors still exceed `max_size_mb`, delete oldest-first
///      among the unprotected until under budget (0 = skip).
fn apply_retention(
    mut cands: Vec<Candidate>,
    keep_days: u64,
    keep_count: usize,
    max_size_mb: u64,
) -> Vec<Candidate> {
    cands.sort_by(|a, b| b.mtime.cmp(&a.mtime)); // newest first
    let cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(keep_days.saturating_mul(86_400)))
        .unwrap_or(SystemTime::UNIX_EPOCH);

    let mut doomed: Vec<Candidate> = Vec::new();
    let mut kept: Vec<Candidate> = Vec::new();
    for (i, c) in cands.into_iter().enumerate() {
        let protected = i < keep_count;
        if !protected && keep_days > 0 && c.mtime < cutoff {
            doomed.push(c);
        } else {
            kept.push(c);
        }
    }

    if max_size_mb > 0 {
        let budget = max_size_mb.saturating_mul(1024 * 1024);
        let mut total: u64 = kept.iter().map(|c| c.size).sum();
        // kept is newest-first; pop from the back = oldest-first deletion.
        while total > budget && kept.len() > keep_count {
            let c = kept.pop().expect("len > keep_count >= 0");
            total -= c.size;
            doomed.push(c);
        }
    }
    doomed
}

/// Delete a file only if it canonicalizes INSIDE the data root (symlink-swap
/// defense, mirrors `cmd_rm`). Errors are reported, never fatal — clean is
/// best-effort.
fn remove_confined(root: &Path, path: &Path) {
    let Ok(root_c) = root.canonicalize() else {
        eprintln!("    !! skip {}: data root unresolvable", path.display());
        return;
    };
    // Canonicalize the PARENT (the file itself may be a dangling symlink —
    // those were already filtered, but a TOCTOU swap lands here).
    let parent_ok = path
        .parent()
        .and_then(|p| p.canonicalize().ok())
        .map(|p| p.starts_with(&root_c))
        .unwrap_or(false);
    if !parent_ok {
        eprintln!("    !! skip {}: escapes the data root", path.display());
        return;
    }
    if let Err(e) = fs::remove_file(path) {
        eprintln!("    !! {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn touch(dir: &Path, name: &str, age_days: u64, bytes: usize) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, vec![b'x'; bytes]).unwrap();
        let mtime = SystemTime::now() - Duration::from_secs(age_days * 86_400);
        let ft = filetime::FileTime::from_system_time(mtime);
        filetime::set_file_mtime(&p, ft).unwrap();
        p
    }

    fn names(v: &[Candidate]) -> Vec<String> {
        v.iter()
            .map(|c| c.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn retention_age_and_count_floor() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "new.md", 1, 10);
        touch(dir.path(), "mid.md", 10, 10);
        touch(dir.path(), "old.md", 40, 10);
        touch(dir.path(), "ancient.md", 90, 10);
        let cands = files_in(dir.path(), &["md"]);
        assert_eq!(cands.len(), 4);

        // keep-days=30: the two older than 30d go.
        let doomed = apply_retention(cands.clone(), 30, 0, 0);
        let mut n = names(&doomed);
        n.sort();
        assert_eq!(n, ["ancient.md", "old.md"]);

        // keep-count=3 protects the newest 3 even past the age cutoff.
        let doomed = apply_retention(cands.clone(), 30, 3, 0);
        assert_eq!(names(&doomed), ["ancient.md"]);

        // keep-days=0: age alone deletes nothing.
        assert!(apply_retention(cands, 0, 0, 0).is_empty());
    }

    #[test]
    fn retention_size_budget_drops_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "a-new.md", 1, 600 * 1024);
        touch(dir.path(), "b-mid.md", 5, 600 * 1024);
        touch(dir.path(), "c-old.md", 9, 600 * 1024);
        let cands = files_in(dir.path(), &["md"]);
        // Budget 1 MiB against ~1.76 MiB total: dropping c-old leaves
        // ~1.17 MiB (still over), so b-mid goes too — oldest-first order.
        let doomed = apply_retention(cands.clone(), 0, 0, 1);
        assert_eq!(names(&doomed), ["c-old.md", "b-mid.md"], "oldest-first until under budget");
        // A 2 MiB budget already fits — nothing deleted.
        assert!(apply_retention(cands, 0, 0, 2).is_empty());
    }

    #[test]
    fn files_in_skips_symlinks_and_wrong_extensions() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "keep.md", 1, 4);
        touch(dir.path(), "skip.txt", 1, 4);
        #[cfg(unix)]
        std::os::unix::fs::symlink("/etc/hostname", dir.path().join("evil.md")).unwrap();
        let cands = files_in(dir.path(), &["md"]);
        assert_eq!(names(&cands), ["keep.md"]);
    }

    #[test]
    fn sessions_pair_snapshot_with_archive() {
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sandbox/sessions");
        let untracked = root.path().join("sandbox/untracked");
        fs::create_dir_all(&sessions).unwrap();
        fs::create_dir_all(&untracked).unwrap();
        touch(&sessions, "s1.toml", 40, 10);
        fs::write(untracked.join("s1.tar.zst"), b"zzzz").unwrap();
        touch(&sessions, "s2.toml", 40, 10); // no archive
        let cands = collect(root.path(), Category::Sessions);
        let s1 = cands.iter().find(|c| c.path.ends_with("s1.toml")).unwrap();
        assert_eq!(s1.twins.len(), 1);
        assert!(s1.twins[0].ends_with("s1.tar.zst"));
        assert_eq!(s1.size, 14, "archive bytes counted with the snapshot");
        let s2 = cands.iter().find(|c| c.path.ends_with("s2.toml")).unwrap();
        assert!(s2.twins.is_empty());
    }

    #[test]
    fn legacy_dbs_gated_on_lamu_db_existence() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("conversations.db"), b"legacy").unwrap();
        fs::write(root.path().join("memory.db"), b"legacy").unwrap();
        // No lamu.db → the import never ran → these ARE the live data:
        // the category must offer NOTHING.
        assert!(
            collect(root.path(), Category::LegacyDbs).is_empty(),
            "legacy dbs must not be candidates before the lamu.db import"
        );
        // lamu.db present → import happened → both legacy dbs are offered.
        fs::write(root.path().join("lamu.db"), b"unified").unwrap();
        let cands = collect(root.path(), Category::LegacyDbs);
        let names: Vec<String> = cands
            .iter()
            .map(|c| c.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"conversations.db".to_string()));
        assert!(names.contains(&"memory.db".to_string()));
        // lamu.db itself is NEVER a candidate.
        assert!(!names.contains(&"lamu.db".to_string()));
    }

    #[test]
    fn legacy_dbs_pair_wal_and_shm_sidecars() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("lamu.db"), b"unified").unwrap();
        fs::write(root.path().join("memory.db"), vec![b'x'; 100]).unwrap();
        fs::write(root.path().join("memory.db-wal"), vec![b'w'; 30]).unwrap();
        fs::write(root.path().join("memory.db-shm"), vec![b's'; 20]).unwrap();
        fs::write(root.path().join("embeddings.db"), vec![b'e'; 50]).unwrap(); // no sidecars
        let cands = collect(root.path(), Category::LegacyDbs);

        let mem = cands.iter().find(|c| c.path.ends_with("memory.db")).unwrap();
        let twin_names: Vec<String> = mem
            .twins
            .iter()
            .map(|t| t.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(twin_names, ["memory.db-wal", "memory.db-shm"]);
        assert_eq!(mem.size, 150, "sidecar bytes counted with the db");

        let emb = cands.iter().find(|c| c.path.ends_with("embeddings.db")).unwrap();
        assert!(emb.twins.is_empty());
        assert_eq!(emb.size, 50);
    }

    #[test]
    fn remove_confined_refuses_escapes() {
        let root = tempfile::tempdir().unwrap();
        let inside = root.path().join("research");
        fs::create_dir_all(&inside).unwrap();
        let victim_dir = tempfile::tempdir().unwrap();
        let victim = victim_dir.path().join("precious.md");
        fs::write(&victim, b"data").unwrap();
        // A file whose parent is OUTSIDE the root must be skipped.
        remove_confined(root.path(), &victim);
        assert!(victim.exists(), "outside file must survive");
        // And a file genuinely inside is deleted.
        let f = inside.join("x.md");
        fs::write(&f, b"d").unwrap();
        remove_confined(root.path(), &f);
        assert!(!f.exists());
    }
}
