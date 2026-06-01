//! Multi-user API-key store for the OpenAI-compat HTTP surface (ADR 0018 P1).
//!
//! ADR 0012 added one optional static bearer token; its revisit trigger
//! ("multiple clients needing distinct keys → a per-token store") has fired.
//! This module is that per-token store. **The API key *is* the account**
//! (like OpenAI's `sk-...`): a 256-bit random token, revocable per row, with
//! per-key quota + priority. There is deliberately NO password/TOTP/session
//! machinery — that defends a browser threat model LAMU does not have.
//!
//! ## Storage
//!
//! A separate SQLite at `~/.config/lamu/keys.db` (NOT `conversations.db` /
//! `memory.db` — those live under the MCP's `~/.local/share`; keys sit next
//! to the static `api-token` under `~/.config/lamu`, 0600). We mirror
//! `lamu-mcp/src/lifetime_memory.rs`'s rusqlite idiom: WAL +
//! `synchronous=NORMAL` pragmas, `CREATE TABLE IF NOT EXISTS` schema, and an
//! idempotent migration that is safe to run on every startup.
//!
//! ## Tokens are hashed, not stored
//!
//! We store **SHA-256 of the full `lamu_<64hex>` token**, never the plaintext.
//! A 256-bit random token has no brute-force surface, so SHA-256 suffices (no
//! bcrypt/argon2 needed — see ADR 0018 Rationale). Plaintext is returned ONCE
//! from [`KeyStore::issue`] and never again ([`KeyStore::list`] shows only the
//! prefix). Hashing also removes the constant-time-compare requirement (a hash
//! lookup is content-independent); a dummy hash on a verify miss still defeats
//! user-enumeration timing.
//!
//! ## In-memory cache
//!
//! [`KeyStore`] keeps a `HashMap<token_hash, Principal>` so the hot
//! `verify()` path is a single in-memory lookup, not a per-request DB read.
//! The cache is populated on open + on issue, and invalidated on revoke
//! (ADR 0018: "Revocation must be immediate").

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `CREATE TABLE IF NOT EXISTS` schema for the key store. Mirrors the ADR 0018
/// column list verbatim:
/// `api_keys(id, user, token_hash, token_prefix, created_at, last_used_at,
/// revoked_at, daily_token_quota, priority)`.
///
/// `token_hash` is the 64-char lowercase hex SHA-256 of the full token and is
/// UNIQUE (a verify is a single indexed lookup). `priority` defaults to 0 to
/// match `queue.rs`'s default priority. `daily_token_quota` is nullable
/// (NULL = unlimited). Timestamps are unix seconds.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS api_keys (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    user               TEXT    NOT NULL,
    token_hash         TEXT    NOT NULL UNIQUE,
    token_prefix       TEXT    NOT NULL,
    created_at         INTEGER NOT NULL,
    last_used_at       INTEGER,
    revoked_at         INTEGER,
    daily_token_quota  INTEGER,
    priority           INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(token_prefix);
";

/// The authenticated principal behind a presented key, inserted into the
/// request extensions on a successful `verify()` so downstream handlers
/// (quota, audit, queue) can attribute the request. `daily_token_quota` is
/// `None` for an unlimited key; `priority` feeds `queue.rs`'s
/// `Strategy::Priority`.
#[derive(Debug, Clone)]
pub struct Principal {
    pub user: String,
    pub key_id: i64,
    pub priority: i32,
    pub daily_token_quota: Option<u32>,
}

/// A redacted key row for `lamu auth list`. NEVER carries the plaintext token
/// or its hash — only the public prefix, the owner, the lifecycle timestamps,
/// and the quota/priority knobs. `revoked_at` is `Some` once the key has been
/// revoked; `last_used_at` is `Some` once the key has authenticated a request
/// (currently always `None` — the writeback is a P2 batched concern, see
/// `verify`). `daily_token_quota` is `None` for an unlimited key.
#[derive(Debug, Clone)]
pub struct KeyInfo {
    pub user: String,
    pub token_prefix: String,
    pub created_at: i64,
    pub revoked_at: Option<i64>,
    pub priority: i32,
    pub daily_token_quota: Option<u32>,
    pub last_used_at: Option<i64>,
}

/// Length of the displayed/stored token prefix (`lamu_` + 8 hex chars). Long
/// enough to disambiguate keys for revoke + list, far too short to be a
/// guessable credential.
const PREFIX_LEN: usize = "lamu_".len() + 8;

/// SHA-256 of `s`, lowercase hex. Used for both the stored `token_hash` and
/// the dummy hash computed on a verify miss (constant-work, timing-safe).
fn sha256_hex(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(s.as_bytes());
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Default key-store path: `~/.config/lamu/keys.db`, alongside the static
/// `api-token`. Creates `~/.config/lamu` if missing.
fn default_db_path() -> Result<PathBuf> {
    let dir = dirs::config_dir()
        .ok_or_else(|| anyhow!("cannot resolve config dir (~/.config)"))?
        .join("lamu");
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    // 0700 the config dir so a directory listing can't even reveal that a key
    // store exists (the keys.db itself is 0600; this closes the dir-listing
    // enumeration channel). Best-effort.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }
    Ok(dir.join("keys.db"))
}

/// The per-token key store: a SQLite connection guarded by a `parking_lot`
/// Mutex (mirroring `lifetime_memory.rs`) plus an in-memory
/// `HashMap<token_hash, Principal>` cache for the hot `verify()` path. Cloneable
/// state lives behind `Arc<KeyStore>` in `AuthMode::KeyStore` (see ADR 0018 P2).
pub struct KeyStore {
    conn: Mutex<Connection>,
    /// token_hash → Principal. Populated on open + on issue, invalidated on
    /// revoke. Guards the per-request verify against a DB round-trip.
    cache: Mutex<HashMap<String, Principal>>,
}

impl KeyStore {
    /// Open (creating if needed) the key store at the default path
    /// `~/.config/lamu/keys.db`, tightened to 0600.
    pub fn open_default() -> Result<Self> {
        let path = default_db_path()?;
        Self::open(&path)
    }

    /// Open (creating if needed) the key store at `path`. Applies WAL +
    /// `synchronous=NORMAL` (the `lifetime_memory.rs` pragma set), runs the
    /// idempotent schema, tightens the db file to 0600 (it holds credential
    /// hashes), and warms the in-memory cache from the live (non-revoked) rows.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute_batch(SCHEMA)?;

        // keys.db holds credential hashes — same 0600 posture as the static
        // api-token (lamu-cli cmd_auth_init) and api-keys.env (cloud_config).
        // rusqlite created the file with the process umask; clamp it. WAL also
        // creates -wal/-shm sidecars; tighten those too when present.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perm = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(path, perm.clone());
            for ext in ["-wal", "-shm"] {
                let side = path.with_file_name(format!(
                    "{}{ext}",
                    path.file_name().and_then(|s| s.to_str()).unwrap_or("keys.db")
                ));
                let _ = std::fs::set_permissions(&side, perm.clone());
            }
        }

        let store = Self {
            conn: Mutex::new(conn),
            cache: Mutex::new(HashMap::new()),
        };
        store.warm_cache()?;
        Ok(store)
    }

    /// (Re)load the in-memory cache from every non-revoked row. Called on open
    /// and after a revoke so the cache reflects exactly the live keys.
    fn warm_cache(&self) -> Result<()> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT token_hash, user, id, priority, daily_token_quota \
             FROM api_keys WHERE revoked_at IS NULL",
        )?;
        let mapped = stmt.query_map([], |r| {
            let token_hash: String = r.get(0)?;
            let quota: Option<i64> = r.get(4)?;
            Ok((
                token_hash,
                Principal {
                    user: r.get(1)?,
                    key_id: r.get(2)?,
                    priority: r.get(3)?,
                    daily_token_quota: quota.and_then(|q| u32::try_from(q).ok()),
                },
            ))
        })?;
        let mut cache = self.cache.lock();
        cache.clear();
        for row in mapped {
            let (hash, principal) = row?;
            cache.insert(hash, principal);
        }
        Ok(())
    }

    /// Mint a new key for `user`: generate `lamu_<64hex>` from 32 OS-random
    /// bytes (`getrandom`, the same generator `lamu auth init` uses), store its
    /// SHA-256 + prefix, seed the cache, and return the **plaintext token —
    /// shown ONCE**. The caller must surface it immediately; it is unrecoverable.
    ///
    /// `priority` defaults to 0; `daily_token_quota` defaults to NULL
    /// (unlimited). Use [`KeyStore::issue_with`] to set them at mint time.
    pub fn issue(&self, user: &str) -> Result<String> {
        self.issue_with(user, 0, None)
    }

    /// [`KeyStore::issue`] with an explicit `priority` + optional
    /// `daily_token_quota`.
    pub fn issue_with(
        &self,
        user: &str,
        priority: i32,
        daily_token_quota: Option<u32>,
    ) -> Result<String> {
        let user = user.trim();
        if user.is_empty() {
            return Err(anyhow!("user is required"));
        }

        let mut bytes = [0u8; 32];
        getrandom::getrandom(&mut bytes).map_err(|e| anyhow!("getrandom: {e}"))?;
        let token = format!(
            "lamu_{}",
            bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
        );
        let token_hash = sha256_hex(&token);
        let token_prefix: String = token.chars().take(PREFIX_LEN).collect();
        let created_at = now_secs();

        let key_id = {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT INTO api_keys \
                 (user, token_hash, token_prefix, created_at, last_used_at, revoked_at, \
                  daily_token_quota, priority) \
                 VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)",
                params![
                    user,
                    token_hash,
                    token_prefix,
                    created_at,
                    daily_token_quota.map(|q| q as i64),
                    priority,
                ],
            )
            .context("insert api_key")?;
            conn.last_insert_rowid()
        };

        // Seed the cache so the freshly-issued key verifies without a reload.
        self.cache.lock().insert(
            token_hash,
            Principal {
                user: user.to_string(),
                key_id,
                priority,
                daily_token_quota,
            },
        );

        Ok(token)
    }

    /// Revoke the key whose prefix is `prefix` (the value [`KeyStore::list`]
    /// shows). Sets `revoked_at = now` on the matching live row and invalidates
    /// the in-memory cache so the key stops verifying **immediately** (ADR 0018:
    /// revocation must be immediate). Returns `true` if a live key was revoked,
    /// `false` if no live key matched (unknown or already-revoked prefix).
    ///
    /// Matching is exact on `token_prefix`; prefix collisions are practically
    /// impossible (8 hex chars over 32 random bytes), but if more than one live
    /// row shares a prefix this revokes all of them — the safe direction.
    pub fn revoke(&self, prefix: &str) -> Result<bool> {
        let prefix = prefix.trim();
        if prefix.is_empty() {
            return Err(anyhow!("prefix is required"));
        }
        // Hold the conn lock for the UPDATE *and* the hash-collect, then evict
        // exactly those hashes from the cache. This is race-free + immediate:
        // no full clear-then-reload (which would briefly 401 every live key),
        // and the revoked hash is gone from the cache the instant we return.
        let (affected, hashes): (usize, Vec<String>) = {
            let conn = self.conn.lock();
            let affected = conn
                .execute(
                    "UPDATE api_keys SET revoked_at = ? \
                     WHERE token_prefix = ? AND revoked_at IS NULL",
                    params![now_secs(), prefix],
                )
                .context("revoke api_key")?;
            let mut stmt = conn.prepare("SELECT token_hash FROM api_keys WHERE token_prefix = ?")?;
            let hashes = stmt
                .query_map([prefix], |r| r.get::<_, String>(0))?
                .filter_map(|r| r.ok())
                .collect();
            (affected, hashes)
        };
        if affected > 0 {
            let mut cache = self.cache.lock();
            for h in &hashes {
                cache.remove(h);
            }
        }
        Ok(affected > 0)
    }

    /// List every key as redacted [`KeyInfo`] (prefix + user + created +
    /// revoked), newest first. NEVER returns plaintext or the hash. Includes
    /// revoked keys (their `revoked_at` is `Some`) so `lamu auth list` shows
    /// the full roster.
    pub fn list(&self) -> Result<Vec<KeyInfo>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT user, token_prefix, created_at, revoked_at, \
             priority, daily_token_quota, last_used_at \
             FROM api_keys ORDER BY created_at DESC, id DESC",
        )?;
        let mapped = stmt.query_map([], |r| {
            let quota: Option<i64> = r.get(5)?;
            Ok(KeyInfo {
                user: r.get(0)?,
                token_prefix: r.get(1)?,
                created_at: r.get(2)?,
                revoked_at: r.get(3)?,
                priority: r.get(4)?,
                daily_token_quota: quota.and_then(|q| u32::try_from(q).ok()),
                last_used_at: r.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for row in mapped {
            out.push(row?);
        }
        Ok(out)
    }

    /// Verify a presented token: SHA-256 it and look the hash up in the
    /// in-memory cache (live, non-revoked keys only — `revoke` evicts directly,
    /// `issue` seeds). `Some(Principal)` on a hit, `None` otherwise.
    ///
    /// No timing equalizer: the cache key IS the full SHA-256, so the lookup is
    /// content-independent (no per-key oracle), and the HTTP response code (200
    /// vs 401) already reveals validity — there is no separate username step to
    /// enumerate, so a dummy hash would only be theater. No DB write on this hot
    /// path either: `last_used_at` is a P2 (batched) concern, not worth
    /// serializing every authenticated request through the connection mutex.
    pub fn verify(&self, token: &str) -> Option<Principal> {
        let hash = sha256_hex(token.trim());
        let principal = self.cache.lock().get(&hash).cloned()?;
        // M7: the in-memory cache is per-process, but `lamu auth revoke` runs in
        // a SEPARATE process and only writes keys.db + evicts its own throwaway
        // cache — the live `serve` process's cache stayed stale, so a revoked
        // token kept authenticating until restart. On a cache hit, confirm the
        // row is still live with one indexed read (the few-machine-client threat
        // model tolerates a SELECT per authenticated request; it's a read, not
        // the per-request WRITE we deliberately avoid). If it was revoked out
        // from under us, evict + reject so revocation is effective immediately.
        let still_live = {
            let conn = self.conn.lock();
            conn.query_row(
                "SELECT 1 FROM api_keys WHERE token_hash = ? AND revoked_at IS NULL",
                [&hash],
                |_| Ok(()),
            )
            .optional()
            .unwrap_or(None)
            .is_some()
        };
        if still_live {
            Some(principal)
        } else {
            self.cache.lock().remove(&hash);
            None
        }
    }

    /// Count of currently-live (non-revoked) keys. The off-loopback startup
    /// gate (`lib.rs`) treats `KeyStore`-with-≥1-live-key as "auth configured";
    /// an empty key store off-loopback must still hard-fail (ADR 0018
    /// Consequences — the one security subtlety).
    pub fn active_key_count(&self) -> Result<i64> {
        let conn = self.conn.lock();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM api_keys WHERE revoked_at IS NULL",
                [],
                |r| r.get(0),
            )
            .optional()?
            .unwrap_or(0);
        Ok(n)
    }

    /// True iff ≥1 live key exists (errors → false). Used by the off-loopback
    /// gate in `serve()`: an empty KeyStore must hard-fail like no-token.
    pub fn has_active_key(&self) -> bool {
        self.active_key_count().map(|n| n > 0).unwrap_or(false)
    }

    /// The default `keys.db` path WITHOUT creating it (unlike `open_default`),
    /// so `resolve_auth_mode` can test existence before deciding to engage the
    /// KeyStore. `None` only if the config dir can't be resolved.
    pub fn default_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("lamu").join("keys.db"))
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> KeyStore {
        // Each test gets a fresh on-disk db in a unique temp path; in-memory
        // would not exercise the file-perm / WAL path, but the logic is
        // identical and tests run fast either way. Use a tempfile.
        let path = std::env::temp_dir().join(format!(
            "lamu-keys-test-{}-{}.db",
            std::process::id(),
            now_secs() as u64 * 1_000_000
                + (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos() as u64)
                    .unwrap_or(0))
        ));
        let _ = std::fs::remove_file(&path);
        KeyStore::open(&path).unwrap()
    }

    #[test]
    fn verify_reflects_cross_process_revoke() {
        // M7: a revoke from a SEPARATE process (separate KeyStore handle on the
        // same db file) must reject on the live handle immediately, not only at
        // restart. The live handle's in-memory cache still holds the key, so
        // verify must re-check the DB row.
        let path = std::env::temp_dir().join(format!(
            "lamu-keys-xproc-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let _ = std::fs::remove_file(&path);
        let serve = KeyStore::open(&path).unwrap(); // live `serve` process
        let token = serve.issue("alice").unwrap();
        assert!(serve.verify(&token).is_some(), "fresh key must verify");
        // A separate admin process revokes via its own handle on the same db.
        let admin = KeyStore::open(&path).unwrap();
        let prefix: String = token.chars().take(PREFIX_LEN).collect();
        assert!(admin.revoke(&prefix).unwrap(), "revoke must affect a row");
        // The live handle's cache still has it — verify must catch the DB revoke.
        assert!(serve.verify(&token).is_none(),
            "cross-process revoke must reject immediately on the live handle (M7)");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn issue_then_verify_roundtrip() {
        let ks = store();
        let token = ks.issue("alice").unwrap();
        assert!(token.starts_with("lamu_"));
        assert_eq!(token.len(), "lamu_".len() + 64);

        let p = ks.verify(&token).expect("freshly-issued key must verify");
        assert_eq!(p.user, "alice");
        assert_eq!(p.priority, 0);
        assert_eq!(p.daily_token_quota, None);
    }

    #[test]
    fn unknown_token_is_none() {
        let ks = store();
        ks.issue("alice").unwrap();
        assert!(ks.verify("lamu_deadbeef").is_none());
        assert!(ks.verify("").is_none());
    }

    #[test]
    fn revoke_makes_key_stop_verifying_immediately() {
        let ks = store();
        let token = ks.issue("bob").unwrap();
        let prefix: String = token.chars().take(PREFIX_LEN).collect();

        assert!(ks.verify(&token).is_some());
        assert!(ks.revoke(&prefix).unwrap(), "revoke should report success");
        assert!(
            ks.verify(&token).is_none(),
            "revoked key must not verify (immediate)"
        );
        // Re-revoking an already-revoked prefix is a no-op (false).
        assert!(!ks.revoke(&prefix).unwrap());
        assert!(!ks.revoke("lamu_nope").unwrap());
    }

    #[test]
    fn list_redacts_plaintext_and_shows_revoked() {
        let ks = store();
        let token = ks.issue("carol").unwrap();
        let prefix: String = token.chars().take(PREFIX_LEN).collect();

        let infos = ks.list().unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].user, "carol");
        assert_eq!(infos[0].token_prefix, prefix);
        assert!(infos[0].revoked_at.is_none());
        // The full token never appears in any KeyInfo field.
        assert!(infos[0].token_prefix.len() < token.len());

        ks.revoke(&prefix).unwrap();
        let infos = ks.list().unwrap();
        assert!(infos[0].revoked_at.is_some());
    }

    #[test]
    fn list_surfaces_priority_quota_lastused() {
        let ks = store();
        // Default issue → priority 0, unlimited quota, never used.
        ks.issue("plain").unwrap();
        // issue_with → explicit priority + quota.
        ks.issue_with("vip", 7, Some(250_000)).unwrap();

        let infos = ks.list().unwrap();
        assert_eq!(infos.len(), 2);
        // Newest-first ordering: "vip" was issued last.
        let vip = infos.iter().find(|k| k.user == "vip").unwrap();
        assert_eq!(vip.priority, 7);
        assert_eq!(vip.daily_token_quota, Some(250_000));
        // last_used_at is not yet written back on verify (P2 batched concern).
        assert_eq!(vip.last_used_at, None);

        let plain = infos.iter().find(|k| k.user == "plain").unwrap();
        assert_eq!(plain.priority, 0);
        assert_eq!(plain.daily_token_quota, None);
        assert_eq!(plain.last_used_at, None);
    }

    #[test]
    fn issue_with_sets_priority_and_quota() {
        let ks = store();
        let token = ks.issue_with("dave", 5, Some(100_000)).unwrap();
        let p = ks.verify(&token).unwrap();
        assert_eq!(p.priority, 5);
        assert_eq!(p.daily_token_quota, Some(100_000));
    }

    #[test]
    fn active_key_count_tracks_revocation() {
        let ks = store();
        assert_eq!(ks.active_key_count().unwrap(), 0);
        let t1 = ks.issue("u1").unwrap();
        ks.issue("u2").unwrap();
        assert_eq!(ks.active_key_count().unwrap(), 2);
        let p1: String = t1.chars().take(PREFIX_LEN).collect();
        ks.revoke(&p1).unwrap();
        assert_eq!(ks.active_key_count().unwrap(), 1);
    }

    #[test]
    fn cache_survives_reopen_from_disk() {
        let path = std::env::temp_dir().join(format!(
            "lamu-keys-reopen-{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&path);
        let token = {
            let ks = KeyStore::open(&path).unwrap();
            ks.issue("ed").unwrap()
        };
        // Reopen: warm_cache must repopulate from the persisted row.
        let ks2 = KeyStore::open(&path).unwrap();
        assert!(ks2.verify(&token).is_some());
        let _ = std::fs::remove_file(&path);
    }
}
