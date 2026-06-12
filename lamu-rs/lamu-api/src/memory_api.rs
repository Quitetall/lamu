//! Memory-as-a-service HTTP routes (ADR 0032).
//!
//! Exposes lamu-memory's temporal fact store over `lamu serve` so an
//! external agent harness (katana) consumes LAMU memory as an
//! out-of-process extension. Four JSON-in/JSON-out POST routes under
//! `/v1/memory/*`, all registered INSIDE the `require_bearer` layer:
//!
//!   * `POST /v1/memory/remember`  {text, kind?, source?} → {id}
//!   * `POST /v1/memory/recall`    {query, k?, include_expired?} → {hits}
//!   * `POST /v1/memory/forget`    {id} → {forgotten}
//!   * `POST /v1/memory/supersede` {old_id, text, kind?, source?} → {id}
//!
//! ## Owner scoping (the deferred ADR 0018 P4)
//!
//! Under `AuthMode::KeyStore` each API key's `Principal.user` is an
//! isolated owner: facts written by one key are invisible to every
//! other key on every recall leg, and forget/supersede against another
//! owner's id behave exactly like a missing id (no existence leak).
//! `StaticToken` / `Off` requests carry no Principal and act as owner
//! `"local"` — the same owner every MCP tool handler uses — so a
//! loopback harness shares the operator's own memory by design.
//!
//! ## Judgment calls (documented per the ADR)
//!
//! * `remember` uses PLAIN `lamu_memory::lifetime_memory::remember`,
//!   NOT `remember_if_novel` — novelty dedup is the MCP autocapture
//!   path's concern; an explicit API write means the caller decided
//!   the fact is worth storing.
//! * Embedding rides the process-global chain (ADR 0030). No embedder
//!   resolved → the fact is stored UNEMBEDDED (recall degrades to
//!   FTS + recency); the route never fails on a missing backend.
//! * Quota: remember/supersede charge the principal's bucket with the
//!   `len/4` approximate token count of the text (the same heuristic
//!   the reasoning-token metric uses) — the embed cost is local but
//!   not free. Recall charges NOTHING in v1 (read path; revisit if
//!   external harnesses hammer it).
//! * Malformed bodies are parsed by hand so the response is the
//!   standard OpenAI-style envelope with status 400 — unlike the chat
//!   surfaces, which let axum's extractor emit `text/plain` 4xx prose
//!   (documented footgun there; new surface, no legacy to preserve).
//! * Storage errors are a generic 500 envelope; the real error goes to
//!   tracing, never to the wire.

use crate::keys::Principal;
use crate::openai_compat::{over_quota, user_label, AppState};
use crate::quota::QuotaCheck;
use axum::body::Bytes;
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use lamu_memory::lifetime_memory::{self, LOCAL_OWNER};
use serde::Deserialize;
use serde_json::json;

/// Default `kind` when the request omits it (mirrors the MCP handler).
const DEFAULT_KIND: &str = "fact";
/// Default `source` when the request omits it — distinguishes HTTP
/// writes from the MCP handlers' "manual" in the stored rows.
const DEFAULT_SOURCE: &str = "api";

/// `k` bounds for recall: clamp to 1..=50, default 8.
const RECALL_K_DEFAULT: usize = 8;
const RECALL_K_MAX: usize = 50;

/// Resolve the request's owner (ADR 0032): KeyStore principal → its
/// user; StaticToken/Off (no Principal) → [`LOCAL_OWNER`].
fn owner_of(principal: Option<&Principal>) -> &str {
    principal.map(|p| p.user.as_str()).unwrap_or(LOCAL_OWNER)
}

/// `len/4` approximate token count — the same heuristic the
/// reasoning-token metric uses. Good enough for quota accounting of a
/// local embed.
fn approx_tokens(text: &str) -> u64 {
    text.len() as u64 / 4
}

/// 400 with the standard OpenAI-style envelope. `msg` must be generic
/// (field-level, not parser internals).
fn bad_request(msg: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": msg, "type": "invalid_request_error"}})),
    )
        .into_response()
}

/// 500 with a generic envelope; the real error is the caller's to
/// trace — never leaks storage internals onto the wire.
fn storage_error() -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": {"message": "memory storage error", "type": "internal_error"}})),
    )
        .into_response()
}

/// Parse a JSON body by hand so malformed input gets the standard 400
/// envelope instead of axum's `text/plain` extractor prose.
// Err IS the ready-to-return axum Response — boxing it would just move
// the bytes for a cold error path.
#[allow(clippy::result_large_err)]
fn parse_body<T: serde::de::DeserializeOwned>(bytes: &Bytes) -> Result<T, Response> {
    match serde_json::from_slice::<T>(bytes) {
        Ok(v) => Ok(v),
        Err(e) => {
            tracing::debug!("memory_api: malformed body: {e}");
            Err(bad_request("malformed JSON body"))
        }
    }
}

// ── POST /v1/memory/remember ────────────────────────────────────────

#[derive(Deserialize)]
struct RememberReq {
    text: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    source: Option<String>,
}

pub(crate) async fn memory_remember(
    State(state): State<AppState>,
    principal: Option<Extension<Principal>>,
    bytes: Bytes,
) -> Response {
    let principal: Option<Principal> = principal.map(|Extension(p)| p);
    let principal_ref = principal.as_ref();
    let route = "/v1/memory/remember";
    if let QuotaCheck::Exhausted { limit } = state.quota.check(principal_ref) {
        return over_quota(route, limit);
    }
    let req: RememberReq = match parse_body(&bytes) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    let text = req.text.trim();
    if text.is_empty() {
        return bad_request("text is required");
    }
    let kind = req.kind.as_deref().map(str::trim).filter(|s| !s.is_empty()).unwrap_or(DEFAULT_KIND);
    let source =
        req.source.as_deref().map(str::trim).filter(|s| !s.is_empty()).unwrap_or(DEFAULT_SOURCE);
    let owner = owner_of(principal_ref);
    match lifetime_memory::remember(text, kind, source, owner).await {
        Ok(id) => {
            // Charge the embed's approximate cost (judgment call: len/4,
            // see module docs). After-the-fact like the chat surfaces.
            state.quota.charge(principal_ref, approx_tokens(text));
            tracing::info!(
                target: "lamu_audit",
                user = user_label(principal_ref),
                route,
                status = 200u16,
                memory_id = id,
                "memory remembered",
            );
            Json(json!({"id": id})).into_response()
        }
        Err(e) => {
            tracing::error!("memory remember failed (owner={owner}): {e}");
            storage_error()
        }
    }
}

// ── POST /v1/memory/recall ──────────────────────────────────────────

#[derive(Deserialize)]
struct RecallReq {
    query: String,
    #[serde(default)]
    k: Option<u64>,
    #[serde(default)]
    include_expired: Option<bool>,
}

pub(crate) async fn memory_recall(
    State(_state): State<AppState>,
    principal: Option<Extension<Principal>>,
    bytes: Bytes,
) -> Response {
    let principal: Option<Principal> = principal.map(|Extension(p)| p);
    let principal_ref = principal.as_ref();
    let req: RecallReq = match parse_body(&bytes) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    let query = req.query.trim();
    if query.is_empty() {
        return bad_request("query is required");
    }
    // Clamp k to 1..=50, default 8. No quota charge on recall (v1 —
    // see module docs).
    let k = (req.k.unwrap_or(RECALL_K_DEFAULT as u64) as usize).clamp(1, RECALL_K_MAX);
    let include_expired = req.include_expired.unwrap_or(false);
    let owner = owner_of(principal_ref);
    match lifetime_memory::recall_memory(query, k, include_expired, owner).await {
        Ok(hits) => {
            let hits: Vec<serde_json::Value> = hits
                .into_iter()
                .map(|h| {
                    json!({
                        "id": h.id,
                        "text": h.text,
                        "kind": h.kind,
                        "source": h.source,
                        "ts": h.ts,
                        "score": h.score,
                        "valid_until": h.valid_until,
                    })
                })
                .collect();
            Json(json!({"hits": hits})).into_response()
        }
        Err(e) => {
            tracing::error!("memory recall failed (owner={owner}): {e}");
            storage_error()
        }
    }
}

// ── POST /v1/memory/forget ──────────────────────────────────────────

#[derive(Deserialize)]
struct ForgetReq {
    id: i64,
}

pub(crate) async fn memory_forget(
    State(_state): State<AppState>,
    principal: Option<Extension<Principal>>,
    bytes: Bytes,
) -> Response {
    let principal: Option<Principal> = principal.map(|Extension(p)| p);
    let principal_ref = principal.as_ref();
    let req: ForgetReq = match parse_body(&bytes) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    let owner = owner_of(principal_ref);
    // Soft-delete; another owner's id (or a missing one) → forgotten:false,
    // indistinguishable on purpose (ADR 0032 — no existence leak).
    match lifetime_memory::forget(req.id, owner) {
        Ok(forgotten) => {
            if forgotten {
                tracing::info!(
                    target: "lamu_audit",
                    user = user_label(principal_ref),
                    route = "/v1/memory/forget",
                    status = 200u16,
                    memory_id = req.id,
                    "memory forgotten",
                );
            }
            Json(json!({"forgotten": forgotten})).into_response()
        }
        Err(e) => {
            tracing::error!("memory forget failed (owner={owner}): {e}");
            storage_error()
        }
    }
}

// ── POST /v1/memory/supersede ───────────────────────────────────────

#[derive(Deserialize)]
struct SupersedeReq {
    old_id: i64,
    text: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    source: Option<String>,
}

pub(crate) async fn memory_supersede(
    State(state): State<AppState>,
    principal: Option<Extension<Principal>>,
    bytes: Bytes,
) -> Response {
    let principal: Option<Principal> = principal.map(|Extension(p)| p);
    let principal_ref = principal.as_ref();
    let route = "/v1/memory/supersede";
    if let QuotaCheck::Exhausted { limit } = state.quota.check(principal_ref) {
        return over_quota(route, limit);
    }
    let req: SupersedeReq = match parse_body(&bytes) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    let text = req.text.trim();
    if text.is_empty() {
        return bad_request("text is required");
    }
    let kind = req.kind.as_deref().map(str::trim).filter(|s| !s.is_empty()).unwrap_or(DEFAULT_KIND);
    let source =
        req.source.as_deref().map(str::trim).filter(|s| !s.is_empty()).unwrap_or(DEFAULT_SOURCE);
    let owner = owner_of(principal_ref);
    match lifetime_memory::supersede(req.old_id, text, kind, source, owner).await {
        // affected=0: the old id is missing OR another owner's — the
        // 404 body is identical for both (no existence leak).
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": {
                "message": format!("no memory with id {}", req.old_id),
                "type": "not_found",
            }})),
        )
            .into_response(),
        Ok(Some(id)) => {
            state.quota.charge(principal_ref, approx_tokens(text));
            tracing::info!(
                target: "lamu_audit",
                user = user_label(principal_ref),
                route,
                status = 200u16,
                memory_id = id,
                superseded = req.old_id,
                "memory superseded",
            );
            Json(json!({"id": id})).into_response()
        }
        Err(e) => {
            tracing::error!("memory supersede failed (owner={owner}): {e}");
            storage_error()
        }
    }
}
