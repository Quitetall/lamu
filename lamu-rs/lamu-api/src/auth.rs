//! Optional single-token bearer auth for the OpenAI-compat HTTP surface.
//!
//! Calibrated to LAMU's threat model: single-user, loopback-default. There is
//! deliberately NO multi-user machinery (accounts, sessions, TOTP, per-token
//! DB, CSP) — that defends a browser/multi-tenant app LAMU is not. The whole
//! mechanism is one static token compared in constant time:
//!
//!   * loopback bind + no token  → frictionless, auth off (the common case);
//!   * any bind + token set       → every route except /health + /metrics
//!                                  requires `Authorization: Bearer <token>`;
//!   * off-loopback bind + no token → `serve()` hard-fails at startup (the
//!                                  caller must `lamu auth init` or set
//!                                  LAMU_ALLOW_INSECURE=1).
//!
//! See docs/decisions/0012-*.md.

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::keys::{KeyStore, Principal};
use crate::openai_compat::{AppState, AuthMode};

/// Resolve the configured API token, or `None` when auth is off. Order:
/// `LAMU_API_TOKEN` env (trimmed, non-empty) → `~/.config/lamu/api-token`
/// (trimmed). Used by the `StaticToken` path (ADR 0012, kept).
pub fn resolve_token() -> Option<String> {
    if let Ok(t) = std::env::var("LAMU_API_TOKEN") {
        let t = t.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    let path = dirs::config_dir()?.join("lamu").join("api-token");
    let body = std::fs::read_to_string(path).ok()?;
    let t = body.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

/// Resolve the active auth backend (ADR 0018). Precedence: `keys.db` exists →
/// `KeyStore` (per-token multi-user); else `LAMU_API_TOKEN`/`api-token` →
/// `StaticToken` (the ADR-0012 path); else `Off`. A keys.db that exists but
/// fails to open degrades to the static path with a warning (a corrupt store
/// must not lock the operator out of a loopback box) — the off-loopback gate
/// in `serve()` still hard-fails on an empty/Off backend.
pub fn resolve_auth_mode() -> AuthMode {
    if let Some(db) = KeyStore::default_path() {
        if db.exists() {
            match KeyStore::open(&db) {
                // Engage KeyStore ONLY with ≥1 active key. An empty keys.db
                // (e.g. created by `lamu auth list` before any key is issued,
                // or after revoking the last key) must NOT flip serve into a
                // mode that 401s every request — fall through to the static/Off
                // path so loopback stays frictionless until a key actually exists.
                Ok(ks) if ks.has_active_key() => return AuthMode::KeyStore(std::sync::Arc::new(ks)),
                Ok(_) => {}
                Err(e) => tracing::warn!(
                    "keys.db at {} exists but failed to open ({e}); falling back to static-token auth",
                    db.display()
                ),
            }
        }
    }
    match resolve_token() {
        Some(t) => AuthMode::StaticToken(t),
        None => AuthMode::Off,
    }
}

/// Bearer-auth middleware (ADR 0018). `Off` → pass (frictionless loopback).
/// `StaticToken` → constant-time (`subtle`) compare of the bearer (ADR 0012,
/// unchanged). `KeyStore` → `verify(token)`, stashing the `Principal` in the
/// request extensions on success for downstream quota/audit. `/health` +
/// `/metrics` are always exempt; failures return the surface-correct 401.
pub async fn require_bearer(State(st): State<AppState>, mut req: Request, next: Next) -> Response {
    let mode = st.auth.as_ref();
    if let AuthMode::Off = mode {
        return next.run(req).await;
    }
    let path = req.uri().path().to_string();
    if path == "/health" || path == "/metrics" {
        return next.run(req).await;
    }
    // Parse leniently per RFC 7235: scheme is case-insensitive, whitespace
    // tolerated. Real clients send "Bearer <token>"; accept "bearer", extra
    // spaces, etc. Empty token → None.
    let presented: Option<String> = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
        .map(|(_, tok)| tok.trim().to_string())
        .filter(|t| !t.is_empty());

    match mode {
        AuthMode::Off => unreachable!("Off handled above"),
        AuthMode::StaticToken(expected) => {
            let ok = match presented.as_deref() {
                Some(t) => {
                    use subtle::ConstantTimeEq;
                    // ct_eq is constant-time over CONTENT for equal-length
                    // inputs; it short-circuits on length mismatch, leaking only
                    // token length — harmless (the lamu_<64hex> length is public).
                    t.as_bytes().ct_eq(expected.as_bytes()).into()
                }
                None => false,
            };
            if ok {
                next.run(req).await
            } else {
                unauthorized(&path)
            }
        }
        AuthMode::KeyStore(ks) => {
            // verify() runs the indexed hash lookup on every path so a wrong vs
            // unknown token cost the same (no user-enumeration timing). On
            // success, stash the Principal for downstream quota/audit handlers.
            let principal: Option<Principal> = presented.as_deref().and_then(|t| ks.verify(t));
            match principal {
                Some(p) => {
                    req.extensions_mut().insert(p);
                    next.run(req).await
                }
                None => unauthorized(&path),
            }
        }
    }
}

/// Surface-correct 401, shared by both authenticating modes: Anthropic shape on
/// /v1/messages, Ollama flat-string on /api/*, else the OpenAI shape. Always
/// carries `WWW-Authenticate: Bearer`.
fn unauthorized(path: &str) -> Response {
    let body = if path.starts_with("/v1/messages") {
        Json(json!({"type": "error", "error": {"type": "authentication_error", "message": "unauthorized"}}))
    } else if path.starts_with("/api/") {
        Json(json!({"error": "unauthorized"}))
    } else {
        Json(json!({"error": {"message": "unauthorized", "type": "invalid_request_error"}}))
    };
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        body,
    )
        .into_response()
}
