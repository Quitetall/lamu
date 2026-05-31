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

use crate::openai_compat::AppState;

/// Resolve the configured API token, or `None` when auth is off. Order:
/// `LAMU_API_TOKEN` env (trimmed, non-empty) → `~/.config/lamu/api-token`
/// (trimmed). Read once at startup into `AppState`.
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

/// Bearer-auth middleware. No token configured → pass (frictionless loopback).
/// `/health` + `/metrics` are always exempt — liveness/scrape probes carry no
/// credentials. The token comparison is constant-time (`subtle`) so a timing
/// side-channel can't recover it byte-by-byte.
pub async fn require_bearer(State(st): State<AppState>, req: Request, next: Next) -> Response {
    let Some(expected) = st.auth_token.as_ref() else {
        return next.run(req).await;
    };
    let path = req.uri().path();
    if path == "/health" || path == "/metrics" {
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let ok = match presented {
        Some(t) => {
            use subtle::ConstantTimeEq;
            t.as_bytes().ct_eq(expected.as_bytes()).into()
        }
        None => false,
    };
    if ok {
        next.run(req).await
    } else {
        // OpenAI error-envelope shape so compat clients surface it correctly.
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            Json(json!({
                "error": { "message": "unauthorized", "type": "invalid_request_error" }
            })),
        )
            .into_response()
    }
}
