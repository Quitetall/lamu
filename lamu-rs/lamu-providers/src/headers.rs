//! Header helpers shared between sync and async transports.

use reqwest::header::HeaderValue;

/// Parse + validate `ANTHROPIC_BETA` env var into a header-safe value.
///
/// Returns `None` when the var is unset, empty after trimming, or
/// contains bytes reqwest would reject as a header value (newlines,
/// NULs, control chars).
///
/// Why a helper: `reqwest::RequestBuilder::header(_, str)` panics on
/// invalid bytes. A trailing newline in a `.env` file would silently
/// poison the build until this function trims + validates first.
pub fn anthropic_beta_header() -> Option<HeaderValue> {
    let raw = std::env::var("ANTHROPIC_BETA").ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match HeaderValue::from_str(trimmed) {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("ANTHROPIC_BETA rejected as header value: {} (ignoring)", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // SAFETY: env access in tests. Each test uses a unique var name to
    // avoid cross-test contamination — anthropic_beta_header() always
    // reads ANTHROPIC_BETA, so we serialize via a mutex.
    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn unset_returns_none() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: serialized by ENV_LOCK; no other thread reads
        // ANTHROPIC_BETA concurrently in this crate's tests.
        unsafe { std::env::remove_var("ANTHROPIC_BETA"); }
        assert!(anthropic_beta_header().is_none());
    }

    #[test]
    fn whitespace_only_returns_none() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("ANTHROPIC_BETA", "   \n\t  "); }
        assert!(anthropic_beta_header().is_none());
        unsafe { std::env::remove_var("ANTHROPIC_BETA"); }
    }

    #[test]
    fn trims_trailing_newline() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("ANTHROPIC_BETA", "context-1m-2025-08-07\n"); }
        let v = anthropic_beta_header().expect("should parse");
        assert_eq!(v.to_str().unwrap(), "context-1m-2025-08-07");
        unsafe { std::env::remove_var("ANTHROPIC_BETA"); }
    }

    #[test]
    fn rejects_embedded_newline() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("ANTHROPIC_BETA", "good\nbad"); }
        assert!(anthropic_beta_header().is_none());
        unsafe { std::env::remove_var("ANTHROPIC_BETA"); }
    }
}
