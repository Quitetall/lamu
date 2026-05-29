//! Single funnel for structured events. Mirrors `lamu/core/observability.py`.
//!
//! Every place in lamu-rs that needs to emit a structured event for
//! operators or downstream tooling calls `emit()` here. Direct
//! `eprintln!("{}",...)` calls bypass the file sink — keep them out of
//! new code.
//!
//! Sinks:
//!   - stderr (always)
//!   - LAMU_EVENT_LOG=/path/to/jsonl (optional file sink)
//!
//! Trace IDs are first-class — pass `trace_id=Some("...")` to thread an
//! MCP request / HTTP traceparent through every event spanned by that
//! work. `new_trace_id()` generates a 16-hex-char id (compatible with
//! the middle 16 chars of W3C TraceContext).

use serde_json::{json, Value};
use std::fs::OpenOptions;
use std::io::Write;

/// Emit a structured event.
///
/// `fields` should be a JSON object — its keys are merged with `event`
/// and `trace_id` to form the final line. Stderr always gets a JSON
/// line; if `LAMU_EVENT_LOG` is set, the same line is appended there.
/// File-sink errors are swallowed so a bad sink path can't wedge the
/// runtime — operators still get the stderr copy.
pub fn emit(event: &str, trace_id: Option<&str>, fields: Value) {
    let mut obj = json!({"event": event});
    if let Some(tid) = trace_id {
        obj["trace_id"] = json!(tid);
    }
    if let Some(map) = fields.as_object() {
        for (k, v) in map {
            obj[k] = v.clone();
        }
    }
    let line = obj.to_string();
    eprintln!("{line}");

    if let Ok(path) = std::env::var("LAMU_EVENT_LOG") {
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Generate a 16-hex-char trace id. Compatible with the middle 16 chars
/// of W3C TraceContext; fine as a standalone id when no traceparent is
/// in scope.
pub fn new_trace_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // Mix high + low halves so successive calls under nano-precision
    // tick still produce distinct ids.
    let hi = (nanos >> 64) as u64;
    let lo = nanos as u64;
    format!("{:08x}{:08x}", hi as u32 ^ lo as u32, (lo >> 32) as u32)
}

/// Parse a W3C `traceparent` header (`00-<32 hex traceid>-<16 hex spanid>-<2 hex flags>`)
/// and return the first 16 hex chars of the traceid, suitable for use
/// as our internal trace_id. Returns None on malformed input.
pub fn trace_id_from_traceparent(tp: &str) -> Option<String> {
    let parts: Vec<&str> = tp.split('-').collect();
    if parts.len() < 4 {
        return None;
    }
    let traceid = parts[1];
    // `str::get` returns None if traceid is shorter than 16 bytes OR if
    // byte 16 is not a UTF-8 char boundary — both "malformed". The old
    // `len() < 16` + `traceid[..16]` byte-slice panicked when an
    // attacker-supplied multibyte char straddled byte 16.
    traceid.get(..16).map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_trace_id_is_16_hex() {
        let t = new_trace_id();
        assert_eq!(t.len(), 16);
        u64::from_str_radix(&t, 16).expect("hex parse");
    }

    #[test]
    fn new_trace_id_distinct_calls_distinct() {
        let a = new_trace_id();
        // sleep a tick so SystemTime advances on platforms with coarse clocks
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = new_trace_id();
        assert_ne!(a, b);
    }

    #[test]
    fn traceparent_extracts_middle_16() {
        let tp = "00-0123456789abcdef0123456789abcdef-0011223344556677-01";
        assert_eq!(
            trace_id_from_traceparent(tp).as_deref(),
            Some("0123456789abcdef")
        );
    }

    #[test]
    fn traceparent_rejects_short_input() {
        assert_eq!(trace_id_from_traceparent("nope"), None);
        assert_eq!(trace_id_from_traceparent("00-short-x-01"), None);
    }

    #[test]
    fn traceparent_multibyte_does_not_panic() {
        // Attacker-controlled traceid whose byte 16 splits a multibyte
        // codepoint: `€` is 3 bytes, so "€€€€€€" is 18 bytes and byte 16
        // lands mid-`€`. Old `traceid[..16]` panicked; `str::get` returns
        // None (treated as malformed) without unwinding the server loop.
        let tp = "00-€€€€€€-0011223344556677-01";
        assert_eq!(trace_id_from_traceparent(tp), None);
        // ASCII traceid exactly 16 bytes still extracts.
        assert_eq!(
            trace_id_from_traceparent("00-0123456789abcdef-aa-01").as_deref(),
            Some("0123456789abcdef")
        );
    }

    #[test]
    fn emit_writes_to_event_log_when_set() {
        let dir = std::env::temp_dir().join(format!("lamu-emit-test-{}", new_trace_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.jsonl");
        // SAFETY: edition-2024 marks env mutation unsafe; this test runs
        // single-threaded under cargo test's per-test process boundary,
        // so racing readers on LAMU_EVENT_LOG aren't a concern here.
        unsafe {
            std::env::set_var("LAMU_EVENT_LOG", &path);
        }
        emit("unit_test_event", Some("abc1234567890def"), json!({"k":"v"}));
        unsafe {
            std::env::remove_var("LAMU_EVENT_LOG");
        }
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("\"event\":\"unit_test_event\""));
        assert!(body.contains("\"trace_id\":\"abc1234567890def\""));
        assert!(body.contains("\"k\":\"v\""));
    }
}
