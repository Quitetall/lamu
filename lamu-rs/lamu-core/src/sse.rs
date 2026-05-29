//! Byte-level SSE line buffering.
//!
//! Decoding raw HTTP body chunks with `String::from_utf8_lossy(&chunk)`
//! corrupts any multibyte UTF-8 codepoint that straddles a chunk boundary:
//! the leading bytes in chunk N and the continuation bytes in chunk N+1
//! each become U+FFFD. Qwen/CJK output (and emoji, em-dashes, arrows) trips
//! this routinely on streaming responses.
//!
//! The fix is to accumulate raw bytes and only decode COMPLETE,
//! `\n`-terminated lines. A newline (0x0A) can never appear inside a UTF-8
//! multibyte sequence (continuation bytes are 0x80..=0xBF, lead bytes are
//! 0xC2..=0xF4), so splitting on `\n` at the byte level always lands on a
//! codepoint boundary — a complete line's bytes are valid UTF-8.

/// Pull the next complete `\n`-terminated line out of `buf`, decoding it as
/// (lossy) UTF-8 including the trailing newline. Returns `None` when no full
/// line is buffered yet; the trailing partial line's bytes stay in `buf` for
/// the next chunk.
pub fn next_sse_line(buf: &mut Vec<u8>) -> Option<String> {
    let nl = buf.iter().position(|&b| b == b'\n')?;
    // Decode the complete line straight from the buffer slice, then drop
    // those bytes — avoids the throwaway Vec that drain().collect() made.
    let line = String::from_utf8_lossy(&buf[..=nl]).into_owned();
    buf.drain(..=nl);
    Some(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_multibyte_split_across_chunks() {
        // "café\n" where the two bytes of 'é' (0xC3 0xA9) arrive in
        // separate chunks. The old push_str(from_utf8_lossy(chunk)) turned
        // each half into U+FFFD; byte-buffering keeps the char intact.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"caf\xc3"); // 'é' lead byte at the boundary
        assert_eq!(next_sse_line(&mut buf), None); // no newline yet
        buf.extend_from_slice(b"\xa9\n"); // continuation byte + newline
        assert_eq!(next_sse_line(&mut buf).as_deref(), Some("café\n"));
        assert!(buf.is_empty());
    }

    #[test]
    fn yields_multiple_lines_and_retains_partial() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"data: a\ndata: b\ndata: c");
        assert_eq!(next_sse_line(&mut buf).as_deref(), Some("data: a\n"));
        assert_eq!(next_sse_line(&mut buf).as_deref(), Some("data: b\n"));
        assert_eq!(next_sse_line(&mut buf), None); // "data: c" has no \n
        assert_eq!(buf, b"data: c");
    }

    #[test]
    fn empty_buffer_yields_none() {
        let mut buf: Vec<u8> = Vec::new();
        assert_eq!(next_sse_line(&mut buf), None);
    }
}
