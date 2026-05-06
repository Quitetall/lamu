//! Streaming markdown renderer for the chat REPL.
//!
//! Tokens arrive one at a time. We accumulate them into a buffer and
//! re-render the *entire* current message after each chunk. Re-render =
//! cursor up N lines, clear, redraw via `termimad::FmtText`. termimad
//! handles inline (`**bold**`, `` ` ``code` `` `, `_em_`) and block
//! (headers, code fences, lists, quotes) markdown — even when the
//! stream is mid-fence, the open block stays styled.
//!
//! Limits:
//!   - Redraw caps at the terminal height. Messages taller than the
//!     viewport stop redrawing the parts that scrolled off; the most
//!     recent N lines stay live.
//!   - Skin is fixed at "dark default" for now. Could be themed.
//!
//! `<think>...</think>` filtering happens BEFORE the buffer here —
//! think blocks are stripped (or shown) in the calling repl, then the
//! visible text comes here.

use crossterm::cursor;
use crossterm::execute;
use crossterm::terminal::{Clear, ClearType};
use std::io::{self, Write};
use termimad::MadSkin;

pub struct StreamMdRenderer {
    /// Full buffered text we've received so far.
    buffer: String,
    /// Lines printed by the last render, so we can erase them next time.
    last_lines_printed: u16,
    /// termimad rendering skin.
    skin: MadSkin,
    /// Width to wrap at — captured at construction so re-renders are stable.
    width: u16,
    /// Cap on lines we actively redraw. Anything older scrolls off and
    /// stays static.
    max_redraw_lines: u16,
}

impl StreamMdRenderer {
    pub fn new() -> Self {
        let (width, height) = terminal_size();
        Self {
            buffer: String::with_capacity(2048),
            last_lines_printed: 0,
            skin: MadSkin::default_dark(),
            width: width.saturating_sub(2).max(40),
            // Leave a couple of rows for the prompt.
            max_redraw_lines: height.saturating_sub(4).max(8),
        }
    }

    /// Drop a token into the buffer + redraw.
    pub fn push_token(&mut self, token: &str) -> io::Result<()> {
        self.buffer.push_str(token);
        self.redraw()
    }

    /// One-shot render of the full buffer; call once at stream end so
    /// the trailing newline lives in scrollback as committed text.
    pub fn finalize(&mut self) -> io::Result<String> {
        // Ensure trailing newline so subsequent prompt lands on a fresh line.
        if !self.buffer.ends_with('\n') {
            self.buffer.push('\n');
        }
        self.redraw()?;
        // Reset internal state so the next message starts clean.
        let final_text = std::mem::take(&mut self.buffer);
        self.last_lines_printed = 0;
        Ok(final_text)
    }

    /// Erase the previous render + emit a fresh one.
    fn redraw(&mut self) -> io::Result<()> {
        let mut out = io::stdout();

        // Clip to the most recent `max_redraw_lines` of the buffer so the
        // top of long messages stays scrolled-up + static.
        let view = clip_to_last_lines(&self.buffer, self.max_redraw_lines as usize);

        // Cursor up + clear the previous render's lines.
        if self.last_lines_printed > 0 {
            execute!(out, cursor::MoveToColumn(0))?;
            for _ in 0..self.last_lines_printed {
                execute!(out, Clear(ClearType::CurrentLine), cursor::MoveUp(1))?;
            }
            execute!(out, Clear(ClearType::CurrentLine))?;
        }

        // Render via termimad. FmtText::from materialises ANSI; print it.
        let fmt = termimad::FmtText::from(&self.skin, view, Some(self.width as usize));
        let rendered = fmt.to_string();
        write!(out, "{}", rendered)?;
        out.flush()?;

        // Count how many lines we just emitted so the next redraw can
        // erase them. termimad always ends with a newline.
        self.last_lines_printed = visible_line_count(&rendered, self.width);
        Ok(())
    }
}

/// Approximate visible line count in a rendered string. termimad already
/// wraps to the requested width, so most lines are bounded; we still
/// count any extra wrapping from CJK / wide chars conservatively by
/// counting every `\n`. Subtracts 1 because the trailing `\n` advances
/// to the start of the next line, where the next render will resume.
fn visible_line_count(rendered: &str, _width: u16) -> u16 {
    let nl = rendered.matches('\n').count() as u16;
    nl
}

/// Return the suffix of `text` that fits in at most `max_lines` newlines.
/// Used to cap the redraw region — once a message is taller than the
/// viewport, the older lines stop being live.
fn clip_to_last_lines(text: &str, max_lines: usize) -> &str {
    if max_lines == 0 {
        return "";
    }
    let total: usize = text.matches('\n').count();
    if total <= max_lines {
        return text;
    }
    // Skip the first (total - max_lines) newlines.
    let skip = total - max_lines;
    let mut seen = 0usize;
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            seen += 1;
            if seen == skip {
                return &text[i + 1..];
            }
        }
    }
    text
}

fn terminal_size() -> (u16, u16) {
    crossterm::terminal::size().unwrap_or((100, 30))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_no_change_when_under_limit() {
        let s = "a\nb\nc\n";
        assert_eq!(clip_to_last_lines(s, 10), s);
    }

    #[test]
    fn clip_keeps_tail_only() {
        let s = "1\n2\n3\n4\n5\n";
        let out = clip_to_last_lines(s, 2);
        assert_eq!(out, "4\n5\n");
    }

    #[test]
    fn clip_zero_returns_empty() {
        assert_eq!(clip_to_last_lines("anything\n", 0), "");
    }

    #[test]
    fn visible_line_count_matches_newlines() {
        assert_eq!(visible_line_count("a\nb\nc\n", 80), 3);
        assert_eq!(visible_line_count("", 80), 0);
    }

    #[test]
    fn renderer_buffers_appends() {
        let mut r = StreamMdRenderer::new();
        r.buffer.push_str("hello");
        r.buffer.push_str(" world");
        assert_eq!(r.buffer, "hello world");
    }
}
