//! Phase 3.5: markdown rendering for the chat conversation pane.
//!
//! `render_markdown` is the entry point: takes raw text (assistant
//! reply / user input) and emits a Vec<Line> ready for ratatui.
//! `parse_inline` handles bold/italic/code/link spans inside a single
//! line. The rest are helpers (strip_html_tags, etc.).

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

pub(super) fn render_markdown(text: &str) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code = false;

    for raw in text.split('\n') {
        // ── fenced code block ──
        if raw.trim_start().starts_with("```") {
            if in_code {
                in_code = false;
                lines.push(Line::from(Span::styled(
                    "  ────────────────────────────────".to_string(),
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                in_code = true;
                let lang = raw.trim_start().trim_start_matches('`').trim();
                let label = if lang.is_empty() {
                    "  ── code ──────────────────────────".to_string()
                } else {
                    format!("  ── {} ──────────────────────────────", lang)
                };
                lines.push(Line::from(Span::styled(label, Style::default().fg(Color::DarkGray))));
            }
            continue;
        }
        if in_code {
            lines.push(Line::from(Span::styled(
                format!("  {}", raw),
                Style::default().fg(Color::Green),
            )));
            continue;
        }

        // ── headers ──
        if let Some(rest) = raw.strip_prefix("### ") {
            lines.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        if let Some(rest) = raw.strip_prefix("## ") {
            lines.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )));
            continue;
        }
        if let Some(rest) = raw.strip_prefix("# ") {
            lines.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )));
            continue;
        }

        // ── blockquote ──
        if let Some(rest) = raw.strip_prefix("> ") {
            lines.push(Line::from(Span::styled(
                format!("│ {}", rest),
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }

        // ── bullet list ──
        let bullet_rest = if raw.starts_with("- ") || raw.starts_with("* ") {
            Some(&raw[2..])
        } else if raw.len() > 3 && raw.as_bytes()[0].is_ascii_digit() && raw.as_bytes()[1] == b'.' && raw.as_bytes()[2] == b' ' {
            Some(&raw[3..])
        } else {
            None
        };
        if let Some(rest) = bullet_rest {
            let mut spans = vec![Span::raw("  • ".to_string())];
            spans.extend(parse_inline(rest));
            lines.push(Line::from(spans));
            continue;
        }

        // ── plain line with inline markup ──
        if raw.is_empty() {
            lines.push(Line::from(""));
        } else {
            lines.push(Line::from(parse_inline(raw)));
        }
    }
    // close unclosed code block
    if in_code {
        lines.push(Line::from(Span::styled(
            "  ────────────────────────────────".to_string(),
            Style::default().fg(Color::DarkGray),
        )));
    }
    lines
}

/// Parse inline markdown: **bold**, *italic*, `code`. Returns Vec<Span<'static>>.
fn parse_inline(s: &str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // backtick inline code
        if chars[i] == '`' {
            if !buf.is_empty() {
                spans.push(Span::raw(buf.clone()));
                buf.clear();
            }
            i += 1;
            let mut code = String::new();
            while i < chars.len() && chars[i] != '`' {
                code.push(chars[i]);
                i += 1;
            }
            if i < chars.len() { i += 1; } // skip closing `
            spans.push(Span::styled(code, Style::default().fg(Color::Green)));
            continue;
        }
        // **bold**
        if i + 1 < chars.len() && chars[i] == '*' && chars[i+1] == '*' {
            if !buf.is_empty() {
                spans.push(Span::raw(buf.clone()));
                buf.clear();
            }
            i += 2;
            let mut bold = String::new();
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i+1] == '*') {
                bold.push(chars[i]);
                i += 1;
            }
            if i + 1 < chars.len() { i += 2; }
            spans.push(Span::styled(bold, Style::default().add_modifier(Modifier::BOLD)));
            continue;
        }
        // *italic* (single star, not followed by another star)
        if chars[i] == '*' && (i + 1 >= chars.len() || chars[i+1] != '*') {
            if !buf.is_empty() {
                spans.push(Span::raw(buf.clone()));
                buf.clear();
            }
            i += 1;
            let mut italic = String::new();
            while i < chars.len() && chars[i] != '*' {
                italic.push(chars[i]);
                i += 1;
            }
            if i < chars.len() { i += 1; }
            spans.push(Span::styled(italic, Style::default().add_modifier(Modifier::ITALIC)));
            continue;
        }
        buf.push(chars[i]);
        i += 1;
    }
    if !buf.is_empty() {
        spans.push(Span::raw(buf));
    }
    if spans.is_empty() {
        spans.push(Span::raw(String::new()));
    }
    spans
}

