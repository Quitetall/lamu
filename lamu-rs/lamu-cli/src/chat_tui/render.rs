//! Phase 3.7: chat conversation rendering.
//!
//! All draw_* functions for the chat TUI plus text-wrap helpers that
//! both renderer and event-loop need (count_wrapped_rows, char
//! boundary walkers).

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use super::{strip_rich_markup, ChatTui};
use crate::theme;

pub(super) fn count_wrapped_rows(text: &str, width: usize) -> u16 {
    if width == 0 { return 1; }
    if text.is_empty() { return 1; }
    let mut rows: u32 = 0;
    for hard_line in text.split('\n') {
        if hard_line.is_empty() { rows += 1; continue; }
        // Word-aware wrap: place words; overflow wraps to next row.
        let mut col: usize = 0;
        let mut row_count: u32 = 1;
        let chars: Vec<char> = hard_line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            // Skip leading spaces only on row continuation, not at start.
            // Walk one "word" (run of non-space chars).
            let start = i;
            while i < chars.len() && !chars[i].is_whitespace() { i += 1; }
            let word_len = i - start;
            // If adding the word would overflow this row, wrap first.
            let would_be = if col == 0 { word_len } else { col + 1 + word_len };
            if would_be > width && col > 0 {
                row_count += 1;
                col = 0;
            }
            // Place the word — long words that exceed width still
            // occupy ceil(len/width) rows on their own.
            if word_len > width {
                let extra = (word_len.saturating_sub(1)) / width;
                row_count += extra as u32;
                col = word_len % width;
                if col == 0 { col = width; }
            } else {
                col += if col == 0 { word_len } else { word_len + 1 };
            }
            // Skip the whitespace separator.
            while i < chars.len() && chars[i].is_whitespace() && chars[i] != '\n' {
                col += 1;
                i += 1;
            }
            if col >= width {
                row_count += 1;
                col = 0;
            }
        }
        rows += row_count;
    }
    rows.min(u16::MAX as u32) as u16
}

pub(super) fn prev_char_boundary(s: &str, i: usize) -> usize {
    let mut j = i.saturating_sub(1);
    while j > 0 && !s.is_char_boundary(j) {
        j -= 1;
    }
    j
}
pub(super) fn next_char_boundary(s: &str, i: usize) -> usize {
    let mut j = i + 1;
    while j < s.len() && !s.is_char_boundary(j) {
        j += 1;
    }
    j
}

pub(super) fn draw(f: &mut ratatui::Frame, state: &mut ChatTui) {
    let banner_h = if state.theme.banner_logo.trim().is_empty() {
        3
    } else {
        (state.theme.banner_logo.lines().count().min(8) as u16) + 2
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(banner_h),
            Constraint::Min(8),
            Constraint::Length(5),  // input
            Constraint::Length(3),  // status
            Constraint::Length(5),  // footer
        ])
        .split(f.area());

    draw_banner(f, chunks[0], state);
    draw_conversation(f, chunks[1], state);
    draw_input(f, chunks[2], state);
    draw_status(f, chunks[3], state);
    draw_footer(f, chunks[4]);
}

fn draw_banner(f: &mut ratatui::Frame, area: Rect, state: &ChatTui) {
    let border = theme::hex_to_color(&state.theme.colors.banner_border, Color::DarkGray);
    let title_color = theme::hex_to_color(&state.theme.colors.banner_title, Color::Cyan);

    let mut lines: Vec<Line> = Vec::new();
    let logo = &state.theme.banner_logo;
    if logo.trim().is_empty() {
        let agent = if state.theme.branding.agent_name.is_empty() {
            "lamu".to_string()
        } else {
            state.theme.branding.agent_name.clone()
        };
        lines.push(Line::from(Span::styled(
            agent,
            Style::default().fg(title_color).add_modifier(Modifier::BOLD),
        )));
    } else {
        for raw in logo.lines().take(8) {
            lines.push(Line::from(Span::styled(
                strip_rich_markup(raw),
                Style::default().fg(title_color),
            )));
        }
    }
    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border)));
    f.render_widget(p, area);
}

fn draw_conversation(f: &mut ratatui::Frame, area: Rect, state: &mut ChatTui) {
    let response_border = theme::hex_to_color(&state.theme.colors.response_border, Color::Cyan);

    let lines = state.build_lines();
    // Estimate wrapped content height. Word-wrap aware: walk each
    // logical line, counting how many display rows it consumes when
    // wrapped at word boundaries — same way Paragraph::wrap renders.
    // Then add a small safety margin so follow_tail always scrolls
    // past the actual bottom (better to over-scroll a hair than to
    // leave the latest tokens off-screen).
    let inner_w = area.width.saturating_sub(2).max(1) as usize;
    let mut content_h: u16 = 0;
    for line in &lines {
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        content_h = content_h.saturating_add(count_wrapped_rows(&text, inner_w));
    }
    // Safety margin — better to over-estimate by a few rows than to
    // leave the bottom of the assistant's reply off-screen.
    content_h = content_h.saturating_add(2);
    let visible = area.height.saturating_sub(2);
    state.content_height = content_h;
    state.visible_height = visible;

    let max = content_h.saturating_sub(visible);
    let scroll = if state.follow_tail {
        state.scroll = max;
        max
    } else {
        state.scroll.min(max)
    };

    let title = if state.follow_tail || max == 0 {
        " conversation ".to_string()
    } else {
        format!(" conversation  [{}/{}] ", scroll, max)
    };

    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(response_border))
                .title(title),
        );
    f.render_widget(p, area);
}

fn draw_input(f: &mut ratatui::Frame, area: Rect, state: &ChatTui) {
    let rule = theme::hex_to_color(&state.theme.colors.input_rule, Color::DarkGray);
    let prompt = if state.theme.branding.prompt_symbol.is_empty() {
        "❯ ".to_string()
    } else {
        state.theme.branding.prompt_symbol.clone()
    };
    let title = if state.streaming() {
        " input — locked while streaming ".to_string()
    } else {
        format!(" {} ", prompt.trim())
    };

    // Render input as styled text with cursor marker. Multi-line splits
    // on '\n' and Paragraph wraps the rest.
    let mut lines: Vec<Line> = Vec::new();
    if state.input.is_empty() && !state.streaming() {
        lines.push(Line::from(Span::styled(
            "(type a message — /help for commands)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        // Split input + insert visible cursor mark
        let mut shown = state.input.clone();
        let cur = state.cursor.min(shown.len());
        if !state.streaming() {
            shown.insert_str(cur, "▏");
        }
        for body_line in shown.split('\n') {
            lines.push(Line::from(body_line.to_string()));
        }
    }
    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(rule))
                .title(title),
        );
    f.render_widget(p, area);
}

fn draw_status(f: &mut ratatui::Frame, area: Rect, state: &ChatTui) {
    let bar_bg = theme::hex_to_color(&state.theme.colors.status_bar_bg, Color::Black);
    let bar_text = theme::hex_to_color(&state.theme.colors.status_bar_text, Color::Gray);
    let bar_strong = theme::hex_to_color(&state.theme.colors.status_bar_strong, Color::Cyan);
    let good = theme::hex_to_color(&state.theme.colors.status_bar_good, Color::Green);

    let total_tokens: usize = state.history.iter().map(|m| m.content.len() / 4).sum::<usize>()
        + state.pending.len() / 4;
    let backend = state.config.backend_label();
    let theme_label = state.theme.name.clone();

    let mut spans: Vec<Span> = Vec::new();
    spans.push(Span::styled("MODEL ", Style::default().fg(bar_text)));
    spans.push(Span::styled(state.model.clone(), Style::default().fg(bar_strong).add_modifier(Modifier::BOLD)));
    spans.push(Span::styled(format!("  THEME {theme_label}"), Style::default().fg(bar_text)));
    spans.push(Span::styled(format!("  BACKEND {backend}"), Style::default().fg(bar_text)));
    spans.push(Span::styled(format!("  ~{total_tokens} tok"), Style::default().fg(bar_text)));
    if state.streaming() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(state.spinner_glyph(), Style::default().fg(good).add_modifier(Modifier::BOLD)));
        spans.push(Span::styled(" streaming…", Style::default().fg(good)));

        // Live timing while we wait. Before first token: count up the
        // prompt-eval clock. After first token: show TTFT + live tok/s.
        match (state.req_started, state.first_token_at, state.last_token_at) {
            (Some(start), None, _) => {
                let dt = start.elapsed().as_secs_f32();
                spans.push(Span::styled(
                    format!("  prompt {:.1}s", dt),
                    Style::default().fg(Color::Yellow),
                ));
            }
            (Some(start), Some(first), Some(last)) => {
                let prompt_s = first.duration_since(start).as_secs_f32();
                let gen_s = last.duration_since(first).as_secs_f32();
                let gen_n = state.tokens_this_req.saturating_sub(1);
                let tps = if gen_s > 0.0 && gen_n > 0 { gen_n as f32 / gen_s } else { 0.0 };
                spans.push(Span::styled(
                    format!("  prompt {:.1}s · {:.1} tok/s", prompt_s, tps),
                    Style::default().fg(bar_strong),
                ));
            }
            _ => {}
        }
    } else if let (Some(p), Some(g)) = (state.last_prompt_secs, state.last_gen_tps) {
        spans.push(Span::styled(
            format!("  last: prompt {:.1}s · {:.1} tok/s", p, g),
            Style::default().fg(bar_text),
        ));
    } else if let Some(p) = state.last_prompt_secs {
        spans.push(Span::styled(
            format!("  last: prompt {:.1}s", p),
            Style::default().fg(bar_text),
        ));
    }
    if state.show_thinking {
        spans.push(Span::styled("  [think:ON]", Style::default().fg(Color::Yellow)));
    }
    if state.search_enabled {
        spans.push(Span::styled("  [🔍search]", Style::default().fg(Color::Cyan)));
    }
    if !state.tool_status.is_empty() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(state.tool_status.clone(), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)));
    }
    if !state.status_msg.is_empty() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(state.status_msg.clone(), Style::default().fg(Color::Yellow)));
    }

    let p = Paragraph::new(Line::from(spans))
        .style(Style::default().bg(bar_bg))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(p, area);
}

fn draw_footer(f: &mut ratatui::Frame, area: Rect) {
    let key = |k: &str| Span::styled(format!("[{k}]"), Style::default().fg(Color::Cyan));
    let lbl = |t: &str| Span::raw(format!(" {t}  "));
    let row1 = Line::from(vec![
        key("Enter"), lbl("send"),
        key("Shift/Alt+Enter"), lbl("newline"),
        key("Ctrl+C"), lbl("quit"),
        key("Ctrl+S"), lbl("save"),
    ]);
    let row2 = Line::from(vec![
        key("↑/↓  PgUp/PgDn  mouse wheel"), lbl("scroll history"),
        key("←/→ Home/End"), lbl("cursor in input"),
    ]);
    let row3 = Line::from(vec![
        key("/help"), lbl("commands"),
        key("/quit  /exit"), lbl("leave"),
        key("/think  /search  /clear  /save  /model"), lbl(""),
    ]);
    let p = Paragraph::new(vec![row1, row2, row3])
        .block(Block::default().borders(Borders::ALL).title("keys + commands"));
    f.render_widget(p, area);
}
