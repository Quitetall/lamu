//! Phase 3.2: dashboard / settings / launchers / MCP rendering.
//!
//! All draw_* functions extracted from `tui/mod.rs` so the parent
//! module focuses on event-loop wiring and the render functions can
//! be reviewed (and replaced) as a unit.

use crate::cloud_models::QuotaState;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::block::{Position, Title};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph};

use super::{AppState, Mode, ModelRef, HARNESSES};
use crate::lamu_config::LamuConfig;
use crate::mcp_servers::{self, ProbeStatus};

pub fn draw(f: &mut ratatui::Frame, state: &AppState) {
    match state.mode {
        Mode::Dashboard => draw_dashboard(f, state),
        Mode::Launchers => draw_launchers(f, state),
        Mode::McpServers => draw_mcp(f, state),
        Mode::Settings => draw_settings(f, state),
    }
}

fn draw_settings(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(4),
            Constraint::Length(5),
        ])
        .split(f.area());

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled("SETTINGS", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(format!("  config={}  ", LamuConfig::path().display())),
        Span::raw("[d/Esc] back"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    // List
    let items_data = state.settings_items();
    let items: Vec<ListItem> = items_data
        .iter()
        .map(|(label, _)| ListItem::new(label.clone()))
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Configurables"))
        .highlight_style(
            Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("→ ");
    let mut s = state.settings_state.clone();
    f.render_stateful_widget(list, chunks[1], &mut s);

    // Detail
    let mut detail: Vec<Line> = Vec::new();
    detail.push(Line::from(format!(
        "Backend URL: {}  ({})",
        state.config.backend_url, state.config.backend_label()
    )));
    detail.push(Line::from(format!(
        "Theme:       {}",
        if state.config.theme.is_empty() { "lamu (default)".into() } else { state.config.theme.clone() }
    )));
    if !state.status_msg.is_empty() {
        detail.push(Line::from(Span::styled(
            state.status_msg.clone(),
            Style::default().fg(Color::Yellow),
        )));
    }
    f.render_widget(
        Paragraph::new(detail).block(Block::default().borders(Borders::ALL).title("Detail")),
        chunks[2],
    );

    // Footer (3 rows simple → advanced)
    let row_basics = Line::from(vec![
        key_span("j/k"), key_lbl("move"),
        key_span("Enter"), key_lbl("activate selected"),
        key_span("d/Esc/Bksp"), key_lbl("back to dashboard"),
        key_span("q"), key_lbl("quit"),
    ]);
    let row_actions = Line::from(vec![
        Span::styled("Items: ", Style::default().fg(Color::DarkGray)),
        Span::raw("Backend cycles direct↔bifrost. "),
        Span::raw("Edit-* opens $EDITOR. "),
        Span::raw("Reset deletes cloud-models.yaml then reloads seed."),
    ]);
    let row_advanced = Line::from(vec![
        Span::styled(
            "Editor resolves: lamu_config.editor → $EDITOR → $VISUAL → vi",
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    let footer = Paragraph::new(vec![row_basics, row_actions, row_advanced])
        .block(Block::default().borders(Borders::ALL).title("keys — simple → advanced"));
    f.render_widget(footer, chunks[3]);
}

fn draw_mcp(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Min(8),    // list
            Constraint::Length(5), // detail
            Constraint::Length(5), // help footer (3 rows)
        ])
        .split(f.area());

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "MCP SERVERS",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  source="),
        Span::styled(
            mcp_servers::config_path().display().to_string(),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw(format!("  count={}  ", state.mcp_servers.len())),
        Span::raw("[d/Esc] back"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    let label = format!(
        "  {:<20}  {:<6}  {:<8}  {:<24}  {}",
        "NAME", "TYPE", "STATUS", "COMMAND", "ARGS"
    );
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(1)])
        .split(chunks[1]);
    f.render_widget(
        Paragraph::new(label).style(
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        split[0],
    );

    let items: Vec<ListItem> = state
        .mcp_servers
        .iter()
        .map(|s| {
            let (status_label, color) = match &s.status {
                ProbeStatus::Healthy { .. } => ("✓ healthy", Color::Green),
                ProbeStatus::Unreachable { .. } => ("✗ down", Color::Red),
                ProbeStatus::Untested => ("? untested", Color::Yellow),
            };
            let line = format!(
                "{:<20}  {:<6}  {:<10}  {:<24}  {}",
                truncate(&s.name, 20),
                s.typ,
                status_label,
                truncate(&s.command, 24),
                s.args.join(" "),
            );
            ListItem::new(line).style(Style::default().fg(color))
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Configured servers"))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("→ ");
    let mut s = state.mcp_state.clone();
    f.render_stateful_widget(list, split[1], &mut s);

    // Detail pane
    let mut lines: Vec<Line> = Vec::new();
    if let Some(idx) = state.selected_mcp_idx() {
        if let Some(entry) = state.mcp_servers.get(idx) {
            lines.push(Line::from(format!(
                "{} {} {}",
                entry.command,
                entry.args.join(" "),
                entry.cwd.as_deref().map(|c| format!("(cwd={})", c)).unwrap_or_default(),
            )));
            match &entry.status {
                ProbeStatus::Healthy { server_name } => {
                    lines.push(Line::from(Span::styled(
                        format!("✓ initialize → server.name={server_name}"),
                        Style::default().fg(Color::Green),
                    )));
                }
                ProbeStatus::Unreachable { reason } => {
                    lines.push(Line::from(Span::styled(
                        format!("✗ {reason}"),
                        Style::default().fg(Color::Red),
                    )));
                }
                ProbeStatus::Untested => {
                    lines.push(Line::from(Span::styled(
                        "? press [p] or Enter to probe (sends initialize, ≤3s)".to_string(),
                        Style::default().fg(Color::Yellow),
                    )));
                }
            }
        }
    }
    if !state.status_msg.is_empty() {
        lines.push(Line::from(Span::styled(
            state.status_msg.clone(),
            Style::default().fg(Color::Yellow),
        )));
    }
    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Detail")),
        chunks[2],
    );

    let row_basics = Line::from(vec![
        key_span("j/k"), key_lbl("move"),
        key_span("Enter"), key_lbl("probe selected"),
        key_span("d/Esc/Bksp"), key_lbl("back to dashboard"),
        key_span("q"), key_lbl("quit"),
    ]);
    let row_actions = Line::from(vec![
        key_span("p"), key_lbl("re-probe selected"),
        key_span("a"), key_lbl("probe all"),
        key_span("r"), key_lbl("reload ~/.claude.json"),
    ]);
    let row_advanced = Line::from(vec![
        Span::styled(
            "(stdio probe sends initialize JSON-RPC, 3s timeout) ",
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    let footer = Paragraph::new(vec![row_basics, row_actions, row_advanced])
        .block(Block::default().borders(Borders::ALL).title("keys — simple → advanced"));
    f.render_widget(footer, chunks[3]);
}

fn draw_dashboard(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(8),     // models
            Constraint::Length(3),  // vram
            Constraint::Length(6),  // status / health (loaded + GPU procs)
            Constraint::Length(5),  // help footer (3 rows of keys)
        ])
        .split(f.area());

    draw_header(f, chunks[0], state);
    draw_models(f, chunks[1], state);
    draw_vram(f, chunks[2], state);
    draw_status(f, chunks[3], state);
    draw_footer(f, chunks[4]);
}

fn draw_header(f: &mut ratatui::Frame, area: Rect, state: &AppState) {
    let serve = if state.serve_up { "✓" } else { "✗" };
    let bifrost = if state.bifrost_up { "✓" } else { "✗" };
    let header = Paragraph::new(Line::from(vec![
        Span::styled("LAMU", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::raw(format!("registry={} ", state.entries.len())),
        Span::raw(format!("serve={} ", serve)),
        Span::raw(format!("bifrost={}", bifrost)),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, area);
}

fn draw_models(f: &mut ratatui::Frame, area: Rect, state: &AppState) {
    // Split off a 1-row banner above the list for column labels.
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(1)])
        .split(area);

    let header_label = format!(
        "   {:<7} {:<40}  {:>5}  {:<6}  {:>5}  {:>9}  {}",
        "SOURCE", "NAME", "PARAMS", "QUANT", "CTX", "VRAM (MB)", "CAPABILITIES",
    );
    let header_widget = Paragraph::new(header_label).style(
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );
    f.render_widget(header_widget, split[0]);

    let items: Vec<ListItem> = state
        .model_view
        .iter()
        .map(|r| match r {
            ModelRef::Local(i) => {
                let e = &state.entries[*i];
                // Compact capability tags. Drop "chat" (universal),
                // abbreviate the rest. Joined with single space inside
                // brackets: e.g. [code think long].
                let caps = e.capabilities.iter().filter_map(|c| match c {
                    lamu_core::types::Capability::Chat => None,
                    lamu_core::types::Capability::Code => Some("code"),
                    lamu_core::types::Capability::Reasoning => Some("think"),
                    lamu_core::types::Capability::Routing => Some("route"),
                    lamu_core::types::Capability::Vision => Some("vis"),
                    lamu_core::types::Capability::LongContext => Some("long"),
                    lamu_core::types::Capability::Embedding => Some("embed"),
                }).collect::<Vec<_>>().join(" ");
                // Empty caps (chat-only models) → blank, not a stray "[]"
                let caps_field = if caps.is_empty() {
                    String::new()
                } else {
                    format!("[{}]", caps)
                };
                let notes_oneline = first_line(&e.notes);
                let fav = state.favorites.has_model(&e.name);
                let deployed = state.model_deployed(&e.name);
                let glyph = if deployed { "●" } else if fav { "★" } else { " " };
                let line = format!(
                    "{} {:<11} {:<28}  {:>4}B  {:<6}  {:>5}  {:>9}  {:<18}  {}",
                    glyph,
                    "[LOCAL]",
                    truncate(&e.name, 28),
                    format_params(e.params_b),
                    e.quant,
                    format_ctx(e.context_max),
                    e.vram_mb,
                    truncate(&caps_field, 18),
                    notes_oneline,
                );
                let style = if deployed {
                    let s = Style::default().fg(Color::Green);
                    if fav { s.add_modifier(Modifier::BOLD) } else { s }
                } else if fav {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(line).style(style)
            }
            ModelRef::Cloud(i) => {
                let m = &state.cloud_models[*i];
                let id = m.full_id();
                let fav = state.favorites.has_model(&id);
                let key_ok = m.key_present();
                // Single-column glyph so cloud rows stay column-aligned
                // with local rows. 🔒 is 2-col which pushed every
                // subsequent column right by one. `!` = key missing
                // (clearer "this row has a problem" than `?` which
                // reads as "unknown").
                let glyph = if !key_ok { "!" } else if fav { "★" } else { " " };
                // Cloud rows: blank tags column (no Capability vec) and
                // notes flow to the right. Column widths match local
                // rows so the LOCAL/CLOUD blocks line up cleanly.
                let line = format!(
                    "{} {:<11} {:<28}  {:>5}  {:<6}  {:>5}  {:>9}  {:<18}  {}",
                    glyph,
                    format!("[{}]", m.provider.to_uppercase()),
                    truncate(&id, 28),
                    "—",
                    "—",
                    format_ctx(m.context_max),
                    "—",
                    "",
                    first_line(&m.notes),
                );
                // Color rule for cloud rows:
                //   key missing               → red (can't auth at all)
                //   quota Exhausted           → red (used up)
                //   quota Low                 → yellow (running out)
                //   quota Available + key ok  → blue
                //   favorited                 → + bold
                let base = if !key_ok {
                    Color::Red
                } else {
                    match m.quota {
                        QuotaState::Available => Color::Blue,
                        QuotaState::Low => Color::Yellow,
                        QuotaState::Exhausted => Color::Red,
                    }
                };
                let mut style = Style::default().fg(base);
                if fav {
                    style = style.add_modifier(Modifier::BOLD);
                }
                ListItem::new(line).style(style)
            }
        })
        .collect();

    let header = format!("Models  [{}]", state.model_view.len());
    let footer = if state.model_filter.is_empty() {
        format!(
            "source={}  sort={}  L/C/A switch  /filter  *favorite",
            state.source_filter.label(),
            state.model_sort.label()
        )
    } else {
        format!(
            "source={}  filter='{}'  sort={}  Esc clears",
            state.source_filter.label(),
            state.model_filter,
            state.model_sort.label()
        )
    };

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Title::from(header))
                .title(Title::from(footer).position(Position::Bottom))
        )
        .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD))
        .highlight_symbol("→ ");

    let mut s = state.list_state.clone();
    f.render_stateful_widget(list, split[1], &mut s);
}

/// Render a context-window count as a compact human-readable label
/// (e.g. 131072 → "128K", 262144 → "256K", 4096 → "4K").
pub(super) fn format_ctx(ctx: u32) -> String {
    if ctx >= 1024 * 1024 {
        format!("{}M", ctx / (1024 * 1024))
    } else if ctx >= 1024 {
        format!("{}K", ctx / 1024)
    } else {
        ctx.to_string()
    }
}

fn draw_vram(f: &mut ratatui::Frame, area: Rect, state: &AppState) {
    let total = state.vram.total_mb.max(1) as f64;
    let used = state.vram.used_mb as f64;
    let pct = ((used / total).min(1.0).max(0.0) * 100.0) as u16;
    let label = if state.gpu_available {
        format!(
            "VRAM {} / {} MB · {} MB free · available {} MB",
            state.vram.used_mb, state.vram.total_mb, state.vram.free_mb, state.vram.available_mb
        )
    } else {
        format!("GPU UNAVAILABLE: {}", state.gpu_reason.as_deref().unwrap_or("unknown"))
    };
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("VRAM"))
        .gauge_style(Style::default().fg(if state.gpu_available { Color::Green } else { Color::Red }))
        .percent(if state.gpu_available { pct } else { 0 })
        .label(label);
    f.render_widget(gauge, area);
}

fn draw_status(f: &mut ratatui::Frame, area: Rect, state: &AppState) {
    let mut lines: Vec<Line> = Vec::new();

    let loaded_count = state.vram.loaded_models.len();
    lines.push(Line::from(format!(
        "Loaded models: {}{}",
        loaded_count,
        if loaded_count == 0 { "  (no model in scheduler)" } else { "" }
    )));

    // GPU process snapshot: shows what's eating VRAM, even if lamu's
    // scheduler didn't spawn it. Untracked → orange so user can spot
    // orphans (`just swap`-style externally-launched llama-server, etc.).
    if state.gpu_procs.is_empty() {
        lines.push(Line::from(Span::styled(
            "GPU procs: (none)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for (pid, mb, name) in &state.gpu_procs {
            // Tracked = scheduler.loaded has any model with matching pid (we
            // don't have that mapping here, so approximate: if loaded_models
            // is empty, EVERY proc is untracked).
            let untracked = state.vram.loaded_models.is_empty();
            let line = format!("GPU proc: pid={pid:<7} {name:<24} {mb:>6} MB");
            let style = if untracked {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(line, style)));
        }
    }

    if !state.status_msg.is_empty() {
        lines.push(Line::from(Span::styled(
            state.status_msg.clone(),
            Style::default().fg(Color::Yellow),
        )));
    }

    let p = Paragraph::new(lines).block(
        Block::default().borders(Borders::ALL).title("Status — what's on the GPU"),
    );
    f.render_widget(p, area);
}

/// Cyan `[key]` chip span — used in every footer for visual consistency.
fn key_span(k: &str) -> Span<'static> {
    Span::styled(format!("[{k}]"), Style::default().fg(Color::Cyan))
}
/// Yellow chip — for actions like ★ fav that we want to call out.
fn key_span_warn(k: &str) -> Span<'static> {
    Span::styled(format!("[{k}]"), Style::default().fg(Color::Yellow))
}
/// Inline footer label after a key chip. Renamed from `label` so it
/// doesn't shadow any local `let label = ...` bindings in draw_*.
fn key_lbl(text: &str) -> Span<'static> {
    Span::raw(format!(" {text}  "))
}

fn draw_footer(f: &mut ratatui::Frame, area: Rect) {
    // Three rows, simple → advanced. Same shape on every screen so users
    // build muscle memory for where each tier of action lives.
    let row_basics = Line::from(vec![
        key_span("j/k"), key_lbl("move"),
        key_span("Enter"), key_lbl("chat"),
        key_span("r"), key_lbl("refresh"),
        key_span("q"), key_lbl("quit (×2)"),
        key_span("x"), key_lbl("instant exit"),
    ]);
    let row_list_ops = Line::from(vec![
        key_span_warn("*"), key_lbl("fav"),
        key_span("/"), key_lbl("filter"),
        key_span("o"), key_lbl("sort"),
        key_span("L/C/A"), key_lbl("source local/cloud/all"),
        key_span("a"), key_lbl("set api key"),
    ]);
    let row_advanced = Line::from(vec![
        key_span("h"), key_lbl("harnesses"),
        key_span("m"), key_lbl("mcp servers"),
        key_span("s"), key_lbl("settings"),
        key_span("S"), key_lbl("start lamu serve"),
        key_span("B"), key_lbl("start bifrost"),
        key_span("n"), key_lbl("add cloud"),
        key_span("K"), key_lbl("key status"),
        key_span("e"), key_lbl("edit cloud yaml"),
    ]);
    let footer = Paragraph::new(vec![row_basics, row_list_ops, row_advanced])
        .block(Block::default().borders(Borders::ALL).title("keys — simple → advanced"));
    f.render_widget(footer, area);
}

fn draw_launchers(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header (last harness tag)
            Constraint::Min(8),    // list
            Constraint::Length(4), // detail / status
            Constraint::Length(5), // help footer (3 rows)
        ])
        .split(f.area());

    // Header — show the most recent harness used + return-to-dashboard hint.
    let last = state.last_harness.unwrap_or("(none yet)");
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "HARNESSES",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  last="),
        Span::styled(last, Style::default().fg(Color::Yellow)),
        Span::raw("  press [d] or [Esc] to return to dashboard"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    // List
    let label = format!(
        "  {:<18}  {:<10}  {:<8}  {}",
        "NAME", "BINARY", "STATUS", "NOTES"
    );
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(1)])
        .split(chunks[1]);
    f.render_widget(
        Paragraph::new(label).style(
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        split[0],
    );

    let items: Vec<ListItem> = state
        .harness_view
        .iter()
        .map(|&i| {
            let h = &HARNESSES[i];
            let installed = *state.harness_installed.get(i).unwrap_or(&false);
            let status_label = if installed { "✓ on PATH" } else { "✗ missing" };
            let is_default = state.favorites.default_harness() == Some(h.name);
            let glyph = if is_default {
                "▶"
            } else if state.favorites.has_harness(h.name) {
                "★"
            } else {
                " "
            };
            let line = format!(
                "{} {:<18}  {:<10}  {:<8}  {}",
                glyph,
                h.name,
                h.bin,
                status_label,
                h.notes,
            );
            let mut style = if installed {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Red)
            };
            if state.favorites.has_harness(h.name) {
                style = style.add_modifier(Modifier::BOLD);
            }
            if is_default {
                style = style.add_modifier(Modifier::UNDERLINED);
            }
            ListItem::new(line).style(style)
        })
        .collect();

    let header = format!("Harnesses  [{}]", state.harness_view.len());
    let footer = if state.harness_filter.is_empty() {
        format!("sort={}  /filter  *favorite", state.harness_sort.label())
    } else {
        format!("filter='{}'  Esc clears", state.harness_filter)
    };

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Title::from(header))
                .title(Title::from(footer).position(Position::Bottom))
        )
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("→ ");
    let mut s = state.launcher_state.clone();
    f.render_stateful_widget(list, split[1], &mut s);

    // Detail pane: install hint for selected harness.
    let mut detail_lines: Vec<Line> = Vec::new();
    if let Some(h) = state.selected_harness() {
        // harness_installed is ORIGINAL-indexed (built from HARNESSES); the
        // launcher list is a sorted VIEW, so launcher_state.selected() (a
        // view index) read the wrong row's install status once any harness
        // was favorited/installed. Map view→original. (#24)
        let idx = state.selected_harness_orig_idx().unwrap_or(0);
        let installed = *state.harness_installed.get(idx).unwrap_or(&false);
        if installed {
            detail_lines.push(Line::from(format!("[Enter] launches: {}", h.launch_argv.join(" "))));
        } else {
            detail_lines.push(Line::from(Span::styled(
                format!("Install: {}", h.install),
                Style::default().fg(Color::Yellow),
            )));
            detail_lines.push(Line::from("Press [i] to run the install command."));
        }
    }
    if !state.status_msg.is_empty() {
        detail_lines.push(Line::from(Span::styled(
            state.status_msg.clone(),
            Style::default().fg(Color::Yellow),
        )));
    }
    f.render_widget(
        Paragraph::new(detail_lines).block(Block::default().borders(Borders::ALL).title("Detail")),
        chunks[2],
    );

    let row_basics = Line::from(vec![
        key_span("j/k"), key_lbl("move"),
        key_span("Enter"), key_lbl("launch selected"),
        key_span("d/Esc/Bksp"), key_lbl("back to dashboard"),
        key_span("q"), key_lbl("quit"),
    ]);
    let row_list_ops = Line::from(vec![
        key_span_warn("*"), key_lbl("fav"),
        key_span("/"), key_lbl("filter"),
        key_span("o"), key_lbl("sort"),
        key_span("r"), key_lbl("refresh PATH detection"),
    ]);
    let row_advanced = Line::from(vec![
        key_span("i"), key_lbl("install missing harness"),
        key_span("D"), key_lbl("set/unset default harness for Dashboard Enter"),
    ]);
    let footer = Paragraph::new(vec![row_basics, row_list_ops, row_advanced])
        .block(Block::default().borders(Borders::ALL).title("keys — simple → advanced"));
    f.render_widget(footer, chunks[3]);
}

pub(super) fn truncate(s: &str, max: usize) -> String {
    // Guard against max == 0: subtracting from zero would underflow
    // and silently return a value longer than the input. No current
    // caller passes 0, but a future caller with a dynamic width
    // calculation might.
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}

/// Pull the first line of a multi-line string, trimmed. Empty input → "".
fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").trim().to_string()
}

pub(super) fn format_params(p: f32) -> String {
    if p >= 10.0 {
        format!("{:.0}", p)
    } else {
        format!("{:.1}", p)
    }
}

