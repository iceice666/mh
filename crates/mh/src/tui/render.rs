//! Drawing only. No I/O, no state mutation beyond the row cache the app owns.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};

use super::app::{
    App, COMPOSER_MAX, COMPOSER_MIN, Focus, Level, MIN_COLS, MIN_ROWS, RowKind, SIDEBAR_COLS,
    Target, WIDE_COLS, display_line, wrap,
};

pub(super) fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    if area.width < MIN_COLS || area.height < MIN_ROWS {
        let text = format!("Terminal too small (min {MIN_COLS}x{MIN_ROWS})\nCtrl-Q quit");
        frame.render_widget(Paragraph::new(text).style(warn_style()), area);
        return;
    }
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .areas(area);

    frame.render_widget(
        Paragraph::new(display_line(&app.header())).style(title_style()),
        header,
    );
    frame.render_widget(
        Paragraph::new(display_line(&app.footer())).style(dim_style()),
        footer,
    );

    let wide = area.width >= WIDE_COLS;
    let (sidebar, main) = if wide {
        let [left, right] =
            Layout::horizontal([Constraint::Length(SIDEBAR_COLS), Constraint::Min(1)]).areas(body);
        (Some(left), right)
    } else {
        (None, body)
    };

    if let Some(sidebar) = sidebar {
        draw_sidebar(frame, app, sidebar);
    }
    draw_main(frame, app, main);

    // Narrow layout: the sidebar becomes an overlay so tasks stay reachable.
    if !wide && app.focus() == Focus::Tasks {
        let overlay = centered(body, body.width.min(SIDEBAR_COLS + 12), body.height);
        frame.render_widget(Clear, overlay);
        draw_sidebar(frame, app, overlay);
    }
    if app.trust().is_some() {
        draw_trust(frame, app, body);
    } else if app.help_open() {
        draw_help(frame, body);
    }
}

fn draw_sidebar(frame: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus() == Focus::Tasks;
    let block = Block::default()
        .borders(Borders::ALL)
        .title("Tasks")
        .border_style(border_style(focused));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let width = usize::from(inner.width.max(4));
    let items: Vec<ListItem> = app
        .sidebar_items()
        .into_iter()
        .flat_map(|item| {
            let style = if item.selected {
                kind_style(item.kind).add_modifier(Modifier::REVERSED)
            } else {
                kind_style(item.kind)
            };
            wrap(&item.text, width)
                .into_iter()
                .map(move |line| ListItem::new(Line::from(Span::styled(line, style))))
                .collect::<Vec<_>>()
        })
        .collect();
    frame.render_widget(List::new(items), inner);
}

fn draw_main(frame: &mut Frame, app: &mut App, area: Rect) {
    // The composer grows with its content up to half the body, so a long
    // paste never squeezes the transcript out of existence.
    let (composer_rows, cursor) = app.composer_view(area.width.saturating_sub(2));
    let wanted = u16::try_from(composer_rows.len()).unwrap_or(COMPOSER_MAX) + 2;
    let composer_height = wanted
        .clamp(COMPOSER_MIN, COMPOSER_MAX)
        .min(area.height / 2)
        .max(COMPOSER_MIN.min(area.height));
    let notices = app.notices();
    let notice_height = u16::try_from(notices.len().min(3)).unwrap_or(0);
    let [info, transcript, notice_area, composer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(notice_height),
        Constraint::Length(composer_height),
    ])
    .areas(area);

    frame.render_widget(
        Paragraph::new(display_line(&app.info_line())).style(dim_style()),
        info,
    );
    draw_transcript(frame, app, transcript);

    if notice_height > 0 {
        let lines: Vec<Line> = notices
            .iter()
            .rev()
            .take(usize::from(notice_height))
            .rev()
            .map(|notice| {
                Line::from(Span::styled(
                    display_line(&notice.text),
                    level_style(notice.level),
                ))
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), notice_area);
    }

    let focused = app.focus() == Focus::Composer;
    let title = match app.target() {
        Target::New => "New task".to_string(),
        Target::Task(id) => format!("Message task #{}", id.0),
    };
    let title = if app.composer_locked() {
        format!("{title} (locked)")
    } else {
        title
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(border_style(focused));
    let inner = block.inner(composer);
    frame.render_widget(block, composer);
    let visible = usize::from(inner.height.max(1));
    let skip = composer_rows.len().saturating_sub(visible);
    let shown: Vec<Line> = composer_rows
        .iter()
        .skip(skip)
        .map(|row| Line::from(row.as_str()))
        .collect();
    frame.render_widget(Paragraph::new(shown), inner);
    if focused && !app.composer_locked() && app.trust().is_none() {
        let row = cursor.0.saturating_sub(skip);
        let x = inner.x
            + u16::try_from(cursor.1)
                .unwrap_or(0)
                .min(inner.width.saturating_sub(1));
        let y = inner.y
            + u16::try_from(row)
                .unwrap_or(0)
                .min(inner.height.saturating_sub(1));
        frame.set_cursor_position(Position { x, y });
    }
}

fn draw_transcript(frame: &mut Frame, app: &mut App, area: Rect) {
    let focused = app.focus() == Focus::Transcript;
    let mut title = "Conversation".to_string();
    if app.unseen_activity() {
        title.push_str(" · new activity (End)");
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(border_style(focused));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let height = usize::from(inner.height.max(1));
    app.note_metrics(height);
    let offset = app.scroll_offset();
    let rows = app.rows(inner.width);
    // Only the visible window is materialized: scroll state is `usize`, so a
    // long session is not truncated by a 16-bit widget offset.
    let end = rows.len().saturating_sub(offset.min(rows.len()));
    let start = end.saturating_sub(height);
    let lines: Vec<Line> = rows[start..end]
        .iter()
        .map(|row| Line::from(Span::styled(row.text.clone(), kind_style(row.kind))))
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

fn draw_trust(frame: &mut Frame, app: &App, area: Rect) {
    let Some(prompt) = app.trust() else {
        return;
    };
    let overlay = centered(
        area,
        area.width.saturating_sub(4),
        area.height.saturating_sub(2),
    );
    frame.render_widget(Clear, overlay);
    let block = Block::default()
        .borders(Borders::ALL)
        .title("Untrusted workspace prelude")
        .border_style(warn_style());
    let inner = block.inner(overlay);
    frame.render_widget(block, overlay);
    let width = usize::from(inner.width.max(8));
    let prelude = &prompt.prelude;
    let mut lines: Vec<Line> = Vec::new();
    let mut push = |text: String, style: Style| {
        for piece in wrap(&display_line(&text), width) {
            lines.push(Line::from(Span::styled(piece, style)));
        }
    };
    push(format!("{}", prelude.path.display()), title_style());
    push(
        "It runs before every PTC program with your own capabilities: it can \
         read and write this workspace and start subprocesses."
            .to_string(),
        warn_style(),
    );
    push(
        format!(
            "{} lines · {}",
            prelude.source.lines().count(),
            prelude.identity().0
        ),
        dim_style(),
    );
    match prelude.description() {
        Some(tools) => {
            push("Advertised tools:".to_string(), dim_style());
            for line in tools.lines() {
                push(format!("  {line}"), Style::default());
            }
        }
        None => push(
            "It advertises no tools to the model.".to_string(),
            dim_style(),
        ),
    }
    push(
        "y trust · n / Enter / Esc reject · PageUp/PageDown scroll source".to_string(),
        title_style(),
    );
    push("--- source ---".to_string(), dim_style());
    for line in prelude.source.lines().skip(prompt.scroll) {
        push(line.to_string(), Style::default());
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

fn draw_help(frame: &mut Frame, area: Rect) {
    let overlay = centered(area, area.width.min(64), area.height.min(20));
    frame.render_widget(Clear, overlay);
    let block = Block::default()
        .borders(Borders::ALL)
        .title("Help (F1 / Esc closes)")
        .border_style(title_style());
    let inner = block.inner(overlay);
    frame.render_widget(block, overlay);
    let entries = [
        ("Tab / Shift-Tab", "cycle Composer, Tasks, Transcript"),
        ("Enter", "send the composer; select a task row"),
        ("Alt-Enter", "insert a newline"),
        ("Up/Down, Home/End", "move in the composer or scroll"),
        ("PageUp/PageDown", "scroll the transcript by a page"),
        ("Ctrl-N", "start a new task draft"),
        ("Ctrl-R", "resume the selected active task"),
        ("Ctrl-X", "queue durable cancellation"),
        ("Ctrl-C", "interrupt the local run, else clear the draft"),
        ("Ctrl-Q", "quit (interrupts the local run first)"),
    ];
    let lines: Vec<Line> = entries
        .iter()
        .map(|(key, description)| {
            Line::from(vec![
                Span::styled(format!("{key:<18}"), title_style()),
                Span::raw((*description).to_string()),
            ])
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.clamp(1, area.width);
    let height = height.clamp(1, area.height);
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

fn border_style(focused: bool) -> Style {
    if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().add_modifier(Modifier::DIM)
    }
}

fn title_style() -> Style {
    Style::default().fg(Color::Cyan)
}

fn dim_style() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

fn warn_style() -> Style {
    Style::default().fg(Color::Yellow)
}

fn level_style(level: Level) -> Style {
    match level {
        Level::Info => Style::default(),
        Level::Warn => warn_style(),
        Level::Error => Style::default().fg(Color::Red),
    }
}

fn kind_style(kind: RowKind) -> Style {
    match kind {
        RowKind::User => Style::default().fg(Color::Cyan),
        RowKind::Assistant => Style::default(),
        RowKind::Steering => Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM),
        RowKind::Activity | RowKind::Reasoning | RowKind::Hint => dim_style(),
        RowKind::Status => Style::default().fg(Color::Green),
        RowKind::Warn => warn_style(),
        RowKind::Error => Style::default().fg(Color::Red),
        RowKind::Provisional => Style::default().add_modifier(Modifier::ITALIC),
    }
}
