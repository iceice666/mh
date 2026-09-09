//! Terminal boundary: raw mode, alternate screen, bracketed paste.
//!
//! Every step that succeeds is recorded individually, so a failure halfway
//! through initialization reverses exactly what was applied. Mouse capture is
//! deliberately never enabled: the terminal's own text selection stays usable.

use std::io::{self, Stdout, Write};

use crossterm::cursor::{Hide, Show};
use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use crossterm::{ExecutableCommand, QueueableCommand};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

pub(super) type Screen = Terminal<CrosstermBackend<Stdout>>;

/// Restores whatever the TUI turned on, in reverse order.
#[derive(Debug, Default)]
pub(super) struct TerminalGuard {
    raw: bool,
    alternate: bool,
    paste: bool,
    cursor_hidden: bool,
}

impl TerminalGuard {
    /// Reverses the applied terminal state. Idempotent, and safe to call from a
    /// panic hook: each step is best effort and independent.
    fn leave(&mut self) {
        let mut out = io::stdout();
        if self.paste {
            let _ = out.queue(DisableBracketedPaste);
            self.paste = false;
        }
        if self.cursor_hidden {
            let _ = out.queue(Show);
            self.cursor_hidden = false;
        }
        if self.alternate {
            let _ = out.queue(LeaveAlternateScreen);
            self.alternate = false;
        }
        let _ = out.flush();
        if self.raw {
            let _ = disable_raw_mode();
            self.raw = false;
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.leave();
    }
}

/// Enters the full-screen TUI, or restores everything it managed to apply and
/// returns the original error.
pub(super) fn enter() -> io::Result<(Screen, TerminalGuard)> {
    let mut guard = TerminalGuard::default();
    let mut out = io::stdout();
    let mut step = || -> io::Result<()> {
        enable_raw_mode()?;
        guard.raw = true;
        out.execute(EnterAlternateScreen)?;
        guard.alternate = true;
        out.execute(EnableBracketedPaste)?;
        guard.paste = true;
        out.execute(Hide)?;
        guard.cursor_hidden = true;
        Ok(())
    };
    if let Err(error) = step() {
        guard.leave();
        return Err(error);
    }
    match Terminal::new(CrosstermBackend::new(io::stdout())) {
        Ok(terminal) => Ok((terminal, guard)),
        Err(error) => {
            guard.leave();
            Err(error)
        }
    }
}

/// Best-effort restore for the panic hook, which has no access to the guard.
pub(super) fn restore_best_effort() {
    let mut out = io::stdout();
    let _ = out.queue(DisableBracketedPaste);
    let _ = out.queue(Show);
    let _ = out.queue(LeaveAlternateScreen);
    let _ = out.flush();
    let _ = disable_raw_mode();
}
