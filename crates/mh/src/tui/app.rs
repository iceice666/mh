//! Pure screen state and input reducer.
//!
//! Everything here is testable without a terminal, a session, or a model: the
//! app owns selection, drafts, scrolling, the durable projection it was last
//! given, and the transient stream state of the one local worker. It performs
//! no I/O — an [`Action`] is turned into a [`Plan`] that the event loop
//! executes, so validation ("that task is finished") is decided from durable
//! state rather than from whatever the executor happens to observe later.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::mpsc::Sender;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use mh::agent::{AgentEvent, TaskOutcome};
use mh::goal::TaskStatus;
use mh::identity::{ExecutionId, TaskId};
use mh::model::ModelEvent;
use mh::ptc::{Prelude, TrustDecision};
use mh::session::{CommandReceipt, SessionEvent, TaskView};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use super::runtime::{RunRequest, Snapshot};
use crate::CliError;

/// Smallest terminal the UI will paint into.
pub(super) const MIN_COLS: u16 = 40;
pub(super) const MIN_ROWS: u16 = 12;
/// Width at and above which the task sidebar is shown beside the conversation.
pub(super) const WIDE_COLS: u16 = 90;
pub(super) const SIDEBAR_COLS: u16 = 28;
pub(super) const COMPOSER_MIN: u16 = 3;
pub(super) const COMPOSER_MAX: u16 = 8;
/// Notices kept per target. Older ones are dropped, never the newest.
const NOTICE_LIMIT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Target {
    New,
    Task(TaskId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Focus {
    Composer,
    Tasks,
    Transcript,
}

/// What a key press asked for. Captured with its target and text so a later
/// selection change cannot redirect work already requested.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Action {
    Submit { target: Target, text: String },
    Resume(TaskId),
    Cancel(TaskId),
    Interrupt,
    Quit,
}

/// The one thing the event loop should do for an [`Action`], after the app has
/// checked it against durable state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Plan {
    Nothing,
    StartRun { run_id: u64, request: RunRequest },
    LocalSteer { task_id: TaskId, text: String },
    DurableSteer { task_id: TaskId, text: String },
    DurableCancel { task_id: TaskId },
    Interrupt,
    Quit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Level {
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Notice {
    pub level: Level,
    pub text: String,
}

/// Kind of a rendered transcript row. Style lives in `render`; the app only
/// decides meaning, and every kind is distinguishable by its label text too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RowKind {
    User,
    Assistant,
    Steering,
    Activity,
    Status,
    Warn,
    Error,
    Reasoning,
    Provisional,
    Hint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Row {
    pub kind: RowKind,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SidebarItem {
    pub target: Target,
    pub text: String,
    pub kind: RowKind,
    pub selected: bool,
}

/// Composer text with a grapheme-aligned cursor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Draft {
    text: String,
    /// Byte offset, always on a grapheme boundary of `text`.
    cursor: usize,
}

impl Draft {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    /// Inserts text at the cursor. CRLF and lone CR both become LF so a paste
    /// from any terminal keeps exactly the line structure the user saw.
    pub fn insert(&mut self, raw: &str) {
        let normalized = raw.replace("\r\n", "\n").replace('\r', "\n");
        self.text.insert_str(self.cursor, &normalized);
        self.cursor += normalized.len();
    }

    pub fn backspace(&mut self) {
        if let Some(previous) = self.prev_boundary(self.cursor) {
            self.text.replace_range(previous..self.cursor, "");
            self.cursor = previous;
        }
    }

    pub fn delete(&mut self) {
        if let Some(next) = self.next_boundary(self.cursor) {
            self.text.replace_range(self.cursor..next, "");
        }
    }

    pub fn left(&mut self) {
        if let Some(previous) = self.prev_boundary(self.cursor) {
            self.cursor = previous;
        }
    }

    pub fn right(&mut self) {
        if let Some(next) = self.next_boundary(self.cursor) {
            self.cursor = next;
        }
    }

    pub fn home(&mut self) {
        self.cursor = self.line_start(self.cursor);
    }

    pub fn end(&mut self) {
        self.cursor = self.line_end(self.cursor);
    }

    pub fn up(&mut self) {
        let start = self.line_start(self.cursor);
        if start == 0 {
            return;
        }
        let column = self.column(start, self.cursor);
        let previous_start = self.line_start(start - 1);
        self.cursor = self.offset_at_column(previous_start, start - 1, column);
    }

    pub fn down(&mut self) {
        let start = self.line_start(self.cursor);
        let end = self.line_end(self.cursor);
        if end >= self.text.len() {
            return;
        }
        let column = self.column(start, self.cursor);
        let next_start = end + 1;
        let next_end = self.line_end(next_start);
        self.cursor = self.offset_at_column(next_start, next_end, column);
    }

    /// Cursor as (logical line index, display column).
    pub fn position(&self) -> (usize, usize) {
        let start = self.line_start(self.cursor);
        let line = self.text[..start].matches('\n').count();
        (line, self.column(start, self.cursor))
    }

    fn prev_boundary(&self, at: usize) -> Option<usize> {
        self.text[..at]
            .grapheme_indices(true)
            .next_back()
            .map(|(index, _)| index)
    }

    fn next_boundary(&self, at: usize) -> Option<usize> {
        self.text[at..]
            .graphemes(true)
            .next()
            .map(|grapheme| at + grapheme.len())
    }

    fn line_start(&self, at: usize) -> usize {
        self.text[..at].rfind('\n').map_or(0, |index| index + 1)
    }

    fn line_end(&self, at: usize) -> usize {
        self.text[at..]
            .find('\n')
            .map_or(self.text.len(), |index| at + index)
    }

    fn column(&self, start: usize, at: usize) -> usize {
        UnicodeWidthStr::width(&self.text[start..at])
    }

    /// Byte offset within `start..end` whose display column is nearest to
    /// `column` without passing it, always on a grapheme boundary.
    fn offset_at_column(&self, start: usize, end: usize, column: usize) -> usize {
        let mut used = 0;
        let mut offset = start;
        for grapheme in self.text[start..end].graphemes(true) {
            let width = UnicodeWidthStr::width(grapheme);
            if used + width > column {
                return offset;
            }
            used += width;
            offset += grapheme.len();
        }
        end
    }
}

/// One provisional model turn, reconstructed from live stream events.
#[derive(Clone, Debug, Default)]
struct Block {
    reasoning: String,
    text: String,
    function: Option<String>,
    response_completed: bool,
    superseded: bool,
    interrupted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SteerState {
    Sent,
    Consumed,
    Applied,
    NotApplied,
}

#[derive(Clone, Debug)]
struct PendingSteer {
    text: String,
    state: SteerState,
}

/// The single local worker, if one is running.
#[derive(Debug)]
struct LocalRun {
    run_id: u64,
    task_id: Option<TaskId>,
    /// Journal sequence the run was admitted after; live turns pair with
    /// durable `ModelStarted` events beyond it.
    after_seq: Option<u64>,
    blocks: Vec<Block>,
    steering: Vec<PendingSteer>,
    activity: Option<String>,
}

impl LocalRun {
    const fn new(run_id: u64) -> Self {
        Self {
            run_id,
            task_id: None,
            after_seq: None,
            blocks: Vec::new(),
            steering: Vec::new(),
            activity: None,
        }
    }
}

/// A prelude awaiting an explicit decision. Default is refusal.
pub(super) struct TrustPrompt {
    pub run_id: u64,
    pub prelude: Prelude,
    pub scroll: usize,
    reply: Sender<TrustDecision>,
}

impl TrustPrompt {
    fn answer(self, decision: TrustDecision) {
        let _ = self.reply.send(decision);
    }
}

/// A durable command in flight. At most one at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingCommand {
    target: TaskId,
    cancel: bool,
}

#[derive(Clone, Debug, Default)]
struct Scroll {
    /// Rows scrolled up from the bottom. Zero follows new output.
    offset: usize,
    unseen: bool,
}

struct RowCache {
    key: RowKey,
    rows: Vec<Row>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RowKey {
    session: String,
    last_seq: u64,
    target: Target,
    width: u16,
    live: u64,
    stale: bool,
    warnings: usize,
}

pub(super) struct App {
    workspace: String,
    target: Target,
    focus: Focus,
    drafts: BTreeMap<Target, Draft>,
    scrolls: BTreeMap<Target, Scroll>,
    notices: BTreeMap<Target, VecDeque<Notice>>,
    snapshot: Option<Snapshot>,
    session_id: Option<String>,
    stale: Option<String>,
    local: Option<LocalRun>,
    command: Option<PendingCommand>,
    /// Target whose composer is locked while its submit is in flight.
    submitting: Option<Target>,
    /// Set when a durable steer should be followed by an explicit resume.
    resume_after_receipt: Option<TaskId>,
    followup: Option<Plan>,
    /// Message submitted after durable completion but before the local worker
    /// has been reaped.
    pending_completed_followup: Option<(TaskId, String)>,
    cancel_interrupt: Option<TaskId>,
    trust: Option<TrustPrompt>,
    help: bool,
    quitting: bool,
    fatal: bool,
    /// Set until the user re-selects a task after the journal was replaced.
    /// Durable actions stay blocked while it holds.
    session_changed: bool,
    /// One-shot edge for the event loop: the change was just observed.
    session_change_signal: bool,
    selection_initialized: bool,
    next_run_id: u64,
    live_revision: u64,
    rows: Option<RowCache>,
    dirty: bool,
    metrics: Metrics,
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    transcript_height: usize,
}

impl App {
    pub fn new(workspace: &std::path::Path) -> Self {
        Self {
            workspace: workspace.file_name().map_or_else(
                || workspace.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            ),
            target: Target::New,
            focus: Focus::Composer,
            drafts: BTreeMap::new(),
            scrolls: BTreeMap::new(),
            notices: BTreeMap::new(),
            snapshot: None,
            session_id: None,
            stale: None,
            local: None,
            command: None,
            submitting: None,
            resume_after_receipt: None,
            followup: None,
            pending_completed_followup: None,
            cancel_interrupt: None,
            trust: None,
            help: false,
            quitting: false,
            fatal: false,
            session_changed: false,
            session_change_signal: false,
            selection_initialized: false,
            next_run_id: 0,
            live_revision: 0,
            rows: None,
            dirty: true,
            metrics: Metrics {
                transcript_height: 10,
            },
        }
    }

    pub const fn target(&self) -> Target {
        self.target
    }

    pub const fn focus(&self) -> Focus {
        self.focus
    }

    pub const fn is_quitting(&self) -> bool {
        self.quitting
    }

    pub const fn help_open(&self) -> bool {
        self.help
    }

    pub const fn trust(&self) -> Option<&TrustPrompt> {
        self.trust.as_ref()
    }

    pub fn local_task(&self) -> Option<TaskId> {
        self.local.as_ref().and_then(|run| run.task_id)
    }

    #[cfg(test)]
    pub fn local_run_id(&self) -> Option<u64> {
        self.local.as_ref().map(|run| run.run_id)
    }

    #[cfg(test)]
    pub const fn stale(&self) -> Option<&String> {
        self.stale.as_ref()
    }

    #[cfg(test)]
    pub const fn snapshot(&self) -> Option<&Snapshot> {
        self.snapshot.as_ref()
    }

    /// Consumes the one-shot edge. The blocking `session_changed` gate stays
    /// set until the user selects a task again.
    pub fn take_session_change(&mut self) -> bool {
        std::mem::take(&mut self.session_change_signal)
    }

    pub fn take_followup(&mut self) -> Option<Plan> {
        self.followup.take()
    }

    pub fn take_cancel_interrupt(&mut self) -> Option<TaskId> {
        self.cancel_interrupt.take()
    }

    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// Marks the process unusable for drawing: a panic escaped somewhere.
    pub fn set_fatal(&mut self) {
        self.fatal = true;
        self.quitting = true;
        self.reject_trust();
    }

    pub const fn is_fatal(&self) -> bool {
        self.fatal
    }

    pub fn note_metrics(&mut self, transcript_height: usize) {
        self.metrics.transcript_height = transcript_height.max(1);
    }

    pub fn notice(&mut self, target: Target, level: Level, text: impl Into<String>) {
        let entry = self.notices.entry(target).or_default();
        entry.push_back(Notice {
            level,
            text: text.into(),
        });
        while entry.len() > NOTICE_LIMIT {
            entry.pop_front();
        }
        self.dirty = true;
    }

    pub fn notices(&self) -> Vec<Notice> {
        let mut notices: Vec<Notice> = self
            .notices
            .get(&self.target)
            .map(|queue| queue.iter().cloned().collect())
            .unwrap_or_default();
        if let Some(run) = self.local.as_ref() {
            let mine = run.task_id == self.task_target();
            for steer in &run.steering {
                let text = match steer.state {
                    SteerState::Sent | SteerState::Consumed if mine => {
                        "Steering pending locally (not durable yet)".to_string()
                    }
                    SteerState::NotApplied => {
                        format!("Steering not applied: {}", display_line(&steer.text))
                    }
                    _ => continue,
                };
                notices.push(Notice {
                    level: Level::Warn,
                    text,
                });
            }
        }
        notices
    }

    fn task_target(&self) -> Option<TaskId> {
        match self.target {
            Target::New => None,
            Target::Task(id) => Some(id),
        }
    }

    pub fn draft(&self) -> &Draft {
        static EMPTY: Draft = Draft {
            text: String::new(),
            cursor: 0,
        };
        self.drafts.get(&self.target).unwrap_or(&EMPTY)
    }

    fn draft_mut(&mut self) -> &mut Draft {
        self.dirty = true;
        self.drafts.entry(self.target).or_default()
    }

    /// Whether the current composer accepts edits. A submit in flight for this
    /// target locks it so a late acceptance cannot erase newer typing.
    pub fn composer_locked(&self) -> bool {
        self.submitting == Some(self.target) || self.trust.is_some() || self.fatal
    }
}

// ---------------------------------------------------------------------------
// Durable and live state intake
// ---------------------------------------------------------------------------

impl App {
    pub fn apply_snapshot(&mut self, snapshot: Option<Snapshot>) {
        self.dirty = true;
        let Some(snapshot) = snapshot else {
            // No session: an empty, read-only screen that still polls.
            if self.session_id.is_some() {
                self.mark_session_changed();
                self.reset_selection();
            }
            self.session_id = None;
            self.snapshot = None;
            self.stale = None;
            self.selection_initialized = false;
            return;
        };
        let replaced = self
            .session_id
            .as_ref()
            .is_some_and(|id| *id != snapshot.session_id);
        let rewound = self
            .snapshot
            .as_ref()
            .is_some_and(|previous| snapshot.last_seq < previous.last_seq);
        if replaced || rewound {
            self.mark_session_changed();
            self.reset_selection();
            self.selection_initialized = false;
        }
        self.session_id = Some(snapshot.session_id.clone());
        self.stale = None;
        self.snapshot = Some(snapshot);
        self.initialize_selection();
    }

    pub fn apply_snapshot_error(&mut self, error: &str) {
        self.stale = Some(error.to_string());
        self.dirty = true;
    }

    /// Records that the journal identity changed under us. Both the one-shot
    /// edge and the durable-action gate are raised; the gate is only lowered
    /// by an explicit re-selection.
    fn mark_session_changed(&mut self) {
        self.session_changed = true;
        self.session_change_signal = true;
    }

    fn reset_selection(&mut self) {
        self.target = Target::New;
        self.focus = Focus::Composer;
        self.notices.clear();
        self.scrolls.clear();
    }

    fn initialize_selection(&mut self) {
        if self.selection_initialized {
            return;
        }
        self.selection_initialized = true;
        let roots = self.root_tasks();
        let preferred = self
            .snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.root_task)
            .filter(|id| roots.iter().any(|view| view.task_id == *id))
            .or_else(|| roots.first().map(|view| view.task_id));
        if let Some(id) = preferred {
            self.target = Target::Task(id);
        }
    }

    /// Root tasks newest first. A delegated worker's task is never selectable:
    /// it is shown as a resource of its root.
    pub fn root_tasks(&self) -> Vec<&TaskView> {
        let Some(snapshot) = self.snapshot.as_ref() else {
            return Vec::new();
        };
        let mut roots: Vec<&TaskView> = snapshot
            .tasks
            .iter()
            .filter(|view| view.agent.is_root())
            .collect();
        roots.sort_by_key(|view| std::cmp::Reverse(view.task_id.0));
        roots
    }

    pub fn selected_view(&self) -> Option<&TaskView> {
        let id = self.task_target()?;
        self.snapshot
            .as_ref()?
            .tasks
            .iter()
            .find(|view| view.task_id == id)
    }

    pub fn apply_registered(&mut self, run_id: u64, task_id: TaskId) {
        self.dirty = true;
        let Some(run) = self.local.as_mut().filter(|run| run.run_id == run_id) else {
            return;
        };
        run.task_id = Some(task_id);
        self.live_revision += 1;
        self.target = Target::Task(task_id);
        if matches!(self.submitting, Some(Target::New | Target::Task(_))) {
            self.submitting = None;
        }
    }

    pub fn apply_admitted(&mut self, run_id: u64, task_id: TaskId, after_seq: u64) {
        self.dirty = true;
        if let Some(run) = self.local.as_mut().filter(|run| run.run_id == run_id) {
            run.task_id = Some(task_id);
            run.after_seq = Some(after_seq);
            self.live_revision += 1;
        }
    }

    /// Folds one live agent event into the transient stream state. Returns
    /// whether the durable projection should be refreshed.
    ///
    /// Both the run id and the task id must match the live run: an event from
    /// a superseded run never rewrites the current one's provisional turn.
    pub fn apply_agent(&mut self, run_id: u64, task_id: TaskId, event: AgentEvent) -> bool {
        self.dirty = true;
        let Some(run) = self
            .local
            .as_mut()
            .filter(|run| run.run_id == run_id && run.task_id == Some(task_id))
        else {
            return false;
        };
        self.live_revision += 1;
        match event {
            AgentEvent::Model(model) => {
                apply_model_event(run, model);
                false
            }
            AgentEvent::SteeringQueued { content } => {
                consume_steering(run, &content);
                if let Some(block) = run.blocks.last_mut()
                    && !block.response_completed
                {
                    block.superseded = true;
                }
                true
            }
            AgentEvent::SteeringApplied { content } => {
                apply_steering(run, &content);
                true
            }
            other => {
                run.activity = Some(activity_label(&other));
                true
            }
        }
    }

    /// Bridges a prelude decision to the run that asked. A prompt from a run
    /// that has already ended is refused: nothing may consent on its behalf.
    pub fn apply_trust(&mut self, run_id: u64, prelude: Prelude, reply: Sender<TrustDecision>) {
        self.dirty = true;
        let live = self.local.as_ref().is_some_and(|run| run.run_id == run_id);
        if self.quitting || self.fatal || !live {
            let _ = reply.send(TrustDecision::Rejected);
            return;
        }
        if let Some(stale) = self.trust.take() {
            stale.answer(TrustDecision::Rejected);
        }
        self.trust = Some(TrustPrompt {
            run_id,
            prelude,
            scroll: 0,
            reply,
        });
    }

    pub fn apply_receipt(
        &mut self,
        target: TaskId,
        cancel: bool,
        result: Result<CommandReceipt, String>,
    ) {
        self.dirty = true;
        self.command = None;
        if self.submitting == Some(Target::Task(target)) {
            self.submitting = None;
        }
        match result {
            Ok(receipt) => {
                let kind = if cancel { "Cancellation" } else { "Steering" };
                self.notice(
                    Target::Task(target),
                    Level::Info,
                    format!(
                        "{kind} queued: task {}, command {}, event {}",
                        receipt.task_id.0, receipt.command_id, receipt.seq
                    ),
                );
                if let Some(warning) = receipt.cache_warning {
                    self.notice(Target::Task(target), Level::Warn, display_line(&warning));
                }
                if cancel {
                    if self.local_task() == Some(target) {
                        self.cancel_interrupt = Some(target);
                    }
                } else {
                    self.drafts.entry(Target::Task(target)).or_default().clear();
                    if self.resume_after_receipt == Some(target)
                        && !self.quitting
                        && self.local.is_none()
                    {
                        let run_id = self.allocate_run(target);
                        self.followup = Some(Plan::StartRun {
                            run_id,
                            request: RunRequest::Resume(target),
                        });
                    }
                }
            }
            Err(error) => self.notice(Target::Task(target), Level::Error, display_line(&error)),
        }
        self.resume_after_receipt = None;
    }

    /// Refuses a prompt whose run has ended: an unanswered prelude question
    /// from a dead worker must never be left waiting.
    fn reject_trust_for(&mut self, run_id: u64) {
        if self
            .trust
            .as_ref()
            .is_some_and(|prompt| prompt.run_id == run_id)
        {
            self.reject_trust();
        }
    }

    /// Records how the local worker ended. No durable event is invented here:
    /// the journal already holds whatever actually happened.
    pub fn worker_finished(&mut self, run_id: u64, result: Result<TaskOutcome, CliError>) {
        self.dirty = true;
        self.reject_trust_for(run_id);
        let Some(mut run) = self.local.take().filter(|run| run.run_id == run_id) else {
            return;
        };
        let target = run.task_id.map_or(Target::New, Target::Task);
        let failed = !matches!(
            result,
            Ok(TaskOutcome::Completed(_) | TaskOutcome::AwaitingUser(_))
        );
        match result {
            Ok(TaskOutcome::Completed(summary)) => {
                self.notice(
                    target,
                    Level::Info,
                    format!("completed: {}", display_line(&summary)),
                );
                if let Some((previous_task, message)) = self.pending_completed_followup.take()
                    && Some(previous_task) == run.task_id
                    && !self.quitting
                {
                    let run_id = self.allocate_new_run();
                    self.submitting = Some(Target::Task(previous_task));
                    self.followup = Some(Plan::StartRun {
                        run_id,
                        request: RunRequest::Followup {
                            previous_task,
                            message,
                        },
                    });
                }
            }
            Ok(TaskOutcome::AwaitingUser(message)) => self.notice(
                target,
                Level::Info,
                format!("waiting-user: {}", display_line(&message)),
            ),
            Err(CliError::Cancelled) => self.notice(target, Level::Warn, "run cancelled"),
            Err(error) => self.notice(target, Level::Error, display_line(&error.to_string())),
        }
        for steer in &mut run.steering {
            if steer.state != SteerState::Applied {
                steer.state = SteerState::NotApplied;
            }
        }
        let unapplied: Vec<String> = run
            .steering
            .iter()
            .filter(|steer| steer.state == SteerState::NotApplied)
            .map(|steer| steer.text.clone())
            .collect();
        for text in unapplied {
            let draft = self.drafts.entry(target).or_default();
            if draft.is_blank() {
                draft.clear();
                draft.insert(&text);
            } else {
                self.notice(
                    target,
                    Level::Warn,
                    format!("Steering not applied: {}", display_line(&text)),
                );
            }
        }
        if failed {
            for block in &mut run.blocks {
                if !block.response_completed {
                    block.interrupted = true;
                }
            }
        }
        if self.submitting == Some(target) || self.submitting == Some(Target::New) {
            self.submitting = None;
        }
        self.live_revision += 1;
    }

    fn allocate_run(&mut self, task_id: TaskId) -> u64 {
        self.next_run_id += 1;
        let run_id = self.next_run_id;
        let mut run = LocalRun::new(run_id);
        run.task_id = Some(task_id);
        self.local = Some(run);
        self.live_revision += 1;
        run_id
    }

    fn allocate_new_run(&mut self) -> u64 {
        self.next_run_id += 1;
        let run_id = self.next_run_id;
        self.local = Some(LocalRun::new(run_id));
        self.live_revision += 1;
        run_id
    }

    /// Drops the local run without inventing an outcome: used when the journal
    /// was replaced under us and the run can no longer be trusted.
    pub fn drop_local(&mut self) {
        self.local = None;
        self.submitting = None;
        self.pending_completed_followup = None;
        self.live_revision += 1;
        self.dirty = true;
    }

    pub fn submit_accepted(&mut self, target: Target) {
        if let Some(draft) = self.drafts.get_mut(&target) {
            draft.clear();
        }
        if self.submitting == Some(target) {
            self.submitting = None;
        }
        self.dirty = true;
    }

    pub fn submit_failed(&mut self, target: Target, error: &str) {
        if self.submitting == Some(target) {
            self.submitting = None;
        }
        self.notice(target, Level::Error, display_line(error));
        if matches!(target, Target::New) {
            self.local = None;
        }
    }

    /// Records a local steering message as sent but not yet durable.
    pub fn local_steer_sent(&mut self, text: String) {
        if let Some(run) = self.local.as_mut() {
            run.steering.push(PendingSteer {
                text,
                state: SteerState::Sent,
            });
            self.live_revision += 1;
        }
        self.dirty = true;
    }
}

fn apply_model_event(run: &mut LocalRun, event: ModelEvent) {
    match event {
        ModelEvent::RequestStarted { .. } => run.blocks.push(Block::default()),
        ModelEvent::ReasoningSummaryDelta(delta) => {
            if let Some(block) = run.blocks.last_mut() {
                block.reasoning.push_str(&delta);
            }
        }
        ModelEvent::OutputTextDelta(delta) => {
            if let Some(block) = run.blocks.last_mut() {
                block.text.push_str(&delta);
            }
        }
        ModelEvent::FunctionCall { name } => {
            if let Some(block) = run.blocks.last_mut() {
                block.function = Some(name);
            }
        }
        ModelEvent::ResponseCompleted { .. } => {
            if let Some(block) = run.blocks.last_mut() {
                block.response_completed = true;
            }
        }
        ModelEvent::ResponseCreated { .. } | ModelEvent::ReasoningSummaryDone => {}
    }
}

/// Marks the oldest unconsumed local steering as seen by the worker loop.
fn consume_steering(run: &mut LocalRun, content: &str) {
    let exact = run
        .steering
        .iter()
        .position(|steer| steer.state == SteerState::Sent && steer.text == content);
    let index = exact.or_else(|| {
        run.steering
            .iter()
            .position(|steer| steer.state == SteerState::Sent)
    });
    if let Some(index) = index {
        run.steering[index].state = SteerState::Consumed;
    }
}

/// Confirms durability for the oldest consumed local steering. An applied
/// event with nothing consumed came from another process's durable command.
fn apply_steering(run: &mut LocalRun, content: &str) {
    let exact = run
        .steering
        .iter()
        .position(|steer| steer.state == SteerState::Consumed && steer.text == content);
    let index = exact.or_else(|| {
        run.steering
            .iter()
            .position(|steer| steer.state == SteerState::Consumed)
    });
    if let Some(index) = index {
        run.steering[index].state = SteerState::Applied;
    }
}

fn activity_label(event: &AgentEvent) -> String {
    match event {
        AgentEvent::PreludeLoaded { path, .. } => {
            format!("prelude {}", display_line(&path.display().to_string()))
        }
        AgentEvent::PreludeRejected { path, reason } => format!(
            "prelude not loaded: {} — {}",
            display_line(&path.display().to_string()),
            reason.explain()
        ),
        AgentEvent::ContextWindowStarted { window } => format!("context window {window}"),
        AgentEvent::Compacted { window, tokens } => {
            format!("compacted window {window} (~{tokens} tokens)")
        }
        AgentEvent::AssistantMessage { .. } => "assistant message recorded".to_string(),
        AgentEvent::FinishProposed { accepted, .. } => {
            if *accepted {
                "finish accepted".to_string()
            } else {
                "finish refused".to_string()
            }
        }
        AgentEvent::TaskCompleted { .. } => "task completed".to_string(),
        AgentEvent::GoalUpdated => "goal updated".to_string(),
        AgentEvent::AgentSpawned { agent, .. } => format!("worker {} spawned", agent.0),
        AgentEvent::AgentSettled { agent, state } => {
            format!("worker {} {}", agent.0, state.label())
        }
        AgentEvent::ProcessSpawned { process, .. } => format!("process {} spawned", process.0),
        AgentEvent::ProcessExited { process, exit_code } => match exit_code {
            Some(code) => format!("process {} exited {code}", process.0),
            None => format!("process {} exited without a status", process.0),
        },
        AgentEvent::PtcStarted => "ptc executing".to_string(),
        AgentEvent::PtcHostCallStarted { name, .. } => format!("tool {}", display_line(name)),
        AgentEvent::PtcHostCallCompleted {
            name,
            ok,
            duration_ms,
            ..
        } => format!(
            "tool {} {} ({duration_ms} ms)",
            display_line(name),
            if *ok { "ok" } else { "failed" }
        ),
        AgentEvent::EvidenceRecorded { kind, ok } => format!(
            "verify {}: {}",
            display_line(kind),
            if *ok { "pass" } else { "fail" }
        ),
        AgentEvent::RepeatedActionDetected { .. } => "repeated action detected".to_string(),
        AgentEvent::ToolCompleted {
            outcome,
            tool_calls,
            duration_ms,
        } => format!(
            "ptc {}; {tool_calls} tool call(s); {duration_ms} ms",
            display_line(outcome)
        ),
        // Stream and steering events are folded into the live block and the
        // pending-steering list instead of becoming activity text.
        AgentEvent::Model(_)
        | AgentEvent::SteeringQueued { .. }
        | AgentEvent::SteeringApplied { .. } => "model stream".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

impl App {
    /// Reduces a key event. Only `Press` (and `Repeat`, for editing) is acted
    /// on; a `Release` never triggers anything.
    pub fn on_key(&mut self, key: KeyEvent) -> Option<Action> {
        if key.kind == KeyEventKind::Release {
            return None;
        }
        self.dirty = true;
        let repeat = key.kind == KeyEventKind::Repeat;
        if self.trust.is_some() {
            return self.trust_key(key);
        }
        if !repeat && let Some(action) = self.global_key(key) {
            return Some(action);
        }
        if self.help {
            match key.code {
                KeyCode::PageUp | KeyCode::PageDown | KeyCode::Up | KeyCode::Down => {}
                _ => self.help = false,
            }
            return None;
        }
        match key.code {
            KeyCode::PageUp => {
                self.scroll_by(self.metrics.transcript_height as isize);
                return None;
            }
            KeyCode::PageDown => {
                self.scroll_by(-(self.metrics.transcript_height as isize));
                return None;
            }
            _ => {}
        }
        match self.focus {
            Focus::Composer => self.composer_key(key, repeat),
            Focus::Tasks => {
                self.tasks_key(key);
                None
            }
            Focus::Transcript => {
                self.transcript_key(key);
                None
            }
        }
    }

    pub fn on_paste(&mut self, text: &str) {
        if self.composer_locked() || self.trust.is_some() {
            return;
        }
        self.focus = Focus::Composer;
        self.draft_mut().insert(text);
    }

    pub fn on_resize(&mut self) {
        self.rows = None;
        self.dirty = true;
    }

    fn global_key(&mut self, key: KeyEvent) -> Option<Action> {
        if !key.modifiers.contains(KeyModifiers::CONTROL) {
            if key.code == KeyCode::F(1) {
                self.help = !self.help;
            }
            return None;
        }
        match key.code {
            KeyCode::Char('q' | 'Q') => Some(Action::Quit),
            KeyCode::Char('c' | 'C') => Some(Action::Interrupt),
            KeyCode::Char('r' | 'R') => match self.task_target() {
                Some(id) => Some(Action::Resume(id)),
                None => {
                    self.notice(self.target, Level::Warn, "Select an active task");
                    None
                }
            },
            KeyCode::Char('x' | 'X') => match self.task_target() {
                Some(id) => Some(Action::Cancel(id)),
                None => {
                    self.notice(self.target, Level::Warn, "Select an active task");
                    None
                }
            },
            KeyCode::Char('n' | 'N') => {
                self.select(Target::New);
                self.focus = Focus::Composer;
                None
            }
            _ => None,
        }
    }

    fn composer_key(&mut self, key: KeyEvent, repeat: bool) -> Option<Action> {
        let locked = self.composer_locked();
        match key.code {
            KeyCode::Tab if !repeat => {
                self.focus = Focus::Tasks;
                None
            }
            KeyCode::BackTab if !repeat => {
                self.focus = Focus::Transcript;
                None
            }
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => {
                if !locked {
                    self.draft_mut().insert("\n");
                }
                None
            }
            KeyCode::Enter if !repeat => {
                if locked || self.draft().is_blank() {
                    return None;
                }
                Some(Action::Submit {
                    target: self.target,
                    text: self.draft().text().to_string(),
                })
            }
            KeyCode::Char(character)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                if !locked {
                    self.draft_mut().insert(&character.to_string());
                }
                None
            }
            KeyCode::Backspace => {
                if !locked {
                    self.draft_mut().backspace();
                }
                None
            }
            KeyCode::Delete => {
                if !locked {
                    self.draft_mut().delete();
                }
                None
            }
            KeyCode::Left => {
                self.draft_mut().left();
                None
            }
            KeyCode::Right => {
                self.draft_mut().right();
                None
            }
            KeyCode::Up => {
                self.draft_mut().up();
                None
            }
            KeyCode::Down => {
                self.draft_mut().down();
                None
            }
            KeyCode::Home => {
                self.draft_mut().home();
                None
            }
            KeyCode::End => {
                self.draft_mut().end();
                None
            }
            _ => None,
        }
    }

    fn tasks_key(&mut self, key: KeyEvent) {
        let targets = self.sidebar_targets();
        let current = targets
            .iter()
            .position(|target| *target == self.target)
            .unwrap_or(0);
        match key.code {
            KeyCode::Tab => self.focus = Focus::Transcript,
            KeyCode::BackTab => self.focus = Focus::Composer,
            // Enter is the explicit "I have reviewed the list" act, which is
            // what clears the post-replacement gate on durable actions.
            KeyCode::Enter => {
                self.focus = Focus::Composer;
                self.session_changed = false;
            }
            KeyCode::Esc => self.focus = Focus::Composer,
            KeyCode::Up => self.select(targets[current.saturating_sub(1)]),
            KeyCode::Down => {
                self.select(targets[(current + 1).min(targets.len().saturating_sub(1))]);
            }
            KeyCode::Home => self.select(targets[0]),
            KeyCode::End => self.select(targets[targets.len().saturating_sub(1)]),
            _ => {}
        }
    }

    fn select(&mut self, target: Target) {
        self.target = target;
        self.session_changed = false;
    }

    /// Test hook for selecting a target directly, equivalent to moving the
    /// sidebar cursor and pressing Enter.
    #[cfg(test)]
    pub fn select_for_test(&mut self, target: Target) {
        self.select(target);
    }

    fn transcript_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Tab => self.focus = Focus::Composer,
            KeyCode::BackTab => self.focus = Focus::Tasks,
            KeyCode::Esc => self.focus = Focus::Composer,
            KeyCode::Up => self.scroll_by(1),
            KeyCode::Down => self.scroll_by(-1),
            KeyCode::Home => self.scroll_to_top(),
            KeyCode::End => self.scroll_to_bottom(),
            _ => {}
        }
    }

    fn trust_key(&mut self, key: KeyEvent) -> Option<Action> {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('q' | 'Q') if control => {
                self.reject_trust();
                Some(Action::Quit)
            }
            KeyCode::Char('c' | 'C') if control => {
                self.reject_trust();
                Some(Action::Interrupt)
            }
            // Consent must be an unmodified `y`. A modified chord is not an
            // answer, and defaulting it to yes would make the prompt theatre.
            KeyCode::Char('y' | 'Y')
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                if let Some(prompt) = self.trust.take() {
                    prompt.answer(TrustDecision::Trusted);
                }
                None
            }
            KeyCode::Char('n' | 'N') | KeyCode::Enter | KeyCode::Esc => {
                self.reject_trust();
                None
            }
            KeyCode::PageUp => {
                if let Some(prompt) = self.trust.as_mut() {
                    prompt.scroll = prompt.scroll.saturating_sub(5);
                }
                None
            }
            KeyCode::PageDown => {
                if let Some(prompt) = self.trust.as_mut() {
                    prompt.scroll += 5;
                }
                None
            }
            _ => None,
        }
    }

    fn reject_trust(&mut self) {
        if let Some(prompt) = self.trust.take() {
            prompt.answer(TrustDecision::Rejected);
        }
        self.dirty = true;
    }

    pub fn sidebar_targets(&self) -> Vec<Target> {
        let mut targets = vec![Target::New];
        targets.extend(
            self.root_tasks()
                .iter()
                .map(|view| Target::Task(view.task_id)),
        );
        targets
    }

    fn scroll_by(&mut self, delta: isize) {
        let scroll = self.scrolls.entry(self.target).or_default();
        let offset = isize::try_from(scroll.offset).unwrap_or(isize::MAX);
        scroll.offset = usize::try_from(offset.saturating_add(delta).max(0)).unwrap_or(0);
        if scroll.offset == 0 {
            scroll.unseen = false;
        }
        self.dirty = true;
    }

    fn scroll_to_top(&mut self) {
        let rows = self.rows.as_ref().map_or(0, |cache| cache.rows.len());
        let scroll = self.scrolls.entry(self.target).or_default();
        scroll.offset = rows;
        self.dirty = true;
    }

    fn scroll_to_bottom(&mut self) {
        let scroll = self.scrolls.entry(self.target).or_default();
        scroll.offset = 0;
        scroll.unseen = false;
        self.dirty = true;
    }

    pub fn scroll_offset(&self) -> usize {
        self.scrolls
            .get(&self.target)
            .map_or(0, |scroll| scroll.offset)
    }

    pub fn unseen_activity(&self) -> bool {
        self.scrolls
            .get(&self.target)
            .is_some_and(|scroll| scroll.unseen && scroll.offset > 0)
    }
}

// ---------------------------------------------------------------------------
// Action → Plan
// ---------------------------------------------------------------------------

impl App {
    pub fn plan(&mut self, action: Action) -> Plan {
        self.dirty = true;
        match action {
            Action::Quit => {
                self.quitting = true;
                self.reject_trust();
                Plan::Quit
            }
            Action::Interrupt => {
                if self.local.is_some() {
                    Plan::Interrupt
                } else {
                    self.drafts.entry(self.target).or_default().clear();
                    Plan::Nothing
                }
            }
            Action::Submit { target, text } => self.plan_submit(target, text),
            Action::Resume(task_id) => self.plan_resume(task_id),
            Action::Cancel(task_id) => self.plan_cancel(task_id),
        }
    }

    /// Whether durable, journal-writing work may start right now.
    fn durable_blocked(&mut self, target: Target) -> bool {
        if self.quitting || self.fatal {
            return true;
        }
        if self.trust.is_some() {
            self.notice(target, Level::Warn, "Answer the prelude prompt first");
            return true;
        }
        if let Some(error) = self.stale.clone() {
            self.notice(
                target,
                Level::Error,
                format!("Session state is stale: {}", display_line(&error)),
            );
            return true;
        }
        if self.session_changed {
            self.notice(
                target,
                Level::Warn,
                "Session changed; review task selection",
            );
            return true;
        }
        if self.command.is_some() {
            self.notice(target, Level::Warn, "Command pending");
            return true;
        }
        false
    }

    fn plan_submit(&mut self, target: Target, text: String) -> Plan {
        if text.trim().is_empty() || self.submitting.is_some() {
            return Plan::Nothing;
        }
        if self.durable_blocked(target) {
            return Plan::Nothing;
        }
        match target {
            Target::New => self.plan_new_run(text),
            Target::Task(task_id) => self.plan_task_submit(task_id, text),
        }
    }

    fn plan_new_run(&mut self, text: String) -> Plan {
        if self.local.is_some() {
            self.notice(Target::New, Level::Warn, "Another task is running locally");
            return Plan::Nothing;
        }
        let run_id = self.allocate_new_run();
        self.submitting = Some(Target::New);
        Plan::StartRun {
            run_id,
            request: RunRequest::New(text),
        }
    }
    fn plan_task_submit(&mut self, task_id: TaskId, text: String) -> Plan {
        let target = Target::Task(task_id);
        let Some(view) = self.selected_view().filter(|view| view.task_id == task_id) else {
            self.notice(target, Level::Error, "Task no longer exists");
            return Plan::Nothing;
        };
        let status = view.status();
        let local_here = self.local_task() == Some(task_id);
        if status == TaskStatus::Completed {
            if local_here {
                if self.pending_completed_followup.is_some() {
                    self.notice(target, Level::Warn, "Follow-up already pending");
                    return Plan::Nothing;
                }
                self.pending_completed_followup = Some((task_id, text));
                self.submitting = Some(target);
                return Plan::Nothing;
            }
            if self.local.is_some() {
                self.notice(target, Level::Warn, "Another task is running locally");
                return Plan::Nothing;
            }
            let run_id = self.allocate_new_run();
            self.submitting = Some(target);
            return Plan::StartRun {
                run_id,
                request: RunRequest::Followup {
                    previous_task: task_id,
                    message: text,
                },
            };
        }
        if !status.is_active() {
            self.notice(
                target,
                Level::Warn,
                format!("Task is {}; use Ctrl-N for a new task", status.label()),
            );
            return Plan::Nothing;
        }
        if local_here {
            if self.has_completed_event(task_id) {
                if self.pending_completed_followup.is_some() {
                    self.notice(target, Level::Warn, "Follow-up already pending");
                    return Plan::Nothing;
                }
                self.pending_completed_followup = Some((task_id, text));
                self.submitting = Some(target);
                return Plan::Nothing;
            }
            let admitted = self
                .local
                .as_ref()
                .is_some_and(|run| run.after_seq.is_some());
            if !admitted {
                self.notice(target, Level::Warn, "Starting...");
                return Plan::Nothing;
            }
            return Plan::LocalSteer { task_id, text };
        }
        // Another process may own this task. Queue durably; only take over
        // when the durable state says nobody is mid-turn.
        let resume =
            self.local.is_none() && matches!(status, TaskStatus::WaitingUser | TaskStatus::Queued);
        self.resume_after_receipt = resume.then_some(task_id);
        self.submitting = Some(target);
        self.command = Some(PendingCommand {
            target: task_id,
            cancel: false,
        });
        Plan::DurableSteer { task_id, text }
    }

    fn has_completed_event(&self, task_id: TaskId) -> bool {
        self.snapshot.as_ref().is_some_and(|snapshot| {
            snapshot.events.iter().rev().any(|record| {
                matches!(
                    record.event,
                    SessionEvent::TaskCompleted { task_id: id, .. } if id == task_id
                )
            })
        })
    }

    fn plan_resume(&mut self, task_id: TaskId) -> Plan {
        let target = Target::Task(task_id);
        if self.durable_blocked(target) {
            return Plan::Nothing;
        }
        let Some(view) = self.selected_view().filter(|view| view.task_id == task_id) else {
            self.notice(target, Level::Error, "Task no longer exists");
            return Plan::Nothing;
        };
        let status = view.status();
        if !status.is_active() {
            self.notice(
                target,
                Level::Warn,
                format!("Task is {}; use Ctrl-N for a new task", status.label()),
            );
            return Plan::Nothing;
        }
        if self.local_task() == Some(task_id) {
            self.notice(target, Level::Warn, "Already running locally");
            return Plan::Nothing;
        }
        if self.local.is_some() {
            self.notice(target, Level::Warn, "Another task is running locally");
            return Plan::Nothing;
        }
        let run_id = self.allocate_run(task_id);
        Plan::StartRun {
            run_id,
            request: RunRequest::Resume(task_id),
        }
    }

    fn plan_cancel(&mut self, task_id: TaskId) -> Plan {
        let target = Target::Task(task_id);
        if self.durable_blocked(target) {
            return Plan::Nothing;
        }
        let Some(view) = self.selected_view().filter(|view| view.task_id == task_id) else {
            self.notice(target, Level::Error, "Task no longer exists");
            return Plan::Nothing;
        };
        if !view.status().is_active() {
            self.notice(
                target,
                Level::Warn,
                format!("Task is already {}", view.status().label()),
            );
            return Plan::Nothing;
        }
        self.command = Some(PendingCommand {
            target: task_id,
            cancel: true,
        });
        Plan::DurableCancel { task_id }
    }
}

// ---------------------------------------------------------------------------
// Projection to visual rows
// ---------------------------------------------------------------------------

/// A merged timeline entry, keyed by the durable sequence it first appeared at.
struct Entry {
    seq: u64,
    kind: RowKind,
    label: &'static str,
    text: String,
}

impl App {
    pub fn header(&self) -> String {
        let mut parts = vec![format!("mh {}", self.workspace)];
        if let Some(snapshot) = self.snapshot.as_ref() {
            parts.push(format!("{} events", snapshot.last_seq));
        } else {
            parts.push("no session".to_string());
        }
        if self.local_task().is_some() {
            parts.push("local run".to_string());
        } else if self.task_target().is_some() {
            parts.push("journal view".to_string());
        }
        if self.stale.is_some() {
            parts.push("stale".to_string());
        }
        if self.session_changed {
            parts.push("session changed; review task selection".to_string());
        }
        if self.quitting {
            parts.push("Stopping...".to_string());
        }
        parts.join(" · ")
    }

    pub fn footer(&self) -> String {
        "Tab focus · Enter send · Alt-Enter newline · Ctrl-R resume · Ctrl-X cancel · \
         Ctrl-N new · Ctrl-C interrupt · Ctrl-Q quit · F1 help"
            .to_string()
    }

    pub fn info_line(&self) -> String {
        let Some(view) = self.selected_view() else {
            return match self.target {
                Target::New => "New task".to_string(),
                Target::Task(id) => format!("task #{} (not in this session)", id.0),
            };
        };
        let mut parts = vec![
            format!("task #{}", view.task_id.0),
            view.status().label().to_string(),
            format!("window {}", view.window),
            format!("rev {}", short_revision(&view.goal.current_revision.0)),
        ];
        let workers = view.unresolved_agents().len();
        if workers > 0 {
            parts.push(format!("{workers} unresolved worker(s)"));
        }
        let processes = view.running_processes().len();
        if processes > 0 {
            parts.push(format!("{processes} running process(es)"));
        }
        if !view.pending_steering.is_empty() {
            parts.push(format!("{} pending steering", view.pending_steering.len()));
        }
        if view.cancel_requested {
            parts.push("cancel requested".to_string());
        }
        if self.local_task() == Some(view.task_id) {
            parts.push("local".to_string());
        }
        parts.join(" · ")
    }

    pub fn sidebar_items(&self) -> Vec<SidebarItem> {
        let mut items = vec![SidebarItem {
            target: Target::New,
            text: "New task".to_string(),
            kind: RowKind::Hint,
            selected: self.target == Target::New,
        }];
        let local = self.local_task();
        for view in self.root_tasks() {
            let mut text = format!(
                "#{} {} {}",
                view.task_id.0,
                view.status().label(),
                display_line(view.objective())
            );
            if local == Some(view.task_id) {
                text.push_str(" [local]");
            }
            items.push(SidebarItem {
                target: Target::Task(view.task_id),
                text,
                kind: status_kind(view.status()),
                selected: self.target == Target::Task(view.task_id),
            });
        }
        if items.len() == 1 {
            items.push(SidebarItem {
                target: Target::New,
                text: "No tasks yet".to_string(),
                kind: RowKind::Hint,
                selected: false,
            });
        }
        items
    }

    /// Visual transcript rows for the selected target, rebuilt only when the
    /// data or the available width changed.
    pub fn rows(&mut self, width: u16) -> &[Row] {
        let key = RowKey {
            session: self.session_id.clone().unwrap_or_default(),
            last_seq: self
                .snapshot
                .as_ref()
                .map_or(0, |snapshot| snapshot.last_seq),
            target: self.target,
            width,
            live: self.live_revision,
            stale: self.stale.is_some(),
            warnings: self
                .snapshot
                .as_ref()
                .map_or(0, |snapshot| snapshot.warnings.len()),
        };
        let fresh = self.rows.as_ref().is_some_and(|cache| cache.key == key);
        if !fresh {
            let rows = self.build_rows(width);
            let grew = self
                .rows
                .as_ref()
                .is_some_and(|cache| rows.len() > cache.rows.len());
            self.rows = Some(RowCache { key, rows });
            if grew
                && let Some(scroll) = self.scrolls.get_mut(&self.target)
                && scroll.offset > 0
            {
                scroll.unseen = true;
            }
        }
        self.rows
            .as_ref()
            .map_or(&[], |cache| cache.rows.as_slice())
    }

    fn build_rows(&self, width: u16) -> Vec<Row> {
        let content = usize::from(width.max(8)).saturating_sub(1);
        let mut rows = Vec::new();
        if let Some(error) = self.stale.as_ref() {
            push_wrapped(
                &mut rows,
                RowKind::Error,
                "session",
                &display_text(error),
                content,
            );
        }
        for warning in self
            .snapshot
            .as_ref()
            .map(|snapshot| snapshot.warnings.as_slice())
            .unwrap_or_default()
        {
            push_wrapped(
                &mut rows,
                RowKind::Warn,
                "warning",
                &display_text(warning),
                content,
            );
        }
        let Some(task_id) = self.task_target() else {
            push_wrapped(
                &mut rows,
                RowKind::Hint,
                "new task",
                "Type an objective and press Enter. Alt-Enter inserts a newline.",
                content,
            );
            return rows;
        };
        for entry in self.timeline(task_id) {
            push_wrapped(&mut rows, entry.kind, entry.label, &entry.text, content);
        }
        self.push_live_rows(&mut rows, task_id, content);
        rows
    }

    /// Durable conversation and activity for one task, merged and ordered.
    fn timeline(&self, task_id: TaskId) -> Vec<Entry> {
        let Some(snapshot) = self.snapshot.as_ref() else {
            return Vec::new();
        };
        let mut conversation = Vec::new();
        let mut current = Some(task_id);
        while let Some(id) = current {
            if conversation.contains(&id) {
                break;
            }
            conversation.push(id);
            current = snapshot
                .tasks
                .iter()
                .find(|view| view.task_id == id)
                .and_then(|view| view.previous_task);
        }
        let mut entries: Vec<Entry> = Vec::new();
        let mut steering: HashMap<u64, usize> = HashMap::new();
        let mut calls: HashMap<(ExecutionId, u64), usize> = HashMap::new();
        let mut executions: HashMap<ExecutionId, usize> = HashMap::new();
        for record in &snapshot.events {
            let event = &record.event;
            let belongs = event.task_id().is_some_and(|id| conversation.contains(&id))
                || matches!(event, SessionEvent::AgentSpawned { parent_task_id, .. }
                    if conversation.contains(parent_task_id));
            if !belongs {
                continue;
            }
            project_event(
                record.seq,
                event,
                &mut entries,
                &mut steering,
                &mut calls,
                &mut executions,
            );
        }
        entries.sort_by_key(|entry| entry.seq);
        entries
    }

    /// Live rows: the provisional turn and any local activity the journal has
    /// not caught up with. A turn whose durable assistant message already
    /// exists contributes no second copy of that text.
    fn push_live_rows(&self, rows: &mut Vec<Row>, task_id: TaskId, content: usize) {
        let Some(run) = self
            .local
            .as_ref()
            .filter(|run| run.task_id == Some(task_id))
        else {
            return;
        };
        let starts = self.model_started_seqs(task_id, run.after_seq);
        let last = run.blocks.len().saturating_sub(1);
        for (index, block) in run.blocks.iter().enumerate() {
            let bounds = starts
                .get(index)
                .map(|start| (*start, starts.get(index + 1).copied().unwrap_or(u64::MAX)));
            let finalized = bounds.is_some_and(|(start, end)| {
                self.has_event(task_id, start, end, |event| {
                    matches!(event, SessionEvent::AssistantMessage { .. })
                })
            });
            let superseded = block.superseded
                || bounds.is_some_and(|(start, end)| {
                    self.has_event(task_id, start, end, |event| {
                        matches!(event, SessionEvent::ModelSuperseded { .. })
                    })
                });
            if finalized && index != last {
                continue;
            }
            if !block.reasoning.is_empty() {
                push_wrapped(
                    rows,
                    RowKind::Reasoning,
                    "Reasoning summary",
                    &display_text(&block.reasoning),
                    content,
                );
            }
            if finalized {
                continue;
            }
            if !block.text.is_empty() {
                let label = if superseded {
                    "Assistant · superseded"
                } else if block.interrupted {
                    "Assistant · interrupted"
                } else {
                    "Assistant · streaming"
                };
                push_wrapped(
                    rows,
                    RowKind::Provisional,
                    label,
                    &display_text(&block.text),
                    content,
                );
            }
            if let Some(name) = block.function.as_ref() {
                push_wrapped(
                    rows,
                    RowKind::Activity,
                    "Provider tool",
                    &display_line(name),
                    content,
                );
            }
        }
        if let Some(activity) = run.activity.as_ref() {
            push_wrapped(rows, RowKind::Activity, "Live", activity, content);
        }
        if run.after_seq.is_none() {
            push_wrapped(rows, RowKind::Hint, "Local", "Starting...", content);
        }
    }

    fn model_started_seqs(&self, task_id: TaskId, after_seq: Option<u64>) -> Vec<u64> {
        let (Some(snapshot), Some(after)) = (self.snapshot.as_ref(), after_seq) else {
            return Vec::new();
        };
        snapshot
            .events
            .iter()
            .filter(|record| record.seq > after)
            .filter(|record| {
                matches!(&record.event, SessionEvent::ModelStarted { task_id: id } if *id == task_id)
            })
            .map(|record| record.seq)
            .collect()
    }

    fn has_event(
        &self,
        task_id: TaskId,
        after: u64,
        before: u64,
        matches: impl Fn(&SessionEvent) -> bool,
    ) -> bool {
        self.snapshot.as_ref().is_some_and(|snapshot| {
            snapshot.events.iter().any(|record| {
                record.seq > after
                    && record.seq < before
                    && record.event.task_id() == Some(task_id)
                    && matches(&record.event)
            })
        })
    }

    /// Composer text laid out for `width`, plus the cursor's visual cell.
    pub fn composer_view(&self, width: u16) -> (Vec<String>, (usize, usize)) {
        let content = usize::from(width.max(4));
        let draft = self.draft();
        let (line, column) = draft.position();
        let mut rows: Vec<String> = Vec::new();
        let mut cursor = (0, 0);
        for (index, logical) in draft.text().split('\n').enumerate() {
            let wrapped = wrap(&display_text(logical), content);
            if index == line {
                let mut remaining = column;
                let mut row = rows.len();
                for (offset, piece) in wrapped.iter().enumerate() {
                    let width_of = UnicodeWidthStr::width(piece.as_str());
                    if remaining <= width_of || offset + 1 == wrapped.len() {
                        row = rows.len() + offset;
                        break;
                    }
                    remaining -= width_of;
                }
                cursor = (row, remaining.min(content.saturating_sub(1)));
            }
            rows.extend(wrapped);
        }
        if rows.is_empty() {
            rows.push(String::new());
        }
        (rows, cursor)
    }
}

fn status_kind(status: TaskStatus) -> RowKind {
    match status {
        TaskStatus::Completed => RowKind::Status,
        TaskStatus::Failed | TaskStatus::Cancelled => RowKind::Error,
        TaskStatus::WaitingUser | TaskStatus::Blocked => RowKind::Warn,
        _ => RowKind::Activity,
    }
}

fn short_revision(revision: &str) -> String {
    let trimmed = revision.trim();
    if trimmed.is_empty() {
        return "-".to_string();
    }
    trimmed.chars().take(12).collect()
}

#[expect(clippy::too_many_lines, reason = "one arm per projected journal event")]
fn project_event(
    seq: u64,
    event: &SessionEvent,
    entries: &mut Vec<Entry>,
    steering: &mut HashMap<u64, usize>,
    calls: &mut HashMap<(ExecutionId, u64), usize>,
    executions: &mut HashMap<ExecutionId, usize>,
) {
    match event {
        SessionEvent::UserMessage { content, .. } => entries.push(Entry {
            seq,
            kind: RowKind::User,
            label: "You",
            text: display_text(content),
        }),
        SessionEvent::AssistantMessage { content, .. } => entries.push(Entry {
            seq,
            kind: RowKind::Assistant,
            label: "Assistant",
            text: display_text(content),
        }),
        SessionEvent::SteeringQueued {
            command_id,
            content,
            ..
        } => {
            steering.insert(*command_id, entries.len());
            entries.push(Entry {
                seq,
                kind: RowKind::Steering,
                label: "You · steering",
                text: format!("{} (queued)", display_text(content)),
            });
        }
        SessionEvent::SteeringApplied { command_id, .. } => {
            if let Some(index) = steering.get(command_id) {
                let entry = &mut entries[*index];
                entry.text = entry.text.replace(" (queued)", " (applied)");
            }
        }
        SessionEvent::HostCallStarted {
            execution_id,
            call_id,
            name,
            ..
        } => {
            calls.insert((*execution_id, *call_id), entries.len());
            entries.push(Entry {
                seq,
                kind: RowKind::Activity,
                label: "Tool",
                text: format!("{} running", display_line(name)),
            });
        }
        SessionEvent::ToolCompleted {
            execution_id,
            call_id,
            name,
            ok,
            duration_ms,
            ..
        } => {
            let text = format!(
                "{} {} ({duration_ms} ms)",
                display_line(name),
                if *ok { "ok" } else { "failed" }
            );
            match calls.get(&(*execution_id, *call_id)) {
                Some(index) => entries[*index].text = text,
                None => entries.push(Entry {
                    seq,
                    kind: RowKind::Activity,
                    label: "Tool",
                    text,
                }),
            }
        }
        SessionEvent::PtcStarted { execution_id, .. } => {
            executions.insert(*execution_id, entries.len());
            entries.push(Entry {
                seq,
                kind: RowKind::Activity,
                label: "Program",
                text: format!("execution {} running", execution_id.0),
            });
        }
        SessionEvent::PtcCompleted {
            execution_id,
            tool_calls,
            duration_ms,
            ..
        } => {
            let text = format!(
                "execution {} ok; {tool_calls} tool call(s); {duration_ms} ms",
                execution_id.0
            );
            set_or_push(
                entries,
                executions.get(execution_id),
                seq,
                RowKind::Activity,
                "Program",
                text,
            );
        }
        SessionEvent::PtcFailed {
            execution_id,
            error,
            tool_calls,
            duration_ms,
            ..
        } => {
            let text = format!(
                "execution {} failed: {}; {tool_calls} tool call(s); {duration_ms} ms",
                execution_id.0,
                display_line(error)
            );
            set_or_push(
                entries,
                executions.get(execution_id),
                seq,
                RowKind::Error,
                "Program",
                text,
            );
        }
        SessionEvent::TaskCompleted {
            summary,
            unresolved,
            ..
        } => {
            let mut text = display_text(summary);
            if !unresolved.is_empty() {
                text.push_str(&format!("\nunresolved: {}", unresolved.join("; ")));
            }
            entries.push(Entry {
                seq,
                kind: RowKind::Status,
                label: "Task completed",
                text,
            });
        }
        SessionEvent::TaskFailed { error, .. } => entries.push(Entry {
            seq,
            kind: RowKind::Error,
            label: "Task failed",
            text: display_text(error),
        }),
        SessionEvent::TaskInterrupted { .. } => entries.push(Entry {
            seq,
            kind: RowKind::Warn,
            label: "Task",
            text: "interrupted".to_string(),
        }),
        SessionEvent::TaskCancelRequested { command_id, .. } => entries.push(Entry {
            seq,
            kind: RowKind::Warn,
            label: "Task",
            text: format!("cancel requested (command {command_id})"),
        }),
        SessionEvent::TaskStatusChanged { status, note, .. } => entries.push(Entry {
            seq,
            kind: status_kind(*status),
            label: "Status",
            text: match note {
                Some(note) => format!("{} — {}", status.label(), display_line(note)),
                None => status.label().to_string(),
            },
        }),
        SessionEvent::ContextWindowStarted { window, .. } => entries.push(Entry {
            seq,
            kind: RowKind::Activity,
            label: "Window",
            text: format!("context window {window}"),
        }),
        SessionEvent::Compacted { window, .. } => entries.push(Entry {
            seq,
            kind: RowKind::Activity,
            label: "Window",
            text: format!("compacted window {window}"),
        }),
        SessionEvent::ModelSuperseded { .. } => entries.push(Entry {
            seq,
            kind: RowKind::Activity,
            label: "Model",
            text: "request superseded".to_string(),
        }),
        SessionEvent::FinishProposed { verdict, .. } => {
            let mut text = if verdict.accepted {
                "accepted".to_string()
            } else {
                "refused".to_string()
            };
            for objection in &verdict.objections {
                text.push_str(&format!("\n{}", display_line(&objection.explain())));
            }
            entries.push(Entry {
                seq,
                kind: if verdict.accepted {
                    RowKind::Status
                } else {
                    RowKind::Warn
                },
                label: "Finish",
                text,
            });
        }
        SessionEvent::EvidenceRecorded { evidence } => entries.push(Entry {
            seq,
            kind: if evidence.ok {
                RowKind::Status
            } else {
                RowKind::Error
            },
            label: "Verify",
            text: format!(
                "{}: {}",
                display_line(&evidence.kind),
                if evidence.ok { "pass" } else { "fail" }
            ),
        }),
        SessionEvent::AgentSpawned {
            agent,
            objective,
            access,
            ..
        } => entries.push(Entry {
            seq,
            kind: RowKind::Activity,
            label: "Worker",
            text: format!(
                "{} spawned ({}): {}",
                agent.0,
                access.label(),
                display_line(objective)
            ),
        }),
        SessionEvent::AgentStateChanged { agent, state, .. } => entries.push(Entry {
            seq,
            kind: RowKind::Activity,
            label: "Worker",
            text: format!("{} {}", agent.0, state.label()),
        }),
        SessionEvent::ProcessSpawned { process, argv, .. } => entries.push(Entry {
            seq,
            kind: RowKind::Activity,
            label: "Process",
            text: format!("{} {}", process.0, display_line(&argv.join(" "))),
        }),
        SessionEvent::ProcessStateChanged {
            process,
            state,
            exit_code,
            ..
        } => entries.push(Entry {
            seq,
            kind: RowKind::Activity,
            label: "Process",
            text: match exit_code {
                Some(code) => format!("{} {} (exit {code})", process.0, display_line(state)),
                None => format!("{} {}", process.0, display_line(state)),
            },
        }),
        _ => {}
    }
}

fn set_or_push(
    entries: &mut Vec<Entry>,
    index: Option<&usize>,
    seq: u64,
    kind: RowKind,
    label: &'static str,
    text: String,
) {
    match index {
        Some(index) => {
            entries[*index].kind = kind;
            entries[*index].text = text;
        }
        None => entries.push(Entry {
            seq,
            kind,
            label,
            text,
        }),
    }
}

/// Appends `text` as labelled, wrapped rows. Continuations are indented so a
/// multi-line message is still visibly one message.
fn push_wrapped(rows: &mut Vec<Row>, kind: RowKind, label: &str, text: &str, width: usize) {
    let width = width.max(8);
    rows.push(Row {
        kind,
        text: format!("{label}:"),
    });
    for logical in text.split('\n') {
        let wrapped = wrap(logical, width.saturating_sub(2));
        if wrapped.is_empty() {
            rows.push(Row {
                kind,
                text: String::new(),
            });
            continue;
        }
        for piece in wrapped {
            rows.push(Row {
                kind,
                text: format!("  {piece}"),
            });
        }
    }
}

/// Wraps to `width` display cells without splitting a grapheme cluster,
/// preferring a space boundary when one is available on the line.
pub(super) fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut used = 0;
    let mut last_space: Option<(usize, usize)> = None;
    for grapheme in text.graphemes(true) {
        let cells = UnicodeWidthStr::width(grapheme);
        if used + cells > width && !current.is_empty() {
            match last_space.filter(|(byte, _)| *byte > 0) {
                Some((byte, _)) => {
                    let rest = current.split_off(byte);
                    lines.push(current.trim_end().to_string());
                    current = rest.trim_start().to_string();
                    used = UnicodeWidthStr::width(current.as_str());
                }
                None => {
                    lines.push(std::mem::take(&mut current));
                    used = 0;
                }
            }
            last_space = None;
        }
        if grapheme == " " {
            last_space = Some((current.len(), used));
        }
        current.push_str(grapheme);
        used += cells;
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Renders untrusted text safely: control sequences become visible escapes, so
/// model or tool output cannot move the cursor or drive the terminal.
pub(super) fn display_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for character in input.chars() {
        let code = character as u32;
        match character {
            '\n' => out.push('\n'),
            '\t' => out.push_str("    "),
            _ if code < 0x20 || code == 0x7f => {
                out.push_str(&format!("\\x{code:02x}"));
            }
            _ if (0x80..=0x9f).contains(&code) => {
                out.push_str(&format!("\\u{{{code:02x}}}"));
            }
            _ => out.push(character),
        }
    }
    out
}

/// Single-line variant for labels, paths and objectives.
pub(super) fn display_line(input: &str) -> String {
    display_text(input).replace('\n', " ")
}
