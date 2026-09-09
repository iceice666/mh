//! Behavioural tests for the TUI front end.
//!
//! These drive the real reducer, the real read-only observer, and the real
//! durable runtime; only the model is faked, and it is faked through the same
//! `Model` trait the provider adapter implements. Rendering is checked against
//! Ratatui's `TestBackend`, so a claim about the screen is a claim about the
//! cells that would actually be written.

#![expect(clippy::too_many_lines, reason = "scenario tests read as scripts")]

#[path = "../../tests/support/mod.rs"]
mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
use mh::agent::{Agent, AgentControl, AgentEvent, TaskOutcome};
use mh::context::CompiledContext;
use mh::identity::TaskId;
use mh::model::{
    GenerationStop, Model, ModelError, ModelEvent, ModelOutput, OpenAiResponses, ProgramLanguage,
};
use mh::ptc::prelude::PreludeOrigin;
use mh::ptc::{Prelude, TrustDecision, TrustStore};
use mh::session::{Session, SessionEvent, WorkspaceRunnerLock};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

use super::app::{Action, App, Level, Target, display_text};
use super::runtime::{RunRequest, UiEvent, execute_run, spawn_observer, tui_agent_config};
use super::{Loop, render};
use support::{config, finish, program, text};

const DEADLINE: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn press(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

fn ctrl(character: char) -> KeyEvent {
    KeyEvent {
        code: KeyCode::Char(character),
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

fn alt(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::ALT,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

fn release(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Release,
        state: KeyEventState::NONE,
    }
}

fn type_text(app: &mut App, value: &str) {
    for character in value.chars() {
        app.on_key(press(KeyCode::Char(character)));
    }
}

/// Blocks until `check` holds, failing the test rather than hanging forever.
fn until(what: &str, mut check: impl FnMut() -> bool) {
    let deadline = Instant::now() + DEADLINE;
    while Instant::now() < deadline {
        if check() {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("timed out waiting for {what}");
}

/// A model that emits scripted stream events and can be held inside a
/// generation until the test releases it.
struct StreamingModel {
    turns: Mutex<Vec<Turn>>,
    /// Signalled by `generate` once it is inside a held turn.
    entered: Sender<()>,
    release: Mutex<Receiver<()>>,
}

struct Turn {
    events: Vec<ModelEvent>,
    output: ModelOutput,
    hold: bool,
}

impl StreamingModel {
    fn new(turns: Vec<Turn>) -> (Arc<Self>, Receiver<()>, Sender<()>) {
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mut turns = turns;
        turns.reverse();
        (
            Arc::new(Self {
                turns: Mutex::new(turns),
                entered: entered_tx,
                release: Mutex::new(release_rx),
            }),
            entered_rx,
            release_tx,
        )
    }
}

impl Model for StreamingModel {
    fn generate(
        &self,
        _context: &CompiledContext,
        stop: &GenerationStop,
        events: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput, ModelError> {
        let turn = self
            .turns
            .lock()
            .pop()
            .ok_or_else(|| ModelError::Protocol("stream script exhausted".to_string()))?;
        for event in turn.events {
            events(event);
        }
        if turn.hold {
            let _ = self.entered.send(());
            let release = self.release.lock();
            // Poll the stop flag rather than sleeping: this is exactly how the
            // real adapter learns about supersede and interrupt.
            loop {
                if let Some(reason) = stop.reason() {
                    return Err(ModelError::Stopped(reason));
                }
                match release.recv_timeout(Duration::from_millis(5)) {
                    Ok(()) => break,
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
        }
        if let Some(reason) = stop.reason() {
            return Err(ModelError::Stopped(reason));
        }
        Ok(turn.output)
    }
}

fn turn(events: Vec<ModelEvent>, output: ModelOutput) -> Turn {
    Turn {
        events,
        output,
        hold: false,
    }
}

fn held(events: Vec<ModelEvent>, output: ModelOutput) -> Turn {
    Turn {
        events,
        output,
        hold: true,
    }
}

/// Drains everything currently queued into the app, like the event loop does.
fn drain(app: &mut App, events: &Receiver<UiEvent>) {
    while let Ok(event) = events.try_recv() {
        match event {
            UiEvent::Snapshot(Ok(snapshot)) => app.apply_snapshot(snapshot),
            UiEvent::Snapshot(Err(error)) => app.apply_snapshot_error(&error),
            UiEvent::Registered { run_id, task_id } => app.apply_registered(run_id, task_id),
            UiEvent::Admitted {
                run_id,
                task_id,
                after_seq,
            } => app.apply_admitted(run_id, task_id, after_seq),
            UiEvent::Agent {
                run_id,
                task_id,
                event,
            } => {
                app.apply_agent(run_id, task_id, event);
            }
            UiEvent::Trust {
                run_id,
                prelude,
                reply,
            } => app.apply_trust(run_id, prelude, reply),
            UiEvent::Receipt {
                target,
                cancel,
                result,
            } => app.apply_receipt(target, cancel, result),
        }
    }
}

/// Refreshes the app's durable projection straight from the journal.
fn observe_once(app: &mut App, workspace: &std::path::Path) {
    match Session::inspect(workspace) {
        Ok(session) => {
            let state = session.state();
            app.apply_snapshot(Some(super::runtime::Snapshot {
                session_id: state.id,
                last_seq: state.last_event_seq,
                root_task: session.root_task(),
                tasks: session.tasks(),
                events: session.events(),
                warnings: session.warnings(),
            }));
        }
        Err(mh::session::SessionError::NoSession(_)) => app.apply_snapshot(None),
        Err(error) => app.apply_snapshot_error(&error.to_string()),
    }
}

fn transcript(app: &mut App, width: u16) -> String {
    app.rows(width)
        .iter()
        .map(|row| row.text.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

fn frame(app: &mut App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| render::draw(frame, app)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    let mut out = String::new();
    for row in 0..buffer.area.height {
        for column in 0..buffer.area.width {
            out.push_str(buffer.cell((column, row)).unwrap().symbol());
        }
        out.push('\n');
    }
    out
}

fn seed_task(workspace: &std::path::Path, objective: &str) -> TaskId {
    Agent::<OpenAiResponses>::start_detached_task(workspace, objective).unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn empty_observation_is_read_only() {
    let dir = tempfile::tempdir().unwrap();
    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let observer = spawn_observer(dir.path().to_path_buf(), event_tx).unwrap();

    let first = event_rx.recv_timeout(DEADLINE).unwrap();
    assert!(matches!(first, UiEvent::Snapshot(Ok(None))));
    observer.refresh();
    // A second poll must not invent a session, and must not re-report the
    // unchanged absence either.
    assert!(matches!(
        event_rx.recv_timeout(Duration::from_millis(400)),
        Err(RecvTimeoutError::Timeout)
    ));
    observer.stop();
    assert!(!dir.path().join(".mh").exists(), "observation created .mh");

    // Without an API key the UI still paints: the first screen needs no model.
    let mut app = App::new(dir.path());
    app.apply_snapshot(None);
    let screen = frame(&mut app, 100, 24);
    assert!(screen.contains("New task"), "{screen}");
    assert!(screen.contains("No tasks yet"), "{screen}");
    assert!(!dir.path().join(".mh").exists());
}

#[test]
fn selected_task_controls_are_explicit() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();
    let task_a = seed_task(workspace, "task a");
    let task_b = seed_task(workspace, "task b");

    let mut app = App::new(workspace);
    observe_once(&mut app, workspace);
    // The session's own root is B; the user is looking at A.
    assert_eq!(app.snapshot().unwrap().root_task, Some(task_b));
    app.on_key(press(KeyCode::Tab));
    while app.target() != Target::Task(task_a) {
        app.on_key(press(KeyCode::Down));
    }
    app.on_key(press(KeyCode::Enter));
    assert_eq!(app.target(), Target::Task(task_a));

    // Durable steering goes to A, not to the session root.
    type_text(&mut app, "look here");
    let action = app.on_key(press(KeyCode::Enter)).expect("submit");
    let plan = app.plan(action);
    let super::Plan::DurableSteer { task_id, text } = plan else {
        panic!("expected durable steer, got {plan:?}");
    };
    assert_eq!(task_id, task_a);
    let receipt = mh::runtime::steer_task(workspace, Some(task_id), &text).unwrap();
    assert_eq!(receipt.task_id, task_a);

    // Cancellation likewise.
    app.apply_receipt(task_a, false, Ok(receipt));
    let cancel = app.plan(Action::Cancel(task_a));
    assert_eq!(cancel, super::Plan::DurableCancel { task_id: task_a });
    let cancel_receipt = mh::runtime::cancel_task(workspace, Some(task_a)).unwrap();
    assert_eq!(cancel_receipt.task_id, task_a);
    app.apply_receipt(task_a, true, Ok(cancel_receipt));

    let commands: Vec<TaskId> = Session::inspect(workspace)
        .unwrap()
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            SessionEvent::SteeringQueued { task_id, .. }
            | SessionEvent::TaskCancelRequested { task_id, .. } => Some(task_id),
            _ => None,
        })
        .collect();
    assert_eq!(commands, vec![task_a, task_a], "B received a command");

    // A terminal task refuses resume and keeps the draft.
    let session = Session::command(workspace).unwrap();
    session
        .append_shared_event(SessionEvent::TaskStatusChanged {
            task_id: task_a,
            status: mh::goal::TaskStatus::Cancelled,
            note: None,
        })
        .unwrap();
    observe_once(&mut app, workspace);
    type_text(&mut app, "again");
    assert_eq!(app.plan(Action::Resume(task_a)), super::Plan::Nothing);
    let submit = app.plan(Action::Submit {
        target: Target::Task(task_a),
        text: "again".to_string(),
    });
    assert_eq!(submit, super::Plan::Nothing);
    assert_eq!(app.draft().text(), "again", "draft was discarded");
    let notices = app.notices();
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("cancelled") && notice.text.contains("Ctrl-N")),
        "{notices:?}"
    );
}

#[test]
fn new_task_busy_keeps_registered_task() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().to_path_buf();
    // Establish the session, then hold the runner lock from "another process".
    Session::open(&workspace).unwrap();
    let guard = WorkspaceRunnerLock::acquire(&workspace).unwrap();

    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    let controls = std::sync::mpsc::channel();

    let mut app = App::new(&workspace);
    let plan = app.plan(Action::Submit {
        target: Target::New,
        text: "busy start".to_string(),
    });
    let super::Plan::StartRun { run_id, request } = plan else {
        panic!("expected a new run, got {plan:?}");
    };
    let agent = Agent::new(support::RoleModel::new(vec![finish("done")]), config());
    let outcome = execute_run(
        &agent,
        &workspace,
        request,
        &cancelled,
        &controls.1,
        &event_tx,
        run_id,
    );
    assert!(outcome.is_err(), "resume should have been refused");

    // The app must learn the registered id even though the run was refused.
    drain(&mut app, &event_rx);
    let task_id = app.local_task().expect("registration reported an id");
    app.worker_finished(run_id, outcome);

    let events = Session::inspect(&workspace).unwrap().events();
    let started: Vec<TaskId> = events
        .iter()
        .filter_map(|record| match &record.event {
            SessionEvent::TaskStarted { task_id, .. } => Some(*task_id),
            _ => None,
        })
        .collect();
    assert_eq!(started, vec![task_id], "registration was duplicated");
    let messages: Vec<&str> = events
        .iter()
        .filter_map(|record| match &record.event {
            SessionEvent::UserMessage { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(messages, vec!["busy start"]);
    assert!(
        app.notices()
            .iter()
            .any(|notice| notice.level == Level::Error),
        "ownership failure was not reported"
    );

    // With the foreign owner gone, the same id resumes.
    drop(guard);
    observe_once(&mut app, &workspace);
    let plan = app.plan(Action::Resume(task_id));
    let super::Plan::StartRun { run_id, request } = plan else {
        panic!("expected a resume run, got {plan:?}");
    };
    assert_eq!(request, RunRequest::Resume(task_id));
    let agent = Agent::new(support::RoleModel::new(vec![finish("done")]), config());
    let outcome = execute_run(
        &agent,
        &workspace,
        request,
        &cancelled,
        &controls.1,
        &event_tx,
        run_id,
    )
    .expect("resume succeeded");
    assert!(outcome.is_complete());
    let started_again = Session::inspect(&workspace)
        .unwrap()
        .events()
        .into_iter()
        .filter(|record| matches!(record.event, SessionEvent::TaskStarted { .. }))
        .count();
    assert_eq!(started_again, 1, "resume registered a second task");
}

#[test]
fn stream_reconciles_without_duplicate_messages() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();
    let task_id = seed_task(workspace, "stream");
    let mut app = App::new(workspace);
    observe_once(&mut app, workspace);
    app.plan(Action::Resume(task_id));
    let run_id = app.local_run_id().unwrap();

    // Baseline before the first durable ModelStarted.
    let baseline = Session::inspect(workspace).unwrap().state().last_event_seq;
    app.apply_admitted(run_id, task_id, baseline);

    let session = Session::command(workspace).unwrap();
    // Deltas arrive before the journal catches up: the provisional turn shows.
    app.apply_agent(
        run_id,
        task_id,
        AgentEvent::Model(ModelEvent::RequestStarted {
            endpoint: "local".to_string(),
        }),
    );
    app.apply_agent(
        run_id,
        task_id,
        AgentEvent::Model(ModelEvent::OutputTextDelta("修復".to_string())),
    );
    app.apply_agent(
        run_id,
        task_id,
        AgentEvent::Model(ModelEvent::OutputTextDelta("測試".to_string())),
    );
    session
        .append_shared_event(SessionEvent::ModelStarted { task_id })
        .unwrap();
    observe_once(&mut app, workspace);
    let streaming = transcript(&mut app, 60);
    assert!(streaming.contains("修復測試"), "{streaming}");
    assert!(streaming.contains("streaming"), "{streaming}");

    // The journal finalizes the same text: exactly one copy remains.
    session
        .append_shared_event(SessionEvent::AssistantMessage {
            task_id,
            content: "修復測試".to_string(),
        })
        .unwrap();
    app.apply_agent(
        run_id,
        task_id,
        AgentEvent::AssistantMessage {
            content: "修復測試".to_string(),
        },
    );
    observe_once(&mut app, workspace);
    let settled = transcript(&mut app, 60);
    assert_eq!(
        settled.matches("修復測試").count(),
        1,
        "duplicated assistant message:\n{settled}"
    );
    assert!(!settled.contains("streaming"), "{settled}");

    // A second turn with byte-identical text is a second message, not a dedup.
    session
        .append_shared_event(SessionEvent::ModelStarted { task_id })
        .unwrap();
    app.apply_agent(
        run_id,
        task_id,
        AgentEvent::Model(ModelEvent::RequestStarted {
            endpoint: "local".to_string(),
        }),
    );
    app.apply_agent(
        run_id,
        task_id,
        AgentEvent::Model(ModelEvent::OutputTextDelta("修復測試".to_string())),
    );
    session
        .append_shared_event(SessionEvent::AssistantMessage {
            task_id,
            content: "修復測試".to_string(),
        })
        .unwrap();
    observe_once(&mut app, workspace);
    let twice = transcript(&mut app, 60);
    assert_eq!(
        twice.matches("修復測試").count(),
        2,
        "second identical turn was swallowed:\n{twice}"
    );

    // Snapshot-first ordering reconciles the same way.
    session
        .append_shared_event(SessionEvent::ModelStarted { task_id })
        .unwrap();
    session
        .append_shared_event(SessionEvent::AssistantMessage {
            task_id,
            content: "late".to_string(),
        })
        .unwrap();
    observe_once(&mut app, workspace);
    app.apply_agent(
        run_id,
        task_id,
        AgentEvent::Model(ModelEvent::RequestStarted {
            endpoint: "local".to_string(),
        }),
    );
    app.apply_agent(
        run_id,
        task_id,
        AgentEvent::Model(ModelEvent::OutputTextDelta("late".to_string())),
    );
    let late = transcript(&mut app, 60);
    assert_eq!(
        late.matches("late").count(),
        1,
        "late delta duplicated a finalized turn:\n{late}"
    );
}

#[test]
fn local_steering_supersedes_and_preserves_target() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().to_path_buf();
    let other = {
        // A second root task the user will browse while A runs.
        Session::open(&workspace).unwrap();
        seed_task(&workspace, "other task")
    };
    let (model, entered, _release) = StreamingModel::new(vec![
        held(
            vec![
                ModelEvent::RequestStarted {
                    endpoint: "local".to_string(),
                },
                ModelEvent::OutputTextDelta("partial".to_string()),
            ],
            text("unused"),
        ),
        turn(vec![], finish("steered")),
    ]);
    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let (control_tx, control_rx) = std::sync::mpsc::channel();
    let cancelled = Arc::new(AtomicBool::new(false));

    let run_workspace = workspace.clone();
    let run_events = event_tx.clone();
    let run_model = model.clone();
    let run_cancelled = cancelled.clone();
    let worker = std::thread::spawn(move || {
        let agent = Agent::new(ModelHandle(run_model), config());
        execute_run(
            &agent,
            &run_workspace,
            RunRequest::New("steer me".to_string()),
            &run_cancelled,
            &control_rx,
            &run_events,
            1,
        )
    });
    entered.recv_timeout(DEADLINE).unwrap();

    let mut app = App::new(&workspace);
    app.plan(Action::Submit {
        target: Target::New,
        text: "steer me".to_string(),
    });
    drain(&mut app, &event_rx);
    let task_id = app.local_task().expect("registered");
    observe_once(&mut app, &workspace);

    // Browse the other task; the running task's stream must not leak into it.
    app.on_key(press(KeyCode::Tab));
    while app.target() != Target::Task(other) {
        app.on_key(press(KeyCode::Down));
    }
    app.on_key(press(KeyCode::Enter));
    let elsewhere = transcript(&mut app, 60);
    assert!(!elsewhere.contains("partial"), "{elsewhere}");

    // Steering the running task from its own page.
    app.on_key(ctrl('n'));
    app.select_for_test(Target::Task(task_id));
    let plan = app.plan(Action::Submit {
        target: Target::Task(task_id),
        text: "focus tests".to_string(),
    });
    assert_eq!(
        plan,
        super::Plan::LocalSteer {
            task_id,
            text: "focus tests".to_string()
        }
    );
    control_tx
        .send(AgentControl::Steer("focus tests".to_string()))
        .unwrap();
    app.local_steer_sent("focus tests".to_string());
    // The release channel is deliberately never used and never dropped: the
    // only way out of the held generation is the supersede stop, so this
    // asserts supersede actually works rather than timing out.

    let outcome = worker.join().unwrap().expect("run finished");
    assert!(outcome.is_complete());
    drain(&mut app, &event_rx);
    app.worker_finished(1, Ok(outcome));
    observe_once(&mut app, &workspace);

    // The steering was journalled exactly once and superseded the turn.
    let events = Session::inspect(&workspace).unwrap().events();
    let queued = events
        .iter()
        .filter(|record| matches!(record.event, SessionEvent::SteeringQueued { .. }))
        .count();
    let applied = events
        .iter()
        .filter(|record| matches!(record.event, SessionEvent::SteeringApplied { .. }))
        .count();
    assert_eq!((queued, applied), (1, 1), "steering was double-sent");
    assert!(
        events
            .iter()
            .any(|record| matches!(record.event, SessionEvent::ModelSuperseded { .. })),
        "the held turn was not superseded"
    );
    assert!(
        !events.iter().any(|record| matches!(
            &record.event,
            SessionEvent::AssistantMessage { content, .. } if content == "partial"
        )),
        "provisional text became a durable assistant message"
    );

    // Unconsumed steering after the worker ends is reported, never resent.
    // The completed task cannot be resumed, so this uses the still-active
    // bystander task.
    let mut orphan = App::new(&workspace);
    observe_once(&mut orphan, &workspace);
    orphan.select_for_test(Target::Task(other));
    let plan = orphan.plan(Action::Resume(other));
    let super::Plan::StartRun { run_id, .. } = plan else {
        panic!("expected a resume run, got {plan:?}");
    };
    orphan.apply_admitted(run_id, other, 0);
    orphan.local_steer_sent("never seen".to_string());
    orphan.worker_finished(run_id, Err(crate::CliError::Cancelled));
    orphan.select_for_test(Target::Task(other));
    assert!(
        orphan.draft().text().contains("never seen")
            || orphan
                .notices()
                .iter()
                .any(|notice| notice.text.contains("not applied")),
        "unapplied steering was silently dropped"
    );
}

/// Newtype so an `Arc<StreamingModel>` can be handed to `Agent::new`.
struct ModelHandle(Arc<StreamingModel>);

impl Model for ModelHandle {
    fn generate(
        &self,
        context: &CompiledContext,
        stop: &GenerationStop,
        events: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput, ModelError> {
        self.0.generate(context, stop, events)
    }
}

#[test]
fn waiting_user_followup_resumes_same_task() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().to_path_buf();
    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    let controls = std::sync::mpsc::channel();

    let agent = Agent::new(support::RoleModel::new(vec![text("waiting")]), config());
    let outcome = execute_run(
        &agent,
        &workspace,
        RunRequest::New("first".to_string()),
        &cancelled,
        &controls.1,
        &event_tx,
        1,
    )
    .expect("first run parked");
    assert!(
        matches!(outcome, TaskOutcome::AwaitingUser(_)),
        "text completed the task"
    );

    let mut app = App::new(&workspace);
    app.plan(Action::Submit {
        target: Target::New,
        text: "first".to_string(),
    });
    drain(&mut app, &event_rx);
    let task_id = app.local_task().unwrap();
    app.worker_finished(1, Ok(outcome));
    observe_once(&mut app, &workspace);
    assert_eq!(
        app.selected_view().unwrap().status().label(),
        "waiting-user"
    );

    // A follow-up message queues durably and then resumes the same task.
    let plan = app.plan(Action::Submit {
        target: Target::Task(task_id),
        text: "carry on".to_string(),
    });
    assert_eq!(
        plan,
        super::Plan::DurableSteer {
            task_id,
            text: "carry on".to_string()
        }
    );
    let receipt = mh::runtime::steer_task(&workspace, Some(task_id), "carry on").unwrap();
    app.apply_receipt(task_id, false, Ok(receipt));
    let followup = app.take_followup().expect("auto-resume was planned");
    let super::Plan::StartRun { run_id, request } = followup else {
        panic!("expected resume, got {followup:?}");
    };
    assert_eq!(request, RunRequest::Resume(task_id));

    let agent = Agent::new(support::RoleModel::new(vec![finish("done")]), config());
    let outcome = execute_run(
        &agent,
        &workspace,
        request,
        &cancelled,
        &controls.1,
        &event_tx,
        run_id,
    )
    .expect("resume completed");
    assert!(outcome.is_complete());

    let events = Session::inspect(&workspace).unwrap().events();
    assert_eq!(
        events
            .iter()
            .filter(|record| matches!(record.event, SessionEvent::TaskStarted { .. }))
            .count(),
        1,
        "the follow-up started a second task"
    );
    let (queued, applied) = (
        events
            .iter()
            .filter(|record| matches!(record.event, SessionEvent::SteeringQueued { .. }))
            .count(),
        events
            .iter()
            .filter(|record| matches!(record.event, SessionEvent::SteeringApplied { .. }))
            .count(),
    );
    assert_eq!((queued, applied), (1, 1));
    assert_eq!(
        Session::inspect(&workspace)
            .unwrap()
            .task_view(task_id)
            .status()
            .label(),
        "completed"
    );
}

#[test]
fn cancel_and_quit_join_local_worker() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().to_path_buf();
    let other = {
        Session::open(&workspace).unwrap();
        seed_task(&workspace, "bystander")
    };
    let (model, entered, _release) = StreamingModel::new(vec![held(
        vec![ModelEvent::RequestStarted {
            endpoint: "local".to_string(),
        }],
        text("never"),
    )]);
    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let (_control_tx, control_rx) = std::sync::mpsc::channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    let run_cancelled = cancelled.clone();
    let run_workspace = workspace.clone();
    let run_events = event_tx.clone();
    let worker = std::thread::spawn(move || {
        let agent = Agent::new(ModelHandle(model), config());
        execute_run(
            &agent,
            &run_workspace,
            RunRequest::New("hold".to_string()),
            &run_cancelled,
            &control_rx,
            &run_events,
            1,
        )
    });
    entered.recv_timeout(DEADLINE).unwrap();

    let mut app = App::new(&workspace);
    app.plan(Action::Submit {
        target: Target::New,
        text: "hold".to_string(),
    });
    drain(&mut app, &event_rx);
    let task_id = app.local_task().unwrap();

    // Ctrl-C interrupts the local run and keeps the UI alive.
    assert_eq!(app.plan(Action::Interrupt), super::Plan::Interrupt);
    cancelled.store(true, Ordering::Relaxed);
    let result = worker.join().unwrap();
    assert!(result.is_err(), "held run should have been cancelled");
    app.worker_finished(1, result);
    assert!(!app.is_quitting(), "interrupt closed the UI");
    observe_once(&mut app, &workspace);

    // A new run resets the flag rather than inheriting the old cancellation.
    let plan = app.plan(Action::Resume(task_id));
    let super::Plan::StartRun { run_id, request } = plan else {
        panic!("expected resume, got {plan:?}");
    };
    let (model, entered, release) = StreamingModel::new(vec![held(
        vec![ModelEvent::RequestStarted {
            endpoint: "local".to_string(),
        }],
        text("never"),
    )]);
    let restart_cancelled = Arc::new(AtomicBool::new(false));
    let run_workspace = workspace.clone();
    let run_events = event_tx.clone();
    let flag = restart_cancelled.clone();
    let worker = std::thread::spawn(move || {
        let agent = Agent::new(ModelHandle(model), config());
        execute_run(
            &agent,
            &run_workspace,
            request,
            &flag,
            &std::sync::mpsc::channel().1,
            &run_events,
            run_id,
        )
    });
    entered.recv_timeout(DEADLINE).unwrap();
    assert!(
        !restart_cancelled.load(Ordering::Relaxed),
        "the new run started already cancelled"
    );

    // Ctrl-X queues a durable receipt first, and only then interrupts.
    observe_once(&mut app, &workspace);
    assert_eq!(
        app.plan(Action::Cancel(task_id)),
        super::Plan::DurableCancel { task_id }
    );
    let receipt = mh::runtime::cancel_task(&workspace, Some(task_id)).unwrap();
    assert!(app.take_cancel_interrupt().is_none(), "interrupted early");
    app.apply_receipt(task_id, true, Ok(receipt));
    assert_eq!(app.take_cancel_interrupt(), Some(task_id));

    // Quit stops the local worker and never touches the other task.
    assert_eq!(app.plan(Action::Quit), super::Plan::Quit);
    assert!(app.is_quitting());
    assert!(app.header().contains("Stopping..."), "{}", app.header());
    restart_cancelled.store(true, Ordering::Relaxed);
    let _ = release.send(());
    let result = worker.join().unwrap();
    app.worker_finished(run_id, result);
    until("worker join", || true);
    let bystander = Session::inspect(&workspace).unwrap().task_view(other);
    assert!(
        !bystander.cancel_requested,
        "quitting cancelled an unrelated task"
    );
}

#[test]
fn prelude_decision_bridge_never_reads_stdin() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = tempfile::tempdir().unwrap();
    let store = TrustStore::at(store_dir.path().join("trust.json"));
    let prelude = Prelude {
        path: dir.path().join(".mh/prelude.js"),
        source: "//! tool doc\nfunction helper() { return 1; }\n".to_string(),
        origin: PreludeOrigin::Workspace,
    };

    // Every decisive key, and exactly what it decides.
    for (key, expected) in [
        (press(KeyCode::Char('y')), TrustDecision::Trusted),
        (press(KeyCode::Char('n')), TrustDecision::Rejected),
        (press(KeyCode::Enter), TrustDecision::Rejected),
        (press(KeyCode::Esc), TrustDecision::Rejected),
    ] {
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let config = tui_agent_config(1, event_tx);
        let confirm = config.confirm_prelude.clone().expect("callback installed");
        let asked = prelude.clone();
        let answer = std::thread::spawn(move || confirm(&asked));

        let mut app = App::new(dir.path());
        // A prompt is only honoured for the live run.
        app.plan(Action::Submit {
            target: Target::New,
            text: "prelude".to_string(),
        });
        app.apply_registered(1, TaskId(1));
        until("trust request", || {
            drain(&mut app, &event_rx);
            app.trust().is_some()
        });
        let screen = frame(&mut app, 100, 30);
        assert!(screen.contains("prelude"), "{screen}");
        assert!(screen.contains("sha256:"), "hash not shown:\n{screen}");
        assert!(!screen.contains("mh>"), "readline prompt leaked:\n{screen}");

        // A `Tab`, a stray letter, or a modified `y` is not an answer: the
        // modal must still be waiting afterwards.
        for ignored in [press(KeyCode::Tab), press(KeyCode::Char('z')), ctrl('y')] {
            app.on_key(ignored);
            assert!(
                app.trust().is_some(),
                "{ignored:?} was treated as a decision"
            );
            assert!(!answer.is_finished(), "{ignored:?} unblocked the callback");
        }

        app.on_key(key);
        assert!(app.trust().is_none(), "the modal outlived its answer");
        assert_eq!(
            answer.join().unwrap(),
            expected,
            "key {key:?} produced the wrong decision"
        );
    }

    // A dropped receiver is refusal, not consent.
    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let config = tui_agent_config(2, event_tx);
    let confirm = config.confirm_prelude.clone().unwrap();
    let asked = prelude.clone();
    let answer = std::thread::spawn(move || confirm(&asked));
    let request = event_rx.recv_timeout(DEADLINE).unwrap();
    let UiEvent::Trust { reply, .. } = request else {
        panic!("expected a trust request");
    };
    drop(reply);
    assert_eq!(answer.join().unwrap(), TrustDecision::Rejected);

    // Quitting also refuses rather than waiting for an answer nobody gives.
    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let config = tui_agent_config(3, event_tx);
    let confirm = config.confirm_prelude.clone().unwrap();
    let asked = prelude.clone();
    let answer = std::thread::spawn(move || confirm(&asked));
    let mut app = App::new(dir.path());
    app.plan(Action::Quit);
    until("quit refusal", || {
        drain(&mut app, &event_rx);
        answer.is_finished()
    });
    assert_eq!(answer.join().unwrap(), TrustDecision::Rejected);

    // Trust is still recorded per exact content by the existing store.
    store.record(&prelude, TrustDecision::Trusted).unwrap();
    assert_eq!(store.status(&prelude).unwrap(), mh::ptc::Trust::Trusted);
    let edited = Prelude {
        source: format!("{}// edit\n", prelude.source),
        ..prelude.clone()
    };
    assert_eq!(store.status(&edited).unwrap(), mh::ptc::Trust::Unknown);
}

#[test]
fn unicode_drafts_and_paste_are_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();
    let task_a = seed_task(workspace, "task a");
    let task_b = seed_task(workspace, "task b");
    let mut app = App::new(workspace);
    observe_once(&mut app, workspace);
    // Type into the New-task composer so the per-target checks below start
    // from genuinely empty drafts.
    app.select_for_test(Target::New);

    type_text(&mut app, "修復e\u{301}程式");
    assert_eq!(app.draft().text(), "修復e\u{301}程式");
    // Backspace removes the whole combining cluster, not one code point.
    app.on_key(press(KeyCode::Left));
    app.on_key(press(KeyCode::Left));
    app.on_key(press(KeyCode::Backspace));
    assert_eq!(app.draft().text(), "修復程式");
    // A release event changes nothing.
    app.on_key(release(KeyCode::Backspace));
    assert_eq!(app.draft().text(), "修復程式");

    // Display width, not char count, decides the cursor column.
    app.on_key(press(KeyCode::End));
    let (_, column) = app.draft().position();
    assert_eq!(column, 8, "wide characters counted as one cell");

    // Paste keeps indentation and newlines, and never submits.
    app.on_paste("  first\r\n\tsecond\rthird");
    assert_eq!(
        app.draft().text(),
        "修復程式  first\n\tsecond\nthird",
        "paste was normalized incorrectly"
    );
    app.on_key(alt(KeyCode::Enter));
    assert!(app.draft().text().ends_with("third\n"));

    // Enter judges emptiness on trim but submits the original text.
    let mut blank = App::new(workspace);
    type_text(&mut blank, "   ");
    assert!(blank.on_key(press(KeyCode::Enter)).is_none());
    blank.on_key(press(KeyCode::Backspace));
    type_text(&mut blank, "  keep  ");
    let action = blank.on_key(press(KeyCode::Enter)).expect("submit");
    assert_eq!(
        action,
        Action::Submit {
            target: Target::New,
            text: "    keep  ".to_string()
        }
    );

    // Drafts and scroll are per target.
    app.select_for_test(Target::Task(task_a));
    type_text(&mut app, "alpha");
    app.select_for_test(Target::Task(task_b));
    type_text(&mut app, "beta");
    assert_eq!(app.draft().text(), "beta");
    app.on_key(press(KeyCode::Tab));
    app.on_key(press(KeyCode::Tab));
    app.on_key(press(KeyCode::Up));
    let scrolled = app.scroll_offset();
    app.select_for_test(Target::Task(task_a));
    assert_eq!(app.draft().text(), "alpha");
    assert_eq!(app.scroll_offset(), 0, "scroll leaked between targets");
    app.select_for_test(Target::Task(task_b));
    assert_eq!(app.scroll_offset(), scrolled);
}

#[test]
fn readonly_corruption_remains_visible() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();
    let task_id = seed_task(workspace, "valid work");
    let session = Session::command(workspace).unwrap();
    session
        .append_shared_event(SessionEvent::AssistantMessage {
            task_id,
            content: "durable answer".to_string(),
        })
        .unwrap();
    let journal = workspace.join(".mh/session.jsonl");
    let before = std::fs::read(&journal).unwrap();

    // An unterminated tail is a warning plus the intact prefix.
    let mut torn = before.clone();
    torn.extend_from_slice(b"{\"seq\":99,\"timestamp_ms\":0,\"type\":\"model_started\"");
    std::fs::write(&journal, &torn).unwrap();
    let mut app = App::new(workspace);
    observe_once(&mut app, workspace);
    app.select_for_test(Target::Task(task_id));
    let rows = transcript(&mut app, 60);
    assert!(rows.contains("torn journal tail"), "{rows}");
    assert!(rows.contains("durable answer"), "{rows}");
    assert_eq!(
        std::fs::read(&journal).unwrap(),
        torn,
        "read-only observation repaired the journal"
    );

    // A complete but malformed record is a hard, visible error.
    let mut broken = before.clone();
    broken.extend_from_slice(b"{\"seq\":99,\"timestamp_ms\":0,\"type\":\"model_started\"}\n");
    std::fs::write(&journal, &broken).unwrap();
    observe_once(&mut app, workspace);
    assert!(app.stale().is_some(), "corrupt journal read as empty");
    let rows = transcript(&mut app, 60);
    assert!(rows.contains("corrupt session journal"), "{rows}");
    // Durable actions are refused while the view is stale.
    assert_eq!(app.plan(Action::Resume(task_id)), super::Plan::Nothing);
    assert_eq!(app.plan(Action::Cancel(task_id)), super::Plan::Nothing);
    assert_eq!(
        app.plan(Action::Submit {
            target: Target::Task(task_id),
            text: "hello".to_string()
        }),
        super::Plan::Nothing
    );
    // Quitting still works.
    assert_eq!(app.plan(Action::Quit), super::Plan::Quit);
}

#[test]
fn responsive_frames_preserve_controls() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();
    let task_id = seed_task(
        workspace,
        "渲染測試 with a very long objective that must wrap",
    );
    let session = Session::command(workspace).unwrap();
    session
        .append_shared_event(SessionEvent::AssistantMessage {
            task_id,
            content: format!("長答案 {}", "資料".repeat(80)),
        })
        .unwrap();
    session
        .append_shared_event(SessionEvent::HostCallStarted {
            task_id,
            execution_id: mh::identity::ExecutionId(1),
            call_id: 1,
            name: "read".to_string(),
            args_hash: 0,
        })
        .unwrap();
    session
        .append_shared_event(SessionEvent::TaskFailed {
            task_id,
            error: "boom \u{1b}]0;title\u{7}\u{1b}[2J".to_string(),
        })
        .unwrap();

    let mut app = App::new(workspace);
    observe_once(&mut app, workspace);
    app.select_for_test(Target::Task(task_id));

    let wide = frame(&mut app, 120, 36);
    assert!(wide.contains("Tasks"), "no sidebar at 120 cols:\n{wide}");
    assert!(wide.contains("Conversation"), "{wide}");
    assert!(wide.contains("Ctrl-Q quit"), "{wide}");
    assert!(wide.contains("Ctrl-X cancel"), "{wide}");
    assert!(!wide.contains('\u{1b}'), "raw escape reached the screen");
    assert!(!wide.contains('\u{7}'), "raw BEL reached the screen");
    assert!(wide.contains("\\x1b"), "escape not made visible:\n{wide}");
    for line in wide.lines() {
        assert!(
            line.chars().count() <= 120,
            "row overflowed the frame: {line:?}"
        );
    }

    // Narrow: full-width conversation, sidebar reachable as an overlay.
    let narrow = frame(&mut app, 80, 24);
    assert!(narrow.contains("Conversation"), "{narrow}");
    assert!(
        !narrow.contains("Tasks"),
        "sidebar shown below 90 cols:\n{narrow}"
    );
    app.on_key(press(KeyCode::Tab));
    let overlay = frame(&mut app, 80, 24);
    assert!(overlay.contains("Tasks"), "overlay missing:\n{overlay}");
    app.on_key(press(KeyCode::Esc));

    // Too small: a bounded message, and quit still works.
    let tiny = frame(&mut app, 39, 11);
    assert!(tiny.contains("Terminal too small (min 40x12)"), "{tiny}");
    assert!(tiny.contains("Ctrl-Q"), "{tiny}");
    assert_eq!(app.plan(Action::Quit), super::Plan::Quit);

    // Back to normal: unchanged structure.
    let restored = frame(&mut app, 120, 36);
    assert!(restored.contains("Tasks"), "{restored}");
    assert!(restored.contains("Ctrl-Q quit"), "{restored}");
}

#[test]
fn display_text_neutralizes_control_sequences() {
    let rendered = display_text("a\u{1b}[2Jb\u{7}c\td\ne");
    assert_eq!(rendered, "a\\x1b[2Jb\\x07c    d\ne");
}

#[test]
fn headless_loop_routes_a_plan_without_a_terminal() {
    let dir = tempfile::tempdir().unwrap();
    let (mut headless, _events) = Loop::headless(dir.path());
    // Nothing is running: Ctrl-C clears the draft rather than quitting.
    headless.action(dir.path(), Action::Interrupt);
    headless.key(dir.path(), ctrl('q'));
}

#[test]
fn program_output_is_projected_as_activity() {
    // Guards the fixture contract the other tests rely on.
    let ModelOutput::Program { language, source } = program("return 1;") else {
        panic!("program fixture changed shape");
    };
    assert_eq!(language, ProgramLanguage::JavaScript);
    assert_eq!(source, "return 1;");
}

#[test]
fn frame_rows_are_full_width() {
    // Guards against a border column being left unpainted, which the PTY
    // smoke harness can only observe indirectly.
    let dir = tempfile::tempdir().unwrap();
    let task_id = seed_task(dir.path(), "layout");
    let mut app = App::new(dir.path());
    observe_once(&mut app, dir.path());
    app.select_for_test(Target::Task(task_id));
    let rendered = frame(&mut app, 120, 36);
    for (index, line) in rendered.lines().enumerate() {
        assert_eq!(
            line.chars().count(),
            120,
            "row {index} is not full width: {line:?}"
        );
    }
}
