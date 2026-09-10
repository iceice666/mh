//! Bridge to the durable runtime: read-only observation plus one local worker.
//!
//! Observation and execution are deliberately separate. The observer only ever
//! calls [`Session::inspect`]/`refresh`, so browsing a workspace never creates
//! `.mh`, never takes the runner lock, and never repairs a torn journal. The
//! worker is the only thing that writes, and it always names the exact task id
//! it was given rather than relying on implicit root selection.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use mh::agent::{Agent, AgentConfig, AgentControl, AgentError, AgentEvent, TaskOutcome};
use mh::identity::TaskId;
use mh::model::{Model, OpenAiResponses};
use mh::ptc::{Prelude, TrustDecision};
use mh::session::{CommandReceipt, EventRecord, Session, SessionError, TaskView};

use crate::CliError;

/// How often the observer re-reads the journal when nothing asked it to.
const POLL: Duration = Duration::from_millis(250);

/// An owned, read-only projection of the journal.
#[derive(Clone, Debug)]
pub(super) struct Snapshot {
    /// Session identity, so a replaced journal is recognized rather than
    /// silently merged into the previous one.
    pub session_id: String,
    pub last_seq: u64,
    pub root_task: Option<TaskId>,
    pub tasks: Vec<TaskView>,
    pub events: Vec<EventRecord>,
    pub warnings: Vec<String>,
}

/// Everything the UI thread consumes. Payloads are owned: nothing borrows from
/// a worker that may already have exited.
pub(super) enum UiEvent {
    Snapshot(Result<Option<Snapshot>, String>),
    Registered {
        run_id: u64,
        task_id: TaskId,
    },
    Admitted {
        run_id: u64,
        task_id: TaskId,
        after_seq: u64,
    },
    Agent {
        run_id: u64,
        /// The exact task the worker was driving, filled in by the worker
        /// itself so a stale run's stream is never attributed to a new one.
        task_id: TaskId,
        event: AgentEvent,
    },
    Trust {
        run_id: u64,
        prelude: Prelude,
        reply: Sender<TrustDecision>,
    },
    Receipt {
        target: TaskId,
        cancel: bool,
        result: Result<CommandReceipt, String>,
    },
}

pub(super) enum ObserverControl {
    Refresh,
    Stop,
}

/// What a local run should do once it owns the workspace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum RunRequest {
    New(String),
    Resume(TaskId),
    Followup {
        previous_task: TaskId,
        message: String,
    },
}

pub(super) struct Observer {
    control: Sender<ObserverControl>,
    handle: JoinHandle<()>,
}

impl Observer {
    pub fn refresh(&self) {
        let _ = self.control.send(ObserverControl::Refresh);
    }

    pub fn stop(self) {
        let _ = self.control.send(ObserverControl::Stop);
        let _ = self.handle.join();
    }
}

/// Starts the read-only observer. It reports immediately, then on every tick or
/// explicit refresh, and only when the observable state actually changed.
pub(super) fn spawn_observer(
    workspace: PathBuf,
    events: Sender<UiEvent>,
) -> Result<Observer, CliError> {
    let (control_tx, control_rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("mh-tui-observer".to_string())
        .spawn(move || observe(&workspace, &events, &control_rx))
        .map_err(CliError::Io)?;
    Ok(Observer {
        control: control_tx,
        handle,
    })
}

/// Last observable state, used to suppress identical projections.
#[derive(PartialEq, Eq)]
struct Seen {
    session_id: Option<String>,
    last_seq: Option<u64>,
    warnings: Vec<String>,
    error: Option<String>,
}

fn observe(workspace: &Path, events: &Sender<UiEvent>, control: &Receiver<ObserverControl>) {
    let mut session: Option<Session> = None;
    let mut seen = Seen {
        session_id: None,
        last_seq: None,
        warnings: Vec::new(),
        error: Some("<unobserved>".to_string()),
    };
    loop {
        let outcome = match session.as_ref() {
            // The handle is reused so a long session is not re-read from
            // scratch every quarter second.
            Some(open) => open.refresh().map_err(|error| describe(&error)),
            None => match Session::inspect(workspace) {
                Ok(open) => {
                    session = Some(open);
                    Ok(())
                }
                Err(SessionError::NoSession(_)) => Err(String::new()),
                Err(error) => Err(describe(&error)),
            },
        };
        match outcome {
            Ok(()) => {
                let open = session.as_ref().expect("session present after refresh");
                let state = open.state();
                let warnings = open.warnings();
                let changed = seen.session_id.as_ref() != Some(&state.id)
                    || seen.last_seq != Some(state.last_event_seq)
                    || seen.warnings != warnings
                    || seen.error.is_some();
                if changed {
                    let snapshot = Snapshot {
                        session_id: state.id.clone(),
                        last_seq: state.last_event_seq,
                        root_task: open.root_task(),
                        tasks: open.tasks(),
                        events: open.events(),
                        warnings: warnings.clone(),
                    };
                    seen = Seen {
                        session_id: Some(state.id),
                        last_seq: Some(state.last_event_seq),
                        warnings,
                        error: None,
                    };
                    if events.send(UiEvent::Snapshot(Ok(Some(snapshot)))).is_err() {
                        return;
                    }
                }
            }
            // Empty string marks "no session": an absence, not a failure.
            Err(error) if error.is_empty() => {
                if seen.session_id.is_some() || seen.error != Some(String::new()) {
                    seen = Seen {
                        session_id: None,
                        last_seq: None,
                        warnings: Vec::new(),
                        error: Some(String::new()),
                    };
                    if events.send(UiEvent::Snapshot(Ok(None))).is_err() {
                        return;
                    }
                }
            }
            Err(error) => {
                // Drop the handle: the next pass re-inspects, which is what a
                // replaced or truncated journal needs.
                session = None;
                if seen.error.as_ref() != Some(&error) {
                    seen.error = Some(error.clone());
                    if events.send(UiEvent::Snapshot(Err(error))).is_err() {
                        return;
                    }
                }
            }
        }
        match control.recv_timeout(POLL) {
            Ok(ObserverControl::Refresh) | Err(RecvTimeoutError::Timeout) => {}
            Ok(ObserverControl::Stop) | Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn describe(error: &SessionError) -> String {
    error.to_string()
}

/// The one local worker. Its exact task id is known as soon as registration
/// commits; before that it is `None`.
pub(super) struct Worker {
    pub run_id: u64,
    controls: Sender<AgentControl>,
    handle: JoinHandle<Result<TaskOutcome, CliError>>,
}

impl Worker {
    pub fn steer(&self, text: String) -> bool {
        self.controls.send(AgentControl::Steer(text)).is_ok()
    }

    pub fn interrupt(&self) {
        let _ = self.controls.send(AgentControl::Interrupt);
    }

    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    pub fn join(self) -> Result<TaskOutcome, CliError> {
        match self.handle.join() {
            Ok(result) => result,
            Err(_) => Err(CliError::Model("agent worker panicked".to_string())),
        }
    }
}

/// Starts the local worker for `request`. The model and config are built inside
/// the thread so a missing API key surfaces as a UI error, not a panic, and
/// never leaves a half-registered task behind.
pub(super) fn spawn_worker(
    workspace: PathBuf,
    run_id: u64,
    request: RunRequest,
    cancelled: Arc<AtomicBool>,
    events: Sender<UiEvent>,
) -> Result<Worker, CliError> {
    cancelled.store(false, Ordering::Relaxed);
    let (control_tx, control_rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name(format!("mh-tui-run-{run_id}"))
        .spawn(move || {
            let model =
                OpenAiResponses::from_env().map_err(|error| CliError::Model(error.to_string()))?;
            let agent = Agent::new(model, tui_agent_config(run_id, events.clone()));
            execute_run(
                &agent,
                &workspace,
                request,
                &cancelled,
                &control_rx,
                &events,
                run_id,
            )
        })
        .map_err(CliError::Io)?;
    Ok(Worker {
        run_id,
        controls: control_tx,
        handle,
    })
}

/// Agent config whose prelude confirmation is answered by the UI thread.
///
/// The reply channel is one-shot and the wait is bounded by liveness, not by
/// time: a closed channel, a cancelled run, or a vanished UI all mean refusal.
/// A timeout is never read as consent.
pub(super) fn tui_agent_config(run_id: u64, events: Sender<UiEvent>) -> AgentConfig {
    AgentConfig {
        confirm_prelude: Some(Arc::new(move |prelude: &Prelude| {
            let (reply_tx, reply_rx) = std::sync::mpsc::channel();
            let sent = events.send(UiEvent::Trust {
                run_id,
                prelude: prelude.clone(),
                reply: reply_tx,
            });
            if sent.is_err() {
                return TrustDecision::Rejected;
            }
            loop {
                match reply_rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(decision) => return decision,
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return TrustDecision::Rejected,
                }
            }
        })),
        ..AgentConfig::default()
    }
}

/// Registers (when new), reports admission, and drives one root task.
///
/// Generic over the model so the production thread and the tests exercise the
/// same registration, admission-baseline and callback wiring.
pub(super) fn execute_run<M: Model + 'static>(
    agent: &Agent<M>,
    workspace: &Path,
    request: RunRequest,
    cancelled: &Arc<AtomicBool>,
    controls: &Receiver<AgentControl>,
    events: &Sender<UiEvent>,
    run_id: u64,
) -> Result<TaskOutcome, CliError> {
    let task_id = match request {
        RunRequest::Resume(task_id) => task_id,
        RunRequest::New(task) => {
            let task_id = Agent::<M>::start_detached_task(workspace, &task)?;
            let _ = events.send(UiEvent::Registered { run_id, task_id });
            task_id
        }
        RunRequest::Followup {
            previous_task,
            message,
        } => {
            let task_id =
                Agent::<M>::start_detached_followup_task(workspace, previous_task, &message)?;
            let _ = events.send(UiEvent::Registered { run_id, task_id });
            task_id
        }
    };
    let mut admitted = || -> Result<(), AgentError> {
        // Baseline is read from the journal, never guessed: live turns are
        // paired with durable `ModelStarted` events after this point.
        let after_seq = Session::inspect(workspace)?.state().last_event_seq;
        let _ = events.send(UiEvent::Admitted {
            run_id,
            task_id,
            after_seq,
        });
        Ok(())
    };
    let mut emit = |event: AgentEvent| {
        let _ = events.send(UiEvent::Agent {
            run_id,
            task_id,
            event,
        });
    };
    agent
        .resume_task_admitted(
            workspace,
            Some(task_id),
            cancelled,
            Some(controls),
            &mut admitted,
            &mut emit,
        )
        .map_err(CliError::from)
}

/// A durable command against a possibly foreign task. At most one at a time.
pub(super) struct Command {
    handle: JoinHandle<()>,
}

impl Command {
    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    pub fn join(self) {
        let _ = self.handle.join();
    }
}

pub(super) fn spawn_steer(
    workspace: PathBuf,
    task_id: TaskId,
    text: String,
    events: Sender<UiEvent>,
) -> Result<Command, CliError> {
    spawn_command(move || {
        let result = mh::runtime::steer_task(&workspace, Some(task_id), &text)
            .map_err(|error| error.to_string());
        let _ = events.send(UiEvent::Receipt {
            target: task_id,
            cancel: false,
            result,
        });
    })
}

pub(super) fn spawn_cancel(
    workspace: PathBuf,
    task_id: TaskId,
    events: Sender<UiEvent>,
) -> Result<Command, CliError> {
    spawn_command(move || {
        let result =
            mh::runtime::cancel_task(&workspace, Some(task_id)).map_err(|error| error.to_string());
        let _ = events.send(UiEvent::Receipt {
            target: task_id,
            cancel: true,
            result,
        });
    })
}

fn spawn_command<F>(body: F) -> Result<Command, CliError>
where
    F: FnOnce() + Send + 'static,
{
    let handle = std::thread::Builder::new()
        .name("mh-tui-command".to_string())
        .spawn(body)
        .map_err(CliError::Io)?;
    Ok(Command { handle })
}
