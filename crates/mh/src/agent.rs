//! Durable agent runtime: one execution loop for root and worker agents.
//!
//! The v0.5 invariant: a task is durable, and model calls, context windows,
//! PTC executions, workers, and subprocesses are disposable mechanisms. The
//! loop therefore runs *windows*, not a task: `max_turns_per_window` bounds one
//! window, a window that fills up compacts into durable semantic state and a
//! fresh window opens, and only an accepted `finish()` completes the task.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

use serde_json::Value;

use crate::checkpoint::{CheckpointError, CheckpointStore};
use crate::context::{ContextCompiler, ContextError};
use crate::delegation::{
    AgentAccess, AgentSpawnOptions, AgentState, AgentStatus, DelegateResult, DelegationBudget,
    DelegationHost, IntegrationResponse, delegate_batch_sync, delegate_sync,
};
use crate::goal::{FinishRequest, FinishVerdict, GoalUpdate, TaskStatus};
use crate::identity::{AgentId, ExecutionId, IsolatedWorkspaceId, ProcessId, RevisionId, TaskId};
use crate::isolation::{IntegrationResult, IsolationStore};
use crate::model::{
    GenerationStop, GenerationStopReason, Model, ModelError, ModelEvent, ModelOutput, lower,
};
use crate::process::{ProcessSnapshot, ProcessSpec, ProcessStream, ProcessTail};
use crate::ptc::prelude::{self, Prelude, PreludeError};
use crate::ptc::runtime::{PtcEvent, PtcEventSink, PtcExecution};
use crate::ptc::trust::{Trust, TrustDecision, TrustError, TrustStore};
use crate::ptc::{PtcBudget, PtcOutcome, PtcRuntime};
use crate::runtime::{Runtime, bounded_context};
use crate::session::{Session, SessionError, SessionEvent, build_checkpoint, validate_finish};
use crate::tools::Capabilities;
use crate::workspace::{WorkspaceError, WorkspaceTracker, WorkspaceTrackerImpl};

/// Decides whether a never-before-seen workspace prelude may run.
///
/// The library never prompts: a front end supplies this, so a non-interactive
/// run can refuse by default instead of blocking on a terminal that is not
/// there.
pub type PreludeConfirm = Arc<dyn Fn(&Prelude) -> TrustDecision + Send + Sync>;

#[derive(Clone)]
pub struct AgentConfig {
    /// Model turns per context window. This is a *window* bound: crossing it
    /// rolls over into a fresh window rather than ending the task.
    pub max_turns_per_window: usize,
    /// Context windows a task may use before it is failed as non-converging.
    /// A task crossing many windows is normal; crossing unboundedly is not.
    pub max_windows: u32,
    pub context_tokens: usize,
    /// Compaction threshold as a fraction of `context_tokens`.
    pub context_soft_limit: f32,
    pub ptc_budget: PtcBudget,
    pub delegation_budget: DelegationBudget,
    /// Fallback prelude used when the workspace defines none.
    pub user_prelude: Option<PathBuf>,
    /// Where prelude trust decisions are recorded. Outside every workspace, so
    /// a repository cannot ship its own authorization.
    pub trust_store: Option<TrustStore>,
    /// Consulted once per unseen workspace prelude. Absent means refuse.
    pub confirm_prelude: Option<PreludeConfirm>,
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("max_turns_per_window", &self.max_turns_per_window)
            .field("max_windows", &self.max_windows)
            .field("context_tokens", &self.context_tokens)
            .field("context_soft_limit", &self.context_soft_limit)
            .field("ptc_budget", &self.ptc_budget)
            .field("delegation_budget", &self.delegation_budget)
            .field("user_prelude", &self.user_prelude)
            .field("trust_store", &self.trust_store)
            .field("confirm_prelude", &self.confirm_prelude.is_some())
            .finish()
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns_per_window: 32,
            max_windows: 64,
            context_tokens: 32_000,
            context_soft_limit: 0.75,
            ptc_budget: PtcBudget::default(),
            delegation_budget: DelegationBudget::default(),
            user_prelude: prelude::user_prelude_path(),
            trust_store: TrustStore::user().ok(),
            confirm_prelude: None,
        }
    }
}

impl AgentConfig {
    fn soft_limit_tokens(&self) -> usize {
        let fraction = self.context_soft_limit.clamp(0.1, 0.95);
        ((self.context_tokens as f32) * fraction) as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentControl {
    Steer(String),
    Interrupt,
}

/// Why a discovered prelude did not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreludeRejection {
    /// The operator declined it at the confirmation prompt.
    Declined,
    /// Declined in an earlier run; the recorded decision still stands.
    PreviouslyRejected,
    /// Nothing could ask: no confirmation callback was supplied.
    CannotConfirm,
    /// No user config directory, so no tamper-proof place to record trust.
    NoTrustStore,
}

impl PreludeRejection {
    pub const fn explain(self) -> &'static str {
        match self {
            Self::Declined => "declined at the confirmation prompt",
            Self::PreviouslyRejected => "previously declined for this exact content",
            Self::CannotConfirm => {
                "not confirmed: this run cannot prompt, so an unreviewed prelude is refused"
            }
            Self::NoTrustStore => {
                "no user config directory, so a trust decision cannot be recorded"
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    Model(ModelEvent),
    PreludeLoaded {
        path: PathBuf,
        described: bool,
    },
    PreludeRejected {
        path: PathBuf,
        reason: PreludeRejection,
    },
    /// A fresh context window opened for the task.
    ContextWindowStarted {
        window: u32,
    },
    /// The previous window was compacted into durable semantic state.
    Compacted {
        window: u32,
        tokens: usize,
    },
    /// The model emitted a progress message; the task is still active.
    AssistantMessage {
        content: String,
    },
    FinishProposed {
        accepted: bool,
        objections: Vec<String>,
    },
    TaskCompleted {
        summary: String,
    },
    GoalUpdated,
    AgentSpawned {
        agent: AgentId,
        objective: String,
    },
    AgentSettled {
        agent: AgentId,
        state: AgentState,
    },
    ProcessSpawned {
        process: ProcessId,
        argv: Vec<String>,
    },
    ProcessExited {
        process: ProcessId,
        exit_code: Option<i64>,
    },
    PtcStarted,
    PtcHostCallStarted {
        call_id: u64,
        name: String,
    },
    PtcHostCallCompleted {
        call_id: u64,
        name: String,
        ok: bool,
        duration_ms: u64,
    },
    EvidenceRecorded {
        kind: String,
        ok: bool,
    },
    SteeringQueued {
        content: String,
    },
    SteeringApplied {
        content: String,
    },
    RepeatedActionDetected {
        fingerprint: String,
    },
    ToolCompleted {
        outcome: String,
        tool_calls: usize,
        duration_ms: u64,
    },
}

#[derive(Debug)]
pub enum AgentError {
    Session(SessionError),
    Workspace(WorkspaceError),
    Checkpoint(CheckpointError),
    Model(ModelError),
    Context(ContextError),
    Prelude(PreludeError),
    Trust(TrustError),
    Cancelled,
    /// The task crossed its context-window ceiling without finishing.
    WindowLimit(u32),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Session(error) => error.fmt(f),
            Self::Workspace(error) => error.fmt(f),
            Self::Checkpoint(error) => error.fmt(f),
            Self::Model(error) => error.fmt(f),
            Self::Context(error) => error.fmt(f),
            Self::Prelude(error) => error.fmt(f),
            Self::Trust(error) => error.fmt(f),
            Self::Cancelled => write!(f, "agent cancelled"),
            Self::WindowLimit(limit) => {
                write!(f, "task did not converge within {limit} context windows")
            }
        }
    }
}

impl std::error::Error for AgentError {}

impl From<SessionError> for AgentError {
    fn from(value: SessionError) -> Self {
        Self::Session(value)
    }
}
impl From<WorkspaceError> for AgentError {
    fn from(value: WorkspaceError) -> Self {
        Self::Workspace(value)
    }
}
impl From<CheckpointError> for AgentError {
    fn from(value: CheckpointError) -> Self {
        Self::Checkpoint(value)
    }
}
impl From<ModelError> for AgentError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}
impl From<ContextError> for AgentError {
    fn from(value: ContextError) -> Self {
        Self::Context(value)
    }
}
impl From<PreludeError> for AgentError {
    fn from(value: PreludeError) -> Self {
        Self::Prelude(value)
    }
}
impl From<TrustError> for AgentError {
    fn from(value: TrustError) -> Self {
        Self::Trust(value)
    }
}

/// How a run of the loop ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskOutcome {
    /// `finish()` was accepted; the task is complete.
    Completed(String),
    /// The model reported progress and is waiting on the user. The task stays
    /// active and durable; a later message or `mh resume` continues it.
    AwaitingUser(String),
}

impl TaskOutcome {
    pub fn message(&self) -> &str {
        match self {
            Self::Completed(text) | Self::AwaitingUser(text) => text,
        }
    }

    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Completed(_))
    }
}

pub struct Agent<M> {
    model: Arc<M>,
    compiler: ContextCompiler,
    config: AgentConfig,
}

impl<M: Model + 'static> Agent<M> {
    pub fn new(model: M, config: AgentConfig) -> Self {
        let compiler = ContextCompiler {
            max_tokens: config.context_tokens,
            soft_limit_tokens: config.soft_limit_tokens(),
            ..ContextCompiler::default()
        };
        Self {
            model: Arc::new(model),
            compiler,
            config,
        }
    }

    /// Clones the compiler with the active prelude's tool documentation, so a
    /// prelude the runtime will evaluate is also one the model knows about.
    fn compiler_with_prelude(&self, prelude: Option<&Prelude>) -> ContextCompiler {
        ContextCompiler {
            prelude_tools: prelude.and_then(Prelude::description),
            ..self.compiler.clone()
        }
    }

    /// Applies the trust decision for a discovered prelude.
    ///
    /// A workspace prelude arrives with the repository and runs arbitrary code
    /// with the caller's capabilities, so it runs only once confirmed. The
    /// decision is keyed by content and recorded outside the workspace, so
    /// editing the prelude asks again and the repository cannot authorize
    /// itself. Refusal drops the prelude rather than failing the run: the
    /// agent still works, just without the derived tools.
    fn authorize_prelude(
        &self,
        discovered: Option<Prelude>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<Option<Prelude>, AgentError> {
        let Some(prelude) = discovered else {
            return Ok(None);
        };
        if !prelude.origin.requires_confirmation() {
            return Ok(Some(prelude));
        }
        let Some(store) = self.config.trust_store.as_ref() else {
            // No tamper-proof place to record a decision means no way to honor
            // one; running the prelude anyway would make the prompt theatre.
            events(AgentEvent::PreludeRejected {
                path: prelude.path.clone(),
                reason: PreludeRejection::NoTrustStore,
            });
            return Ok(None);
        };
        match store.status(&prelude)? {
            Trust::Trusted => Ok(Some(prelude)),
            Trust::Rejected => {
                events(AgentEvent::PreludeRejected {
                    path: prelude.path.clone(),
                    reason: PreludeRejection::PreviouslyRejected,
                });
                Ok(None)
            }
            Trust::Unknown => {
                let Some(confirm) = self.config.confirm_prelude.as_ref() else {
                    events(AgentEvent::PreludeRejected {
                        path: prelude.path.clone(),
                        reason: PreludeRejection::CannotConfirm,
                    });
                    return Ok(None);
                };
                let decision = confirm(&prelude);
                store.record(&prelude, decision)?;
                match decision {
                    TrustDecision::Trusted => Ok(Some(prelude)),
                    TrustDecision::Rejected => {
                        events(AgentEvent::PreludeRejected {
                            path: prelude.path.clone(),
                            reason: PreludeRejection::Declined,
                        });
                        Ok(None)
                    }
                }
            }
        }
    }

    pub fn run_task(
        &self,
        workspace: &Path,
        task: &str,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<String, AgentError> {
        self.run_task_with_events(workspace, task, cancelled, &mut |_| {})
    }

    pub fn run_task_with_events(
        &self,
        workspace: &Path,
        task: &str,
        cancelled: &Arc<AtomicBool>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<String, AgentError> {
        self.run_task_controlled(workspace, task, cancelled, None, events)
            .map(|outcome| outcome.message().to_string())
    }

    pub fn run_task_controlled(
        &self,
        workspace: &Path,
        task: &str,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<TaskOutcome, AgentError> {
        let mut session = Session::open(workspace)?;
        let task_id = session.begin_task(task)?;
        session.append(SessionEvent::UserMessage {
            task_id,
            content: task.to_string(),
        })?;
        self.drive(session, task_id, cancelled, controls, events)
    }

    /// Starts a task without running it, for a detached launch.
    pub fn start_detached_task(workspace: &Path, task: &str) -> Result<TaskId, AgentError> {
        let mut session = Session::open(workspace)?;
        let task_id = session.begin_task(task)?;
        session.append(SessionEvent::UserMessage {
            task_id,
            content: task.to_string(),
        })?;
        session.set_status(task_id, TaskStatus::Queued, None)?;
        Ok(task_id)
    }

    pub fn resume(
        &self,
        workspace: &Path,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<String, AgentError> {
        self.resume_with_events(workspace, cancelled, &mut |_| {})
    }

    pub fn resume_with_events(
        &self,
        workspace: &Path,
        cancelled: &Arc<AtomicBool>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<String, AgentError> {
        self.resume_controlled(workspace, cancelled, None, events)
            .map(|outcome| outcome.message().to_string())
    }

    /// Continues an existing durable task from its persisted semantic state.
    pub fn resume_controlled(
        &self,
        workspace: &Path,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<TaskOutcome, AgentError> {
        self.resume_task(workspace, None, cancelled, controls, events)
    }

    /// Continues a specific durable task, which is what a detached runner and
    /// `mh attach` both need.
    pub fn resume_task(
        &self,
        workspace: &Path,
        task_id: Option<TaskId>,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<TaskOutcome, AgentError> {
        let session = Session::resume(workspace)?;
        let task_id = match task_id.or_else(|| session.root_task()) {
            Some(task_id) => task_id,
            None => {
                return Err(AgentError::Session(SessionError::State(
                    "no task to resume".to_string(),
                )));
            }
        };
        if !session.task_view(task_id).status().is_active() {
            return Err(AgentError::Session(SessionError::State(format!(
                "task {} already {}",
                task_id.0,
                session.task_view(task_id).status().label()
            ))));
        }
        self.drive(session, task_id, cancelled, controls, events)
    }

    pub fn send_message(
        &self,
        workspace: &Path,
        message: &str,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<String, AgentError> {
        self.send_message_with_events(workspace, message, cancelled, &mut |_| {})
    }

    pub fn send_message_with_events(
        &self,
        workspace: &Path,
        message: &str,
        cancelled: &Arc<AtomicBool>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<String, AgentError> {
        self.send_message_controlled(workspace, message, cancelled, None, events)
            .map(|outcome| outcome.message().to_string())
    }

    /// Delivers a user message. An active parked task continues; otherwise a
    /// new task starts.
    pub fn send_message_controlled(
        &self,
        workspace: &Path,
        message: &str,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<TaskOutcome, AgentError> {
        let mut session = Session::open(workspace)?;
        let parked = session
            .root_task()
            .filter(|task_id| session.task_view(*task_id).status().is_active());
        let task_id = match parked {
            Some(task_id) => task_id,
            None => session.begin_task(message)?,
        };
        session.append(SessionEvent::UserMessage {
            task_id,
            content: message.to_string(),
        })?;
        self.drive(session, task_id, cancelled, controls, events)
    }

    /// Sets up durable services and runs the task's context windows.
    fn drive(
        &self,
        session: Session,
        task_id: TaskId,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<TaskOutcome, AgentError> {
        let workspace = session.state().workspace.clone();
        // Discovered once per session run: a prelude edited mid-run would
        // otherwise change the tool set the model was told about.
        let discovered = prelude::discover(&workspace, self.config.user_prelude.as_deref())?;
        let prelude = self.authorize_prelude(discovered, events)?.map(Arc::new);
        if let Some(prelude) = prelude.as_deref() {
            // Durable provenance: the journal records which tool environment
            // every later host call and evidence record was produced under.
            session.append_shared_event(SessionEvent::PreludeLoaded {
                path: prelude.path.clone(),
                prelude: prelude.identity(),
                described: prelude.description().is_some(),
            })?;
            events(AgentEvent::PreludeLoaded {
                path: prelude.path.clone(),
                described: prelude.description().is_some(),
            });
        }
        let runtime = Arc::new(Runtime::new(
            session,
            self.config.delegation_budget.clone(),
        )?);
        runtime.set_prelude(prelude.clone());
        // A previous runtime's processes cannot have survived it; say so
        // instead of pretending they are live.
        for snapshot in runtime.recover_processes() {
            events(AgentEvent::ProcessExited {
                process: snapshot.id,
                exit_code: snapshot.exit_code,
            });
        }

        let host = Arc::new(AgentRuntimeHost::new(
            self.model.clone(),
            self.compiler_with_prelude(prelude.as_deref()),
            self.config.clone(),
            runtime.clone(),
            cancelled.clone(),
        ));
        let outcome = self.run_windows(
            &runtime,
            Some(host.clone() as Arc<dyn DelegationHost>),
            task_id,
            AgentId::ROOT,
            cancelled,
            controls,
            events,
        );

        // A task that stops owning execution must not leave orphans behind.
        runtime.workers().cancel_all();
        runtime.workers().join_all();
        runtime.shutdown_task_processes(task_id);
        if matches!(outcome, Ok(TaskOutcome::Completed(_))) {
            runtime.cleanup_isolation(task_id);
        }
        outcome
    }

    /// Runs context windows until the task completes, parks, or is cancelled.
    #[allow(clippy::too_many_arguments)]
    fn run_windows(
        &self,
        runtime: &Arc<Runtime>,
        host: Option<Arc<dyn DelegationHost>>,
        task_id: TaskId,
        agent: AgentId,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<TaskOutcome, AgentError> {
        let session = runtime.session();
        let compiler = ContextCompiler {
            max_tokens: self.config.context_tokens,
            soft_limit_tokens: self.config.soft_limit_tokens(),
            ..self.compiler_with_prelude(runtime.prelude().as_deref())
        };
        let workspace = if agent.is_root() {
            runtime.workspace_root()
        } else {
            worker_workspace(runtime, agent)?
        };
        let tracker: Arc<dyn WorkspaceTracker> = if agent.is_root() {
            runtime.tracker()
        } else {
            Arc::new(WorkspaceTrackerImpl::open(&workspace)?)
        };
        let capabilities = match worker_access(runtime, agent) {
            Some(AgentAccess::Read) => Capabilities::read_only(&workspace),
            _ => Capabilities::new(&workspace),
        };
        let prelude = runtime.prelude();
        let checkpoints = CheckpointStore::open(&workspace, tracker.clone())?
            .with_prelude(prelude.as_deref().map(Prelude::identity));
        let isolated = worker_workspace_id(runtime, agent);
        let mut ptc = PtcRuntime::new(
            capabilities,
            session.result_store()?,
            self.config.ptc_budget.clone(),
            tracker.clone(),
            checkpoints,
        )
        .with_prelude(prelude);
        if let Some(host) = host {
            ptc = ptc.with_delegation(host);
        }
        let sink: Arc<dyn PtcEventSink> = Arc::new(session.child_ptc_event_sink(isolated));

        session.set_status(task_id, TaskStatus::Running, None)?;
        let mut repeated = RepeatState::default();
        let start_window = session.task_view(task_id).window;
        session.append_shared_event(SessionEvent::ContextWindowStarted {
            task_id,
            agent,
            window: start_window,
        })?;
        events(AgentEvent::ContextWindowStarted {
            window: start_window,
        });

        loop {
            let view = session.task_view(task_id);
            if view.window >= self.config.max_windows {
                session.set_status(
                    task_id,
                    TaskStatus::Failed,
                    Some("context window limit reached".to_string()),
                )?;
                return Err(AgentError::WindowLimit(self.config.max_windows));
            }
            match self.run_window(
                runtime,
                &ptc,
                &sink,
                &compiler,
                task_id,
                agent,
                cancelled,
                controls,
                &mut repeated,
                events,
            )? {
                WindowOutcome::Finished(summary) => return Ok(TaskOutcome::Completed(summary)),
                WindowOutcome::AwaitingUser(text) => return Ok(TaskOutcome::AwaitingUser(text)),
                WindowOutcome::Rollover => {
                    let view = session.task_view(task_id);
                    let checkpoint = build_checkpoint(&view);
                    let rendered = checkpoint.render();
                    session.append_shared_event(SessionEvent::Compacted {
                        task_id,
                        agent,
                        window: view.window,
                        summary: rendered.clone(),
                        checkpoint: Some(checkpoint),
                    })?;
                    events(AgentEvent::Compacted {
                        window: view.window,
                        tokens: crate::context::estimate_tokens(&rendered),
                    });
                    let window = session.task_view(task_id).window;
                    session.append_shared_event(SessionEvent::ContextWindowStarted {
                        task_id,
                        agent,
                        window,
                    })?;
                    events(AgentEvent::ContextWindowStarted { window });
                    repeated = RepeatState::default();
                }
            }
        }
    }

    /// Runs model turns inside one context window.
    #[allow(clippy::too_many_arguments)]
    fn run_window(
        &self,
        runtime: &Arc<Runtime>,
        ptc: &PtcRuntime,
        sink: &Arc<dyn PtcEventSink>,
        compiler: &ContextCompiler,
        task_id: TaskId,
        agent: AgentId,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        repeated: &mut RepeatState,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<WindowOutcome, AgentError> {
        let session = runtime.session();
        for _turn in 0..self.config.max_turns_per_window.max(1) {
            // A detached run has no control channel: durable steering and
            // cancellation arrive through the journal instead.
            session.refresh()?;
            let steering = drain_controls(cancelled, controls, events)?;
            if !steering.is_empty() {
                apply_steering(session, task_id, steering, events, repeated)?;
            }
            apply_durable_steering(session, task_id, events, repeated)?;
            if session.task_view(task_id).cancel_requested {
                cancelled.store(true, Ordering::Relaxed);
            }
            check_cancelled(session, task_id, cancelled)?;
            if agent.is_root() {
                session.reconcile_workspace()?;
            }

            let context = compiler.compile(session, task_id)?;
            if compiler.needs_rollover(&context) {
                return Ok(WindowOutcome::Rollover);
            }
            session.append_shared_event(SessionEvent::ModelStarted { task_id })?;
            let (output, steering) =
                self.generate_controlled(&context, cancelled, controls, events)?;
            if !steering.is_empty() {
                session.append_shared_event(SessionEvent::ModelSuperseded { task_id })?;
                apply_steering(session, task_id, steering, events, repeated)?;
                continue;
            }
            session.append_shared_event(SessionEvent::ModelCompleted {
                task_id,
                output_kind: output_kind(&output).to_string(),
            })?;
            check_cancelled(session, task_id, cancelled)?;

            match output {
                ModelOutput::Text(text) => {
                    // Text is a progress report, never completion. The task
                    // stays durable and active.
                    session.append_shared_event(SessionEvent::AssistantMessage {
                        task_id,
                        content: text.clone(),
                    })?;
                    events(AgentEvent::AssistantMessage {
                        content: text.clone(),
                    });
                    if !agent.is_root() {
                        // A worker must settle so its parent's join resolves:
                        // parking would strand the parent forever.
                        return Ok(WindowOutcome::AwaitingUser(text));
                    }
                    session.set_status(task_id, TaskStatus::WaitingUser, None)?;
                    return Ok(WindowOutcome::AwaitingUser(text));
                }
                executable => {
                    let source = lower(&executable).expect("executable model output lowers to PTC");
                    let normalized = normalize_source(&source);
                    if repeated.identical_executions >= 2
                        && repeated.normalized_source.as_deref() == Some(&normalized)
                    {
                        let fingerprint = repeated
                            .fingerprint
                            .clone()
                            .unwrap_or_else(|| fingerprint_text(&normalized));
                        record_repeat(session, task_id, events, fingerprint)?;
                        continue;
                    }
                    let steering = drain_controls(cancelled, controls, events)?;
                    if !steering.is_empty() {
                        apply_steering(session, task_id, steering, events, repeated)?;
                        continue;
                    }
                    check_cancelled(session, task_id, cancelled)?;
                    let execution_id = session.begin_execution();
                    let start_revision = runtime
                        .tracker()
                        .current_revision()
                        .map(|revision| revision.id)
                        .unwrap_or_else(|_| session.task_view(task_id).goal.current_revision);
                    let start_revision = if agent.is_root() {
                        start_revision
                    } else {
                        session.task_view(task_id).goal.current_revision
                    };
                    session.append_shared_event(SessionEvent::PtcStarted {
                        task_id,
                        execution_id,
                        source: source.clone(),
                        start_revision: start_revision.clone(),
                    })?;
                    events(AgentEvent::PtcStarted);
                    let execution = PtcExecution {
                        task_id,
                        agent,
                        execution_id,
                        start_revision,
                    };
                    let result =
                        ptc.execute(&source, cancelled.clone(), execution, Some(sink.clone()));
                    translate_ptc_events(&result.events, events);
                    events(AgentEvent::ToolCompleted {
                        outcome: format!("{:?}", result.outcome),
                        tool_calls: result.tool_calls,
                        duration_ms: result.duration_ms,
                    });
                    session.append_ptc_result(&result)?;
                    if result.outcome == PtcOutcome::Interrupted {
                        session.interrupt_task(task_id)?;
                        return Err(AgentError::Cancelled);
                    }
                    // An accepted finish() inside the program completes the
                    // task; nothing else does.
                    if let Some(summary) = accepted_finish(session, task_id) {
                        events(AgentEvent::TaskCompleted {
                            summary: summary.clone(),
                        });
                        return Ok(WindowOutcome::Finished(summary));
                    }
                    let fingerprint =
                        action_fingerprint(&normalized, &result.outcome, &result.value);
                    if repeated.normalized_source.as_deref() == Some(&normalized)
                        && repeated.fingerprint.as_deref() == Some(&fingerprint)
                    {
                        repeated.identical_executions += 1;
                    } else {
                        repeated.normalized_source = Some(normalized);
                        repeated.fingerprint = Some(fingerprint.clone());
                        repeated.identical_executions = 1;
                    }
                    if repeated.identical_executions == 2 {
                        record_repeat(session, task_id, events, fingerprint)?;
                    }
                }
            }
        }
        // The window filled without finishing: that is a rollover, not the end
        // of the task.
        Ok(WindowOutcome::Rollover)
    }

    fn generate_controlled(
        &self,
        context: &crate::context::CompiledContext,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<(ModelOutput, Vec<String>), AgentError> {
        let stop = GenerationStop::new(cancelled.clone());
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let mut steering = Vec::new();
        std::thread::scope(|scope| {
            let model_stop = stop.clone();
            scope.spawn(move || {
                let result = self.model.generate(context, &model_stop, &mut |event| {
                    let _ = event_tx.send(event);
                });
                let _ = result_tx.send(result);
            });
            loop {
                while let Ok(event) = event_rx.try_recv() {
                    events(AgentEvent::Model(event));
                }
                if let Some(receiver) = controls {
                    loop {
                        match receiver.try_recv() {
                            Ok(AgentControl::Steer(content)) => {
                                events(AgentEvent::SteeringQueued {
                                    content: content.clone(),
                                });
                                steering.push(content);
                                stop.stop(GenerationStopReason::Superseded);
                            }
                            Ok(AgentControl::Interrupt) => {
                                stop.stop(GenerationStopReason::UserInterrupt)
                            }
                            Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                        }
                    }
                }
                if cancelled.load(Ordering::Relaxed) {
                    stop.stop(GenerationStopReason::UserInterrupt);
                }
                match result_rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(result) => {
                        while let Ok(event) = event_rx.try_recv() {
                            events(AgentEvent::Model(event));
                        }
                        return match result {
                            Ok(output) => Ok((output, steering)),
                            Err(ModelError::Stopped(GenerationStopReason::Superseded))
                                if !steering.is_empty() =>
                            {
                                Ok((ModelOutput::Text(String::new()), steering))
                            }
                            Err(ModelError::Stopped(GenerationStopReason::UserInterrupt)) => {
                                Err(AgentError::Cancelled)
                            }
                            Err(error) => Err(error.into()),
                        };
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        return Err(AgentError::Model(ModelError::Protocol(
                            "model worker stopped unexpectedly".to_string(),
                        )));
                    }
                }
            }
        })
    }
}

/// How one context window ended.
enum WindowOutcome {
    Finished(String),
    AwaitingUser(String),
    Rollover,
}

/// Newest accepted finish summary for a task, if it has one.
fn accepted_finish(session: &Session, task_id: TaskId) -> Option<String> {
    session
        .events()
        .iter()
        .rev()
        .find_map(|record| match &record.event {
            SessionEvent::TaskCompleted {
                task_id: id,
                summary,
                ..
            } if *id == task_id => Some(summary.clone()),
            _ => None,
        })
}

fn worker_access(runtime: &Runtime, agent: AgentId) -> Option<AgentAccess> {
    (!agent.is_root())
        .then(|| runtime.session().agent(agent))
        .flatten()
        .map(|record| record.access)
}

fn worker_workspace_id(runtime: &Runtime, agent: AgentId) -> Option<IsolatedWorkspaceId> {
    (!agent.is_root())
        .then(|| runtime.session().agent(agent))
        .flatten()
        .and_then(|record| record.isolated_workspace)
}

/// Resolves the workspace a worker executes in: its isolated worktree when it
/// has one, otherwise the parent workspace.
fn worker_workspace(runtime: &Runtime, agent: AgentId) -> Result<PathBuf, AgentError> {
    let Some(workspace) = worker_workspace_id(runtime, agent) else {
        return Ok(runtime.workspace_root());
    };
    let isolation = runtime.isolation().ok_or_else(|| {
        AgentError::Session(SessionError::State(
            "isolated-write requires a Git-root workspace".to_string(),
        ))
    })?;
    isolation
        .load(workspace)
        .map(|record| record.path)
        .map_err(|error| AgentError::Session(SessionError::State(error.to_string())))
}

/// The runtime's PTC host: workers, processes, goal state, and completion.
struct AgentRuntimeHost<M> {
    model: Arc<M>,
    compiler: ContextCompiler,
    config: AgentConfig,
    runtime: Arc<Runtime>,
    cancelled: Arc<AtomicBool>,
}

impl<M: Model + 'static> AgentRuntimeHost<M> {
    fn new(
        model: Arc<M>,
        compiler: ContextCompiler,
        config: AgentConfig,
        runtime: Arc<Runtime>,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            model,
            compiler,
            config,
            runtime,
            cancelled,
        }
    }

    fn session(&self) -> &Session {
        self.runtime.session()
    }

    fn status(&self, agent: AgentId) -> Result<AgentStatus, String> {
        // Prefer the live registry; fall back to the journal so a worker from a
        // previous runtime still reports honestly after a restart.
        if let Some(snapshot) = self.runtime.workers().snapshot(agent) {
            return Ok(snapshot.status(self.session()));
        }
        let record = self
            .session()
            .agent(agent)
            .ok_or_else(|| format!("unknown agent {}", agent.0))?;
        let view = self.session().task_view(record.task_id);
        Ok(AgentStatus {
            agent,
            task_id: record.task_id,
            state: record.state,
            objective: record.objective,
            access: record.access,
            profile: record.profile,
            done: record.state.is_terminal(),
            window: view.window,
            turns_in_window: view.turns_in_window,
            result: record.result,
            pending: view
                .goal
                .pending_work
                .iter()
                .map(|item| item.title.clone())
                .collect(),
            next_actions: view.goal.next_actions.clone(),
            current_revision: Some(view.goal.current_revision),
        })
    }

    /// Materializes a worker's private workspace at exactly the parent
    /// revision, or fails the spawn.
    fn prepare_workspace(
        &self,
        access: AgentAccess,
        agent: AgentId,
        task_id: TaskId,
        revision: &RevisionId,
    ) -> Result<Option<IsolatedWorkspaceId>, String> {
        if access == AgentAccess::Read {
            return Ok(None);
        }
        let isolation: &IsolationStore = self
            .runtime
            .isolation()
            .ok_or("isolated-write requires a Git-root workspace")?;
        let workspace = self.runtime.allocate_workspace();
        isolation
            .create(workspace, task_id, ExecutionId(agent.0), revision.clone())
            .map_err(|error| error.to_string())?;
        Ok(Some(workspace))
    }

    /// Runs a worker to completion on its own thread.
    ///
    /// A worker is a normal agent execution with a parent: it drives the same
    /// window loop, so it gets rollover, goal state, evidence, and resume for
    /// free. It gets no delegation host, which is what keeps depth at one.
    fn run_worker(
        &self,
        agent: AgentId,
        task_id: TaskId,
        base_revision: RevisionId,
        worker_cancelled: Arc<AtomicBool>,
    ) -> DelegateResult {
        let session = self.session();
        let workspace_id = worker_workspace_id(&self.runtime, agent);
        let _ = session.append_shared_event(SessionEvent::AgentStateChanged {
            agent,
            task_id,
            state: AgentState::Running,
        });
        self.runtime.workers().set_state(agent, AgentState::Running);

        let profile = session
            .agent(agent)
            .and_then(|record| record.profile)
            .and_then(|name| crate::delegation::profile(&name));
        let worker_agent = Agent {
            model: self.model.clone(),
            compiler: self.compiler.clone(),
            config: AgentConfig {
                max_turns_per_window: profile
                    .map(|profile| profile.max_turns_per_window)
                    .unwrap_or(self.config.delegation_budget.max_child_turns),
                max_windows: self.config.delegation_budget.max_child_windows,
                ..self.config.clone()
            },
        };
        // A worker gets a host of its own so it gains goal state, evidence,
        // and process ownership like any agent execution. Its own agent id is
        // what refuses agent_spawn/delegate/integrate, which is what actually
        // keeps delegation depth at one.
        let worker_host = Arc::new(self.clone_for_thread()) as Arc<dyn DelegationHost>;
        let outcome = worker_agent.run_windows(
            &self.runtime,
            Some(worker_host),
            task_id,
            agent,
            &worker_cancelled,
            None,
            &mut |_| {},
        );

        // Whatever happened, the worker must not leave processes running: its
        // parent can no longer see them.
        self.runtime.processes().kill_agent(agent);
        let view = session.task_view(task_id);
        let final_record = workspace_id.and_then(|id| {
            self.runtime
                .isolation()
                .and_then(|isolation| isolation.finalize(id).ok())
        });
        let final_revision = final_record
            .as_ref()
            .map(|record| record.final_revision.clone())
            .unwrap_or_else(|| view.goal.current_revision.clone());
        let changed = final_revision != base_revision;
        let (ok, summary) = match &outcome {
            Ok(outcome) => (true, outcome.message().to_string()),
            Err(AgentError::Cancelled) => (false, "cancelled".to_string()),
            Err(error) => (false, error.to_string()),
        };
        let findings = view
            .latest_ptc
            .map(|summary| summary.value)
            .unwrap_or(Value::Null);
        let result = DelegateResult {
            task_id,
            agent,
            ok,
            summary,
            base_revision,
            final_revision,
            changed,
            workspace: final_record.as_ref().map(|record| record.id),
            evidence: session.evidence_for_task(task_id),
            findings: bounded_context(findings, self.config.delegation_budget.max_findings_bytes),
        };
        let _ = session.append_shared_event(SessionEvent::AgentCompleted {
            agent,
            task_id,
            result: result.clone(),
        });
        // A worker's own task is durable, so it must reach a terminal status
        // too: an inspector reading a settled worker's task would otherwise
        // see it as still running forever.
        //
        // An accepted finish() already completed it; only settle a task the
        // worker left active.
        if session.task_view(task_id).status().is_active() {
            let status = if ok {
                TaskStatus::Completed
            } else if worker_cancelled.load(Ordering::Relaxed) {
                TaskStatus::Cancelled
            } else {
                TaskStatus::Failed
            };
            let _ = session.set_status(task_id, status, Some(result.summary.clone()));
        }
        self.runtime.workers().settle(agent, result.clone());
        self.runtime.release_worker_slot();
        result
    }
}

impl<M: Model + 'static> DelegationHost for AgentRuntimeHost<M> {
    fn agent_spawn(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        options: AgentSpawnOptions,
    ) -> Result<AgentStatus, String> {
        if !parent.agent.is_root() {
            // Unprefixed: the dispatch layer names whichever primitive the
            // program actually called, so `delegate()` does not report that
            // `agent_spawn` failed.
            return Err("unavailable in a delegated worker".to_string());
        }
        let (access, profile) = options.resolve()?;
        self.runtime.reserve_worker_slot()?;
        let session = self.session();
        let (agent, task_id) = session.allocate_agent();

        let workspace = match self.prepare_workspace(access, agent, task_id, revision) {
            Ok(workspace) => workspace,
            Err(error) => {
                self.runtime.release_worker_slot();
                return Err(error);
            }
        };
        session
            .append_shared_event(SessionEvent::AgentSpawned {
                parent_task_id: parent.task_id,
                parent_agent: parent.agent,
                agent,
                task_id,
                objective: options.task.clone(),
                access,
                profile: profile.map(|profile| profile.name.to_string()),
                base_revision: revision.clone(),
                context: bounded_context(options.context.clone(), 8 * 1024),
                isolated_workspace: workspace,
            })
            .map_err(|error| error.to_string())?;

        let worker_cancelled = self.runtime.workers().register(
            agent,
            task_id,
            options.task.clone(),
            access,
            profile.map(|profile| profile.name.to_string()),
        );
        let status = self.status(agent)?;

        // The worker runs on its own thread: agent_spawn returns immediately
        // and the root agent keeps doing model and PTC work.
        let host = self.clone_for_thread();
        let base = revision.clone();
        let handle = std::thread::Builder::new()
            .name(format!("mh-agent-{}", agent.0))
            .spawn(move || {
                host.run_worker(agent, task_id, base, worker_cancelled);
            })
            .map_err(|error| format!("agent_spawn: {error}"))?;
        self.runtime.workers().attach_thread(agent, handle);
        Ok(status)
    }

    fn agent_poll(&self, agent: AgentId) -> Result<AgentStatus, String> {
        self.status(agent)
    }

    fn agent_join(&self, agent: AgentId) -> Result<DelegateResult, String> {
        self.runtime.workers().join(agent)
    }

    fn agent_cancel(&self, agent: AgentId) -> Result<AgentStatus, String> {
        self.runtime.workers().cancel(agent)?;
        let _ = self
            .session()
            .append_shared_event(SessionEvent::AgentStateChanged {
                agent,
                task_id: self
                    .session()
                    .agent(agent)
                    .map(|record| record.task_id)
                    .unwrap_or(TaskId(0)),
                state: AgentState::Cancelled,
            });
        self.status(agent)
    }

    fn agent_send(&self, agent: AgentId, message: String) -> Result<AgentStatus, String> {
        let record = self
            .session()
            .agent(agent)
            .ok_or_else(|| format!("unknown agent {}", agent.0))?;
        if record.state.is_terminal() {
            return Err(format!(
                "agent {} already {}",
                agent.0,
                record.state.label()
            ));
        }
        // Durable delivery: the worker applies it at its next safe point, so a
        // message is never lost to a race with its model call.
        self.session()
            .append_shared_event(SessionEvent::AgentMessage {
                agent,
                task_id: record.task_id,
                content: message,
            })
            .map_err(|error| error.to_string())?;
        self.status(agent)
    }

    fn agent_list(&self) -> Vec<AgentStatus> {
        self.runtime
            .workers()
            .list()
            .into_iter()
            .map(|snapshot| snapshot.status(self.session()))
            .collect()
    }

    fn integrate(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        workspace: IsolatedWorkspaceId,
    ) -> IntegrationResponse {
        if !parent.agent.is_root() {
            return IntegrationResponse::error("integrate: unavailable in a delegated worker");
        }
        let Some(isolation) = self.runtime.isolation() else {
            return IntegrationResponse::error("integration requires a Git-root workspace");
        };
        // Integrating a worker that is still running would merge a half-written
        // tree; refuse instead.
        if let Some(record) = self
            .session()
            .agents()
            .into_iter()
            .find(|record| record.isolated_workspace == Some(workspace))
            && !record.is_terminal()
        {
            return IntegrationResponse::error(format!(
                "agent {} still owns workspace {}; join or cancel it first",
                record.agent.0, workspace.0
            ));
        }
        let record = match isolation.load(workspace) {
            Ok(record) => record,
            Err(error) => return IntegrationResponse::error(error.to_string()),
        };
        let owner = self
            .session()
            .agents()
            .into_iter()
            .find(|agent| agent.isolated_workspace == Some(workspace))
            .map(|agent| agent.agent)
            .unwrap_or(AgentId::ROOT);
        let session = self.session();
        let _ = session.append_shared_event(SessionEvent::IntegrationStarted {
            task_id: parent.task_id,
            execution_id: parent.execution_id,
            workspace,
            agent: owner,
            child_base: record.base_revision.clone(),
            parent_base: revision.clone(),
        });
        match isolation.integrate(workspace) {
            Ok(IntegrationResult::Applied {
                previous_revision,
                current_revision,
                changed_paths,
            }) => {
                let paths = changed_paths
                    .iter()
                    .map(|path| path.to_string_lossy().into_owned())
                    .collect::<Vec<_>>();
                let _ = session.append_shared_event(SessionEvent::WorkspaceRevisionChanged {
                    task_id: parent.task_id,
                    execution_id: Some(parent.execution_id),
                    from: previous_revision.clone(),
                    to: current_revision.clone(),
                    added: paths.clone(),
                    modified: Vec::new(),
                    deleted: Vec::new(),
                    source: crate::workspace::RevisionSource::Tool,
                    isolated_workspace: None,
                });
                let _ = session.append_shared_event(SessionEvent::IntegrationCompleted {
                    task_id: parent.task_id,
                    execution_id: parent.execution_id,
                    agent: owner,
                    parent_revision: current_revision.clone(),
                    changed_paths: paths,
                });
                IntegrationResponse {
                    ok: true,
                    conflict: false,
                    previous_revision: Some(previous_revision),
                    parent_revision: Some(current_revision),
                    child_base: None,
                    parent_current: None,
                    paths: changed_paths,
                    error: None,
                    requires_reverification: true,
                }
            }
            Ok(IntegrationResult::Conflict {
                child_base,
                parent_current,
                paths,
            }) => {
                let _ = session.append_shared_event(SessionEvent::IntegrationConflict {
                    task_id: parent.task_id,
                    execution_id: parent.execution_id,
                    agent: owner,
                    child_base: child_base.clone(),
                    parent_current: parent_current.clone(),
                    paths: paths
                        .iter()
                        .map(|path| path.to_string_lossy().into_owned())
                        .collect(),
                });
                IntegrationResponse {
                    ok: false,
                    conflict: true,
                    previous_revision: None,
                    parent_revision: None,
                    child_base: Some(child_base),
                    parent_current: Some(parent_current),
                    paths,
                    error: None,
                    requires_reverification: false,
                }
            }
            Err(error) => IntegrationResponse::error(error.to_string()),
        }
    }

    fn discard(
        &self,
        parent: &PtcExecution,
        workspace: IsolatedWorkspaceId,
        reason: String,
    ) -> Result<Value, String> {
        if !parent.agent.is_root() {
            return Err("discard: unavailable in a delegated worker".to_string());
        }
        let session = self.session();
        let owner = session
            .agents()
            .into_iter()
            .find(|record| record.isolated_workspace == Some(workspace));
        // Discarding a live worker's workspace would pull the tree out from
        // under it mid-write.
        if let Some(record) = owner.as_ref()
            && !record.is_terminal()
        {
            return Err(format!(
                "agent {} still owns workspace {}; join or cancel it first",
                record.agent.0, workspace.0
            ));
        }
        let isolation = self
            .runtime
            .isolation()
            .ok_or("discard requires a Git-root workspace")?;
        isolation
            .discard(workspace)
            .map_err(|error| error.to_string())?;
        session
            .append_shared_event(SessionEvent::IntegrationDiscarded {
                task_id: parent.task_id,
                execution_id: parent.execution_id,
                workspace,
                agent: owner.map_or(AgentId::ROOT, |record| record.agent),
                reason: reason.clone(),
            })
            .map_err(|error| error.to_string())?;
        Ok(serde_json::json!({
            "ok": true,
            "workspace": workspace.0,
            "reason": reason,
        }))
    }

    fn goal_update(&self, parent: &PtcExecution, update: GoalUpdate) -> Result<Value, String> {
        if update.is_empty() {
            return Err("goal(update): no recognized fields".to_string());
        }
        self.session()
            .append_shared_event(SessionEvent::GoalUpdated {
                task_id: parent.task_id,
                update,
            })
            .map_err(|error| error.to_string())?;
        let view = self.session().task_view(parent.task_id);
        serde_json::to_value(&view.goal).map_err(|error| error.to_string())
    }

    fn finish(
        &self,
        parent: &PtcExecution,
        request: FinishRequest,
    ) -> Result<FinishVerdict, String> {
        let session = self.session();
        // Poll workers first: a worker that finished a moment ago should not
        // block completion just because nothing has observed it yet.
        self.reap_settled_workers();
        let view = session.task_view(parent.task_id);
        let verdict = validate_finish(&view, &request);
        session
            .append_shared_event(SessionEvent::FinishProposed {
                task_id: parent.task_id,
                request: request.clone(),
                verdict: verdict.clone(),
            })
            .map_err(|error| error.to_string())?;
        if verdict.accepted {
            session
                .complete_task(
                    parent.task_id,
                    request.summary.clone(),
                    request.unresolved.clone(),
                    verdict.waived.clone(),
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(verdict)
    }

    fn process_spawn(
        &self,
        parent: &PtcExecution,
        spec: ProcessSpec,
    ) -> Result<ProcessSnapshot, String> {
        let session = self.session();
        let workspace = if parent.agent.is_root() {
            self.runtime.workspace_root()
        } else {
            worker_workspace(&self.runtime, parent.agent).map_err(|error| error.to_string())?
        };
        let capabilities = match worker_access(&self.runtime, parent.agent) {
            Some(AgentAccess::Read) => Capabilities::read_only(&workspace),
            _ => Capabilities::new(&workspace),
        };
        let process = session.allocate_process();
        let snapshot = self
            .runtime
            .processes()
            .spawn(process, parent.task_id, parent.agent, &capabilities, spec)
            .map_err(|error| error.to_string())?;
        session
            .append_shared_event(SessionEvent::ProcessSpawned {
                task_id: parent.task_id,
                agent: parent.agent,
                process: snapshot.id,
                argv: snapshot.argv.clone(),
                cwd: snapshot.cwd.clone(),
                label: snapshot.label.clone(),
                pid: snapshot.pid,
            })
            .map_err(|error| error.to_string())?;
        Ok(snapshot)
    }

    fn process_poll(&self, process: ProcessId) -> Result<ProcessSnapshot, String> {
        let snapshot = self
            .runtime
            .processes()
            .poll(process)
            .map_err(|error| error.to_string())?;
        self.record_terminal_process(&snapshot);
        Ok(snapshot)
    }

    fn process_tail(
        &self,
        process: ProcessId,
        stream: ProcessStream,
        lines: usize,
    ) -> Result<ProcessTail, String> {
        self.runtime
            .processes()
            .tail(process, stream, lines)
            .map_err(|error| error.to_string())
    }

    fn process_wait(
        &self,
        process: ProcessId,
        timeout_ms: Option<u64>,
    ) -> Result<ProcessSnapshot, String> {
        let snapshot = self
            .runtime
            .processes()
            .wait(process, timeout_ms, &self.cancelled)
            .map_err(|error| error.to_string())?;
        self.record_terminal_process(&snapshot);
        Ok(snapshot)
    }

    fn process_kill(&self, process: ProcessId) -> Result<ProcessSnapshot, String> {
        let snapshot = self
            .runtime
            .processes()
            .kill(process)
            .map_err(|error| error.to_string())?;
        self.record_terminal_process(&snapshot);
        Ok(snapshot)
    }

    fn process_write(&self, process: ProcessId, data: &str) -> Result<Value, String> {
        self.runtime
            .processes()
            .write_stdin(process, data)
            .map(|()| serde_json::json!({ "ok": true, "id": process.0 }))
            .map_err(|error| error.to_string())
    }

    fn process_list(&self) -> Vec<ProcessSnapshot> {
        self.runtime.processes().list()
    }
}

impl<M: Model + 'static> AgentRuntimeHost<M> {
    /// A second handle for a worker thread. Everything shared is behind `Arc`.
    fn clone_for_thread(&self) -> Self {
        Self {
            model: self.model.clone(),
            compiler: self.compiler.clone(),
            config: self.config.clone(),
            runtime: self.runtime.clone(),
            cancelled: self.cancelled.clone(),
        }
    }

    /// Records a process transition once, when it reaches a terminal state.
    fn record_terminal_process(&self, snapshot: &ProcessSnapshot) {
        if snapshot.state == crate::process::ProcessState::Running {
            return;
        }
        let recorded = self
            .session()
            .processes()
            .into_iter()
            .find(|record| record.process == snapshot.id)
            .map(|record| record.state);
        let label = crate::runtime::process_state_label(snapshot.state);
        if recorded.as_deref() != Some(label.as_str()) {
            self.runtime
                .record_process_state(snapshot.owner_task, snapshot);
        }
    }

    /// Cleans up after workers that have already settled.
    ///
    /// A settled worker's processes have no owner left to inspect them, so they
    /// are killed and journaled here. A *live* worker's processes are left
    /// alone: killing a running worker's build because its parent called
    /// finish() would destroy the work the refusal is about to demand.
    fn reap_settled_workers(&self) {
        for snapshot in self.runtime.workers().list() {
            if !snapshot.state.is_terminal() {
                continue;
            }
            for process in self.runtime.processes().kill_agent(snapshot.agent) {
                self.record_terminal_process(&process);
            }
        }
    }
}

/// Synchronous delegation, kept for compatibility with existing PTC programs.
pub fn delegate_compat(
    host: &dyn DelegationHost,
    parent: &PtcExecution,
    revision: &RevisionId,
    options: AgentSpawnOptions,
) -> Result<DelegateResult, String> {
    delegate_sync(host, parent, revision, options)
}

pub fn delegate_batch_compat(
    host: &dyn DelegationHost,
    parent: &PtcExecution,
    revision: &RevisionId,
    options: Vec<AgentSpawnOptions>,
) -> Vec<DelegateResult> {
    delegate_batch_sync(host, parent, revision, options)
}

#[derive(Default)]
struct RepeatState {
    normalized_source: Option<String>,
    fingerprint: Option<String>,
    identical_executions: usize,
}

fn drain_controls(
    cancelled: &Arc<AtomicBool>,
    controls: Option<&Receiver<AgentControl>>,
    events: &mut dyn FnMut(AgentEvent),
) -> Result<Vec<String>, AgentError> {
    let Some(controls) = controls else {
        return Ok(Vec::new());
    };
    let mut steering = Vec::new();
    loop {
        match controls.try_recv() {
            Ok(AgentControl::Steer(content)) => {
                events(AgentEvent::SteeringQueued {
                    content: content.clone(),
                });
                steering.push(content);
            }
            Ok(AgentControl::Interrupt) => cancelled.store(true, Ordering::Relaxed),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
        }
    }
    Ok(steering)
}

fn apply_steering(
    session: &Session,
    task_id: TaskId,
    steering: Vec<String>,
    events: &mut dyn FnMut(AgentEvent),
    repeated: &mut RepeatState,
) -> Result<(), AgentError> {
    for content in steering {
        session.append_shared_event(SessionEvent::SteeringQueued {
            task_id,
            content: content.clone(),
        })?;
        session.append_shared_event(SessionEvent::SteeringApplied {
            task_id,
            content: content.clone(),
        })?;
        events(AgentEvent::SteeringApplied { content });
    }
    *repeated = RepeatState::default();
    Ok(())
}

/// Applies steering and worker messages appended by another process, which is
/// how a detached task is steered.
fn apply_durable_steering(
    session: &Session,
    task_id: TaskId,
    events: &mut dyn FnMut(AgentEvent),
    repeated: &mut RepeatState,
) -> Result<(), AgentError> {
    let view = session.task_view(task_id);
    for content in view.pending_steering {
        session.append_shared_event(SessionEvent::SteeringApplied {
            task_id,
            content: content.clone(),
        })?;
        events(AgentEvent::SteeringApplied { content });
        *repeated = RepeatState::default();
    }
    // A worker consumes parent messages the same way.
    if !view.agent.is_root()
        && let Some(record) = session.agent(view.agent)
    {
        for content in record.inbox {
            session.append_shared_event(SessionEvent::AgentMessageApplied {
                agent: view.agent,
                task_id,
                content: content.clone(),
            })?;
            events(AgentEvent::SteeringApplied { content });
            *repeated = RepeatState::default();
        }
    }
    Ok(())
}

fn translate_ptc_events(ptc_events: &[PtcEvent], events: &mut dyn FnMut(AgentEvent)) {
    for event in ptc_events {
        match event {
            PtcEvent::HostCallStarted { call_id, name, .. } => {
                events(AgentEvent::PtcHostCallStarted {
                    call_id: *call_id,
                    name: name.clone(),
                })
            }
            PtcEvent::HostCallCompleted {
                call_id,
                name,
                ok,
                duration_ms,
                ..
            } => events(AgentEvent::PtcHostCallCompleted {
                call_id: *call_id,
                name: name.clone(),
                ok: *ok,
                duration_ms: *duration_ms,
            }),
            PtcEvent::EvidenceRecorded { evidence, .. } => events(AgentEvent::EvidenceRecorded {
                kind: evidence.kind.clone(),
                ok: evidence.ok,
            }),
            PtcEvent::WorkspaceRevisionChanged { .. } => {}
        }
    }
}

fn check_cancelled(
    session: &Session,
    task_id: TaskId,
    cancelled: &Arc<AtomicBool>,
) -> Result<(), AgentError> {
    if cancelled.load(Ordering::Relaxed) {
        session.interrupt_task(task_id)?;
        Err(AgentError::Cancelled)
    } else {
        Ok(())
    }
}

fn record_repeat(
    session: &Session,
    task_id: TaskId,
    events: &mut dyn FnMut(AgentEvent),
    fingerprint: String,
) -> Result<(), AgentError> {
    session.append_shared_event(SessionEvent::RepeatedActionDetected {
        task_id,
        fingerprint: fingerprint.clone(),
    })?;
    events(AgentEvent::RepeatedActionDetected { fingerprint });
    Ok(())
}

fn normalize_source(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn action_fingerprint(source: &str, outcome: &PtcOutcome, value: &Value) -> String {
    fingerprint_text(&format!("{source}\n{outcome:?}\n{}", compact_value(value)))
}

fn fingerprint_text(text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn compact_value(value: &Value) -> String {
    let text = serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
    text.chars().take(4096).collect()
}

fn output_kind(output: &ModelOutput) -> &'static str {
    match output {
        ModelOutput::Text(_) => "text",
        ModelOutput::Program { .. } => "program",
        ModelOutput::FunctionCall { .. } => "function_call",
    }
}

/// Renders a task's durable state for `mh inspect`.
pub fn inspect(workspace: &Path) -> Result<Vec<crate::runtime::TaskReport>, AgentError> {
    crate::runtime::task_reports(workspace).map_err(AgentError::Session)
}

#[cfg(test)]
mod tests {
    use std::sync::{Barrier, Mutex, mpsc};

    use super::*;
    use crate::context::CompiledContext;
    use crate::model::ProgramLanguage;

    struct ScriptedModel {
        outputs: Mutex<Vec<ModelOutput>>,
    }
    impl ScriptedModel {
        fn new(mut outputs: Vec<ModelOutput>) -> Self {
            outputs.reverse();
            Self {
                outputs: Mutex::new(outputs),
            }
        }
    }
    impl Model for ScriptedModel {
        fn generate(
            &self,
            _context: &CompiledContext,
            _stop: &GenerationStop,
            _events: &mut dyn FnMut(ModelEvent),
        ) -> Result<ModelOutput, ModelError> {
            self.outputs
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| ModelError::Protocol("script exhausted".to_string()))
        }
    }

    struct BlockingModel {
        barrier: Arc<Barrier>,
        outputs: Mutex<Vec<ModelOutput>>,
    }
    impl Model for BlockingModel {
        fn generate(
            &self,
            _context: &CompiledContext,
            stop: &GenerationStop,
            _events: &mut dyn FnMut(ModelEvent),
        ) -> Result<ModelOutput, ModelError> {
            let output = self
                .outputs
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| ModelError::Protocol("script exhausted".to_string()))?;
            if matches!(output, ModelOutput::Program { .. }) {
                self.barrier.wait();
                while stop.reason().is_none() {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(ModelError::Stopped(stop.reason().unwrap()))
            } else {
                Ok(output)
            }
        }
    }

    fn program(source: &str) -> ModelOutput {
        ModelOutput::Program {
            language: ProgramLanguage::JavaScript,
            source: source.to_string(),
        }
    }

    fn finish(summary: &str) -> ModelOutput {
        program(&format!(
            "return finish({{ summary: \"{summary}\", force: true }});"
        ))
    }

    fn config() -> AgentConfig {
        AgentConfig {
            trust_store: None,
            user_prelude: None,
            ..AgentConfig::default()
        }
    }

    #[test]
    fn text_output_does_not_complete_the_task() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::new(
            ScriptedModel::new(vec![
                program("write(\"a.txt\", \"hello\");"),
                ModelOutput::Text("Implemented most of it.".into()),
            ]),
            config(),
        );
        let outcome = agent
            .run_task_controlled(
                dir.path(),
                "write",
                &Arc::new(AtomicBool::new(false)),
                None,
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(
            outcome,
            TaskOutcome::AwaitingUser("Implemented most of it.".to_string())
        );
        assert!(!outcome.is_complete());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "hello"
        );

        let session = Session::resume(dir.path()).unwrap();
        let task = session.root_task().unwrap();
        assert_eq!(session.task_view(task).status(), TaskStatus::WaitingUser);
        assert!(
            session.task_view(task).status().is_active(),
            "a progress report must leave the task active"
        );
        assert!(
            !session
                .events()
                .iter()
                .any(|record| matches!(record.event, SessionEvent::TaskCompleted { .. })),
            "text must not append TaskCompleted"
        );
    }

    #[test]
    fn only_finish_completes_the_task() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::new(
            ScriptedModel::new(vec![
                ModelOutput::Text("Still working.".into()),
                finish("all done"),
            ]),
            config(),
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        let parked = agent
            .run_task_controlled(dir.path(), "work", &cancelled, None, &mut |_| {})
            .unwrap();
        assert!(!parked.is_complete());

        // A follow-up message continues the same durable task.
        let completed = agent
            .send_message_controlled(dir.path(), "continue", &cancelled, None, &mut |_| {})
            .unwrap();
        assert_eq!(completed, TaskOutcome::Completed("all done".to_string()));

        let session = Session::resume(dir.path()).unwrap();
        let tasks = session.tasks();
        assert_eq!(tasks.len(), 1, "a follow-up must not start a second task");
        assert_eq!(tasks[0].status(), TaskStatus::Completed);
    }

    #[test]
    fn a_full_window_rolls_over_instead_of_ending_the_task() {
        let dir = tempfile::tempdir().unwrap();
        let mut outputs = vec![program(
            "return goal({ findings: [{ summary: \"EARLY_FINDING\" }], pending: [{ title: \"finish up\" }] });",
        )];
        // Fill two windows of two turns each, then finish in the third.
        for _ in 0..4 {
            outputs.push(program("return read(\"missing\");"));
        }
        outputs.push(finish("converged"));
        let agent = Agent::new(
            ScriptedModel::new(outputs),
            AgentConfig {
                max_turns_per_window: 2,
                ..config()
            },
        );
        let mut windows = Vec::new();
        let outcome = agent
            .run_task_controlled(
                dir.path(),
                "long task",
                &Arc::new(AtomicBool::new(false)),
                None,
                &mut |event| {
                    if let AgentEvent::ContextWindowStarted { window } = event {
                        windows.push(window);
                    }
                },
            )
            .unwrap();
        assert!(outcome.is_complete());
        assert!(
            windows.len() >= 3,
            "expected at least two rollovers, saw windows {windows:?}"
        );

        let session = Session::resume(dir.path()).unwrap();
        let task = session.root_task().unwrap();
        let view = session.task_view(task);
        assert!(view.window >= 2, "task crossed multiple context windows");
        assert_eq!(view.objective(), "long task", "the goal survives rollover");
        let checkpoint = view
            .checkpoint
            .expect("a rolled-over task has a checkpoint");
        assert!(
            checkpoint
                .important_findings
                .iter()
                .any(|finding| finding.summary == "EARLY_FINDING"),
            "an early finding must survive every rollover"
        );
        assert_eq!(
            session.tasks().len(),
            1,
            "rollover keeps the same durable task id"
        );
    }

    /// Routes turns by role and gates the worker, so the root provably calls
    /// `finish()` while the worker is still live. A shared output script would
    /// race: the worker might settle first and the assertion would be vacuous.
    struct GatedRoleModel {
        root: Mutex<Vec<ModelOutput>>,
        /// Cleared until the root reaches its join turn.
        release_worker: Arc<AtomicBool>,
        /// Set once the worker has actually started its model turn.
        worker_running: Arc<AtomicBool>,
    }

    impl Model for GatedRoleModel {
        fn generate(
            &self,
            context: &CompiledContext,
            _stop: &GenerationStop,
            _events: &mut dyn FnMut(ModelEvent),
        ) -> Result<ModelOutput, ModelError> {
            if context.render().contains("delegated worker with a parent") {
                self.worker_running.store(true, Ordering::Relaxed);
                while !self.release_worker.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                return Ok(ModelOutput::Text("inspected".into()));
            }
            let mut root = self.root.lock().unwrap();
            let output = root
                .pop()
                .ok_or_else(|| ModelError::Protocol("script exhausted".to_string()))?;
            // The remaining turns are join-then-finish: the worker may settle.
            if root.len() == 1 {
                self.release_worker.store(true, Ordering::Relaxed);
            }
            Ok(output)
        }
    }

    #[test]
    fn finish_is_refused_while_a_worker_is_unresolved() {
        let dir = repository();
        let release_worker = Arc::new(AtomicBool::new(false));
        let worker_running = Arc::new(AtomicBool::new(false));
        let mut root = vec![
            program(
                "var a = agent_spawn({ task: \"inspect\", access: \"read\" });\nreturn { spawned: a.agent, state: a.state };",
            ),
            // The worker is still live here, so this must be refused.
            program("return finish({ summary: \"early\", force: true });"),
            program(
                "var joined = agent_join(1);\nreturn { joined: joined.ok, summary: joined.summary };",
            ),
            finish("done"),
        ];
        root.reverse();
        let agent = Agent::new(
            GatedRoleModel {
                root: Mutex::new(root),
                release_worker: release_worker.clone(),
                worker_running: worker_running.clone(),
            },
            config(),
        );
        let outcome = agent
            .run_task_controlled(
                dir.path(),
                "orchestrate",
                &Arc::new(AtomicBool::new(false)),
                None,
                &mut |_| {},
            )
            .unwrap();
        assert!(outcome.is_complete());
        assert!(
            worker_running.load(Ordering::Relaxed),
            "the worker must have run concurrently with the root"
        );

        let session = Session::resume(dir.path()).unwrap();
        let verdicts = session
            .events()
            .into_iter()
            .filter_map(|record| match record.event {
                SessionEvent::FinishProposed { verdict, .. } => Some(verdict),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(verdicts.len(), 2, "one refusal, then one acceptance");
        assert!(
            !verdicts[0].accepted
                && verdicts[0].objections.iter().any(|objection| matches!(
                    objection,
                    crate::goal::FinishObjection::UnresolvedAgents { .. }
                )),
            "force must not complete over a live worker: {:?}",
            verdicts[0]
        );
        assert!(verdicts[1].accepted);

        // The joined worker settled as its own durable task.
        let worker = session.agent(AgentId(1)).expect("worker was journaled");
        assert_eq!(worker.state, AgentState::Completed);
        assert_eq!(worker.result.unwrap().summary, "inspected");
        assert_eq!(session.tasks().len(), 2, "worker owns a durable task");
    }

    fn git(dir: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(output.status.success(), "git {args:?}");
    }

    fn repository() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        git(dir.path(), &["init", "-q"]);
        git(dir.path(), &["config", "user.name", "mh test"]);
        git(dir.path(), &["config", "user.email", "mh@example.invalid"]);
        std::fs::write(dir.path().join("tracked"), "base").unwrap();
        git(dir.path(), &["add", "."]);
        git(dir.path(), &["commit", "-qm", "base"]);
        dir
    }

    #[test]
    fn steering_during_inference_discards_stale_program() {
        let dir = tempfile::tempdir().unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let model = BlockingModel {
            barrier: barrier.clone(),
            outputs: Mutex::new(vec![
                ModelOutput::Text("done".into()),
                program("write(\"stale.txt\", \"bad\");"),
            ]),
        };
        let agent = Agent::new(model, config());
        let (tx, rx) = mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let workspace = dir.path();
        std::thread::scope(|scope| {
            let worker_cancelled = cancelled.clone();
            let handle = scope.spawn(move || {
                agent.run_task_controlled(
                    workspace,
                    "start",
                    &worker_cancelled,
                    Some(&rx),
                    &mut |_| {},
                )
            });
            barrier.wait();
            tx.send(AgentControl::Steer("do not write".into())).unwrap();
            assert_eq!(
                handle.join().unwrap().unwrap().message(),
                "done",
                "steering supersedes the stale program"
            );
        });
        assert!(!dir.path().join("stale.txt").exists());
    }

    #[test]
    fn durable_steering_from_another_process_is_applied() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::new(
            ScriptedModel::new(vec![program("return read(\"missing\");"), finish("done")]),
            config(),
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        // Start the task, park it, then steer it as `mh steer` would.
        let task = Agent::<ScriptedModel>::start_detached_task(dir.path(), "detached").unwrap();
        crate::runtime::steer_task(dir.path(), Some(task), "also check the docs").unwrap();
        let outcome = agent
            .resume_task(dir.path(), Some(task), &cancelled, None, &mut |_| {})
            .unwrap();
        assert!(outcome.is_complete());
        let session = Session::resume(dir.path()).unwrap();
        assert!(
            session.events().iter().any(|record| matches!(
                &record.event,
                SessionEvent::SteeringApplied { content, .. } if content == "also check the docs"
            )),
            "durable steering must be applied by the running task"
        );
    }

    #[test]
    fn durable_cancellation_stops_a_running_task() {
        let dir = tempfile::tempdir().unwrap();
        let task = Agent::<ScriptedModel>::start_detached_task(dir.path(), "cancel me").unwrap();
        crate::runtime::cancel_task(dir.path(), Some(task)).unwrap();
        let agent = Agent::new(ScriptedModel::new(vec![finish("never")]), config());
        let error = agent
            .resume_task(
                dir.path(),
                Some(task),
                &Arc::new(AtomicBool::new(false)),
                None,
                &mut |_| {},
            )
            .unwrap_err();
        assert!(matches!(error, AgentError::Cancelled));
        assert_eq!(
            Session::resume(dir.path())
                .unwrap()
                .task_view(task)
                .status(),
            TaskStatus::Cancelled
        );
    }

    #[test]
    fn translates_runtime_events_into_session_and_agent_events() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::new(
            ScriptedModel::new(vec![program("return read(\"missing\");"), finish("done")]),
            config(),
        );
        let mut emitted = Vec::new();
        agent
            .run_task_controlled(
                dir.path(),
                "inspect",
                &Arc::new(AtomicBool::new(false)),
                None,
                &mut |event| emitted.push(event),
            )
            .unwrap();
        assert!(emitted.iter().any(
            |event| matches!(event, AgentEvent::PtcHostCallStarted { name, .. } if name == "read")
        ));
        assert!(emitted.iter().any(
            |event| matches!(event, AgentEvent::PtcHostCallCompleted { name, .. } if name == "read")
        ));
        assert!(
            emitted
                .iter()
                .any(|event| matches!(event, AgentEvent::TaskCompleted { .. }))
        );
    }

    #[test]
    fn third_identical_action_is_not_executed() {
        let dir = tempfile::tempdir().unwrap();
        let same = "return read(\"missing\");";
        let agent = Agent::new(
            ScriptedModel::new(vec![
                program(same),
                program(same),
                program(same),
                finish("different"),
            ]),
            config(),
        );
        let mut emitted = Vec::new();
        assert!(
            agent
                .run_task_controlled(
                    dir.path(),
                    "inspect",
                    &Arc::new(AtomicBool::new(false)),
                    None,
                    &mut |event| emitted.push(event)
                )
                .unwrap()
                .is_complete()
        );
        assert_eq!(
            emitted
                .iter()
                .filter(|event| matches!(event, AgentEvent::PtcStarted))
                .count(),
            3,
            "two executions of the repeat plus the finish program"
        );
        assert!(
            emitted
                .iter()
                .any(|event| matches!(event, AgentEvent::RepeatedActionDetected { .. }))
        );
    }

    #[test]
    fn context_overflow_propagates() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::new(
            ScriptedModel::new(vec![]),
            AgentConfig {
                context_tokens: 1,
                ..config()
            },
        );
        assert!(matches!(
            agent.run_task(dir.path(), "too large", &Arc::new(AtomicBool::new(false))),
            Err(AgentError::Context(_))
        ));
    }
}
