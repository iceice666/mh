//! PTC-first agent loop with safe-point steering (spec §16–§18).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

use crate::checkpoint::{CheckpointError, CheckpointStore};
use crate::context::{ContextCompiler, ContextError};
use crate::delegation::{
    DelegateResult, DelegationAccess, DelegationBudget, DelegationHost, DelegationOptions,
    IntegrationResponse,
};
use crate::identity::{ExecutionId, IsolatedWorkspaceId, RevisionId, TaskId};
use crate::isolation::{IntegrationResult, IsolationStore};
use crate::model::{
    GenerationStop, GenerationStopReason, Model, ModelError, ModelEvent, ModelOutput, lower,
};
use crate::ptc::prelude::{self, Prelude, PreludeError};
use crate::ptc::runtime::{PtcEvent, PtcEventSink, PtcExecution};
use crate::ptc::trust::{Trust, TrustDecision, TrustError, TrustStore};
use crate::ptc::{PtcBudget, PtcOutcome, PtcRuntime};
use crate::session::{Session, SessionError, SessionEvent};
use crate::tools::Capabilities;
use crate::workspace::{WorkspaceError, WorkspaceTracker, WorkspaceTrackerImpl};

const REPEAT_WARNING: &str = "The previous action was repeated without changing the result. Choose a different approach instead of retrying the same PTC program.";

/// Decides whether a never-before-seen workspace prelude may run.
///
/// The library never prompts: a front end supplies this, so a non-interactive
/// run can refuse by default instead of blocking on a terminal that is not
/// there.
pub type PreludeConfirm = Arc<dyn Fn(&Prelude) -> TrustDecision + Send + Sync>;

#[derive(Clone)]
pub struct AgentConfig {
    pub max_turns: usize,
    pub context_tokens: usize,
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
            .field("max_turns", &self.max_turns)
            .field("context_tokens", &self.context_tokens)
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
            max_turns: 32,
            context_tokens: 32_000,
            ptc_budget: PtcBudget::default(),
            delegation_budget: DelegationBudget::default(),
            user_prelude: prelude::user_prelude_path(),
            trust_store: TrustStore::user().ok(),
            confirm_prelude: None,
        }
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
    TurnLimit(usize),
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
            Self::TurnLimit(limit) => write!(f, "agent reached {limit} model turns"),
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

pub struct Agent<M> {
    model: Arc<M>,
    compiler: ContextCompiler,
    config: AgentConfig,
}

impl<M: Model + 'static> Agent<M> {
    pub fn new(model: M, config: AgentConfig) -> Self {
        let compiler = ContextCompiler {
            max_tokens: config.context_tokens,
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
    }
    pub fn run_task_controlled(
        &self,
        workspace: &Path,
        task: &str,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<String, AgentError> {
        let mut session = Session::open(workspace)?;
        let task_id = session.begin_task(task)?;
        session.append(SessionEvent::UserMessage {
            task_id,
            content: task.to_string(),
        })?;
        self.run_session(&mut session, cancelled, controls, events)
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
    }

    pub fn resume_controlled(
        &self,
        workspace: &Path,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<String, AgentError> {
        let mut session = Session::resume(workspace)?;
        self.run_session(&mut session, cancelled, controls, events)
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
    }
    pub fn send_message_controlled(
        &self,
        workspace: &Path,
        message: &str,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<String, AgentError> {
        let mut session = Session::open(workspace)?;
        let task_id = session.begin_task(message)?;
        session.append(SessionEvent::UserMessage {
            task_id,
            content: message.to_string(),
        })?;
        self.run_session(&mut session, cancelled, controls, events)
    }

    fn run_session(
        &self,
        session: &mut Session,
        cancelled: &Arc<AtomicBool>,
        controls: Option<&Receiver<AgentControl>>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<String, AgentError> {
        let workspace = session.state().workspace.clone();
        let tracker_impl = WorkspaceTrackerImpl::open(&workspace)?;
        let tracker: Arc<dyn WorkspaceTracker> = Arc::new(tracker_impl);
        // Discovered once per session run: a prelude edited mid-run would
        // otherwise change the tool set the model was told about.
        let discovered = prelude::discover(&workspace, self.config.user_prelude.as_deref())?;
        let prelude = self.authorize_prelude(discovered, events)?.map(Arc::new);
        let prelude_id = prelude.as_deref().map(Prelude::identity);
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
        let checkpoints =
            CheckpointStore::open(&workspace, tracker.clone())?.with_prelude(prelude_id.clone());
        let compiler = self.compiler_with_prelude(prelude.as_deref());
        let runtime = PtcRuntime::new(
            Capabilities::new(workspace),
            session.result_store()?,
            self.config.ptc_budget.clone(),
            tracker,
            checkpoints,
        )
        .with_prelude(prelude.clone())
        .with_delegation(Arc::new(AgentDelegationHost::new(
            self.model.clone(),
            compiler.clone(),
            self.config.clone(),
            session.clone(),
            cancelled.clone(),
            prelude,
        )));
        let mut repeated = RepeatState::default();

        for _turn in 0..self.config.max_turns {
            let task_id = session.state().active_task.ok_or_else(|| {
                SessionError::State("cannot run an agent without an active task".to_string())
            })?;
            let steering = drain_controls(cancelled, controls, events)?;
            if !steering.is_empty() {
                apply_steering(session, task_id, steering, events, &mut repeated)?;
                session.reconcile_workspace()?;
            }
            check_cancelled(session, cancelled)?;
            session.reconcile_workspace()?;
            let context = compiler.compile(session)?;
            session.append(SessionEvent::ModelStarted { task_id })?;
            let (output, steering) =
                self.generate_controlled(&context, cancelled, controls, events)?;
            if !steering.is_empty() {
                session.append(SessionEvent::ModelSuperseded { task_id })?;
                apply_steering(session, task_id, steering, events, &mut repeated)?;
                session.reconcile_workspace()?;
                continue;
            }
            session.append(SessionEvent::ModelCompleted {
                task_id,
                output_kind: output_kind(&output).to_string(),
            })?;
            check_cancelled(session, cancelled)?;
            match output {
                ModelOutput::Text(text) => {
                    session.append(SessionEvent::AssistantMessage {
                        task_id,
                        content: text.clone(),
                    })?;
                    session.complete_task(Some(text.clone()))?;
                    return Ok(text);
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
                        apply_steering(session, task_id, steering, events, &mut repeated)?;
                        session.reconcile_workspace()?;
                        continue;
                    }
                    check_cancelled(session, cancelled)?;
                    let execution_id = session.begin_execution()?;
                    let start_revision = session.state().current_revision.clone();
                    session.append(SessionEvent::PtcStarted {
                        task_id,
                        execution_id,
                        source: source.clone(),
                        start_revision: start_revision.clone(),
                    })?;
                    events(AgentEvent::PtcStarted);
                    let execution = PtcExecution {
                        task_id,
                        execution_id,
                        start_revision,
                    };
                    let sink: Arc<dyn PtcEventSink> = Arc::new(session.ptc_event_sink());
                    let result = runtime.execute(&source, cancelled.clone(), execution, Some(sink));
                    translate_ptc_events(&result.events, events);
                    events(AgentEvent::ToolCompleted {
                        outcome: format!("{:?}", result.outcome),
                        tool_calls: result.tool_calls,
                        duration_ms: result.duration_ms,
                    });
                    session.append_ptc_result(&result)?;
                    if result.outcome == PtcOutcome::Interrupted {
                        session.interrupt_task()?;
                        return Err(AgentError::Cancelled);
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
        Err(AgentError::TurnLimit(self.config.max_turns))
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

struct AgentDelegationHost<M> {
    model: Arc<M>,
    compiler: ContextCompiler,
    config: AgentConfig,
    session: Session,
    cancelled: Arc<AtomicBool>,
    prelude: Option<Arc<Prelude>>,
    isolation: Option<IsolationStore>,
    owner_task_id: std::sync::atomic::AtomicU64,
    children_started: std::sync::atomic::AtomicUsize,
    active_children: std::sync::atomic::AtomicUsize,
    next_child_task: std::sync::atomic::AtomicU64,
    next_child_execution: std::sync::atomic::AtomicU64,
    next_workspace: std::sync::atomic::AtomicU64,
}

impl<M> Drop for AgentDelegationHost<M> {
    fn drop(&mut self) {
        let owner = self.owner_task_id.load(Ordering::Relaxed);
        if owner == 0 {
            return;
        }
        if let Some(isolation) = self.isolation.as_ref() {
            let _ = isolation.cleanup_owner(TaskId(owner));
        }
    }
}

impl<M: Model> AgentDelegationHost<M> {
    fn new(
        model: Arc<M>,
        compiler: ContextCompiler,
        config: AgentConfig,
        session: Session,
        cancelled: Arc<AtomicBool>,
        prelude: Option<Arc<Prelude>>,
    ) -> Self {
        let isolation = IsolationStore::open(session.workspace_root()).ok();
        let (next_task, next_execution) = session.allocate_child_ids();
        let next_workspace = session.next_isolated_workspace_id();
        Self {
            model,
            compiler,
            config,
            session,
            cancelled,
            prelude,
            isolation,
            owner_task_id: std::sync::atomic::AtomicU64::new(0),
            children_started: std::sync::atomic::AtomicUsize::new(0),
            active_children: std::sync::atomic::AtomicUsize::new(0),
            next_child_task: std::sync::atomic::AtomicU64::new(next_task.0),
            next_child_execution: std::sync::atomic::AtomicU64::new(next_execution.0),
            next_workspace: std::sync::atomic::AtomicU64::new(next_workspace.0),
        }
    }

    fn failure(
        task_id: TaskId,
        execution_id: ExecutionId,
        revision: RevisionId,
        summary: impl Into<String>,
    ) -> DelegateResult {
        DelegateResult {
            task_id,
            execution_id,
            ok: false,
            summary: summary.into(),
            base_revision: revision.clone(),
            final_revision: revision,
            changed: false,
            workspace: None,
            evidence: Vec::new(),
            findings: serde_json::Value::Null,
        }
    }

    fn run_child(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        options: DelegationOptions,
    ) -> DelegateResult {
        let index = self.children_started.fetch_add(1, Ordering::Relaxed);
        let task_id = TaskId(self.next_child_task.fetch_add(1, Ordering::Relaxed));
        let execution_id = ExecutionId(self.next_child_execution.fetch_add(1, Ordering::Relaxed));
        if index >= self.config.delegation_budget.max_children {
            return Self::failure(
                task_id,
                execution_id,
                revision.clone(),
                "delegation child budget exceeded",
            );
        }
        let active = self.active_children.fetch_add(1, Ordering::AcqRel) + 1;
        if active > self.config.delegation_budget.max_parallel_children.max(1) {
            self.active_children.fetch_sub(1, Ordering::AcqRel);
            return Self::failure(
                task_id,
                execution_id,
                revision.clone(),
                "parallel delegation budget exceeded",
            );
        }
        struct ActiveGuard<'a>(&'a std::sync::atomic::AtomicUsize);
        impl Drop for ActiveGuard<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        let _guard = ActiveGuard(&self.active_children);
        let started = SessionEvent::DelegationStarted {
            parent_task_id: parent.task_id,
            parent_execution_id: parent.execution_id,
            child_task_id: task_id,
            child_execution_id: execution_id,
            access: options.access,
            base_revision: revision.clone(),
        };
        if let Err(error) = self.session.append_shared_event(started) {
            return Self::failure(task_id, execution_id, revision.clone(), error.to_string());
        }
        let workspace_id = (options.access == DelegationAccess::IsolatedWrite)
            .then(|| IsolatedWorkspaceId(self.next_workspace.fetch_add(1, Ordering::Relaxed)));
        self.owner_task_id
            .store(parent.task_id.0, Ordering::Relaxed);
        let workspace_record = match workspace_id {
            Some(id) => match self.isolation.as_ref().map_or_else(
                || Err("isolated-write requires a Git-root workspace".to_string()),
                |isolation| {
                    isolation
                        .create(id, parent.task_id, execution_id, revision.clone())
                        .map_err(|error| error.to_string())
                },
            ) {
                Ok(record) => Some(record),
                Err(error) => {
                    let result =
                        Self::failure(task_id, execution_id, revision.clone(), error.to_string());
                    let _ = self
                        .session
                        .append_shared_event(SessionEvent::DelegationCompleted {
                            parent_task_id: parent.task_id,
                            parent_execution_id: parent.execution_id,
                            result: result.clone(),
                        });
                    return result;
                }
            },
            None => None,
        };
        let workspace = workspace_record.as_ref().map_or_else(
            || self.session.workspace_root(),
            |record| record.path.clone(),
        );
        let tracker_impl = match WorkspaceTrackerImpl::open(&workspace) {
            Ok(tracker) => tracker,
            Err(error) => {
                return Self::failure(task_id, execution_id, revision.clone(), error.to_string());
            }
        };
        let tracker: Arc<dyn WorkspaceTracker> = Arc::new(tracker_impl);
        let child_revision = match tracker.current_revision() {
            Ok(child_revision) => child_revision,
            Err(error) => {
                return Self::failure(task_id, execution_id, revision.clone(), error.to_string());
            }
        };
        if child_revision.id != *revision {
            return Self::failure(
                task_id,
                execution_id,
                revision.clone(),
                format!(
                    "delegation base revision changed before child start: expected {}, found {}",
                    revision.0, child_revision.id.0
                ),
            );
        }
        let checkpoints = match CheckpointStore::open(&workspace, tracker.clone()) {
            Ok(checkpoints) => {
                checkpoints.with_prelude(self.prelude.as_deref().map(Prelude::identity))
            }
            Err(error) => {
                return Self::failure(task_id, execution_id, revision.clone(), error.to_string());
            }
        };
        let capabilities = match options.access {
            DelegationAccess::Read => Capabilities::read_only(&workspace),
            DelegationAccess::IsolatedWrite => Capabilities::new(&workspace),
        };
        let runtime = PtcRuntime::new(
            capabilities,
            match self.session.result_store() {
                Ok(store) => store,
                Err(error) => {
                    return Self::failure(
                        task_id,
                        execution_id,
                        revision.clone(),
                        error.to_string(),
                    );
                }
            },
            self.config.ptc_budget.clone(),
            tracker.clone(),
            checkpoints,
        )
        .with_prelude(self.prelude.clone());
        let sink: Arc<dyn PtcEventSink> = Arc::new(self.session.child_ptc_event_sink(workspace_id));
        let mut context = match self.compiler.compile_child(
            &options.task,
            &options.context,
            &child_revision,
            options.access,
        ) {
            Ok(context) => context,
            Err(error) => {
                return Self::failure(task_id, execution_id, revision.clone(), error.to_string());
            }
        };
        let mut summary = None;
        let mut last_value = serde_json::Value::Null;
        let mut last_error = None;
        let turns = self.config.delegation_budget.max_child_turns.max(1);
        for _ in 0..turns {
            if self.cancelled.load(Ordering::Relaxed) {
                let _ = self
                    .session
                    .append_shared_event(SessionEvent::DelegationCancelled {
                        parent_task_id: parent.task_id,
                        parent_execution_id: parent.execution_id,
                        child_task_id: task_id,
                        child_execution_id: execution_id,
                    });
                return Self::failure(task_id, execution_id, revision.clone(), "cancelled");
            }
            let stop = GenerationStop::new(self.cancelled.clone());
            let output = match self.model.generate(&context, &stop, &mut |_| {}) {
                Ok(output) => output,
                Err(error) => {
                    last_error = Some(error.to_string());
                    break;
                }
            };
            match output {
                ModelOutput::Text(text) => {
                    summary = Some(text);
                    break;
                }
                executable => {
                    let source = lower(&executable).expect("executable child output lowers to PTC");
                    let current = match tracker.current_revision() {
                        Ok(revision) => revision,
                        Err(error) => {
                            last_error = Some(error.to_string());
                            break;
                        }
                    };
                    let execution = PtcExecution {
                        task_id,
                        execution_id,
                        start_revision: current.id,
                    };
                    let result = runtime.execute(
                        &source,
                        self.cancelled.clone(),
                        execution,
                        Some(sink.clone()),
                    );
                    let _ = self.session.append_child_ptc_result(&result);
                    last_value = result.value.clone();
                    last_error = match &result.outcome {
                        PtcOutcome::Completed => None,
                        PtcOutcome::Interrupted => Some("cancelled".to_string()),
                        PtcOutcome::BudgetExceeded(kind) => {
                            Some(format!("budget exceeded: {kind}"))
                        }
                        PtcOutcome::Failed(error) => Some(error.clone()),
                    };
                    let current = match tracker.current_revision() {
                        Ok(revision) => revision,
                        Err(error) => {
                            last_error = Some(error.to_string());
                            break;
                        }
                    };
                    context = match self.compiler.compile_child_followup(
                        &options.task,
                        &options.context,
                        &current,
                        options.access,
                        &last_value,
                        last_error.as_deref(),
                    ) {
                        Ok(context) => context,
                        Err(error) => {
                            last_error = Some(error.to_string());
                            break;
                        }
                    };
                }
            }
        }
        let final_record = workspace_id.and_then(|id| {
            self.isolation
                .as_ref()
                .and_then(|isolation| isolation.finalize(id).ok())
        });
        let final_revision = final_record
            .as_ref()
            .map(|record| record.final_revision.clone())
            .or_else(|| tracker.current_revision().ok().map(|revision| revision.id))
            .unwrap_or_else(|| revision.clone());
        let changed = final_revision != *revision;
        let ok = summary.is_some();
        let result = DelegateResult {
            task_id,
            execution_id,
            ok,
            summary: summary.unwrap_or_else(|| {
                last_error.unwrap_or_else(|| format!("child reached {turns} model turns"))
            }),
            base_revision: revision.clone(),
            final_revision,
            changed,
            workspace: final_record.as_ref().map(|record| record.id),
            evidence: self.session.evidence_for_execution(execution_id),
            findings: {
                let encoded = serde_json::to_vec(&last_value).unwrap_or_default();
                if encoded.len() <= self.config.delegation_budget.max_findings_bytes {
                    last_value
                } else {
                    serde_json::json!({
                        "truncated": true,
                        "bytes": encoded.len(),
                    })
                }
            },
        };
        let _ = self
            .session
            .append_shared_event(SessionEvent::DelegationCompleted {
                parent_task_id: parent.task_id,
                parent_execution_id: parent.execution_id,
                result: result.clone(),
            });
        result
    }
}

impl<M: Model> DelegationHost for AgentDelegationHost<M> {
    fn delegate(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        options: DelegationOptions,
    ) -> DelegateResult {
        self.run_child(parent, revision, options)
    }

    fn delegate_batch(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        options: Vec<DelegationOptions>,
    ) -> Vec<DelegateResult> {
        let limit = self.config.delegation_budget.max_parallel_children.max(1);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let output = std::sync::Mutex::new(vec![None; options.len()]);
        std::thread::scope(|scope| {
            for _ in 0..limit.min(options.len()) {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= options.len() {
                            break;
                        }
                        let result = self.run_child(parent, revision, options[index].clone());
                        output.lock().expect("delegate output poisoned")[index] = Some(result);
                    }
                });
            }
        });
        output
            .into_inner()
            .expect("delegate output poisoned")
            .into_iter()
            .map(|result| result.expect("delegate worker filled every result"))
            .collect()
    }

    fn integrate(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        workspace: IsolatedWorkspaceId,
    ) -> IntegrationResponse {
        let Some(isolation) = self.isolation.as_ref() else {
            return IntegrationResponse::error("integration requires a Git-root workspace");
        };
        let record = match isolation.load(workspace) {
            Ok(record) => record,
            Err(error) => return IntegrationResponse::error(error.to_string()),
        };
        let _ = self
            .session
            .append_shared_event(SessionEvent::IntegrationStarted {
                parent_task_id: parent.task_id,
                parent_execution_id: parent.execution_id,
                workspace,
                child_execution_id: record.owner_execution_id,
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
                let _ = self
                    .session
                    .append_shared_event(SessionEvent::WorkspaceRevisionChanged {
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
                let _ = self
                    .session
                    .append_shared_event(SessionEvent::IntegrationCompleted {
                        parent_task_id: parent.task_id,
                        parent_execution_id: parent.execution_id,
                        child_execution_id: record.owner_execution_id,
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
                let _ = self
                    .session
                    .append_shared_event(SessionEvent::IntegrationConflict {
                        parent_task_id: parent.task_id,
                        parent_execution_id: parent.execution_id,
                        child_execution_id: record.owner_execution_id,
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
    session: &mut Session,
    task_id: crate::identity::TaskId,
    steering: Vec<String>,
    events: &mut dyn FnMut(AgentEvent),
    repeated: &mut RepeatState,
) -> Result<(), AgentError> {
    for content in steering {
        session.append(SessionEvent::SteeringQueued {
            task_id,
            content: content.clone(),
        })?;
        session.append(SessionEvent::SteeringApplied {
            task_id,
            content: content.clone(),
        })?;
        events(AgentEvent::SteeringApplied { content });
    }
    *repeated = RepeatState::default();
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

fn check_cancelled(session: &mut Session, cancelled: &Arc<AtomicBool>) -> Result<(), AgentError> {
    if cancelled.load(Ordering::Relaxed) {
        session.interrupt_task()?;
        Err(AgentError::Cancelled)
    } else {
        Ok(())
    }
}

fn record_repeat(
    session: &mut Session,
    task_id: crate::identity::TaskId,
    events: &mut dyn FnMut(AgentEvent),
    fingerprint: String,
) -> Result<(), AgentError> {
    session.append(SessionEvent::RepeatedActionDetected {
        task_id,
        fingerprint: fingerprint.clone(),
    })?;
    events(AgentEvent::RepeatedActionDetected { fingerprint });
    // The context compiler renders this durable event as an instruction.
    let _ = REPEAT_WARNING;
    Ok(())
}

fn normalize_source(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn action_fingerprint(source: &str, outcome: &PtcOutcome, value: &serde_json::Value) -> String {
    fingerprint_text(&format!("{source}\n{outcome:?}\n{}", compact_value(value)))
}

fn fingerprint_text(text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn compact_value(value: &serde_json::Value) -> String {
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

    #[test]
    fn existing_wrappers_execute_then_finish() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::new(
            ScriptedModel::new(vec![
                program("write(\"a.txt\", \"hello\");"),
                ModelOutput::Text("done".into()),
            ]),
            AgentConfig::default(),
        );
        let answer = agent
            .run_task(dir.path(), "write", &Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert_eq!(answer, "done");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "hello"
        );
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
        let agent = Agent::new(model, AgentConfig::default());
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

            assert_eq!(handle.join().unwrap().unwrap(), "done");
        });
        assert!(!dir.path().join("stale.txt").exists());
    }

    #[test]
    fn steering_during_ptc_is_applied_after_completion() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::new(
            ScriptedModel::new(vec![
                program("write(\"during.txt\", \"completed\");"),
                ModelOutput::Text("done".into()),
            ]),
            AgentConfig::default(),
        );
        let (tx, rx) = mpsc::channel();
        let mut sent = false;
        let answer = agent
            .run_task_controlled(
                dir.path(),
                "write",
                &Arc::new(AtomicBool::new(false)),
                Some(&rx),
                &mut |event| {
                    if event == AgentEvent::PtcStarted && !sent {
                        tx.send(AgentControl::Steer("continue only after the write".into()))
                            .unwrap();
                        sent = true;
                    }
                },
            )
            .unwrap();
        assert_eq!(answer, "done");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("during.txt")).unwrap(),
            "completed"
        );
        let session = Session::resume(dir.path()).unwrap();
        let completed = session
            .events()
            .iter()
            .position(|record| matches!(record.event, SessionEvent::PtcCompleted { .. }))
            .unwrap();
        let applied = session
            .events()
            .iter()
            .position(|record| matches!(record.event, SessionEvent::SteeringApplied { .. }))
            .unwrap();
        assert!(completed < applied);
    }

    #[test]
    fn translates_runtime_events_into_session_and_agent_events() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::new(
            ScriptedModel::new(vec![
                program("return read(\"missing\");"),
                ModelOutput::Text("done".into()),
            ]),
            AgentConfig::default(),
        );
        let mut emitted = Vec::new();
        agent
            .run_task_with_events(
                dir.path(),
                "inspect",
                &Arc::new(AtomicBool::new(false)),
                &mut |event| emitted.push(event),
            )
            .unwrap();
        assert!(emitted.iter().any(
            |event| matches!(event, AgentEvent::PtcHostCallStarted { name, .. } if name == "read")
        ));
        assert!(emitted.iter().any(
            |event| matches!(event, AgentEvent::PtcHostCallCompleted { name, .. } if name == "read")
        ));
        let session = Session::resume(dir.path()).unwrap();
        assert!(
            session
                .events()
                .iter()
                .any(|record| matches!(record.event, SessionEvent::ToolCompleted { .. }))
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
                ModelOutput::Text("different".into()),
            ]),
            AgentConfig::default(),
        );
        let mut emitted = Vec::new();
        assert_eq!(
            agent
                .run_task_with_events(
                    dir.path(),
                    "inspect",
                    &Arc::new(AtomicBool::new(false)),
                    &mut |event| emitted.push(event)
                )
                .unwrap(),
            "different"
        );
        assert_eq!(
            emitted
                .iter()
                .filter(|event| matches!(event, AgentEvent::PtcStarted))
                .count(),
            2
        );
        assert!(
            emitted
                .iter()
                .filter(|event| matches!(event, AgentEvent::RepeatedActionDetected { .. }))
                .count()
                >= 2
        );
    }

    #[test]
    fn context_overflow_propagates() {
        let dir = tempfile::tempdir().unwrap();
        let config = AgentConfig {
            context_tokens: 1,
            ..AgentConfig::default()
        };
        let agent = Agent::new(ScriptedModel::new(vec![]), config);
        assert!(matches!(
            agent.run_task(dir.path(), "too large", &Arc::new(AtomicBool::new(false))),
            Err(AgentError::Context(_))
        ));
    }
}
