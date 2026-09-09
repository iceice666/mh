//! Durable append-only v5 session store.
//!
//! The journal is the only durable coordination point in mh: a detached run, an
//! inspector, a steering command, and a restarted runtime all agree because they
//! all read the same append-only event log. Task, goal, worker, and process
//! state are therefore *derived* from events rather than cached anywhere else.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::delegation::{AgentAccess, AgentState, DelegateResult};
use crate::goal::{
    Blocker, ContextCheckpoint, Decision, Failure, Finding, FinishObjection, FinishRequest,
    FinishVerdict, GoalState, GoalUpdate, TaskStatus, WorkItem,
};
use crate::identity::{AgentId, ExecutionId, IsolatedWorkspaceId, ProcessId, RevisionId, TaskId};
use crate::ptc::prelude::PreludeId;
use crate::ptc::{PtcEvent, PtcEventSink, PtcEventSinkError, PtcOutcome, PtcResult};
use crate::tools::{ResultId, ResultStore, ToolEffects};
use crate::workspace::{
    RevisionSource, WorkspaceDelta, WorkspaceError, WorkspaceTracker, WorkspaceTrackerImpl,
};

const SESSION_DIR: &str = ".mh";
const EVENT_FILE: &str = "session.jsonl";
const STATE_FILE: &str = "state.json";
const JOURNAL_LOCK_FILE: &str = "journal.lock";
const RUNNER_LOCK_FILE: &str = "runner.lock";
const CORRUPT_TAIL_PREFIX: &str = "session.jsonl.torn";
pub const SESSION_VERSION: u32 = 5;

#[derive(Debug)]
pub enum SessionError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Workspace(WorkspaceError),
    NoSession(PathBuf),
    Busy(PathBuf),
    CorruptJournal {
        path: PathBuf,
        offset: u64,
        message: String,
    },
    CommitOutcomeUnknown(String),
    State(String),
}
impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "session I/O error: {e}"),
            Self::Json(e) => write!(f, "session data error: {e}"),
            Self::Workspace(e) => write!(f, "workspace error: {e}"),
            Self::NoSession(p) => write!(f, "no session found at {}", p.display()),
            Self::Busy(p) => write!(f, "workspace runner is busy: {}", p.display()),
            Self::CorruptJournal {
                path,
                offset,
                message,
            } => write!(
                f,
                "corrupt session journal {} at byte {offset}: {message}",
                path.display()
            ),
            Self::CommitOutcomeUnknown(message) => {
                write!(f, "session journal commit outcome unknown: {message}")
            }
            Self::State(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for SessionError {}
impl From<std::io::Error> for SessionError {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<serde_json::Error> for SessionError {
    fn from(v: serde_json::Error) -> Self {
        Self::Json(v)
    }
}
impl From<WorkspaceError> for SessionError {
    fn from(v: WorkspaceError) -> Self {
        Self::Workspace(v)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionState {
    pub id: String,
    pub workspace: PathBuf,
    /// The root task of this session. A delegated worker owns its own task id
    /// but never becomes the session's active task.
    pub active_task: Option<TaskId>,
    pub current_revision: RevisionId,
    pub last_event_seq: u64,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMetadata {
    pub id: String,
    pub workspace: PathBuf,
    pub created_at_ms: u64,
    pub version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandReceipt {
    pub command_id: u64,
    pub task_id: TaskId,
    pub seq: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunnerInfo {
    pub generation: u64,
    pub instance_id: String,
    pub pid: u32,
    pub task_id: Option<TaskId>,
    pub started_at_ms: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub kind: String,
    pub ok: bool,
    pub revision: RevisionId,
    pub result_ids: Vec<ResultId>,
    pub note: Option<String>,
    pub timestamp_ms: u64,
    pub task_id: TaskId,
    pub execution_id: ExecutionId,
    /// Tool environment that produced this verification. Absent when no
    /// prelude was active, and for records written before preludes existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prelude: Option<PreludeId>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PtcSummary {
    pub ok: bool,
    pub value: Value,
    pub error: Option<String>,
    pub tool_calls: usize,
    pub duration_ms: u64,
    pub revision: RevisionId,
    pub timestamp_ms: u64,
    pub task_id: TaskId,
    pub execution_id: ExecutionId,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyEvidenceRecord {
    pub kind: String,
    pub ok: bool,
    pub mutation_epoch: u64,
    #[serde(default)]
    pub result_ids: Vec<ResultId>,
    #[serde(default)]
    pub note: Option<String>,
    pub timestamp_ms: u64,
}

/// A delegated worker as recorded in the journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentRecord {
    pub agent: AgentId,
    pub task_id: TaskId,
    pub parent: AgentId,
    pub parent_task_id: TaskId,
    pub objective: String,
    pub access: AgentAccess,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    pub state: AgentState,
    pub base_revision: RevisionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub isolated_workspace: Option<IsolatedWorkspaceId>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub context: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<DelegateResult>,
    /// Messages the parent sent that the worker has not yet consumed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inbox: Vec<String>,
}

impl AgentRecord {
    pub const fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
}

/// A background process as recorded in the journal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessRecord {
    pub process: ProcessId,
    pub task_id: TaskId,
    pub agent: AgentId,
    pub argv: Vec<String>,
    pub cwd: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    /// Best-effort liveness observation made during owner recovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid_alive: Option<bool>,
}

impl ProcessRecord {
    pub fn is_running(&self) -> bool {
        self.state == "running"
    }
}

/// Everything a context window or an inspector needs about one task.
///
/// Derived from the journal on demand: no separate durable representation can
/// drift from the events that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskView {
    pub task_id: TaskId,
    pub agent: AgentId,
    pub goal: GoalState,
    /// Zero-based index of the active context window.
    pub window: u32,
    /// Model turns already spent in the active window.
    pub turns_in_window: usize,
    pub base_revision: Option<RevisionId>,
    pub latest_user_message: Option<String>,
    pub changed_paths: Vec<PathBuf>,
    pub latest_ptc: Option<PtcSummary>,
    pub latest_failure: Option<PtcSummary>,
    pub evidence: Vec<EvidenceRecord>,
    pub prelude: Option<PreludeId>,
    /// Semantic state carried over from the previous window, when the task has
    /// rolled over at least once.
    pub checkpoint: Option<ContextCheckpoint>,
    /// Steering appended but not yet applied to a model turn.
    pub pending_steering: Vec<String>,
    /// Cancellation requested durably, e.g. by `mh cancel` against a detached
    /// run.
    pub cancel_requested: bool,
    /// Note attached to the newest durable status transition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_note: Option<String>,
    /// Sequence number of the newest event belonging to this task's window.
    pub window_start_seq: u64,
    pub legacy_evidence: Vec<LegacyEvidenceRecord>,
    /// Workers spawned by this task.
    pub agents: Vec<AgentRecord>,
    /// Processes owned by this task.
    pub processes: Vec<ProcessRecord>,
    /// Repeat-action fingerprint recorded in this window, if any.
    pub repeated_fingerprint: Option<String>,
    /// Isolated worker workspaces the task integrated or explicitly discarded.
    pub resolved_workspaces: Vec<u64>,
    #[serde(default)]
    pub unresolved_recovery_workspaces: Vec<u64>,
    /// PTC executions started by a lost runner with no durable completion.
    #[serde(default)]
    pub outcome_unknown_executions: Vec<ExecutionId>,
}

impl TaskView {
    pub fn objective(&self) -> &str {
        &self.goal.objective
    }

    pub const fn status(&self) -> TaskStatus {
        self.goal.status
    }

    /// Workers that neither completed, failed, nor were cancelled.
    pub fn unresolved_agents(&self) -> Vec<AgentId> {
        self.agents
            .iter()
            .filter(|record| !record.is_terminal())
            .map(|record| record.agent)
            .collect()
    }

    pub fn running_processes(&self) -> Vec<ProcessId> {
        self.processes
            .iter()
            .filter(|record| record.is_running())
            .map(|record| record.process)
            .collect()
    }

    /// Isolated worker deltas that were never integrated or discarded.
    ///
    /// A changed delta nobody resolved is silent lost work, so it blocks
    /// completion. Integration and explicit discard both resolve it.
    pub fn unintegrated_workspaces(&self) -> Vec<u64> {
        let mut workspaces = self
            .agents
            .iter()
            .filter_map(|record| {
                let workspace = record.isolated_workspace?;
                let changed = record
                    .result
                    .as_ref()
                    .is_some_and(|result| result.changed && result.workspace.is_some());
                (changed && !self.resolved_workspaces.contains(&workspace.0)).then_some(workspace.0)
            })
            .chain(self.unresolved_recovery_workspaces.iter().copied())
            .collect::<Vec<_>>();
        workspaces.sort_unstable();
        workspaces.dedup();
        workspaces
    }

    /// Newest evidence per kind, which is what freshness is judged on.
    pub fn latest_evidence(&self) -> Vec<&EvidenceRecord> {
        let mut newest: BTreeMap<&str, &EvidenceRecord> = BTreeMap::new();
        for record in &self.evidence {
            newest
                .entry(record.kind.as_str())
                .and_modify(|current| {
                    if record.timestamp_ms >= current.timestamp_ms {
                        *current = record;
                    }
                })
                .or_insert(record);
        }
        newest.into_values().collect()
    }

    /// Whether an evidence record still describes the current workspace and
    /// tool environment.
    pub fn evidence_is_fresh(&self, record: &EvidenceRecord) -> bool {
        record.revision == self.goal.current_revision && record.prelude == self.prelude
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    SessionInitialized {
        metadata: SessionMetadata,
        initial_revision: RevisionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        migrated_from: Option<u32>,
    },
    IdsReserved {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<TaskId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<AgentId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        execution_id: Option<ExecutionId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        process_id: Option<ProcessId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        workspace_id: Option<IsolatedWorkspaceId>,
    },
    PreludeLoaded {
        path: PathBuf,
        prelude: PreludeId,
        described: bool,
    },
    TaskStarted {
        task_id: TaskId,
        objective: String,
        base_revision: RevisionId,
        #[serde(default = "default_activate_task")]
        activate: bool,
    },
    TaskStatusChanged {
        task_id: TaskId,
        status: TaskStatus,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
    TaskCompleted {
        task_id: TaskId,
        final_revision: RevisionId,
        summary: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        unresolved: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        waived: Vec<FinishObjection>,
    },
    TaskInterrupted {
        task_id: TaskId,
    },
    TaskCancelRequested {
        command_id: u64,
        task_id: TaskId,
    },
    TaskFailed {
        task_id: TaskId,
        error: String,
    },
    GoalUpdated {
        task_id: TaskId,
        update: GoalUpdate,
    },
    FinishProposed {
        task_id: TaskId,
        request: FinishRequest,
        verdict: FinishVerdict,
    },
    UserMessage {
        task_id: TaskId,
        content: String,
    },
    AssistantMessage {
        task_id: TaskId,
        content: String,
    },
    ContextWindowStarted {
        task_id: TaskId,
        agent: AgentId,
        window: u32,
    },
    Compacted {
        task_id: TaskId,
        agent: AgentId,
        window: u32,
        summary: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        checkpoint: Option<ContextCheckpoint>,
    },
    ModelStarted {
        task_id: TaskId,
    },
    ModelCompleted {
        task_id: TaskId,
        output_kind: String,
    },
    ModelSuperseded {
        task_id: TaskId,
    },
    PtcStarted {
        task_id: TaskId,
        execution_id: ExecutionId,
        source: String,
        start_revision: RevisionId,
    },
    PtcCompleted {
        task_id: TaskId,
        execution_id: ExecutionId,
        value: Value,
        tool_calls: usize,
        duration_ms: u64,
        end_revision: RevisionId,
    },
    PtcFailed {
        task_id: TaskId,
        execution_id: ExecutionId,
        error: String,
        value: Value,
        tool_calls: usize,
        duration_ms: u64,
        end_revision: RevisionId,
    },
    AgentSpawned {
        parent_task_id: TaskId,
        parent_agent: AgentId,
        agent: AgentId,
        task_id: TaskId,
        objective: String,
        access: AgentAccess,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        profile: Option<String>,
        base_revision: RevisionId,
        #[serde(default, skip_serializing_if = "Value::is_null")]
        context: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        isolated_workspace: Option<IsolatedWorkspaceId>,
    },
    AgentStateChanged {
        agent: AgentId,
        task_id: TaskId,
        state: AgentState,
    },
    AgentCompleted {
        agent: AgentId,
        task_id: TaskId,
        result: DelegateResult,
    },
    /// Parent-to-worker message, delivered at the worker's next safe point.
    AgentMessage {
        agent: AgentId,
        task_id: TaskId,
        content: String,
    },
    AgentMessageApplied {
        agent: AgentId,
        task_id: TaskId,
        content: String,
    },
    IntegrationStarted {
        task_id: TaskId,
        execution_id: ExecutionId,
        workspace: IsolatedWorkspaceId,
        agent: AgentId,
        child_base: RevisionId,
        parent_base: RevisionId,
    },
    IntegrationCompleted {
        task_id: TaskId,
        execution_id: ExecutionId,
        agent: AgentId,
        parent_revision: RevisionId,
        changed_paths: Vec<String>,
    },
    IntegrationConflict {
        task_id: TaskId,
        execution_id: ExecutionId,
        agent: AgentId,
        child_base: RevisionId,
        parent_current: RevisionId,
        paths: Vec<String>,
    },
    /// The task deliberately abandoned an isolated worker's delta. Recorded
    /// because discarding work must be a visible decision, not an omission.
    IntegrationDiscarded {
        task_id: TaskId,
        execution_id: ExecutionId,
        workspace: IsolatedWorkspaceId,
        agent: AgentId,
        reason: String,
    },
    ProcessSpawned {
        task_id: TaskId,
        agent: AgentId,
        process: ProcessId,
        argv: Vec<String>,
        cwd: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pid: Option<u32>,
    },
    ProcessStateChanged {
        task_id: TaskId,
        process: ProcessId,
        state: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code: Option<i64>,
        /// Only meaningful for a process recovered after a runtime restart.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pid_alive: Option<bool>,
    },
    HostCallStarted {
        task_id: TaskId,
        execution_id: ExecutionId,
        call_id: u64,
        name: String,
        args_hash: u64,
    },
    ToolCompleted {
        task_id: TaskId,
        execution_id: ExecutionId,
        call_id: u64,
        name: String,
        args_hash: u64,
        effects: ToolEffects,
        outcome: crate::ptc::HostCallOutcome,
        ok: bool,
        duration_ms: u64,
        result_ids: Vec<ResultId>,
        paths: Vec<String>,
    },
    WorkspaceRevisionChanged {
        task_id: TaskId,
        execution_id: Option<ExecutionId>,
        from: RevisionId,
        to: RevisionId,
        added: Vec<String>,
        modified: Vec<String>,
        deleted: Vec<String>,
        source: RevisionSource,
        /// Set when the revision belongs to an isolated child workspace rather
        /// than the parent workspace, so parent state is never advanced to it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        isolated_workspace: Option<IsolatedWorkspaceId>,
    },
    WorkspaceDriftDetected {
        task_id: Option<TaskId>,
        expected: RevisionId,
        actual: RevisionId,
        delta: WorkspaceDelta,
    },
    EvidenceRecorded {
        evidence: EvidenceRecord,
    },
    SteeringQueued {
        command_id: u64,
        task_id: TaskId,
        content: String,
    },
    SteeringApplied {
        command_id: u64,
        task_id: TaskId,
    },
    RepeatedActionDetected {
        task_id: TaskId,
        fingerprint: String,
    },
    SessionMigrated {
        from_version: u32,
        to_version: u32,
    },
    RunnerAdmitted {
        info: RunnerInfo,
    },
    RunnerReleased {
        generation: u64,
        instance_id: String,
    },
    RecoveryApplied {
        lost_generation: u64,
        owner_lost_agents: Vec<AgentId>,
        #[serde(default)]
        outcome_unknown_executions: Vec<ExecutionId>,
        outcome_unknown_processes: Vec<ProcessId>,
        unresolved_workspaces: Vec<IsolatedWorkspaceId>,
    },
    RecoveryResolved {
        task_id: TaskId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        execution_id: Option<ExecutionId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        process_id: Option<ProcessId>,
        reason: String,
    },
    #[serde(skip)]
    Legacy {
        event_type: String,
        payload: Value,
    },
}

impl SessionEvent {
    /// Task this event belongs to, when it belongs to one.
    pub const fn task_id(&self) -> Option<TaskId> {
        match self {
            Self::TaskStarted { task_id, .. }
            | Self::TaskStatusChanged { task_id, .. }
            | Self::TaskCompleted { task_id, .. }
            | Self::TaskInterrupted { task_id }
            | Self::TaskCancelRequested { task_id, .. }
            | Self::TaskFailed { task_id, .. }
            | Self::GoalUpdated { task_id, .. }
            | Self::FinishProposed { task_id, .. }
            | Self::UserMessage { task_id, .. }
            | Self::AssistantMessage { task_id, .. }
            | Self::ContextWindowStarted { task_id, .. }
            | Self::Compacted { task_id, .. }
            | Self::ModelStarted { task_id }
            | Self::ModelCompleted { task_id, .. }
            | Self::ModelSuperseded { task_id }
            | Self::PtcStarted { task_id, .. }
            | Self::PtcCompleted { task_id, .. }
            | Self::PtcFailed { task_id, .. }
            | Self::AgentSpawned { task_id, .. }
            | Self::AgentStateChanged { task_id, .. }
            | Self::IntegrationDiscarded { task_id, .. }
            | Self::AgentCompleted { task_id, .. }
            | Self::AgentMessage { task_id, .. }
            | Self::AgentMessageApplied { task_id, .. }
            | Self::IntegrationStarted { task_id, .. }
            | Self::IntegrationCompleted { task_id, .. }
            | Self::IntegrationConflict { task_id, .. }
            | Self::ProcessSpawned { task_id, .. }
            | Self::ProcessStateChanged { task_id, .. }
            | Self::HostCallStarted { task_id, .. }
            | Self::ToolCompleted { task_id, .. }
            | Self::WorkspaceRevisionChanged { task_id, .. }
            | Self::SteeringQueued { task_id, .. }
            | Self::SteeringApplied { task_id, .. }
            | Self::RepeatedActionDetected { task_id, .. }
            | Self::RecoveryResolved { task_id, .. } => Some(*task_id),
            Self::EvidenceRecorded { evidence } => Some(evidence.task_id),
            Self::WorkspaceDriftDetected { task_id, .. } => *task_id,
            Self::SessionInitialized { .. }
            | Self::IdsReserved { .. }
            | Self::PreludeLoaded { .. }
            | Self::SessionMigrated { .. }
            | Self::RunnerAdmitted { .. }
            | Self::RunnerReleased { .. }
            | Self::RecoveryApplied { .. }
            | Self::Legacy { .. } => None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventRecord {
    pub seq: u64,
    pub timestamp_ms: u64,
    #[serde(flatten)]
    pub event: SessionEvent,
}
#[derive(Debug)]
struct Journal {
    root: PathBuf,
    state: SessionState,
    events: Vec<EventRecord>,
    writable: bool,
    poisoned: bool,
    warnings: Vec<String>,
}

#[derive(Debug)]
pub struct WorkspaceRunnerLock {
    workspace: PathBuf,
    file: File,
}

impl WorkspaceRunnerLock {
    pub fn acquire(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = canonical_workspace(workspace.as_ref())?;
        let root = workspace.join(SESSION_DIR);
        std::fs::create_dir_all(&root)?;
        let path = root.join(RUNNER_LOCK_FILE);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;
        match file.try_lock() {
            Ok(()) => Ok(Self { workspace, file }),
            Err(TryLockError::WouldBlock) => Err(SessionError::Busy(path)),
            Err(TryLockError::Error(error)) => Err(error.into()),
        }
    }
}

#[derive(Debug)]
pub struct RunnerGuard {
    file: File,
    session: Session,
    info: RunnerInfo,
}

impl RunnerGuard {
    pub fn info(&self) -> &RunnerInfo {
        &self.info
    }
    pub fn resolve_recovery(
        &self,
        task_id: TaskId,
        execution_id: Option<ExecutionId>,
        process_id: Option<ProcessId>,
        reason: String,
    ) -> Result<EventRecord, SessionError> {
        self.session
            .resolve_recovery_owned(task_id, execution_id, process_id, reason)
    }
}

impl Drop for RunnerGuard {
    fn drop(&mut self) {
        let _ = self.session.append_shared(SessionEvent::RunnerReleased {
            generation: self.info.generation,
            instance_id: self.info.instance_id.clone(),
        });
        let _ = self.file.unlock();
    }
}
impl Clone for Session {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            tracker: self.tracker.clone(),
            journal: self.journal.clone(),
        }
    }
}
pub struct Session {
    root: PathBuf,
    tracker: Option<Arc<dyn WorkspaceTracker>>,
    journal: Arc<Mutex<Journal>>,
}
impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").field("root", &self.root).finish()
    }
}

#[derive(Debug, Clone)]
pub struct SessionPtcEventSink {
    session: Session,
    isolated_workspace: Option<IsolatedWorkspaceId>,
}

impl Session {
    pub fn open(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = canonical_workspace(workspace.as_ref())?;
        let root = workspace.join(SESSION_DIR);
        std::fs::create_dir_all(&root)?;
        let tracker: Arc<dyn WorkspaceTracker> = Arc::new(WorkspaceTrackerImpl::open(&workspace)?);
        let revision = tracker.current_revision()?.id;
        migrate_legacy_session(&root, &workspace, &revision)?;
        let session = Self::empty(
            root,
            workspace.clone(),
            Some(tracker),
            true,
            revision.clone(),
        );
        session.transact(|events, _state| {
            if !events.is_empty() {
                return Ok(None);
            }
            let now = now_ms();
            let metadata = SessionMetadata {
                id: new_instance_id(now),
                workspace,
                created_at_ms: now,
                version: SESSION_VERSION,
            };
            Ok(Some(SessionEvent::SessionInitialized {
                metadata,
                initial_revision: revision,
                migrated_from: None,
            }))
        })?;
        session.ensure_current_format()?;
        session.refresh_cache();
        Ok(session)
    }

    pub fn resume(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = canonical_workspace(workspace.as_ref())?;
        let root = workspace.join(SESSION_DIR);
        if !root.join(EVENT_FILE).exists() && !root.join(STATE_FILE).exists() {
            return Err(SessionError::NoSession(root));
        }
        let tracker: Arc<dyn WorkspaceTracker> = Arc::new(WorkspaceTrackerImpl::open(&workspace)?);
        let revision = tracker.current_revision()?.id;
        migrate_legacy_session(&root, &workspace, &revision)?;
        let session = Self::empty(root, workspace, Some(tracker), true, revision);
        session.reload(false)?;
        session.ensure_current_format()?;
        session.refresh_cache();
        Ok(session)
    }

    /// Opens an existing current-format session for queued task registration
    /// without touching the workspace tracker. First initialization briefly
    /// takes the canonical runner lock because it must establish the initial
    /// workspace revision and session identity.
    pub fn registration(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = canonical_workspace(workspace.as_ref())?;
        let event_path = workspace.join(SESSION_DIR).join(EVENT_FILE);
        if event_path.exists() {
            return Self::command(workspace);
        }
        let _initialization = WorkspaceRunnerLock::acquire(&workspace)?;
        if event_path.exists() {
            Self::command(workspace)
        } else {
            Self::open(workspace)
        }
    }

    /// Opens a session as a pure journal snapshot. It neither creates `.mh`,
    /// opens the mutable workspace revision cache, repairs a torn tail, nor
    /// refreshes `state.json`.
    pub fn inspect(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = canonical_workspace(workspace.as_ref())?;
        let root = workspace.join(SESSION_DIR);
        if !root.join(EVENT_FILE).exists() && !root.join(STATE_FILE).exists() {
            return Err(SessionError::NoSession(root));
        }
        if !journal_has_current_header(&root.join(EVENT_FILE))? {
            return Err(SessionError::State(
                "legacy session requires `mh resume` to perform the locked format upgrade"
                    .to_string(),
            ));
        }
        let seed_revision = state_revision(&root).unwrap_or_else(|| RevisionId(String::new()));
        let session = Self::empty(root, workspace, None, false, seed_revision);
        session.reload(false)?;
        Ok(session)
    }

    /// Opens only the current journal format for a command transaction. This
    /// path never opens the mutable workspace tracker or performs migration.
    pub fn command(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = canonical_workspace(workspace.as_ref())?;
        let root = workspace.join(SESSION_DIR);
        if !root.join(EVENT_FILE).exists() {
            return Err(SessionError::NoSession(root));
        }
        if !journal_has_current_header(&root.join(EVENT_FILE))? {
            return Err(SessionError::State(
                "legacy session must be upgraded with `mh resume` before commands are accepted"
                    .to_string(),
            ));
        }
        let seed_revision = state_revision(&root).unwrap_or_else(|| RevisionId(String::new()));
        let session = Self::empty(root, workspace, None, true, seed_revision);
        session.reload(false)?;
        session.ensure_current_format()?;
        Ok(session)
    }

    fn empty(
        root: PathBuf,
        workspace: PathBuf,
        tracker: Option<Arc<dyn WorkspaceTracker>>,
        writable: bool,
        revision: RevisionId,
    ) -> Self {
        let now = now_ms();
        Self {
            root: root.clone(),
            tracker,
            journal: Arc::new(Mutex::new(Journal {
                root,
                state: SessionState {
                    id: String::new(),
                    workspace,
                    active_task: None,
                    current_revision: revision,
                    last_event_seq: 0,
                    created_at_ms: now,
                    updated_at_ms: now,
                },
                events: Vec::new(),
                writable,
                poisoned: false,
                warnings: Vec::new(),
            })),
        }
    }

    fn ensure_current_format(&self) -> Result<(), SessionError> {
        let metadata = session_metadata(&self.events());
        if let Some(metadata) = metadata {
            if metadata.version != SESSION_VERSION {
                return Err(SessionError::State(format!(
                    "unsupported session format {}; expected {SESSION_VERSION}",
                    metadata.version
                )));
            }
            return Ok(());
        }
        Err(SessionError::State(
            "legacy session lacks replayable session metadata; run the previous mh release once before upgrading"
                .to_string(),
        ))
    }

    fn reload(&self, repair_torn_tail: bool) -> Result<(), SessionError> {
        let mut journal = self.journal.lock().expect("session journal poisoned");
        let replay = read_events(&journal.root.join(EVENT_FILE), repair_torn_tail)?;
        journal.events = replay.events;
        journal.warnings = replay.warning.into_iter().collect();
        let events = journal.events.clone();
        rebuild_from_journal(&mut journal.state, &events)?;
        Ok(())
    }

    fn refresh_cache(&self) {
        let mut journal = self.journal.lock().expect("session journal poisoned");
        if let Err(error) = write_state(&journal) {
            journal
                .warnings
                .push(format!("state cache refresh failed: {error}"));
        }
    }
    pub fn append(&mut self, event: SessionEvent) -> Result<EventRecord, SessionError> {
        self.append_shared(event)
    }

    fn append_shared(&self, event: SessionEvent) -> Result<EventRecord, SessionError> {
        self.transact(|_, _| Ok(Some(event)))?
            .ok_or_else(|| SessionError::State("journal transaction produced no event".to_string()))
    }

    pub fn append_shared_event(&self, event: SessionEvent) -> Result<EventRecord, SessionError> {
        self.append_shared(event)
    }

    pub fn begin_task(&self, objective: &str) -> Result<TaskId, SessionError> {
        self.begin_task_with_activation(objective, true)
    }

    pub fn register_queued_task(&self, objective: &str) -> Result<TaskId, SessionError> {
        self.begin_task_with_activation(objective, false)
    }

    fn begin_task_with_activation(
        &self,
        objective: &str,
        activate: bool,
    ) -> Result<TaskId, SessionError> {
        let objective = objective.to_string();
        let record = self.transact(|events, state| {
            let task_id = next_id(events.iter().filter_map(task_id_in_event))?;
            Ok(Some(SessionEvent::TaskStarted {
                task_id: TaskId(task_id),
                objective,
                base_revision: state.current_revision.clone(),
                activate,
            }))
        })?;
        match record.map(|record| record.event) {
            Some(SessionEvent::TaskStarted { task_id, .. }) => Ok(task_id),
            _ => Err(SessionError::State(
                "task reservation did not commit".to_string(),
            )),
        }
    }

    pub fn reserve_agent(&self) -> Result<(AgentId, TaskId), SessionError> {
        let record = self.transact(|events, _| {
            let agent = AgentId(next_id(events.iter().filter_map(agent_id_in_event))?);
            let task_id = TaskId(next_id(events.iter().filter_map(task_id_in_event))?);
            Ok(Some(SessionEvent::IdsReserved {
                task_id: Some(task_id),
                agent_id: Some(agent),
                execution_id: None,
                process_id: None,
                workspace_id: None,
            }))
        })?;
        match record.map(|record| record.event) {
            Some(SessionEvent::IdsReserved {
                task_id: Some(task_id),
                agent_id: Some(agent),
                ..
            }) => Ok((agent, task_id)),
            _ => Err(SessionError::State(
                "agent reservation did not commit".to_string(),
            )),
        }
    }

    pub fn reserve_process(&self) -> Result<ProcessId, SessionError> {
        let record = self.transact(|events, _| {
            let process_id = ProcessId(next_id(events.iter().filter_map(process_id_in_event))?);
            Ok(Some(SessionEvent::IdsReserved {
                task_id: None,
                agent_id: None,
                execution_id: None,
                process_id: Some(process_id),
                workspace_id: None,
            }))
        })?;
        match record.map(|record| record.event) {
            Some(SessionEvent::IdsReserved {
                process_id: Some(id),
                ..
            }) => Ok(id),
            _ => Err(SessionError::State(
                "process reservation did not commit".to_string(),
            )),
        }
    }

    pub fn reserve_execution(&self) -> Result<ExecutionId, SessionError> {
        let record = self.transact(|events, _| {
            let execution_id =
                ExecutionId(next_id(events.iter().filter_map(execution_id_in_event))?);
            Ok(Some(SessionEvent::IdsReserved {
                task_id: None,
                agent_id: None,
                execution_id: Some(execution_id),
                process_id: None,
                workspace_id: None,
            }))
        })?;
        match record.map(|record| record.event) {
            Some(SessionEvent::IdsReserved {
                execution_id: Some(id),
                ..
            }) => Ok(id),
            _ => Err(SessionError::State(
                "execution reservation did not commit".to_string(),
            )),
        }
    }

    pub fn reserve_workspace(&self) -> Result<IsolatedWorkspaceId, SessionError> {
        let record = self.transact(|events, _| {
            let workspace_id =
                IsolatedWorkspaceId(next_id(events.iter().filter_map(workspace_id_in_event))?);
            Ok(Some(SessionEvent::IdsReserved {
                task_id: None,
                agent_id: None,
                execution_id: None,
                process_id: None,
                workspace_id: Some(workspace_id),
            }))
        })?;
        match record.map(|record| record.event) {
            Some(SessionEvent::IdsReserved {
                workspace_id: Some(id),
                ..
            }) => Ok(id),
            _ => Err(SessionError::State(
                "workspace reservation did not commit".to_string(),
            )),
        }
    }
    pub fn queue_steering(
        &self,
        task_id: Option<TaskId>,
        content: String,
    ) -> Result<CommandReceipt, SessionError> {
        let record = self.transact(|events, state| {
            let task_id = resolve_command_task(events, state, task_id, "steer")?;
            let command_id = next_id(events.iter().filter_map(command_id_in_event))?;
            Ok(Some(SessionEvent::SteeringQueued {
                command_id,
                task_id,
                content,
            }))
        })?;
        match record {
            Some(EventRecord {
                seq,
                event:
                    SessionEvent::SteeringQueued {
                        command_id,
                        task_id,
                        ..
                    },
                ..
            }) => Ok(CommandReceipt {
                command_id,
                task_id,
                seq,
                cache_warning: self.cache_warning_for(seq),
            }),
            _ => Err(SessionError::State(
                "steering command did not commit".to_string(),
            )),
        }
    }

    pub fn request_cancel(&self, task_id: Option<TaskId>) -> Result<CommandReceipt, SessionError> {
        let record = self.transact(|events, state| {
            let task_id = resolve_command_task(events, state, task_id, "cancel")?;
            let command_id = next_id(events.iter().filter_map(command_id_in_event))?;
            Ok(Some(SessionEvent::TaskCancelRequested {
                task_id,
                command_id,
            }))
        })?;
        match record {
            Some(EventRecord {
                seq,
                event:
                    SessionEvent::TaskCancelRequested {
                        task_id,
                        command_id,
                    },
                ..
            }) => Ok(CommandReceipt {
                command_id,
                task_id,
                seq,
                cache_warning: self.cache_warning_for(seq),
            }),
            _ => Err(SessionError::State(
                "cancel command did not commit".to_string(),
            )),
        }
    }
    fn resolve_recovery_owned(
        &self,
        task_id: TaskId,
        execution_id: Option<ExecutionId>,
        process_id: Option<ProcessId>,
        reason: String,
    ) -> Result<EventRecord, SessionError> {
        let targets = usize::from(execution_id.is_some()) + usize::from(process_id.is_some());
        if targets != 1 {
            return Err(SessionError::State(
                "recovery resolution requires exactly one execution or process".to_string(),
            ));
        }
        if reason.trim().is_empty() {
            return Err(SessionError::State(
                "recovery resolution requires a reason".to_string(),
            ));
        }
        self.transact(|events, state| {
            let view = derive_task(state, events, task_id);
            let unresolved_execution =
                execution_id.is_some_and(|id| view.outcome_unknown_executions.contains(&id));
            let unresolved_process = process_id.is_some_and(|id| {
                view.processes
                    .iter()
                    .any(|process| process.process == id && process.state == "outcome-unknown")
            });
            if !(unresolved_execution || unresolved_process) {
                return Err(SessionError::State(format!(
                    "task {} has no matching unresolved recovery resource",
                    task_id.0
                )));
            }
            Ok(Some(SessionEvent::RecoveryResolved {
                task_id,
                execution_id,
                process_id,
                reason,
            }))
        })?
        .ok_or_else(|| SessionError::State("recovery resolution did not commit".to_string()))
    }

    pub fn apply_steering_command(
        &self,
        task_id: TaskId,
        command_id: u64,
    ) -> Result<(), SessionError> {
        self.transact(|events, _| {
            let queued = events.iter().any(|record| {
                matches!(
                    record.event,
                    SessionEvent::SteeringQueued {
                        command_id: id,
                        task_id: task,
                        ..
                    } if id == command_id && task == task_id
                )
            });
            let applied = events.iter().any(|record| {
                matches!(
                    record.event,
                    SessionEvent::SteeringApplied {
                        command_id: id,
                        task_id: task,
                    } if id == command_id && task == task_id
                )
            });
            if !queued || applied {
                return Ok(None);
            }
            Ok(Some(SessionEvent::SteeringApplied {
                command_id,
                task_id,
            }))
        })?;
        Ok(())
    }

    pub fn pending_steering_commands(&self, task_id: TaskId) -> Vec<(u64, String)> {
        let mut queued = Vec::new();
        for record in self.events() {
            match record.event {
                SessionEvent::SteeringQueued {
                    command_id,
                    task_id: id,
                    content,
                } if id == task_id => queued.push((command_id, content)),
                SessionEvent::SteeringApplied {
                    command_id,
                    task_id: id,
                } if id == task_id => {
                    if let Some(index) = queued
                        .iter()
                        .position(|(queued_id, _)| *queued_id == command_id)
                    {
                        queued.remove(index);
                    }
                }
                _ => {}
            }
        }
        queued
    }

    pub fn warnings(&self) -> Vec<String> {
        self.journal
            .lock()
            .expect("session journal poisoned")
            .warnings
            .clone()
    }

    fn cache_warning_for(&self, seq: u64) -> Option<String> {
        let suffix = format!("after seq {seq}");
        self.warnings()
            .into_iter()
            .find(|warning| warning.contains(&suffix))
    }
    pub fn evidence_for_task(&self, task_id: TaskId) -> Vec<EvidenceRecord> {
        self.events()
            .into_iter()
            .filter_map(|record| match record.event {
                SessionEvent::EvidenceRecorded { evidence } if evidence.task_id == task_id => {
                    Some(evidence)
                }
                _ => None,
            })
            .collect()
    }

    /// Records explicit, validated task completion.
    pub fn complete_task(
        &self,
        task_id: TaskId,
        summary: String,
        unresolved: Vec<String>,
        waived: Vec<FinishObjection>,
    ) -> Result<(), SessionError> {
        self.transact(|events, state| {
            let view = derive_task(state, events, task_id);
            if !view.status().is_active() {
                return Err(SessionError::State(format!(
                    "task {} already {}",
                    task_id.0,
                    view.status().label()
                )));
            }
            if view.cancel_requested {
                return Err(SessionError::State(format!(
                    "task {} has a durable cancellation request",
                    task_id.0
                )));
            }
            if !view.pending_steering.is_empty() {
                return Err(SessionError::State(format!(
                    "task {} has unapplied steering",
                    task_id.0
                )));
            }
            Ok(Some(SessionEvent::TaskCompleted {
                task_id,
                final_revision: state.current_revision.clone(),
                summary,
                unresolved,
                waived,
            }))
        })?;
        Ok(())
    }

    pub fn interrupt_task(&self, task_id: TaskId) -> Result<(), SessionError> {
        self.transact(|events, state| {
            let view = derive_task(state, events, task_id);
            if !view.status().is_active() {
                return Err(SessionError::State(format!(
                    "task {} already {}",
                    task_id.0,
                    view.status().label()
                )));
            }
            Ok(Some(SessionEvent::TaskInterrupted { task_id }))
        })?;
        Ok(())
    }

    pub fn set_status(
        &self,
        task_id: TaskId,
        status: TaskStatus,
        note: Option<String>,
    ) -> Result<(), SessionError> {
        self.transact(|events, state| {
            let view = derive_task(state, events, task_id);
            if !view.status().is_active() {
                return Err(SessionError::State(format!(
                    "task {} already {}",
                    task_id.0,
                    view.status().label()
                )));
            }
            Ok(Some(SessionEvent::TaskStatusChanged {
                task_id,
                status,
                note,
            }))
        })?;
        Ok(())
    }

    pub fn append_ptc_result(&self, result: &PtcResult) -> Result<(), SessionError> {
        let event = match &result.outcome {
            PtcOutcome::Completed => SessionEvent::PtcCompleted {
                task_id: result.task_id,
                execution_id: result.execution_id,
                value: result.value.clone(),
                tool_calls: result.tool_calls,
                duration_ms: result.duration_ms,
                end_revision: result.end_revision.clone(),
            },
            PtcOutcome::Interrupted => SessionEvent::PtcFailed {
                task_id: result.task_id,
                execution_id: result.execution_id,
                error: "cancelled".to_string(),
                value: result.value.clone(),
                tool_calls: result.tool_calls,
                duration_ms: result.duration_ms,
                end_revision: result.end_revision.clone(),
            },
            PtcOutcome::BudgetExceeded(kind) => SessionEvent::PtcFailed {
                task_id: result.task_id,
                execution_id: result.execution_id,
                error: format!("budget exceeded: {kind}"),
                value: result.value.clone(),
                tool_calls: result.tool_calls,
                duration_ms: result.duration_ms,
                end_revision: result.end_revision.clone(),
            },
            PtcOutcome::Failed(error) => SessionEvent::PtcFailed {
                task_id: result.task_id,
                execution_id: result.execution_id,
                error: error.clone(),
                value: result.value.clone(),
                tool_calls: result.tool_calls,
                duration_ms: result.duration_ms,
                end_revision: result.end_revision.clone(),
            },
        };
        self.append_shared(event)?;
        Ok(())
    }
    pub fn reconcile_workspace(&self) -> Result<Option<WorkspaceDelta>, SessionError> {
        let expected = self.state().current_revision;
        let actual = self.tracker().current_revision()?.id;
        if expected == actual {
            return Ok(None);
        }
        let delta = self.tracker().delta(&expected, &actual)?;
        self.append_shared(SessionEvent::WorkspaceDriftDetected {
            task_id: self.state().active_task,
            expected,
            actual,
            delta: delta.clone(),
        })?;
        Ok(Some(delta))
    }

    pub fn state(&self) -> SessionState {
        self.journal
            .lock()
            .expect("session journal poisoned")
            .state
            .clone()
    }

    pub fn events(&self) -> Vec<EventRecord> {
        self.journal
            .lock()
            .expect("session journal poisoned")
            .events
            .clone()
    }

    /// Re-reads the journal from disk, picking up events appended by another
    /// process. `mh steer` and `mh cancel` write into the same log a detached
    /// run is executing from.
    pub fn refresh(&self) -> Result<(), SessionError> {
        self.reload(false)
    }

    /// The root task of this session: active if one is running, otherwise the
    /// most recently started one.
    pub fn root_task(&self) -> Option<TaskId> {
        let state = self.state();
        state.active_task.or_else(|| {
            self.events()
                .iter()
                .rev()
                .find_map(|record| match record.event {
                    SessionEvent::TaskStarted { task_id, .. } => Some(task_id),
                    _ => None,
                })
        })
    }

    pub fn recover_previous_owner(&self, current_generation: u64) -> Result<(), SessionError> {
        loop {
            let recovered = self.transact(|events, _| {
                let previous = events.iter().enumerate().find_map(|(index, record)| {
                    let SessionEvent::RunnerAdmitted { info } = &record.event else {
                        return None;
                    };
                    if info.generation >= current_generation {
                        return None;
                    }
                    let released = events.iter().any(|record| {
                        matches!(
                            &record.event,
                            SessionEvent::RunnerReleased {
                                generation,
                                instance_id,
                            } if *generation == info.generation && instance_id == &info.instance_id
                        )
                    });
                    let already_recovered = events.iter().any(|record| {
                        matches!(
                            record.event,
                            SessionEvent::RecoveryApplied { lost_generation, .. }
                                if lost_generation == info.generation
                        )
                    });
                    (!released && !already_recovered).then_some((index, info.clone()))
                });
                let Some((previous_index, previous)) = previous else {
                    return Ok(None);
                };
                let owned_end = events[previous_index + 1..]
                    .iter()
                    .position(|record| matches!(record.event, SessionEvent::RunnerAdmitted { .. }))
                    .map_or(events.len(), |offset| previous_index + 1 + offset);
                let owned_events = &events[previous_index + 1..owned_end];
                let agents = derive_agents(owned_events);
                let owner_lost_agents = agents
                    .values()
                    .filter(|agent| !agent.is_terminal())
                    .map(|agent| agent.agent)
                    .collect();
                let outcome_unknown_processes = derive_processes(owned_events)
                    .into_values()
                    .filter(ProcessRecord::is_running)
                    .map(|process| process.process)
                    .collect();
                let outcome_unknown_executions = in_flight_executions(owned_events)
                    .into_values()
                    .map(|(_, execution)| execution)
                    .collect();
                let unresolved_workspaces = agents
                    .values()
                    .filter(|agent| !agent.is_terminal())
                    .filter_map(|agent| agent.isolated_workspace)
                    .collect();
                Ok(Some(SessionEvent::RecoveryApplied {
                    lost_generation: previous.generation,
                    owner_lost_agents,
                    outcome_unknown_executions,
                    outcome_unknown_processes,
                    unresolved_workspaces,
                }))
            })?;
            if recovered.is_none() {
                return Ok(());
            }
        }
    }

    /// Every task in the journal, root tasks first, in id order.
    pub fn tasks(&self) -> Vec<TaskView> {
        let mut ids: Vec<TaskId> = self
            .events()
            .iter()
            .filter_map(|record| match &record.event {
                SessionEvent::TaskStarted { task_id, .. } => Some(*task_id),
                SessionEvent::AgentSpawned { task_id, .. } => Some(*task_id),
                _ => None,
            })
            .collect();
        ids.sort();
        ids.dedup();
        ids.into_iter().map(|id| self.task_view(id)).collect()
    }

    /// Durable view of one task, derived from the journal.
    pub fn task_view(&self, task_id: TaskId) -> TaskView {
        derive_task(&self.state(), &self.events(), task_id)
    }

    pub fn agents(&self) -> Vec<AgentRecord> {
        derive_agents(&self.events())
            .into_values()
            .collect::<Vec<_>>()
    }

    pub fn agent(&self, agent: AgentId) -> Option<AgentRecord> {
        derive_agents(&self.events()).remove(&agent.0)
    }

    pub fn processes(&self) -> Vec<ProcessRecord> {
        derive_processes(&self.events()).into_values().collect()
    }

    pub fn ptc_event_sink(&self) -> SessionPtcEventSink {
        SessionPtcEventSink {
            session: self.clone(),
            isolated_workspace: None,
        }
    }

    pub fn child_ptc_event_sink(
        &self,
        isolated_workspace: Option<IsolatedWorkspaceId>,
    ) -> SessionPtcEventSink {
        SessionPtcEventSink {
            session: self.clone(),
            isolated_workspace,
        }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn workspace_root(&self) -> PathBuf {
        self.state().workspace
    }
    pub fn tracker(&self) -> Arc<dyn WorkspaceTracker> {
        self.tracker
            .as_ref()
            .expect("execution session has workspace tracker")
            .clone()
    }
    pub fn result_store(&self) -> Result<ResultStore, SessionError> {
        ResultStore::persistent(self.root.join("blobs")).map_err(SessionError::Io)
    }

    pub fn acquire_runner(&self, task_id: Option<TaskId>) -> Result<RunnerGuard, SessionError> {
        let lock = WorkspaceRunnerLock::acquire(self.workspace_root())?;
        self.admit_runner(lock, task_id)
    }

    pub fn admit_runner(
        &self,
        lock: WorkspaceRunnerLock,
        task_id: Option<TaskId>,
    ) -> Result<RunnerGuard, SessionError> {
        if lock.workspace != self.workspace_root() {
            return Err(SessionError::State(
                "runner lock belongs to a different workspace".to_string(),
            ));
        }
        let now = now_ms();
        let record = self.transact(|events, state| {
            if let Some(task_id) = task_id {
                let exists = events.iter().any(|record| {
                    matches!(record.event, SessionEvent::TaskStarted { task_id: id, .. } if id == task_id)
                });
                if !exists {
                    return Err(SessionError::State(format!("no root task {}", task_id.0)));
                }
                let view = derive_task(state, events, task_id);
                if !view.agent.is_root() {
                    return Err(SessionError::State(format!(
                        "task {} is delegated child agent {}; only root tasks can be resumed",
                        task_id.0, view.agent.0
                    )));
                }
                if !view.status().is_active() {
                    return Err(SessionError::State(format!(
                        "task {} already {}",
                        task_id.0,
                        view.status().label()
                    )));
                }
            }
            let generation = events
                .iter()
                .filter_map(|record| match &record.event {
                    SessionEvent::RunnerAdmitted { info } => Some(info.generation),
                    _ => None,
                })
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| SessionError::State("runner generation exhausted".to_string()))?;
            Ok(Some(SessionEvent::RunnerAdmitted {
                info: RunnerInfo {
                    generation,
                    instance_id: new_instance_id(now),
                    pid: std::process::id(),
                    task_id,
                    started_at_ms: now,
                },
            }))
        })?;
        let Some(EventRecord {
            event: SessionEvent::RunnerAdmitted { info },
            ..
        }) = record
        else {
            return Err(SessionError::State(
                "runner admission did not commit".to_string(),
            ));
        };
        Ok(RunnerGuard {
            file: lock.file,
            session: self.clone(),
            info,
        })
    }

    fn transact<F>(&self, build: F) -> Result<Option<EventRecord>, SessionError>
    where
        F: FnOnce(&[EventRecord], &SessionState) -> Result<Option<SessionEvent>, SessionError>,
    {
        let mut journal = self.journal.lock().expect("session journal poisoned");
        if !journal.writable {
            return Err(SessionError::State("session is read-only".to_string()));
        }
        if journal.poisoned {
            return Err(SessionError::CommitOutcomeUnknown(
                "this session handle must be reopened before writing".to_string(),
            ));
        }
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(journal.root.join(JOURNAL_LOCK_FILE))?;
        lock.lock()?;
        let replay = read_events(&journal.root.join(EVENT_FILE), true)?;
        journal.events = replay.events;
        journal.warnings = replay.warning.into_iter().collect();
        let events = journal.events.clone();
        rebuild_from_journal(&mut journal.state, &events)?;
        let Some(event) = build(&journal.events, &journal.state)? else {
            return Ok(None);
        };
        let record = append_record(&mut journal, event)?;
        Ok(Some(record))
    }
}

impl PtcEventSink for SessionPtcEventSink {
    fn emit(&self, event: PtcEvent) -> Result<(), PtcEventSinkError> {
        let event = match event {
            PtcEvent::HostCallStarted {
                task_id,
                execution_id,
                call_id,
                name,
                args_hash,
            } => SessionEvent::HostCallStarted {
                task_id,
                execution_id,
                call_id,
                name,
                args_hash,
            },
            PtcEvent::HostCallCompleted {
                task_id,
                execution_id,
                call_id,
                name,
                args_hash,
                effects,
                outcome,
                ok,
                duration_ms,
                result_ids,
                paths,
            } => SessionEvent::ToolCompleted {
                task_id,
                execution_id,
                call_id,
                name,
                args_hash,
                effects,
                outcome,
                ok,
                duration_ms,
                result_ids,
                paths,
            },
            PtcEvent::WorkspaceRevisionChanged {
                task_id,
                execution_id,
                from,
                to,
                added,
                modified,
                deleted,
                source,
            } => SessionEvent::WorkspaceRevisionChanged {
                task_id,
                execution_id: Some(execution_id),
                from,
                to,
                added,
                modified,
                deleted,
                source,
                isolated_workspace: self.isolated_workspace,
            },
            PtcEvent::EvidenceRecorded { evidence } => SessionEvent::EvidenceRecorded { evidence },
        };
        self.session
            .append_shared_event(event)
            .map(|_| ())
            .map_err(|e| PtcEventSinkError::new(e.to_string()))
    }
}

fn derive_agents(events: &[EventRecord]) -> BTreeMap<u64, AgentRecord> {
    let mut agents: BTreeMap<u64, AgentRecord> = BTreeMap::new();
    for record in events {
        match &record.event {
            SessionEvent::AgentSpawned {
                parent_task_id,
                parent_agent,
                agent,
                task_id,
                objective,
                access,
                profile,
                base_revision,
                context,
                isolated_workspace,
            } => {
                agents.insert(
                    agent.0,
                    AgentRecord {
                        agent: *agent,
                        task_id: *task_id,
                        parent: *parent_agent,
                        parent_task_id: *parent_task_id,
                        objective: objective.clone(),
                        access: *access,
                        profile: profile.clone(),
                        state: AgentState::Queued,
                        base_revision: base_revision.clone(),
                        isolated_workspace: *isolated_workspace,
                        context: context.clone(),
                        result: None,
                        inbox: Vec::new(),
                    },
                );
            }
            SessionEvent::AgentStateChanged { agent, state, .. } => {
                if let Some(entry) = agents.get_mut(&agent.0) {
                    entry.state = *state;
                }
            }
            SessionEvent::AgentCompleted {
                agent,
                result: agent_result,
                ..
            } => {
                if let Some(entry) = agents.get_mut(&agent.0) {
                    entry.state = if agent_result.ok {
                        AgentState::Completed
                    } else {
                        AgentState::Failed
                    };
                    entry.result = Some(agent_result.clone());
                }
            }
            SessionEvent::AgentMessage { agent, content, .. } => {
                if let Some(entry) = agents.get_mut(&agent.0) {
                    entry.inbox.push(content.clone());
                }
            }
            SessionEvent::AgentMessageApplied { agent, content, .. } => {
                if let Some(entry) = agents.get_mut(&agent.0)
                    && let Some(index) = entry.inbox.iter().position(|queued| queued == content)
                {
                    entry.inbox.remove(index);
                }
            }
            SessionEvent::RecoveryApplied {
                owner_lost_agents, ..
            } => {
                for agent in owner_lost_agents {
                    if let Some(entry) = agents.get_mut(&agent.0)
                        && !entry.is_terminal()
                    {
                        entry.state = AgentState::Failed;
                    }
                }
            }
            _ => {}
        }
    }
    agents
}

fn derive_processes(events: &[EventRecord]) -> BTreeMap<u64, ProcessRecord> {
    let mut processes: BTreeMap<u64, ProcessRecord> = BTreeMap::new();
    for record in events {
        match &record.event {
            SessionEvent::ProcessSpawned {
                task_id,
                agent,
                process,
                argv,
                cwd,
                label,
                pid,
            } => {
                processes.insert(
                    process.0,
                    ProcessRecord {
                        process: *process,
                        task_id: *task_id,
                        agent: *agent,
                        argv: argv.clone(),
                        cwd: cwd.clone(),
                        label: label.clone(),
                        pid: *pid,
                        state: "running".to_string(),
                        exit_code: None,
                        pid_alive: None,
                    },
                );
            }
            SessionEvent::ProcessStateChanged {
                process,
                state,
                exit_code,
                pid_alive,
                ..
            } => {
                if let Some(entry) = processes.get_mut(&process.0) {
                    if entry.state != "outcome-unknown" && entry.state != "recovery-resolved" {
                        entry.state.clone_from(state);
                    }
                    if exit_code.is_some() {
                        entry.exit_code = *exit_code;
                    }
                    if pid_alive.is_some() {
                        entry.pid_alive = *pid_alive;
                    }
                }
            }
            SessionEvent::RecoveryApplied {
                outcome_unknown_processes,
                ..
            } => {
                for process in outcome_unknown_processes {
                    if let Some(entry) = processes.get_mut(&process.0)
                        && entry.is_running()
                    {
                        entry.state = "outcome-unknown".to_string();
                    }
                }
            }
            SessionEvent::RecoveryResolved {
                process_id: Some(process),
                ..
            } => {
                if let Some(entry) = processes.get_mut(&process.0)
                    && entry.state == "outcome-unknown"
                {
                    entry.state = "recovery-resolved".to_string();
                }
            }
            _ => {}
        }
    }
    processes
}
fn in_flight_executions(events: &[EventRecord]) -> BTreeMap<u64, (TaskId, ExecutionId)> {
    let mut executions = BTreeMap::new();
    for record in events {
        match record.event {
            SessionEvent::PtcStarted {
                task_id,
                execution_id,
                ..
            } => {
                executions.insert(execution_id.0, (task_id, execution_id));
            }
            SessionEvent::PtcCompleted { execution_id, .. }
            | SessionEvent::PtcFailed { execution_id, .. } => {
                executions.remove(&execution_id.0);
            }
            _ => {}
        }
    }
    executions
}

/// Rebuilds one task's durable state from the journal.
///
/// Window scoping matters here: after a rollover, only the compacted
/// checkpoint and the events that follow it belong to the live window, so the
/// derivation records where the window began instead of replaying everything.
fn derive_task(state: &SessionState, events: &[EventRecord], task_id: TaskId) -> TaskView {
    let mut goal = GoalState::new(task_id, String::new(), state.current_revision.clone());
    let mut view = TaskView {
        task_id,
        agent: AgentId::ROOT,
        goal: goal.clone(),
        window: 0,
        turns_in_window: 0,
        base_revision: None,
        latest_user_message: None,
        changed_paths: vec![],
        latest_ptc: None,
        latest_failure: None,
        evidence: vec![],
        prelude: None,
        checkpoint: None,
        pending_steering: vec![],
        cancel_requested: false,
        window_start_seq: 0,
        legacy_evidence: vec![],
        agents: vec![],
        processes: vec![],
        repeated_fingerprint: None,
        resolved_workspaces: vec![],
        unresolved_recovery_workspaces: vec![],
        outcome_unknown_executions: vec![],
        status_note: None,
    };
    let mut queued_steering: Vec<(u64, String)> = Vec::new();
    let mut isolated_revision: Option<RevisionId> = None;
    let mut terminal = false;

    for record in events {
        match &record.event {
            SessionEvent::PreludeLoaded { prelude, .. } => view.prelude = Some(prelude.clone()),
            SessionEvent::TaskStarted {
                task_id: id,
                objective,
                base_revision,
                activate,
            } if *id == task_id => {
                goal = GoalState::new(task_id, objective.clone(), base_revision.clone());
                if !*activate {
                    goal.status = TaskStatus::Queued;
                }
                view.base_revision = Some(base_revision.clone());
                view.changed_paths.clear();
                view.evidence.clear();
                view.legacy_evidence.clear();
                view.window = 0;
                view.turns_in_window = 0;
                view.window_start_seq = record.seq;
                view.checkpoint = None;
                terminal = false;
            }
            SessionEvent::AgentSpawned {
                agent,
                task_id: id,
                objective,
                base_revision,
                isolated_workspace,
                ..
            } if *id == task_id => {
                // A worker is a normal agent execution with a parent: its task
                // gets the same durable treatment as a root task.
                goal = GoalState::new(task_id, objective.clone(), base_revision.clone());
                view.agent = *agent;
                view.base_revision = Some(base_revision.clone());
                view.window = 0;
                view.turns_in_window = 0;
                view.window_start_seq = record.seq;
                if isolated_workspace.is_some() {
                    isolated_revision = Some(base_revision.clone());
                }
            }
            SessionEvent::GoalUpdated {
                task_id: id,
                update,
            } if *id == task_id => {
                goal.apply(update.clone());
            }
            SessionEvent::TaskStatusChanged {
                task_id: id,
                status,
                note,
            } if *id == task_id && !terminal => {
                goal.status = *status;
                view.status_note.clone_from(note);
                terminal = !status.is_active();
            }
            SessionEvent::TaskCompleted { task_id: id, .. } if *id == task_id && !terminal => {
                goal.status = TaskStatus::Completed;
                terminal = true;
            }
            SessionEvent::TaskInterrupted { task_id: id } if *id == task_id && !terminal => {
                goal.status = TaskStatus::Cancelled;
                terminal = true;
            }
            SessionEvent::TaskFailed { task_id: id, .. } if *id == task_id && !terminal => {
                goal.status = TaskStatus::Failed;
                terminal = true;
            }
            SessionEvent::TaskCancelRequested { task_id: id, .. } if *id == task_id => {
                if !terminal {
                    view.cancel_requested = true;
                }
            }
            SessionEvent::UserMessage {
                task_id: id,
                content,
            } if *id == task_id => {
                view.latest_user_message = Some(content.clone());
            }
            SessionEvent::SteeringQueued {
                command_id,
                task_id: id,
                content,
            } if *id == task_id => queued_steering.push((*command_id, content.clone())),
            SessionEvent::SteeringApplied {
                command_id,
                task_id: id,
            } if *id == task_id => {
                if let Some(index) = queued_steering
                    .iter()
                    .position(|(queued_id, _)| queued_id == command_id)
                {
                    let (_, content) = queued_steering.remove(index);
                    view.latest_user_message = Some(content);
                }
            }
            SessionEvent::AgentMessageApplied {
                task_id: id,
                content,
                ..
            } if *id == task_id => view.latest_user_message = Some(content.clone()),
            SessionEvent::ContextWindowStarted {
                task_id: id,
                window,
                agent,
            } if *id == task_id => {
                view.window = *window;
                view.agent = *agent;
                view.turns_in_window = 0;
                view.window_start_seq = record.seq;
                view.repeated_fingerprint = None;
            }
            SessionEvent::Compacted {
                task_id: id,
                window,
                checkpoint,
                ..
            } if *id == task_id => {
                view.checkpoint = checkpoint.clone();
                view.window = window.saturating_add(1);
                view.turns_in_window = 0;
                view.window_start_seq = record.seq;
                view.repeated_fingerprint = None;
            }
            SessionEvent::ModelCompleted { task_id: id, .. } if *id == task_id => {
                view.turns_in_window = view.turns_in_window.saturating_add(1);
            }
            SessionEvent::RepeatedActionDetected {
                task_id: id,
                fingerprint,
            } if *id == task_id => view.repeated_fingerprint = Some(fingerprint.clone()),
            SessionEvent::WorkspaceRevisionChanged {
                task_id: id,
                added,
                modified,
                deleted,
                to,
                isolated_workspace,
                ..
            } if *id == task_id => {
                view.changed_paths.extend(
                    added
                        .iter()
                        .chain(modified)
                        .chain(deleted)
                        .map(PathBuf::from),
                );
                // An isolated worker's own revisions are the truth for that
                // worker's task, while the parent workspace stays untouched.
                if isolated_workspace.is_some() {
                    isolated_revision = Some(to.clone());
                }
            }
            SessionEvent::WorkspaceDriftDetected {
                task_id: id, delta, ..
            } if *id == Some(task_id) => view.changed_paths.extend(
                delta
                    .added
                    .iter()
                    .chain(&delta.modified)
                    .chain(&delta.deleted)
                    .cloned(),
            ),
            SessionEvent::PtcCompleted {
                task_id: id,
                execution_id,
                value,
                tool_calls,
                duration_ms,
                end_revision,
            } if *id == task_id => {
                view.latest_ptc = Some(PtcSummary {
                    ok: true,
                    value: value.clone(),
                    error: None,
                    tool_calls: *tool_calls,
                    duration_ms: *duration_ms,
                    revision: end_revision.clone(),
                    timestamp_ms: record.timestamp_ms,
                    task_id,
                    execution_id: *execution_id,
                });
            }
            SessionEvent::PtcFailed {
                task_id: id,
                execution_id,
                error,
                value,
                tool_calls,
                duration_ms,
                end_revision,
            } if *id == task_id => {
                view.latest_failure = Some(PtcSummary {
                    ok: false,
                    value: value.clone(),
                    error: Some(error.clone()),
                    tool_calls: *tool_calls,
                    duration_ms: *duration_ms,
                    revision: end_revision.clone(),
                    timestamp_ms: record.timestamp_ms,
                    task_id,
                    execution_id: *execution_id,
                });
            }
            // Integration and explicit discard both resolve a worker delta.
            SessionEvent::IntegrationCompleted {
                task_id: id, agent, ..
            } if *id == task_id => {
                if let Some(workspace) = events.iter().rev().find_map(|other| match &other.event {
                    SessionEvent::AgentSpawned {
                        agent: spawned,
                        isolated_workspace,
                        ..
                    } if spawned == agent => *isolated_workspace,
                    _ => None,
                }) {
                    view.resolved_workspaces.push(workspace.0);
                }
            }
            SessionEvent::IntegrationDiscarded {
                task_id: id,
                workspace,
                ..
            } if *id == task_id => view.resolved_workspaces.push(workspace.0),
            SessionEvent::EvidenceRecorded { evidence } if evidence.task_id == task_id => {
                view.evidence.push(evidence.clone());
            }
            SessionEvent::Legacy {
                event_type,
                payload,
            } if event_type == "evidence_recorded" => {
                if let Some(value) = payload.get("evidence")
                    && let Ok(evidence) = serde_json::from_value(value.clone())
                {
                    view.legacy_evidence.push(evidence);
                }
            }
            _ => {}
        }
    }

    let agents = derive_agents(events);

    for record in events {
        match &record.event {
            SessionEvent::RecoveryApplied {
                owner_lost_agents,
                outcome_unknown_executions,
                unresolved_workspaces,
                ..
            } => {
                if owner_lost_agents.contains(&view.agent)
                    && !view.agent.is_root()
                    && goal.status.is_active()
                {
                    goal.status = TaskStatus::Failed;
                }
                view.outcome_unknown_executions.extend(
                    in_flight_executions(events)
                        .values()
                        .filter(|(owner_task, execution)| {
                            task_belongs_to_root(&agents, *owner_task, task_id)
                                && outcome_unknown_executions.contains(execution)
                        })
                        .map(|(_, execution)| *execution),
                );
                for workspace in unresolved_workspaces {
                    if !view.resolved_workspaces.contains(&workspace.0)
                        && agents.values().any(|agent| {
                            agent.parent_task_id == task_id
                                && agent.isolated_workspace == Some(*workspace)
                        })
                    {
                        view.unresolved_recovery_workspaces.push(workspace.0);
                    }
                }
            }
            SessionEvent::RecoveryResolved {
                execution_id: Some(execution),
                ..
            } => {
                view.outcome_unknown_executions
                    .retain(|unknown| unknown != execution);
            }
            _ => {}
        }
    }
    view.outcome_unknown_executions.sort_unstable();
    view.outcome_unknown_executions.dedup();
    view.unresolved_recovery_workspaces.sort_unstable();
    view.unresolved_recovery_workspaces.dedup();

    // The revision a task's work is judged against is the *current* state of
    // the workspace it runs in, not the revision it started from: leaving it
    // pinned to the base would make every later verification look fresh and
    // let finish() accept unverified work. An isolated worker never advances
    // the parent revision, so it is judged against its private workspace.
    goal.current_revision = isolated_revision.unwrap_or_else(|| state.current_revision.clone());
    goal.last_verified_revision = view
        .evidence
        .iter()
        .filter(|record| record.ok)
        .max_by_key(|record| record.timestamp_ms)
        .map(|record| record.revision.clone());
    view.pending_steering = queued_steering
        .into_iter()
        .map(|(_, content)| content)
        .collect();
    view.changed_paths.sort();
    view.changed_paths.dedup();
    view.agents = agents
        .values()
        .filter(|record| record.parent_task_id == task_id)
        .cloned()
        .collect();
    view.processes = derive_processes(events)
        .into_values()
        .filter(|record| {
            record.task_id == task_id
                || (view.agent.is_root() && task_belongs_to_root(&agents, record.task_id, task_id))
        })
        .collect();
    let recovery_unresolved = !view.outcome_unknown_executions.is_empty()
        || view
            .processes
            .iter()
            .any(|process| process.state == "outcome-unknown")
        || !view.unresolved_recovery_workspaces.is_empty();
    if !terminal && view.agent.is_root() && recovery_unresolved {
        goal.status = TaskStatus::Blocked;
    }
    view.goal = goal;
    view
}

/// Builds the durable semantic checkpoint handed to the next context window.
fn task_belongs_to_root(
    agents: &BTreeMap<u64, AgentRecord>,
    mut task_id: TaskId,
    root_task_id: TaskId,
) -> bool {
    if task_id == root_task_id {
        return true;
    }
    for _ in 0..agents.len() {
        let Some(agent) = agents.values().find(|agent| agent.task_id == task_id) else {
            return false;
        };
        if agent.parent_task_id == root_task_id {
            return true;
        }
        if agent.parent_task_id == task_id {
            return false;
        }
        task_id = agent.parent_task_id;
    }
    false
}

pub fn build_checkpoint(view: &TaskView) -> ContextCheckpoint {
    ContextCheckpoint {
        task_id: view.task_id,
        window: view.window,
        objective: view.goal.objective.clone(),
        status: view.goal.status,
        acceptance_criteria: view.goal.acceptance_criteria.clone(),
        decisions: merge_lists(
            view.checkpoint.as_ref().map(|c| c.decisions.clone()),
            view.goal.decisions.clone(),
            |decision: &Decision| decision.decision.clone(),
        ),
        completed: merge_lists(
            view.checkpoint.as_ref().map(|c| c.completed.clone()),
            view.goal.completed_work.clone(),
            |item: &WorkItem| item.title.clone(),
        ),
        pending: if view.goal.pending_work.is_empty() {
            view.checkpoint
                .as_ref()
                .map(|c| c.pending.clone())
                .unwrap_or_default()
        } else {
            view.goal.pending_work.clone()
        },
        blockers: if view.goal.blockers.is_empty() {
            view.checkpoint
                .as_ref()
                .map(|c| c.blockers.clone())
                .unwrap_or_default()
        } else {
            view.goal.blockers.clone()
        },
        important_findings: merge_lists(
            view.checkpoint
                .as_ref()
                .map(|c| c.important_findings.clone()),
            view.goal.findings.clone(),
            |finding: &Finding| finding.summary.clone(),
        ),
        failed_approaches: merge_lists(
            view.checkpoint
                .as_ref()
                .map(|c| c.failed_approaches.clone()),
            view.goal.failed_approaches.clone(),
            |failure: &Failure| failure.approach.clone(),
        ),
        changed_paths: view.changed_paths.clone(),
        verification: view
            .latest_evidence()
            .into_iter()
            .map(|record| crate::goal::VerificationSummary {
                kind: record.kind.clone(),
                ok: record.ok,
                revision: record.revision.clone(),
                fresh: view.evidence_is_fresh(record),
                note: record.note.clone(),
            })
            .collect(),
        active_agents: view
            .agents
            .iter()
            .map(|record| crate::goal::AgentSummary {
                agent: record.agent,
                task: record.objective.clone(),
                state: record.state.label().to_string(),
                summary: record
                    .result
                    .as_ref()
                    .map(|result| bounded(&result.summary, 400)),
            })
            .collect(),
        active_processes: view
            .processes
            .iter()
            .map(|record| crate::goal::ProcessSummary {
                process: record.process,
                argv: record.argv.clone(),
                state: record.state.clone(),
                exit_code: record.exit_code,
            })
            .collect(),
        next_actions: view.goal.next_actions.clone(),
        current_revision: view.goal.current_revision.clone(),
        last_verified_revision: view.goal.last_verified_revision.clone(),
    }
}

/// Carries an earlier window's durable items forward, newest last, without
/// duplicating an item the model restated.
fn merge_lists<T: Clone>(
    previous: Option<Vec<T>>,
    current: Vec<T>,
    key: impl Fn(&T) -> String,
) -> Vec<T> {
    let mut out = previous.unwrap_or_default();
    for item in current {
        let id = key(&item);
        if let Some(existing) = out.iter_mut().find(|other| key(other) == id) {
            *existing = item;
        } else {
            out.push(item);
        }
    }
    out
}

fn bounded(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

/// Blockers reported by a worker are the parent's problem too; exposed for the
/// runtime host when a worker fails.
pub fn blocker_from_failure(summary: &str) -> Blocker {
    Blocker {
        summary: bounded(summary, 400),
        needs: None,
    }
}

/// Objections a `finish()` request must clear, computed from durable state.
pub fn finish_objections(view: &TaskView, request: &FinishRequest) -> Vec<FinishObjection> {
    let mut objections = Vec::new();
    let unresolved = view.unresolved_agents();
    if !unresolved.is_empty() {
        objections.push(FinishObjection::UnresolvedAgents { agents: unresolved });
    }
    let running = view
        .processes
        .iter()
        .filter(|record| record.is_running() || record.state == "outcome-unknown")
        .map(|record| record.process)
        .collect::<Vec<_>>();
    if !running.is_empty() {
        objections.push(FinishObjection::RunningProcesses { processes: running });
    }
    if !view.outcome_unknown_executions.is_empty() {
        objections.push(FinishObjection::OutcomeUnknownExecutions {
            executions: view.outcome_unknown_executions.clone(),
        });
    }
    let failed: Vec<String> = view
        .latest_evidence()
        .into_iter()
        .filter(|record| !record.ok)
        .map(|record| record.kind.clone())
        .collect();
    if !failed.is_empty() {
        objections.push(FinishObjection::FailedVerification { kinds: failed });
    }
    let workspaces = view.unintegrated_workspaces();
    if !workspaces.is_empty() {
        objections.push(FinishObjection::UnintegratedWorkspaces { workspaces });
    }
    // Staleness is only a separate objection once something actually passed.
    // A task that never verified anything (pure inspection) must still be able
    // to finish, and a task whose only verification *failed* is already covered
    // by FailedVerification — reporting both would tell the model to re-run a
    // verification it must first fix.
    if view.goal.last_verified_revision.is_some() && !view.goal.verification_is_current() {
        objections.push(FinishObjection::StaleVerification {
            current: view.goal.current_revision.clone(),
            verified: view.goal.last_verified_revision.clone(),
        });
    }
    if !view.goal.pending_work.is_empty() {
        objections.push(FinishObjection::PendingWork {
            items: view
                .goal
                .pending_work
                .iter()
                .map(|item| item.title.clone())
                .collect(),
        });
    }
    let unmet: Vec<String> = view
        .goal
        .acceptance_criteria
        .iter()
        .filter(|criterion| !criterion.met)
        .map(|criterion| criterion.description.clone())
        .collect();
    if !unmet.is_empty() {
        objections.push(FinishObjection::UnmetCriteria { criteria: unmet });
    }
    let _ = request;
    objections
}

/// Validates a `finish()` request against durable state.
pub fn validate_finish(view: &TaskView, request: &FinishRequest) -> FinishVerdict {
    FinishVerdict::evaluate(finish_objections(view, request), request.force)
}

struct Replay {
    events: Vec<EventRecord>,
    warning: Option<String>,
}

fn append_record(journal: &mut Journal, event: SessionEvent) -> Result<EventRecord, SessionError> {
    let seq = journal.events.last().map_or(Ok(1), |record| {
        record
            .seq
            .checked_add(1)
            .ok_or_else(|| SessionError::State("event sequence exhausted".to_string()))
    })?;
    let record = EventRecord {
        seq,
        timestamp_ms: now_ms(),
        event,
    };
    let mut bytes = serde_json::to_vec(&record)?;
    bytes.push(b'\n');
    let path = journal.root.join(EVENT_FILE);
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_data()) {
        journal.poisoned = true;
        return Err(SessionError::CommitOutcomeUnknown(format!(
            "write or sync failed for {}: {error}",
            path.display()
        )));
    }
    apply(&mut journal.state, &record);
    journal.events.push(record.clone());
    if let Err(error) = write_state(journal) {
        journal.warnings.push(format!(
            "state cache refresh failed after seq {seq}: {error}"
        ));
    }
    Ok(record)
}

fn apply(state: &mut SessionState, record: &EventRecord) {
    state.last_event_seq = record.seq;
    state.updated_at_ms = state.updated_at_ms.max(record.timestamp_ms);
    match &record.event {
        SessionEvent::SessionInitialized {
            metadata,
            initial_revision,
            ..
        } => {
            state.id.clone_from(&metadata.id);
            state.workspace.clone_from(&metadata.workspace);
            state.created_at_ms = metadata.created_at_ms;
            state.updated_at_ms = metadata.created_at_ms;
            state.current_revision.clone_from(initial_revision);
        }
        SessionEvent::TaskStarted {
            task_id,
            base_revision,
            activate,
            ..
        } => {
            if *activate {
                state.active_task = Some(*task_id);
                state.current_revision = base_revision.clone();
            }
        }
        SessionEvent::RunnerAdmitted { info } => {
            if let Some(task_id) = info.task_id {
                state.active_task = Some(task_id);
            }
        }
        SessionEvent::TaskCompleted {
            task_id,
            final_revision,
            ..
        } => {
            if state.active_task == Some(*task_id) {
                state.active_task = None;
            }
            state.current_revision = final_revision.clone();
        }
        SessionEvent::TaskInterrupted { task_id } | SessionEvent::TaskFailed { task_id, .. } => {
            if state.active_task == Some(*task_id) {
                state.active_task = None;
            }
        }
        SessionEvent::TaskStatusChanged {
            task_id, status, ..
        } if !status.is_active() => {
            if state.active_task == Some(*task_id) {
                state.active_task = None;
            }
        }
        SessionEvent::WorkspaceRevisionChanged {
            to,
            isolated_workspace: None,
            ..
        } => state.current_revision = to.clone(),
        SessionEvent::WorkspaceDriftDetected { actual, .. } => {
            state.current_revision = actual.clone();
        }
        _ => {}
    }
}

fn rebuild_from_journal(
    state: &mut SessionState,
    events: &[EventRecord],
) -> Result<(), SessionError> {
    let Some(first) = events.first() else {
        state.active_task = None;
        state.last_event_seq = 0;
        return Ok(());
    };
    let SessionEvent::SessionInitialized {
        metadata,
        initial_revision,
        ..
    } = &first.event
    else {
        return Err(SessionError::State(
            "journal has no replayable session header".to_string(),
        ));
    };
    *state = SessionState {
        id: metadata.id.clone(),
        workspace: metadata.workspace.clone(),
        active_task: None,
        current_revision: initial_revision.clone(),
        last_event_seq: 0,
        created_at_ms: metadata.created_at_ms,
        updated_at_ms: metadata.created_at_ms,
    };
    for event in events {
        apply(state, event);
    }
    Ok(())
}

fn write_state(journal: &Journal) -> Result<(), SessionError> {
    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp = journal.root.join(format!(
        ".{STATE_FILE}.{}.{}.tmp",
        std::process::id(),
        nonce
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        serde_json::to_writer_pretty(&mut file, &journal.state)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        std::fs::rename(&temp, journal.root.join(STATE_FILE))?;
        sync_directory(&journal.root)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

fn read_events(path: &Path, repair_torn_tail: bool) -> Result<Replay, SessionError> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Replay {
                events: Vec::new(),
                warning: None,
            });
        }
        Err(error) => return Err(error.into()),
    };
    let mut events = Vec::new();
    let mut offset = 0usize;
    let mut warning = None;
    for frame in bytes.split_inclusive(|byte| *byte == b'\n') {
        let terminated = frame.last() == Some(&b'\n');
        let payload = if terminated {
            &frame[..frame.len().saturating_sub(1)]
        } else {
            frame
        };
        if payload.is_empty() {
            offset += frame.len();
            continue;
        }
        if !terminated {
            warning = Some(format!("ignored torn journal tail at byte {offset}"));
            if repair_torn_tail {
                preserve_and_truncate_torn_tail(path, &bytes, offset)?;
            }
            break;
        }
        let record = decode_record(payload, path, offset as u64)?;
        let expected = events.last().map_or_else(
            || {
                if matches!(record.event, SessionEvent::SessionInitialized { .. })
                    && record.seq <= 1
                {
                    record.seq
                } else {
                    1
                }
            },
            |prior: &EventRecord| prior.seq + 1,
        );
        if record.seq != expected {
            return Err(SessionError::CorruptJournal {
                path: path.to_path_buf(),
                offset: offset as u64,
                message: format!("expected sequence {expected}, found {}", record.seq),
            });
        }
        events.push(record);
        offset += frame.len();
    }
    Ok(Replay { events, warning })
}

fn decode_record(payload: &[u8], path: &Path, offset: u64) -> Result<EventRecord, SessionError> {
    match serde_json::from_slice(payload) {
        Ok(record) => Ok(record),
        Err(primary) => {
            let mut raw: Value =
                serde_json::from_slice(payload).map_err(|_| SessionError::CorruptJournal {
                    path: path.to_path_buf(),
                    offset,
                    message: primary.to_string(),
                })?;
            let seq = raw.get("seq").and_then(Value::as_u64).ok_or_else(|| {
                SessionError::CorruptJournal {
                    path: path.to_path_buf(),
                    offset,
                    message: "record has no integer seq".to_string(),
                }
            })?;
            let timestamp_ms = raw.get("timestamp_ms").and_then(Value::as_u64).unwrap_or(0);
            let event_type = raw.get("type").and_then(Value::as_str).ok_or_else(|| {
                SessionError::CorruptJournal {
                    path: path.to_path_buf(),
                    offset,
                    message: "record has no event type".to_string(),
                }
            })?;
            if known_event_type(event_type) {
                return Err(SessionError::CorruptJournal {
                    path: path.to_path_buf(),
                    offset,
                    message: primary.to_string(),
                });
            }
            let event_type = event_type.to_string();
            if let Some(object) = raw.as_object_mut() {
                object.remove("seq");
                object.remove("timestamp_ms");
                object.remove("type");
            }
            Ok(EventRecord {
                seq,
                timestamp_ms,
                event: SessionEvent::Legacy {
                    event_type,
                    payload: raw,
                },
            })
        }
    }
}

fn preserve_and_truncate_torn_tail(
    path: &Path,
    bytes: &[u8],
    offset: usize,
) -> Result<(), SessionError> {
    let backup = path.with_file_name(format!(
        "{CORRUPT_TAIL_PREFIX}.{}.{}",
        now_ms(),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&backup)?;
    file.write_all(&bytes[offset..])?;
    file.sync_all()?;
    OpenOptions::new()
        .write(true)
        .open(path)?
        .set_len(offset as u64)?;
    sync_directory(path.parent().expect("journal has parent"))?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), SessionError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn canonical_workspace(path: &Path) -> Result<PathBuf, std::io::Error> {
    path.canonicalize()
}

fn state_revision(root: &Path) -> Option<RevisionId> {
    std::fs::read(root.join(STATE_FILE))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<SessionState>(&bytes).ok())
        .map(|state| state.current_revision)
}

fn journal_has_current_header(path: &Path) -> Result<bool, SessionError> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let Some(line) = bytes
        .split(|byte| *byte == b'\n')
        .find(|line| !line.is_empty())
    else {
        return Ok(false);
    };
    let value: Value =
        serde_json::from_slice(line).map_err(|error| SessionError::CorruptJournal {
            path: path.to_path_buf(),
            offset: 0,
            message: error.to_string(),
        })?;
    Ok(
        value.get("type").and_then(Value::as_str) == Some("session_initialized")
            && value.pointer("/metadata/version").and_then(Value::as_u64)
                == Some(u64::from(SESSION_VERSION)),
    )
}

fn migrate_legacy_session(
    root: &Path,
    workspace: &Path,
    fallback_revision: &RevisionId,
) -> Result<(), SessionError> {
    let event_path = root.join(EVENT_FILE);
    if journal_has_current_header(&event_path)? {
        return Ok(());
    }
    let state_path = root.join(STATE_FILE);
    if !state_path.exists() {
        return Ok(());
    }
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(JOURNAL_LOCK_FILE))?;
    lock.lock()?;
    if journal_has_current_header(&event_path)? {
        return Ok(());
    }

    let raw_state: Value =
        serde_json::from_slice(&std::fs::read(&state_path)?).map_err(|error| {
            SessionError::State(format!(
                "legacy session metadata cannot be recovered from {}: {error}",
                state_path.display()
            ))
        })?;
    let id = raw_state
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            SessionError::State(
                "legacy session lacks a recoverable immutable session identity".to_string(),
            )
        })?
        .to_string();
    let created_at_ms = raw_state
        .get("created_at_ms")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            SessionError::State("legacy session lacks a recoverable creation timestamp".to_string())
        })?;
    let initial_revision = raw_state
        .get("current_revision")
        .cloned()
        .and_then(|value| serde_json::from_value::<RevisionId>(value).ok())
        .unwrap_or_else(|| fallback_revision.clone());

    let bytes = match std::fs::read(&event_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    if !bytes.is_empty() && !bytes.ends_with(b"\n") {
        return Err(SessionError::CorruptJournal {
            path: event_path,
            offset: bytes
                .iter()
                .rposition(|byte| *byte == b'\n')
                .map_or(0, |offset| offset + 1) as u64,
            message: "legacy journal has an incomplete final frame".to_string(),
        });
    }
    let mut values = Vec::new();
    let mut expected = 1u64;
    let mut next_command = 1u64;
    let mut queued: Vec<(u64, u64, String)> = Vec::new();
    for (index, line) in bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .enumerate()
    {
        let mut value: Value =
            serde_json::from_slice(line).map_err(|error| SessionError::CorruptJournal {
                path: event_path.clone(),
                offset: index as u64,
                message: error.to_string(),
            })?;
        let seq = value.get("seq").and_then(Value::as_u64).ok_or_else(|| {
            SessionError::CorruptJournal {
                path: event_path.clone(),
                offset: index as u64,
                message: "legacy record has no integer seq".to_string(),
            }
        })?;
        if seq != expected {
            return Err(SessionError::CorruptJournal {
                path: event_path.clone(),
                offset: index as u64,
                message: format!("expected sequence {expected}, found {seq}"),
            });
        }
        expected = expected.checked_add(1).ok_or_else(|| {
            SessionError::State("event sequence exhausted during migration".to_string())
        })?;
        let event_type = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let object = value
            .as_object_mut()
            .ok_or_else(|| SessionError::CorruptJournal {
                path: event_path.clone(),
                offset: index as u64,
                message: "legacy record is not a JSON object".to_string(),
            })?;
        match event_type.as_str() {
            "steering_queued" => {
                let task = object
                    .get("task_id")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| {
                        SessionError::State("legacy steering has no task id".to_string())
                    })?;
                let content = object
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        SessionError::State("legacy steering has no content".to_string())
                    })?
                    .to_string();
                object.insert("command_id".to_string(), Value::from(next_command));
                queued.push((next_command, task, content));
                next_command = next_command.checked_add(1).ok_or_else(|| {
                    SessionError::State("command identity exhausted during migration".to_string())
                })?;
            }
            "steering_applied" => {
                let task = object
                    .get("task_id")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| {
                        SessionError::State("legacy applied steering has no task id".to_string())
                    })?;
                let content = object
                    .remove("content")
                    .and_then(|value| value.as_str().map(str::to_string))
                    .ok_or_else(|| {
                        SessionError::State("legacy applied steering has no content".to_string())
                    })?;
                let position = queued
                    .iter()
                    .position(|(_, queued_task, queued_content)| {
                        *queued_task == task && queued_content == &content
                    })
                    .ok_or_else(|| {
                        SessionError::State(
                            "legacy applied steering has no matching queued command".to_string(),
                        )
                    })?;
                let (command_id, _, _) = queued.remove(position);
                object.insert("command_id".to_string(), Value::from(command_id));
            }
            "task_cancel_requested" => {
                object.insert("command_id".to_string(), Value::from(next_command));
                next_command = next_command.checked_add(1).ok_or_else(|| {
                    SessionError::State("command identity exhausted during migration".to_string())
                })?;
            }
            _ => {}
        }
        serde_json::from_value::<EventRecord>(value.clone()).map_err(|error| {
            SessionError::CorruptJournal {
                path: event_path.clone(),
                offset: index as u64,
                message: format!("unsupported legacy event schema: {error}"),
            }
        })?;
        values.push(value);
    }

    let metadata = SessionMetadata {
        id,
        workspace: workspace.to_path_buf(),
        created_at_ms,
        version: SESSION_VERSION,
    };
    let header = EventRecord {
        seq: 0,
        timestamp_ms: created_at_ms,
        event: SessionEvent::SessionInitialized {
            metadata,
            initial_revision,
            migrated_from: Some(4),
        },
    };
    let migration = EventRecord {
        seq: expected,
        timestamp_ms: now_ms(),
        event: SessionEvent::SessionMigrated {
            from_version: 4,
            to_version: SESSION_VERSION,
        },
    };
    static MIGRATION_COUNTER: AtomicU64 = AtomicU64::new(0);
    let temp = root.join(format!(
        ".{EVENT_FILE}.migration.{}.{}.tmp",
        std::process::id(),
        MIGRATION_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    serde_json::to_writer(&mut file, &header)?;
    file.write_all(b"\n")?;
    for value in values {
        serde_json::to_writer(&mut file, &value)?;
        file.write_all(b"\n")?;
    }
    serde_json::to_writer(&mut file, &migration)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    std::fs::rename(&temp, &event_path)?;
    sync_directory(root)?;
    Ok(())
}

fn session_metadata(events: &[EventRecord]) -> Option<SessionMetadata> {
    events.iter().find_map(|record| match &record.event {
        SessionEvent::SessionInitialized { metadata, .. } => Some(metadata.clone()),
        _ => None,
    })
}

fn new_instance_id(now: u64) -> String {
    static INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "{now:x}-{:x}-{:x}",
        std::process::id(),
        INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn command_id_in_event(record: &EventRecord) -> Option<u64> {
    match record.event {
        SessionEvent::SteeringQueued { command_id, .. }
        | SessionEvent::SteeringApplied { command_id, .. }
        | SessionEvent::TaskCancelRequested { command_id, .. } => Some(command_id),
        _ => None,
    }
}

fn resolve_command_task(
    events: &[EventRecord],
    state: &SessionState,
    requested: Option<TaskId>,
    verb: &str,
) -> Result<TaskId, SessionError> {
    let task_id = match requested {
        Some(task_id) => task_id,
        None => {
            let active = events
                .iter()
                .filter_map(|record| match record.event {
                    SessionEvent::TaskStarted { task_id, .. } => Some(task_id),
                    _ => None,
                })
                .filter(|task_id| derive_task(state, events, *task_id).status().is_active())
                .collect::<Vec<_>>();
            match active.as_slice() {
                [task_id] => *task_id,
                [] => return Err(SessionError::State(format!("no task to {verb}"))),
                _ => {
                    return Err(SessionError::State(format!(
                        "multiple active root tasks; specify a task id to {verb}"
                    )));
                }
            }
        }
    };
    let view = derive_task(state, events, task_id);
    if !view.status().is_active() {
        return Err(SessionError::State(format!(
            "cannot {verb} task {}: already {}",
            task_id.0,
            view.status().label()
        )));
    }
    if !events.iter().any(|record| {
        matches!(
            record.event,
            SessionEvent::TaskStarted { task_id: id, .. }
                | SessionEvent::AgentSpawned { task_id: id, .. }
                if id == task_id
        )
    }) {
        return Err(SessionError::State(format!("no task {}", task_id.0)));
    }
    Ok(task_id)
}

fn next_id(ids: impl Iterator<Item = u64>) -> Result<u64, SessionError> {
    ids.max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| SessionError::State("durable id namespace exhausted".to_string()))
}

fn task_id_in_event(record: &EventRecord) -> Option<u64> {
    match &record.event {
        SessionEvent::IdsReserved { task_id, .. } => task_id.map(|id| id.0),
        SessionEvent::TaskStarted { task_id, .. } | SessionEvent::AgentSpawned { task_id, .. } => {
            Some(task_id.0)
        }
        _ => None,
    }
}

fn agent_id_in_event(record: &EventRecord) -> Option<u64> {
    match &record.event {
        SessionEvent::IdsReserved { agent_id, .. } => agent_id.map(|id| id.0),
        SessionEvent::AgentSpawned { agent, .. } => Some(agent.0),
        _ => None,
    }
}

fn execution_id_in_event(record: &EventRecord) -> Option<u64> {
    match &record.event {
        SessionEvent::IdsReserved { execution_id, .. } => execution_id.map(|id| id.0),
        SessionEvent::PtcStarted { execution_id, .. }
        | SessionEvent::PtcCompleted { execution_id, .. }
        | SessionEvent::PtcFailed { execution_id, .. }
        | SessionEvent::HostCallStarted { execution_id, .. }
        | SessionEvent::ToolCompleted { execution_id, .. }
        | SessionEvent::IntegrationStarted { execution_id, .. } => Some(execution_id.0),
        _ => None,
    }
}

fn process_id_in_event(record: &EventRecord) -> Option<u64> {
    match &record.event {
        SessionEvent::IdsReserved { process_id, .. } => process_id.map(|id| id.0),
        SessionEvent::ProcessSpawned { process, .. } => Some(process.0),
        _ => None,
    }
}

fn default_activate_task() -> bool {
    true
}

fn workspace_id_in_event(record: &EventRecord) -> Option<u64> {
    match &record.event {
        SessionEvent::IdsReserved { workspace_id, .. } => workspace_id.map(|id| id.0),
        SessionEvent::AgentSpawned {
            isolated_workspace, ..
        } => isolated_workspace.map(|id| id.0),
        SessionEvent::IntegrationStarted { workspace, .. }
        | SessionEvent::IntegrationDiscarded { workspace, .. } => Some(workspace.0),
        _ => None,
    }
}

fn known_event_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "session_initialized"
            | "ids_reserved"
            | "prelude_loaded"
            | "task_started"
            | "task_status_changed"
            | "task_completed"
            | "task_interrupted"
            | "task_cancel_requested"
            | "task_failed"
            | "goal_updated"
            | "finish_proposed"
            | "user_message"
            | "assistant_message"
            | "context_window_started"
            | "compacted"
            | "model_started"
            | "model_completed"
            | "model_superseded"
            | "ptc_started"
            | "ptc_completed"
            | "ptc_failed"
            | "agent_spawned"
            | "agent_state_changed"
            | "agent_completed"
            | "agent_message"
            | "agent_message_applied"
            | "integration_started"
            | "integration_completed"
            | "integration_conflict"
            | "integration_discarded"
            | "process_spawned"
            | "process_state_changed"
            | "host_call_started"
            | "tool_completed"
            | "workspace_revision_changed"
            | "workspace_drift_detected"
            | "evidence_recorded"
            | "steering_queued"
            | "steering_applied"
            | "repeated_action_detected"
            | "session_migrated"
            | "runner_admitted"
            | "runner_released"
            | "recovery_applied"
            | "recovery_resolved"
    )
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::AcceptanceCriterion;

    fn session() -> (Session, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let session = Session::open(dir.path()).unwrap();
        (session, dir)
    }

    fn evidence(task_id: TaskId, kind: &str, ok: bool, revision: &str, ts: u64) -> EvidenceRecord {
        EvidenceRecord {
            kind: kind.to_string(),
            ok,
            revision: RevisionId(revision.to_string()),
            result_ids: vec![],
            note: None,
            timestamp_ms: ts,
            task_id,
            execution_id: ExecutionId(1),
            prelude: None,
        }
    }

    #[test]
    fn task_completion_requires_an_explicit_event() {
        let (mut session, _dir) = session();
        let task = session.begin_task("refactor").unwrap();
        session
            .append(SessionEvent::AssistantMessage {
                task_id: task,
                content: "Implemented most of it.".to_string(),
            })
            .unwrap();
        assert_eq!(
            session.state().active_task,
            Some(task),
            "an assistant message must not complete the task"
        );
        assert!(session.task_view(task).status().is_active());

        session
            .complete_task(task, "done".to_string(), vec![], vec![])
            .unwrap();
        assert_eq!(session.state().active_task, None);
        assert_eq!(session.task_view(task).status(), TaskStatus::Completed);
    }

    #[test]
    fn parked_task_does_not_block_a_new_runner_owned_task() {
        let (session, _dir) = session();
        let parked = session.begin_task("parked").unwrap();
        session
            .set_status(parked, TaskStatus::WaitingUser, None)
            .unwrap();

        let next = session.begin_task("next").unwrap();

        assert_ne!(next, parked);
        assert_eq!(session.state().active_task, Some(next));
        assert_eq!(session.task_view(parked).status(), TaskStatus::WaitingUser);
    }

    #[test]
    fn every_terminal_root_event_releases_the_active_pointer() {
        let (session, _dir) = session();
        let interrupted = session.begin_task("interrupt").unwrap();
        session.interrupt_task(interrupted).unwrap();
        assert_eq!(session.state().active_task, None);

        let failed = session.begin_task("fail").unwrap();
        session
            .append_shared_event(SessionEvent::TaskFailed {
                task_id: failed,
                error: "failed".to_string(),
            })
            .unwrap();
        assert_eq!(session.state().active_task, None);
    }

    #[test]
    fn goal_updates_accumulate_across_windows() {
        let (mut session, _dir) = session();
        let task = session.begin_task("refactor storage").unwrap();
        session
            .append(SessionEvent::GoalUpdated {
                task_id: task,
                update: GoalUpdate {
                    pending: Some(vec![WorkItem::new("update callers")]),
                    findings: Some(vec![Finding {
                        summary: "storage is behind a trait".to_string(),
                        paths: vec![],
                    }]),
                    ..GoalUpdate::default()
                },
            })
            .unwrap();
        let view = session.task_view(task);
        assert_eq!(view.goal.pending_work.len(), 1);
        assert_eq!(view.goal.findings.len(), 1);

        let checkpoint = build_checkpoint(&view);
        session
            .append(SessionEvent::Compacted {
                task_id: task,
                agent: AgentId::ROOT,
                window: 0,
                summary: checkpoint.render(),
                checkpoint: Some(checkpoint),
            })
            .unwrap();
        let rolled = session.task_view(task);
        assert_eq!(rolled.window, 1, "compaction opens the next window");
        assert_eq!(rolled.turns_in_window, 0);
        let carried = rolled.checkpoint.unwrap();
        assert_eq!(carried.important_findings.len(), 1);
        assert_eq!(carried.pending.len(), 1);
        assert_eq!(
            rolled.goal.objective, "refactor storage",
            "the goal survives a rollover"
        );
    }

    #[test]
    fn finish_is_refused_while_a_worker_is_unresolved() {
        let (mut session, _dir) = session();
        let task = session.begin_task("delegate").unwrap();
        let (agent, child_task) = session.reserve_agent().unwrap();
        session
            .append(SessionEvent::AgentSpawned {
                parent_task_id: task,
                parent_agent: AgentId::ROOT,
                agent,
                task_id: child_task,
                objective: "inspect".to_string(),
                access: AgentAccess::Read,
                profile: None,
                base_revision: session.state().current_revision,
                context: Value::Null,
                isolated_workspace: None,
            })
            .unwrap();
        session
            .append(SessionEvent::AgentStateChanged {
                agent,
                task_id: child_task,
                state: AgentState::Running,
            })
            .unwrap();

        let request = FinishRequest {
            summary: "done".to_string(),
            ..FinishRequest::default()
        };
        let view = session.task_view(task);
        let verdict = validate_finish(&view, &request);
        assert!(!verdict.accepted);
        assert_eq!(
            verdict.objections,
            vec![FinishObjection::UnresolvedAgents {
                agents: vec![agent]
            }]
        );

        let forced = validate_finish(
            &view,
            &FinishRequest {
                force: true,
                ..request.clone()
            },
        );
        assert!(
            !forced.accepted,
            "force must not complete over a live worker"
        );

        session
            .append(SessionEvent::AgentCompleted {
                agent,
                task_id: child_task,
                result: DelegateResult {
                    task_id: child_task,
                    agent,
                    ok: true,
                    summary: "inspected".to_string(),
                    base_revision: session.state().current_revision,
                    final_revision: session.state().current_revision,
                    changed: false,
                    workspace: None,
                    evidence: vec![],
                    findings: Value::Null,
                },
            })
            .unwrap();
        assert!(validate_finish(&session.task_view(task), &request).accepted);
    }

    #[test]
    fn failed_and_stale_verification_block_completion() {
        let (mut session, _dir) = session();
        let task = session.begin_task("verify").unwrap();
        let revision = session.state().current_revision;
        session
            .append(SessionEvent::EvidenceRecorded {
                evidence: evidence(task, "tests", false, &revision.0, 10),
            })
            .unwrap();
        let request = FinishRequest {
            summary: "shipped".to_string(),
            ..FinishRequest::default()
        };
        let verdict = validate_finish(&session.task_view(task), &request);
        assert_eq!(
            verdict.objections,
            vec![FinishObjection::FailedVerification {
                kinds: vec!["tests".to_string()]
            }]
        );
        assert!(
            !FinishVerdict::evaluate(verdict.objections.clone(), true).accepted,
            "a failing verification is not overridable"
        );

        session
            .append(SessionEvent::EvidenceRecorded {
                evidence: evidence(task, "tests", true, "older", 20),
            })
            .unwrap();
        let stale = validate_finish(&session.task_view(task), &request);
        assert_eq!(
            stale.objections,
            vec![FinishObjection::StaleVerification {
                current: revision.clone(),
                verified: Some(RevisionId("older".to_string())),
            }]
        );

        session
            .append(SessionEvent::EvidenceRecorded {
                evidence: evidence(task, "tests", true, &revision.0, 30),
            })
            .unwrap();
        assert!(validate_finish(&session.task_view(task), &request).accepted);
    }

    #[test]
    fn unmet_criteria_are_overridable_but_reported() {
        let (mut session, _dir) = session();
        let task = session.begin_task("criteria").unwrap();
        session
            .append(SessionEvent::GoalUpdated {
                task_id: task,
                update: GoalUpdate {
                    acceptance_criteria: Some(vec![AcceptanceCriterion {
                        description: "docs updated".to_string(),
                        met: false,
                        evidence_kind: None,
                    }]),
                    ..GoalUpdate::default()
                },
            })
            .unwrap();
        let verdict = validate_finish(
            &session.task_view(task),
            &FinishRequest {
                summary: "partial".to_string(),
                force: true,
                ..FinishRequest::default()
            },
        );
        assert!(verdict.accepted);
        assert_eq!(
            verdict.waived,
            vec![FinishObjection::UnmetCriteria {
                criteria: vec!["docs updated".to_string()]
            }]
        );
    }

    #[test]
    fn steering_appended_by_another_process_is_pending_until_applied() {
        let (session, dir) = session();
        let task = session.begin_task("steer").unwrap();

        // A separate handle stands in for `mh steer` running as its own process.
        let external = Session::resume(dir.path()).unwrap();
        let receipt = external
            .queue_steering(Some(task), "also update the docs".to_string())
            .unwrap();

        session.refresh().unwrap();
        let view = session.task_view(task);
        assert_eq!(view.pending_steering, vec!["also update the docs"]);

        session
            .apply_steering_command(task, receipt.command_id)
            .unwrap();
        assert!(session.task_view(task).pending_steering.is_empty());
        assert_eq!(
            session.task_view(task).latest_user_message.as_deref(),
            Some("also update the docs")
        );
    }

    #[test]
    fn durable_cancel_request_survives_for_a_detached_run() {
        let (session, dir) = session();
        let task = session.begin_task("cancel me").unwrap();
        Session::resume(dir.path())
            .unwrap()
            .request_cancel(Some(task))
            .unwrap();
        session.refresh().unwrap();
        assert!(session.task_view(task).cancel_requested);
    }

    #[test]
    fn worker_tasks_are_derived_independently_of_the_root_task() {
        let (mut session, _dir) = session();
        let root = session.begin_task("root work").unwrap();
        let (agent, child) = session.reserve_agent().unwrap();
        let revision = session.state().current_revision;
        session
            .append(SessionEvent::AgentSpawned {
                parent_task_id: root,
                parent_agent: AgentId::ROOT,
                agent,
                task_id: child,
                objective: "inspect the parser".to_string(),
                access: AgentAccess::Read,
                profile: Some("explore".to_string()),
                base_revision: revision.clone(),
                context: Value::Null,
                isolated_workspace: None,
            })
            .unwrap();
        session
            .append(SessionEvent::GoalUpdated {
                task_id: child,
                update: GoalUpdate {
                    findings: Some(vec![Finding {
                        summary: "precedence lives in parser.rs".to_string(),
                        paths: vec![],
                    }]),
                    ..GoalUpdate::default()
                },
            })
            .unwrap();

        let child_view = session.task_view(child);
        assert_eq!(child_view.agent, agent);
        assert_eq!(child_view.objective(), "inspect the parser");
        assert_eq!(child_view.goal.findings.len(), 1);

        let root_view = session.task_view(root);
        assert_eq!(root_view.objective(), "root work");
        assert!(
            root_view.goal.findings.is_empty(),
            "worker goal state must not leak into the parent"
        );
        assert_eq!(root_view.agents.len(), 1);
        assert_eq!(
            session.state().active_task,
            Some(root),
            "spawning a worker never changes the session's active task"
        );
        assert_eq!(session.tasks().len(), 2);
    }

    #[test]
    fn allocation_never_reuses_an_id_after_resume() {
        let dir = tempfile::tempdir().unwrap();
        let (first_agent, first_task, process) = {
            let mut session = Session::open(dir.path()).unwrap();
            let root = session.begin_task("work").unwrap();
            let (agent, task) = session.reserve_agent().unwrap();
            session
                .append(SessionEvent::AgentSpawned {
                    parent_task_id: root,
                    parent_agent: AgentId::ROOT,
                    agent,
                    task_id: task,
                    objective: "child".to_string(),
                    access: AgentAccess::Read,
                    profile: None,
                    base_revision: session.state().current_revision,
                    context: Value::Null,
                    isolated_workspace: None,
                })
                .unwrap();
            let process = session.reserve_process().unwrap();
            session
                .append(SessionEvent::ProcessSpawned {
                    task_id: root,
                    agent: AgentId::ROOT,
                    process,
                    argv: vec!["sleep".to_string()],
                    cwd: dir.path().to_path_buf(),
                    label: None,
                    pid: Some(1234),
                })
                .unwrap();
            (agent, task, process)
        };

        let resumed = Session::resume(dir.path()).unwrap();
        let (next_agent, next_task) = resumed.reserve_agent().unwrap();
        assert!(next_agent.0 > first_agent.0 && next_task.0 > first_task.0);
        assert!(resumed.reserve_process().unwrap().0 > process.0);
        assert_eq!(resumed.agents().len(), 1);
        assert_eq!(resumed.processes().len(), 1);
        assert!(resumed.processes()[0].is_running());
    }

    #[test]
    fn cache_failure_returns_a_committed_command_receipt() {
        let (session, dir) = session();
        let task = session.begin_task("steer durably").unwrap();
        let state = session.root().join(STATE_FILE);
        std::fs::remove_file(&state).unwrap();
        std::fs::create_dir(&state).unwrap();
        let receipt = session
            .queue_steering(Some(task), "once".to_string())
            .unwrap();

        assert!(receipt.cache_warning.is_some());
        assert!(
            !std::fs::read_dir(session.root())
                .unwrap()
                .filter_map(Result::ok)
                .any(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
        );
        let reopened = Session::inspect(dir.path()).unwrap();
        assert!(
            reopened
                .pending_steering_commands(task)
                .iter()
                .any(|(command, content)| *command == receipt.command_id && content == "once")
        );
    }

    #[test]
    fn identical_steering_commands_keep_distinct_identities() {
        let (session, _dir) = session();
        let task = session.begin_task("steer twice").unwrap();
        let first = session
            .queue_steering(Some(task), "same".to_string())
            .unwrap();
        let second = session
            .queue_steering(Some(task), "same".to_string())
            .unwrap();
        assert_ne!(first.command_id, second.command_id);
        session
            .apply_steering_command(task, first.command_id)
            .unwrap();
        assert_eq!(
            session.pending_steering_commands(task),
            vec![(second.command_id, "same".to_string())]
        );
    }

    #[test]
    fn committed_controls_prevent_late_completion() {
        let (session, _dir) = session();
        let cancelled = session.begin_task("cancel before finish").unwrap();
        session.request_cancel(Some(cancelled)).unwrap();
        assert!(
            session
                .complete_task(cancelled, "too late".to_string(), vec![], vec![])
                .is_err()
        );

        session
            .set_status(cancelled, TaskStatus::Cancelled, None)
            .unwrap();
        let steered = session.begin_task("steer before finish").unwrap();
        session
            .queue_steering(Some(steered), "pending".to_string())
            .unwrap();
        assert!(
            session
                .complete_task(steered, "too late".to_string(), vec![], vec![])
                .is_err()
        );
    }

    #[test]
    fn unknown_well_formed_event_degrades_to_legacy() {
        let dir = tempfile::tempdir().unwrap();
        let session = Session::open(dir.path()).unwrap();
        session.begin_task("work").unwrap();
        let path = dir.path().join(".mh/session.jsonl");
        let next = session.state().last_event_seq + 1;
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(
            file,
            "{{\"seq\":{next},\"timestamp_ms\":1,\"type\":\"from_the_future\"}}"
        )
        .unwrap();
        let session = Session::resume(dir.path()).unwrap();
        assert!(session.events().iter().any(
            |record| matches!(&record.event, SessionEvent::Legacy { event_type, .. }
                    if event_type == "from_the_future")
        ));
    }
    #[test]
    fn terminal_decision_wins_over_late_cancel_and_steering() {
        let (session, _dir) = session();
        let task = session.begin_task("finish atomically").unwrap();
        session
            .complete_task(task, "done".to_string(), vec![], vec![])
            .unwrap();
        assert!(session.request_cancel(Some(task)).is_err());
        assert!(
            session
                .queue_steering(Some(task), "too late".to_string())
                .is_err()
        );
        let view = session.task_view(task);
        assert_eq!(view.status(), TaskStatus::Completed);
        assert!(!view.cancel_requested);
        assert!(view.pending_steering.is_empty());
        assert!(
            session
                .set_status(
                    task,
                    TaskStatus::Blocked,
                    Some("late detached error".to_string()),
                )
                .is_err()
        );
        assert_eq!(session.task_view(task).status(), TaskStatus::Completed);
    }

    #[test]
    fn owner_loss_recovery_is_idempotent_and_blocks_unknown_effects() {
        let (session, _dir) = session();
        let root = session.begin_task("recover").unwrap();
        let (agent, child) = session.reserve_agent().unwrap();
        let workspace = session.reserve_workspace().unwrap();
        let process = session.reserve_process().unwrap();
        let revision = session.state().current_revision;
        session
            .append_shared_event(SessionEvent::RunnerAdmitted {
                info: RunnerInfo {
                    generation: 1,
                    instance_id: "lost".to_string(),
                    pid: 42,
                    task_id: Some(root),
                    started_at_ms: 1,
                },
            })
            .unwrap();
        session
            .append_shared_event(SessionEvent::AgentSpawned {
                parent_task_id: root,
                parent_agent: AgentId::ROOT,
                agent,
                task_id: child,
                objective: "unknown child".to_string(),
                access: AgentAccess::IsolatedWrite,
                profile: None,
                base_revision: revision,
                context: Value::Null,
                isolated_workspace: Some(workspace),
            })
            .unwrap();
        session
            .append_shared_event(SessionEvent::ProcessSpawned {
                task_id: root,
                agent: AgentId::ROOT,
                process,
                argv: vec!["unknown".to_string()],
                cwd: session.workspace_root(),
                label: None,
                pid: Some(1234),
            })
            .unwrap();
        session.recover_previous_owner(2).unwrap();
        session.recover_previous_owner(2).unwrap();
        assert_eq!(
            session
                .events()
                .iter()
                .filter(|record| matches!(
                    record.event,
                    SessionEvent::RecoveryApplied {
                        lost_generation: 1,
                        ..
                    }
                ))
                .count(),
            1
        );
        let view = session.task_view(root);
        assert_eq!(view.status(), TaskStatus::Blocked);
        assert_eq!(view.processes[0].state, "outcome-unknown");
        assert!(view.agents[0].is_terminal());
        assert!(matches!(
            finish_objections(&view, &FinishRequest::default()).as_slice(),
            [
                FinishObjection::RunningProcesses { .. },
                FinishObjection::UnintegratedWorkspaces { .. }
            ]
        ));
        assert_eq!(view.unintegrated_workspaces(), vec![workspace.0]);
    }

    #[test]
    fn lost_worker_resources_block_and_resolve_on_the_root_task() {
        let (session, _dir) = session();
        let root = session.begin_task("recover worker resources").unwrap();
        let (agent, child) = session.reserve_agent().unwrap();
        let execution = session.reserve_execution().unwrap();
        let process = session.reserve_process().unwrap();
        session
            .append_shared_event(SessionEvent::RunnerAdmitted {
                info: RunnerInfo {
                    generation: 1,
                    instance_id: "lost".to_string(),
                    pid: 42,
                    task_id: Some(root),
                    started_at_ms: 1,
                },
            })
            .unwrap();
        session
            .append_shared_event(SessionEvent::AgentSpawned {
                parent_task_id: root,
                parent_agent: AgentId::ROOT,
                agent,
                task_id: child,
                objective: "unknown child effects".to_string(),
                access: AgentAccess::Read,
                profile: None,
                base_revision: session.state().current_revision,
                context: Value::Null,
                isolated_workspace: None,
            })
            .unwrap();
        session
            .append_shared_event(SessionEvent::PtcStarted {
                task_id: child,
                execution_id: execution,
                source: "read()".to_string(),
                start_revision: session.state().current_revision,
            })
            .unwrap();
        session
            .append_shared_event(SessionEvent::ProcessSpawned {
                task_id: child,
                agent,
                process,
                argv: vec!["unknown".to_string()],
                cwd: session.workspace_root(),
                label: None,
                pid: Some(1234),
            })
            .unwrap();

        session.recover_previous_owner(2).unwrap();

        let view = session.task_view(root);
        assert_eq!(view.status(), TaskStatus::Blocked);
        assert_eq!(view.outcome_unknown_executions, vec![execution]);
        assert_eq!(view.processes[0].process, process);
        assert_eq!(view.processes[0].state, "outcome-unknown");
        assert!(matches!(
            finish_objections(&view, &FinishRequest::default()).as_slice(),
            [
                FinishObjection::RunningProcesses { processes },
                FinishObjection::OutcomeUnknownExecutions { executions }
            ] if processes == &vec![process] && executions == &vec![execution]
        ));

        let guard = session.acquire_runner(Some(root)).unwrap();
        guard
            .resolve_recovery(
                root,
                Some(execution),
                None,
                "verified child call manually".to_string(),
            )
            .unwrap();
        guard
            .resolve_recovery(
                root,
                None,
                Some(process),
                "verified child process manually".to_string(),
            )
            .unwrap();
        let view = session.task_view(root);
        assert!(view.outcome_unknown_executions.is_empty());
        assert_eq!(view.processes[0].state, "recovery-resolved");
    }

    #[test]
    fn recovery_does_not_skip_older_crashed_generations() {
        let (session, _dir) = session();
        let root = session.begin_task("recover chained crashes").unwrap();
        let execution = session.reserve_execution().unwrap();
        session
            .append_shared_event(SessionEvent::RunnerAdmitted {
                info: RunnerInfo {
                    generation: 1,
                    instance_id: "first-lost".to_string(),
                    pid: 41,
                    task_id: Some(root),
                    started_at_ms: 1,
                },
            })
            .unwrap();
        session
            .append_shared_event(SessionEvent::PtcStarted {
                task_id: root,
                execution_id: execution,
                source: "write()".to_string(),
                start_revision: session.state().current_revision,
            })
            .unwrap();
        session
            .append_shared_event(SessionEvent::RunnerAdmitted {
                info: RunnerInfo {
                    generation: 2,
                    instance_id: "second-lost".to_string(),
                    pid: 42,
                    task_id: Some(root),
                    started_at_ms: 2,
                },
            })
            .unwrap();

        session.recover_previous_owner(3).unwrap();
        let recovered = session
            .events()
            .into_iter()
            .filter_map(|record| match record.event {
                SessionEvent::RecoveryApplied {
                    lost_generation, ..
                } => Some(lost_generation),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(recovered, vec![1, 2]);
        assert_eq!(
            session.task_view(root).outcome_unknown_executions,
            vec![execution]
        );
    }

    #[test]
    fn recovery_resolution_requires_and_unblocks_the_current_owner() {
        let (session, _dir) = session();
        let root = session.begin_task("recover execution").unwrap();
        let execution = session.reserve_execution().unwrap();
        session
            .append_shared_event(SessionEvent::RunnerAdmitted {
                info: RunnerInfo {
                    generation: 1,
                    instance_id: "lost".to_string(),
                    pid: 42,
                    task_id: Some(root),
                    started_at_ms: 1,
                },
            })
            .unwrap();
        session
            .append_shared_event(SessionEvent::PtcStarted {
                task_id: root,
                execution_id: execution,
                source: "write()".to_string(),
                start_revision: session.state().current_revision,
            })
            .unwrap();
        session.recover_previous_owner(2).unwrap();
        assert_eq!(session.task_view(root).status(), TaskStatus::Blocked);

        let guard = session.acquire_runner(Some(root)).unwrap();
        guard
            .resolve_recovery(
                root,
                Some(execution),
                None,
                "verified side effect manually".to_string(),
            )
            .unwrap();
        assert_eq!(session.task_view(root).status(), TaskStatus::Running);
    }
}
