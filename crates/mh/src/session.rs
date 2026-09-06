//! Durable append-only v4 session store.
//!
//! The journal is the only durable coordination point in mh: a detached run, an
//! inspector, a steering command, and a restarted runtime all agree because they
//! all read the same append-only event log. Task, goal, worker, and process
//! state are therefore *derived* from events rather than cached anywhere else.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
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
pub const SESSION_VERSION: u32 = 4;

#[derive(Debug)]
pub enum SessionError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Workspace(WorkspaceError),
    NoSession(PathBuf),
    State(String),
}
impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "session I/O error: {e}"),
            Self::Json(e) => write!(f, "session data error: {e}"),
            Self::Workspace(e) => write!(f, "workspace error: {e}"),
            Self::NoSession(p) => write!(f, "no session found at {}", p.display()),
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
        self.agents
            .iter()
            .filter_map(|record| {
                let workspace = record.isolated_workspace?;
                let changed = record
                    .result
                    .as_ref()
                    .is_some_and(|result| result.changed && result.workspace.is_some());
                (changed && !self.resolved_workspaces.contains(&workspace.0)).then_some(workspace.0)
            })
            .collect()
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
    PreludeLoaded {
        path: PathBuf,
        prelude: PreludeId,
        described: bool,
    },
    TaskStarted {
        task_id: TaskId,
        objective: String,
        base_revision: RevisionId,
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
    /// Durable cancellation request. A detached run has no control channel, so
    /// `mh cancel` records intent here and the running loop honors it.
    TaskCancelRequested {
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
        task_id: TaskId,
        content: String,
    },
    SteeringApplied {
        task_id: TaskId,
        content: String,
    },
    RepeatedActionDetected {
        task_id: TaskId,
        fingerprint: String,
    },
    SessionMigrated {
        from_version: u32,
        to_version: u32,
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
            | Self::TaskCancelRequested { task_id }
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
            | Self::RepeatedActionDetected { task_id, .. } => Some(*task_id),
            Self::EvidenceRecorded { evidence } => Some(evidence.task_id),
            Self::WorkspaceDriftDetected { task_id, .. } => *task_id,
            Self::PreludeLoaded { .. } | Self::SessionMigrated { .. } | Self::Legacy { .. } => None,
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
    tracker: Arc<dyn WorkspaceTracker>,
    journal: Arc<Mutex<Journal>>,
}
#[derive(Debug, Clone)]
pub struct SessionPtcEventSink {
    journal: Arc<Mutex<Journal>>,
    isolated_workspace: Option<IsolatedWorkspaceId>,
}
impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").field("root", &self.root).finish()
    }
}

impl Session {
    pub fn open(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = absolute(workspace.as_ref())?;
        let root = workspace.join(SESSION_DIR);
        if root.join(STATE_FILE).exists() {
            return Self::load(root, workspace);
        }
        std::fs::create_dir_all(root.join("blobs"))?;
        let tracker: Arc<dyn WorkspaceTracker> = Arc::new(WorkspaceTrackerImpl::open(&workspace)?);
        let now = now_ms();
        let revision = tracker.current_revision()?.id;
        let state = SessionState {
            id: format!("{now:x}-{:x}", std::process::id()),
            workspace,
            active_task: None,
            current_revision: revision,
            last_event_seq: 0,
            created_at_ms: now,
            updated_at_ms: now,
        };
        let session = Self {
            root: root.clone(),
            tracker,
            journal: Arc::new(Mutex::new(Journal {
                root,
                state,
                events: vec![],
            })),
        };
        session.write_state()?;
        Ok(session)
    }
    pub fn resume(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = absolute(workspace.as_ref())?;
        let root = workspace.join(SESSION_DIR);
        if !root.join(STATE_FILE).exists() {
            return Err(SessionError::NoSession(root));
        }
        Self::load(root, workspace)
    }

    /// Opens a session without reconciling workspace drift or migrating.
    ///
    /// Inspection must not mutate the journal: `mh tasks` running beside a live
    /// task would otherwise append drift events for changes that task is in the
    /// middle of making.
    pub fn inspect(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = absolute(workspace.as_ref())?;
        let root = workspace.join(SESSION_DIR);
        if !root.join(STATE_FILE).exists() {
            return Err(SessionError::NoSession(root));
        }
        let raw: Value = serde_json::from_slice(&std::fs::read(root.join(STATE_FILE))?)?;
        let tracker: Arc<dyn WorkspaceTracker> = Arc::new(WorkspaceTrackerImpl::open(&workspace)?);
        let events = read_events(&root.join(EVENT_FILE))?;
        let mut state = match serde_json::from_value::<SessionState>(raw) {
            Ok(state) => state,
            Err(error) => return Err(SessionError::Json(error)),
        };
        rebuild(&mut state, &events);
        Ok(Self {
            root: root.clone(),
            tracker,
            journal: Arc::new(Mutex::new(Journal {
                root,
                state,
                events,
            })),
        })
    }

    fn load(root: PathBuf, workspace: PathBuf) -> Result<Self, SessionError> {
        let raw: Value = serde_json::from_slice(&std::fs::read(root.join(STATE_FILE))?)?;
        let tracker: Arc<dyn WorkspaceTracker> = Arc::new(WorkspaceTrackerImpl::open(&workspace)?);
        let actual = tracker.current_revision()?.id;
        let events = read_events(&root.join(EVENT_FILE))?;
        let legacy = raw.get("active_task").is_none() || raw.get("current_revision").is_none();
        let now = now_ms();
        let mut state = if legacy {
            SessionState {
                id: raw["id"].as_str().unwrap_or("migrated").into(),
                workspace,
                active_task: None,
                current_revision: actual.clone(),
                last_event_seq: 0,
                created_at_ms: raw["created_at_ms"].as_u64().unwrap_or(now),
                updated_at_ms: now,
            }
        } else {
            serde_json::from_value(raw.clone())?
        };
        rebuild(&mut state, &events);
        let session = Self {
            root: root.clone(),
            tracker,
            journal: Arc::new(Mutex::new(Journal {
                root,
                state,
                events,
            })),
        };
        if legacy {
            if let Some(objective) = raw["task"].as_str() {
                let task_id = session.next_task_id();
                session.append_shared(SessionEvent::TaskStarted {
                    task_id,
                    objective: objective.into(),
                    base_revision: actual.clone(),
                })?;
                match raw["status"].as_str() {
                    Some("completed") => {
                        session.append_shared(SessionEvent::TaskCompleted {
                            task_id,
                            final_revision: actual.clone(),
                            summary: String::new(),
                            unresolved: Vec::new(),
                            waived: Vec::new(),
                        })?;
                    }
                    Some("interrupted") => {
                        session.append_shared(SessionEvent::TaskInterrupted { task_id })?;
                    }
                    _ => {}
                }
            }
            session.append_shared(SessionEvent::SessionMigrated {
                from_version: 2,
                to_version: SESSION_VERSION,
            })?;
        }
        session.reconcile_workspace()?;
        Ok(session)
    }
    pub fn append(&mut self, event: SessionEvent) -> Result<EventRecord, SessionError> {
        self.append_shared(event)
    }
    fn append_shared(&self, event: SessionEvent) -> Result<EventRecord, SessionError> {
        append_locked(
            &mut self.journal.lock().expect("session journal poisoned"),
            event,
        )
    }
    pub fn begin_task(&mut self, objective: &str) -> Result<TaskId, SessionError> {
        let task_id = self.next_task_id();
        let base_revision = self.state().current_revision;
        self.append_shared(SessionEvent::TaskStarted {
            task_id,
            objective: objective.into(),
            base_revision,
        })?;
        Ok(task_id)
    }

    pub fn append_shared_event(&self, event: SessionEvent) -> Result<EventRecord, SessionError> {
        self.append_shared(event)
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

    pub fn next_isolated_workspace_id(&self) -> IsolatedWorkspaceId {
        let max_event_id = self
            .events()
            .into_iter()
            .filter_map(|record| match record.event {
                SessionEvent::IntegrationStarted { workspace, .. } => Some(workspace.0),
                SessionEvent::AgentSpawned {
                    isolated_workspace, ..
                } => isolated_workspace.map(|workspace| workspace.0),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        IsolatedWorkspaceId(max_event_id.saturating_add(1))
    }

    /// Allocates the identity of a new delegated worker: its agent id plus the
    /// task id it owns as a durable session of its own.
    pub fn allocate_agent(&self) -> (AgentId, TaskId) {
        let journal = self.journal.lock().expect("session journal poisoned");
        let agent = journal
            .events
            .iter()
            .filter_map(|record| match record.event {
                SessionEvent::AgentSpawned { agent, .. } => Some(agent.0),
                _ => None,
            })
            .max()
            .unwrap_or(AgentId::ROOT.0);
        let task = journal
            .events
            .iter()
            .filter_map(|record| match &record.event {
                SessionEvent::TaskStarted { task_id, .. } => Some(task_id.0),
                SessionEvent::AgentSpawned { task_id, .. } => Some(task_id.0),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        (
            AgentId(agent.saturating_add(1)),
            TaskId(task.saturating_add(1)),
        )
    }

    pub fn allocate_process(&self) -> ProcessId {
        ProcessId(
            self.events()
                .iter()
                .filter_map(|record| match record.event {
                    SessionEvent::ProcessSpawned { process, .. } => Some(process.0),
                    _ => None,
                })
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        )
    }

    pub fn begin_execution(&self) -> ExecutionId {
        self.next_execution_id()
    }

    /// Records explicit, validated task completion.
    pub fn complete_task(
        &self,
        task_id: TaskId,
        summary: String,
        unresolved: Vec<String>,
        waived: Vec<FinishObjection>,
    ) -> Result<(), SessionError> {
        let final_revision = self.state().current_revision;
        self.append_shared(SessionEvent::TaskCompleted {
            task_id,
            final_revision,
            summary,
            unresolved,
            waived,
        })?;
        Ok(())
    }

    pub fn interrupt_task(&self, task_id: TaskId) -> Result<(), SessionError> {
        self.append_shared(SessionEvent::TaskInterrupted { task_id })?;
        Ok(())
    }

    pub fn set_status(
        &self,
        task_id: TaskId,
        status: TaskStatus,
        note: Option<String>,
    ) -> Result<(), SessionError> {
        self.append_shared(SessionEvent::TaskStatusChanged {
            task_id,
            status,
            note,
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
        let actual = self.tracker.current_revision()?.id;
        if expected == actual {
            return Ok(None);
        }
        let delta = self.tracker.delta(&expected, &actual)?;
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
        let mut journal = self.journal.lock().expect("session journal poisoned");
        let events = read_events(&journal.root.join(EVENT_FILE))?;
        if events.len() <= journal.events.len() {
            return Ok(());
        }
        journal.events = events;
        let events = journal.events.clone();
        rebuild(&mut journal.state, &events);
        Ok(())
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
            journal: self.journal.clone(),
            isolated_workspace: None,
        }
    }

    pub fn child_ptc_event_sink(
        &self,
        isolated_workspace: Option<IsolatedWorkspaceId>,
    ) -> SessionPtcEventSink {
        SessionPtcEventSink {
            journal: self.journal.clone(),
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
        self.tracker.clone()
    }
    pub fn result_store(&self) -> Result<ResultStore, SessionError> {
        ResultStore::persistent(self.root.join("blobs")).map_err(SessionError::Io)
    }
    fn next_task_id(&self) -> TaskId {
        TaskId(
            self.events()
                .iter()
                .filter_map(|r| match &r.event {
                    SessionEvent::TaskStarted { task_id, .. } => Some(task_id.0),
                    SessionEvent::AgentSpawned { task_id, .. } => Some(task_id.0),
                    _ => None,
                })
                .max()
                .unwrap_or(0)
                + 1,
        )
    }
    fn next_execution_id(&self) -> ExecutionId {
        ExecutionId(
            self.events()
                .iter()
                .filter_map(|r| match r.event {
                    SessionEvent::PtcStarted { execution_id, .. }
                    | SessionEvent::PtcCompleted { execution_id, .. }
                    | SessionEvent::PtcFailed { execution_id, .. }
                    | SessionEvent::HostCallStarted { execution_id, .. }
                    | SessionEvent::ToolCompleted { execution_id, .. }
                    | SessionEvent::IntegrationStarted { execution_id, .. } => Some(execution_id.0),
                    _ => None,
                })
                .max()
                .unwrap_or(0)
                + 1,
        )
    }
    fn write_state(&self) -> Result<(), SessionError> {
        write_state(&self.journal.lock().expect("session journal poisoned"))
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
        append_locked(
            &mut self.journal.lock().expect("session journal poisoned"),
            event,
        )
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
                    },
                );
            }
            SessionEvent::ProcessStateChanged {
                process,
                state,
                exit_code,
                ..
            } => {
                if let Some(entry) = processes.get_mut(&process.0) {
                    entry.state = state.clone();
                    if exit_code.is_some() {
                        entry.exit_code = *exit_code;
                    }
                }
            }
            _ => {}
        }
    }
    processes
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
    };
    let mut applied_steering: Vec<String> = Vec::new();
    let mut queued_steering: Vec<String> = Vec::new();
    let mut isolated_revision: Option<RevisionId> = None;
    let mut terminal = false;

    for record in events {
        match &record.event {
            SessionEvent::PreludeLoaded { prelude, .. } => view.prelude = Some(prelude.clone()),
            SessionEvent::TaskStarted {
                task_id: id,
                objective,
                base_revision,
            } if *id == task_id => {
                goal = GoalState::new(task_id, objective.clone(), base_revision.clone());
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
                ..
            } if *id == task_id => {
                goal.status = *status;
                terminal = !status.is_active();
            }
            SessionEvent::TaskCompleted { task_id: id, .. } if *id == task_id => {
                goal.status = TaskStatus::Completed;
                terminal = true;
            }
            SessionEvent::TaskInterrupted { task_id: id } if *id == task_id => {
                goal.status = TaskStatus::Cancelled;
                terminal = true;
            }
            SessionEvent::TaskFailed { task_id: id, .. } if *id == task_id => {
                goal.status = TaskStatus::Failed;
                terminal = true;
            }
            SessionEvent::TaskCancelRequested { task_id: id } if *id == task_id => {
                view.cancel_requested = true;
            }
            SessionEvent::UserMessage {
                task_id: id,
                content,
            } if *id == task_id => {
                view.latest_user_message = Some(content.clone());
                if terminal {
                    // A follow-up message revives a parked task rather than
                    // starting a second one; the objective and history stand.
                    goal.status = TaskStatus::Running;
                    terminal = false;
                }
            }
            SessionEvent::SteeringQueued {
                task_id: id,
                content,
            } if *id == task_id => queued_steering.push(content.clone()),
            SessionEvent::SteeringApplied {
                task_id: id,
                content,
            } if *id == task_id => {
                view.latest_user_message = Some(content.clone());
                applied_steering.push(content.clone());
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
        .filter(|content| {
            // Steering applied earlier consumes exactly one queued copy.
            match applied_steering.iter().position(|other| other == content) {
                Some(index) => {
                    applied_steering.remove(index);
                    false
                }
                None => true,
            }
        })
        .collect();
    view.changed_paths.sort();
    view.changed_paths.dedup();
    view.goal = goal;
    view.agents = derive_agents(events)
        .into_values()
        .filter(|record| record.parent_task_id == task_id)
        .collect();
    view.processes = derive_processes(events)
        .into_values()
        .filter(|record| record.task_id == task_id)
        .collect();
    view
}

/// Builds the durable semantic checkpoint handed to the next context window.
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
    let running = view.running_processes();
    if !running.is_empty() {
        objections.push(FinishObjection::RunningProcesses { processes: running });
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

fn append_locked(journal: &mut Journal, event: SessionEvent) -> Result<EventRecord, SessionError> {
    let record = EventRecord {
        seq: journal.state.last_event_seq + 1,
        timestamp_ms: now_ms(),
        event,
    };
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(journal.root.join(EVENT_FILE))?;
    let len = file.metadata()?.len();
    if len > 0 {
        file.seek(SeekFrom::End(-1))?;
        let mut byte = [0];
        file.read_exact(&mut byte)?;
        if byte[0] != b'\n' {
            file.seek(SeekFrom::End(0))?;
            file.write_all(b"\n")?;
        }
    }
    serde_json::to_writer(&mut file, &record)?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    apply(&mut journal.state, &record);
    journal.events.push(record.clone());
    write_state(journal)?;
    Ok(record)
}
fn apply(state: &mut SessionState, record: &EventRecord) {
    state.last_event_seq = state.last_event_seq.max(record.seq);
    state.updated_at_ms = state.updated_at_ms.max(record.timestamp_ms);
    match &record.event {
        SessionEvent::TaskStarted {
            task_id,
            base_revision,
            ..
        } => {
            state.active_task = Some(*task_id);
            state.current_revision = base_revision.clone();
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
        // A follow-up user message revives the task it addresses, so a parked
        // root task keeps its identity across turns.
        SessionEvent::UserMessage { task_id, .. } => {
            if state.active_task.is_none() {
                state.active_task = Some(*task_id);
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
fn rebuild(state: &mut SessionState, events: &[EventRecord]) {
    state.active_task = None;
    state.last_event_seq = 0;
    for event in events {
        apply(state, event);
    }
}
fn write_state(journal: &Journal) -> Result<(), SessionError> {
    let temp = journal.root.join(format!(".{STATE_FILE}.tmp"));
    let mut file = File::create(&temp)?;
    serde_json::to_writer_pretty(&mut file, &journal.state)?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    std::fs::rename(temp, journal.root.join(STATE_FILE))?;
    Ok(())
}
fn read_events(path: &Path) -> Result<Vec<EventRecord>, SessionError> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(e.into()),
    };
    let mut out = vec![];
    for line in BufReader::new(file).split(b'\n') {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        if let Ok(record) = serde_json::from_slice(&line) {
            out.push(record);
            continue;
        }
        let Ok(mut raw) = serde_json::from_slice::<Value>(&line) else {
            continue;
        };
        let Some(seq) = raw["seq"].as_u64() else {
            continue;
        };
        let timestamp_ms = raw["timestamp_ms"].as_u64().unwrap_or(0);
        let event_type = raw["type"].as_str().unwrap_or("unknown").into();
        if let Some(object) = raw.as_object_mut() {
            object.remove("seq");
            object.remove("timestamp_ms");
            object.remove("type");
        }
        out.push(EventRecord {
            seq,
            timestamp_ms,
            event: SessionEvent::Legacy {
                event_type,
                payload: raw,
            },
        });
    }
    out.sort_by_key(|r| r.seq);
    Ok(out)
}
fn absolute(path: &Path) -> Result<PathBuf, std::io::Error> {
    if path.is_absolute() {
        Ok(path.into())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
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
        let (agent, child_task) = session.allocate_agent();
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
        let (mut session, dir) = session();
        let task = session.begin_task("steer").unwrap();

        // A separate handle stands in for `mh steer` running as its own process.
        let external = Session::resume(dir.path()).unwrap();
        external
            .append_shared_event(SessionEvent::SteeringQueued {
                task_id: task,
                content: "also update the docs".to_string(),
            })
            .unwrap();

        session.refresh().unwrap();
        let view = session.task_view(task);
        assert_eq!(view.pending_steering, vec!["also update the docs"]);

        session
            .append(SessionEvent::SteeringApplied {
                task_id: task,
                content: "also update the docs".to_string(),
            })
            .unwrap();
        assert!(session.task_view(task).pending_steering.is_empty());
        assert_eq!(
            session.task_view(task).latest_user_message.as_deref(),
            Some("also update the docs")
        );
    }

    #[test]
    fn durable_cancel_request_survives_for_a_detached_run() {
        let (mut session, dir) = session();
        let task = session.begin_task("cancel me").unwrap();
        Session::resume(dir.path())
            .unwrap()
            .append_shared_event(SessionEvent::TaskCancelRequested { task_id: task })
            .unwrap();
        session.refresh().unwrap();
        assert!(session.task_view(task).cancel_requested);
    }

    #[test]
    fn worker_tasks_are_derived_independently_of_the_root_task() {
        let (mut session, _dir) = session();
        let root = session.begin_task("root work").unwrap();
        let (agent, child) = session.allocate_agent();
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
            let (agent, task) = session.allocate_agent();
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
            let process = session.allocate_process();
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
        let (next_agent, next_task) = resumed.allocate_agent();
        assert!(next_agent.0 > first_agent.0 && next_task.0 > first_task.0);
        assert!(resumed.allocate_process().0 > process.0);
        assert_eq!(resumed.agents().len(), 1);
        assert_eq!(resumed.processes().len(), 1);
        assert!(resumed.processes()[0].is_running());
    }

    #[test]
    fn an_unreadable_event_line_degrades_to_legacy_instead_of_failing() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut session = Session::open(dir.path()).unwrap();
            session.begin_task("work").unwrap();
        }
        let path = dir.path().join(".mh/session.jsonl");
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"{\"seq\":99,\"timestamp_ms\":1,\"type\":\"from_the_future\"}\n")
            .unwrap();
        let session = Session::resume(dir.path()).unwrap();
        assert!(session.events().iter().any(
            |record| matches!(&record.event, SessionEvent::Legacy { event_type, .. }
                    if event_type == "from_the_future")
        ));
    }
}
