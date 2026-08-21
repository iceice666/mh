//! Durable append-only v3 session store.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::identity::{ExecutionId, RevisionId, TaskId};
use crate::ptc::{PtcEvent, PtcEventSink, PtcEventSinkError, PtcOutcome, PtcResult};
use crate::tools::{ResultId, ResultStore, ToolEffects};
use crate::workspace::{
    RevisionSource, WorkspaceDelta, WorkspaceError, WorkspaceTracker, WorkspaceTrackerImpl,
};

const SESSION_DIR: &str = ".mh";
const EVENT_FILE: &str = "session.jsonl";
const STATE_FILE: &str = "state.json";

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkState {
    pub task_id: Option<TaskId>,
    pub objective: Option<String>,
    pub base_revision: Option<RevisionId>,
    pub current_revision: RevisionId,
    pub latest_user_message: Option<String>,
    pub changed_paths: Vec<PathBuf>,
    pub latest_ptc: Option<PtcSummary>,
    pub latest_failure: Option<PtcSummary>,
    pub evidence: Vec<EvidenceRecord>,
    pub legacy_evidence: Vec<LegacyEvidenceRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    TaskStarted {
        task_id: TaskId,
        objective: String,
        base_revision: RevisionId,
    },
    TaskCompleted {
        task_id: TaskId,
        final_revision: RevisionId,
        answer: Option<String>,
    },
    TaskInterrupted {
        task_id: TaskId,
    },
    UserMessage {
        task_id: TaskId,
        content: String,
    },
    AssistantMessage {
        task_id: TaskId,
        content: String,
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
    Compacted {
        task_id: TaskId,
        summary: String,
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
pub struct Session {
    root: PathBuf,
    tracker: Arc<dyn WorkspaceTracker>,
    journal: Arc<Mutex<Journal>>,
}
#[derive(Debug, Clone)]
pub struct SessionPtcEventSink {
    journal: Arc<Mutex<Journal>>,
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
                            answer: None,
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
                to_version: 3,
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
    pub fn begin_execution(&mut self) -> Result<ExecutionId, SessionError> {
        if self.state().active_task.is_none() {
            Err(SessionError::State("session has no active task".into()))
        } else {
            Ok(self.next_execution_id())
        }
    }
    pub fn complete_task(&mut self, answer: Option<String>) -> Result<(), SessionError> {
        let state = self.state();
        let task_id = state
            .active_task
            .ok_or_else(|| SessionError::State("session has no active task".into()))?;
        self.append_shared(SessionEvent::TaskCompleted {
            task_id,
            final_revision: state.current_revision,
            answer,
        })?;
        Ok(())
    }
    pub fn interrupt_task(&mut self) -> Result<(), SessionError> {
        let task_id = self
            .state()
            .active_task
            .ok_or_else(|| SessionError::State("session has no active task".into()))?;
        self.append_shared(SessionEvent::TaskInterrupted { task_id })?;
        Ok(())
    }
    pub fn append_ptc_result(&mut self, result: &PtcResult) -> Result<(), SessionError> {
        let event = match &result.outcome {
            PtcOutcome::Completed => SessionEvent::PtcCompleted {
                task_id: result.task_id,
                execution_id: result.execution_id,
                value: result.value.clone(),
                tool_calls: result.tool_calls,
                duration_ms: result.duration_ms,
                end_revision: result.end_revision.clone(),
            },
            PtcOutcome::Interrupted => {
                self.interrupt_task()?;
                return Ok(());
            }
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
        self.append(event)?;
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
    pub fn work_state(&self) -> WorkState {
        derive_work(self.state(), self.events())
    }
    pub fn ptc_event_sink(&self) -> SessionPtcEventSink {
        SessionPtcEventSink {
            journal: self.journal.clone(),
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
                .filter_map(|r| {
                    if let SessionEvent::TaskStarted { task_id, .. } = r.event {
                        Some(task_id.0)
                    } else {
                        None
                    }
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
                    | SessionEvent::ToolCompleted { execution_id, .. } => Some(execution_id.0),
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

fn derive_work(state: SessionState, events: Vec<EventRecord>) -> WorkState {
    let mut work = WorkState {
        task_id: state.active_task,
        objective: None,
        base_revision: None,
        current_revision: state.current_revision,
        latest_user_message: None,
        changed_paths: vec![],
        latest_ptc: None,
        latest_failure: None,
        evidence: vec![],
        legacy_evidence: vec![],
    };
    for record in events {
        match record.event {
            SessionEvent::TaskStarted {
                task_id,
                objective,
                base_revision,
            } if Some(task_id) == work.task_id => {
                work.objective = Some(objective);
                work.base_revision = Some(base_revision);
                work.changed_paths.clear();
                work.evidence.clear();
                work.legacy_evidence.clear();
            }
            SessionEvent::UserMessage { task_id, content }
            | SessionEvent::SteeringApplied { task_id, content }
                if Some(task_id) == work.task_id =>
            {
                work.latest_user_message = Some(content)
            }
            SessionEvent::WorkspaceRevisionChanged {
                task_id,
                added,
                modified,
                deleted,
                ..
            } if Some(task_id) == work.task_id => work.changed_paths.extend(
                added
                    .into_iter()
                    .chain(modified)
                    .chain(deleted)
                    .map(PathBuf::from),
            ),
            SessionEvent::WorkspaceDriftDetected { task_id, delta, .. }
                if task_id == work.task_id =>
            {
                work.changed_paths.extend(
                    delta
                        .added
                        .into_iter()
                        .chain(delta.modified)
                        .chain(delta.deleted),
                )
            }
            SessionEvent::PtcCompleted {
                task_id,
                execution_id,
                value,
                tool_calls,
                duration_ms,
                end_revision,
            } if Some(task_id) == work.task_id => {
                work.latest_ptc = Some(PtcSummary {
                    ok: true,
                    value,
                    error: None,
                    tool_calls,
                    duration_ms,
                    revision: end_revision,
                    timestamp_ms: record.timestamp_ms,
                    task_id,
                    execution_id,
                })
            }
            SessionEvent::PtcFailed {
                task_id,
                execution_id,
                error,
                value,
                tool_calls,
                duration_ms,
                end_revision,
            } if Some(task_id) == work.task_id => {
                work.latest_failure = Some(PtcSummary {
                    ok: false,
                    value,
                    error: Some(error),
                    tool_calls,
                    duration_ms,
                    revision: end_revision,
                    timestamp_ms: record.timestamp_ms,
                    task_id,
                    execution_id,
                })
            }
            SessionEvent::EvidenceRecorded { evidence }
                if Some(evidence.task_id) == work.task_id =>
            {
                work.evidence.push(evidence)
            }
            SessionEvent::Legacy {
                ref event_type,
                ref payload,
            } if event_type == "evidence_recorded" => {
                if let Some(v) = payload.get("evidence")
                    && let Ok(e) = serde_json::from_value(v.clone())
                {
                    work.legacy_evidence.push(e);
                }
            }
            _ => {}
        }
    }
    work.changed_paths.sort();
    work.changed_paths.dedup();
    work
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
        SessionEvent::TaskInterrupted { task_id } => {
            if state.active_task == Some(*task_id) {
                state.active_task = None;
            }
        }
        SessionEvent::WorkspaceRevisionChanged { to, .. } => state.current_revision = to.clone(),
        SessionEvent::WorkspaceDriftDetected { actual, .. } => {
            state.current_revision = actual.clone()
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
