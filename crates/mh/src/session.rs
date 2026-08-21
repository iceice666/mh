//! Durable append-only session store (spec §21–§22).

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ptc::{PtcOutcome, PtcResult};
use crate::tools::ResultStore;

const SESSION_DIR: &str = ".mh";
const EVENT_FILE: &str = "session.jsonl";
const STATE_FILE: &str = "state.json";

#[derive(Debug)]
pub enum SessionError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NoSession(PathBuf),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "session I/O error: {error}"),
            Self::Json(error) => write!(f, "session data error: {error}"),
            Self::NoSession(path) => write!(f, "no session found at {}", path.display()),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<std::io::Error> for SessionError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for SessionError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Completed,
    Interrupted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub id: String,
    pub workspace: PathBuf,
    pub task: Option<String>,
    pub status: SessionStatus,
    pub last_event_seq: u64,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    UserMessage {
        content: String,
    },
    AssistantMessage {
        content: String,
    },
    ModelStarted,
    ModelCompleted {
        output_kind: String,
    },
    PtcStarted {
        source: String,
    },
    PtcCompleted {
        value: Value,
        tool_calls: usize,
        duration_ms: u64,
    },
    PtcFailed {
        error: String,
        value: Value,
        tool_calls: usize,
        duration_ms: u64,
    },
    Compacted {
        summary: String,
    },
    Interrupted {
        operation: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    pub seq: u64,
    pub timestamp_ms: u64,
    #[serde(flatten)]
    pub event: SessionEvent,
}

#[derive(Debug)]
pub struct Session {
    root: PathBuf,
    state: SessionState,
    events: Vec<EventRecord>,
}

impl Session {
    /// Opens the workspace session, creating it when absent.
    pub fn open(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = absolute(workspace.as_ref())?;
        let root = workspace.join(SESSION_DIR);
        if root.join(STATE_FILE).exists() {
            return Self::load_root(root);
        }
        std::fs::create_dir_all(root.join("blobs"))?;
        let now = now_ms();
        let state = SessionState {
            id: format!("{now:x}-{:x}", std::process::id()),
            workspace,
            task: None,
            status: SessionStatus::Active,
            last_event_seq: 0,
            created_at_ms: now,
            updated_at_ms: now,
        };
        let session = Self {
            root,
            state,
            events: Vec::new(),
        };
        session.write_state()?;
        Ok(session)
    }

    /// Loads an existing workspace session without creating one.
    pub fn resume(workspace: impl AsRef<Path>) -> Result<Self, SessionError> {
        let workspace = absolute(workspace.as_ref())?;
        let root = workspace.join(SESSION_DIR);
        if !root.join(STATE_FILE).exists() {
            return Err(SessionError::NoSession(root));
        }
        Self::load_root(root)
    }

    fn load_root(root: PathBuf) -> Result<Self, SessionError> {
        let state: SessionState = serde_json::from_slice(&std::fs::read(root.join(STATE_FILE))?)?;
        let events = read_events(&root.join(EVENT_FILE))?;
        let mut session = Self {
            root,
            state,
            events,
        };
        // JSONL is authoritative at safe append boundaries. A stale
        // state file after a crash is repaired on open.
        session.state.last_event_seq = session.events.last().map_or(0, |record| record.seq);
        session.write_state()?;
        Ok(session)
    }

    pub fn append(&mut self, event: SessionEvent) -> Result<&EventRecord, SessionError> {
        let record = EventRecord {
            seq: self.state.last_event_seq.saturating_add(1),
            timestamp_ms: now_ms(),
            event,
        };
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join(EVENT_FILE))?;
        serde_json::to_writer(&mut file, &record)?;
        file.write_all(b"\n")?;
        file.sync_data()?;

        self.state.last_event_seq = record.seq;
        self.state.updated_at_ms = record.timestamp_ms;
        self.events.push(record);
        self.write_state()?;
        Ok(self.events.last().expect("event was just appended"))
    }

    pub fn append_ptc_result(&mut self, result: &PtcResult) -> Result<(), SessionError> {
        match &result.outcome {
            PtcOutcome::Completed => {
                self.append(SessionEvent::PtcCompleted {
                    value: result.value.clone(),
                    tool_calls: result.tool_calls,
                    duration_ms: result.duration_ms,
                })?;
            }
            PtcOutcome::Interrupted => {
                self.append(SessionEvent::Interrupted {
                    operation: "ptc".to_string(),
                })?;
                self.state.status = SessionStatus::Interrupted;
                self.write_state()?;
            }
            PtcOutcome::BudgetExceeded(kind) => {
                self.append(SessionEvent::PtcFailed {
                    error: format!("budget exceeded: {kind}"),
                    value: result.value.clone(),
                    tool_calls: result.tool_calls,
                    duration_ms: result.duration_ms,
                })?;
            }
            PtcOutcome::Failed(error) => {
                self.append(SessionEvent::PtcFailed {
                    error: error.clone(),
                    value: result.value.clone(),
                    tool_calls: result.tool_calls,
                    duration_ms: result.duration_ms,
                })?;
            }
        }
        Ok(())
    }

    pub fn begin_task(&mut self, task: &str) -> Result<(), SessionError> {
        if self.state.task.is_none() {
            self.state.task = Some(task.to_string());
        }
        self.state.status = SessionStatus::Active;
        self.write_state()
    }

    pub fn mark_completed(&mut self) -> Result<(), SessionError> {
        self.state.status = SessionStatus::Completed;
        self.write_state()
    }

    pub fn mark_interrupted(&mut self) -> Result<(), SessionError> {
        self.state.status = SessionStatus::Interrupted;
        self.write_state()
    }

    pub fn events(&self) -> &[EventRecord] {
        &self.events
    }

    pub fn state(&self) -> &SessionState {
        &self.state
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn result_store(&self) -> Result<ResultStore, SessionError> {
        ResultStore::persistent(self.root.join("blobs")).map_err(SessionError::Io)
    }

    fn write_state(&self) -> Result<(), SessionError> {
        std::fs::create_dir_all(&self.root)?;
        let temp = self.root.join(format!(".{STATE_FILE}.tmp"));
        let mut file = File::create(&temp)?;
        serde_json::to_writer_pretty(&mut file, &self.state)?;
        file.write_all(b"\n")?;
        file.sync_data()?;
        std::fs::rename(temp, self.root.join(STATE_FILE))?;
        Ok(())
    }
}

fn read_events(path: &Path) -> Result<Vec<EventRecord>, SessionError> {
    let file = match OpenOptions::new().read(true).write(true).open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut reader = BufReader::new(file.try_clone()?);
    let mut events = Vec::new();
    let mut valid_len = 0u64;
    let mut position = 0u64;
    loop {
        let mut line = Vec::new();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        position = position.saturating_add(read as u64);
        if !line.ends_with(b"\n") {
            break;
        }
        while matches!(line.last(), Some(b'\n' | b'\r')) {
            line.pop();
        }
        if line.is_empty() {
            valid_len = position;
            continue;
        }
        match serde_json::from_slice::<EventRecord>(&line) {
            Ok(record) => {
                events.push(record);
                valid_len = position;
            }
            Err(_) => break,
        }
    }
    if file.metadata()?.len() != valid_len {
        file.set_len(valid_len)?;
        file.sync_data()?;
    }
    Ok(events)
}

fn absolute(path: &Path) -> Result<PathBuf, std::io::Error> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
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

    #[test]
    fn persists_and_resumes_conversation() {
        let dir = tempfile::tempdir().unwrap();
        let id = {
            let mut session = Session::open(dir.path()).unwrap();
            session.begin_task("fix it").unwrap();
            session
                .append(SessionEvent::UserMessage {
                    content: "fix it".to_string(),
                })
                .unwrap();
            session.state().id.clone()
        };
        let resumed = Session::resume(dir.path()).unwrap();
        assert_eq!(resumed.state().id, id);
        assert_eq!(resumed.state().task.as_deref(), Some("fix it"));
        assert_eq!(resumed.events().len(), 1);
    }

    #[test]
    fn ignores_partial_trailing_event() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = Session::open(dir.path()).unwrap();
        session
            .append(SessionEvent::UserMessage {
                content: "safe".to_string(),
            })
            .unwrap();
        let mut file = OpenOptions::new()
            .append(true)
            .open(session.root().join(EVENT_FILE))
            .unwrap();
        file.write_all(b"{\"seq\":2").unwrap();
        drop(file);
        let resumed = Session::resume(dir.path()).unwrap();
        assert_eq!(resumed.events().len(), 1);
        assert_eq!(resumed.state().last_event_seq, 1);
        drop(resumed);
        let mut reopened = Session::resume(dir.path()).unwrap();
        reopened
            .append(SessionEvent::AssistantMessage {
                content: "after recovery".to_string(),
            })
            .unwrap();
        assert_eq!(Session::resume(dir.path()).unwrap().events().len(), 2);
    }
}
