//! Durable background process runtime (spec §5, Phase 4).
//!
//! [`crate::tools::fs_tools::exec`] runs a foreground command to completion;
//! this module owns children whose lifetime deliberately outlives the PTC
//! execution that started them. Every state transition is persisted so a
//! restarted runtime enumerates what the previous one left behind instead of
//! silently forgetting it — and never pretends such a process is still ours.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::identity::{AgentId, ProcessId, TaskId};
use crate::tools::capability::{Capabilities, ProcessPolicy};

/// Retained tail per stream. Total byte counts stay exact; only the retained
/// window is bounded, so `tail` is cheap regardless of how chatty a child is.
const TAIL_CAP: usize = 256 * 1024;
/// Simultaneously live processes allowed per manager.
const DEFAULT_MAX_LIVE: usize = 8;
/// Reap poll interval; `wait` sleeps here rather than holding the map lock.
const REAP_INTERVAL: Duration = Duration::from_millis(10);
/// Bound on waiting for reader threads to observe EOF once the child exited.
/// A grandchild holding the inherited pipe open must not wedge a reap.
const EOF_GRACE: Duration = Duration::from_millis(250);
/// SIGKILL is not catchable, so reaping after `kill` converges; the deadline
/// only guards against an unexpected uninterruptible state.
const KILL_REAP_TIMEOUT_MS: u64 = 5_000;
/// Never handed to a child, matching `exec`'s scrubbing.
const SECRET_ENV: [&str; 2] = ["MH_API_KEY", "OPENAI_API_KEY"];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessSpec {
    pub command: Vec<String>,
    /// Workspace-relative; resolved through [`Capabilities`].
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessState {
    Running,
    Exited,
    Killed,
    Failed,
    /// Recorded by a previous runtime; this process is no longer ours to reap.
    Orphaned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessStream {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessSnapshot {
    pub id: ProcessId,
    pub owner_task: TaskId,
    pub owner_agent: AgentId,
    pub argv: Vec<String>,
    pub cwd: PathBuf,
    pub label: Option<String>,
    pub pid: Option<u32>,
    pub state: ProcessState,
    pub started_at_ms: u64,
    pub exited_at_ms: Option<u64>,
    pub exit_code: Option<i64>,
    pub stdout_bytes: u64,
    pub stderr_bytes: u64,
    /// Set for Orphaned entries recovered after a runtime restart: whether the
    /// recorded OS pid still appears to exist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid_alive: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessTail {
    pub id: ProcessId,
    pub stream: ProcessStream,
    pub content: String,
    pub total_lines: usize,
    pub truncated: bool,
}

#[derive(Debug)]
pub enum ProcessError {
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    Disabled,
    InvalidSpec(String),
    Sandbox(String),
    NotFound(ProcessId),
    Budget {
        limit: usize,
    },
    NotRunning(ProcessId),
    Timeout {
        id: ProcessId,
        timeout_ms: u64,
    },
    Cancelled,
}

impl fmt::Display for ProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(f, "{operation} failed at {}: {source}", path.display()),
            Self::Disabled => write!(f, "process_spawn: disabled by policy"),
            Self::InvalidSpec(message) => write!(f, "process_spawn: {message}"),
            Self::Sandbox(path) => write!(f, "process cwd escapes workspace: {path}"),
            Self::NotFound(id) => write!(f, "process {} was not found", id.0),
            Self::Budget { limit } => {
                write!(f, "process_spawn: live process budget exhausted ({limit})")
            }
            Self::NotRunning(id) => write!(f, "process {} is not running", id.0),
            Self::Timeout { id, timeout_ms } => {
                write!(f, "process {} did not exit within {timeout_ms}ms", id.0)
            }
            Self::Cancelled => write!(f, "process wait was cancelled"),
        }
    }
}

impl std::error::Error for ProcessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Bounded tail plus an exact byte counter, shared with the reader thread.
struct StreamBuffer {
    cap: usize,
    tail: Mutex<VecDeque<u8>>,
    total: AtomicU64,
    /// Reader thread observed EOF; the retained tail is final.
    done: AtomicBool,
}

impl StreamBuffer {
    fn new(cap: usize) -> Arc<Self> {
        Arc::new(Self {
            cap,
            tail: Mutex::new(VecDeque::new()),
            total: AtomicU64::new(0),
            done: AtomicBool::new(false),
        })
    }

    fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Retained bytes plus whether anything was dropped off the head.
    fn retained(&self) -> (Vec<u8>, bool) {
        let tail = self.tail.lock().expect("process tail poisoned");
        let bytes: Vec<u8> = tail.iter().copied().collect();
        drop(tail);
        let dropped = self.total() > bytes.len() as u64;
        (bytes, dropped)
    }
}

struct Entry {
    snap: ProcessSnapshot,
    /// `Some` until reaped; `None` implies a terminal or recovered state.
    child: Option<Child>,
    /// Behind its own lock so a stdin write never holds the map lock.
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    out: Option<Arc<StreamBuffer>>,
    err: Option<Arc<StreamBuffer>>,
    /// We delivered the signal, so exit means Killed rather than Exited.
    killed: bool,
}

impl Entry {
    fn stream(&self, stream: ProcessStream) -> Option<&Arc<StreamBuffer>> {
        match stream {
            ProcessStream::Stdout => self.out.as_ref(),
            ProcessStream::Stderr => self.err.as_ref(),
        }
    }

    fn sync_counters(&mut self) {
        if let Some(out) = &self.out {
            self.snap.stdout_bytes = out.total();
        }
        if let Some(err) = &self.err {
            self.snap.stderr_bytes = err.total();
        }
    }

    fn live_buffers(&self) -> Vec<Arc<StreamBuffer>> {
        [self.out.clone(), self.err.clone()]
            .into_iter()
            .flatten()
            .collect()
    }

    /// Records a terminal state exactly once; the first observer wins.
    fn finish(&mut self, exit_code: Option<i64>, failed: bool) {
        if self.snap.state != ProcessState::Running {
            return;
        }
        self.snap.state = if failed {
            ProcessState::Failed
        } else if self.killed {
            ProcessState::Killed
        } else {
            ProcessState::Exited
        };
        self.snap.exit_code = exit_code;
        self.snap.exited_at_ms = Some(now_ms());
        self.child = None;
        *self.stdin.lock().expect("process stdin poisoned") = None;
        self.sync_counters();
    }
}

struct Inner {
    dir: PathBuf,
    max_live: AtomicUsize,
    procs: Mutex<BTreeMap<u64, Entry>>,
}

/// Handle onto the session's background processes. Cheap to clone and shared
/// across agent threads.
#[derive(Clone)]
pub struct ProcessManager {
    inner: Arc<Inner>,
}

impl ProcessManager {
    /// `root` is the session directory (`<workspace>/.mh`); process metadata
    /// and logs live in `<root>/processes`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, ProcessError> {
        let dir = root.as_ref().join("processes");
        fs::create_dir_all(&dir).map_err(|source| ProcessError::Io {
            operation: "create process directory",
            path: dir.clone(),
            source,
        })?;
        Ok(Self {
            inner: Arc::new(Inner {
                dir,
                max_live: AtomicUsize::new(DEFAULT_MAX_LIVE),
                procs: Mutex::new(BTreeMap::new()),
            }),
        })
    }

    /// Maximum simultaneously live processes; default 8.
    pub fn with_max_live(self, max_live: usize) -> Self {
        self.inner.max_live.store(max_live, Ordering::Relaxed);
        self
    }

    /// Spawns a long-lived process and returns immediately; it is never waited
    /// on here. `id` is allocated by the caller (session journal).
    pub fn spawn(
        &self,
        id: ProcessId,
        owner_task: TaskId,
        owner_agent: AgentId,
        caps: &Capabilities,
        spec: ProcessSpec,
    ) -> Result<ProcessSnapshot, ProcessError> {
        if caps.process == ProcessPolicy::Disabled {
            return Err(ProcessError::Disabled);
        }
        if spec.command.is_empty() || spec.command[0].is_empty() {
            return Err(ProcessError::InvalidSpec("empty 'command'".into()));
        }
        for key in spec.env.keys() {
            if SECRET_ENV.contains(&key.as_str()) {
                return Err(ProcessError::InvalidSpec(format!(
                    "env key {key} is not assignable"
                )));
            }
        }
        let cwd = match &spec.cwd {
            Some(path) => {
                let resolved = caps
                    .resolve_existing_read(path)
                    .map_err(|error| ProcessError::Sandbox(error.0))?;
                if !resolved.is_dir() {
                    return Err(ProcessError::Sandbox(path.clone()));
                }
                resolved
            }
            None => caps.workspace.clone(),
        };

        let snap = self.spawn_locked(id, owner_task, owner_agent, cwd, spec)?;
        if let Err(error) = self.persist(&snap) {
            // An unrecorded live process is worse than no process at all.
            let _ = self.kill(id);
            return Err(error);
        }
        Ok(snap)
    }

    /// Budget check, `Command::spawn` and registration under one lock hold so
    /// concurrent spawns cannot both win the last budget slot.
    fn spawn_locked(
        &self,
        id: ProcessId,
        owner_task: TaskId,
        owner_agent: AgentId,
        cwd: PathBuf,
        spec: ProcessSpec,
    ) -> Result<ProcessSnapshot, ProcessError> {
        let mut guard = self.lock();
        if guard.contains_key(&id.0) {
            return Err(ProcessError::InvalidSpec(format!(
                "process id {} already exists",
                id.0
            )));
        }
        // Cheap, non-blocking refresh so children that already exited do not
        // hold budget slots hostage.
        let mut live = 0usize;
        for entry in guard.values_mut() {
            if entry.snap.state != ProcessState::Running {
                continue;
            }
            match entry.child.as_mut().map(Child::try_wait) {
                Some(Ok(Some(status))) => entry.finish(exit_code(&status), false),
                Some(Err(_)) => entry.finish(None, true),
                _ => live += 1,
            }
        }
        let limit = self.inner.max_live.load(Ordering::Relaxed);
        if live >= limit {
            return Err(ProcessError::Budget { limit });
        }

        let out_log = self.create_log(id, ProcessStream::Stdout)?;
        let err_log = self.create_log(id, ProcessStream::Stderr)?;

        let mut command = Command::new(&spec.command[0]);
        command
            .args(&spec.command[1..])
            .current_dir(&cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in &spec.env {
            command.env(key, value);
        }
        for key in SECRET_ENV {
            command.env_remove(key);
        }
        let mut child = command.spawn().map_err(|source| ProcessError::Io {
            operation: "spawn process",
            path: PathBuf::from(&spec.command[0]),
            source,
        })?;

        let out = StreamBuffer::new(TAIL_CAP);
        let err = StreamBuffer::new(TAIL_CAP);
        pump(child.stdout.take(), out_log, Arc::clone(&out));
        pump(child.stderr.take(), err_log, Arc::clone(&err));

        let entry = Entry {
            snap: ProcessSnapshot {
                id,
                owner_task,
                owner_agent,
                argv: spec.command,
                cwd,
                label: spec.label,
                pid: Some(child.id()),
                state: ProcessState::Running,
                started_at_ms: now_ms(),
                exited_at_ms: None,
                exit_code: None,
                stdout_bytes: 0,
                stderr_bytes: 0,
                pid_alive: None,
            },
            stdin: Arc::new(Mutex::new(child.stdin.take())),
            child: Some(child),
            out: Some(out),
            err: Some(err),
            killed: false,
        };
        let snap = entry.snap.clone();
        guard.insert(id.0, entry);
        Ok(snap)
    }

    /// Observes exit without anyone having called [`Self::wait`].
    pub fn poll(&self, id: ProcessId) -> Result<ProcessSnapshot, ProcessError> {
        self.reap(id)
    }

    pub fn tail(
        &self,
        id: ProcessId,
        stream: ProcessStream,
        lines: usize,
    ) -> Result<ProcessTail, ProcessError> {
        let buffer = {
            let guard = self.lock();
            let entry = guard.get(&id.0).ok_or(ProcessError::NotFound(id))?;
            entry.stream(stream).map(Arc::clone)
        };
        // Recovered entries have no in-memory tail; their log file is the only
        // remaining evidence.
        let (bytes, dropped) = match buffer {
            Some(buffer) => buffer.retained(),
            None => {
                let path = self.log_path(id, stream);
                let (bytes, total) = read_log_tail(&path, TAIL_CAP)?;
                let dropped = total > bytes.len() as u64;
                (bytes, dropped)
            }
        };
        let text = String::from_utf8_lossy(&bytes);
        let all: Vec<&str> = text.lines().collect();
        let total_lines = all.len();
        let start = total_lines.saturating_sub(lines);
        Ok(ProcessTail {
            id,
            stream,
            content: all[start..].join("\n"),
            total_lines,
            truncated: dropped || start > 0,
        })
    }

    /// Blocks until exit, `timeout_ms` elapses (`Err(Timeout)`), or `cancelled`
    /// flips (`Err(Cancelled)`).
    pub fn wait(
        &self,
        id: ProcessId,
        timeout_ms: Option<u64>,
        cancelled: &AtomicBool,
    ) -> Result<ProcessSnapshot, ProcessError> {
        let deadline = timeout_ms.map(|ms| Instant::now() + Duration::from_millis(ms));
        loop {
            let snap = self.reap(id)?;
            if snap.state != ProcessState::Running {
                return Ok(snap);
            }
            if cancelled.load(Ordering::Relaxed) {
                return Err(ProcessError::Cancelled);
            }
            if let Some(deadline) = deadline
                && Instant::now() >= deadline
            {
                return Err(ProcessError::Timeout {
                    id,
                    timeout_ms: timeout_ms.unwrap_or_default(),
                });
            }
            std::thread::sleep(REAP_INTERVAL);
        }
    }

    /// Terminates and reaps the child we spawned. Idempotent: killing an
    /// already-exited process returns its final snapshot. Orphaned entries are
    /// left alone — their pid is no longer ours to signal.
    pub fn kill(&self, id: ProcessId) -> Result<ProcessSnapshot, ProcessError> {
        {
            let mut guard = self.lock();
            let entry = guard.get_mut(&id.0).ok_or(ProcessError::NotFound(id))?;
            if entry.child.is_none() {
                return Ok(entry.snap.clone());
            }
            *entry.stdin.lock().expect("process stdin poisoned") = None;
            if let Some(child) = entry.child.as_mut()
                && matches!(child.try_wait(), Ok(None))
            {
                let _ = child.kill();
                entry.killed = true;
            }
        }
        self.wait(id, Some(KILL_REAP_TIMEOUT_MS), &AtomicBool::new(false))
            .or_else(|_| self.poll(id))
    }

    pub fn write_stdin(&self, id: ProcessId, data: &str) -> Result<(), ProcessError> {
        if self.reap(id)?.state != ProcessState::Running {
            return Err(ProcessError::NotRunning(id));
        }
        let slot = {
            let guard = self.lock();
            let entry = guard.get(&id.0).ok_or(ProcessError::NotFound(id))?;
            Arc::clone(&entry.stdin)
        };
        // Written outside the map lock: a full pipe blocks until the child reads.
        let mut held = slot.lock().expect("process stdin poisoned");
        let stdin = held.as_mut().ok_or(ProcessError::NotRunning(id))?;
        let write = if data.ends_with('\n') {
            stdin.write_all(data.as_bytes())
        } else {
            stdin
                .write_all(data.as_bytes())
                .and_then(|()| stdin.write_all(b"\n"))
        };
        write
            .and_then(|()| stdin.flush())
            .map_err(|source| ProcessError::Io {
                operation: "write process stdin",
                path: self.meta_path(id),
                source,
            })
    }

    pub fn list(&self) -> Vec<ProcessSnapshot> {
        let running: Vec<ProcessId> = self
            .lock()
            .values()
            .filter(|entry| entry.snap.state == ProcessState::Running)
            .map(|entry| entry.snap.id)
            .collect();
        for id in running {
            let _ = self.poll(id);
        }
        self.lock()
            .values()
            .map(|entry| entry.snap.clone())
            .collect()
    }

    /// Kills every live process owned by `agent`; used when an agent execution
    /// ends.
    pub fn kill_agent(&self, agent: AgentId) -> Vec<ProcessSnapshot> {
        self.kill_owned(|snap| snap.owner_agent == agent)
    }

    /// Kills every live process owned by `task`.
    pub fn kill_task(&self, task: TaskId) -> Vec<ProcessSnapshot> {
        self.kill_owned(|snap| snap.owner_task == task)
    }

    /// Loads metadata written by a previous runtime and marks every entry that
    /// was still `Running` as `Orphaned`, with `pid_alive` filled in from a
    /// liveness probe. Never pretends a process survived as ours.
    pub fn recover(&self) -> Vec<ProcessSnapshot> {
        let mut recovered = Vec::new();
        let Ok(items) = fs::read_dir(&self.inner.dir) else {
            return recovered;
        };
        let mut paths: Vec<PathBuf> = items
            .filter_map(Result::ok)
            .map(|item| item.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
            .collect();
        paths.sort();

        for path in paths {
            let Ok(bytes) = fs::read(&path) else { continue };
            let Ok(mut snap) = serde_json::from_slice::<ProcessSnapshot>(&bytes) else {
                continue;
            };
            // A live in-memory entry is authoritative: this runtime owns it.
            if self.lock().contains_key(&snap.id.0) {
                continue;
            }
            let stale = snap.state == ProcessState::Running;
            if stale {
                snap.state = ProcessState::Orphaned;
                snap.pid_alive = match snap.pid {
                    Some(pid) => probe_pid(pid),
                    None => Some(false),
                };
            }
            let entry = Entry {
                snap: snap.clone(),
                child: None,
                stdin: Arc::new(Mutex::new(None)),
                out: None,
                err: None,
                killed: false,
            };
            self.lock().insert(snap.id.0, entry);
            if stale {
                let _ = self.persist(&snap);
            }
            recovered.push(snap);
        }
        recovered
    }

    // -- internals ----------------------------------------------------------

    fn lock(&self) -> MutexGuard<'_, BTreeMap<u64, Entry>> {
        self.inner.procs.lock().expect("process manager poisoned")
    }

    fn kill_owned(&self, owned: impl Fn(&ProcessSnapshot) -> bool) -> Vec<ProcessSnapshot> {
        let targets: Vec<ProcessId> = self
            .lock()
            .values()
            .filter(|entry| entry.snap.state == ProcessState::Running && owned(&entry.snap))
            .map(|entry| entry.snap.id)
            .collect();
        targets
            .into_iter()
            .filter_map(|id| self.kill(id).ok())
            .collect()
    }

    /// Non-blocking exit observation. The map lock is released before waiting
    /// on reader threads, and is never held across a blocking child wait.
    fn reap(&self, id: ProcessId) -> Result<ProcessSnapshot, ProcessError> {
        let (outcome, buffers) = {
            let mut guard = self.lock();
            let entry = guard.get_mut(&id.0).ok_or(ProcessError::NotFound(id))?;
            match entry.child.as_mut().map(Child::try_wait) {
                None => return Ok(entry.snap.clone()),
                Some(Ok(None)) => {
                    entry.sync_counters();
                    return Ok(entry.snap.clone());
                }
                Some(Ok(Some(status))) => ((exit_code(&status), false), entry.live_buffers()),
                Some(Err(_)) => ((None, true), entry.live_buffers()),
            }
        };
        drain_streams(&buffers);

        let mut guard = self.lock();
        let entry = guard.get_mut(&id.0).ok_or(ProcessError::NotFound(id))?;
        entry.finish(outcome.0, outcome.1);
        let snap = entry.snap.clone();
        drop(guard);
        self.persist(&snap)?;
        Ok(snap)
    }

    fn meta_path(&self, id: ProcessId) -> PathBuf {
        self.inner.dir.join(format!("{}.json", id.0))
    }

    fn log_path(&self, id: ProcessId, stream: ProcessStream) -> PathBuf {
        let ext = match stream {
            ProcessStream::Stdout => "out",
            ProcessStream::Stderr => "err",
        };
        self.inner.dir.join(format!("{}.{ext}", id.0))
    }

    fn create_log(&self, id: ProcessId, stream: ProcessStream) -> Result<fs::File, ProcessError> {
        let path = self.log_path(id, stream);
        fs::File::create(&path).map_err(|source| ProcessError::Io {
            operation: "create process log",
            path,
            source,
        })
    }

    /// Atomic temp-then-rename so a crash mid-write cannot leave a torn record.
    fn persist(&self, snap: &ProcessSnapshot) -> Result<(), ProcessError> {
        let path = self.meta_path(snap.id);
        let bytes = serde_json::to_vec(snap).map_err(|error| ProcessError::Io {
            operation: "encode process metadata",
            path: path.clone(),
            source: io::Error::other(error),
        })?;
        let temp = path.with_extension("json.tmp");
        let mut file = fs::File::create(&temp).map_err(|source| ProcessError::Io {
            operation: "create process metadata",
            path: temp.clone(),
            source,
        })?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_data())
            .map_err(|source| ProcessError::Io {
                operation: "write process metadata",
                path: temp.clone(),
                source,
            })?;
        fs::rename(&temp, &path).map_err(|source| ProcessError::Io {
            operation: "replace process metadata",
            path,
            source,
        })
    }
}

/// Drains one pipe into its log file and bounded tail until EOF.
fn pump(pipe: Option<impl Read + Send + 'static>, mut log: fs::File, buffer: Arc<StreamBuffer>) {
    let Some(mut pipe) = pipe else {
        buffer.done.store(true, Ordering::Release);
        return;
    };
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        while let Ok(read) = pipe.read(&mut chunk) {
            if read == 0 {
                break;
            }
            let bytes = &chunk[..read];
            let _ = log.write_all(bytes);
            buffer.total.fetch_add(read as u64, Ordering::Relaxed);
            if buffer.cap == 0 {
                continue;
            }
            let mut tail = buffer.tail.lock().expect("process tail poisoned");
            for byte in bytes {
                if tail.len() == buffer.cap {
                    tail.pop_front();
                }
                tail.push_back(*byte);
            }
        }
        buffer.done.store(true, Ordering::Release);
    });
}

/// Waits, bounded, for reader threads to reach EOF so a terminal snapshot
/// reports the child's complete output.
fn drain_streams(buffers: &[Arc<StreamBuffer>]) {
    let deadline = Instant::now() + EOF_GRACE;
    while buffers
        .iter()
        .any(|buffer| !buffer.done.load(Ordering::Acquire))
    {
        if Instant::now() >= deadline {
            return;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn read_log_tail(path: &Path, cap: usize) -> Result<(Vec<u8>, u64), ProcessError> {
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(source) => {
            return Err(ProcessError::Io {
                operation: "open process log",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let total = file.metadata().map(|meta| meta.len()).unwrap_or_default();
    let start = total.saturating_sub(cap as u64);
    if start > 0 {
        file.seek(SeekFrom::Start(start))
            .map_err(|source| ProcessError::Io {
                operation: "seek process log",
                path: path.to_path_buf(),
                source,
            })?;
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|source| ProcessError::Io {
            operation: "read process log",
            path: path.to_path_buf(),
            source,
        })?;
    Ok((bytes, total))
}

/// `None` for a signalled child: it has no exit code to report.
fn exit_code(status: &std::process::ExitStatus) -> Option<i64> {
    status.code().map(i64::from)
}

/// Liveness probe for a pid this runtime does not own. `kill -0` reports
/// success only for a signallable process, so a foreign-uid pid reads as gone.
#[cfg(unix)]
fn probe_pid(pid: u32) -> Option<bool> {
    Some(
        Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success()),
    )
}

#[cfg(not(unix))]
fn probe_pid(_pid: u32) -> Option<bool> {
    None
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TASK: TaskId = TaskId(1);

    fn fixture() -> (ProcessManager, Capabilities, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let caps = Capabilities::new(dir.path().to_path_buf());
        let manager = ProcessManager::open(caps.workspace.join(".mh")).unwrap();
        (manager, caps, dir)
    }

    fn spec(script: &str) -> ProcessSpec {
        ProcessSpec {
            command: vec!["/bin/sh".into(), "-c".into(), script.into()],
            cwd: None,
            env: BTreeMap::new(),
            label: None,
        }
    }

    #[test]
    fn spawn_returns_immediately_and_kill_terminates() {
        let (manager, caps, _dir) = fixture();
        let start = Instant::now();
        let snap = manager
            .spawn(ProcessId(1), TASK, AgentId::ROOT, &caps, spec("sleep 5"))
            .unwrap();
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(snap.state, ProcessState::Running);
        let pid = snap.pid.expect("pid recorded");

        let killed = manager.kill(ProcessId(1)).unwrap();
        assert_ne!(killed.state, ProcessState::Running);
        assert!(killed.exited_at_ms.is_some());
        assert_eq!(probe_pid(pid), Some(false));
    }

    #[test]
    fn output_is_tailable_after_exit() {
        let (manager, caps, _dir) = fixture();
        manager
            .spawn(
                ProcessId(7),
                TASK,
                AgentId::ROOT,
                &caps,
                spec("printf 'a\nb\nc\n'"),
            )
            .unwrap();
        let done = manager
            .wait(ProcessId(7), None, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(done.state, ProcessState::Exited);
        assert_eq!(done.exit_code, Some(0));
        assert_eq!(done.stdout_bytes, 6);

        let tail = manager
            .tail(ProcessId(7), ProcessStream::Stdout, 2)
            .unwrap();
        assert_eq!(tail.content, "b\nc");
        assert_eq!(tail.total_lines, 3);
        assert!(tail.truncated);
    }

    #[test]
    fn spawn_never_blocks_on_a_long_child() {
        let (manager, caps, _dir) = fixture();
        let start = Instant::now();
        let snap = manager
            .spawn(ProcessId(2), TASK, AgentId::ROOT, &caps, spec("sleep 3"))
            .unwrap();
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(snap.state, ProcessState::Running);
        assert_eq!(
            manager.poll(ProcessId(2)).unwrap().state,
            ProcessState::Running
        );
        manager.kill(ProcessId(2)).unwrap();
    }

    #[test]
    fn wait_times_out_without_disturbing_the_child() {
        let (manager, caps, _dir) = fixture();
        manager
            .spawn(ProcessId(3), TASK, AgentId::ROOT, &caps, spec("sleep 5"))
            .unwrap();
        let error = manager
            .wait(ProcessId(3), Some(50), &AtomicBool::new(false))
            .expect_err("must time out");
        assert!(matches!(
            error,
            ProcessError::Timeout {
                id: ProcessId(3),
                timeout_ms: 50
            }
        ));
        assert_eq!(
            manager.poll(ProcessId(3)).unwrap().state,
            ProcessState::Running
        );
        assert_ne!(
            manager.kill(ProcessId(3)).unwrap().state,
            ProcessState::Running
        );
    }

    #[test]
    fn recover_marks_previous_runtime_processes_orphaned() {
        let (manager, caps, _dir) = fixture();
        let root = caps.workspace.join(".mh");
        manager
            .spawn(ProcessId(4), TASK, AgentId::ROOT, &caps, spec("sleep 5"))
            .unwrap();

        let restarted = ProcessManager::open(&root).unwrap();
        let recovered = restarted.recover();
        assert!(recovered.iter().all(|s| s.state != ProcessState::Running));
        let entry = recovered
            .iter()
            .find(|s| s.id == ProcessId(4))
            .expect("entry recovered");
        assert_eq!(entry.state, ProcessState::Orphaned);
        assert_eq!(entry.pid_alive, Some(true));
        assert_eq!(
            restarted.poll(ProcessId(4)).unwrap().state,
            ProcessState::Orphaned
        );
        assert!(
            restarted
                .list()
                .iter()
                .any(|s| s.id == ProcessId(4) && s.state == ProcessState::Orphaned)
        );

        manager.kill(ProcessId(4)).unwrap();
    }

    #[test]
    fn live_budget_is_enforced() {
        let (manager, caps, _dir) = fixture();
        let manager = manager.with_max_live(1);
        manager
            .spawn(ProcessId(5), TASK, AgentId::ROOT, &caps, spec("sleep 5"))
            .unwrap();
        let error = manager
            .spawn(ProcessId(6), TASK, AgentId::ROOT, &caps, spec("sleep 5"))
            .expect_err("budget exhausted");
        assert!(matches!(error, ProcessError::Budget { limit: 1 }));
        manager.kill(ProcessId(5)).unwrap();
    }

    #[test]
    fn policy_and_sandbox_are_enforced() {
        let (manager, caps, dir) = fixture();
        let denied = Capabilities::read_only(dir.path().to_path_buf());
        assert!(matches!(
            manager.spawn(ProcessId(8), TASK, AgentId::ROOT, &denied, spec("true")),
            Err(ProcessError::Disabled)
        ));

        let mut escaping = spec("true");
        escaping.cwd = Some("../..".into());
        assert!(matches!(
            manager.spawn(ProcessId(9), TASK, AgentId::ROOT, &caps, escaping),
            Err(ProcessError::Sandbox(_))
        ));
    }

    #[test]
    fn stdin_round_trips_and_requires_a_running_child() {
        let (manager, caps, _dir) = fixture();
        manager
            .spawn(
                ProcessId(10),
                TASK,
                AgentId::ROOT,
                &caps,
                spec("read line; printf 'got:%s\\n' \"$line\""),
            )
            .unwrap();
        manager.write_stdin(ProcessId(10), "ping").unwrap();
        let done = manager
            .wait(ProcessId(10), Some(5_000), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(done.exit_code, Some(0));
        let tail = manager
            .tail(ProcessId(10), ProcessStream::Stdout, 1)
            .unwrap();
        assert_eq!(tail.content, "got:ping");
        assert!(matches!(
            manager.write_stdin(ProcessId(10), "again"),
            Err(ProcessError::NotRunning(ProcessId(10)))
        ));
    }

    #[test]
    fn ownership_kills_reap_only_matching_processes() {
        let (manager, caps, _dir) = fixture();
        manager
            .spawn(ProcessId(11), TASK, AgentId(3), &caps, spec("sleep 5"))
            .unwrap();
        manager
            .spawn(ProcessId(12), TaskId(2), AgentId(4), &caps, spec("sleep 5"))
            .unwrap();

        let killed = manager.kill_agent(AgentId(3));
        assert_eq!(killed.len(), 1);
        assert_eq!(killed[0].id, ProcessId(11));
        assert_ne!(
            manager.poll(ProcessId(12)).unwrap().state,
            ProcessState::Exited
        );

        let killed = manager.kill_task(TaskId(2));
        assert_eq!(killed.len(), 1);
        assert_eq!(killed[0].id, ProcessId(12));
        // Already terminal: nothing left to signal.
        assert!(manager.kill_task(TaskId(2)).is_empty());
    }

    #[test]
    fn secret_env_keys_are_rejected() {
        let (manager, caps, _dir) = fixture();
        let mut leaking = spec("true");
        leaking.env.insert("MH_API_KEY".into(), "sk-x".into());
        assert!(matches!(
            manager.spawn(ProcessId(13), TASK, AgentId::ROOT, &caps, leaking),
            Err(ProcessError::InvalidSpec(_))
        ));
    }
}
