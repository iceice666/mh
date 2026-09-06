//! Runtime service layer (spec v0.5 §"Runtime architecture").
//!
//! [`Runtime`] owns everything that outlives a single model call: the session
//! journal, the workspace tracker, the isolation store, the worker scheduler,
//! and the process manager. `Agent<M>` drives model turns; the runtime owns
//! durable state, so a detached run, an inspector, and a restart all see the
//! same thing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use serde_json::Value;

use crate::delegation::{AgentAccess, AgentState, AgentStatus, DelegateResult, DelegationBudget};
use crate::goal::TaskStatus;
use crate::identity::{AgentId, IsolatedWorkspaceId, ProcessId, RevisionId, TaskId};
use crate::isolation::IsolationStore;
use crate::process::{ProcessManager, ProcessSnapshot};
use crate::ptc::prelude::Prelude;
use crate::session::{Session, SessionError, SessionEvent};
use crate::workspace::WorkspaceTracker;

/// Shared durable services for one workspace session.
pub struct Runtime {
    session: Session,
    tracker: Arc<dyn WorkspaceTracker>,
    isolation: Option<IsolationStore>,
    processes: ProcessManager,
    workers: WorkerRegistry,
    prelude: Mutex<Option<Arc<Prelude>>>,
    next_workspace: AtomicU64,
    workers_started: AtomicUsize,
    budget: DelegationBudget,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("workspace", &self.session.workspace_root())
            .finish()
    }
}

impl Runtime {
    pub fn new(session: Session, budget: DelegationBudget) -> Result<Self, SessionError> {
        let tracker = session.tracker();
        let workspace = session.workspace_root();
        let processes = ProcessManager::open(session.root())
            .map_err(|error| SessionError::State(error.to_string()))?
            .with_max_live(MAX_LIVE_PROCESSES);
        let next_workspace = session.next_isolated_workspace_id().0;
        Ok(Self {
            isolation: IsolationStore::open(&workspace).ok(),
            session,
            tracker,
            processes,
            workers: WorkerRegistry::default(),
            prelude: Mutex::new(None),
            next_workspace: AtomicU64::new(next_workspace),
            workers_started: AtomicUsize::new(0),
            budget,
        })
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn tracker(&self) -> Arc<dyn WorkspaceTracker> {
        self.tracker.clone()
    }

    pub fn isolation(&self) -> Option<&IsolationStore> {
        self.isolation.as_ref()
    }

    pub fn processes(&self) -> &ProcessManager {
        &self.processes
    }

    pub fn workers(&self) -> &WorkerRegistry {
        &self.workers
    }

    pub const fn budget(&self) -> &DelegationBudget {
        &self.budget
    }

    pub fn workspace_root(&self) -> PathBuf {
        self.session.workspace_root()
    }

    pub fn set_prelude(&self, prelude: Option<Arc<Prelude>>) {
        *self.prelude.lock().expect("prelude poisoned") = prelude;
    }

    pub fn prelude(&self) -> Option<Arc<Prelude>> {
        self.prelude.lock().expect("prelude poisoned").clone()
    }

    pub fn current_revision(&self) -> Result<RevisionId, SessionError> {
        Ok(self.tracker.current_revision()?.id)
    }

    /// Allocates the next isolated workspace id without racing sibling workers.
    pub fn allocate_workspace(&self) -> IsolatedWorkspaceId {
        IsolatedWorkspaceId(self.next_workspace.fetch_add(1, Ordering::Relaxed))
    }

    /// Reserves one slot against the per-task child budget.
    pub fn reserve_worker_slot(&self) -> Result<(), String> {
        let started = self.workers_started.fetch_add(1, Ordering::Relaxed);
        if started >= self.budget.max_children {
            self.workers_started.fetch_sub(1, Ordering::Relaxed);
            return Err(format!(
                "delegation child budget exceeded ({} children)",
                self.budget.max_children
            ));
        }
        Ok(())
    }

    pub fn release_worker_slot(&self) {
        self.workers_started.fetch_sub(1, Ordering::Relaxed);
    }

    /// Marks processes from a previous runtime as orphaned and records that in
    /// the journal, so a resumed task never believes a dead process is live.
    pub fn recover_processes(&self) -> Vec<ProcessSnapshot> {
        let recovered = self.processes.recover();
        for snapshot in &recovered {
            let _ = self
                .session
                .append_shared_event(SessionEvent::ProcessStateChanged {
                    task_id: snapshot.owner_task,
                    process: snapshot.id,
                    state: process_state_label(snapshot.state),
                    exit_code: snapshot.exit_code,
                    pid_alive: snapshot.pid_alive,
                });
        }
        recovered
    }

    /// Kills every process a task owns and records the transition.
    pub fn shutdown_task_processes(&self, task_id: TaskId) {
        for snapshot in self.processes.kill_task(task_id) {
            let _ = self
                .session
                .append_shared_event(SessionEvent::ProcessStateChanged {
                    task_id,
                    process: snapshot.id,
                    state: process_state_label(snapshot.state),
                    exit_code: snapshot.exit_code,
                    pid_alive: None,
                });
        }
    }

    pub fn record_process_state(&self, task_id: TaskId, snapshot: &ProcessSnapshot) {
        let _ = self
            .session
            .append_shared_event(SessionEvent::ProcessStateChanged {
                task_id,
                process: snapshot.id,
                state: process_state_label(snapshot.state),
                exit_code: snapshot.exit_code,
                pid_alive: snapshot.pid_alive,
            });
    }

    /// Cleans up isolated workspaces belonging to a finished task.
    pub fn cleanup_isolation(&self, task_id: TaskId) {
        if let Some(isolation) = self.isolation.as_ref() {
            let _ = isolation.cleanup_owner(task_id);
        }
    }
}

/// Live processes allowed per session. Long-running must not mean unbounded.
const MAX_LIVE_PROCESSES: usize = 8;

pub fn process_state_label(state: crate::process::ProcessState) -> String {
    match state {
        crate::process::ProcessState::Running => "running",
        crate::process::ProcessState::Exited => "exited",
        crate::process::ProcessState::Killed => "killed",
        crate::process::ProcessState::Failed => "failed",
        crate::process::ProcessState::Orphaned => "orphaned",
    }
    .to_string()
}

/// One delegated worker's live handle.
struct Worker {
    agent: AgentId,
    task_id: TaskId,
    objective: String,
    access: AgentAccess,
    profile: Option<String>,
    /// Cancellation private to this worker: cancelling one must not disturb
    /// its siblings or the root agent.
    cancelled: Arc<AtomicBool>,
    state: AgentState,
    result: Option<DelegateResult>,
    /// Set while the OS thread is joinable.
    handle: Option<std::thread::JoinHandle<()>>,
}

/// Tracks running workers and lets `agent_join` block without holding a lock.
#[derive(Default)]
pub struct WorkerRegistry {
    inner: Mutex<HashMap<u64, Worker>>,
    settled: Condvar,
}

impl WorkerRegistry {
    /// Registers a worker before its thread starts, so `agent_poll` never
    /// observes a spawned-but-unknown agent.
    pub fn register(
        &self,
        agent: AgentId,
        task_id: TaskId,
        objective: String,
        access: AgentAccess,
        profile: Option<String>,
    ) -> Arc<AtomicBool> {
        let cancelled = Arc::new(AtomicBool::new(false));
        self.inner.lock().expect("workers poisoned").insert(
            agent.0,
            Worker {
                agent,
                task_id,
                objective,
                access,
                profile,
                cancelled: cancelled.clone(),
                state: AgentState::Queued,
                result: None,
                handle: None,
            },
        );
        cancelled
    }

    pub fn attach_thread(&self, agent: AgentId, handle: std::thread::JoinHandle<()>) {
        if let Some(worker) = self
            .inner
            .lock()
            .expect("workers poisoned")
            .get_mut(&agent.0)
        {
            worker.handle = Some(handle);
        }
    }

    pub fn set_state(&self, agent: AgentId, state: AgentState) {
        let mut workers = self.inner.lock().expect("workers poisoned");
        if let Some(worker) = workers.get_mut(&agent.0) {
            worker.state = state;
        }
        drop(workers);
        self.settled.notify_all();
    }

    /// Records the worker's outcome and wakes anyone joining it.
    pub fn settle(&self, agent: AgentId, result: DelegateResult) {
        let mut workers = self.inner.lock().expect("workers poisoned");
        if let Some(worker) = workers.get_mut(&agent.0) {
            worker.state = if result.ok {
                AgentState::Completed
            } else if worker.cancelled.load(Ordering::Relaxed) {
                AgentState::Cancelled
            } else {
                AgentState::Failed
            };
            worker.result = Some(result);
        }
        drop(workers);
        self.settled.notify_all();
    }

    pub fn cancel(&self, agent: AgentId) -> Result<(), String> {
        let workers = self.inner.lock().expect("workers poisoned");
        let worker = workers
            .get(&agent.0)
            .ok_or_else(|| format!("unknown agent {}", agent.0))?;
        worker.cancelled.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Cancels every live worker, e.g. on Ctrl-C.
    pub fn cancel_all(&self) {
        for worker in self.inner.lock().expect("workers poisoned").values() {
            worker.cancelled.store(true, Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self, agent: AgentId) -> Option<WorkerSnapshot> {
        self.inner
            .lock()
            .expect("workers poisoned")
            .get(&agent.0)
            .map(Worker::snapshot)
    }

    pub fn list(&self) -> Vec<WorkerSnapshot> {
        let mut out: Vec<WorkerSnapshot> = self
            .inner
            .lock()
            .expect("workers poisoned")
            .values()
            .map(Worker::snapshot)
            .collect();
        out.sort_by_key(|worker| worker.agent.0);
        out
    }

    pub fn is_cancelled(&self, agent: AgentId) -> bool {
        self.inner
            .lock()
            .expect("workers poisoned")
            .get(&agent.0)
            .is_some_and(|worker| worker.cancelled.load(Ordering::Relaxed))
    }

    /// Blocks until the worker settles, then returns its result.
    ///
    /// The condvar wait releases the registry lock, so sibling workers keep
    /// reporting progress while one join is outstanding.
    pub fn join(&self, agent: AgentId) -> Result<DelegateResult, String> {
        let mut workers = self.inner.lock().expect("workers poisoned");
        loop {
            let worker = workers
                .get(&agent.0)
                .ok_or_else(|| format!("unknown agent {}", agent.0))?;
            if let Some(result) = worker.result.clone() {
                // Reap the thread so a joined worker leaks nothing.
                if let Some(handle) = workers
                    .get_mut(&agent.0)
                    .and_then(|worker| worker.handle.take())
                {
                    drop(workers);
                    let _ = handle.join();
                } else {
                    drop(workers);
                }
                return Ok(result);
            }
            workers = self
                .settled
                .wait(workers)
                .expect("worker registry condvar poisoned");
        }
    }

    /// Joins every outstanding worker thread; used when a task ends.
    pub fn join_all(&self) {
        let handles: Vec<_> = self
            .inner
            .lock()
            .expect("workers poisoned")
            .values_mut()
            .filter_map(|worker| worker.handle.take())
            .collect();
        for handle in handles {
            let _ = handle.join();
        }
    }
}

impl Worker {
    fn snapshot(&self) -> WorkerSnapshot {
        WorkerSnapshot {
            agent: self.agent,
            task_id: self.task_id,
            objective: self.objective.clone(),
            access: self.access,
            profile: self.profile.clone(),
            state: self.state,
            result: self.result.clone(),
        }
    }
}

/// Live view of a worker held in memory by this runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSnapshot {
    pub agent: AgentId,
    pub task_id: TaskId,
    pub objective: String,
    pub access: AgentAccess,
    pub profile: Option<String>,
    pub state: AgentState,
    pub result: Option<DelegateResult>,
}

impl WorkerSnapshot {
    /// Combines the live worker with its durable task state into the status a
    /// PTC program sees.
    pub fn status(&self, session: &Session) -> AgentStatus {
        let view = session.task_view(self.task_id);
        AgentStatus {
            agent: self.agent,
            task_id: self.task_id,
            state: self.state,
            objective: self.objective.clone(),
            access: self.access,
            profile: self.profile.clone(),
            done: self.state.is_terminal(),
            window: view.window,
            turns_in_window: view.turns_in_window,
            result: self.result.clone(),
            pending: view
                .goal
                .pending_work
                .iter()
                .map(|item| item.title.clone())
                .collect(),
            next_actions: view.goal.next_actions.clone(),
            current_revision: Some(view.goal.current_revision),
        }
    }
}

/// A durable task as reported by `mh tasks` / `mh inspect`.
#[derive(Debug, Clone)]
pub struct TaskReport {
    pub task_id: TaskId,
    pub agent: AgentId,
    pub objective: String,
    pub status: TaskStatus,
    pub window: u32,
    pub revision: RevisionId,
    pub workers: Vec<(AgentId, AgentState, String)>,
    pub processes: Vec<(ProcessId, String, Vec<String>)>,
    pub pending: Vec<String>,
    pub blockers: Vec<String>,
    pub cancel_requested: bool,
    /// Steering appended but not yet applied by a running loop. Visible so a
    /// detached task's queued instruction is not invisible until it lands.
    pub pending_steering: Vec<String>,
    pub last_message: Option<String>,
}

/// Reads durable task state without mutating the journal.
pub fn task_reports(workspace: &Path) -> Result<Vec<TaskReport>, SessionError> {
    let session = Session::inspect(workspace)?;
    Ok(session
        .tasks()
        .into_iter()
        .map(|view| TaskReport {
            task_id: view.task_id,
            agent: view.agent,
            objective: view.goal.objective.clone(),
            status: view.goal.status,
            window: view.window,
            revision: view.goal.current_revision.clone(),
            workers: view
                .agents
                .iter()
                .map(|record| (record.agent, record.state, record.objective.clone()))
                .collect(),
            processes: view
                .processes
                .iter()
                .map(|record| (record.process, record.state.clone(), record.argv.clone()))
                .collect(),
            pending: view
                .goal
                .pending_work
                .iter()
                .map(|item| item.title.clone())
                .collect(),
            blockers: view
                .goal
                .blockers
                .iter()
                .map(|blocker| blocker.summary.clone())
                .collect(),
            cancel_requested: view.cancel_requested,
            pending_steering: view.pending_steering.clone(),
            last_message: view.latest_user_message.clone(),
        })
        .collect())
}

/// Appends durable steering for a task, including one running in another
/// process. The journal is the only channel a detached run needs.
///
/// A terminal task is refused rather than accumulating steering nothing will
/// ever read: silently queueing into a finished task looks like it worked.
pub fn steer_task(
    workspace: &Path,
    task_id: Option<TaskId>,
    message: &str,
) -> Result<TaskId, SessionError> {
    let session = Session::resume(workspace)?;
    let task_id = resolve_active(&session, task_id, "steer")?;
    session.append_shared_event(SessionEvent::SteeringQueued {
        task_id,
        content: message.to_string(),
    })?;
    Ok(task_id)
}

/// Requests durable cancellation of a task.
pub fn cancel_task(workspace: &Path, task_id: Option<TaskId>) -> Result<TaskId, SessionError> {
    let session = Session::resume(workspace)?;
    let task_id = resolve_active(&session, task_id, "cancel")?;
    session.append_shared_event(SessionEvent::TaskCancelRequested { task_id })?;
    Ok(task_id)
}

/// Resolves the target task for a journal command, refusing a terminal one.
fn resolve_active(
    session: &Session,
    task_id: Option<TaskId>,
    verb: &str,
) -> Result<TaskId, SessionError> {
    let task_id = task_id
        .or_else(|| session.root_task())
        .ok_or_else(|| SessionError::State(format!("no task to {verb}")))?;
    let status = session.task_view(task_id).status();
    if status.is_active() {
        Ok(task_id)
    } else {
        Err(SessionError::State(format!(
            "cannot {verb} task {}: already {}",
            task_id.0,
            status.label()
        )))
    }
}

/// Serializes a parent-supplied context value under the delegation bound.
pub fn bounded_context(value: Value, max_bytes: usize) -> Value {
    let encoded = serde_json::to_vec(&value).unwrap_or_default();
    if encoded.len() <= max_bytes {
        value
    } else {
        serde_json::json!({ "truncated": true, "bytes": encoded.len() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_worker(registry: &WorkerRegistry, agent: AgentId) -> Arc<AtomicBool> {
        registry.register(
            agent,
            TaskId(agent.0 + 10),
            format!("objective {}", agent.0),
            AgentAccess::Read,
            None,
        )
    }

    fn result(agent: AgentId, ok: bool) -> DelegateResult {
        DelegateResult {
            task_id: TaskId(agent.0 + 10),
            agent,
            ok,
            summary: if ok { "done" } else { "failed" }.to_string(),
            base_revision: RevisionId("r".to_string()),
            final_revision: RevisionId("r".to_string()),
            changed: false,
            workspace: None,
            evidence: vec![],
            findings: Value::Null,
        }
    }

    #[test]
    fn join_blocks_until_the_worker_settles() {
        let registry = Arc::new(WorkerRegistry::default());
        registry_worker(&registry, AgentId(1));
        let settling = registry.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            settling.settle(AgentId(1), result(AgentId(1), true));
        });
        let joined = registry.join(AgentId(1)).unwrap();
        assert!(joined.ok);
        assert_eq!(
            registry.snapshot(AgentId(1)).unwrap().state,
            AgentState::Completed
        );
    }

    #[test]
    fn cancellation_affects_only_the_selected_worker() {
        let registry = WorkerRegistry::default();
        let first = registry_worker(&registry, AgentId(1));
        let second = registry_worker(&registry, AgentId(2));
        registry.cancel(AgentId(1)).unwrap();
        assert!(first.load(Ordering::Relaxed));
        assert!(
            !second.load(Ordering::Relaxed),
            "cancelling one worker must not touch its sibling"
        );
        assert!(registry.is_cancelled(AgentId(1)));
        assert!(!registry.is_cancelled(AgentId(2)));
    }

    #[test]
    fn a_cancelled_worker_settles_as_cancelled_not_failed() {
        let registry = WorkerRegistry::default();
        registry_worker(&registry, AgentId(3));
        registry.cancel(AgentId(3)).unwrap();
        registry.settle(AgentId(3), result(AgentId(3), false));
        assert_eq!(
            registry.snapshot(AgentId(3)).unwrap().state,
            AgentState::Cancelled
        );
    }

    #[test]
    fn joining_an_unknown_agent_is_an_error_not_a_hang() {
        let registry = WorkerRegistry::default();
        assert!(
            registry
                .join(AgentId(99))
                .unwrap_err()
                .contains("unknown agent")
        );
        assert!(registry.cancel(AgentId(99)).is_err());
    }

    #[test]
    fn oversized_parent_context_is_replaced_by_a_marker() {
        let big = Value::String("x".repeat(2048));
        assert_eq!(
            bounded_context(big, 512)["truncated"],
            Value::Bool(true),
            "an oversized context must not silently enter the child"
        );
        let small = serde_json::json!({ "hint": "ok" });
        assert_eq!(bounded_context(small.clone(), 512), small);
    }
}
