//! Shared harness for the v0.5 durable-runtime integration tests.
//!
//! The model is faked, but nothing else is: these tests drive the real agent
//! loop, the real MicroQuickJS PTC runtime, real Git-backed isolated
//! workspaces, and real subprocesses.
//!
//! Each integration binary uses a different subset of these helpers, so an
//! item unused by one binary is still load-bearing for another.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mh::agent::{Agent, AgentConfig, AgentEvent, TaskOutcome};
use mh::context::CompiledContext;
use mh::model::{GenerationStop, Model, ModelError, ModelEvent, ModelOutput, ProgramLanguage};
use mh::session::{EventRecord, Session, SessionEvent};

pub fn program(source: &str) -> ModelOutput {
    ModelOutput::Program {
        language: ProgramLanguage::JavaScript,
        source: source.to_string(),
    }
}

/// A `finish()` program. `force` waives judgement-level objections only, so a
/// live worker or a failed verification still refuses.
pub fn finish(summary: &str) -> ModelOutput {
    program(&format!(
        "return finish({{ summary: \"{summary}\", force: true }});"
    ))
}

pub fn text(message: &str) -> ModelOutput {
    ModelOutput::Text(message.to_string())
}

/// Marker the compiled context carries for every delegated worker.
const WORKER_MARKER: &str = "delegated worker with a parent";

/// A model whose turns are scripted per role.
///
/// Routing by rendered context rather than by call order is what makes these
/// tests deterministic under real concurrency: the root and each worker draw
/// from their own script no matter which thread reaches the model first.
pub struct RoleModel {
    root: Mutex<Vec<ModelOutput>>,
    /// Keyed by a substring of the worker's delegated objective.
    workers: Mutex<HashMap<String, Vec<ModelOutput>>>,
    /// Runs for every worker turn; used to observe real overlap.
    worker_hook: Option<WorkerHook>,
    /// Every rendered context, for assertions about what the model saw.
    seen: Mutex<Vec<String>>,
}

/// Observer invoked with a worker's objective on each of its model turns.
pub type WorkerHook = Arc<dyn Fn(&str) + Send + Sync>;

impl RoleModel {
    pub fn new(root: Vec<ModelOutput>) -> Self {
        Self {
            root: Mutex::new(reversed(root)),
            workers: Mutex::new(HashMap::new()),
            worker_hook: None,
            seen: Mutex::new(Vec::new()),
        }
    }

    /// Scripts one worker, matched by a substring of its objective.
    pub fn worker(self, objective: &str, outputs: Vec<ModelOutput>) -> Self {
        self.workers
            .lock()
            .unwrap()
            .insert(objective.to_string(), reversed(outputs));
        self
    }

    pub fn with_worker_hook(mut self, hook: WorkerHook) -> Self {
        self.worker_hook = Some(hook);
        self
    }
}

fn reversed(mut outputs: Vec<ModelOutput>) -> Vec<ModelOutput> {
    outputs.reverse();
    outputs
}

impl Model for RoleModel {
    fn generate(
        &self,
        context: &CompiledContext,
        _stop: &GenerationStop,
        _events: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput, ModelError> {
        let rendered = context.render();
        self.seen.lock().unwrap().push(rendered.clone());
        if !rendered.contains(WORKER_MARKER) {
            return self
                .root
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| ModelError::Protocol("root script exhausted".to_string()));
        }
        let objective = worker_objective(&rendered);
        if let Some(hook) = self.worker_hook.as_ref() {
            hook(&objective);
        }
        let mut workers = self.workers.lock().unwrap();
        let Some((_, script)) = workers
            .iter_mut()
            .find(|(key, _)| objective.contains(key.as_str()))
        else {
            return Err(ModelError::Protocol(format!(
                "no worker script matches objective {objective:?}"
            )));
        };
        script.pop().ok_or_else(|| {
            ModelError::Protocol(format!("worker script for {objective:?} exhausted"))
        })
    }
}

fn worker_objective(rendered: &str) -> String {
    rendered
        .split("[delegated objective]\n")
        .nth(1)
        .and_then(|rest| rest.split("\n\n").next())
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Config with prelude discovery and trust disabled: these tests assert about
/// the runtime, not about whatever prelude the developer's machine has.
pub fn config() -> AgentConfig {
    AgentConfig {
        user_prelude: None,
        trust_store: None,
        confirm_prelude: None,
        ..AgentConfig::default()
    }
}

pub fn run(
    workspace: &Path,
    model: RoleModel,
    task: &str,
    config: AgentConfig,
) -> (TaskOutcome, Vec<AgentEvent>) {
    let agent = Agent::new(model, config);
    let mut events = Vec::new();
    let outcome = agent
        .run_task_controlled(
            workspace,
            task,
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |event| events.push(event),
        )
        .expect("task ran");
    (outcome, events)
}

/// A Git workspace, which isolated-write delegation requires.
pub fn repository() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    git(dir.path(), &["init", "-q"]);
    git(dir.path(), &["config", "user.name", "mh test"]);
    git(dir.path(), &["config", "user.email", "mh@example.invalid"]);
    std::fs::write(dir.path().join("tracked"), "base\n").unwrap();
    git(dir.path(), &["add", "."]);
    git(dir.path(), &["commit", "-qm", "base"]);
    dir
}

pub fn git(dir: &Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Journal helpers. Every assertion about durability goes through the journal,
/// because that is what actually survives a restart.
pub fn events(workspace: &Path) -> Vec<EventRecord> {
    Session::inspect(workspace).unwrap().events()
}

/// Sequence numbers of events matching a predicate, in journal order.
pub fn seqs(workspace: &Path, matches: impl Fn(&SessionEvent) -> bool) -> Vec<u64> {
    events(workspace)
        .into_iter()
        .filter(|record| matches(&record.event))
        .map(|record| record.seq)
        .collect()
}

/// The last PTC value carrying `key`; root and worker results share the log.
pub fn ptc_value(workspace: &Path, key: &str) -> serde_json::Value {
    events(workspace)
        .into_iter()
        .rev()
        .find_map(|record| match record.event {
            SessionEvent::PtcCompleted { value, .. } if value.get(key).is_some() => Some(value),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no PTC result carried {key}"))
}

/// Tracks the largest number of workers observed running at once.
#[derive(Default)]
pub struct Overlap {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl Overlap {
    /// Blocks the caller until `target` workers are simultaneously inside this
    /// call, or the deadline passes.
    ///
    /// Real concurrency releases the gate; serialized execution hits the
    /// deadline and leaves `peak` at 1, which the test then reports as a
    /// failure instead of hanging forever.
    pub fn rendezvous(&self, target: usize, timeout: Duration) {
        let live = self.live.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak.fetch_max(live, Ordering::AcqRel);
        let deadline = Instant::now() + timeout;
        while self.live.load(Ordering::Acquire) < target && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        self.live.fetch_sub(1, Ordering::AcqRel);
    }

    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }
}
