//! PTC-first agent loop with safe-point steering (spec §16–§18).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

use crate::checkpoint::{CheckpointError, CheckpointStore};
use crate::context::{ContextCompiler, ContextError};
use crate::model::{
    GenerationStop, GenerationStopReason, Model, ModelError, ModelEvent, ModelOutput, lower,
};
use crate::ptc::runtime::{PtcEvent, PtcEventSink, PtcExecution};
use crate::ptc::{PtcBudget, PtcOutcome, PtcRuntime};
use crate::session::{Session, SessionError, SessionEvent};
use crate::tools::Capabilities;
use crate::workspace::{WorkspaceError, WorkspaceTracker, WorkspaceTrackerImpl};

const REPEAT_WARNING: &str = "The previous action was repeated without changing the result. Choose a different approach instead of retrying the same PTC program.";

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_turns: usize,
    pub context_tokens: usize,
    pub ptc_budget: PtcBudget,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 32,
            context_tokens: 32_000,
            ptc_budget: PtcBudget::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentControl {
    Steer(String),
    Interrupt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    Model(ModelEvent),
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

pub struct Agent<M> {
    model: M,
    compiler: ContextCompiler,
    config: AgentConfig,
}

impl<M: Model> Agent<M> {
    pub fn new(model: M, config: AgentConfig) -> Self {
        let compiler = ContextCompiler {
            max_tokens: config.context_tokens,
            ..ContextCompiler::default()
        };
        Self {
            model,
            compiler,
            config,
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
        let tracker_impl = WorkspaceTrackerImpl::open(&session.state().workspace)?;
        let tracker: Arc<dyn WorkspaceTracker> = Arc::new(tracker_impl);
        let checkpoints = CheckpointStore::open(&session.state().workspace, tracker.clone())?;
        let runtime = PtcRuntime::new(
            Capabilities::new(session.state().workspace.clone()),
            session.result_store()?,
            self.config.ptc_budget.clone(),
            tracker,
            checkpoints,
        );
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
            let context = self.compiler.compile(session)?;
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
