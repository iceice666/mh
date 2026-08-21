//! PTC-first agent loop with safe-point steering (spec §16–§18).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};

use crate::context::{ContextCompiler, ContextError};
use crate::model::{Model, ModelError, ModelEvent, ModelOutput, lower};
use crate::ptc::runtime::PtcEvent;
use crate::ptc::{PtcBudget, PtcOutcome, PtcRuntime};
use crate::session::{Session, SessionError, SessionEvent};
use crate::tools::Capabilities;

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
    Model(ModelError),
    Context(ContextError),
    Cancelled,
    TurnLimit(usize),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Session(error) => error.fmt(f),
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
        session.begin_task(task)?;
        session.append(SessionEvent::UserMessage {
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
        session.begin_task(message)?;
        session.append(SessionEvent::UserMessage {
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
        let store = session.result_store()?;
        let runtime = PtcRuntime::new(
            Capabilities::new(session.state().workspace.clone()),
            store,
            self.config.ptc_budget.clone(),
        );
        let mut repeated = RepeatState::default();

        for _turn in 0..self.config.max_turns {
            apply_controls(session, cancelled, controls, events, &mut repeated)?;
            check_cancelled(session, cancelled)?;

            let context = self.compiler.compile(session)?;
            session.append(SessionEvent::ModelStarted)?;
            let output = match self.model.generate(&context, cancelled, &mut |event| {
                events(AgentEvent::Model(event))
            }) {
                Ok(output) => output,
                Err(ModelError::Cancelled) => {
                    interrupt(session, "model")?;
                    return Err(AgentError::Cancelled);
                }
                Err(error) => return Err(error.into()),
            };
            session.append(SessionEvent::ModelCompleted {
                output_kind: output_kind(&output).to_string(),
            })?;

            // A control queued during inference invalidates all unexecuted output,
            // including a nominally final text response.
            if apply_controls(session, cancelled, controls, events, &mut repeated)? {
                continue;
            }
            check_cancelled(session, cancelled)?;

            match output {
                ModelOutput::Text(text) => {
                    session.append(SessionEvent::AssistantMessage {
                        content: text.clone(),
                    })?;
                    session.mark_completed()?;
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
                        record_repeat(session, events, fingerprint)?;
                        continue;
                    }

                    // Final safe point immediately before execution.
                    if apply_controls(session, cancelled, controls, events, &mut repeated)? {
                        continue;
                    }
                    check_cancelled(session, cancelled)?;

                    session.append(SessionEvent::PtcStarted {
                        source: source.clone(),
                    })?;
                    events(AgentEvent::PtcStarted);
                    let epoch = session.work_state().mutation_epoch;
                    let result = runtime.execute_at_epoch(&source, cancelled, epoch);
                    translate_ptc_events(session, &result.events, events)?;
                    events(AgentEvent::ToolCompleted {
                        outcome: format!("{:?}", result.outcome),
                        tool_calls: result.tool_calls,
                        duration_ms: result.duration_ms,
                    });
                    session.append_ptc_result(&result)?;
                    if result.outcome == PtcOutcome::Interrupted {
                        session.mark_interrupted()?;
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
                        record_repeat(session, events, fingerprint)?;
                    }

                    // Steering received after execution began is applied only now.
                    apply_controls(session, cancelled, controls, events, &mut repeated)?;
                    check_cancelled(session, cancelled)?;
                }
            }
        }
        Err(AgentError::TurnLimit(self.config.max_turns))
    }
}

#[derive(Default)]
struct RepeatState {
    normalized_source: Option<String>,
    fingerprint: Option<String>,
    identical_executions: usize,
}

fn apply_controls(
    session: &mut Session,
    cancelled: &Arc<AtomicBool>,
    controls: Option<&Receiver<AgentControl>>,
    events: &mut dyn FnMut(AgentEvent),
    repeated: &mut RepeatState,
) -> Result<bool, AgentError> {
    let Some(controls) = controls else {
        return Ok(false);
    };
    let mut steering = Vec::new();
    loop {
        match controls.try_recv() {
            Ok(AgentControl::Steer(content)) => steering.push(content),
            Ok(AgentControl::Interrupt) => cancelled.store(true, Ordering::Relaxed),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
        }
    }
    let applied = !steering.is_empty();
    for content in &steering {
        session.append(SessionEvent::SteeringQueued {
            content: content.clone(),
        })?;
        events(AgentEvent::SteeringQueued {
            content: content.clone(),
        });
    }
    for content in steering {
        session.append(SessionEvent::SteeringApplied {
            content: content.clone(),
        })?;
        events(AgentEvent::SteeringApplied { content });
    }
    if applied {
        *repeated = RepeatState::default();
    }
    Ok(applied)
}

fn translate_ptc_events(
    session: &mut Session,
    ptc_events: &[PtcEvent],
    events: &mut dyn FnMut(AgentEvent),
) -> Result<(), AgentError> {
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
                args_hash,
                effect,
                ok,
                duration_ms,
                result_ids,
                paths,
                ..
            } => {
                session.append(SessionEvent::ToolCompleted {
                    call_id: *call_id,
                    name: name.clone(),
                    args_hash: *args_hash,
                    effect: *effect,
                    ok: *ok,
                    duration_ms: *duration_ms,
                    result_ids: result_ids.clone(),
                    paths: paths.clone(),
                })?;
                events(AgentEvent::PtcHostCallCompleted {
                    call_id: *call_id,
                    name: name.clone(),
                    ok: *ok,
                    duration_ms: *duration_ms,
                });
            }
            PtcEvent::EvidenceRecorded { evidence } => {
                session.append(SessionEvent::EvidenceRecorded {
                    evidence: evidence.clone(),
                })?;
                events(AgentEvent::EvidenceRecorded {
                    kind: evidence.kind.clone(),
                    ok: evidence.ok,
                });
            }
        }
    }
    Ok(())
}

fn check_cancelled(session: &mut Session, cancelled: &Arc<AtomicBool>) -> Result<(), AgentError> {
    if cancelled.load(Ordering::Relaxed) {
        interrupt(session, "agent")?;
        Err(AgentError::Cancelled)
    } else {
        Ok(())
    }
}

fn interrupt(session: &mut Session, operation: &str) -> Result<(), SessionError> {
    session.append(SessionEvent::Interrupted {
        operation: operation.to_string(),
    })?;
    session.mark_interrupted()
}

fn record_repeat(
    session: &mut Session,
    events: &mut dyn FnMut(AgentEvent),
    fingerprint: String,
) -> Result<(), AgentError> {
    session.append(SessionEvent::RepeatedActionDetected {
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
            _cancelled: &Arc<AtomicBool>,
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
            _cancelled: &Arc<AtomicBool>,
            _events: &mut dyn FnMut(ModelEvent),
        ) -> Result<ModelOutput, ModelError> {
            self.barrier.wait();
            self.barrier.wait();
            self.outputs
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| ModelError::Protocol("script exhausted".to_string()))
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
            barrier.wait();
            barrier.wait();
            barrier.wait();
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
