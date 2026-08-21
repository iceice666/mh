//! Small PTC-first agent loop (spec §6).

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::context::ContextCompiler;
use crate::model::{Model, ModelError, ModelEvent, ModelOutput, lower};
use crate::ptc::{PtcBudget, PtcOutcome, PtcRuntime};
use crate::session::{Session, SessionError, SessionEvent};
use crate::tools::Capabilities;

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
pub enum AgentEvent {
    Model(ModelEvent),
    PtcStarted,
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
    Cancelled,
    TurnLimit(usize),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Session(error) => error.fmt(f),
            Self::Model(error) => error.fmt(f),
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
        let mut session = Session::open(workspace)?;
        session.begin_task(task)?;
        session.append(SessionEvent::UserMessage {
            content: task.to_string(),
        })?;
        self.run_session(&mut session, cancelled, events)
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
        let mut session = Session::resume(workspace)?;
        self.run_session(&mut session, cancelled, events)
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
        let mut session = Session::open(workspace)?;
        session.begin_task(message)?;
        session.append(SessionEvent::UserMessage {
            content: message.to_string(),
        })?;
        self.run_session(&mut session, cancelled, events)
    }

    fn run_session(
        &self,
        session: &mut Session,
        cancelled: &Arc<AtomicBool>,
        events: &mut dyn FnMut(AgentEvent),
    ) -> Result<String, AgentError> {
        let store = session.result_store()?;
        let runtime = PtcRuntime::new(
            Capabilities::new(session.state().workspace.clone()),
            store,
            self.config.ptc_budget.clone(),
        );

        for _turn in 0..self.config.max_turns {
            if cancelled.load(Ordering::Relaxed) {
                session.append(SessionEvent::Interrupted {
                    operation: "agent".to_string(),
                })?;
                session.mark_interrupted()?;
                return Err(AgentError::Cancelled);
            }

            let context = self.compiler.compile(session);
            session.append(SessionEvent::ModelStarted)?;
            let output = match self.model.generate(&context, cancelled, &mut |event| {
                events(AgentEvent::Model(event))
            }) {
                Ok(output) => output,
                Err(ModelError::Cancelled) => {
                    session.append(SessionEvent::Interrupted {
                        operation: "model".to_string(),
                    })?;
                    session.mark_interrupted()?;
                    return Err(AgentError::Cancelled);
                }
                Err(error) => return Err(error.into()),
            };
            session.append(SessionEvent::ModelCompleted {
                output_kind: output_kind(&output).to_string(),
            })?;

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
                    session.append(SessionEvent::PtcStarted {
                        source: source.clone(),
                    })?;
                    events(AgentEvent::PtcStarted);
                    let result = runtime.execute(&source, cancelled);
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
                }
            }
        }

        Err(AgentError::TurnLimit(self.config.max_turns))
    }
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
    use std::sync::Mutex;

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

    #[test]
    fn ptc_program_loop_executes_then_finishes() {
        let dir = tempfile::tempdir().unwrap();
        let model = ScriptedModel::new(vec![
            ModelOutput::Program {
                language: ProgramLanguage::JavaScript,
                source: "write(\"a.txt\", \"hello\"); return read(\"a.txt\").content;".to_string(),
            },
            ModelOutput::Text("done".to_string()),
        ]);
        let agent = Agent::new(model, AgentConfig::default());
        let cancelled = Arc::new(AtomicBool::new(false));
        let answer = agent
            .run_task(dir.path(), "write and inspect", &cancelled)
            .unwrap();
        assert_eq!(answer, "done");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "hello"
        );
        let session = Session::resume(dir.path()).unwrap();
        assert!(
            session
                .events()
                .iter()
                .any(|record| { matches!(record.event, SessionEvent::PtcCompleted { .. }) })
        );
    }

    #[test]
    fn reports_ptc_progress_events() {
        let dir = tempfile::tempdir().unwrap();
        let model = ScriptedModel::new(vec![
            ModelOutput::Program {
                language: ProgramLanguage::JavaScript,
                source: "return write(\"event.txt\", \"ok\");".to_string(),
            },
            ModelOutput::Text("done".to_string()),
        ]);
        let agent = Agent::new(model, AgentConfig::default());
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut events = Vec::new();
        agent
            .run_task_with_events(dir.path(), "report progress", &cancelled, &mut |event| {
                events.push(event)
            })
            .unwrap();
        assert!(events.contains(&AgentEvent::PtcStarted));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolCompleted { tool_calls: 1, .. }))
        );
    }
}
