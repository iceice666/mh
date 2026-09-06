//! Model adapter and compatibility/lowering layer (spec §17–§18).

use std::io::{BufRead, BufReader, Read};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::context::CompiledContext;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgramLanguage {
    JavaScript,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelOutput {
    Text(String),
    Program {
        language: ProgramLanguage,
        source: String,
    },
    FunctionCall {
        name: String,
        arguments: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelEvent {
    RequestStarted { endpoint: String },
    ResponseCreated { id: String },
    ReasoningSummaryDelta(String),
    ReasoningSummaryDone,
    OutputTextDelta(String),
    FunctionCall { name: String },
    ResponseCompleted { id: Option<String> },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationStopReason {
    UserInterrupt,
    Superseded,
}

#[derive(Debug, Clone)]
pub struct GenerationStop {
    user_interrupted: Arc<AtomicBool>,
    superseded: Arc<AtomicBool>,
}

impl GenerationStop {
    pub fn new(user_interrupted: Arc<AtomicBool>) -> Self {
        Self {
            user_interrupted,
            superseded: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn stop(&self, reason: GenerationStopReason) {
        match reason {
            GenerationStopReason::UserInterrupt => {
                self.user_interrupted.store(true, Ordering::Relaxed)
            }
            GenerationStopReason::Superseded => self.superseded.store(true, Ordering::Relaxed),
        }
    }

    pub fn reason(&self) -> Option<GenerationStopReason> {
        if self.user_interrupted.load(Ordering::Relaxed) {
            Some(GenerationStopReason::UserInterrupt)
        } else if self.superseded.load(Ordering::Relaxed) {
            Some(GenerationStopReason::Superseded)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub enum ModelError {
    Stopped(GenerationStopReason),
    Transport(String),
    Protocol(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped(GenerationStopReason::UserInterrupt) => {
                write!(f, "model request interrupted")
            }
            Self::Stopped(GenerationStopReason::Superseded) => {
                write!(f, "model request superseded")
            }
            Self::Transport(error) => write!(f, "model transport error: {error}"),
            Self::Protocol(error) => write!(f, "model protocol error: {error}"),
        }
    }
}

impl std::error::Error for ModelError {}

pub trait Model: Send + Sync {
    fn generate(
        &self,
        context: &CompiledContext,
        stop: &GenerationStop,
        events: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput, ModelError>;
}

#[derive(Debug, Clone)]
pub struct OpenAiResponses {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub request_timeout: Duration,
}

impl OpenAiResponses {
    pub fn from_env() -> Result<Self, ModelError> {
        let api_key = std::env::var("MH_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .map_err(|_| {
                ModelError::Protocol(
                    "set MH_API_KEY (or OPENAI_API_KEY) before running the agent".to_string(),
                )
            })?;
        Ok(Self {
            base_url: std::env::var("MH_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            api_key,
            model: std::env::var("MH_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_string()),
            request_timeout: Duration::from_secs(
                std::env::var("MH_MODEL_TIMEOUT_SECS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(300),
            ),
        })
    }

    fn endpoint(&self) -> String {
        format!("{}/responses", self.base_url.trim_end_matches('/'))
    }

    fn request_body(&self, context: &CompiledContext) -> Value {
        json!({
            "model": self.model,
            "input": context.render(),
            "tools": tool_definitions(),
            "tool_choice": "auto",
            "reasoning": { "summary": "auto" },
            "stream": true,
            "store": false
        })
    }
}

impl Model for OpenAiResponses {
    fn generate(
        &self,
        context: &CompiledContext,
        stop: &GenerationStop,
        events: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput, ModelError> {
        if let Some(reason) = stop.reason() {
            return Err(ModelError::Stopped(reason));
        }
        let endpoint = self.endpoint();
        events(ModelEvent::RequestStarted {
            endpoint: endpoint.clone(),
        });
        let body = self.request_body(context);
        let this = self.clone();
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let agent = ureq::AgentBuilder::new()
                .timeout(this.request_timeout)
                .build();
            let result = agent
                .post(&endpoint)
                .set("Authorization", &format!("Bearer {}", this.api_key))
                .set("Content-Type", "application/json")
                .set("Accept", "text/event-stream")
                .send_json(body)
                .map(|response| Box::new(response.into_reader()) as Box<dyn Read + Send>)
                .map_err(|error| ModelError::Transport(error.to_string()));
            let _ = send.send(result);
        });

        let reader = loop {
            if let Some(reason) = stop.reason() {
                return Err(ModelError::Stopped(reason));
            }
            match receive.recv_timeout(Duration::from_millis(50)) {
                Ok(result) => break result?,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(ModelError::Transport(
                        "model request worker stopped unexpectedly".to_string(),
                    ));
                }
            }
        };
        parse_responses_stream(reader, stop, events)
    }
}

fn parse_responses_stream(
    reader: Box<dyn Read + Send>,
    stop: &GenerationStop,
    events: &mut dyn FnMut(ModelEvent),
) -> Result<ModelOutput, ModelError> {
    let mut stream = ResponseStreamState::default();
    let mut data_lines = Vec::new();
    for line in BufReader::new(reader).lines() {
        if let Some(reason) = stop.reason() {
            return Err(ModelError::Stopped(reason));
        }
        let line = line.map_err(|error| ModelError::Transport(error.to_string()))?;
        if line.is_empty() {
            if !data_lines.is_empty() {
                let data = data_lines.join("\n");
                data_lines.clear();
                if data == "[DONE]" {
                    break;
                }
                let event = serde_json::from_str(&data).map_err(|error| {
                    ModelError::Protocol(format!("invalid Responses SSE event: {error}"))
                })?;
                stream.apply_event(&event, events)?;
            }
        } else if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        }
    }
    if !data_lines.is_empty() {
        let event = serde_json::from_str(&data_lines.join("\n")).map_err(|error| {
            ModelError::Protocol(format!("invalid trailing Responses SSE event: {error}"))
        })?;
        stream.apply_event(&event, events)?;
    }
    stream.finish()
}

#[derive(Default)]
struct ResponseStreamState {
    response_id: Option<String>,
    output_text: String,
    function_name: Option<String>,
    function_arguments: String,
    completed: bool,
}

impl ResponseStreamState {
    fn apply_event(
        &mut self,
        event: &Value,
        events: &mut dyn FnMut(ModelEvent),
    ) -> Result<(), ModelError> {
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| ModelError::Protocol("Responses SSE event missing type".to_string()))?;
        match event_type {
            "response.created" => {
                if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                    self.response_id = Some(id.to_string());
                    events(ModelEvent::ResponseCreated { id: id.to_string() });
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    events(ModelEvent::ReasoningSummaryDelta(delta.to_string()));
                }
            }
            "response.reasoning_summary_text.done" => {
                events(ModelEvent::ReasoningSummaryDone);
            }
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    self.output_text.push_str(delta);
                    events(ModelEvent::OutputTextDelta(delta.to_string()));
                }
            }
            "response.output_item.added" | "response.output_item.done" => {
                if event.pointer("/item/type").and_then(Value::as_str) == Some("function_call") {
                    if let Some(name) = event.pointer("/item/name").and_then(Value::as_str) {
                        if self.function_name.as_deref() != Some(name) {
                            events(ModelEvent::FunctionCall {
                                name: name.to_string(),
                            });
                        }
                        self.function_name = Some(name.to_string());
                    }
                    if let Some(arguments) =
                        event.pointer("/item/arguments").and_then(Value::as_str)
                        && !arguments.is_empty()
                    {
                        self.function_arguments = arguments.to_string();
                    }
                }
            }
            "response.function_call_arguments.done" => {
                if let Some(name) = event.get("name").and_then(Value::as_str) {
                    if self.function_name.as_deref() != Some(name) {
                        events(ModelEvent::FunctionCall {
                            name: name.to_string(),
                        });
                    }
                    self.function_name = Some(name.to_string());
                }
                self.function_arguments = event
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}")
                    .to_string();
            }
            "response.completed" => {
                self.completed = true;
                if self.response_id.is_none() {
                    self.response_id = event
                        .pointer("/response/id")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
                events(ModelEvent::ResponseCompleted {
                    id: self.response_id.clone(),
                });
                self.read_completed_output(event)?;
            }
            "response.failed" | "response.incomplete" => {
                let message = event
                    .pointer("/response/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or(event_type);
                return Err(ModelError::Protocol(message.to_string()));
            }
            "error" => {
                let message = event
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Responses API stream error");
                return Err(ModelError::Protocol(message.to_string()));
            }
            _ => {}
        }
        Ok(())
    }

    fn read_completed_output(&mut self, event: &Value) -> Result<(), ModelError> {
        let Some(output) = event.pointer("/response/output").and_then(Value::as_array) else {
            return Ok(());
        };
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("function_call") if self.function_name.is_none() => {
                    self.function_name =
                        item.get("name").and_then(Value::as_str).map(str::to_string);
                    self.function_arguments = item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}")
                        .to_string();
                }
                Some("message") if self.output_text.is_empty() => {
                    if let Some(content) = item.get("content").and_then(Value::as_array) {
                        self.output_text = content
                            .iter()
                            .filter(|part| {
                                part.get("type").and_then(Value::as_str) == Some("output_text")
                            })
                            .filter_map(|part| part.get("text").and_then(Value::as_str))
                            .collect();
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<ModelOutput, ModelError> {
        if !self.completed {
            return Err(ModelError::Protocol(
                "Responses stream ended before response.completed".to_string(),
            ));
        }
        if let Some(name) = self.function_name {
            let arguments = serde_json::from_str(&self.function_arguments).map_err(|error| {
                ModelError::Protocol(format!("invalid function arguments: {error}"))
            })?;
            return normalize_function_call(name, arguments);
        }
        if self.output_text.is_empty() {
            return Err(ModelError::Protocol(
                "Responses stream completed without text or function call".to_string(),
            ));
        }
        Ok(normalize_text(&self.output_text))
    }
}
pub fn normalize_text(content: &str) -> ModelOutput {
    if let Some(source) = extract_js_fence(content) {
        return ModelOutput::Program {
            language: ProgramLanguage::JavaScript,
            source,
        };
    }
    ModelOutput::Text(content.trim().to_string())
}

fn normalize_function_call(name: String, arguments: Value) -> Result<ModelOutput, ModelError> {
    if name == "ptc" {
        let source = arguments
            .get("source")
            .and_then(Value::as_str)
            .filter(|source| !source.trim().is_empty())
            .ok_or_else(|| ModelError::Protocol("ptc function call missing source".to_string()))?;
        return Ok(ModelOutput::Program {
            language: ProgramLanguage::JavaScript,
            source: source.to_string(),
        });
    }
    Ok(ModelOutput::FunctionCall { name, arguments })
}

/// Lowers every executable provider output into canonical PTC.
pub fn lower(output: &ModelOutput) -> Option<String> {
    match output {
        ModelOutput::Text(_) => None,
        ModelOutput::Program { source, .. } => Some(source.clone()),
        ModelOutput::FunctionCall { name, arguments } => {
            let name = serde_json::to_string(name).expect("serializing a string cannot fail");
            let arguments = serde_json::to_string(arguments).expect("serializing JSON cannot fail");
            Some(format!("return tool({name}, {arguments});"))
        }
    }
}

fn extract_js_fence(content: &str) -> Option<String> {
    for marker in ["```javascript", "```js", "```ptc"] {
        let Some(start) = content.find(marker) else {
            continue;
        };
        let body_start = start + marker.len();
        let body = content[body_start..]
            .strip_prefix('\n')
            .unwrap_or(&content[body_start..]);
        let end = body.find("```")?;
        return Some(body[..end].trim().to_string());
    }
    None
}

fn tool_definitions() -> Value {
    json!([{
        "type": "function",
        "name": "ptc",
        "description": "Execute one bounded synchronous JavaScript Programmatic Tool Calling program inside the workspace sandbox. The program may call tool, read, write, edit, glob, grep, exec, batch(name, args[]) for procedural parallel host calls, goal(update) to record durable task state that survives a context rollover, finish({summary, unresolved?, evidence?, force?}) to explicitly complete the task, agent_spawn/agent_poll/agent_join/agent_cancel/agent_send/agent_list for asynchronous delegated workers, delegate({task, access|profile, context?}) or delegate_batch(options[]) for spawn-then-join reasoning, integrate(workspace) for explicit isolated change integration, process_spawn/process_poll/process_tail/process_wait/process_kill/process_write/process_list for long-lived background processes, evidence(kind, ok, metadata?) to record verification, checkpoint() to capture the current workspace revision, and restore(checkpoint) to explicitly restore one. Plain text is a progress message and never completes the task; only an accepted finish() does. Return only compact evidence needed for the next inference; result handles expose id, length, totalBytes, truncated, kind, and bounded read/head/tail/grep/json methods.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "A complete synchronous PTC JavaScript program with a top-level return statement."
                }
            },
            "required": ["source"],
            "additionalProperties": false
        },
        "strict": true
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_stream_normalizes_ptc_function_into_program() {
        let stream = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"Inspecting\"}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.done\",\"text\":\"Inspecting\"}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"in_progress\",\"arguments\":\"\",\"call_id\":\"call_1\",\"name\":\"ptc\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_1\",\"output_index\":0,\"arguments\":\"{\\\"source\\\":\\\"return read(\\\\\\\"src/main.rs\\\\\\\");\\\"}\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"completed\",\"arguments\":\"{\\\"source\\\":\\\"return read(\\\\\\\"src/main.rs\\\\\\\");\\\"}\",\"call_id\":\"call_1\",\"name\":\"ptc\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"output\":[]}}\n\n",
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        let stop = GenerationStop::new(cancelled);
        let mut events = Vec::new();
        let output = parse_responses_stream(
            Box::new(std::io::Cursor::new(stream.as_bytes().to_vec())),
            &stop,
            &mut |event| events.push(event),
        )
        .unwrap();
        assert_eq!(
            output,
            ModelOutput::Program {
                language: ProgramLanguage::JavaScript,
                source: "return read(\"src/main.rs\");".to_string(),
            }
        );
        assert!(events.contains(&ModelEvent::ResponseCreated {
            id: "resp_1".to_string(),
        }));
        assert!(events.contains(&ModelEvent::ReasoningSummaryDelta("Inspecting".to_string(),)));
        assert!(events.contains(&ModelEvent::FunctionCall {
            name: "ptc".to_string(),
        }));
    }

    #[test]
    fn responses_stream_collects_output_text() {
        let stream = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"done\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"output\":[]}}\n\n",
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        let stop = GenerationStop::new(cancelled);
        let output = parse_responses_stream(
            Box::new(std::io::Cursor::new(stream.as_bytes().to_vec())),
            &stop,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(output, ModelOutput::Text("done".to_string()));
    }
    #[test]
    fn exposes_only_strict_ptc_provider_function() {
        let tools = tool_definitions();
        let tools = tools.as_array().unwrap();
        assert_eq!(tools.len(), 1);
        let tool = &tools[0];
        assert_eq!(tool.get("name").and_then(Value::as_str), Some("ptc"));
        assert_eq!(tool.get("strict"), Some(&Value::Bool(true)));
        let parameters = tool.get("parameters").unwrap();
        assert_eq!(parameters.get("required"), Some(&json!(["source"])));
        assert_eq!(
            parameters
                .pointer("/properties/source/type")
                .and_then(Value::as_str),
            Some("string")
        );
        let description = tool.get("description").and_then(Value::as_str).unwrap();
        assert!(description.contains("batch(name, args[])"));
        assert!(description.contains("evidence(kind, ok, metadata?)"));
        assert!(description.contains("totalBytes"));
    }

    #[test]
    fn generation_stop_prioritizes_user_interrupt() {
        let interrupted = Arc::new(AtomicBool::new(false));
        let stop = GenerationStop::new(interrupted);
        stop.stop(GenerationStopReason::Superseded);
        assert_eq!(stop.reason(), Some(GenerationStopReason::Superseded));
        stop.stop(GenerationStopReason::UserInterrupt);
        assert_eq!(stop.reason(), Some(GenerationStopReason::UserInterrupt));
    }

    #[test]
    fn provider_description_exposes_checkpoint_and_restore() {
        let definitions = tool_definitions();
        let description = definitions[0]["description"].as_str().unwrap();
        assert!(description.contains("checkpoint()"));
        assert!(description.contains("restore(checkpoint)"));
    }

    #[test]
    fn ptc_function_requires_non_empty_source() {
        let error =
            normalize_function_call("ptc".to_string(), json!({"source": "  "})).unwrap_err();
        assert!(error.to_string().contains("missing source"));
    }

    #[test]
    fn extracts_program_fence() {
        let output = normalize_text("Working.\n```js\nreturn glob(\"src/**\");\n```");
        assert!(matches!(output, ModelOutput::Program { .. }));
    }

    #[test]
    fn function_call_lowers_to_canonical_tool() {
        let source = lower(&ModelOutput::FunctionCall {
            name: "read".to_string(),
            arguments: json!({"path": "a.rs"}),
        })
        .unwrap();
        assert_eq!(source, "return tool(\"read\", {\"path\":\"a.rs\"});");
    }
}
