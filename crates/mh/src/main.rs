use std::io::{self, BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use mh::agent::{Agent, AgentConfig, AgentError, AgentEvent};
use mh::model::{ModelEvent, OpenAiResponses};
use mh::session::{Session, SessionError, SessionEvent};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Cancelled) => ExitCode::from(130),
        Err(error) => {
            eprintln!("mh: {error}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug)]
enum CliError {
    Io(io::Error),
    Session(SessionError),
    Agent(AgentError),
    Model(String),
    Cancelled,
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::Session(error) => error.fmt(f),
            Self::Agent(error) => error.fmt(f),
            Self::Model(error) => f.write_str(error),
            Self::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl From<io::Error> for CliError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<SessionError> for CliError {
    fn from(value: SessionError) -> Self {
        Self::Session(value)
    }
}

impl From<AgentError> for CliError {
    fn from(value: AgentError) -> Self {
        if matches!(value, AgentError::Cancelled) {
            Self::Cancelled
        } else {
            Self::Agent(value)
        }
    }
}

fn run() -> Result<(), CliError> {
    let workspace = std::env::current_dir()?;
    let cancelled = install_ctrl_c()?;
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    if args
        .first()
        .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        print_help();
        return Ok(());
    }
    if args
        .first()
        .is_some_and(|arg| arg == "--version" || arg == "-V")
    {
        println!("mh {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let mut printer = EventPrinter {
        reasoning_open: false,
    };
    match args.first().map(String::as_str) {
        Some("resume") if args.len() == 1 => {
            let agent = new_agent()?;
            let answer = agent.resume_with_events(&workspace, &cancelled, &mut |event| {
                printer.print(event);
            })?;
            println!("{answer}");
        }
        Some("sessions") if args.len() == 1 => print_session(&workspace)?,
        Some(_) => {
            let task = std::mem::take(&mut args).join(" ");
            let agent = new_agent()?;
            let answer =
                agent.run_task_with_events(&workspace, &task, &cancelled, &mut |event| {
                    printer.print(event)
                })?;
            println!("{answer}");
        }
        None => repl(&workspace, &cancelled)?,
    }
    Ok(())
}

fn new_agent() -> Result<Agent<OpenAiResponses>, CliError> {
    let model = OpenAiResponses::from_env().map_err(|error| CliError::Model(error.to_string()))?;
    Ok(Agent::new(model, AgentConfig::default()))
}

fn repl(workspace: &Path, cancelled: &Arc<AtomicBool>) -> Result<(), CliError> {
    println!("mh — PTC coding agent. Type /help for commands.");
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut printer = EventPrinter {
        reasoning_open: false,
    };
    for line in stdin.lock().lines() {
        write!(stdout, "mh> ")?;
        stdout.flush()?;
        let line = line?;
        let message = line.trim();
        if message.is_empty() {
            continue;
        }
        match message {
            "/quit" | "/exit" => return Ok(()),
            "/help" => {
                writeln!(stdout, "/resume  resume the workspace session")?;
                writeln!(stdout, "/session show session state")?;
                writeln!(stdout, "/quit    exit")?;
            }
            "/session" => print_session(workspace)?,
            "/resume" => {
                cancelled.store(false, Ordering::Relaxed);
                let answer =
                    new_agent()?.resume_with_events(workspace, cancelled, &mut |event| {
                        printer.print(event)
                    })?;
                writeln!(stdout, "{answer}")?;
            }
            _ => {
                cancelled.store(false, Ordering::Relaxed);
                let answer = new_agent()?.send_message_with_events(
                    workspace,
                    message,
                    cancelled,
                    &mut |event| printer.print(event),
                )?;
                writeln!(stdout, "{answer}")?;
            }
        }
    }
    Ok(())
}

struct EventPrinter {
    reasoning_open: bool,
}

impl EventPrinter {
    fn print(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Model(ModelEvent::ReasoningSummaryDelta(delta)) => {
                if !self.reasoning_open {
                    eprint!("[reasoning] ");
                    self.reasoning_open = true;
                }
                eprint!("{delta}");
                let _ = io::stderr().flush();
            }
            AgentEvent::Model(ModelEvent::ReasoningSummaryDone) => {
                if self.reasoning_open {
                    eprintln!();
                    self.reasoning_open = false;
                }
            }
            event => print_event(event),
        }
    }
}

fn print_event(event: AgentEvent) {
    match event {
        AgentEvent::Model(ModelEvent::RequestStarted { endpoint }) => {
            eprintln!("[request] POST {endpoint}");
        }
        AgentEvent::Model(ModelEvent::ResponseCreated { id }) => {
            eprintln!("[response] {id}");
        }
        AgentEvent::Model(ModelEvent::ReasoningSummaryDelta(delta)) => {
            eprintln!("[reasoning] {delta}");
        }
        AgentEvent::Model(ModelEvent::ReasoningSummaryDone) => {}
        AgentEvent::Model(ModelEvent::OutputTextDelta(_)) => {}
        AgentEvent::Model(ModelEvent::FunctionCall { name }) => {
            eprintln!("[provider tool] {name}");
        }
        AgentEvent::Model(ModelEvent::ResponseCompleted { id }) => {
            if let Some(id) = id {
                eprintln!("[response] completed {id}");
            } else {
                eprintln!("[response] completed");
            }
        }
        AgentEvent::PtcStarted => eprintln!("[ptc] executing"),
        AgentEvent::ToolCompleted {
            outcome,
            tool_calls,
            duration_ms,
        } => eprintln!("[ptc] {outcome}; {tool_calls} tool call(s); {duration_ms} ms"),
    }
}

fn print_session(workspace: &Path) -> Result<(), CliError> {
    let session = Session::resume(workspace)?;
    let state = session.state();
    println!("id: {}", state.id);
    println!("status: {:?}", state.status);
    println!("events: {}", session.events().len());
    if let Some(task) = state.task.as_deref() {
        println!("task: {task}");
    }
    if let Some(answer) = session
        .events()
        .iter()
        .rev()
        .find_map(|record| match &record.event {
            SessionEvent::AssistantMessage { content } => Some(content.as_str()),
            _ => None,
        })
    {
        println!("last answer: {answer}");
    }
    Ok(())
}

fn install_ctrl_c() -> Result<Arc<AtomicBool>, CliError> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let signal = cancelled.clone();
    ctrlc::set_handler(move || {
        signal.store(true, Ordering::Relaxed);
    })
    .map_err(|error| CliError::Model(format!("failed to install Ctrl-C handler: {error}")))?;
    Ok(cancelled)
}

fn print_help() {
    println!("mh — minimal PTC coding agent");
    println!();
    println!("USAGE:");
    println!("  mh                     open an interactive session");
    println!("  mh \"fix the tests\"    start a task");
    println!("  mh resume              resume the workspace session");
    println!("  mh sessions            inspect the workspace session");
    println!();
    println!("ENVIRONMENT:");
    println!("  MH_API_KEY or OPENAI_API_KEY");
    println!("  MH_BASE_URL             Responses API root; default: https://api.openai.com/v1");
    println!("  MH_MODEL                default: gpt-4.1-mini");
    println!("  MH_MODEL_TIMEOUT_SECS   default: 300");
}
