use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::Duration;

use mh::agent::{Agent, AgentConfig, AgentControl, AgentError, AgentEvent};
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
    let (input_tx, input_rx) = mpsc::channel();
    std::thread::spawn(move || read_stdin(input_tx));
    let (event_tx, event_rx) = mpsc::channel();
    let mut worker: Option<AgentWorker> = None;
    let mut printer = EventPrinter {
        reasoning_open: false,
    };
    print_prompt()?;

    loop {
        drain_worker_events(&event_rx, &mut printer);
        if worker
            .as_ref()
            .is_some_and(|worker| worker.handle.is_finished())
        {
            let finished = worker.take().expect("worker was present");
            match finished.handle.join() {
                Ok(Ok(answer)) => println!("{answer}"),
                Ok(Err(AgentError::Cancelled)) => eprintln!("[agent] cancelled"),
                Ok(Err(error)) => eprintln!("[agent] {error}"),
                Err(_) => eprintln!("[agent] worker panicked"),
            }
            drain_worker_events(&event_rx, &mut printer);
            print_prompt()?;
        }

        match input_rx.recv_timeout(Duration::from_millis(25)) {
            Ok(Ok(line)) => {
                let message = line.trim();
                if message.is_empty() {
                    print_prompt()?;
                    continue;
                }
                if let Some(active) = worker.as_ref() {
                    match message {
                        "/quit" | "/exit" => {
                            let _ = active.controls.send(AgentControl::Interrupt);
                            cancelled.store(true, Ordering::Relaxed);
                            let active = worker.take().expect("worker was present");
                            let _ = active.handle.join();
                            return Ok(());
                        }
                        "/help" => print_repl_help(),
                        _ => {
                            active
                                .controls
                                .send(AgentControl::Steer(message.to_string()))
                                .map_err(|_| {
                                    CliError::Model("agent control channel closed".to_string())
                                })?;
                            eprintln!("[steering] queued");
                        }
                    }
                    print_prompt()?;
                    continue;
                }

                match message {
                    "/quit" | "/exit" => return Ok(()),
                    "/help" => print_repl_help(),
                    "/session" => print_session(workspace)?,
                    "/resume" => {
                        worker = Some(spawn_agent_worker(
                            workspace.to_path_buf(),
                            None,
                            true,
                            cancelled.clone(),
                            event_tx.clone(),
                        )?)
                    }
                    _ => {
                        worker = Some(spawn_agent_worker(
                            workspace.to_path_buf(),
                            Some(message.to_string()),
                            false,
                            cancelled.clone(),
                            event_tx.clone(),
                        )?)
                    }
                }
                print_prompt()?;
            }
            Ok(Err(error)) => return Err(error.into()),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                if let Some(active) = worker.take() {
                    let _ = active.controls.send(AgentControl::Interrupt);
                    cancelled.store(true, Ordering::Relaxed);
                    let _ = active.handle.join();
                }
                return Ok(());
            }
        }
    }
}

struct AgentWorker {
    controls: Sender<AgentControl>,
    handle: JoinHandle<Result<String, AgentError>>,
}

fn spawn_agent_worker(
    workspace: PathBuf,
    message: Option<String>,
    resume: bool,
    cancelled: Arc<AtomicBool>,
    event_tx: Sender<AgentEvent>,
) -> Result<AgentWorker, CliError> {
    let agent = new_agent()?;
    cancelled.store(false, Ordering::Relaxed);
    let (control_tx, control_rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let mut emit = |event| {
            let _ = event_tx.send(event);
        };
        if resume {
            agent.resume_controlled(&workspace, &cancelled, Some(&control_rx), &mut emit)
        } else {
            agent.send_message_controlled(
                &workspace,
                message.as_deref().expect("message task"),
                &cancelled,
                Some(&control_rx),
                &mut emit,
            )
        }
    });
    Ok(AgentWorker {
        controls: control_tx,
        handle,
    })
}

fn read_stdin(sender: Sender<Result<String, io::Error>>) {
    for line in io::stdin().lock().lines() {
        let done = line.is_err();
        if sender.send(line).is_err() || done {
            return;
        }
    }
}

fn drain_worker_events(receiver: &Receiver<AgentEvent>, printer: &mut EventPrinter) {
    loop {
        match receiver.try_recv() {
            Ok(event) => printer.print(event),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => return,
        }
    }
}

fn print_prompt() -> Result<(), io::Error> {
    print!("mh> ");
    io::stdout().flush()
}

fn print_repl_help() {
    println!("/resume  resume the workspace session");
    println!("/session show session state");
    println!("/quit    exit (interrupts an active agent)");
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
        AgentEvent::PreludeLoaded { path, described } => {
            let note = if described {
                ""
            } else {
                " (no //! tool descriptions; the model will not be told about it)"
            };
            eprintln!("[prelude] {}{note}", path.display());
        }
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
        AgentEvent::PtcHostCallStarted { call_id, name } => {
            eprintln!("[ptc:{call_id}] {name}");
        }
        AgentEvent::PtcHostCallCompleted {
            call_id,
            name,
            ok,
            duration_ms,
        } => {
            let status = if ok { "ok" } else { "failed" };
            eprintln!("[ptc:{call_id}] {name}: {status} ({duration_ms} ms)");
        }
        AgentEvent::EvidenceRecorded { kind, ok } => {
            let status = if ok { "pass" } else { "fail" };
            eprintln!("[verify] {kind}: {status}");
        }
        AgentEvent::SteeringQueued { content } => eprintln!("[steering] queued: {content}"),
        AgentEvent::SteeringApplied { content } => eprintln!("[steering] applied: {content}"),
        AgentEvent::RepeatedActionDetected { fingerprint } => {
            eprintln!("[ptc] repeated action detected ({fingerprint})");
        }
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
    println!(
        "active task: {}",
        state
            .active_task
            .map_or_else(|| "none".to_string(), |id| id.0.to_string())
    );
    println!("current revision: {}", state.current_revision.0);
    println!("events: {}", session.events().len());
    if let Some(answer) = session
        .events()
        .iter()
        .rev()
        .find_map(|record| match &record.event {
            SessionEvent::AssistantMessage { content, .. } => Some(content.as_str()),
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
