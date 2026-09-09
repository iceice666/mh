use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use mh::agent::{Agent, AgentConfig, AgentControl, AgentError, AgentEvent, TaskOutcome};
use mh::goal::TaskStatus;
use mh::identity::{ExecutionId, ProcessId, TaskId};
use mh::model::{ModelEvent, OpenAiResponses};
use mh::ptc::{Prelude, TrustDecision};
use mh::runtime::{TaskReport, cancel_task, steer_task, task_reports};
use mh::session::{Session, SessionError, SessionEvent, TaskView, WorkspaceRunnerLock};

/// Subcommand the detach path re-execs itself with. Deliberately undocumented:
/// it is an implementation detail of `--detach`, not a user-facing verb.
const DETACHED_RUNNER: &str = "__run-detached";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Cancelled) => ExitCode::from(130),
        Err(error) => {
            eprintln!("mh: {error}");
            if matches!(error, CliError::Usage(_)) {
                eprintln!("mh: run `mh --help` for usage");
            }
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
    Usage(String),
    NoTask(u64),
    Cancelled,
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::Session(error) => error.fmt(f),
            Self::Agent(error) => error.fmt(f),
            Self::Model(error) => f.write_str(error),
            Self::Usage(message) => f.write_str(message),
            Self::NoTask(id) => write!(f, "no task {id} in this session"),
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = || args[1..].to_vec();

    match args.first().map(String::as_str) {
        None => repl(&workspace, &cancelled),
        Some("--help" | "-h") => {
            print_help();
            Ok(())
        }
        Some("--version" | "-V") => {
            println!("mh {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("run") => cmd_run(&workspace, rest(), &cancelled),
        Some("resume") => cmd_resume(&workspace, rest(), &cancelled),
        Some("attach") => cmd_attach(&workspace, rest(), &cancelled),
        Some("tasks") if args.len() == 1 => cmd_tasks(&workspace),
        Some("inspect") => cmd_inspect(&workspace, rest()),
        Some("steer") => cmd_steer(&workspace, rest()),
        Some("cancel") => cmd_cancel(&workspace, rest()),
        Some("recover") => cmd_recover(&workspace, rest()),
        Some("sessions") if args.len() == 1 => print_session(&workspace),
        Some(DETACHED_RUNNER) => cmd_run_detached(&workspace, rest(), &cancelled),
        // Anything else is the task itself: `mh "fix the tests"`.
        Some(_) => run_foreground(&workspace, &args.join(" "), &cancelled),
    }
}

/// `mh run <task> [--detach]` and the bare `mh "<task>"` form.
fn cmd_run(
    workspace: &Path,
    args: Vec<String>,
    cancelled: &Arc<AtomicBool>,
) -> Result<(), CliError> {
    let detach = args.iter().any(|arg| arg == "--detach");
    let task = args
        .iter()
        .filter(|arg| *arg != "--detach")
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    if task.trim().is_empty() {
        return Err(CliError::Usage("run needs a task".to_string()));
    }
    if detach {
        detach_task(workspace, &task)
    } else {
        run_foreground(workspace, &task, cancelled)
    }
}

fn run_foreground(
    workspace: &Path,
    task: &str,
    cancelled: &Arc<AtomicBool>,
) -> Result<(), CliError> {
    let agent = new_agent()?;
    let mut printer = EventPrinter::new();
    let outcome = agent.run_task_controlled(workspace, task, cancelled, None, &mut |event| {
        printer.print(event);
    })?;
    report_outcome(&outcome);
    Ok(())
}

/// Journals the task, then re-execs this binary to drive it in a child process.
///
/// The task id is allocated before the fork so the parent can report something
/// the user can steer immediately, and so a child that dies before its first
/// model call still leaves a durable, inspectable task behind.
fn detach_task(workspace: &Path, task: &str) -> Result<(), CliError> {
    let task_id = Agent::<OpenAiResponses>::start_detached_task(workspace, task)?;
    let log_path = task_log_path(workspace, task_id);
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let log = fs::File::create(&log_path)?;
    let errors = log.try_clone()?;
    let admission_path = log_path.with_extension(format!("{}.admission", std::process::id()));
    let _ = fs::remove_file(&admission_path);
    let exe = std::env::current_exe()?;
    let mut child = Command::new(exe)
        .arg(DETACHED_RUNNER)
        .arg(task_id.0.to_string())
        .current_dir(workspace)
        // No stdin: a detached run must never block on a prompt, and that is
        // also what makes it refuse an unreviewed prelude instead of hanging.
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(errors))
        .env("MH_DETACHED_ADMISSION_FILE", &admission_path)
        .spawn()?;
    wait_for_detached_admission(&mut child, &admission_path).map_err(|error| {
        CliError::Model(format!(
            "task {} queued but not admitted: {error}",
            task_id.0
        ))
    })?;
    eprintln!("[detach] task {} admitted", task_id.0);
    eprintln!("[detach] log {}", log_path.display());
    eprintln!(
        "[detach] steer with `mh steer {} \"...\"`, watch with `mh inspect {}`",
        task_id.0, task_id.0
    );
    println!("{}", task_id.0);
    Ok(())
}

fn cmd_run_detached(
    workspace: &Path,
    args: Vec<String>,
    cancelled: &Arc<AtomicBool>,
) -> Result<(), CliError> {
    let [id] = args.as_slice() else {
        return Err(CliError::Usage(format!(
            "{DETACHED_RUNNER} needs one task id"
        )));
    };
    let task_id = parse_task_id(id)?;
    let result: Result<(), CliError> = (|| {
        let agent = new_agent()?;
        let mut printer = EventPrinter::new();
        let mut admitted = || {
            let path = std::env::var("MH_DETACHED_ADMISSION_FILE").map_err(|error| {
                AgentError::Session(SessionError::State(format!(
                    "detached admission channel missing: {error}"
                )))
            })?;
            fs::write(path, b"admitted\n").map_err(SessionError::from)?;
            Ok(())
        };
        let outcome = agent.resume_task_admitted(
            workspace,
            Some(task_id),
            cancelled,
            None,
            &mut admitted,
            &mut |event| printer.print(event),
        )?;
        report_outcome(&outcome);
        Ok(())
    })();
    if let Err(error) = &result
        && let Ok(path) = std::env::var("MH_DETACHED_ADMISSION_FILE")
        && !Path::new(&path).exists()
    {
        let _ = fs::write(path, format!("{error}\n"));
    }
    if let Err(error) = &result
        && !matches!(
            error,
            CliError::Session(SessionError::Busy(_))
                | CliError::Agent(AgentError::Session(SessionError::Busy(_)))
        )
        && let Ok(session) = Session::command(workspace)
        && session.task_view(task_id).status().is_active()
    {
        let _ = session.set_status(task_id, TaskStatus::Blocked, Some(error.to_string()));
    }
    result
}

fn wait_for_detached_admission(
    child: &mut std::process::Child,
    path: &Path,
) -> Result<(), CliError> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(message) = fs::read_to_string(path)
            && message.ends_with('\n')
        {
            let _ = fs::remove_file(path);
            if message.trim() == "admitted" {
                return Ok(());
            }
            return Err(CliError::Model(format!(
                "detached admission failed: {}",
                message.trim()
            )));
        }
        if let Some(status) = child.try_wait()? {
            return Err(CliError::Model(format!(
                "detached runner exited before admission ({status})"
            )));
        }
        if Instant::now() >= deadline {
            return Err(CliError::Model(
                "timed out waiting for detached runner admission".to_string(),
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn cmd_resume(
    workspace: &Path,
    args: Vec<String>,
    cancelled: &Arc<AtomicBool>,
) -> Result<(), CliError> {
    let task_id = match args.as_slice() {
        [] => None,
        [id] => Some(parse_task_id(id)?),
        _ => {
            return Err(CliError::Usage(
                "resume takes at most one task id".to_string(),
            ));
        }
    };
    let agent = new_agent()?;
    let mut printer = EventPrinter::new();
    let outcome = agent.resume_task(workspace, task_id, cancelled, None, &mut |event| {
        printer.print(event);
    })?;
    report_outcome(&outcome);
    Ok(())
}

/// `mh attach <task-id>`: report the task's current durable state.
///
/// Streaming attachment is not available without a daemon. Continuing work is
/// an explicit `mh resume`, which must first acquire workspace ownership.
fn cmd_attach(
    workspace: &Path,
    args: Vec<String>,
    _cancelled: &Arc<AtomicBool>,
) -> Result<(), CliError> {
    let [id] = args.as_slice() else {
        return Err(CliError::Usage("attach needs one task id".to_string()));
    };
    let task_id = parse_task_id(id)?;
    let session = Session::inspect(workspace)?;
    if !session.tasks().iter().any(|view| view.task_id == task_id) {
        return Err(CliError::NoTask(task_id.0));
    }
    let view = session.task_view(task_id);
    print_task_view(&view);
    eprintln!(
        "[attach] observation only; use `mh resume {}` to compete for runner ownership.",
        task_id.0
    );
    Ok(())
}

fn cmd_tasks(workspace: &Path) -> Result<(), CliError> {
    let reports = task_reports(workspace)?;
    if reports.is_empty() {
        println!("no tasks");
        return Ok(());
    }
    println!("{:>4}  {:<14}  {:>6}  OBJECTIVE", "ID", "STATUS", "WINDOW");
    for report in &reports {
        print_task_report(report);
    }
    Ok(())
}

fn print_task_report(report: &TaskReport) {
    println!(
        "{:>4}  {:<14}  {:>6}  {}",
        report.task_id.0,
        report.status.label(),
        report.window,
        summarize(&report.objective, 64)
    );
    let workers = report
        .workers
        .iter()
        .filter(|(_, state, _)| !state.is_terminal())
        .count();
    let processes = report
        .processes
        .iter()
        .filter(|(_, state, _)| state == "running")
        .count();
    let mut notes = Vec::new();
    if report.cancel_requested {
        notes.push("cancel requested".to_string());
    }
    if !report.pending_steering.is_empty() {
        notes.push(format!("{} steering queued", report.pending_steering.len()));
    }
    if workers > 0 {
        notes.push(format!("{workers} live worker(s)"));
    }
    if processes > 0 {
        notes.push(format!("{processes} running process(es)"));
    }
    if !report.pending.is_empty() {
        notes.push(format!("{} pending", report.pending.len()));
    }
    if !report.blockers.is_empty() {
        notes.push(format!("{} blocker(s)", report.blockers.len()));
    }
    if !notes.is_empty() {
        println!("{:>4}  {}", "", notes.join(", "));
    }
}

fn cmd_inspect(workspace: &Path, args: Vec<String>) -> Result<(), CliError> {
    let [id] = args.as_slice() else {
        return Err(CliError::Usage("inspect needs one task id".to_string()));
    };
    let task_id = parse_task_id(id)?;
    // Read-only: inspecting a live task must not append drift to its journal.
    let session = Session::inspect(workspace)?;
    if !session.tasks().iter().any(|view| view.task_id == task_id) {
        return Err(CliError::NoTask(task_id.0));
    }
    print_task_view(&session.task_view(task_id));
    Ok(())
}

fn print_task_view(view: &TaskView) {
    let goal = &view.goal;
    println!("task {}", view.task_id.0);
    println!("agent: {}", view.agent.0);
    println!("status: {}", goal.status.label());
    println!("objective: {}", goal.objective);
    println!(
        "window: {} ({} turn(s) so far)",
        view.window, view.turns_in_window
    );
    println!("revision: {}", goal.current_revision.0);
    println!(
        "verified revision: {}",
        goal.last_verified_revision
            .as_ref()
            .map_or("none", |revision| revision.0.as_str())
    );
    if view.cancel_requested {
        println!("cancel: requested");
    }
    if let Some(checkpoint) = &view.checkpoint {
        println!("checkpoint: carried over from window {}", checkpoint.window);
    }
    if let Some(message) = &view.latest_user_message {
        println!("last message: {}", summarize(message, 200));
    }

    print_section("acceptance criteria", &goal.acceptance_criteria, |item| {
        let mark = if item.met { "x" } else { " " };
        match &item.evidence_kind {
            Some(kind) => format!("[{mark}] {} (evidence: {kind})", item.description),
            None => format!("[{mark}] {}", item.description),
        }
    });
    print_section("completed", &goal.completed_work, |item| item.title.clone());
    print_section("pending", &goal.pending_work, |item| match &item.detail {
        Some(detail) => format!("{} — {detail}", item.title),
        None => item.title.clone(),
    });
    print_section("blockers", &goal.blockers, |item| match &item.needs {
        Some(needs) => format!("{} (needs: {needs})", item.summary),
        None => item.summary.clone(),
    });
    print_section("decisions", &goal.decisions, |item| match &item.rationale {
        Some(rationale) => format!("{} — {rationale}", item.decision),
        None => item.decision.clone(),
    });
    print_section("findings", &goal.findings, |item| {
        if item.paths.is_empty() {
            item.summary.clone()
        } else {
            let paths: Vec<String> = item
                .paths
                .iter()
                .map(|path| path.display().to_string())
                .collect();
            format!("{} [{}]", item.summary, paths.join(", "))
        }
    });
    print_section("failed approaches", &goal.failed_approaches, |item| {
        format!("{} — {}", item.approach, item.reason)
    });
    print_section("next actions", &goal.next_actions, Clone::clone);
    print_section("changed paths", &view.changed_paths, |path| {
        path.display().to_string()
    });
    print_section("pending steering", &view.pending_steering, |content| {
        summarize(content, 120)
    });

    let evidence = view.latest_evidence();
    print_section("verification", &evidence, |record| {
        let status = if record.ok { "pass" } else { "fail" };
        let freshness = if view.evidence_is_fresh(record) {
            "fresh"
        } else {
            "stale"
        };
        match &record.note {
            Some(note) => format!("{}: {status} ({freshness}) — {note}", record.kind),
            None => format!("{}: {status} ({freshness})", record.kind),
        }
    });
    print_section("workers", &view.agents, |record| {
        format!(
            "{} {} — {}",
            record.agent.0,
            record.state.label(),
            summarize(&record.objective, 80)
        )
    });
    print_section("processes", &view.processes, |record| {
        let exit = record
            .exit_code
            .map_or_else(String::new, |code| format!(" exit {code}"));
        format!(
            "{} {}{exit} — {}",
            record.process.0,
            record.state,
            record.argv.join(" ")
        )
    });
    print_section(
        "outcome-unknown executions",
        &view.outcome_unknown_executions,
        |execution| execution.0.to_string(),
    );
    print_section(
        "unresolved recovery workspaces",
        &view.unresolved_recovery_workspaces,
        u64::to_string,
    );
}

fn print_section<T>(title: &str, items: &[T], render: impl Fn(&T) -> String) {
    if items.is_empty() {
        return;
    }
    println!("{title}:");
    for item in items {
        println!("  {}", render(item));
    }
}

fn cmd_steer(workspace: &Path, args: Vec<String>) -> Result<(), CliError> {
    let (id, message) = match args.split_first() {
        Some((id, rest)) if !rest.is_empty() => (id, rest.join(" ")),
        _ => {
            return Err(CliError::Usage(
                "steer needs a task id and a message".to_string(),
            ));
        }
    };
    let receipt = steer_task(workspace, Some(parse_task_id(id)?), &message)?;
    eprintln!(
        "[steering] queued for task {} as command {} (event {})",
        receipt.task_id.0, receipt.command_id, receipt.seq
    );
    if let Some(warning) = &receipt.cache_warning {
        eprintln!("[steering] warning: {warning}");
    }
    Ok(())
}

fn cmd_cancel(workspace: &Path, args: Vec<String>) -> Result<(), CliError> {
    let task_id = match args.as_slice() {
        [] => None,
        [id] => Some(parse_task_id(id)?),
        _ => {
            return Err(CliError::Usage(
                "cancel takes at most one task id".to_string(),
            ));
        }
    };
    let receipt = cancel_task(workspace, task_id)?;
    eprintln!(
        "[cancel] requested for task {} as command {} (event {})",
        receipt.task_id.0, receipt.command_id, receipt.seq
    );
    if let Some(warning) = &receipt.cache_warning {
        eprintln!("[cancel] warning: {warning}");
    }
    Ok(())
}

fn cmd_recover(workspace: &Path, args: Vec<String>) -> Result<(), CliError> {
    let [task, kind, id, reason @ ..] = args.as_slice() else {
        return Err(CliError::Usage(
            "recover needs <task-id> <execution|process> <resource-id> <reason>".to_string(),
        ));
    };
    if reason.is_empty() {
        return Err(CliError::Usage("recover needs an audit reason".to_string()));
    }
    let task_id = parse_task_id(task)?;
    let resource_id = id
        .parse::<u64>()
        .map_err(|_| CliError::Usage(format!("`{id}` is not a resource id")))?;
    let (execution_id, process_id) = match kind.as_str() {
        "execution" => (Some(ExecutionId(resource_id)), None),
        "process" => (None, Some(ProcessId(resource_id))),
        _ => {
            return Err(CliError::Usage(
                "recover resource kind must be `execution` or `process`".to_string(),
            ));
        }
    };
    let lock = WorkspaceRunnerLock::acquire(workspace)?;
    let session = Session::resume(workspace)?;
    let runner = session.admit_runner(lock, Some(task_id))?;
    session.recover_previous_owner(runner.info().generation)?;
    session.refresh()?;
    let record = runner.resolve_recovery(task_id, execution_id, process_id, reason.join(" "))?;
    eprintln!(
        "[recovery] resolved {kind} {resource_id} for task {} (event {})",
        task_id.0, record.seq
    );
    Ok(())
}

fn parse_task_id(raw: &str) -> Result<TaskId, CliError> {
    raw.parse::<u64>()
        .map(TaskId)
        .map_err(|_| CliError::Usage(format!("`{raw}` is not a task id")))
}

fn task_log_path(workspace: &Path, task_id: TaskId) -> PathBuf {
    workspace
        .join(".mh")
        .join("tasks")
        .join(format!("{}.log", task_id.0))
}

/// Reports how a run ended. A parked task and a completed one are different
/// facts — one needs the user, the other does not — so they never share wording.
fn report_outcome(outcome: &TaskOutcome) {
    if outcome.is_complete() {
        eprintln!("[task] complete");
    } else {
        eprintln!("[task] parked — waiting for you; reply, or run `mh resume`");
    }
    println!("{}", outcome.message());
}

fn summarize(text: &str, max: usize) -> String {
    let line = text.lines().next().unwrap_or_default().trim();
    let truncated = text.lines().nth(1).is_some();
    if line.chars().count() <= max {
        return if truncated {
            format!("{line} …")
        } else {
            line.to_string()
        };
    }
    let head: String = line.chars().take(max).collect();
    format!("{head} …")
}

fn new_agent() -> Result<Agent<OpenAiResponses>, CliError> {
    let model = OpenAiResponses::from_env().map_err(|error| CliError::Model(error.to_string()))?;
    Ok(Agent::new(model, agent_config()))
}

/// Supplies the interactive prelude confirmation.
///
/// Only attached when both stdin and stderr are terminals. A piped or CI run
/// therefore refuses an unreviewed workspace prelude rather than blocking on a
/// prompt nobody can answer, and never consumes task input as an answer.
fn agent_config() -> AgentConfig {
    let mut config = AgentConfig::default();
    if io::stdin().is_terminal() && io::stderr().is_terminal() {
        config.confirm_prelude = Some(Arc::new(confirm_prelude));
    }
    config
}

fn confirm_prelude(prelude: &Prelude) -> TrustDecision {
    let lines = prelude.source.lines().count();
    eprintln!("\n[prelude] {} is not yet trusted.", prelude.path.display());
    eprintln!(
        "  It runs before every PTC program with your own capabilities: it can\n  \
         read and write this workspace and start subprocesses."
    );
    eprintln!("  {} lines, {}", lines, prelude.identity().0);
    match prelude.description() {
        Some(tools) => {
            eprintln!("  Advertised tools:");
            for line in tools.lines() {
                eprintln!("    {line}");
            }
        }
        None => eprintln!("  It advertises no tools to the model."),
    }
    eprintln!("  Review it before trusting. The decision is remembered for this");
    eprintln!("  exact content; editing the prelude asks again.");

    // Read from the terminal directly: in the REPL a reader thread owns stdin,
    // and this runs before that thread starts.
    loop {
        eprint!("  Trust this prelude? [y/N] ");
        if io::stderr().flush().is_err() {
            return TrustDecision::Rejected;
        }
        let mut answer = String::new();
        match io::stdin().read_line(&mut answer) {
            // EOF: no answer is not consent.
            Ok(0) => return TrustDecision::Rejected,
            Ok(_) => {}
            Err(_) => return TrustDecision::Rejected,
        }
        match answer.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => return TrustDecision::Trusted,
            "" | "n" | "no" => return TrustDecision::Rejected,
            _ => eprintln!("  Answer y or n."),
        }
    }
}

fn repl(workspace: &Path, cancelled: &Arc<AtomicBool>) -> Result<(), CliError> {
    println!("mh — PTC coding agent. Type /help for commands.");
    let (input_tx, input_rx) = mpsc::channel();
    std::thread::spawn(move || read_stdin(input_tx));
    let (event_tx, event_rx) = mpsc::channel();
    let mut worker: Option<AgentWorker> = None;
    let mut printer = EventPrinter::new();
    print_prompt()?;

    loop {
        drain_worker_events(&event_rx, &mut printer);
        if worker
            .as_ref()
            .is_some_and(|worker| worker.handle.is_finished())
        {
            let finished = worker.take().expect("worker was present");
            match finished.handle.join() {
                Ok(Ok(outcome)) => report_outcome(&outcome),
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
                        "/resume" => eprintln!("[agent] already running"),
                        // Inspection and durable cancellation stay available
                        // while the agent works; anything else is steering.
                        _ if repl_shared_command(workspace, message)? => {}
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
                    "/resume" => {
                        worker = Some(spawn_agent_worker(
                            workspace.to_path_buf(),
                            None,
                            cancelled.clone(),
                            event_tx.clone(),
                        )?);
                    }
                    _ if repl_shared_command(workspace, message)? => {}
                    _ => {
                        worker = Some(spawn_agent_worker(
                            workspace.to_path_buf(),
                            Some(message.to_string()),
                            cancelled.clone(),
                            event_tx.clone(),
                        )?);
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

/// Handles the REPL commands that mean the same thing whether or not an agent
/// is running. Returns whether the line was one of them.
fn repl_shared_command(workspace: &Path, message: &str) -> Result<bool, CliError> {
    let mut words = message.split_whitespace();
    let Some(command) = words.next() else {
        return Ok(false);
    };
    let args: Vec<String> = words.map(str::to_string).collect();
    match command {
        "/help" => print_repl_help(),
        "/session" => report(print_session(workspace)),
        "/tasks" if args.is_empty() => report(cmd_tasks(workspace)),
        "/inspect" => {
            let args = match args.as_slice() {
                [] => match repl_root_task(workspace)? {
                    Some(task_id) => vec![task_id.0.to_string()],
                    None => {
                        eprintln!("[inspect] no task yet");
                        return Ok(true);
                    }
                },
                _ => args,
            };
            report(cmd_inspect(workspace, args));
        }
        "/cancel" => report(cmd_cancel(workspace, args)),
        _ => return Ok(false),
    }
    Ok(true)
}

fn repl_root_task(workspace: &Path) -> Result<Option<TaskId>, CliError> {
    match Session::inspect(workspace) {
        Ok(session) => Ok(session.root_task()),
        Err(SessionError::NoSession(_)) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Reports a failed REPL command without ending the session: a typo or a
/// missing session is not a reason to drop the user's context.
fn report(result: Result<(), CliError>) {
    if let Err(error) = result {
        eprintln!("mh: {error}");
    }
}

struct AgentWorker {
    controls: Sender<AgentControl>,
    handle: JoinHandle<Result<TaskOutcome, AgentError>>,
}

fn spawn_agent_worker(
    workspace: PathBuf,
    message: Option<String>,
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
        match message {
            Some(message) => agent.send_message_controlled(
                &workspace,
                &message,
                &cancelled,
                Some(&control_rx),
                &mut emit,
            ),
            None => agent.resume_controlled(&workspace, &cancelled, Some(&control_rx), &mut emit),
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
    println!("/resume          resume the workspace session");
    println!("/tasks           list durable tasks");
    println!("/inspect [<id>]  full durable state of a task (default: this session's)");
    println!("/cancel [<id>]   request durable cancellation");
    println!("/session         show session state");
    println!("/help            this list");
    println!("/quit            exit (interrupts an active agent)");
    println!();
    println!("Any other line starts a task, or steers the one already running.");
}

struct EventPrinter {
    reasoning_open: bool,
}

impl EventPrinter {
    const fn new() -> Self {
        Self {
            reasoning_open: false,
        }
    }

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
            AgentEvent::Model(ModelEvent::ReasoningSummaryDone) => self.close_reasoning(),
            event => {
                // A streamed reasoning line has no newline yet; anything else
                // printing over it would splice two messages together.
                self.close_reasoning();
                print_event(event);
            }
        }
    }

    fn close_reasoning(&mut self) {
        if self.reasoning_open {
            eprintln!();
            self.reasoning_open = false;
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
        AgentEvent::PreludeRejected { path, reason } => {
            eprintln!(
                "[prelude] not loaded: {} — {}",
                path.display(),
                reason.explain()
            );
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
        AgentEvent::ContextWindowStarted { window } => eprintln!("[window] {window}"),
        AgentEvent::Compacted { window, tokens } => {
            eprintln!("[compact] window {window} -> ~{tokens} tokens");
        }
        AgentEvent::AssistantMessage { content } => eprintln!("[assistant] {content}"),
        AgentEvent::FinishProposed {
            accepted,
            objections,
        } => {
            if accepted {
                eprintln!("[finish] accepted");
            } else {
                eprintln!("[finish] refused");
                for objection in objections {
                    eprintln!("  {objection}");
                }
            }
        }
        AgentEvent::TaskCompleted { summary } => eprintln!("[task] completed: {summary}"),
        AgentEvent::GoalUpdated => eprintln!("[goal] updated"),
        AgentEvent::AgentSpawned { agent, objective } => {
            eprintln!("[worker {}] spawned: {objective}", agent.0);
        }
        AgentEvent::AgentSettled { agent, state } => {
            eprintln!("[worker {}] {}", agent.0, state.label());
        }
        AgentEvent::ProcessSpawned { process, argv } => {
            eprintln!("[process {}] {}", process.0, argv.join(" "));
        }
        AgentEvent::ProcessExited {
            process,
            exit_code: Some(code),
        } => eprintln!("[process {}] exited {code}", process.0),
        AgentEvent::ProcessExited {
            process,
            exit_code: None,
        } => eprintln!("[process {}] exited without a status", process.0),
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
    let session = Session::inspect(workspace)?;
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
    println!("tasks: {}", session.tasks().len());
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
    println!("mh — durable PTC coding agent");
    println!();
    println!("USAGE:");
    println!("  mh                              open an interactive session");
    println!("  mh \"fix the tests\"              run a task in the foreground");
    println!("  mh run <task> [--detach]        register and request task admission");
    println!("  mh resume [<task-id>]           acquire workspace ownership and continue");
    println!("  mh attach <task-id>             show an observation-only task snapshot");
    println!("  mh tasks                        list durable tasks without mutation");
    println!("  mh inspect <task-id>            full read-only durable state");
    println!("  mh steer <task-id> <message>    queue durable steering with a receipt");
    println!("  mh cancel [<task-id>]           queue durable cancellation with a receipt");
    println!("  mh recover <task-id> <execution|process> <id> <reason>");
    println!("  mh sessions                     inspect the workspace session");
    println!("  mh --help | --version");
    println!();
    println!("ENVIRONMENT:");
    println!("  MH_API_KEY or OPENAI_API_KEY");
    println!("  MH_BASE_URL             Responses API root; default: https://api.openai.com/v1");
    println!("  MH_MODEL                default: gpt-4.1-mini");
    println!("  MH_MODEL_TIMEOUT_SECS   default: 300");
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADMISSION_CHILD: &str = "MH_ADMISSION_WAIT_CHILD";

    #[test]
    fn admission_wait_child() {
        if std::env::var_os(ADMISSION_CHILD).is_some() {
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    #[test]
    fn detached_admission_waits_for_a_complete_frame() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("admission");
        fs::write(&path, b"admitted").unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::admission_wait_child", "--nocapture"])
            .env(ADMISSION_CHILD, "1")
            .spawn()
            .unwrap();
        let completed = path.clone();
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            let mut file = fs::OpenOptions::new().append(true).open(completed).unwrap();
            file.write_all(b"\n").unwrap();
        });

        wait_for_detached_admission(&mut child, &path).unwrap();

        writer.join().unwrap();
        assert!(!path.exists());
        let status = child.wait().unwrap();
        assert!(status.success());
    }
}
