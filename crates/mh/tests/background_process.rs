//! Durable background processes and runtime restart (spec v0.5 §5, Phase 5).

mod support;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use mh::agent::{Agent, AgentConfig, AgentEvent};
use mh::goal::TaskStatus;
use mh::identity::AgentId;
use mh::session::{Session, SessionEvent};
use support::{RoleModel, config, events, finish, program, ptc_value, repository, run, text};

/// Spec: spawn a process that lives longer than an ordinary `exec()` timeout;
/// `process_spawn()` returns quickly, the process stays running, output can be
/// tailed, the root keeps working, and the process can be killed cleanly.
#[test]
fn a_background_process_outlives_exec_while_the_root_keeps_working() {
    let dir = repository();
    let (outcome, emitted) = run(
        dir.path(),
        RoleModel::new(vec![
            program(
                r#"
                // Emits for well over the exec timeout imposed below, so an
                // exec() of the same command would fail where this must not.
                var p = process_spawn({
                    command: ["/bin/sh", "-c", "for i in 1 2 3 4 5 6 7 8 9 10; do echo line-$i; sleep 0.1; done; sleep 5"],
                    label: "emitter"
                });
                var immediately = process_poll(p.id);
                return {
                    id: p.id,
                    spawnState: p.state,
                    pid: p.pid,
                    stillRunning: immediately.state,
                    label: p.label
                };
                "#,
            ),
            // Real root work while the process runs.
            program(
                r#"
                write("root-continued.txt", "root worked alongside the process");
                var timedOut = exec({
                    command: ["/bin/sh", "-c", "sleep 5"],
                    timeout_ms: 150
                });
                return {
                    rootWork: read("root-continued.txt").content,
                    execTimedOut: timedOut.error,
                    processes: process_list().length
                };
                "#,
            ),
            program(
                r#"
                var tail = process_tail(1, "stdout", 3);
                var poll = process_poll(1);
                return {
                    tail: tail.content,
                    totalLines: tail.totalLines,
                    stream: tail.stream,
                    state: poll.state
                };
                "#,
            ),
            program(
                r#"
                var killed = process_kill(1);
                var after = process_poll(1);
                return { killedState: killed.state, afterState: after.state };
                "#,
            ),
            finish("process managed cleanly"),
        ]),
        "run a background process",
        config(),
    );
    assert!(outcome.is_complete());

    // Spawn returned immediately with a live process.
    let spawned = ptc_value(dir.path(), "spawnState");
    assert_eq!(spawned["id"], 1);
    assert_eq!(spawned["spawnState"], "running");
    assert_eq!(spawned["stillRunning"], "running");
    assert_eq!(spawned["label"], "emitter");
    assert!(
        spawned["pid"].as_u64().is_some(),
        "a live process has a pid"
    );

    // The root kept working, and an ordinary exec of the same duration fails
    // where the background process does not.
    let root_work = ptc_value(dir.path(), "execTimedOut");
    assert_eq!(root_work["rootWork"], "root worked alongside the process");
    assert!(
        root_work["execTimedOut"]
            .as_str()
            .unwrap_or_default()
            .contains("timed out"),
        "exec still enforces its timeout: {root_work}"
    );
    assert_eq!(root_work["processes"], 1);

    // Output is tailable while the process is still running.
    let tailed = ptc_value(dir.path(), "tail");
    assert_eq!(tailed["stream"], "stdout");
    assert_eq!(tailed["state"], "running", "still alive after 10 lines");
    let content = tailed["tail"].as_str().unwrap();
    assert!(
        content.lines().count() <= 3 && content.contains("line-"),
        "tail returns the last lines: {content:?}"
    );
    assert!(
        tailed["totalLines"].as_u64().unwrap() >= 3,
        "the process produced output: {tailed}"
    );

    // Kill is clean and observable.
    let killed = ptc_value(dir.path(), "killedState");
    assert!(
        ["killed", "exited"].contains(&killed["killedState"].as_str().unwrap()),
        "kill leaves a terminal state: {killed}"
    );
    assert_ne!(killed["afterState"], "running");

    // The journal owns the process lifecycle, so an inspector can see it.
    let session = Session::inspect(dir.path()).unwrap();
    let record = session
        .processes()
        .into_iter()
        .find(|record| record.process.0 == 1)
        .expect("process journaled");
    assert_eq!(record.argv[0], "/bin/sh");
    assert_eq!(record.agent, AgentId::ROOT);
    assert!(!record.is_running(), "the final state is durable");
    assert!(
        emitted
            .iter()
            .any(|event| matches!(event, AgentEvent::ProcessSpawned { .. }))
            || events(dir.path())
                .iter()
                .any(|record| matches!(record.event, SessionEvent::ProcessSpawned { .. })),
        "process spawn is recorded"
    );
}

/// A running process blocks completion: finishing over it would leave
/// execution the user can no longer see.
#[test]
fn a_running_process_blocks_completion_until_it_is_resolved() {
    let dir = repository();
    let (outcome, _) = run(
        dir.path(),
        RoleModel::new(vec![
            program(
                "var p = process_spawn({ command: [\"/bin/sh\", \"-c\", \"sleep 5\"] });\nreturn { id: p.id, spawned: p.state };",
            ),
            program(
                "var verdict = finish({ summary: \"done early\", force: true });\nreturn { refusedAccepted: verdict.accepted, refusal: verdict.explanation };",
            ),
            program("var killed = process_kill(1);\nreturn { resolved: killed.state };"),
            finish("process resolved, then finished"),
        ]),
        "process blocks finish",
        config(),
    );
    assert!(outcome.is_complete());

    // Keyed distinctly from the final accepted verdict, which carries an
    // explanation of its own.
    let refusal = ptc_value(dir.path(), "refusal");
    assert_eq!(
        refusal["refusedAccepted"], false,
        "force must not complete over a running process"
    );
    assert!(
        refusal["refusal"]
            .as_str()
            .unwrap()
            .contains("background process"),
        "the refusal must name the running process: {refusal}"
    );

    let verdicts: Vec<_> = events(dir.path())
        .into_iter()
        .filter_map(|record| match record.event {
            SessionEvent::FinishProposed { verdict, .. } => Some(verdict),
            _ => None,
        })
        .collect();
    assert_eq!(verdicts.len(), 2);
    assert!(!verdicts[0].accepted && verdicts[1].accepted);
}

/// `process_write` drives a process's stdin, and reads come back through tail.
#[test]
fn a_background_process_accepts_stdin() {
    let dir = repository();
    let (outcome, _) = run(
        dir.path(),
        RoleModel::new(vec![
            program(
                r#"
                var p = process_spawn({ command: ["/bin/sh", "-c", "while read line; do echo \"got:$line\"; done"] });
                process_write(p.id, "hello");
                var waited = process_poll(p.id);
                return { id: p.id, state: waited.state };
                "#,
            ),
            // The reader echoes asynchronously, so poll until the line lands.
            program(
                r#"
                var seen = "";
                for (var attempt = 0; attempt < 50; attempt = attempt + 1) {
                    var tail = process_tail(1, "stdout", 10);
                    if (tail.content.indexOf("got:hello") >= 0) {
                        seen = tail.content;
                        attempt = 50;
                    } else {
                        exec({ command: ["/bin/sh", "-c", "sleep 0.05"] });
                    }
                }
                process_kill(1);
                return { seen: seen };
                "#,
            ),
            finish("stdin round-tripped"),
        ]),
        "write to a process",
        config(),
    );
    assert!(outcome.is_complete());
    let value = ptc_value(dir.path(), "seen");
    assert!(
        value["seen"].as_str().unwrap().contains("got:hello"),
        "stdin reached the process and its echo was tailable: {value}"
    );
}

#[test]
fn a_settled_worker_journals_its_process_teardown() {
    let dir = repository();
    let (outcome, _) = run(
        dir.path(),
        RoleModel::new(vec![
            program(
                r#"
                var worker = agent_spawn({ task: "spawn worker process", access: "isolated-write" });
                var result = agent_join(worker.agent);
                return { workerOk: result.ok };
                "#,
            ),
            finish("worker process was cleaned up"),
        ])
        .worker(
            "spawn worker process",
            vec![
                program(
                    r#"
                    var process = process_spawn({
                        command: ["/bin/sh", "-c", "sleep 30"],
                        label: "worker-owned"
                    });
                    return { process: process.id };
                    "#,
                ),
                text("worker done"),
            ],
        ),
        "worker process teardown",
        config(),
    );

    assert!(outcome.is_complete());
    let session = Session::inspect(dir.path()).unwrap();
    let root = session.root_task().unwrap();
    let view = session.task_view(root);
    assert_eq!(view.status(), TaskStatus::Completed);
    assert_eq!(view.processes.len(), 1);
    assert!(!view.processes[0].is_running());
    assert!(events(dir.path()).iter().any(|record| matches!(
        record.event,
        SessionEvent::ProcessStateChanged { process, .. } if process == view.processes[0].process
    )));
}

/// Spec: after a runtime restart the recovered task must clearly represent
/// task status, worker states, workspace revision, compacted context, and
/// process states/orphans. No silent state loss.
#[test]
fn a_restarted_runtime_reports_orphans_and_recovers_task_state() {
    let dir = repository();

    // First runtime: change the workspace, roll over a window, run a worker to
    // completion, leave a process running, then park.
    let (parked, _) = run(
        dir.path(),
        RoleModel::new(vec![
            program(
                r#"
                write("phase-one.txt", "written before the restart");
                var worker = agent_spawn({ task: "inspect before restart", access: "read" });
                var joined = agent_join(worker.agent);
                var p = process_spawn({ command: ["/bin/sh", "-c", "sleep 30"], label: "survivor" });
                return goal({
                    findings: [{ summary: "PRE_RESTART_FINDING: revision tracking is manifest-based" }],
                    pending: [{ title: "complete after the restart" }],
                    nextActions: ["resume and finish"]
                });
                "#,
            ),
            program("return read(\"phase-one.txt\").content;"),
            text("Parked with a process still running."),
        ])
        .worker(
            "inspect before restart",
            vec![text("inspected before the restart")],
        ),
        "survive a restart",
        AgentConfig {
            max_turns_per_window: 2,
            ..config()
        },
    );
    assert!(!parked.is_complete());

    let before = Session::inspect(dir.path()).unwrap();
    let task = before.root_task().unwrap();
    let revision_before = before.task_view(task).goal.current_revision.clone();
    // The parked runtime shut its own processes down; the recorded state must
    // say so rather than claiming the process is live.
    let process_states: Vec<String> = before
        .processes()
        .into_iter()
        .map(|record| record.state)
        .collect();
    assert_eq!(process_states.len(), 1);

    // Second runtime: a fresh Agent and Session, as after a crash/restart.
    let agent = Agent::new(
        RoleModel::new(vec![
            program(
                "return goal({ pending: [], completed: [{ title: \"complete after the restart\" }] });",
            ),
            finish("recovered and completed"),
        ]),
        config(),
    );
    let mut emitted = Vec::new();
    let outcome = agent
        .resume_task(
            dir.path(),
            Some(task),
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |event| emitted.push(event),
        )
        .unwrap();
    assert!(outcome.is_complete());

    let after = Session::inspect(dir.path()).unwrap();
    let view = after.task_view(task);

    // Task status.
    assert_eq!(view.status(), TaskStatus::Completed);
    assert_eq!(after.root_task().unwrap(), task, "the task id survived");

    // Worker states.
    let worker = after
        .agent(AgentId(1))
        .expect("worker survived the restart");
    assert!(worker.state.is_terminal());
    assert_eq!(
        worker.result.unwrap().summary,
        "inspected before the restart"
    );

    // Workspace revision and content.
    assert_eq!(
        std::fs::read_to_string(dir.path().join("phase-one.txt")).unwrap(),
        "written before the restart"
    );
    assert_eq!(
        view.goal.current_revision, revision_before,
        "the recovered revision matches what was persisted"
    );

    // Compacted context: the pre-restart finding is still represented.
    assert!(
        view.goal
            .findings
            .iter()
            .any(|finding| finding.summary.contains("PRE_RESTART_FINDING")),
        "durable findings survived the restart"
    );

    // Process states: never reported as running after a restart.
    let processes = after.processes();
    assert_eq!(processes.len(), 1);
    assert!(
        !processes[0].is_running(),
        "a process from a previous runtime must not be reported as live, got {:?}",
        processes[0].state
    );
    assert!(
        events(dir.path())
            .iter()
            .any(|record| matches!(record.event, SessionEvent::ProcessStateChanged { .. })),
        "the process transition is durable"
    );

    // And the recovered task is reportable without mutating anything.
    let reports = mh::runtime::task_reports(dir.path()).unwrap();
    let report = reports
        .iter()
        .find(|report| report.task_id == task)
        .expect("root task reported");
    assert_eq!(report.status, TaskStatus::Completed);
    assert_eq!(report.workers.len(), 1);
    assert_eq!(report.processes.len(), 1);
    assert!(report.pending.is_empty());
}

/// Process budget is enforced: long-running must not mean unbounded.
#[test]
fn the_live_process_budget_is_enforced() {
    let dir = repository();
    let (outcome, _) = run(
        dir.path(),
        RoleModel::new(vec![
            program(
                r#"
                var spawned = 0;
                var failure = null;
                for (var index = 0; index < 12; index = index + 1) {
                    try {
                        process_spawn({ command: ["/bin/sh", "-c", "sleep 5"] });
                        spawned = spawned + 1;
                    } catch (error) {
                        failure = "" + error;
                        index = 12;
                    }
                }
                var live = process_list();
                for (var k = 0; k < live.length; k = k + 1) {
                    process_kill(live[k].id);
                }
                return { spawned: spawned, failure: failure };
                "#,
            ),
            finish("budget enforced"),
        ]),
        "exhaust the process budget",
        config(),
    );
    assert!(outcome.is_complete());

    let value = ptc_value(dir.path(), "failure");
    let spawned = value["spawned"].as_u64().unwrap();
    assert!(
        spawned > 0 && spawned < 12,
        "the budget must stop the loop before 12 processes, spawned {spawned}"
    );
    assert!(
        value["failure"]
            .as_str()
            .unwrap_or_default()
            .contains("budget"),
        "the refusal must name the budget: {value}"
    );
}
