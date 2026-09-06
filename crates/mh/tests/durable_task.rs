//! Long-horizon durable-task behaviour (spec v0.5 §"Tests required").
//!
//! These cover explicit finish, multi-window rollover, and resume across a
//! restart. The model is faked; the agent loop, journal, and PTC runtime are
//! real.

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use mh::agent::{Agent, AgentConfig, AgentEvent, TaskOutcome};
use mh::goal::TaskStatus;
use mh::session::{Session, SessionEvent};
use support::{RoleModel, config, events, finish, program, run, text};

/// Spec: "Model emits 'Implemented most of it.' — task must remain active.
/// Only finish() completes it."
#[test]
fn text_output_alone_cannot_complete_a_task() {
    let dir = tempfile::tempdir().unwrap();
    let (outcome, emitted) = run(
        dir.path(),
        RoleModel::new(vec![
            program("return write(\"progress.txt\", \"partial\");"),
            text("Implemented most of it."),
        ]),
        "refactor the storage layer",
        config(),
    );

    assert_eq!(
        outcome,
        TaskOutcome::AwaitingUser("Implemented most of it.".to_string()),
        "text parks the task, it does not complete it"
    );
    assert!(!outcome.is_complete());
    assert!(
        emitted.iter().any(|event| matches!(
            event,
            AgentEvent::AssistantMessage { content } if content == "Implemented most of it."
        )),
        "the progress report reaches the caller as a message"
    );
    assert!(
        !emitted
            .iter()
            .any(|event| matches!(event, AgentEvent::TaskCompleted { .. })),
        "no completion event may be emitted for plain text"
    );

    let session = Session::inspect(dir.path()).unwrap();
    let task = session.root_task().unwrap();
    let view = session.task_view(task);
    assert_eq!(view.status(), TaskStatus::WaitingUser);
    assert!(view.status().is_active(), "the task is still owed work");
    assert!(
        !events(dir.path())
            .iter()
            .any(|record| matches!(record.event, SessionEvent::TaskCompleted { .. })),
        "the journal must record no completion"
    );

    // The same durable task then completes only through finish().
    let agent = Agent::new(RoleModel::new(vec![finish("storage refactored")]), config());
    let completed = agent
        .send_message_controlled(
            dir.path(),
            "carry on",
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(
        completed,
        TaskOutcome::Completed("storage refactored".to_string())
    );

    let session = Session::inspect(dir.path()).unwrap();
    assert_eq!(
        session.tasks().len(),
        1,
        "the follow-up continued the same durable task"
    );
    assert_eq!(session.task_view(task).status(), TaskStatus::Completed);
}

/// Spec: force enough turns to require at least two rollovers, then assert the
/// task id, goal, pending work, and workspace state all survive.
#[test]
fn a_task_survives_multiple_context_rollovers() {
    let dir = tempfile::tempdir().unwrap();
    let mut root = vec![
        program(
            r#"
            write("window-one.txt", "created in window one");
            return goal({
                objective: "refactor the storage layer",
                pending: [{ title: "update all callers" }, { title: "add tests" }],
                findings: [{ summary: "EARLY_FINDING: storage sits behind StorageBackend" }],
                decisions: [{ decision: "keep the trait object", rationale: "callers are dynamic" }],
                failedApproaches: [{ approach: "generic backend", reason: "not object safe" }]
            });
            "#,
        ),
        program("return read(\"window-one.txt\").content;"),
    ];
    // Two turns per window: these four fill windows 0 and 1.
    for index in 0..4 {
        root.push(program(&format!("return {index} + 1;")));
    }
    root.push(program(
        "return goal({ pending: [], completed: [{ title: \"update all callers\" }, { title: \"add tests\" }] });",
    ));
    root.push(finish("refactor complete across windows"));

    let (outcome, emitted) = run(
        dir.path(),
        RoleModel::new(root),
        "refactor the storage layer",
        AgentConfig {
            max_turns_per_window: 2,
            ..config()
        },
    );
    assert!(outcome.is_complete());

    let windows: Vec<u32> = emitted
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ContextWindowStarted { window } => Some(*window),
            _ => None,
        })
        .collect();
    let compactions = emitted
        .iter()
        .filter(|event| matches!(event, AgentEvent::Compacted { .. }))
        .count();
    assert!(
        compactions >= 2,
        "expected at least two rollovers, saw {compactions} (windows {windows:?})"
    );
    assert_eq!(
        windows,
        (0..=windows.len() as u32 - 1).collect::<Vec<_>>(),
        "windows advance one at a time without gaps"
    );

    let session = Session::inspect(dir.path()).unwrap();
    let tasks = session.tasks();
    assert_eq!(tasks.len(), 1, "the same task id survived every rollover");
    let view = &tasks[0];
    assert_eq!(
        view.objective(),
        "refactor the storage layer",
        "the goal survived rollover"
    );
    assert_eq!(view.status(), TaskStatus::Completed);
    assert!(view.window >= 2);

    // Semantic state, not a transcript: an early finding is still represented.
    let checkpoint = view
        .checkpoint
        .as_ref()
        .expect("a rolled-over task carries a checkpoint");
    assert!(
        checkpoint
            .important_findings
            .iter()
            .any(|finding| finding.summary.contains("EARLY_FINDING")),
        "a window-one finding must survive to the final window"
    );
    assert!(
        checkpoint
            .failed_approaches
            .iter()
            .any(|failure| failure.approach == "generic backend"),
        "failed approaches must survive so they are not retried"
    );
    assert!(
        checkpoint
            .decisions
            .iter()
            .any(|decision| decision.decision == "keep the trait object"),
        "decisions must survive rollover"
    );
    assert_eq!(
        checkpoint.pending.len(),
        2,
        "pending work recorded in window one reached the final window"
    );
    // The final window's own goal updates land in live goal state; only the
    // next rollover would fold them into a checkpoint.
    assert_eq!(
        view.goal
            .completed_work
            .iter()
            .map(|item| item.title.as_str())
            .collect::<Vec<_>>(),
        vec!["update all callers", "add tests"]
    );
    assert!(
        view.goal.pending_work.is_empty(),
        "the model cleared its pending work before finishing"
    );

    // Workspace state survives too, and is real on disk.
    assert_eq!(
        std::fs::read_to_string(dir.path().join("window-one.txt")).unwrap(),
        "created in window one"
    );
    assert!(
        view.changed_paths
            .iter()
            .any(|path| path.ends_with("window-one.txt")),
        "changed paths survive rollover: {:?}",
        view.changed_paths
    );
}

/// Spec: interrupt/restart during an active task, then resume; the task
/// continues from persisted semantic state instead of starting over.
#[test]
fn resume_after_a_restart_continues_from_persisted_state() {
    let dir = tempfile::tempdir().unwrap();

    // First process: record durable state, then park.
    let (parked, _) = run(
        dir.path(),
        RoleModel::new(vec![
            program(
                r#"
                write("started.txt", "phase one");
                return goal({
                    pending: [{ title: "finish phase two" }],
                    findings: [{ summary: "PHASE_ONE_FINDING: the parser owns precedence" }],
                    nextActions: ["resume and complete phase two"]
                });
                "#,
            ),
            text("Phase one done; pausing."),
        ]),
        "two-phase task",
        AgentConfig {
            max_turns_per_window: 2,
            ..config()
        },
    );
    assert!(!parked.is_complete());
    let first_task = Session::inspect(dir.path()).unwrap().root_task().unwrap();

    // Second process: a fresh Agent and a fresh Session, as after a restart.
    let resumed_model = RoleModel::new(vec![
        program("return goal({ pending: [], completed: [{ title: \"finish phase two\" }] });"),
        finish("both phases complete"),
    ]);
    let agent = Agent::new(resumed_model, config());
    let mut emitted = Vec::new();
    let outcome = agent
        .resume_controlled(
            dir.path(),
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |event| emitted.push(event),
        )
        .unwrap();
    assert_eq!(
        outcome,
        TaskOutcome::Completed("both phases complete".to_string())
    );

    let session = Session::inspect(dir.path()).unwrap();
    assert_eq!(
        session.root_task().unwrap(),
        first_task,
        "resume continued the same durable task rather than starting over"
    );
    assert_eq!(session.tasks().len(), 1);
    let view = session.task_view(first_task);
    assert_eq!(view.objective(), "two-phase task");
    assert_eq!(view.status(), TaskStatus::Completed);
    assert!(
        view.goal
            .findings
            .iter()
            .any(|finding| finding.summary.contains("PHASE_ONE_FINDING")),
        "a pre-restart finding is still durable after resume"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("started.txt")).unwrap(),
        "phase one",
        "workspace state persisted across the restart"
    );
}

/// A refused finish is a value the model can act on, and the refusal reaches
/// the next context window.
#[test]
fn a_refused_finish_is_reported_and_the_task_keeps_running() {
    let dir = tempfile::tempdir().unwrap();
    let (outcome, emitted) = run(
        dir.path(),
        RoleModel::new(vec![
            // Verification fails, so completion must be refused even forced.
            program(
                "evidence(\"tests\", false, { note: \"3 failing\" });\nreturn { recorded: true };",
            ),
            program(
                "var verdict = finish({ summary: \"shipped\", force: true });\nreturn { refusedAccepted: verdict.accepted, refusal: verdict.explanation };",
            ),
            // Fix it and record passing evidence, then finish for real.
            program("evidence(\"tests\", true, { note: \"all green\" });\nreturn { fixed: true };"),
            finish("tests fixed and passing"),
        ]),
        "make the tests pass",
        config(),
    );
    assert!(outcome.is_complete());

    // Keyed distinctly from the final accepted verdict, which also carries an
    // explanation field.
    let refusal = support::ptc_value(dir.path(), "refusal");
    assert_eq!(refusal["refusedAccepted"], false);
    let explanation = refusal["refusal"].as_str().unwrap();
    assert!(
        explanation.contains("finish() refused") && explanation.contains("tests"),
        "the refusal must name the failing verification: {explanation}"
    );

    let verdicts: Vec<_> = events(dir.path())
        .into_iter()
        .filter_map(|record| match record.event {
            SessionEvent::FinishProposed { verdict, .. } => Some(verdict),
            _ => None,
        })
        .collect();
    assert_eq!(verdicts.len(), 2);
    assert!(
        !verdicts[0].accepted,
        "a failed verification is not waivable"
    );
    assert!(verdicts[1].accepted);
    assert!(
        emitted
            .iter()
            .any(|event| matches!(event, AgentEvent::TaskCompleted { .. }))
    );
}

/// Durable cancellation: `mh cancel` appends intent, and a running loop honors
/// it at its next safe point.
#[test]
fn durable_cancellation_is_honored_by_a_detached_run() {
    let dir = tempfile::tempdir().unwrap();
    let task = Agent::<RoleModel>::start_detached_task(dir.path(), "long job").unwrap();
    mh::runtime::cancel_task(dir.path(), Some(task)).unwrap();

    let agent = Agent::new(RoleModel::new(vec![finish("should never run")]), config());
    let error = agent
        .resume_task(
            dir.path(),
            Some(task),
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |_| {},
        )
        .unwrap_err();
    assert!(matches!(error, mh::agent::AgentError::Cancelled));
    assert_eq!(
        Session::inspect(dir.path())
            .unwrap()
            .task_view(task)
            .status(),
        TaskStatus::Cancelled
    );
}

/// Durable steering from another process reaches a running task.
#[test]
fn durable_steering_reaches_a_running_task() {
    let dir = tempfile::tempdir().unwrap();
    let task = Agent::<RoleModel>::start_detached_task(dir.path(), "detached job").unwrap();
    mh::runtime::steer_task(dir.path(), Some(task), "also update the changelog").unwrap();

    let seen_instruction = Arc::new(AtomicBool::new(false));
    let probe = seen_instruction.clone();
    let model = RoleModel::new(vec![program("return 1;"), finish("done")])
        .with_worker_hook(Arc::new(move |_| probe.store(true, Ordering::Relaxed)));
    let agent = Agent::new(model, config());
    let outcome = agent
        .resume_task(
            dir.path(),
            Some(task),
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |_| {},
        )
        .unwrap();
    assert!(outcome.is_complete());
    assert!(
        events(dir.path()).iter().any(|record| matches!(
            &record.event,
            SessionEvent::SteeringApplied { content, .. } if content == "also update the changelog"
        )),
        "steering appended by another process must be applied"
    );
    assert!(
        Session::inspect(dir.path())
            .unwrap()
            .task_view(task)
            .pending_steering
            .is_empty(),
        "applied steering is no longer pending"
    );
}
