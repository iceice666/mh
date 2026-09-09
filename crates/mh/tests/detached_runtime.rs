//! Detached task execution (spec v0.5 Phase 5).
//!
//! These drive the journal as separate `Session` handles the way separate
//! processes do: a task is allocated by one, steered and cancelled by another,
//! and driven to completion by a third. That is the whole point of making the
//! journal the coordination point — no daemon is required for a task to
//! outlive the command that started it.

mod support;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use mh::agent::{Agent, AgentError, TaskOutcome};
use mh::goal::TaskStatus;
use mh::runtime::{cancel_task, steer_task, task_reports};
use mh::session::{Session, SessionEvent};
use support::{RoleModel, config, events, finish, program, repository};

/// A task allocated by one process is visible to, and runnable by, another.
#[test]
fn a_detached_task_is_journaled_before_anything_runs() {
    let dir = repository();
    let task = Agent::<RoleModel>::start_detached_task(dir.path(), "detached objective").unwrap();

    // A separate reader — as `mh tasks` in another process — sees it queued.
    let reports = task_reports(dir.path()).unwrap();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].task_id, task);
    assert_eq!(reports[0].objective, "detached objective");
    assert_eq!(reports[0].status, TaskStatus::Queued);
    assert!(!reports[0].cancel_requested);

    // Inspection must not mutate the journal of a task nobody is driving.
    let before = events(dir.path()).len();
    let _ = task_reports(dir.path()).unwrap();
    let _ = Session::inspect(dir.path()).unwrap().tasks();
    assert_eq!(
        events(dir.path()).len(),
        before,
        "inspecting a task must not append to its journal"
    );

    // A third handle then drives it to completion.
    let agent = Agent::new(
        RoleModel::new(vec![
            program("return write(\"detached-ran\", \"yes\");"),
            finish("detached work complete"),
        ]),
        config(),
    );
    let outcome = agent
        .resume_task(
            dir.path(),
            Some(task),
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(
        outcome,
        TaskOutcome::Completed("detached work complete".to_string())
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("detached-ran")).unwrap(),
        "yes"
    );
    assert_eq!(
        task_reports(dir.path()).unwrap()[0].status,
        TaskStatus::Completed
    );
}

/// Steering queued while nothing is running stays pending, is visible to an
/// inspector, and is applied by whichever process next drives the task.
#[test]
fn steering_queued_against_an_idle_task_is_pending_then_applied() {
    let dir = repository();
    let task = Agent::<RoleModel>::start_detached_task(dir.path(), "steerable").unwrap();
    steer_task(dir.path(), Some(task), "prefer the smaller change").unwrap();

    let reports = task_reports(dir.path()).unwrap();
    assert_eq!(
        reports[0].pending_steering,
        vec!["prefer the smaller change"],
        "queued steering is visible before anything applies it"
    );

    let agent = Agent::new(
        RoleModel::new(vec![program("return 1;"), finish("steered")]),
        config(),
    );
    assert!(
        agent
            .resume_task(
                dir.path(),
                Some(task),
                &Arc::new(AtomicBool::new(false)),
                None,
                &mut |_| {},
            )
            .unwrap()
            .is_complete()
    );

    assert!(
        events(dir.path())
            .iter()
            .any(|record| matches!(&record.event, SessionEvent::SteeringApplied { .. })),
        "the running loop must apply steering left by another process"
    );
    assert!(
        task_reports(dir.path()).unwrap()[0]
            .pending_steering
            .is_empty()
    );
}

/// Cancellation requested with no id targets the root task and is honored.
#[test]
fn cancellation_without_an_id_targets_the_root_task() {
    let dir = repository();
    let task = Agent::<RoleModel>::start_detached_task(dir.path(), "cancellable").unwrap();

    let cancelled = cancel_task(dir.path(), None).unwrap();
    assert_eq!(cancelled.task_id, task, "no id means the root task");
    assert!(task_reports(dir.path()).unwrap()[0].cancel_requested);

    let agent = Agent::new(RoleModel::new(vec![finish("unreachable")]), config());
    let error = agent
        .resume_task(
            dir.path(),
            Some(task),
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |_| {},
        )
        .unwrap_err();
    assert!(matches!(error, AgentError::Cancelled));
    assert_eq!(
        task_reports(dir.path()).unwrap()[0].status,
        TaskStatus::Cancelled
    );
}

/// A completed task is not resumable: resume must not silently restart work
/// that already finished.
#[test]
fn a_terminal_task_is_not_resumed() {
    let dir = repository();
    let agent = Agent::new(RoleModel::new(vec![finish("all done")]), config());
    let task = Agent::<RoleModel>::start_detached_task(dir.path(), "finishes").unwrap();
    assert!(
        agent
            .resume_task(
                dir.path(),
                Some(task),
                &Arc::new(AtomicBool::new(false)),
                None,
                &mut |_| {},
            )
            .unwrap()
            .is_complete()
    );

    let second = Agent::new(RoleModel::new(vec![finish("should not run")]), config());
    let error = second
        .resume_task(
            dir.path(),
            Some(task),
            &Arc::new(AtomicBool::new(false)),
            None,
            &mut |_| {},
        )
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("already completed"),
        "resuming a completed task must be refused, got {message:?}"
    );

    // Steering or cancelling a finished task is refused rather than queued
    // into a journal nothing will read again.
    let steer = steer_task(dir.path(), Some(task), "too late").unwrap_err();
    assert!(
        steer.to_string().contains("already completed"),
        "steering a completed task must be refused, got {steer}"
    );
    assert!(
        task_reports(dir.path()).unwrap()[0]
            .pending_steering
            .is_empty(),
        "a refused steer leaves nothing queued"
    );
    assert!(
        cancel_task(dir.path(), Some(task))
            .unwrap_err()
            .to_string()
            .contains("already completed")
    );
}

/// Inspection races the task it observes by design. A file the task replaces
/// between the directory walk and the content read must not fail a read-only
/// caller, or `mh tasks` breaks exactly when a task is busiest.
#[test]
fn inspection_tolerates_a_workspace_changing_underneath_it() {
    let dir = repository();
    Agent::<RoleModel>::start_detached_task(dir.path(), "churn").unwrap();
    let root = dir.path().to_path_buf();
    let stop = Arc::new(AtomicBool::new(false));

    let churn_stop = stop.clone();
    let churn_root = root.clone();
    let churn = std::thread::spawn(move || {
        // Create-then-remove is what a real task does constantly: writes,
        // build output, temp files.
        while !churn_stop.load(std::sync::atomic::Ordering::Relaxed) {
            for index in 0..24 {
                let file = churn_root.join(format!("churn-{index}.txt"));
                let _ = std::fs::write(&file, "x");
                let _ = std::fs::remove_file(&file);
            }
            let nested = churn_root.join("churn-dir/inner");
            let _ = std::fs::create_dir_all(&nested);
            let _ = std::fs::write(nested.join("f"), "y");
            let _ = std::fs::remove_dir_all(churn_root.join("churn-dir"));
        }
    });

    for _ in 0..40 {
        task_reports(&root).expect("inspection must not fail while the workspace churns");
        Session::inspect(&root)
            .expect("opening a session must not fail while the workspace churns")
            .tasks();
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    churn.join().unwrap();
}

/// Steering or cancelling a workspace with no session fails cleanly rather
/// than panicking or inventing a task.
#[test]
fn journal_commands_fail_cleanly_without_a_session() {
    let dir = tempfile::tempdir().unwrap();
    assert!(task_reports(dir.path()).is_err());
    assert!(steer_task(dir.path(), None, "hello").is_err());
    assert!(cancel_task(dir.path(), None).is_err());
}
