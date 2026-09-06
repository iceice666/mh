//! Asynchronous delegated workers (spec v0.5 §3, §4).
//!
//! Covers genuine overlap, independent polling, per-worker cancellation,
//! durable worker context across a rollover, and isolated integration.

mod support;

use std::sync::Arc;
use std::time::Duration;

use mh::delegation::AgentState;
use mh::goal::TaskStatus;
use mh::identity::AgentId;
use mh::session::{Session, SessionEvent};
use support::{
    Overlap, RoleModel, config, events, finish, program, ptc_value, repository, run, seqs, text,
};

/// Spec: root spawns A and B; A and B overlap, the root executes another PTC
/// action before joining, and results are polled independently.
#[test]
fn workers_overlap_while_the_root_keeps_working() {
    let dir = repository();
    let overlap = Arc::new(Overlap::default());
    let gate = overlap.clone();

    let model = RoleModel::new(vec![
        program(
            r#"
            var a = agent_spawn({ task: "inspect the parser architecture", access: "read" });
            var b = agent_spawn({ task: "inspect the parser tests", access: "read" });
            return { a: a.agent, b: b.agent, aState: a.state, bState: b.state };
            "#,
        ),
        // Real root work while both workers run: this must not block on them.
        program(
            r#"
            write("root-worked.txt", "root ran between spawn and join");
            var statuses = agent_list();
            return { rootWork: read("root-worked.txt").content, workers: statuses.length };
            "#,
        ),
        program(
            r#"
            var first = agent_join(1);
            var second = agent_join(2);
            return {
                firstOk: first.ok, firstSummary: first.summary,
                secondOk: second.ok, secondSummary: second.summary
            };
            "#,
        ),
        finish("both inspections joined"),
    ])
    .worker(
        "parser architecture",
        vec![
            program("return glob(\"*\").length;"),
            text("architecture: precedence lives in the parser"),
        ],
    )
    .worker(
        "parser tests",
        vec![
            program("return glob(\"*\").length;"),
            text("tests: precedence is covered by two cases"),
        ],
    )
    .with_worker_hook(Arc::new(move |_objective| {
        // Each worker waits for the other to arrive. Serialized execution
        // would time out here and leave the peak at one.
        gate.rendezvous(2, Duration::from_secs(5));
    }));

    let (outcome, _) = run(dir.path(), model, "orchestrate two inspections", config());
    assert!(outcome.is_complete());
    assert_eq!(
        overlap.peak(),
        2,
        "both workers must be inside the model call at the same time"
    );

    let spawned = ptc_value(dir.path(), "aState");
    assert_eq!(spawned["a"], 1);
    assert_eq!(spawned["b"], 2);

    // The root really did work between spawn and join.
    let root_work = ptc_value(dir.path(), "rootWork");
    assert_eq!(root_work["rootWork"], "root ran between spawn and join");
    assert_eq!(root_work["workers"], 2);

    // Results are independent and each worker's own summary comes back.
    let joined = ptc_value(dir.path(), "firstSummary");
    assert_eq!(joined["firstOk"], true);
    assert_eq!(joined["secondOk"], true);
    assert!(
        joined["firstSummary"]
            .as_str()
            .unwrap()
            .contains("precedence lives in the parser")
    );
    assert!(
        joined["secondSummary"]
            .as_str()
            .unwrap()
            .contains("covered by two cases")
    );

    // Ordering proof from the durable journal: root PTC work is interleaved
    // between the spawns and the completions.
    let spawns = seqs(dir.path(), |event| {
        matches!(event, SessionEvent::AgentSpawned { .. })
    });
    let completions = seqs(dir.path(), |event| {
        matches!(event, SessionEvent::AgentCompleted { .. })
    });
    assert_eq!(spawns.len(), 2);
    assert_eq!(completions.len(), 2);
    assert!(
        spawns[1] < completions[0],
        "the second worker started before the first finished: {spawns:?} vs {completions:?}"
    );

    // Each worker is its own durable task with its own agent id.
    let session = Session::inspect(dir.path()).unwrap();
    assert_eq!(session.tasks().len(), 3, "one root task plus two workers");
    for agent in [AgentId(1), AgentId(2)] {
        let record = session.agent(agent).expect("worker journaled");
        assert_eq!(record.state, AgentState::Completed);
        assert_eq!(
            session.task_view(record.task_id).status(),
            TaskStatus::Completed
        );
    }
}

/// Spec: cancellation affects only the selected worker.
#[test]
fn cancellation_affects_only_the_selected_worker() {
    let dir = repository();
    let model = RoleModel::new(vec![
        program(
            r#"
            var doomed = agent_spawn({ task: "inspect the doomed area", access: "read" });
            var survivor = agent_spawn({ task: "inspect the surviving area", access: "read" });
            var cancelled = agent_cancel(doomed.agent);
            var kept = agent_join(survivor.agent);
            var doomedResult = agent_join(doomed.agent);
            return {
                cancelledState: cancelled.state,
                keptOk: kept.ok,
                keptSummary: kept.summary,
                doomedOk: doomedResult.ok
            };
            "#,
        ),
        finish("one cancelled, one kept"),
    ])
    .worker(
        "doomed area",
        vec![
            // Long enough that cancellation lands mid-flight.
            program("return exec({ command: [\"/bin/sh\", \"-c\", \"sleep 1\"] }).exitCode;"),
            text("doomed should not report"),
        ],
    )
    .worker(
        "surviving area",
        vec![text("survivor: inspected successfully")],
    );

    let (outcome, _) = run(dir.path(), model, "cancel one worker", config());
    assert!(outcome.is_complete());

    let value = ptc_value(dir.path(), "keptSummary");
    assert_eq!(value["keptOk"], true);
    assert!(
        value["keptSummary"]
            .as_str()
            .unwrap()
            .contains("inspected successfully"),
        "the surviving worker completed normally: {value}"
    );
    assert_eq!(
        value["doomedOk"], false,
        "the cancelled worker must not report success"
    );

    let session = Session::inspect(dir.path()).unwrap();
    assert_eq!(
        session.agent(AgentId(2)).unwrap().state,
        AgentState::Completed,
        "cancelling worker 1 must leave worker 2 untouched"
    );
    assert!(
        session.agent(AgentId(1)).unwrap().state.is_terminal(),
        "the cancelled worker settled"
    );
}

/// Spec: force a worker to need many turns and at least one rollover, then
/// verify an early finding is still represented afterwards.
#[test]
fn a_worker_survives_its_own_context_rollover() {
    let dir = repository();
    let mut worker_script = vec![program(
        r#"
        return goal({
            findings: [{ summary: "WORKER_EARLY_FINDING: precedence table is in parser.rs" }],
            pending: [{ title: "confirm the test coverage" }]
        });
        "#,
    )];
    // Two turns per worker window: these fill the worker's first two windows.
    for index in 0..4 {
        worker_script.push(program(&format!("return {index};")));
    }
    worker_script.push(text("worker report: precedence confirmed"));

    let model = RoleModel::new(vec![
        program(
            "var a = agent_spawn({ task: \"deeply inspect the parser\", access: \"read\" });\nvar r = agent_join(a.agent);\nreturn { ok: r.ok, summary: r.summary, taskId: r.taskId };",
        ),
        finish("worker rolled over and reported"),
    ])
    .worker("deeply inspect the parser", worker_script);

    let (outcome, _) = run(
        dir.path(),
        model,
        "delegate a long inspection",
        mh::agent::AgentConfig {
            delegation_budget: mh::delegation::DelegationBudget {
                max_child_turns: 2,
                ..mh::delegation::DelegationBudget::default()
            },
            ..config()
        },
    );
    assert!(outcome.is_complete());

    let joined = ptc_value(dir.path(), "summary");
    assert_eq!(joined["ok"], true);
    assert!(
        joined["summary"]
            .as_str()
            .unwrap()
            .contains("precedence confirmed")
    );

    let session = Session::inspect(dir.path()).unwrap();
    let record = session.agent(AgentId(1)).unwrap();
    let view = session.task_view(record.task_id);
    assert!(
        view.window >= 1,
        "the worker crossed at least one context window, saw {}",
        view.window
    );
    let checkpoint = view
        .checkpoint
        .as_ref()
        .expect("a rolled-over worker carries its own checkpoint");
    assert!(
        checkpoint
            .important_findings
            .iter()
            .any(|finding| finding.summary.contains("WORKER_EARLY_FINDING")),
        "the worker's early finding survived its own rollover"
    );
    assert!(
        events(dir.path()).iter().any(|record| matches!(
            &record.event,
            SessionEvent::Compacted { agent, .. } if *agent == AgentId(1)
        )),
        "the worker's compaction is durable and attributed to the worker"
    );
}

/// Spec: two write agents fork from the same revision; non-conflicting
/// integration succeeds, conflicting integration reports a conflict, parent
/// revision tracking stays correct, and verification is marked stale.
#[test]
fn isolated_workers_integrate_or_conflict_without_corrupting_the_parent() {
    let dir = repository();
    let model = RoleModel::new(vec![
        program(
            r#"
            var a = agent_spawn({ task: "add the alpha file", access: "isolated-write" });
            var b = agent_spawn({ task: "rewrite the tracked file", access: "isolated-write" });
            var first = agent_join(a.agent);
            var second = agent_join(b.agent);
            return {
                aWorkspace: first.workspace, aChanged: first.changed,
                bWorkspace: second.workspace, bChanged: second.changed,
                visibleBefore: glob("alpha").length,
                aBase: first.baseRevision, bBase: second.baseRevision
            };
            "#,
        ),
        // Record verification, then integrate: the merge must invalidate it.
        program("evidence(\"tests\", true, { note: \"green before integration\" });\nreturn { verified: true };"),
        program(
            r#"
            var merged = integrate(1);
            return {
                mergedOk: merged.ok, conflict: merged.conflict,
                paths: merged.paths, requiresReverification: merged.requiresReverification,
                previousRevision: merged.previousRevision, parentRevision: merged.parentRevision,
                alpha: read("alpha").content
            };
            "#,
        ),
        // The parent now diverges on `tracked`, so worker B must conflict.
        program(
            r#"
            write("tracked", "parent version\n");
            var conflicted = integrate(2);
            return {
                conflictOk: conflicted.ok, conflict: conflicted.conflict,
                conflictPaths: conflicted.paths,
                childBase: conflicted.childBase, parentCurrent: conflicted.parentCurrent,
                tracked: read("tracked").content
            };
            "#,
        ),
        program("evidence(\"tests\", true, { note: \"re-verified after integration\" });\nreturn { reverified: true };"),
        // The conflicted delta must be resolvable, or the task could never
        // legitimately finish.
        program("return { discarded: discard(2, \"parent diverged on tracked\") };"),
        finish("integrated one worker, refused the other"),
    ])
    .worker(
        "add the alpha file",
        vec![
            program("return write(\"alpha\", \"from worker a\");"),
            text("alpha added"),
        ],
    )
    .worker(
        "rewrite the tracked file",
        vec![
            program("return write(\"tracked\", \"worker b version\\n\");"),
            text("tracked rewritten"),
        ],
    );

    let (outcome, _) = run(dir.path(), model, "two isolated workers", config());
    assert!(outcome.is_complete());

    // Both workers forked from the same parent revision and neither leaked.
    let spawned = ptc_value(dir.path(), "aWorkspace");
    assert_eq!(
        spawned["aBase"], spawned["bBase"],
        "both workers forked from the same revision"
    );
    assert_eq!(spawned["aChanged"], true);
    assert_eq!(spawned["bChanged"], true);
    assert_eq!(
        spawned["visibleBefore"], 0,
        "an isolated delta is invisible to the parent before integration"
    );

    // Non-conflicting integration applied and demanded re-verification.
    let merged = ptc_value(dir.path(), "requiresReverification");
    assert_eq!(merged["mergedOk"], true);
    assert_eq!(merged["conflict"], false);
    assert_eq!(merged["requiresReverification"], true);
    assert_eq!(merged["paths"], serde_json::json!(["alpha"]));
    assert_ne!(
        merged["previousRevision"], merged["parentRevision"],
        "a successful integration creates a new parent revision"
    );
    assert_eq!(merged["alpha"], "from worker a");

    // Conflicting integration reported a conflict and changed nothing.
    let conflicted = ptc_value(dir.path(), "conflictPaths");
    assert_eq!(conflicted["conflictOk"], false);
    assert_eq!(conflicted["conflict"], true);
    assert_eq!(conflicted["conflictPaths"], serde_json::json!(["tracked"]));
    assert_ne!(conflicted["childBase"], conflicted["parentCurrent"]);
    assert_eq!(
        conflicted["tracked"], "parent version\n",
        "a conflicting worker must never overwrite the parent"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("tracked")).unwrap(),
        "parent version\n"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("alpha")).unwrap(),
        "from worker a"
    );

    // The durable journal records exactly one applied integration and one
    // conflict, and evidence recorded pre-merge is stale afterwards.
    let session = Session::inspect(dir.path()).unwrap();
    let applied = events(dir.path())
        .into_iter()
        .filter(|record| matches!(record.event, SessionEvent::IntegrationCompleted { .. }))
        .count();
    let conflicts = events(dir.path())
        .into_iter()
        .filter(|record| matches!(record.event, SessionEvent::IntegrationConflict { .. }))
        .count();
    assert_eq!((applied, conflicts), (1, 1));
    assert_eq!(
        events(dir.path())
            .into_iter()
            .filter(|record| matches!(record.event, SessionEvent::IntegrationDiscarded { .. }))
            .count(),
        1,
        "abandoning the conflicted delta is journaled as a decision"
    );
    assert!(
        session
            .task_view(session.root_task().unwrap())
            .unintegrated_workspaces()
            .is_empty(),
        "one integrated and one discarded leaves nothing unresolved"
    );

    let view = session.task_view(session.root_task().unwrap());
    let stale: Vec<_> = view
        .evidence
        .iter()
        .filter(|record| record.note.as_deref() == Some("green before integration"))
        .collect();
    assert_eq!(stale.len(), 1);
    assert!(
        !view.evidence_is_fresh(stale[0]),
        "verification recorded before integration must read as stale"
    );
    let fresh = view
        .evidence
        .iter()
        .find(|record| record.note.as_deref() == Some("re-verified after integration"))
        .expect("post-merge verification recorded");
    assert!(
        view.evidence_is_fresh(fresh),
        "verification recorded after integration is fresh"
    );
}

/// A worker cannot spawn a worker: delegation depth stays at one, and the
/// refusal names the reason rather than surfacing as an undefined identifier.
#[test]
fn a_worker_cannot_delegate_further() {
    let dir = repository();
    let model = RoleModel::new(vec![
        program(
            "var a = agent_spawn({ task: \"try to recurse\", access: \"read\" });\nvar r = agent_join(a.agent);\nreturn { findings: r.findings };",
        ),
        finish("depth held at one"),
    ])
    .worker(
        "try to recurse",
        vec![
            program(
                // MicroQuickJS rejects reusing one catch identifier across
                // sibling clauses, so each catch binds its own name.
                r#"
                var spawnError = null;
                try { agent_spawn({ task: "grandchild", access: "read" }); }
                catch (spawnFailure) { spawnError = "" + spawnFailure; }
                var delegateError = null;
                try { delegate({ task: "grandchild", access: "read" }); }
                catch (delegateFailure) { delegateError = "" + delegateFailure; }
                // integrate reports refusal as a structured value, not a
                // throw, so a program can branch on it.
                var integrateRefusal = integrate(1);
                return {
                    spawnError: spawnError,
                    delegateError: delegateError,
                    integrateError: integrateRefusal.error
                };
                "#,
            ),
            text("recursion refused"),
        ],
    );

    let (outcome, _) = run(dir.path(), model, "check delegation depth", config());
    assert!(outcome.is_complete());

    let findings = ptc_value(dir.path(), "spawnError");
    for key in ["spawnError", "delegateError", "integrateError"] {
        let message = findings[key].as_str().unwrap_or_default();
        assert!(
            message.contains("unavailable in a delegated worker"),
            "{key} must explain the depth limit, got {message:?}"
        );
    }
    // Each refusal names the primitive the program actually called, so
    // `delegate()` never reports that `agent_spawn` failed.
    for (key, primitive) in [("spawnError", "agent_spawn"), ("delegateError", "delegate")] {
        let message = findings[key].as_str().unwrap();
        assert!(
            message.contains(&format!("{primitive}: unavailable")),
            "{key} must name {primitive}, got {message:?}"
        );
        assert!(
            !message.contains("agent_spawn: agent_spawn"),
            "the primitive must not be named twice: {message:?}"
        );
    }
}

/// Profiles are declarative: an unknown one is refused rather than silently
/// granting unintended access.
#[test]
fn profiles_select_access_and_an_unknown_profile_is_refused() {
    let dir = repository();
    let model = RoleModel::new(vec![
        program(
            r#"
            var unknown = null;
            try { agent_spawn({ task: "x", profile: "wizard" }); }
            catch (error) { unknown = "" + error; }
            var explore = agent_spawn({ task: "explore the tree", profile: "explore" });
            var joined = agent_join(explore.agent);
            return {
                unknown: unknown,
                access: explore.access,
                profile: explore.profile,
                findings: joined.findings
            };
            "#,
        ),
        finish("profiles honored"),
    ])
    .worker(
        "explore the tree",
        vec![
            program("return { write: write(\"denied\", \"x\").error };"),
            text("explored read-only"),
        ],
    );

    let (outcome, _) = run(dir.path(), model, "use a profile", config());
    assert!(outcome.is_complete());

    let value = ptc_value(dir.path(), "access");
    assert!(
        value["unknown"]
            .as_str()
            .unwrap_or_default()
            .contains("unknown agent profile 'wizard'"),
        "an unknown profile must be refused: {value}"
    );
    assert_eq!(value["access"], "read", "the explore profile is read-only");
    assert_eq!(value["profile"], "explore");
    assert_eq!(
        value["findings"]["write"], "write: disabled by policy",
        "the profile's read-only access is enforced in the worker"
    );
    assert!(!dir.path().join("denied").exists());
}

/// A parent message reaches a running worker and shows up in its context.
#[test]
fn a_parent_message_reaches_a_running_worker() {
    let dir = repository();
    let model = RoleModel::new(vec![
        program(
            r#"
            var a = agent_spawn({ task: "inspect and await instruction", access: "read" });
            agent_send(a.agent, "also inspect the parser tests");
            var joined = agent_join(a.agent);
            return { ok: joined.ok, summary: joined.summary };
            "#,
        ),
        finish("message delivered"),
    ])
    .worker(
        "inspect and await instruction",
        vec![
            program("return glob(\"*\").length;"),
            program("return glob(\"*\").length;"),
            text("inspected both areas"),
        ],
    );

    let (outcome, _) = run(dir.path(), model, "steer a worker", config());
    assert!(outcome.is_complete());
    assert!(
        events(dir.path()).iter().any(|record| matches!(
            &record.event,
            SessionEvent::AgentMessageApplied { content, .. }
                if content == "also inspect the parser tests"
        )),
        "the worker must durably consume the parent's message"
    );
    assert!(
        Session::inspect(dir.path())
            .unwrap()
            .agent(AgentId(1))
            .unwrap()
            .inbox
            .is_empty(),
        "a consumed message is no longer pending"
    );
}
