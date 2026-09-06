//! The synchronous delegation surface, preserved for compatibility in v0.5.
//!
//! `delegate` and `delegate_batch` are now spawn-then-join over the same
//! worker runtime as `agent_spawn`, so these tests pin the JS-visible contract
//! documented in README.md: field names, access-mode capability limits, and
//! input-order results.

mod support;

use mh::delegation::AgentState;
use mh::identity::AgentId;
use mh::session::Session;
use support::{RoleModel, config, finish, program, ptc_value, repository, run, text};

#[test]
fn read_delegation_returns_the_documented_result_and_denies_mutation() {
    let dir = repository();
    let model = RoleModel::new(vec![
        program("return delegate({ task: \"inspect the tree\", access: \"read\" });"),
        finish("parent answer"),
    ])
    .worker(
        "inspect the tree",
        vec![
            program(
                r#"
                var wrote = write("child-wrote", "x");
                var ran = exec({ command: ["true"] });
                return {
                    wrote: wrote.error,
                    ran: ran.error,
                    tracked: read("tracked").content
                };
                "#,
            ),
            text("child report"),
        ],
    );
    let (outcome, _) = run(dir.path(), model, "delegate synchronously", config());
    assert!(outcome.is_complete());

    // A read worker reads, but cannot mutate or spawn processes.
    let child = ptc_value(dir.path(), "tracked");
    assert_eq!(child["wrote"], "write: disabled by policy");
    assert_eq!(child["ran"], "exec: disabled by policy");
    assert_eq!(child["tracked"], "base\n", "read workers still read");
    assert!(
        !dir.path().join("child-wrote").exists(),
        "a denied write must not reach the parent workspace"
    );

    // The documented result shape crosses back, and nothing else does.
    let result = ptc_value(dir.path(), "taskId");
    assert_eq!(result["ok"], true);
    assert_eq!(result["summary"], "child report");
    assert_eq!(result["changed"], false);
    assert_eq!(result["baseRevision"], result["finalRevision"]);
    assert!(result["taskId"].is_u64() && result["agent"].is_u64());
    assert!(
        result.get("workspace").is_none(),
        "read delegation allocates no isolated workspace"
    );
    assert_eq!(
        result["findings"], child,
        "findings carries the worker's last PTC value, and nothing else"
    );
}

#[test]
fn delegate_batch_preserves_input_order() {
    let dir = repository();
    let model = RoleModel::new(vec![
        program(
            r#"
            var reports = delegate_batch([
                { task: "report alpha", access: "read" },
                { task: "report beta", access: "read" },
                { task: "report gamma", access: "read" }
            ]);
            var summaries = [];
            for (var index = 0; index < reports.length; index = index + 1) {
                summaries.push(reports[index].summary);
            }
            return { summaries: summaries, count: reports.length };
            "#,
        ),
        finish("batch complete"),
    ])
    .worker("report alpha", vec![text("alpha")])
    .worker("report beta", vec![text("beta")])
    .worker("report gamma", vec![text("gamma")]);

    let (outcome, _) = run(dir.path(), model, "delegate a batch", config());
    assert!(outcome.is_complete());

    let value = ptc_value(dir.path(), "summaries");
    assert_eq!(value["count"], 3);
    assert_eq!(
        value["summaries"],
        serde_json::json!(["alpha", "beta", "gamma"]),
        "results come back in input order regardless of completion order"
    );

    // Each batch entry is a real durable worker.
    let session = Session::inspect(dir.path()).unwrap();
    for agent in [AgentId(1), AgentId(2), AgentId(3)] {
        assert_eq!(
            session.agent(agent).expect("worker journaled").state,
            AgentState::Completed
        );
    }
    assert_eq!(session.tasks().len(), 4, "one root task plus three workers");
}

#[test]
fn isolated_delegation_reaches_the_parent_only_through_integrate() {
    let dir = repository();
    let model = RoleModel::new(vec![
        program(
            r#"
            var child = delegate({ task: "add the file", access: "isolated-write" });
            var visibleBefore = glob("added-by-child").length;
            var merged = integrate(child.workspace);
            return {
                childChanged: child.changed,
                childWorkspace: child.workspace,
                visibleBefore: visibleBefore,
                mergedOk: merged.ok,
                mergedPaths: merged.paths,
                requiresReverification: merged.requiresReverification
            };
            "#,
        ),
        finish("integrated"),
    ])
    .worker(
        "add the file",
        vec![
            program("return write(\"added-by-child\", \"child\");"),
            text("implemented"),
        ],
    );

    let (outcome, _) = run(dir.path(), model, "isolated delegation", config());
    assert!(outcome.is_complete());

    let value = ptc_value(dir.path(), "mergedPaths");
    assert_eq!(value["childChanged"], true);
    assert!(value["childWorkspace"].is_u64());
    assert_eq!(
        value["visibleBefore"], 0,
        "the worker delta is invisible to the parent before integration"
    );
    assert_eq!(value["mergedOk"], true);
    assert_eq!(value["mergedPaths"], serde_json::json!(["added-by-child"]));
    assert_eq!(value["requiresReverification"], true);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("added-by-child")).unwrap(),
        "child"
    );
    assert!(
        !dir.path().join(".mh/workspaces/1").exists(),
        "a successful integration discards the isolated workspace"
    );
}

#[test]
fn a_failing_worker_reports_failure_without_degrading_the_parent() {
    let dir = repository();
    let model = RoleModel::new(vec![
        program(
            r#"
            var child = delegate({ task: "fail immediately", access: "read" });
            write("parent-continued", "parent kept working");
            return {
                ok: child.ok,
                summary: child.summary,
                parentWork: read("parent-continued").content
            };
            "#,
        ),
        finish("parent survived a worker failure"),
    ])
    // No script entry for this objective, so the worker's model call fails.
    ;

    let (outcome, _) = run(dir.path(), model, "worker fails", config());
    assert!(
        outcome.is_complete(),
        "a worker failure must not fail the parent task"
    );

    let value = ptc_value(dir.path(), "summary");
    assert_eq!(value["ok"], false);
    assert!(
        !value["summary"].as_str().unwrap().is_empty(),
        "a failure explains itself: {value}"
    );
    assert_eq!(value["parentWork"], "parent kept working");
    assert_eq!(
        Session::inspect(dir.path())
            .unwrap()
            .agent(AgentId(1))
            .unwrap()
            .state,
        AgentState::Failed
    );
}
