//! End-to-end delegation and integration through the real PTC surface.
//!
//! These tests drive `delegate` / `integrate` from MicroQuickJS with a
//! scripted model and assert the JS-visible contract documented in README.md:
//! field names, access-mode capability limits, nested-delegation refusal, and
//! integration versus conflict semantics.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use mh::agent::{Agent, AgentConfig};
use mh::context::CompiledContext;
use mh::model::{GenerationStop, Model, ModelError, ModelEvent, ModelOutput, ProgramLanguage};
use mh::session::{Session, SessionEvent};
use serde_json::Value;

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
        _stop: &GenerationStop,
        _events: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput, ModelError> {
        self.outputs
            .lock()
            .unwrap()
            .pop()
            .ok_or_else(|| ModelError::Protocol("script exhausted".to_string()))
    }
}

fn program(source: &str) -> ModelOutput {
    ModelOutput::Program {
        language: ProgramLanguage::JavaScript,
        source: source.to_string(),
    }
}

fn git(dir: &Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn repository() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    git(dir.path(), &["init", "-q"]);
    git(dir.path(), &["config", "user.name", "mh test"]);
    git(dir.path(), &["config", "user.email", "mh@example.invalid"]);
    std::fs::write(dir.path().join("tracked"), "base").unwrap();
    git(dir.path(), &["add", "."]);
    git(dir.path(), &["commit", "-qm", "base"]);
    dir
}

fn run(dir: &Path, outputs: Vec<ModelOutput>) -> String {
    Agent::new(ScriptedModel::new(outputs), AgentConfig::default())
        .run_task(dir, "delegate", &Arc::new(AtomicBool::new(false)))
        .unwrap()
}

/// The last PTC value carrying `key`; parent and child results share the journal.
fn ptc_value(dir: &Path, key: &str) -> Value {
    Session::resume(dir)
        .unwrap()
        .events()
        .into_iter()
        .rev()
        .find_map(|record| match record.event {
            SessionEvent::PtcCompleted { value, .. } if value.get(key).is_some() => Some(value),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no PTC result carried {key}"))
}

#[test]
fn read_delegation_returns_the_documented_result_and_denies_mutation() {
    let dir = repository();
    let answer = run(
        dir.path(),
        vec![
            program("return delegate({ task: \"inspect\", access: \"read\" });"),
            program(
                r#"
                var wrote = write("child-wrote", "x");
                var ran = exec({ command: ["true"] });
                var nested = null;
                try {
                    delegate({ task: "n", access: "read" });
                } catch (nestedError) {
                    nested = "" + nestedError;
                }
                var merged = null;
                try {
                    integrate(1);
                } catch (mergeError) {
                    merged = "" + mergeError;
                }
                return {
                    wrote: wrote.error,
                    ran: ran.error,
                    nested: nested,
                    merged: merged,
                    tracked: read("tracked").content
                };
                "#,
            ),
            ModelOutput::Text("child report".to_string()),
            ModelOutput::Text("parent answer".to_string()),
        ],
    );
    assert_eq!(answer, "parent answer");

    let child_value = ptc_value(dir.path(), "tracked");
    assert_eq!(child_value["wrote"], "write: disabled by policy");
    assert_eq!(child_value["ran"], "exec: disabled by policy");
    assert_eq!(
        child_value["nested"], "InternalError: delegate: unavailable in child execution",
        "nested delegation must throw, not silently succeed"
    );
    assert_eq!(
        child_value["merged"],
        "InternalError: integrate: unavailable in child execution"
    );
    assert_eq!(child_value["tracked"], "base", "read children still read");
    assert!(
        !dir.path().join("child-wrote").exists(),
        "a denied write must not reach the parent workspace"
    );

    let result = ptc_value(dir.path(), "taskId");
    assert_eq!(result["ok"], true);
    assert_eq!(result["summary"], "child report");
    assert_eq!(result["changed"], false);
    assert_eq!(result["baseRevision"], result["finalRevision"]);
    assert!(result["taskId"].is_u64() && result["executionId"].is_u64());
    assert!(
        result.get("workspace").is_none(),
        "read delegation allocates no isolated workspace"
    );
    assert_eq!(
        result["findings"], child_value,
        "findings carries the child's last PTC value, and nothing else crosses back"
    );
}

#[test]
fn isolated_child_delta_reaches_the_parent_only_through_integrate() {
    let dir = repository();
    run(
        dir.path(),
        vec![
            program(
                r#"
                var child = delegate({ task: "implement", access: "isolated-write" });
                var visibleBefore = glob("added-by-child").length;
                var merged = integrate(child.workspace);
                return { child: child, visibleBefore: visibleBefore, merged: merged };
                "#,
            ),
            program("return write(\"added-by-child\", \"child\");"),
            ModelOutput::Text("implemented".to_string()),
            ModelOutput::Text("integrated".to_string()),
        ],
    );

    let value = ptc_value(dir.path(), "merged");
    let child = &value["child"];
    assert_eq!(child["ok"], true);
    assert_eq!(child["changed"], true);
    assert_ne!(child["baseRevision"], child["finalRevision"]);
    assert!(
        child["workspace"].is_u64(),
        "isolated-write returns a workspace id"
    );
    assert_eq!(
        value["visibleBefore"], 0,
        "the child delta must be invisible to the parent before integration"
    );

    let merged = &value["merged"];
    assert_eq!(merged["ok"], true);
    assert_eq!(merged["conflict"], false);
    assert_eq!(merged["requiresReverification"], true);
    assert_eq!(merged["paths"], serde_json::json!(["added-by-child"]));
    assert_ne!(merged["previousRevision"], merged["parentRevision"]);
    assert!(merged.get("error").is_none());

    assert_eq!(
        std::fs::read_to_string(dir.path().join("added-by-child")).unwrap(),
        "child"
    );
    assert!(
        !dir.path().join(".mh/workspaces").join("1").exists(),
        "a successful integration discards the isolated workspace"
    );
}

#[test]
fn divergent_parent_turns_integration_into_a_conflict_without_overwriting() {
    let dir = repository();
    run(
        dir.path(),
        vec![
            program(
                r#"
                var child = delegate({ task: "edit tracked", access: "isolated-write" });
                write("tracked", "parent version");
                var merged = integrate(child.workspace);
                return { merged: merged, tracked: read("tracked").content };
                "#,
            ),
            program("return write(\"tracked\", \"child version\");"),
            ModelOutput::Text("edited".to_string()),
            ModelOutput::Text("conflicted".to_string()),
        ],
    );

    let value = ptc_value(dir.path(), "merged");
    let merged = &value["merged"];
    assert_eq!(merged["ok"], false);
    assert_eq!(merged["conflict"], true);
    assert_eq!(merged["requiresReverification"], false);
    assert_eq!(merged["paths"], serde_json::json!(["tracked"]));
    assert_ne!(merged["childBase"], merged["parentCurrent"]);
    assert!(
        merged.get("previousRevision").is_none() && merged.get("parentRevision").is_none(),
        "a conflict creates no parent revision"
    );

    assert_eq!(value["tracked"], "parent version");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("tracked")).unwrap(),
        "parent version",
        "a conflicting child must never overwrite the parent"
    );
}
