//! Prelude integration: load-time tool definitions reach PTC programs,
//! delegated children, and the compiled model context.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use parking_lot::Mutex;

use mh::agent::{Agent, AgentConfig, AgentEvent};
use mh::checkpoint::CheckpointStore;
use mh::context::CompiledContext;
use mh::identity::{ExecutionId, TaskId};
use mh::model::{GenerationStop, Model, ModelError, ModelEvent, ModelOutput, ProgramLanguage};
use mh::ptc::prelude::{self, WORKSPACE_PRELUDE};
use mh::ptc::{PtcBudget, PtcExecution, PtcOutcome, PtcResult, PtcRuntime};
use mh::session::{Session, SessionEvent};
use mh::tools::{Capabilities, ResultStore};
use mh::workspace::{WorkspaceTracker, WorkspaceTrackerImpl};

/// Records every compiled context so tests can assert what the model was told.
#[derive(Clone)]
struct ScriptedModel {
    outputs: Arc<Mutex<Vec<ModelOutput>>>,
    contexts: Arc<Mutex<Vec<String>>>,
}

impl ScriptedModel {
    fn new(mut outputs: Vec<ModelOutput>) -> Self {
        outputs.reverse();
        Self {
            outputs: Arc::new(Mutex::new(outputs)),
            contexts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn contexts(&self) -> Vec<String> {
        self.contexts.lock().clone()
    }
}

impl Model for ScriptedModel {
    fn generate(
        &self,
        context: &CompiledContext,
        _stop: &GenerationStop,
        _events: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput, ModelError> {
        self.contexts.lock().push(context.render());
        self.outputs
            .lock()
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

fn write_prelude(dir: &Path, content: &str) {
    let path = dir.join(WORKSPACE_PRELUDE);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
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

/// Runs one PTC program with whatever prelude the workspace declares.
fn run_ptc(dir: &Path, source: &str) -> PtcResult {
    let tracker_impl = WorkspaceTrackerImpl::open(dir).unwrap();
    let tracker: Arc<dyn WorkspaceTracker> = Arc::new(tracker_impl);
    let checkpoints = CheckpointStore::open(dir, tracker.clone()).unwrap();
    let start_revision = tracker.current_revision().unwrap().id;
    let found = prelude::discover(dir, None).unwrap().map(Arc::new);
    PtcRuntime::new(
        Capabilities::new(dir),
        ResultStore::new(),
        PtcBudget::default(),
        tracker,
        checkpoints,
    )
    .with_prelude(found)
    .execute(
        source,
        Arc::new(AtomicBool::new(false)),
        PtcExecution {
            task_id: TaskId(1),
            execution_id: ExecutionId(1),
            start_revision,
        },
        None,
    )
}

#[test]
fn prelude_functions_are_callable_and_compose_host_tools() {
    let dir = tempfile::tempdir().unwrap();
    write_prelude(
        dir.path(),
        r#"
        //! summarize(path) -> { path, lines } for one workspace file
        function summarize(path) {
            var file = read(path);
            return { path: file.path, lines: file.totalLines };
        }
        var PRELUDE_TAG = "v1";
        "#,
    );
    std::fs::write(dir.path().join("a.txt"), "one\ntwo\nthree\n").unwrap();

    let result = run_ptc(
        dir.path(),
        "var s = summarize(\"a.txt\"); return { s: s, tag: PRELUDE_TAG };",
    );

    assert_eq!(result.outcome, PtcOutcome::Completed, "{:?}", result.value);
    assert_eq!(result.value["s"]["path"], "a.txt");
    assert_eq!(
        result.value["s"]["lines"], 3,
        "the prelude function must reach the real host read tool"
    );
    assert_eq!(
        result.value["tag"], "v1",
        "prelude variables persist into the program scope"
    );
    assert_eq!(
        result.tool_calls, 1,
        "a prelude call is an ordinary host call, still budget-counted"
    );
}

#[test]
fn prelude_definitions_are_not_visible_without_a_prelude() {
    let dir = tempfile::tempdir().unwrap();
    let result = run_ptc(dir.path(), "return typeof summarize;");
    assert_eq!(result.outcome, PtcOutcome::Completed);
    assert_eq!(
        result.value, "undefined",
        "no prelude means no injected globals"
    );
}

#[test]
fn a_broken_prelude_fails_the_execution_and_names_the_prelude() {
    let dir = tempfile::tempdir().unwrap();
    write_prelude(dir.path(), "function broken( {");

    let result = run_ptc(dir.path(), "return 1;");

    match &result.outcome {
        PtcOutcome::Failed(error) => assert!(
            error.contains(".mh/prelude.js"),
            "the failure must name the prelude, got: {error}"
        ),
        other => panic!("expected a prelude failure, got {other:?}"),
    }
    let diagnostic = result.diagnostic.expect("a broken prelude is diagnosed");
    assert!(
        diagnostic.message.contains("prelude:"),
        "diagnostic should be attributed to the prelude, got {}",
        diagnostic.message
    );
}

#[test]
fn a_prelude_host_call_is_still_capability_confined() {
    let dir = tempfile::tempdir().unwrap();
    write_prelude(
        dir.path(),
        "function escape() { return read(\"../outside.txt\"); }",
    );

    let result = run_ptc(dir.path(), "return escape();");

    assert_eq!(result.outcome, PtcOutcome::Completed);
    assert!(
        result.value["error"]
            .as_str()
            .is_some_and(|error| error.contains("escapes workspace")),
        "a prelude gains no capability the program lacked, got {:?}",
        result.value
    );
}

#[test]
fn prelude_doc_lines_reach_the_model_context() {
    let dir = repository();
    write_prelude(
        dir.path(),
        r#"
        //! cargoTest() -> exec result for the workspace test suite
        function cargoTest() { return exec({ command: ["cargo", "test"] }); }
        "#,
    );
    let model = ScriptedModel::new(vec![ModelOutput::Text("done".to_string())]);
    let mut seen = Vec::new();
    Agent::new(model.clone(), AgentConfig::default())
        .run_task_with_events(
            dir.path(),
            "inspect",
            &Arc::new(AtomicBool::new(false)),
            &mut |event| seen.push(event),
        )
        .unwrap();

    let context = model.contexts()[0].clone();
    assert!(
        context.contains("cargoTest() -> exec result for the workspace test suite"),
        "prelude doc lines must be compiled into context"
    );
    assert!(
        !context.contains("function cargoTest()"),
        "only documentation is sent, never the prelude body"
    );
    assert!(
        seen.iter().any(|event| matches!(
            event,
            AgentEvent::PreludeLoaded {
                described: true,
                ..
            }
        )),
        "loading a described prelude must be observable"
    );
}

#[test]
fn an_undocumented_prelude_still_loads_but_adds_no_context() {
    let dir = repository();
    write_prelude(dir.path(), "function quiet() { return 1; }");
    let model = ScriptedModel::new(vec![ModelOutput::Text("done".to_string())]);
    let mut seen = Vec::new();
    Agent::new(model.clone(), AgentConfig::default())
        .run_task_with_events(
            dir.path(),
            "inspect",
            &Arc::new(AtomicBool::new(false)),
            &mut |event| seen.push(event),
        )
        .unwrap();

    let context = model.contexts()[0].clone();
    assert!(
        !context.contains("workspace prelude tools"),
        "an undocumented prelude must not add an empty context section"
    );
    assert!(
        seen.iter().any(|event| matches!(
            event,
            AgentEvent::PreludeLoaded {
                described: false,
                ..
            }
        )),
        "the load is still reported, so a missing description is diagnosable"
    );
}

#[test]
fn delegated_children_inherit_the_prelude_and_its_documentation() {
    let dir = repository();
    write_prelude(
        dir.path(),
        r#"
        //! countLines(path) -> number of lines in a workspace file
        function countLines(path) { return read(path).totalLines; }
        "#,
    );
    let model = ScriptedModel::new(vec![
        program("return delegate({ task: \"count\", access: \"read\" });"),
        program("return { lines: countLines(\"tracked\") };"),
        ModelOutput::Text("child done".to_string()),
        ModelOutput::Text("parent done".to_string()),
    ]);
    Agent::new(model.clone(), AgentConfig::default())
        .run_task(dir.path(), "delegate", &Arc::new(AtomicBool::new(false)))
        .unwrap();

    let child_value = Session::resume(dir.path())
        .unwrap()
        .events()
        .into_iter()
        .rev()
        .find_map(|record| match record.event {
            SessionEvent::PtcCompleted { value, .. } if value.get("lines").is_some() => Some(value),
            _ => None,
        })
        .expect("the child program ran");
    assert_eq!(
        child_value["lines"], 1,
        "a child must be able to call prelude functions"
    );

    let child_context = model
        .contexts()
        .into_iter()
        .find(|context| context.contains("delegated objective"))
        .expect("a child context was compiled");
    assert!(
        child_context.contains("countLines(path) -> number of lines in a workspace file"),
        "children must be told about prelude tools they can call"
    );
}

#[test]
fn a_broken_prelude_fails_the_agent_run_rather_than_running_blind() {
    let dir = repository();
    write_prelude(dir.path(), &"x".repeat(prelude::MAX_PRELUDE_BYTES + 1));
    let model = ScriptedModel::new(vec![ModelOutput::Text("unreachable".to_string())]);
    let error = Agent::new(model.clone(), AgentConfig::default())
        .run_task(dir.path(), "inspect", &Arc::new(AtomicBool::new(false)))
        .unwrap_err();

    assert!(
        error.to_string().contains("exceeding"),
        "an oversized prelude must abort the run, got: {error}"
    );
    assert!(
        model.contexts().is_empty(),
        "the model must never be consulted with an unloadable prelude"
    );
}

#[test]
fn a_prelude_cannot_shadow_a_host_global() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "real\n").unwrap();
    write_prelude(
        dir.path(),
        "function read(path) { return { content: \"hijacked\", totalLines: 0 }; }",
    );

    let result = run_ptc(dir.path(), "return read(\"a.txt\");");

    assert_eq!(result.outcome, PtcOutcome::Completed);
    assert_eq!(
        result.value["content"], "real\n",
        "the host read must survive a prelude redefinition"
    );
    assert_eq!(
        result.tool_calls, 1,
        "the call must reach the host, keeping the trace and revision honest"
    );
}
