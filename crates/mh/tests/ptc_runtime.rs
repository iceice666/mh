//! PTC runtime integration tests against the real MicroQuickJS engine.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use mh::checkpoint::CheckpointStore;
use mh::identity::{AgentId, ExecutionId, TaskId};
use mh::ptc::{PtcBudget, PtcDiagnosticKind, PtcEvent, PtcExecution, PtcOutcome, PtcRuntime};
use mh::tools::{Capabilities, ResultStore, ToolEffects};
use mh::workspace::{WorkspaceTracker, WorkspaceTrackerImpl};

fn runtime_with_budget(dir: &std::path::Path, budget: PtcBudget) -> PtcRuntime {
    let tracker_impl = WorkspaceTrackerImpl::open(dir).unwrap();
    let tracker: Arc<dyn WorkspaceTracker> = Arc::new(tracker_impl);
    let checkpoints = CheckpointStore::open(dir, tracker.clone()).unwrap();
    PtcRuntime::new(
        Capabilities::new(dir),
        ResultStore::new(),
        budget,
        tracker,
        checkpoints,
    )
}

fn runtime(dir: &std::path::Path) -> PtcRuntime {
    runtime_with_budget(dir, PtcBudget::default())
}

fn run(dir: &std::path::Path, program: &str) -> mh::ptc::PtcResult {
    let tracker = WorkspaceTrackerImpl::open(dir).unwrap();
    let start_revision = tracker.current_revision().unwrap().id;
    runtime(dir).execute(
        program,
        Arc::new(AtomicBool::new(false)),
        PtcExecution {
            task_id: TaskId(1),
            agent: AgentId::ROOT,
            execution_id: ExecutionId(1),
            start_revision,
        },
        None,
    )
}

fn execute(
    rt: &PtcRuntime,
    dir: &std::path::Path,
    program: &str,
    cancelled: Arc<AtomicBool>,
) -> mh::ptc::PtcResult {
    let start_revision = WorkspaceTrackerImpl::open(dir)
        .unwrap()
        .current_revision()
        .unwrap()
        .id;
    rt.execute(
        program,
        cancelled,
        PtcExecution {
            task_id: TaskId(1),
            agent: AgentId::ROOT,
            execution_id: ExecutionId(1),
            start_revision,
        },
        None,
    )
}

#[test]
fn read_only_capabilities_disable_mutation_process_and_restore() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "alpha").unwrap();
    let tracker_impl = WorkspaceTrackerImpl::open(dir.path()).unwrap();
    let start_revision = tracker_impl.current_revision().unwrap().id;
    let tracker: Arc<dyn WorkspaceTracker> = Arc::new(tracker_impl);
    let runtime = PtcRuntime::new(
        Capabilities::read_only(dir.path()),
        ResultStore::new(),
        PtcBudget::default(),
        tracker.clone(),
        CheckpointStore::open(dir.path(), tracker).unwrap(),
    );
    let result = runtime.execute(
        r#"
        var readResult = read("a.txt");
        var writeResult = tool("write", {path: "blocked.txt", content: "bad"});
        var execResult = exec({command: ["/usr/bin/touch", "also-blocked.txt"]});
        var checkpointResult = checkpoint();
        return {read: readResult.content, write: writeResult, exec: execResult,
                checkpoint: checkpointResult};
        "#,
        Arc::new(AtomicBool::new(false)),
        PtcExecution {
            task_id: TaskId(2),
            agent: AgentId::ROOT,
            execution_id: ExecutionId(2),
            start_revision,
        },
        None,
    );
    assert_eq!(result.outcome, PtcOutcome::Completed);
    assert_eq!(result.value["read"], "alpha");
    assert_eq!(result.value["write"]["error"], "write: disabled by policy");
    assert_eq!(result.value["exec"]["error"], "exec: disabled by policy");
    assert_eq!(
        result.value["checkpoint"]["error"],
        "checkpoint: disabled by policy"
    );
    assert!(!dir.path().join("blocked.txt").exists());
    assert!(!dir.path().join("also-blocked.txt").exists());
}

#[test]
fn pure_js_evaluates() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(dir.path(), "var x = 1 + 2; return x * 2;");
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value, serde_json::json!(6));
}

#[test]
fn tool_read_write_roundtrip_preserves_trailing_newline() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        var expected = "line1\nline2\n";
        write("hello.txt", expected);
        var out = read("hello.txt");
        return {content: out.content, exact: out.content === expected};
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value["content"], serde_json::json!("line1\nline2\n"));
    assert_eq!(r.value["exact"], serde_json::json!(true));
}

#[test]
fn tool_canonical_abi() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        tool("write", { path: "a.txt", content: "abc" });
        var r = tool("read", { path: "a.txt" });
        return r.content;
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value, serde_json::json!("abc"));
}

#[test]
fn tools_alias_family() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        tools.write("x.txt", "hi");
        var files = tools.glob("*.txt");
        var r = call_tool("read", { path: "x.txt" });
        return { files: files, content: r.content };
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value["files"][0], serde_json::json!("x.txt"));
    assert_eq!(r.value["content"], serde_json::json!("hi"));
}

#[test]
fn glob_grep_workflow() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/a.rs"), "fn a() { x.unwrap() }").unwrap();
    std::fs::write(dir.path().join("src/b.rs"), "fn b() {}").unwrap();
    let r = run(
        dir.path(),
        r#"
        var files = glob("src/**/*.rs");
        var findings = [];
        for (var i = 0; i < files.length; i++) {
            var matches = grep({ pattern: "unwrap", path: files[i] });
            if (matches.length > 0) {
                findings.push({ path: files[i], matches: matches });
            }
        }
        return findings;
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    let findings = r.value.as_array().unwrap();
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0]["path"], serde_json::json!("src/a.rs"));
    assert_eq!(findings[0]["matches"][0]["line"], serde_json::json!(1));
}

#[test]
fn exec_returns_handles() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        var t = exec({ command: ["/bin/sh", "-c", "echo hello; echo oops 1>&2"] });
        return {
            exitCode: t.exitCode,
            out: t.stdout.read(),
            errs: t.stderr.grep("oops")
        };
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value["exitCode"], serde_json::json!(0));
    assert_eq!(r.value["out"]["content"], serde_json::json!("hello"));
    let errs = r.value["errs"]["matches"].as_array().unwrap();
    assert_eq!(errs.len(), 1);
    assert_eq!(errs[0]["text"], serde_json::json!("oops"));
}

#[test]
fn canonical_exec_uses_same_handle_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        var t = tool("exec", { command: ["/bin/echo", "canonical"] });
        return t.stdout.read().content;
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value, serde_json::json!("canonical"));
}

#[test]
fn infinite_loop_is_interrupted() {
    let dir = tempfile::tempdir().unwrap();
    let budget = PtcBudget {
        wall_time_ms: 300,
        instruction_limit: None,
        ..PtcBudget::default()
    };
    let rt = runtime_with_budget(dir.path(), budget);
    let r = execute(
        &rt,
        dir.path(),
        "while (true) { }",
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(r.outcome, PtcOutcome::BudgetExceeded("wall time"));
}

#[test]
fn tool_call_budget_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let budget = PtcBudget {
        max_tool_calls: 3,
        ..PtcBudget::default()
    };
    let rt = runtime_with_budget(dir.path(), budget);
    let r = execute(
        &rt,
        dir.path(),
        r#"
        for (var i = 0; i < 10; i++) {
            glob("*.none");
        }
        return "done";
        "#,
        Arc::new(AtomicBool::new(false)),
    );
    assert!(matches!(r.outcome, PtcOutcome::Failed(_)));
    assert!(r.tool_calls >= 3);
}

#[test]
fn sandbox_escape_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        var r = read("../etc/passwd");
        return r.error;
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert!(r.value.as_str().unwrap().contains("workspace"));
}

#[test]
fn syntax_error_reported_for_model_repair() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(dir.path(), "let x = 1; return x;");
    // `let` is unsupported in this dialect: parse error must surface
    // as Failed with a diagnostic the model can repair from.
    assert!(matches!(r.outcome, PtcOutcome::Failed(_)));
}

#[test]
fn json_object_result_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        var obj = { a: 1, b: "two", c: [true, null], d: { e: 2.5 } };
        return obj;
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value["a"], serde_json::json!(1));
    assert_eq!(r.value["b"], serde_json::json!("two"));
    assert_eq!(r.value["c"][0], serde_json::json!(true));
    assert_eq!(r.value["c"][1], serde_json::json!(null));
    assert_eq!(r.value["d"]["e"], serde_json::json!(2.5));
}

#[test]
fn js_language_features_available() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        function fib(n) {
            if (n < 2) { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        var xs = [];
        for (var i = 0; i < 6; i++) { xs.push(fib(i)); }
        var s = xs.map(function(v) { return v * 10; }).join(",");
        var parsed = JSON.parse("[1,2,3]");
        var m = {};
        try {
            null.x;
        } catch (e) {
            m.caught = true;
        }
        return { xs: xs, s: s, sum: parsed.length, caught: m.caught };
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value["xs"].as_array().unwrap().len(), 6);
    assert_eq!(r.value["s"], serde_json::json!("0,10,10,20,30,50"));
    assert_eq!(r.value["sum"], serde_json::json!(3));
    assert_eq!(r.value["caught"], serde_json::json!(true));
}

#[test]
fn result_handle_supports_filters_and_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        var t = exec({ command: ["/bin/sh", "-c", "printf 'one\\ntwo\\nthree\\n'"] });
        return {
            length: t.stdout.length,
            head: t.stdout.head(1).content,
            tail: t.stdout.tail(1).content,
            range: t.stdout.read(1, 1).content
        };
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value["length"], serde_json::json!(14));
    assert_eq!(r.value["head"], serde_json::json!("one"));
    assert_eq!(r.value["tail"], serde_json::json!("three"));
    assert_eq!(r.value["range"], serde_json::json!("two"));
}

#[test]
fn process_budget_is_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let budget = PtcBudget {
        max_processes: 1,
        ..PtcBudget::default()
    };
    let rt = runtime_with_budget(dir.path(), budget);
    let r = execute(
        &rt,
        dir.path(),
        r#"
        exec({ command: ["/bin/echo", "one"] });
        exec({ command: ["/bin/echo", "two"] });
        return "unreachable";
        "#,
        Arc::new(AtomicBool::new(false)),
    );
    assert!(matches!(r.outcome, PtcOutcome::Failed(_)));
}

#[test]
fn host_calls_are_traced_with_composable_effects_and_revisions() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        write("trace.txt", "hello");
        return read("trace.txt").content;
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_ne!(r.start_revision, r.end_revision);
    assert!(
        matches!(&r.events[0], PtcEvent::HostCallStarted { call_id: 1, name, .. } if name == "write")
    );
    assert!(r.events.iter().any(|event| matches!(event,
        PtcEvent::HostCallCompleted { call_id: 1, name, effects: ToolEffects { reads_workspace: true, mutates_workspace: true, process: false, .. }, ok: true, paths, .. }
        if name == "write" && paths == &["trace.txt".to_string()]
    )));
    assert!(r.events.iter().any(|event| matches!(event,
        PtcEvent::HostCallCompleted { call_id: 2, name, effects: ToolEffects { reads_workspace: true, mutates_workspace: false, process: false, .. }, ok: true, .. }
        if name == "read"
    )));
    assert!(r.events.iter().any(|event| matches!(
        event,
        PtcEvent::WorkspaceRevisionChanged {
            source: mh::workspace::RevisionSource::Tool,
            ..
        }
    )));
}

#[test]
fn batch_preserves_order_isolates_errors_and_rejects_mutation() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "alpha").unwrap();
    std::fs::write(dir.path().join("c.txt"), "charlie").unwrap();
    let r = run(
        dir.path(),
        r#"
        var reads = batch("read", [
            {path: "a.txt"},
            {path: "missing.txt"},
            {path: "c.txt"}
        ]);
        var writes = batch("write", [{path: "x.txt", content: "bad"}]);
        return {reads: reads, writes: writes};
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value["reads"][0]["content"], "alpha");
    assert!(r.value["reads"][1]["error"].as_str().is_some());
    assert_eq!(r.value["reads"][2]["content"], "charlie");
    assert_eq!(
        r.value["writes"][0]["error"],
        "tool 'write' is not batch-safe"
    );
    assert!(!dir.path().join("x.txt").exists());
    assert_eq!(r.tool_calls, 3);
}

#[test]
fn batch_exec_uses_bounded_parallel_host_workers() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime_with_budget(
        dir.path(),
        PtcBudget {
            max_parallel_tools: 2,
            ..PtcBudget::default()
        },
    );
    let started = std::time::Instant::now();
    let r = execute(
        &rt,
        dir.path(),
        r#"
        var out = batch("exec", [
            {command: ["/bin/sh", "-c", "sleep 0.15; printf 0"]},
            {command: ["/bin/sh", "-c", "sleep 0.15; printf 1"]},
            {command: ["/bin/sh", "-c", "sleep 0.15; printf 2"]},
            {command: ["/bin/sh", "-c", "sleep 0.15; printf 3"]}
        ]);
        return [out[0].stdout.read().content, out[1].stdout.read().content,
                out[2].stdout.read().content, out[3].stdout.read().content];
        "#,
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value, serde_json::json!(["0", "1", "2", "3"]));
    assert!(started.elapsed() < std::time::Duration::from_millis(550));
    assert_eq!(r.tool_calls, 4);
}

#[test]
fn result_handles_expose_complete_v2_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime_with_budget(
        dir.path(),
        PtcBudget {
            max_result_bytes: 4,
            ..PtcBudget::default()
        },
    );
    let r = execute(
        &rt,
        dir.path(),
        r#"
        var out = exec({command: ["/bin/sh", "-c", "printf 0123456789"]});
        return {
            durationMs: out.durationMs,
            id: out.stdout.id,
            length: out.stdout.length,
            totalBytes: out.stdout.totalBytes,
            truncated: out.stdout.truncated,
            kind: out.stdout.kind,
            retained: out.stdout.read().content
        };
        "#,
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert!(r.value["durationMs"].as_u64().is_some());
    assert!(r.value["id"].as_u64().is_some());
    assert_eq!(r.value["length"], 4);
    assert_eq!(r.value["totalBytes"], 10);
    assert_eq!(r.value["truncated"], true);
    assert_eq!(r.value["kind"], "stdout");
    assert_eq!(r.value["retained"], "6789");
}

#[test]
fn collection_truncation_remains_visible_on_iterable_arrays() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("many.txt"), "hit one\nhit two\n").unwrap();
    let r = run(
        dir.path(),
        r#"
        var matches = grep({pattern: "hit", path: "many.txt", max: 1});
        return {length: matches.length, first: matches[0].line, truncated: matches.truncated};
        "#,
    );
    assert_eq!(
        r.value,
        serde_json::json!({"length": 1, "first": 1, "truncated": true})
    );
}

#[test]
fn evidence_records_current_revision_and_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        write("changed.txt", "yes");
        var check = exec({command: ["/bin/echo", "verified"]});
        evidence("tests", check.exitCode === 0, {stdout: check.stdout.id, note: "cargo test"});
        return true;
        "#,
    );
    let evidence = r
        .events
        .iter()
        .find_map(|event| match event {
            PtcEvent::EvidenceRecorded { evidence } => Some(evidence),
            _ => None,
        })
        .unwrap();
    assert_eq!(evidence.kind, "tests");
    assert!(evidence.ok);
    assert_eq!(evidence.revision, r.end_revision);
    assert_eq!(evidence.task_id, TaskId(1));
    assert_eq!(evidence.execution_id, ExecutionId(1));
    assert_eq!(evidence.result_ids.len(), 1);
    assert_eq!(evidence.note.as_deref(), Some("cargo test"));
}

#[test]
fn unsupported_syntax_has_structured_repair_diagnostic() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(dir.path(), "let x = 1; return x;");
    let diagnostic = r.diagnostic.unwrap();
    assert_eq!(diagnostic.kind, PtcDiagnosticKind::UnsupportedSyntax);
    assert!(diagnostic.hint.unwrap().contains("var"));
    assert_eq!(r.value["ptcError"]["kind"], "unsupported_syntax");
}

#[test]
fn cancellation_interrupts_a_running_batch() {
    let dir = tempfile::tempdir().unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let signal = cancelled.clone();
    let trigger = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        signal.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let rt = runtime(dir.path());
    let r = execute(
        &rt,
        dir.path(),
        r#"
        batch("exec", [
            {command: ["/bin/sh", "-c", "sleep 2"]},
            {command: ["/bin/sh", "-c", "sleep 2"]}
        ]);
        return "unreachable";
        "#,
        cancelled.clone(),
    );
    trigger.join().unwrap();
    assert_eq!(r.outcome, PtcOutcome::Interrupted);
    assert!(r.duration_ms < 1_000);
}

#[derive(Default)]
struct RecordingSink(Mutex<Vec<PtcEvent>>);

impl mh::ptc::PtcEventSink for RecordingSink {
    fn emit(&self, event: PtcEvent) -> Result<(), mh::ptc::PtcEventSinkError> {
        self.0.lock().expect("recording sink poisoned").push(event);
        Ok(())
    }
}

#[test]
fn exec_effect_is_process_plus_actual_mutation_on_any_exit() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"exec({command:["/bin/sh","-c","printf changed > changed.txt; exit 7"]}); return true;"#,
    );
    assert!(r.events.iter().any(|event| matches!(event,
        PtcEvent::HostCallCompleted { name, effects: ToolEffects { process: true, mutates_workspace: true, .. }, .. } if name == "exec"
    )));
    let r = run(
        dir.path(),
        r#"exec({command:["/bin/sh","-c","exit 3"]}); return true;"#,
    );
    assert!(r.events.iter().any(|event| matches!(event,
        PtcEvent::HostCallCompleted { name, effects: ToolEffects { process: true, mutates_workspace: false, .. }, .. } if name == "exec"
    )));
}

#[test]
fn live_sink_receives_events_before_execute_returns() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path());
    let start_revision = WorkspaceTrackerImpl::open(dir.path())
        .unwrap()
        .current_revision()
        .unwrap()
        .id;
    let sink = Arc::new(RecordingSink::default());
    let result = rt.execute(
        r#"write("live.txt", "yes"); return true;"#,
        Arc::new(AtomicBool::new(false)),
        PtcExecution {
            task_id: TaskId(4),
            agent: AgentId::ROOT,
            execution_id: ExecutionId(9),
            start_revision,
        },
        Some(sink.clone()),
    );
    assert_eq!(
        *sink.0.lock().expect("recording sink poisoned"),
        result.events
    );
    assert!(
        sink.0
            .lock()
            .expect("recording sink poisoned")
            .iter()
            .any(|event| matches!(event, PtcEvent::WorkspaceRevisionChanged { .. }))
    );
}

#[test]
fn checkpoint_restore_returns_exact_revision_and_emits_restore() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        write("state.txt", "one");
        var cp = checkpoint();
        write("state.txt", "two");
        var restored = restore(cp);
        return {cp: cp, restored: restored, content: read("state.txt").content};
    "#,
    );
    assert_eq!(r.value["cp"], r.value["restored"]);
    assert_eq!(r.value["content"], "one");
    assert!(r.events.iter().any(|event| matches!(
        event,
        PtcEvent::WorkspaceRevisionChanged {
            source: mh::workspace::RevisionSource::Restore,
            ..
        }
    )));
}
