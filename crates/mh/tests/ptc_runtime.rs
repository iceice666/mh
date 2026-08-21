//! PTC runtime integration tests against the real MicroQuickJS engine.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use mh::ptc::{PtcBudget, PtcOutcome, PtcRuntime};
use mh::tools::{Capabilities, ResultStore};

fn runtime(dir: &std::path::Path) -> PtcRuntime {
    PtcRuntime::new(
        Capabilities::new(dir.to_path_buf()),
        ResultStore::new(),
        PtcBudget::default(),
    )
}

fn run(dir: &std::path::Path, program: &str) -> mh::ptc::PtcResult {
    let no_cancel = Arc::new(AtomicBool::new(false));
    runtime(dir).execute(program, &no_cancel)
}

#[test]
fn pure_js_evaluates() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(dir.path(), "var x = 1 + 2; return x * 2;");
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value, serde_json::json!(6));
}

#[test]
fn tool_read_write_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        dir.path(),
        r#"
        write("hello.txt", "line1\nline2");
        var out = read("hello.txt");
        return out.content;
        "#,
    );
    assert_eq!(r.outcome, PtcOutcome::Completed);
    assert_eq!(r.value, serde_json::json!("line1\nline2"));
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
    let rt = PtcRuntime::new(
        Capabilities::new(dir.path().to_path_buf()),
        ResultStore::new(),
        budget,
    );
    let no_cancel = Arc::new(AtomicBool::new(false));
    let r = rt.execute("while (true) { }", &no_cancel);
    assert_eq!(r.outcome, PtcOutcome::BudgetExceeded("wall time"));
}

#[test]
fn tool_call_budget_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let budget = PtcBudget {
        max_tool_calls: 3,
        ..PtcBudget::default()
    };
    let rt = PtcRuntime::new(
        Capabilities::new(dir.path().to_path_buf()),
        ResultStore::new(),
        budget,
    );
    let no_cancel = Arc::new(AtomicBool::new(false));
    let r = rt.execute(
        r#"
        for (var i = 0; i < 10; i++) {
            glob("*.none");
        }
        return "done";
        "#,
        &no_cancel,
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
    let rt = PtcRuntime::new(
        Capabilities::new(dir.path().to_path_buf()),
        ResultStore::new(),
        budget,
    );
    let no_cancel = Arc::new(AtomicBool::new(false));
    let r = rt.execute(
        r#"
        exec({ command: ["/bin/echo", "one"] });
        exec({ command: ["/bin/echo", "two"] });
        return "unreachable";
        "#,
        &no_cancel,
    );
    assert!(matches!(r.outcome, PtcOutcome::Failed(_)));
}
