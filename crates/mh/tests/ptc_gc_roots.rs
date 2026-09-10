//! GC root discipline for host-built values.
//!
//! The engine keeps temporary roots on a strict LIFO stack: `JS_PopGCRef`
//! assigns `ctx->top_gc_ref = ref->prev`, so releasing anything other than the
//! current top silently unroots every root pushed after it. The compacting GC
//! then moves or reclaims values the host still holds, which crashed the
//! process (`SIGSEGV`) instead of failing.
//!
//! These programs drive the host value builders that nest roots most deeply —
//! `batch` over `glob`, and `exec` handle objects — and assert on the values
//! that come back. In a debug build `GcRoot` also asserts LIFO order directly,
//! so an out-of-order release fails here loudly rather than corrupting a heap.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use mh::checkpoint::CheckpointStore;
use mh::identity::{AgentId, ExecutionId, TaskId};
use mh::ptc::{PtcBudget, PtcExecution, PtcOutcome, PtcRuntime};
use mh::tools::{Capabilities, ResultStore};
use mh::workspace::{WorkspaceTracker, WorkspaceTrackerImpl};

fn run(dir: &std::path::Path, program: &str) -> mh::ptc::PtcResult {
    let tracker_impl = WorkspaceTrackerImpl::open(dir).unwrap();
    let tracker: Arc<dyn WorkspaceTracker> = Arc::new(tracker_impl);
    let start_revision = tracker.current_revision().unwrap().id;
    let checkpoints = CheckpointStore::open(dir, tracker.clone()).unwrap();
    PtcRuntime::new(
        Capabilities::new(dir),
        ResultStore::new(),
        PtcBudget::default(),
        tracker,
        checkpoints,
    )
    .execute(
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

fn seed(dir: &std::path::Path, files: usize) {
    let sub = dir.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    for index in 0..files {
        std::fs::write(sub.join(format!("file-{index:05}.txt")), "x").unwrap();
    }
}

/// The exact shape that crashed a real session: several globs through `batch`,
/// whose results are collected into one returned object. Allocation continues
/// afterwards, so any unrooted intermediate is exposed to a moving GC.
#[test]
fn batched_collections_survive_allocation_after_construction() {
    let dir = tempfile::tempdir().unwrap();
    seed(dir.path(), 2_000);

    let result = run(
        dir.path(),
        "var files = batch('glob', [{pattern:'*'},{pattern:'**/*'}]);\n\
         var extra = batch('glob', [{pattern:'sub/*'},{pattern:'**/file-0*'}]);\n\
         var pad = [];\n\
         for (var i = 0; i < 20000; i++) { pad.push('padding-' + i); }\n\
         return {a: files[1].length, b: extra[0].length, pad: pad.length};",
    );

    assert_eq!(result.outcome, PtcOutcome::Completed, "{:?}", result.value);
    assert_eq!(
        result.value["a"], 2_000,
        "`**/*` must see every seeded file (directories are not listed)"
    );
    assert_eq!(
        result.value["b"], 2_000,
        "`sub/*` must see every seeded file"
    );
    assert_eq!(
        result.value["pad"], 20_000,
        "post-construction allocation ran"
    );
}

/// `exec` builds the most deeply nested host value: an object holding two
/// handle objects, each holding five closures built from rooted strings.
#[test]
fn exec_handles_survive_allocation_after_construction() {
    let dir = tempfile::tempdir().unwrap();

    let result = run(
        dir.path(),
        "var r = exec({command:['printf','alpha']});\n\
         var pad = [];\n\
         for (var i = 0; i < 20000; i++) { pad.push('padding-' + i); }\n\
         return {out: r.stdout.read(), kind: r.stdout.kind, code: r.exitCode};",
    );

    assert_eq!(result.outcome, PtcOutcome::Completed, "{:?}", result.value);
    assert_eq!(
        result.value["out"]["content"], "alpha",
        "the stdout handle still reads its result after a GC"
    );
    assert_eq!(result.value["kind"], "stdout");
    assert_eq!(result.value["code"], 0);
}

/// Deeply nested JSON exercises `json_to_vm` recursion, where each level roots
/// a parent while building children.
#[test]
fn deeply_nested_host_json_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "alpha\nbeta\n").unwrap();

    let result = run(
        dir.path(),
        "var acc = [];\n\
         for (var i = 0; i < 200; i++) { acc.push(read('a.txt')); }\n\
         var pad = [];\n\
         for (var j = 0; j < 20000; j++) { pad.push('padding-' + j); }\n\
         return {first: acc[0].content, last: acc[199].content, n: acc.length};",
    );

    assert_eq!(result.outcome, PtcOutcome::Completed, "{:?}", result.value);
    assert_eq!(result.value["first"], "alpha\nbeta\n");
    assert_eq!(
        result.value["last"], "alpha\nbeta\n",
        "an early host-built value must not be corrupted by later allocation"
    );
    assert_eq!(result.value["n"], 200);
}
