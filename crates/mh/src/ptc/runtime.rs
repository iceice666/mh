//! PTC runtime (spec §7, §8, §14, §15).
//!
//! One [`PtcRuntime`] executes a single model-generated program:
//! a fresh MicroQuickJS VM (spec invariant III: VMs are disposable)
//! with the mh ABI globals installed. Tools dispatch through the
//! single C closure registered in the vendored stdlib table.
//!
//! GC safety: all JSValues handled inside a closure call live in the
//! interpreter's frame (rooted by the GC stack scan). Values built in
//! Rust that must survive further allocation are pushed through
//! [`GcRoot`], which wraps `JS_PushGCRef`/`JS_PopGCRef`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::checkpoint::{Checkpoint, CheckpointStore};
use crate::delegation::{DelegationHost, DelegationOptions};
use crate::identity::{ExecutionId, RevisionId, TaskId};
use crate::session::EvidenceRecord;
use crate::tools::capability::Capabilities;
use crate::tools::fs_tools;
use crate::tools::store::ResultStore;
use crate::tools::{ResultId, ToolEffects};
use crate::workspace::{RevisionSource, WorkspaceTracker};

use super::wrapper::{JSValue, Vm, VmError};

/// Execution budget (spec §14).
#[derive(Debug, Clone)]
pub struct PtcBudget {
    pub wall_time_ms: u64,
    pub instruction_limit: Option<u64>,
    pub max_tool_calls: usize,
    pub max_processes: usize,
    pub max_result_bytes: usize,
    pub max_output_bytes: usize,
    /// JS heap size for the VM.
    pub heap_bytes: usize,
    /// Maximum number of host operations run concurrently by `batch`.
    pub max_parallel_tools: usize,
}

impl Default for PtcBudget {
    fn default() -> Self {
        Self {
            wall_time_ms: 120_000,
            instruction_limit: Some(100_000_000),
            max_tool_calls: 200,
            max_processes: 16,
            max_parallel_tools: 8,
            max_result_bytes: 4 << 20,
            max_output_bytes: 64 * 1024,
            heap_bytes: 4 << 20,
        }
    }
}

/// Why a PTC execution ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtcOutcome {
    Completed,
    BudgetExceeded(&'static str),
    Interrupted,
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PtcDiagnosticKind {
    Syntax,
    Runtime,
    UnsupportedSyntax,
    ToolBudget,
    ProcessBudget,
    InstructionBudget,
    WallTime,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PtcDiagnostic {
    pub kind: PtcDiagnosticKind,
    pub message: String,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub hint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostCallOutcome {
    Ok,
    Error,
    Cancelled,
    BudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtcEvent {
    HostCallStarted {
        task_id: TaskId,
        execution_id: ExecutionId,
        call_id: u64,
        name: String,
        args_hash: u64,
    },
    HostCallCompleted {
        task_id: TaskId,
        execution_id: ExecutionId,
        call_id: u64,
        name: String,
        args_hash: u64,
        effects: ToolEffects,
        outcome: HostCallOutcome,
        ok: bool,
        duration_ms: u64,
        result_ids: Vec<ResultId>,
        paths: Vec<String>,
    },
    WorkspaceRevisionChanged {
        task_id: TaskId,
        execution_id: ExecutionId,
        from: RevisionId,
        to: RevisionId,
        added: Vec<String>,
        modified: Vec<String>,
        deleted: Vec<String>,
        source: RevisionSource,
    },
    EvidenceRecorded {
        evidence: EvidenceRecord,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtcEventSinkError(pub String);

impl std::fmt::Display for PtcEventSinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PtcEventSinkError {}

impl PtcEventSinkError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
pub trait PtcEventSink: Send + Sync {
    fn emit(&self, event: PtcEvent) -> Result<(), PtcEventSinkError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtcExecution {
    pub task_id: TaskId,
    pub execution_id: ExecutionId,
    pub start_revision: RevisionId,
}

/// Result of one PTC execution.
#[derive(Debug, Clone)]
pub struct PtcResult {
    pub value: Value,
    pub outcome: PtcOutcome,
    pub task_id: TaskId,
    pub execution_id: ExecutionId,
    pub start_revision: RevisionId,
    pub end_revision: RevisionId,
    pub tool_calls: usize,
    pub duration_ms: u64,
    pub events: Vec<PtcEvent>,
    pub diagnostic: Option<PtcDiagnostic>,
}

struct ExecState {
    cancelled: Arc<AtomicBool>,
    deadline: Instant,
    execution: PtcExecution,
    current_revision: Mutex<RevisionId>,
    tool_calls: AtomicU64,
    process_calls: AtomicU64,
    interrupt_polls: AtomicU64,
    next_call_id: AtomicU64,
    events: Mutex<Vec<PtcEvent>>,
    sink: Option<Arc<dyn PtcEventSink>>,
    checkpoints: Mutex<HashMap<u64, Checkpoint>>,
    sink_error: Mutex<Option<PtcEventSinkError>>,
    max_tool_calls: u64,
    max_processes: u64,
    max_interrupt_polls: Option<u64>,
}

/// Interrupt handler bridging cancellation/deadline into the VM.
/// `opaque` is the context-opaque DispatchBox.
unsafe extern "C" fn interrupt_handler(
    _ctx: *mut mh_quickjs_sys::JSContext,
    opaque: *mut c_void,
) -> i32 {
    let dispatch = unsafe { &*(opaque as *const DispatchBox) };
    let state = dispatch.state;
    if state.cancelled.load(Ordering::Relaxed) {
        return 1;
    }
    if Instant::now() >= state.deadline {
        return 1;
    }
    let polls = state.interrupt_polls.fetch_add(1, Ordering::Relaxed) + 1;
    if state.max_interrupt_polls.is_some_and(|max| polls > max) {
        return 1;
    }
    0
}

/// Rooted JS value guard (GC-move-safe).
struct GcRoot<'v> {
    vm: &'v Vm,
    root: Box<mh_quickjs_sys::JSGCRef>,
    active: bool,
}

impl<'v> GcRoot<'v> {
    /// # Safety
    /// `val` must belong to `vm`'s context.
    unsafe fn new(vm: &'v Vm, val: JSValue) -> Self {
        let mut root = Box::new(mh_quickjs_sys::JSGCRef {
            val: JSValue::UNDEFINED,
            prev: std::ptr::null_mut(),
        });
        let slot = unsafe { mh_quickjs_sys::JS_PushGCRef(vm.ctx(), root.as_mut()) };
        unsafe { *slot = val };
        Self {
            vm,
            root,
            active: true,
        }
    }

    fn get(&self) -> JSValue {
        self.root.val
    }

    fn pop(mut self) -> JSValue {
        self.active = false;
        unsafe { mh_quickjs_sys::JS_PopGCRef(self.vm.ctx(), self.root.as_mut()) }
    }
}

impl Drop for GcRoot<'_> {
    fn drop(&mut self) {
        if self.active {
            unsafe { mh_quickjs_sys::JS_PopGCRef(self.vm.ctx(), self.root.as_mut()) };
        }
    }
}

/// The PTC runtime. One instance per program execution.
pub struct PtcRuntime {
    caps: Capabilities,
    store: ResultStore,
    budget: PtcBudget,
    tracker: Arc<dyn WorkspaceTracker>,
    checkpoints: CheckpointStore,
    delegation: Option<Arc<dyn DelegationHost>>,
}

impl PtcRuntime {
    pub fn new(
        caps: Capabilities,
        store: ResultStore,
        budget: PtcBudget,
        tracker: Arc<dyn WorkspaceTracker>,
        checkpoints: CheckpointStore,
    ) -> Self {
        Self {
            caps,
            store,
            budget,
            tracker,
            checkpoints,
            delegation: None,
        }
    }

    pub fn with_delegation(mut self, delegation: Arc<dyn DelegationHost>) -> Self {
        self.delegation = Some(delegation);
        self
    }
    pub fn execute(
        &self,
        program: &str,
        cancelled: Arc<AtomicBool>,
        execution: PtcExecution,
        sink: Option<Arc<dyn PtcEventSink>>,
    ) -> PtcResult {
        let start = Instant::now();
        let state = Box::new(ExecState {
            cancelled,
            deadline: start + Duration::from_millis(self.budget.wall_time_ms),
            current_revision: Mutex::new(execution.start_revision.clone()),
            execution,
            tool_calls: AtomicU64::new(0),
            process_calls: AtomicU64::new(0),
            interrupt_polls: AtomicU64::new(0),
            next_call_id: AtomicU64::new(1),
            events: Mutex::new(Vec::new()),
            checkpoints: Mutex::new(HashMap::new()),
            sink,
            sink_error: Mutex::new(None),
            max_tool_calls: self.budget.max_tool_calls as u64,
            max_processes: self.budget.max_processes as u64,
            max_interrupt_polls: self
                .budget
                .instruction_limit
                .map(|limit| limit.div_ceil(mh_quickjs_sys::JS_INTERRUPT_COUNTER_INIT)),
        });
        let state_ptr = Box::into_raw(state);

        let mut vm = match Vm::with_heap(self.budget.heap_bytes) {
            Ok(v) => v,
            Err(e) => {
                let state = unsafe { Box::from_raw(state_ptr) };
                return empty_result(PtcOutcome::Failed(e.to_string()), start, &state.execution);
            }
        };

        let dispatch = Box::new(DispatchBox {
            handler: dispatch_closure,
            state: unsafe { &*state_ptr },
            caps: &self.caps,
            store: &self.store,
            budget: &self.budget,
            tracker: self.tracker.as_ref(),
            checkpoints: &self.checkpoints,
            delegation: self.delegation.as_deref(),
        });
        let dispatch_ptr = Box::into_raw(dispatch);
        unsafe { vm.install_opaque(dispatch_ptr.cast::<c_void>(), interrupt_handler) };

        let execution_result = (|| -> (Value, PtcOutcome, Option<PtcDiagnostic>) {
            if let Err(e) = self.install_globals(&mut vm) {
                let message = e.to_string();
                return (
                    Value::Null,
                    PtcOutcome::Failed(message.clone()),
                    Some(diagnostic(PtcDiagnosticKind::Runtime, message)),
                );
            }
            let wrapped = format!("(function() {{\n{program}\n}})()");
            let val = match vm.eval(&wrapped, "ptc.js") {
                Ok(v) => v,
                Err(VmError::Exception(msg)) => {
                    let diag = classify_diagnostic(
                        program,
                        &msg,
                        unsafe { &(*state_ptr).cancelled },
                        dispatch_ptr,
                    );
                    let outcome = match diag.kind {
                        PtcDiagnosticKind::Cancelled => PtcOutcome::Interrupted,
                        PtcDiagnosticKind::WallTime => PtcOutcome::BudgetExceeded("wall time"),
                        PtcDiagnosticKind::InstructionBudget => {
                            PtcOutcome::BudgetExceeded("instruction limit")
                        }
                        _ => PtcOutcome::Failed(msg.clone()),
                    };
                    return (json!({ "ptcError": &diag }), outcome, Some(diag));
                }
                Err(e) => {
                    let message = e.to_string();
                    let diag = diagnostic(PtcDiagnosticKind::Runtime, message.clone());
                    return (
                        json!({ "ptcError": &diag }),
                        PtcOutcome::Failed(message),
                        Some(diag),
                    );
                }
            };
            (
                vm_to_json(&vm, val, self.budget.max_output_bytes),
                PtcOutcome::Completed,
                None,
            )
        })();

        // The VM may invoke the opaque callback while it is being destroyed.
        // Its backing state and dispatch box must therefore outlive this drop.
        drop(vm);
        let dispatch = unsafe { Box::from_raw(dispatch_ptr) };
        drop(dispatch);
        let state = unsafe { Box::from_raw(state_ptr) };
        let sink_error = state
            .sink_error
            .lock()
            .expect("sink error poisoned")
            .clone();
        let (outcome, diagnostic) = if let Some(error) = sink_error {
            let message = format!("PTC event sink failed: {error}");
            (
                PtcOutcome::Failed(message.clone()),
                Some(diagnostic(PtcDiagnosticKind::Runtime, message)),
            )
        } else {
            (execution_result.1, execution_result.2)
        };
        let end_revision = state
            .current_revision
            .lock()
            .expect("revision poisoned")
            .clone();
        PtcResult {
            value: execution_result.0,
            outcome,
            task_id: state.execution.task_id,
            execution_id: state.execution.execution_id,
            start_revision: state.execution.start_revision.clone(),
            end_revision,
            tool_calls: state.tool_calls.load(Ordering::Relaxed) as usize,
            duration_ms: elapsed_ms(start),
            events: state.events.lock().expect("event trace poisoned").clone(),
            diagnostic,
        }
    }

    fn install_globals(&self, vm: &mut Vm) -> Result<(), VmError> {
        let v = vm.values();
        let make_fn = |vm: &Vm, name: &str| {
            let v = vm.values();
            let params = unsafe { GcRoot::new(vm, v.string(name)) };
            let f = v.closure(params.get());
            drop(params);
            f
        };
        let g = unsafe { mh_quickjs_sys::JS_GetGlobalObject(vm.ctx()) };
        for name in [
            "tool",
            "call_tool",
            "read",
            "write",
            "edit",
            "glob",
            "grep",
            "exec",
            "batch",
            "delegate",
            "delegate_batch",
            "integrate",
            "evidence",
            "checkpoint",
            "restore",
        ] {
            let f = unsafe { GcRoot::new(vm, make_fn(vm, name)) };
            v.set(g, name, f.get())?;
        }
        // tools.read(...) alias family (spec §18.1).
        let tools = unsafe { GcRoot::new(vm, v.object()) };
        for name in [
            "read",
            "write",
            "edit",
            "glob",
            "grep",
            "exec",
            "call_tool",
            "batch",
            "delegate",
            "delegate_batch",
            "integrate",
            "evidence",
            "checkpoint",
            "restore",
        ] {
            let f = unsafe { GcRoot::new(vm, make_fn(vm, name)) };
            v.set(tools.get(), name, f.get())?;
        }
        v.set(g, "tools", tools.get())?;
        Ok(())
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn empty_result(outcome: PtcOutcome, start: Instant, execution: &PtcExecution) -> PtcResult {
    PtcResult {
        value: Value::Null,
        outcome,
        task_id: execution.task_id,
        execution_id: execution.execution_id,
        start_revision: execution.start_revision.clone(),
        end_revision: execution.start_revision.clone(),
        tool_calls: 0,
        duration_ms: elapsed_ms(start),
        events: Vec::new(),
        diagnostic: None,
    }
}

fn diagnostic(kind: PtcDiagnosticKind, message: String) -> PtcDiagnostic {
    PtcDiagnostic {
        kind,
        message,
        line: None,
        column: None,
        hint: None,
    }
}

fn classify_diagnostic(
    program: &str,
    message: &str,
    cancelled: &AtomicBool,
    dispatch: *mut DispatchBox,
) -> PtcDiagnostic {
    let lower = message.to_ascii_lowercase();
    let mut diag = if cancelled.load(Ordering::Relaxed) || lower.contains("cancelled") {
        diagnostic(PtcDiagnosticKind::Cancelled, message.to_string())
    } else if Instant::now() >= state_deadline(dispatch) {
        diagnostic(PtcDiagnosticKind::WallTime, message.to_string())
    } else if instruction_budget_exceeded(dispatch) {
        diagnostic(PtcDiagnosticKind::InstructionBudget, message.to_string())
    } else if lower.contains("tool call budget") {
        diagnostic(PtcDiagnosticKind::ToolBudget, message.to_string())
    } else if lower.contains("process budget") {
        diagnostic(PtcDiagnosticKind::ProcessBudget, message.to_string())
    } else if contains_unsupported_lexical_declaration(program)
        || lower.contains("let")
        || lower.contains("const")
        || lower.contains("lexical")
    {
        let mut d = diagnostic(PtcDiagnosticKind::UnsupportedSyntax, message.to_string());
        d.hint = Some("MicroQuickJS PTC uses var; replace let/const with var".to_string());
        d
    } else if lower.contains("syntax") || lower.contains("parse") {
        diagnostic(PtcDiagnosticKind::Syntax, message.to_string())
    } else {
        diagnostic(PtcDiagnosticKind::Runtime, message.to_string())
    };
    let numbers: Vec<u32> = message
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|part| part.parse().ok())
        .collect();
    diag.line = numbers.first().copied();
    diag.column = numbers.get(1).copied();
    diag
}

fn contains_unsupported_lexical_declaration(program: &str) -> bool {
    program
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .any(|word| matches!(word, "let" | "const"))
}

// ---------------------------------------------------------------------------
// Dispatch (called from C)
// ---------------------------------------------------------------------------

/// Layout installed as the context opaque: handler first so the
/// interrupt entry can reinterpret the same pointer.
#[repr(C)]
struct DispatchBox<'a> {
    handler: mh_quickjs_sys::MhClosureHandler,
    state: &'a ExecState,
    caps: &'a Capabilities,
    store: &'a ResultStore,
    budget: &'a PtcBudget,
    tracker: &'a dyn WorkspaceTracker,
    checkpoints: &'a CheckpointStore,
    delegation: Option<&'a dyn DelegationHost>,
}

fn instruction_budget_exceeded(p: *mut DispatchBox) -> bool {
    let state = unsafe { (*p).state };
    state
        .max_interrupt_polls
        .is_some_and(|max| state.interrupt_polls.load(Ordering::Relaxed) > max)
}

fn state_deadline(p: *mut DispatchBox) -> Instant {
    unsafe { (*p).state.deadline }
}

/// The single mh closure entry point.
unsafe extern "C" fn dispatch_closure(
    ctx: *mut mh_quickjs_sys::JSContext,
    name: *const std::ffi::c_char,
    name_len: usize,
    this_val: *mut JSValue,
    argc: i32,
    argv: *mut JSValue,
) -> JSValue {
    let dispatch = unsafe { mh_quickjs_sys::JS_GetContextOpaque(ctx) } as *mut DispatchBox;
    if dispatch.is_null() {
        return JSValue::EXCEPTION;
    }
    let name_bytes = unsafe { std::slice::from_raw_parts(name.cast::<u8>(), name_len) };
    let name = String::from_utf8_lossy(name_bytes).into_owned();
    let receiver = unsafe { *this_val };

    let vm = unsafe { Vm::borrow_ctx(ctx) };
    let result = unsafe { run_host_call(&vm, dispatch, &name, receiver, argc, argv) };
    match result {
        Ok(v) => v,
        Err(msg) => unsafe { mh_quickjs_sys::throw_internal_error(ctx, &msg) },
    }
}

/// Executes one host call; all argument values are rooted by the callback frame.
unsafe fn run_host_call(
    vm: &Vm,
    dispatch: *mut DispatchBox,
    name: &str,
    receiver: JSValue,
    argc: i32,
    argv: *mut JSValue,
) -> Result<JSValue, String> {
    let d = unsafe { &*dispatch };
    let arg = |i: usize| -> Option<JSValue> {
        if (i as i32) < argc {
            Some(unsafe { *argv.add(i) })
        } else {
            None
        }
    };
    let v = vm.values();

    match name {
        "tool" | "call_tool" => {
            let tool_name = arg(0)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("tool(name, args): name required")?;
            let args = arg(1)
                .map(|a| vm_to_json(vm, a, d.budget.max_output_bytes))
                .unwrap_or(Value::Null);
            let out = execute_child(d, &tool_name, args, true);
            ensure_direct_host_call(&out)?;
            Ok(tool_value_to_vm(vm, d, &tool_name, &out))
        }
        "read" => {
            let args = read_args(&v, arg(0))?;
            let out = execute_child(d, "read", args, true);
            ensure_direct_host_call(&out)?;
            Ok(json_to_vm(vm, &out))
        }
        "write" => {
            let path = arg(0)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("write(path, content)")?;
            let content = arg(1)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("write(path, content)")?;
            let out = execute_child(
                d,
                "write",
                json!({ "path": path, "content": content }),
                true,
            );
            ensure_direct_host_call(&out)?;
            Ok(json_to_vm(vm, &out))
        }
        "edit" => {
            let args = arg(0).ok_or("edit(argsObject)")?;
            let out = execute_child(
                d,
                "edit",
                vm_to_json(vm, args, d.budget.max_output_bytes),
                true,
            );
            ensure_direct_host_call(&out)?;
            Ok(json_to_vm(vm, &out))
        }
        "glob" => {
            let pattern = arg(0)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("glob(pattern)")?;
            let out = execute_child(d, "glob", json!({ "pattern": pattern }), true);
            ensure_direct_host_call(&out)?;
            Ok(collection_to_vm(vm, &out, "files"))
        }
        "grep" => {
            let args = arg(0).ok_or("grep(argsObject)")?;
            let out = execute_child(
                d,
                "grep",
                vm_to_json(vm, args, d.budget.max_output_bytes),
                true,
            );
            ensure_direct_host_call(&out)?;
            Ok(collection_to_vm(vm, &out, "matches"))
        }
        "exec" => {
            let args = arg(0).ok_or("exec(argsObject)")?;
            let out = execute_child(
                d,
                "exec",
                vm_to_json(vm, args, d.budget.max_output_bytes),
                true,
            );
            ensure_direct_host_call(&out)?;
            Ok(exec_result_to_vm(vm, d, &out))
        }
        "checkpoint" => {
            let out = execute_child(d, "checkpoint", Value::Null, true);
            ensure_direct_host_call(&out)?;
            Ok(json_to_vm(vm, &out))
        }
        "restore" => {
            let checkpoint = arg(0).ok_or("restore(checkpoint)")?;
            let out = execute_child(
                d,
                "restore",
                vm_to_json(vm, checkpoint, d.budget.max_output_bytes),
                true,
            );
            ensure_direct_host_call(&out)?;
            Ok(json_to_vm(vm, &out))
        }
        "batch" => {
            if d.state.cancelled.load(Ordering::Relaxed) {
                return Err("cancelled".to_string());
            }
            let tool_name = arg(0)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("batch(name, args[]): name required")?;
            let items_value = arg(1).ok_or("batch(name, args[]): args array required")?;
            let items = vm_to_json(vm, items_value, d.budget.max_output_bytes);
            let items = items
                .as_array()
                .ok_or("batch(name, args[]): args must be an array")?;
            let outputs = execute_batch(d, &tool_name, items);
            if d.state.cancelled.load(Ordering::Relaxed)
                || outputs.iter().any(|output| {
                    output
                        .get("error")
                        .and_then(Value::as_str)
                        .is_some_and(|error| error.contains("cancel"))
                })
            {
                return Err("cancelled".to_string());
            }
            let arr = unsafe { GcRoot::new(vm, v.array(outputs.len())) };
            for (index, output) in outputs.iter().enumerate() {
                let child = unsafe { GcRoot::new(vm, tool_value_to_vm(vm, d, &tool_name, output)) };
                let _ = v.set_index(arr.get(), index as u32, child.get());
            }
            Ok(arr.pop())
        }
        "delegate" => {
            let host = d
                .delegation
                .ok_or("delegate: unavailable in child execution")?;
            let options = arg(0).ok_or("delegate(options)")?;
            let options: DelegationOptions =
                serde_json::from_value(vm_to_json(vm, options, d.budget.max_output_bytes))
                    .map_err(|error| format!("delegate: invalid options: {error}"))?;
            let revision = d
                .state
                .current_revision
                .lock()
                .expect("revision poisoned")
                .clone();
            Ok(json_to_vm(
                vm,
                &serde_json::to_value(host.delegate(&d.state.execution, &revision, options))
                    .unwrap(),
            ))
        }
        "delegate_batch" => {
            let host = d
                .delegation
                .ok_or("delegate_batch: unavailable in child execution")?;
            let options = arg(0).ok_or("delegate_batch(options[])")?;
            let options: Vec<DelegationOptions> =
                serde_json::from_value(vm_to_json(vm, options, d.budget.max_output_bytes))
                    .map_err(|error| format!("delegate_batch: invalid options: {error}"))?;
            let revision = d
                .state
                .current_revision
                .lock()
                .expect("revision poisoned")
                .clone();
            Ok(json_to_vm(
                vm,
                &serde_json::to_value(host.delegate_batch(&d.state.execution, &revision, options))
                    .unwrap(),
            ))
        }
        "integrate" => {
            let host = d
                .delegation
                .ok_or("integrate: unavailable in child execution")?;
            let workspace = arg(0).ok_or("integrate(workspace)")?;
            let workspace = vm_to_json(vm, workspace, d.budget.max_output_bytes);
            let id = workspace
                .get("id")
                .and_then(Value::as_u64)
                .or_else(|| workspace.as_u64())
                .ok_or("integrate: workspace id required")?;
            let revision = d
                .state
                .current_revision
                .lock()
                .expect("revision poisoned")
                .clone();
            let response = host.integrate(
                &d.state.execution,
                &revision,
                crate::identity::IsolatedWorkspaceId(id),
            );
            if let Some(parent_revision) = response.parent_revision.as_ref() {
                *d.state.current_revision.lock().expect("revision poisoned") =
                    parent_revision.clone();
            }
            Ok(json_to_vm(vm, &serde_json::to_value(response).unwrap()))
        }
        "evidence" => {
            let kind = arg(0)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("evidence(kind, ok, metadata?)")?;
            let ok = arg(1)
                .and_then(|a| v.to_json_scalar(a))
                .and_then(|value| value.as_bool())
                .ok_or("evidence(kind, ok, metadata?): ok must be boolean")?;
            let metadata = arg(2)
                .map(|a| vm_to_json(vm, a, d.budget.max_output_bytes))
                .unwrap_or(Value::Null);
            let evidence = evidence_record(d, kind, ok, &metadata);
            emit_event(d.state, PtcEvent::EvidenceRecorded { evidence })
                .map_err(|error| error.to_string())?;
            Ok(v.undefined())
        }
        "tr_read" => {
            let id = handle_id(&v, receiver)?;
            let offset = arg(0).and_then(|a| v.to_i64(a).ok()).unwrap_or(0) as usize;
            let limit = arg(1).and_then(|a| v.to_i64(a).ok()).unwrap_or(200) as usize;
            Ok(json_to_vm(
                vm,
                &fs_tools::handle_read(d.store, id, offset, limit),
            ))
        }
        "tr_head" => {
            let id = handle_id(&v, receiver)?;
            let limit = arg(0).and_then(|a| v.to_i64(a).ok()).unwrap_or(50) as usize;
            Ok(json_to_vm(
                vm,
                &fs_tools::handle_read(d.store, id, 0, limit),
            ))
        }
        "tr_tail" => {
            let id = handle_id(&v, receiver)?;
            let limit = arg(0).and_then(|a| v.to_i64(a).ok()).unwrap_or(50) as usize;
            Ok(json_to_vm(vm, &fs_tools::handle_tail(d.store, id, limit)))
        }
        "tr_grep" => {
            let id = handle_id(&v, receiver)?;
            let pattern = arg(0)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("handle.grep(pattern)")?;
            let limit = arg(1).and_then(|a| v.to_i64(a).ok()).unwrap_or(100) as usize;
            Ok(json_to_vm(
                vm,
                &fs_tools::handle_grep(d.store, id, &pattern, limit),
            ))
        }
        "tr_json" => {
            let id = handle_id(&v, receiver)?;
            Ok(json_to_vm(vm, &fs_tools::handle_json(d.store, id)))
        }
        _ => Err(format!("unknown mh function: {name}")),
    }
}

fn ensure_direct_host_call(output: &Value) -> Result<(), String> {
    let Some(error) = output.get("error").and_then(Value::as_str) else {
        return Ok(());
    };
    if error.contains("budget exceeded") || error.contains("cancelled") {
        return Err(error.to_string());
    }
    Ok(())
}

fn handle_id(v: &super::wrapper::ValueCtx, receiver: JSValue) -> Result<u64, String> {
    v.get(receiver, "id")
        .and_then(|id| v.to_i64(id).ok())
        .and_then(|id| u64::try_from(id).ok())
        .ok_or_else(|| "handle: id".to_string())
}

fn read_args(v: &super::wrapper::ValueCtx, a: Option<JSValue>) -> Result<Value, String> {
    match a {
        Some(a) if v.is_string(a) => {
            Ok(json!({ "path": v.to_string(a).map_err(|e| e.to_string())? }))
        }
        Some(a) => Ok(vm_to_json(v.0, a, usize::MAX)),
        None => Err("read(path) or read({path})".to_string()),
    }
}

fn execute_batch(d: &DispatchBox, name: &str, items: &[Value]) -> Vec<Value> {
    if !matches!(name, "read" | "grep" | "glob" | "exec") {
        return items
            .iter()
            .map(|_| json!({ "error": format!("tool '{name}' is not batch-safe") }))
            .collect();
    }
    if items.is_empty() {
        return Vec::new();
    }
    let concurrency = d.budget.max_parallel_tools.max(1).min(items.len());
    let next = AtomicUsize::new(0);
    let outputs = Mutex::new(vec![Value::Null; items.len()]);
    std::thread::scope(|scope| {
        for _ in 0..concurrency {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= items.len() {
                        break;
                    }
                    let output = execute_child(d, name, items[index].clone(), true);
                    outputs.lock().expect("batch output poisoned")[index] = output;
                }
            });
        }
    });
    outputs.into_inner().expect("batch output poisoned")
}

fn execute_child(d: &DispatchBox, name: &str, args: Value, count_budget: bool) -> Value {
    let call_id = d.state.next_call_id.fetch_add(1, Ordering::Relaxed);
    let args_hash = stable_args_hash(&args);
    let execution = &d.state.execution;
    let sink_failed = emit_event(
        d.state,
        PtcEvent::HostCallStarted {
            task_id: execution.task_id,
            execution_id: execution.execution_id,
            call_id,
            name: name.to_string(),
            args_hash,
        },
    )
    .is_err();
    let started = Instant::now();
    let mut budget_error = sink_failed.then_some("event sink failed");
    if count_budget {
        let calls = d.state.tool_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if calls > d.state.max_tool_calls {
            budget_error = Some("tool call budget exceeded");
        }
    }
    if budget_error.is_none() && name == "exec" {
        let processes = d.state.process_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if processes > d.state.max_processes {
            budget_error = Some("process budget exceeded");
        }
    }
    let tracks_revision = matches!(name, "write" | "edit" | "exec" | "restore");
    let before = tracks_revision
        .then(|| d.tracker.current_revision().ok())
        .flatten();
    let mut restored_revision = None;
    let output = if let Some(error) = budget_error {
        json!({ "error": error })
    } else if d.state.cancelled.load(Ordering::Relaxed) {
        json!({ "error": "cancelled" })
    } else {
        match name {
            "read" => fs_tools::read(d.caps, &args),
            "write" if d.caps.can_write_workspace() => fs_tools::write(d.caps, &args),
            "write" => json!({ "error": "write: disabled by policy" }),
            "edit" if d.caps.can_write_workspace() => fs_tools::edit(d.caps, &args),
            "edit" => json!({ "error": "edit: disabled by policy" }),
            "glob" => fs_tools::glob(d.caps, &args),
            "grep" => fs_tools::grep(d.caps, &args),
            "exec" => fs_tools::exec(d.caps, d.store, &args, d.budget.max_result_bytes, &|| {
                d.state.cancelled.load(Ordering::Relaxed)
            }),
            "checkpoint" if !d.caps.can_write_workspace() => {
                json!({ "error": "checkpoint: disabled by policy" })
            }
            "checkpoint" => match d.checkpoints.create(execution.task_id) {
                Ok(checkpoint) => {
                    d.state
                        .checkpoints
                        .lock()
                        .expect("checkpoint map poisoned")
                        .insert(checkpoint.id.0, checkpoint.clone());
                    json!({ "id": checkpoint.id.0, "revision": checkpoint.revision.0 })
                }
                Err(error) => json!({ "error": error.to_string() }),
            },
            "restore" if !d.caps.can_write_workspace() => {
                json!({ "error": "restore: disabled by policy" })
            }
            "restore" => {
                let id = args.get("id").and_then(Value::as_u64);
                let revision = args.get("revision").and_then(Value::as_str);
                let checkpoint = id.and_then(|id| {
                    d.state
                        .checkpoints
                        .lock()
                        .expect("checkpoint map poisoned")
                        .get(&id)
                        .cloned()
                });
                match checkpoint {
                    Some(checkpoint) if revision == Some(checkpoint.revision.0.as_str()) => {
                        match d.checkpoints.restore(execution.task_id, &checkpoint) {
                            Ok(restored) => {
                                restored_revision = Some(restored.clone());
                                json!({ "id": checkpoint.id.0, "revision": restored.id.0 })
                            }
                            Err(error) => json!({ "error": error.to_string() }),
                        }
                    }
                    Some(_) => json!({ "error": "checkpoint revision does not match handle" }),
                    None => json!({ "error": "unknown checkpoint handle" }),
                }
            }
            _ => json!({ "error": format!("unknown tool: {name}") }),
        }
    };
    let duration_ms = elapsed_ms(started);
    let mut output = output;
    if name == "exec"
        && output.get("error").is_none()
        && let Some(object) = output.as_object_mut()
    {
        object.insert("durationMs".to_string(), json!(duration_ms));
    }
    let ok = output.get("error").is_none();
    let after = if tracks_revision && before.is_some() {
        restored_revision.or_else(|| d.tracker.current_revision().ok())
    } else {
        None
    };
    let mut effects = tool_effects(name);
    let mut paths = Vec::new();
    if let (Some(before), Some(after)) = (before, after)
        && before.id != after.id
    {
        effects.mutates_workspace = true;
        let (added, modified, deleted) = d.tracker.delta(&before.id, &after.id).ok().map_or_else(
            || (Vec::new(), Vec::new(), Vec::new()),
            |delta| {
                (
                    string_paths(delta.added),
                    string_paths(delta.modified),
                    string_paths(delta.deleted),
                )
            },
        );
        paths.extend(added.iter().chain(&modified).chain(&deleted).cloned());
        *d.state.current_revision.lock().expect("revision poisoned") = after.id.clone();
        let source = match name {
            "exec" => RevisionSource::Process,
            "restore" => RevisionSource::Restore,
            _ => RevisionSource::Tool,
        };
        let _ = emit_event(
            d.state,
            PtcEvent::WorkspaceRevisionChanged {
                task_id: execution.task_id,
                execution_id: execution.execution_id,
                from: before.id,
                to: after.id,
                added,
                modified,
                deleted,
                source,
            },
        );
    }
    let result_ids = result_ids(&output);
    let error = output
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let outcome = if ok {
        HostCallOutcome::Ok
    } else if error.contains("cancel") {
        HostCallOutcome::Cancelled
    } else if error.contains("budget exceeded") {
        HostCallOutcome::BudgetExceeded
    } else {
        HostCallOutcome::Error
    };
    let _ = emit_event(
        d.state,
        PtcEvent::HostCallCompleted {
            task_id: execution.task_id,
            execution_id: execution.execution_id,
            call_id,
            name: name.to_string(),
            args_hash,
            effects,
            outcome,
            ok,
            duration_ms,
            result_ids,
            paths,
        },
    );
    output
}

fn tool_effects(name: &str) -> ToolEffects {
    match name {
        "read" | "glob" | "grep" => ToolEffects::READ,
        "write" | "edit" => ToolEffects {
            reads_workspace: true,
            ..ToolEffects::default()
        },
        "exec" => ToolEffects::PROCESS,
        _ => ToolEffects::META,
    }
}

fn string_paths(paths: Vec<std::path::PathBuf>) -> Vec<String> {
    paths
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

fn emit_event(state: &ExecState, event: PtcEvent) -> Result<(), PtcEventSinkError> {
    state
        .events
        .lock()
        .expect("event trace poisoned")
        .push(event.clone());
    if let Some(error) = state
        .sink_error
        .lock()
        .expect("sink error poisoned")
        .clone()
    {
        return Err(error);
    }
    let Some(sink) = state.sink.clone() else {
        return Ok(());
    };
    if let Err(error) = sink.emit(event) {
        *state.sink_error.lock().expect("sink error poisoned") = Some(error.clone());
        return Err(error);
    }
    Ok(())
}

fn stable_args_hash(args: &Value) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in serde_json::to_vec(args).unwrap_or_default() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn result_ids(output: &Value) -> Vec<ResultId> {
    ["stdoutId", "stderrId", "resultId"]
        .into_iter()
        .filter_map(|key| output.get(key).and_then(Value::as_u64).map(ResultId))
        .collect()
}

fn evidence_record(d: &DispatchBox, kind: String, ok: bool, metadata: &Value) -> EvidenceRecord {
    let mut ids = Vec::new();
    collect_result_ids(d.store, metadata, &mut ids);
    ids.sort_by_key(|id| id.0);
    ids.dedup();
    EvidenceRecord {
        kind,
        ok,
        revision: d
            .state
            .current_revision
            .lock()
            .expect("revision poisoned")
            .clone(),
        result_ids: ids,
        note: metadata
            .get("note")
            .and_then(Value::as_str)
            .map(str::to_string),
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0),
        task_id: d.state.execution.task_id,
        execution_id: d.state.execution.execution_id,
    }
}

fn collect_result_ids(store: &ResultStore, value: &Value, ids: &mut Vec<ResultId>) {
    match value {
        Value::Number(number) => {
            if let Some(id) = number.as_u64().map(ResultId)
                && store.metadata(id).is_some()
            {
                ids.push(id);
            }
        }
        Value::Object(map) => map
            .values()
            .for_each(|value| collect_result_ids(store, value, ids)),
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_result_ids(store, value, ids)),
        _ => {}
    }
}

fn collection_to_vm(vm: &Vm, output: &Value, key: &str) -> JSValue {
    if output.get("error").is_some() {
        return json_to_vm(vm, output);
    }
    let values = output
        .get(key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let v = vm.values();
    let array = unsafe { GcRoot::new(vm, v.array(values.len())) };
    for (index, value) in values.iter().enumerate() {
        let child = unsafe { GcRoot::new(vm, json_to_vm(vm, value)) };
        let _ = v.set_index(array.get(), index as u32, child.get());
    }
    let truncated = unsafe {
        GcRoot::new(
            vm,
            v.bool(
                output
                    .get("truncated")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            ),
        )
    };
    let _ = v.set(array.get(), "truncated", truncated.get());
    array.pop()
}

fn tool_value_to_vm(vm: &Vm, d: &DispatchBox, name: &str, output: &Value) -> JSValue {
    match name {
        "glob" => collection_to_vm(vm, output, "files"),
        "grep" => collection_to_vm(vm, output, "matches"),
        "exec" => exec_result_to_vm(vm, d, output),
        _ => json_to_vm(vm, output),
    }
}

/// Builds the JS-visible exec result: scalar fields plus `stdout`/
/// `stderr` host-backed handle objects.
fn exec_result_to_vm(vm: &Vm, d: &DispatchBox, out: &Value) -> JSValue {
    let v = vm.values();
    if out.get("error").is_some() {
        return json_to_vm(vm, out);
    }
    let obj = unsafe { GcRoot::new(vm, v.object()) };
    let exit = unsafe { GcRoot::new(vm, v.i64(out["exitCode"].as_i64().unwrap_or(-1))) };
    let duration = unsafe { GcRoot::new(vm, v.i64(out["durationMs"].as_i64().unwrap_or(0))) };
    let _ = v.set(obj.get(), "exitCode", exit.get());
    let _ = v.set(obj.get(), "durationMs", duration.get());

    let make_handle = |id: u64, key: &str| -> JSValue {
        let h = unsafe { GcRoot::new(vm, v.object()) };
        for method in ["read", "head", "tail", "grep", "json"] {
            let params = unsafe { GcRoot::new(vm, v.string(&format!("tr_{method}"))) };
            let method_fn = unsafe { GcRoot::new(vm, v.closure(params.get())) };
            let _ = v.set(h.get(), method, method_fn.get());
        }
        let metadata = d.store.metadata(ResultId(id));
        let id_value = unsafe { GcRoot::new(vm, v.i64(i64::try_from(id).unwrap_or(i64::MAX))) };
        let kind = unsafe { GcRoot::new(vm, v.string(key)) };
        let length =
            unsafe { GcRoot::new(vm, v.i64(metadata.map(|m| m.length as i64).unwrap_or(0))) };
        let total = unsafe {
            GcRoot::new(
                vm,
                v.i64(metadata.map(|m| m.total_bytes as i64).unwrap_or(0)),
            )
        };
        let truncated = unsafe { GcRoot::new(vm, v.bool(metadata.is_some_and(|m| m.truncated))) };
        let _ = v.set(h.get(), "id", id_value.get());
        let _ = v.set(h.get(), "kind", kind.get());
        let _ = v.set(h.get(), "length", length.get());
        let _ = v.set(h.get(), "totalBytes", total.get());
        let _ = v.set(h.get(), "truncated", truncated.get());
        h.pop()
    };

    let stdout = unsafe {
        GcRoot::new(
            vm,
            make_handle(out["stdoutId"].as_u64().unwrap_or(0), "stdout"),
        )
    };
    let stderr = unsafe {
        GcRoot::new(
            vm,
            make_handle(out["stderrId"].as_u64().unwrap_or(0), "stderr"),
        )
    };
    let _ = v.set(obj.get(), "stdout", stdout.get());
    let _ = v.set(obj.get(), "stderr", stderr.get());
    obj.pop()
}

// ---------------------------------------------------------------------------
// JS <-> JSON conversion
// ---------------------------------------------------------------------------

/// Converts a rooted JS value into JSON, bounded by `max_bytes`.
fn vm_to_json(vm: &Vm, val: JSValue, max_bytes: usize) -> Value {
    let mut budget = max_bytes;
    match conv(vm, val, &mut budget, 0) {
        Some(v) => v,
        None => Value::Null,
    }
}

fn conv(vm: &Vm, val: JSValue, budget: &mut usize, depth: usize) -> Option<Value> {
    const MAX_DEPTH: usize = 24;
    if depth > MAX_DEPTH {
        return None;
    }
    let v = vm.values();
    if let Some(scalar) = v.to_json_scalar(val) {
        *budget = budget.saturating_sub(estimate(&scalar));
        return Some(scalar);
    }
    if v.is_array(val) {
        let len = v
            .get(val, "length")
            .and_then(|n| v.to_i64(n).ok())
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0)
            .min(100_000);
        let mut arr = Vec::with_capacity(len.min(1024));
        for i in 0..len {
            if *budget == 0 {
                arr.push(json!({ "truncated": true }));
                break;
            }
            let item = v.get_index(val, i as u32).unwrap_or(JSValue::UNDEFINED);
            match conv(vm, item, budget, depth + 1) {
                Some(x) => arr.push(x),
                None => arr.push(Value::Null),
            }
        }
        return Some(Value::Array(arr));
    }
    // Plain object: enumerate via JS Object.keys through a fresh eval
    // is overkill; instead use for-in semantics unavailable here —
    // serialize via JSON.stringify closure trick is unavailable too.
    // mquickjs exposes js_object_keys only internally, so use the
    // engine's own JSON.stringify by calling it with JS_Call.
    let json_str = stringify_via_engine(vm, val)?;
    let parsed: Value = serde_json::from_str(&json_str).ok()?;
    Some(parsed)
}

/// Calls the engine's JSON.stringify through the JS evaluator.
fn stringify_via_engine(vm: &Vm, val: JSValue) -> Option<String> {
    let g = unsafe { mh_quickjs_sys::JS_GetGlobalObject(vm.ctx()) };
    let v = vm.values();
    v.set(g, "__mh_tmp", val).ok()?;
    let src = b"JSON.stringify(__mh_tmp)\0";
    let out = unsafe {
        mh_quickjs_sys::JS_Eval(
            vm.ctx(),
            src.as_ptr().cast(),
            src.len() - 1,
            c"mh_internal.js".as_ptr(),
            mh_quickjs_sys::JS_EVAL_RETVAL,
        )
    };
    if out.is_exception() {
        v.set(g, "__mh_tmp", v.undefined()).ok()?;
        return None;
    }
    let s = v.to_string(out).ok()?;
    v.set(g, "__mh_tmp", v.undefined()).ok()?;
    Some(s)
}

fn estimate(v: &Value) -> usize {
    match v {
        Value::String(s) => s.len() + 2,
        Value::Array(a) => 2 + a.len(),
        Value::Object(o) => 2 + o.len() * 8,
        _ => 8,
    }
}

/// Builds a JS value from JSON (bounded by construction).
fn json_to_vm(vm: &Vm, v: &Value) -> JSValue {
    let vals = vm.values();
    match v {
        Value::Null => vals.null(),
        Value::Bool(b) => vals.bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                vals.i64(i)
            } else {
                vals.f64(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::String(s) => vals.string(s),
        Value::Array(items) => {
            let arr = unsafe { GcRoot::new(vm, vals.array(items.len())) };
            for (i, item) in items.iter().enumerate() {
                let child = unsafe { GcRoot::new(vm, json_to_vm(vm, item)) };
                let _ = vals.set_index(arr.get(), i as u32, child.get());
            }
            arr.pop()
        }
        Value::Object(map) => {
            let obj = unsafe { GcRoot::new(vm, vals.object()) };
            for (k, item) in map {
                let child = unsafe { GcRoot::new(vm, json_to_vm(vm, item)) };
                let _ = vals.set(obj.get(), k, child.get());
            }
            obj.pop()
        }
    }
}
