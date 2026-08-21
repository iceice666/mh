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

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::tools::capability::Capabilities;
use crate::tools::fs_tools;
use crate::tools::store::ResultStore;

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
}

impl Default for PtcBudget {
    fn default() -> Self {
        Self {
            wall_time_ms: 120_000,
            instruction_limit: Some(100_000_000),
            max_tool_calls: 200,
            max_processes: 16,
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

/// Result of one PTC execution.
#[derive(Debug, Clone)]
pub struct PtcResult {
    /// JSON view of the program's returned value (`undefined` → null).
    pub value: Value,
    pub outcome: PtcOutcome,
    pub tool_calls: usize,
    pub duration_ms: u64,
}

struct ExecState {
    cancelled: Arc<AtomicBool>,
    deadline: Instant,
    tool_calls: Arc<AtomicU64>,
    process_calls: AtomicU64,
    interrupt_polls: AtomicU64,
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
}

impl PtcRuntime {
    pub fn new(caps: Capabilities, store: ResultStore, budget: PtcBudget) -> Self {
        Self {
            caps,
            store,
            budget,
        }
    }

    /// Executes `program` and converts its return value to JSON.
    ///
    /// Cancellation is polled by the VM interrupt handler and tools.
    pub fn execute(&self, program: &str, cancelled: &Arc<AtomicBool>) -> PtcResult {
        let start = Instant::now();
        let state = Box::new(ExecState {
            cancelled: cancelled.clone(),
            deadline: start + Duration::from_millis(self.budget.wall_time_ms),
            tool_calls: Arc::new(AtomicU64::new(0)),
            process_calls: AtomicU64::new(0),
            interrupt_polls: AtomicU64::new(0),
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
                unsafe { drop(Box::from_raw(state_ptr)) };
                return PtcResult {
                    value: Value::Null,
                    outcome: PtcOutcome::Failed(e.to_string()),
                    tool_calls: 0,
                    duration_ms: 0,
                };
            }
        };

        // Both the interrupt handler and the mh closure dispatch read
        // the context-opaque DispatchBox (handler field first).
        let dispatch = Box::new(DispatchBox {
            handler: dispatch_closure,
            state: unsafe { &*state_ptr },
            caps: &self.caps,
            store: &self.store,
            budget: &self.budget,
        });
        let dispatch_ptr = Box::into_raw(dispatch);
        unsafe {
            vm.install_opaque(dispatch_ptr.cast::<c_void>(), interrupt_handler);
        }

        let result = (|| {
            // Install ABI globals.
            if let Err(e) = self.install_globals(&mut vm) {
                return PtcResult {
                    value: Value::Null,
                    outcome: PtcOutcome::Failed(e.to_string()),
                    tool_calls: 0,
                    duration_ms: elapsed_ms(start),
                };
            }
            // Run.
            let wrapped = format!("(function() {{\n{program}\n}})()");
            let val = match vm.eval(&wrapped, "ptc.js") {
                Ok(v) => v,
                Err(VmError::Exception(msg)) => {
                    let outcome = if cancelled.load(Ordering::Relaxed) {
                        Some(PtcOutcome::Interrupted)
                    } else if Instant::now() >= state_deadline(dispatch_ptr) {
                        Some(PtcOutcome::BudgetExceeded("wall time"))
                    } else if instruction_budget_exceeded(dispatch_ptr) {
                        Some(PtcOutcome::BudgetExceeded("instruction limit"))
                    } else {
                        None
                    };
                    if let Some(outcome) = outcome {
                        return PtcResult {
                            value: Value::Null,
                            outcome,
                            tool_calls: tool_calls(dispatch_ptr),
                            duration_ms: elapsed_ms(start),
                        };
                    }
                    return PtcResult {
                        value: json!({ "ptcError": msg }),
                        outcome: PtcOutcome::Failed(msg),
                        tool_calls: tool_calls(dispatch_ptr),
                        duration_ms: elapsed_ms(start),
                    };
                }
                Err(e) => {
                    return PtcResult {
                        value: Value::Null,
                        outcome: PtcOutcome::Failed(e.to_string()),
                        tool_calls: tool_calls(dispatch_ptr),
                        duration_ms: elapsed_ms(start),
                    };
                }
            };
            // Convert result to JSON (bounded).
            let js = vm_to_json(&vm, val, self.budget.max_output_bytes);
            PtcResult {
                value: js,
                outcome: PtcOutcome::Completed,
                tool_calls: tool_calls(dispatch_ptr),
                duration_ms: elapsed_ms(start),
            }
        })();

        unsafe {
            drop(Box::from_raw(dispatch_ptr));
            drop(Box::from_raw(state_ptr));
        }
        result
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
        ] {
            let f = unsafe { GcRoot::new(vm, make_fn(vm, name)) };
            v.set(g, name, f.get())?;
        }
        // tools.read(...) alias family (spec §18.1).
        let tools = unsafe { GcRoot::new(vm, v.object()) };
        for name in ["read", "write", "edit", "glob", "grep", "exec", "call_tool"] {
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
}

fn tool_calls(p: *mut DispatchBox) -> usize {
    unsafe { (*p).state.tool_calls.load(Ordering::Relaxed) as usize }
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

/// Executes one host call; all argument values are rooted by the
unsafe fn run_host_call(
    vm: &Vm,
    dispatch: *mut DispatchBox,
    name: &str,
    receiver: JSValue,
    argc: i32,
    argv: *mut JSValue,
) -> Result<JSValue, String> {
    let d = unsafe { &*dispatch };
    // Tool-call accounting & budget.
    if matches!(
        name,
        "tool" | "call_tool" | "read" | "write" | "edit" | "glob" | "grep" | "exec"
    ) {
        let n = d.state.tool_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if n > d.state.max_tool_calls {
            return Err("tool call budget exceeded".to_string());
        }
        if d.state.cancelled.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }
    }

    let arg = |i: usize| -> Option<JSValue> {
        if (i as i32) < argc {
            Some(unsafe { *argv.add(i) })
        } else {
            None
        }
    };

    let v = vm.values();

    match name {
        // canonical ABI ---------------------------------------------------
        "tool" | "call_tool" => {
            let tool_name = arg(0)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("tool(name, args): name required")?;
            let args_json = match arg(1) {
                Some(a) => vm_to_json(vm, a, d.budget.max_output_bytes),
                None => Value::Null,
            };
            if tool_name == "exec" {
                let processes = d.state.process_calls.fetch_add(1, Ordering::Relaxed) + 1;
                if processes > d.state.max_processes {
                    return Err("process budget exceeded".to_string());
                }
                let out = fs_tools::exec(
                    d.caps,
                    d.store,
                    &args_json,
                    d.budget.max_result_bytes,
                    &|| d.state.cancelled.load(Ordering::Relaxed),
                );
                Ok(exec_result_to_vm(vm, d, &out))
            } else {
                let out = dispatch_tool(d, &tool_name, &args_json);
                Ok(json_to_vm(vm, &out))
            }
        }
        // convenience globals (spec §8.1) --------------------------------
        "read" => {
            let args = read_args(&v, arg(0))?;
            let out = fs_tools::read(d.caps, &args);
            Ok(json_to_vm(vm, &out))
        }
        "write" => {
            let path = arg(0)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("write(path, content)")?;
            let content = arg(1)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("write(path, content)")?;
            let out = fs_tools::write(d.caps, &json!({ "path": path, "content": content }));
            Ok(json_to_vm(vm, &out))
        }
        "edit" => {
            let args = arg(0).ok_or("edit(argsObject)")?;
            let j = vm_to_json(vm, args, d.budget.max_output_bytes);
            let out = fs_tools::edit(d.caps, &j);
            Ok(json_to_vm(vm, &out))
        }
        "glob" => {
            let pattern = arg(0)
                .and_then(|a| v.to_string(a).ok())
                .ok_or("glob(pattern)")?;
            let out = fs_tools::glob(d.caps, &json!({ "pattern": pattern }));
            let files = out.get("files").cloned().unwrap_or(out);
            Ok(json_to_vm(vm, &files))
        }
        "grep" => {
            let args = arg(0).ok_or("grep(argsObject)")?;
            let j = vm_to_json(vm, args, d.budget.max_output_bytes);
            let out = fs_tools::grep(d.caps, &j);
            let matches = out.get("matches").cloned().unwrap_or(out);
            Ok(json_to_vm(vm, &matches))
        }
        "exec" => {
            let processes = d.state.process_calls.fetch_add(1, Ordering::Relaxed) + 1;
            if processes > d.state.max_processes {
                return Err("process budget exceeded".to_string());
            }
            let args = arg(0).ok_or("exec(argsObject)")?;
            let j = vm_to_json(vm, args, d.budget.max_output_bytes);
            let out = fs_tools::exec(d.caps, d.store, &j, d.budget.max_result_bytes, &|| {
                d.state.cancelled.load(Ordering::Relaxed)
            });
            Ok(exec_result_to_vm(vm, d, &out))
        }
        // ToolResult handle methods --------------------------------------
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
        Some(a) => {
            // argument object form
            let vm = v.0;
            let j = vm_to_json(vm, a, usize::MAX);
            Ok(j)
        }
        None => Err("read(path) or read({path})".to_string()),
    }
}

/// Routes `tool(name, args)` to the implementing function; native FC
/// lowering reuses this exact path (spec invariant V).
fn dispatch_tool(d: &DispatchBox, name: &str, args: &Value) -> Value {
    let no_cancel = || d.state.cancelled.load(Ordering::Relaxed);
    match name {
        "read" => fs_tools::read(d.caps, args),
        "write" => fs_tools::write(d.caps, args),
        "edit" => fs_tools::edit(d.caps, args),
        "glob" => {
            let output = fs_tools::glob(d.caps, args);
            output.get("files").cloned().unwrap_or(output)
        }
        "grep" => {
            let output = fs_tools::grep(d.caps, args);
            output.get("matches").cloned().unwrap_or(output)
        }
        "exec" => {
            let processes = d.state.process_calls.fetch_add(1, Ordering::Relaxed) + 1;
            if processes > d.state.max_processes {
                return json!({ "error": "process budget exceeded" });
            }
            fs_tools::exec(d.caps, d.store, args, d.budget.max_result_bytes, &no_cancel)
        }
        _ => json!({ "error": format!("unknown tool: {name}") }),
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
    let _ = v.set(obj.get(), "exitCode", exit.get());

    let make_handle = |id: u64, key: &str| -> JSValue {
        let h = unsafe { GcRoot::new(vm, v.object()) };
        for method in ["read", "head", "tail", "grep", "json"] {
            let dispatch_name = format!("tr_{method}");
            let params = unsafe { GcRoot::new(vm, v.string(&dispatch_name)) };
            let method_fn = unsafe { GcRoot::new(vm, v.closure(params.get())) };
            let _ = v.set(h.get(), method, method_fn.get());
        }
        let id_value = unsafe { GcRoot::new(vm, v.i64(i64::try_from(id).unwrap_or(i64::MAX))) };
        let kind = unsafe { GcRoot::new(vm, v.string(key)) };
        let length_value = fs_tools::handle_length(d.store, id)
            .and_then(|length| i64::try_from(length).ok())
            .unwrap_or(0);
        let length = unsafe { GcRoot::new(vm, v.i64(length_value)) };
        let _ = v.set(h.get(), "id", id_value.get());
        let _ = v.set(h.get(), "_kind", kind.get());
        let _ = v.set(h.get(), "length", length.get());
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
