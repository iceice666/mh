//! Safe wrapper over the MicroQuickJS FFI.
//!
//! One [`Vm`] owns the engine memory buffer and a context. The buffer
//! never moves and the Vm is never reused across threads; `!Send`
//! enforces that. All values produced inside a callback are rooted by
//! the interpreter value stack while the callback runs; the helpers
//! here only touch values that are already rooted (callback arguments)
//! or build fresh values handed straight back to the engine.

use std::ffi::c_void;
use std::fmt;

use mh_quickjs_sys as sys;

pub use sys::{JSValue, MhDispatch};

/// A live MicroQuickJS virtual machine.
pub struct Vm {
    ctx: *mut sys::JSContext,
    /// Backing allocation; must outlive the context and stay at a
    /// stable address (the engine stores raw pointers into it).
    /// `None` for borrowed views (callback dispatch) — those never
    /// free the context.
    _mem: Option<Box<[u8]>>,
}

// The engine is single-threaded and full of raw pointers.
// Vm is !Send/!Sync by construction: it owns raw pointers into its
// heap buffer and the engine is single-threaded. (Negative impls are
// unstable; the raw pointer fields below already poison auto-traits.)

/// Errors surfaced by the wrapper.
#[derive(Debug)]
pub enum VmError {
    /// JS exception with the message of the thrown value.
    Exception(String),
    /// Host memory exhausted (engine could not allocate).
    OutOfMemory,
    /// Context creation failed.
    InitFailed,
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::Exception(msg) => write!(f, "js error: {msg}"),
            VmError::OutOfMemory => write!(f, "js out of memory"),
            VmError::InitFailed => write!(f, "vm init failed"),
        }
    }
}

impl std::error::Error for VmError {}

const DEFAULT_HEAP: usize = 4 << 20; // 4 MiB; PTC programs are small

impl Vm {
    /// Creates a VM with the default heap.
    pub fn new() -> Result<Self, VmError> {
        Self::with_heap(DEFAULT_HEAP)
    }

    /// Creates a VM with a custom heap size in bytes.
    pub fn with_heap(bytes: usize) -> Result<Self, VmError> {
        let mut mem = vec![0u8; bytes].into_boxed_slice();
        let def = unsafe { sys::mh_stdlib_def() };
        let ctx = unsafe {
            sys::JS_NewContext(
                mem.as_mut_ptr().cast::<c_void>(),
                mem.len(),
                def as *const sys::JSSTDLibraryDef,
            )
        };
        if ctx.is_null() {
            return Err(VmError::InitFailed);
        }
        Ok(Self {
            ctx,
            _mem: Some(mem),
        })
    }

    pub(crate) fn ctx(&self) -> *mut sys::JSContext {
        self.ctx
    }

    /// Installs the mh closure dispatcher and interrupt handler.
    /// Opaque for both is the same pointer (the interrupt handler
    /// reads the DispatchBox directly).
    ///
    /// # Safety
    /// `opaque` must outlive the Vm.
    pub(crate) unsafe fn install_opaque(
        &self,
        opaque: *mut std::ffi::c_void,
        interrupt: sys::JSInterruptHandler,
    ) {
        unsafe {
            sys::JS_SetContextOpaque(self.ctx, opaque);
            sys::JS_SetInterruptHandler(self.ctx, interrupt);
        }
    }

    /// Borrows an existing context pointer (callback dispatch use).
    /// The returned Vm does not own the buffer and will not free the
    /// context on drop.
    ///
    /// # Safety
    /// `ctx` must be a live context; the returned Vm must not outlive
    /// it.
    pub(crate) unsafe fn borrow_ctx(ctx: *mut sys::JSContext) -> Vm {
        Vm { ctx, _mem: None }
    }

    /// Evaluates `source` and returns the value of the last top-level
    /// statement (the PTC `return` convention, spec §7).
    pub fn eval(&mut self, source: &str, filename: &str) -> Result<JSValue, VmError> {
        // JS_Eval requires a NUL-terminated buffer.
        let mut buf = Vec::with_capacity(source.len() + 1);
        buf.extend_from_slice(source.as_bytes());
        buf.push(0);
        let mut name = Vec::with_capacity(filename.len() + 1);
        name.extend_from_slice(filename.as_bytes());
        name.push(0);
        let val = unsafe {
            sys::JS_Eval(
                self.ctx,
                buf.as_ptr().cast(),
                buf.len() - 1,
                name.as_ptr().cast(),
                sys::JS_EVAL_RETVAL,
            )
        };
        if val.is_exception() {
            return Err(self.take_exception());
        }
        Ok(val)
    }

    /// Extracts and formats the pending exception into a [`VmError`].
    pub fn take_exception(&self) -> VmError {
        let exc = unsafe { sys::JS_GetException(self.ctx) };
        if exc == JSValue::NULL || exc == JSValue::UNDEFINED {
            return VmError::Exception("unknown exception".into());
        }
        // Common case: Error object with a message property.
        let mut sbuf = sys::JSCStringBuf { buf: [0; 5] };
        let mut len = 0usize;
        let msg = unsafe { sys::JS_GetPropertyStr(self.ctx, exc, c"message".as_ptr()) };
        let text = unsafe {
            let mut base = None;
            let v = msg;
            if !v.is_exception() {
                let p = sys::JS_ToCStringLen(self.ctx, &mut len, v, &mut sbuf);
                if !p.is_null() {
                    base = Some(slice_from_raw(p, len));
                }
            }
            match base {
                Some(s) if !s.is_empty() => String::from_utf8_lossy(s).into_owned(),
                _ => {
                    // Not an Error object or no message: stringify.
                    let p = sys::JS_ToCStringLen(self.ctx, &mut len, exc, &mut sbuf);
                    if p.is_null() {
                        "unknown exception".to_string()
                    } else {
                        String::from_utf8_lossy(slice_from_raw(p, len)).into_owned()
                    }
                }
            }
        };
        VmError::Exception(text)
    }
}

fn slice_from_raw<'a>(p: *const std::ffi::c_char, len: usize) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) }
}

impl Drop for Vm {
    fn drop(&mut self) {
        if self._mem.is_some() {
            unsafe { sys::JS_FreeContext(self.ctx) };
        }
    }
}

// ---------------------------------------------------------------------------
// Value construction/inspection helpers (context-bound).
// ---------------------------------------------------------------------------

/// Context-bound helpers for building and reading JSValues.
pub struct ValueCtx<'a>(pub &'a Vm);

impl Vm {
    pub fn values(&self) -> ValueCtx<'_> {
        ValueCtx(self)
    }
}

impl ValueCtx<'_> {
    pub fn undefined(&self) -> JSValue {
        JSValue::UNDEFINED
    }

    pub fn null(&self) -> JSValue {
        JSValue::NULL
    }

    pub fn bool(&self, b: bool) -> JSValue {
        if b { JSValue::TRUE } else { JSValue::FALSE }
    }

    pub fn int(&self, v: i32) -> JSValue {
        unsafe { sys::JS_NewInt32(self.0.ctx, v) }
    }

    pub fn i64(&self, v: i64) -> JSValue {
        unsafe { sys::JS_NewInt64(self.0.ctx, v) }
    }

    pub fn f64(&self, v: f64) -> JSValue {
        unsafe { sys::JS_NewFloat64(self.0.ctx, v) }
    }

    pub fn string(&self, s: &str) -> JSValue {
        unsafe { sys::JS_NewStringLen(self.0.ctx, s.as_ptr().cast(), s.len()) }
    }

    pub fn object(&self) -> JSValue {
        unsafe { sys::JS_NewObject(self.0.ctx) }
    }

    pub fn array(&self, len: usize) -> JSValue {
        unsafe { sys::JS_NewArray(self.0.ctx, i32::try_from(len).unwrap_or(i32::MAX)) }
    }

    pub fn closure(&self, params: JSValue) -> JSValue {
        unsafe { sys::JS_NewCFunctionParams(self.0.ctx, sys::JS_CFUNCTION_MH_CLOSURE, params) }
    }

    /// Reads a property; returns `JSValue::UNDEFINED` for absent
    /// properties (engine semantics for missing props).
    pub fn get(&self, obj: JSValue, key: &str) -> Option<JSValue> {
        let keyz = CStringBuf::from(key);
        let v = unsafe { sys::JS_GetPropertyStr(self.0.ctx, obj, keyz.as_ptr()) };
        if v.is_exception() { None } else { Some(v) }
    }

    pub fn get_index(&self, obj: JSValue, idx: u32) -> Option<JSValue> {
        let v = unsafe { sys::JS_GetPropertyUint32(self.0.ctx, obj, idx) };
        if v.is_exception() { None } else { Some(v) }
    }

    /// Sets a property; the engine reports failures (frozen objects,
    /// OOM) via the exception state, surfaced as `Err`.
    pub fn set(&self, obj: JSValue, key: &str, val: JSValue) -> Result<(), VmError> {
        let keyz = CStringBuf::from(key);
        let r = unsafe { sys::JS_SetPropertyStr(self.0.ctx, obj, keyz.as_ptr(), val) };
        if r.is_exception() {
            Err(self.0.take_exception())
        } else {
            Ok(())
        }
    }

    pub fn set_index(&self, obj: JSValue, idx: u32, val: JSValue) -> Result<(), VmError> {
        let r = unsafe { sys::JS_SetPropertyUint32(self.0.ctx, obj, idx, val) };
        if r.is_exception() {
            Err(self.0.take_exception())
        } else {
            Ok(())
        }
    }

    pub fn is_array(&self, v: JSValue) -> bool {
        unsafe { sys::JS_IsArray(self.0.ctx, v) != 0 }
    }

    pub fn is_string(&self, v: JSValue) -> bool {
        unsafe { sys::JS_IsString(self.0.ctx, v) != 0 }
    }

    pub fn to_string(&self, v: JSValue) -> Result<String, VmError> {
        let mut sbuf = sys::JSCStringBuf { buf: [0; 5] };
        let mut len = 0usize;
        let p = unsafe { sys::JS_ToCStringLen(self.0.ctx, &mut len, v, &mut sbuf) };
        if p.is_null() {
            Err(self.0.take_exception())
        } else {
            Ok(String::from_utf8_lossy(slice_from_raw(p, len)).into_owned())
        }
    }

    pub fn to_i64(&self, v: JSValue) -> Result<i64, VmError> {
        let mut d = 0f64;
        if unsafe { sys::JS_ToNumber(self.0.ctx, &mut d, v) } != 0 {
            return Err(self.0.take_exception());
        }
        if !d.is_finite() {
            d = 0.0;
        }
        Ok(d as i64)
    }

    /// Reads a JS string/number/bool into a JSON value. Objects and
    /// arrays are converted structurally by the PTC runtime, not here.
    pub fn to_json_scalar(&self, v: JSValue) -> Option<serde_json::Value> {
        if v == JSValue::UNDEFINED {
            return Some(serde_json::Value::Null);
        }
        if v == JSValue::NULL {
            return Some(serde_json::Value::Null);
        }
        if v == JSValue::TRUE {
            return Some(serde_json::Value::Bool(true));
        }
        if v == JSValue::FALSE {
            return Some(serde_json::Value::Bool(false));
        }
        if self.is_string(v) {
            return self.to_string(v).ok().map(serde_json::Value::String);
        }
        // Numbers.
        let mut d = 0f64;
        if unsafe { sys::JS_ToNumber(self.0.ctx, &mut d, v) } == 0 {
            if d.fract() == 0.0 && d.abs() < 9.007_199_254_740_992e15 {
                return Some(serde_json::Value::Number((d as i64).into()));
            }
            if let Some(n) = serde_json::Number::from_f64(d) {
                return Some(serde_json::Value::Number(n));
            }
        }
        None
    }
}

/// NUL-terminated key buffer for property access.
struct CStringBuf {
    buf: Vec<u8>,
}

impl CStringBuf {
    fn from(s: &str) -> Self {
        let mut buf = Vec::with_capacity(s.len() + 1);
        buf.extend_from_slice(s.as_bytes());
        buf.push(0);
        Self { buf }
    }

    fn as_ptr(&self) -> *const std::ffi::c_char {
        self.buf.as_ptr().cast()
    }
}
