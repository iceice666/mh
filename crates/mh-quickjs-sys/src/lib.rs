//! Raw FFI bindings to the vendored `MicroQuickJS` engine.
//!
//! This crate exposes the C API exactly as declared in
//! `vendor/mquickjs/mquickjs.h` plus the mh shim entry
//! (`mh_stdlib_def`). The safe wrapper lives in the `mh-ptc` crate.
//!
//! `JSValue` is a tagged word: either an immediate (int, bool, null,
//! undefined, exception, short func, string char) or a pointer into the
//! engine heap. All pointer values point into the buffer that was
//! passed to `JS_NewContext`; there is no per-value free function —
//! storage is reclaimed by the compacting GC.
//!
//! # Caller contract (garbage collection)
//!
//! The engine GC scans the value stack between `ctx.sp` and
//! `ctx.stack_top` for roots. While executing inside a C function
//! called from JS (or inside `JS_Call`), values below the current frame
//! are therefore rooted; a freshly built value that must survive an
//! allocation has to be kept on the value stack (or in a `JSGCRef`
//! root) and read back afterwards, because the GC may compact the heap
//! and move it.

#![allow(
    non_camel_case_types,
    clippy::missing_safety_doc,
    clippy::undocumented_unsafe_blocks
)]

mod host_fns;

pub use host_fns::*;

use std::ffi::c_char;
use std::ffi::c_void;

/// Opaque engine context.
#[repr(C)]
pub struct JSContext {
    _private: [u8; 0],
}

/// Tagged JavaScript value word. Never construct one directly — use
/// the constructor functions or the documented immediates.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct JSValue(pub u64);

pub type JS_BOOL = i32;

/// `JS_INTERRUPT_COUNTER_INIT` — the interpreter polls the interrupt
/// handler roughly every this many bytecode dispatches.
pub const JS_INTERRUPT_COUNTER_INIT: u64 = 10_000;

impl JSValue {
    /// `JS_EXCEPTION` sentinel — a C function must return this after
    /// throwing, and callers must test it after every engine call.
    /// `JS_VALUE_MAKE_SPECIAL(JS_TAG_EXCEPTION, 0)` = 15.
    pub const EXCEPTION: JSValue = JSValue(15);
    /// `JS_NULL` = `JS_VALUE_MAKE_SPECIAL(JS_TAG_NULL, 0)` = 7.
    pub const NULL: JSValue = JSValue(7);
    /// `JS_UNDEFINED` = 11.
    pub const UNDEFINED: JSValue = JSValue(11);
    /// `JS_TRUE` = 35.
    pub const TRUE: JSValue = JSValue(35);
    /// `JS_FALSE` = 3.
    pub const FALSE: JSValue = JSValue(3);

    #[must_use]
    pub fn is_exception(self) -> bool {
        self == Self::EXCEPTION
    }
}

// ---------------------------------------------------------------------------
// Values and immediates
// ---------------------------------------------------------------------------

unsafe extern "C" {
    pub fn JS_NewFloat64(ctx: *mut JSContext, d: f64) -> JSValue;
    pub fn JS_NewInt32(ctx: *mut JSContext, val: i32) -> JSValue;
    pub fn JS_NewUint32(ctx: *mut JSContext, val: u32) -> JSValue;
    pub fn JS_NewInt64(ctx: *mut JSContext, val: i64) -> JSValue;

    pub fn JS_IsNumber(ctx: *mut JSContext, val: JSValue) -> JS_BOOL;
    pub fn JS_IsString(ctx: *mut JSContext, val: JSValue) -> JS_BOOL;
    pub fn JS_IsError(ctx: *mut JSContext, val: JSValue) -> JS_BOOL;
    pub fn JS_IsFunction(ctx: *mut JSContext, val: JSValue) -> JS_BOOL;
    pub fn JS_IsArray(ctx: *mut JSContext, obj: JSValue) -> JS_BOOL;
    pub fn JS_HasException(ctx: *mut JSContext) -> JS_BOOL;
    pub fn JS_GetException(ctx: *mut JSContext) -> JSValue;

    pub fn JS_GetClassID(ctx: *mut JSContext, val: JSValue) -> i32;
    pub fn JS_SetOpaque(ctx: *mut JSContext, val: JSValue, opaque: *mut c_void);
    pub fn JS_GetOpaque(ctx: *mut JSContext, val: JSValue) -> *mut c_void;
}

// ---------------------------------------------------------------------------
// Static stdlib definition (generated at build time by build.rs)
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct JSSTDLibraryDef {
    pub stdlib_table: *const u64,
    pub c_function_table: *const JSCFunctionDef,
    pub c_finalizer_table: *const JSCFinalizer,
    pub stdlib_table_len: u32,
    pub stdlib_table_align: u32,
    pub sorted_atoms_offset: u32,
    pub global_object_offset: u32,
    pub class_count: u32,
}

/// C function entry in the generated stdlib table.
#[repr(C)]
pub struct JSCFunctionDef {
    pub func: JSCFunctionType,
    pub def_type: u8,
    pub arg_count: u8,
    pub magic: i16,
}

/// Plain C function signature (`JSCFunction` in C).
pub type JSCFunction = unsafe extern "C" fn(
    ctx: *mut JSContext,
    this_val: *mut JSValue,
    argc: i32,
    argv: *mut JSValue,
) -> JSValue;

/// Finalizer for user classes; receives the opaque pointer.
pub type JSCFinalizer = Option<unsafe extern "C" fn(ctx: *mut JSContext, opaque: *mut c_void)>;

/// C function signature variants (`JSCFunctionType` in C).
#[repr(C)]
#[derive(Copy, Clone)]
pub union JSCFunctionType {
    pub generic: JSCFunction,
    pub generic_magic: unsafe extern "C" fn(
        ctx: *mut JSContext,
        this_val: *mut JSValue,
        argc: i32,
        argv: *mut JSValue,
        magic: i32,
    ) -> JSValue,
    pub constructor: JSCFunction,
    pub constructor_magic: unsafe extern "C" fn(
        ctx: *mut JSContext,
        this_val: *mut JSValue,
        argc: i32,
        argv: *mut JSValue,
        magic: i32,
    ) -> JSValue,
    pub generic_params: unsafe extern "C" fn(
        ctx: *mut JSContext,
        this_val: *mut JSValue,
        argc: i32,
        argv: *mut JSValue,
        params: JSValue,
    ) -> JSValue,
}
/// Returns the build-time-generated static stdlib definition.
///
/// # Safety
/// The returned reference is valid for the program lifetime and immutable.
#[must_use]
pub unsafe fn mh_stdlib_def() -> &'static JSSTDLibraryDef {
    unsafe { &*ffi::mh_stdlib_def() }
}

mod ffi {
    use super::JSSTDLibraryDef;
    unsafe extern "C" {
        pub fn mh_stdlib_def() -> *const JSSTDLibraryDef;
    }
}

// ---------------------------------------------------------------------------
// Context lifecycle
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Allocates a context inside `mem_start`/`mem_size`. The buffer
    /// must stay alive and unmoved until `JS_FreeContext`.
    pub fn JS_NewContext(
        mem_start: *mut c_void,
        mem_size: usize,
        stdlib_def: *const JSSTDLibraryDef,
    ) -> *mut JSContext;
    /// Calls finalizers for user-class objects; does not free the buffer.
    pub fn JS_FreeContext(ctx: *mut JSContext);
    pub fn JS_SetContextOpaque(ctx: *mut JSContext, opaque: *mut c_void);
    pub fn JS_GetContextOpaque(ctx: *mut JSContext) -> *mut c_void;
    pub fn JS_SetInterruptHandler(ctx: *mut JSContext, handler: JSInterruptHandler);
    pub fn JS_SetRandomSeed(ctx: *mut JSContext, seed: u64);
    pub fn JS_GetGlobalObject(ctx: *mut JSContext) -> JSValue;
    pub fn JS_GC(ctx: *mut JSContext);
}

/// Interrupt handler; return nonzero to abort execution with an
/// uncatchable `InternalError`.
pub type JSInterruptHandler = unsafe extern "C" fn(ctx: *mut JSContext, opaque: *mut c_void) -> i32;

// ---------------------------------------------------------------------------
// GC roots
// ---------------------------------------------------------------------------

/// Stack-list GC root. Push before a value may move, pop after.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct JSGCRef {
    pub val: JSValue,
    pub prev: *mut JSGCRef,
}

unsafe extern "C" {
    /// Roots the value in `ref_` (stack discipline); returns a pointer
    /// to the rooted slot, which the GC updates on compaction.
    pub fn JS_PushGCRef(ctx: *mut JSContext, ref_: *mut JSGCRef) -> *mut JSValue;
    /// Pops the root; returns the (possibly moved) value.
    pub fn JS_PopGCRef(ctx: *mut JSContext, ref_: *mut JSGCRef) -> JSValue;
}

// ---------------------------------------------------------------------------
// Properties and objects
// ---------------------------------------------------------------------------

unsafe extern "C" {
    pub fn JS_GetPropertyStr(
        ctx: *mut JSContext,
        this_obj: JSValue,
        str_: *const c_char,
    ) -> JSValue;
    pub fn JS_GetPropertyUint32(ctx: *mut JSContext, obj: JSValue, idx: u32) -> JSValue;
    pub fn JS_SetPropertyStr(
        ctx: *mut JSContext,
        this_obj: JSValue,
        str_: *const c_char,
        val: JSValue,
    ) -> JSValue;
    pub fn JS_SetPropertyUint32(
        ctx: *mut JSContext,
        this_obj: JSValue,
        idx: u32,
        val: JSValue,
    ) -> JSValue;
    pub fn JS_NewObject(ctx: *mut JSContext) -> JSValue;
    pub fn JS_NewArray(ctx: *mut JSContext, initial_len: i32) -> JSValue;
    /// Creates a C closure. `func_idx` indexes the generated
    /// `js_c_function_table`; the mh closure is
    /// [`JS_CFUNCTION_MH_CLOSURE`]. `params` is stored in the closure
    /// object and handed back on every call; it is GC-managed.
    pub fn JS_NewCFunctionParams(ctx: *mut JSContext, func_idx: i32, params: JSValue) -> JSValue;
}

/// C function table index of the mh closure entry: `js_c_function_decl[]`
/// is `[bound, mh_closure]` under `CONFIG_MH`, and user functions start
/// at `JS_CFUNCTION_USER` (= 1).
pub const JS_CFUNCTION_MH_CLOSURE: i32 = 1;

// ---------------------------------------------------------------------------
// Strings and conversions
// ---------------------------------------------------------------------------

unsafe extern "C" {
    pub fn JS_NewStringLen(ctx: *mut JSContext, buf: *const c_char, buf_len: usize) -> JSValue;
    /// Converts `val` to a string. The returned pointer is borrowed
    /// from the engine heap (or `buf` for short strings) and is only
    /// valid until the next GC-triggering call; copy it if needed.
    pub fn JS_ToCStringLen(
        ctx: *mut JSContext,
        plen: *mut usize,
        val: JSValue,
        buf: *mut JSCStringBuf,
    ) -> *const c_char;
    pub fn JS_ToString(ctx: *mut JSContext, val: JSValue) -> JSValue;
    pub fn JS_ToInt32(ctx: *mut JSContext, pres: *mut i32, val: JSValue) -> i32;
    pub fn JS_ToInt32Sat(ctx: *mut JSContext, pres: *mut i32, val: JSValue) -> i32;
    pub fn JS_ToNumber(ctx: *mut JSContext, pres: *mut f64, val: JSValue) -> i32;
}

/// Small stack buffer backing short-string conversion results.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct JSCStringBuf {
    pub buf: [u8; 5],
}

// ---------------------------------------------------------------------------
// Exceptions
// ---------------------------------------------------------------------------

/// Object class ids for `JS_ThrowError` (`JSObjectClassEnum` in C).
pub type JSObjectClassEnum = u32;
pub const JS_CLASS_RANGE_ERROR: JSObjectClassEnum = 11;
pub const JS_CLASS_TYPE_ERROR: JSObjectClassEnum = 14;
pub const JS_CLASS_INTERNAL_ERROR: JSObjectClassEnum = 16;

unsafe extern "C" {
    pub fn JS_Throw(ctx: *mut JSContext, obj: JSValue) -> JSValue;
    pub fn JS_ThrowError(
        ctx: *mut JSContext,
        error_num: JSObjectClassEnum,
        fmt: *const c_char,
        ...
    ) -> JSValue;
    pub fn JS_ThrowOutOfMemory(ctx: *mut JSContext) -> JSValue;
}

// ---------------------------------------------------------------------------
// Evaluation and calls
// ---------------------------------------------------------------------------

pub const JS_EVAL_RETVAL: i32 = 1 << 0;

unsafe extern "C" {
    /// Parses and runs `input` (which must be NUL-terminated). With
    /// `JS_EVAL_RETVAL`, the value of the last top-level
    /// expression/statement is returned.
    pub fn JS_Eval(
        ctx: *mut JSContext,
        input: *const c_char,
        input_len: usize,
        filename: *const c_char,
        eval_flags: i32,
    ) -> JSValue;
    pub fn JS_Run(ctx: *mut JSContext, val: JSValue) -> JSValue;
    /// Returns nonzero if fewer than `len` stack words remain.
    pub fn JS_StackCheck(ctx: *mut JSContext, len: u32) -> i32;
    pub fn JS_PushArg(ctx: *mut JSContext, val: JSValue);
    /// Calls a function. Args and callee must already be pushed with
    /// `JS_PushArg` in reverse order (arg[n-1] ... arg[0], func, this).
    pub fn JS_Call(ctx: *mut JSContext, call_flags: i32) -> JSValue;
}

// ---------------------------------------------------------------------------
// Debug output (routed through JS_SetLogFunc)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    pub fn JS_SetLogFunc(ctx: *mut JSContext, write_func: JSWriteFunc);
    pub fn JS_PrintValueF(ctx: *mut JSContext, val: JSValue, flags: i32);
}

pub const JS_DUMP_LONG: i32 = 1 << 0;

/// Output callback used by `JS_PrintValueF`.
pub type JSWriteFunc =
    unsafe extern "C" fn(opaque: *mut c_void, buf: *const c_void, buf_len: usize);
