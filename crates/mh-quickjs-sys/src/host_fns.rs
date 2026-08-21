//! Rust-side host functions for the mh engine shim.
//!
//! Only `mh_js_closure` lives here (the single dispatch entry for all
//! mh host functions — `tool`, `read`, `write`, `edit`, `glob`,
//! `grep`, `exec`, `call_tool`, `tools.*`, and `ToolResult` handle
//! methods). The simpler stdlib functions (`js_print`, `js_gc`, ...)
//! are defined in C (`csrc/mh_shim.c`) so the engine archive is
//! self-contained without whole-archive linking.

use std::ffi::c_char;

use crate::{
    JS_CLASS_INTERNAL_ERROR, JS_CLASS_RANGE_ERROR, JS_CLASS_TYPE_ERROR, JS_GetContextOpaque,
    JS_ThrowError, JS_ToCStringLen, JSCStringBuf, JSContext, JSObjectClassEnum, JSValue,
};

/// Handler installed on a context to receive mh closure calls.
///
/// `name` is the closure's params string (dispatch name, e.g. `"tool"`
/// or `"tr_read"`), borrowed for the call duration. `this_val` points
/// at the callee's `this` value. Returns a `JSValue` or
/// `JSValue::EXCEPTION` after throwing.
pub type MhClosureHandler = unsafe extern "C" fn(
    ctx: *mut JSContext,
    name: *const c_char,
    name_len: usize,
    this_val: *mut JSValue,
    argc: i32,
    argv: *mut JSValue,
) -> JSValue;

/// Context-opaque payload: installed with `JS_SetContextOpaque` before
/// any mh closure can run on that context.
#[repr(C)]
pub struct MhDispatch {
    pub handler: MhClosureHandler,
}

/// The single mh dispatch closure registered as `mh_closure` in the
/// stdlib C function table. Forwards to the installed
/// [`MhDispatch::handler`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mh_js_closure(
    ctx: *mut JSContext,
    this_val: *mut JSValue,
    argc: i32,
    argument_values: *mut JSValue,
    params: JSValue,
) -> JSValue {
    let mut len = 0usize;
    let mut sbuf = JSCStringBuf { buf: [0; 5] };
    let name = unsafe { JS_ToCStringLen(ctx, &raw mut len, params, &raw mut sbuf) };
    if name.is_null() {
        return JSValue::EXCEPTION;
    }
    let dispatch = unsafe { JS_GetContextOpaque(ctx) }.cast::<MhDispatch>();
    if dispatch.is_null() {
        return unsafe { throw_internal_error(ctx, "mh dispatch not installed on this context") };
    }
    unsafe { ((*dispatch).handler)(ctx, name, len, this_val, argc, argument_values) }
}

// ---------------------------------------------------------------------------
// Throw helpers (the C macros JS_ThrowTypeError etc. are not callable
// from Rust; JS_ThrowError is variadic).
// ---------------------------------------------------------------------------

/// Throws a `TypeError` with a static message. Returns
/// `JSValue::EXCEPTION` for the return-after-throw pattern.
///
/// # Safety
/// `ctx` must be a live context. `msg` must not contain NUL.
pub unsafe fn throw_type_error(ctx: *mut JSContext, msg: &str) -> JSValue {
    unsafe { throw_error(ctx, JS_CLASS_TYPE_ERROR, msg) }
}

/// Throws an `InternalError`; see [`throw_type_error`].
///
/// # Safety
/// See [`throw_type_error`].
pub unsafe fn throw_internal_error(ctx: *mut JSContext, msg: &str) -> JSValue {
    unsafe { throw_error(ctx, JS_CLASS_INTERNAL_ERROR, msg) }
}

/// Throws a `RangeError`; see [`throw_type_error`].
///
/// # Safety
/// See [`throw_type_error`].
pub unsafe fn throw_range_error(ctx: *mut JSContext, msg: &str) -> JSValue {
    unsafe { throw_error(ctx, JS_CLASS_RANGE_ERROR, msg) }
}

unsafe fn throw_error(ctx: *mut JSContext, class: JSObjectClassEnum, msg: &str) -> JSValue {
    // The engine printf-formats the message; escape any '%' by
    // doubling and append the required NUL terminator.
    let mut formatted = String::with_capacity(msg.len() + 1);
    for c in msg.chars() {
        if c == '%' {
            formatted.push('%');
        }
        formatted.push(c);
    }
    formatted.push('\0');
    let bytes = formatted.as_bytes();
    unsafe { JS_ThrowError(ctx, class, bytes.as_ptr().cast::<std::ffi::c_char>()) }
}
