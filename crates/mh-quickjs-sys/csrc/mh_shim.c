/* mh engine shim.
 *
 * Compiled into libmhquickjs.a by build.rs. It includes the generated
 * stdlib table (mh_stdlib.h) and exposes the JSSTDLibraryDef to Rust.
 *
 * Simple host functions referenced by the generated table are defined
 * here in C (js_print, js_gc, js_load, js_setTimeout,
 * js_clearTimeout, js_date_now, js_performance_now,
 * js_date_constructor). mh_js_closure — the single dispatch entry for
 * all mh host functions — is defined in Rust (src/host_fns.rs).
 */
#include <stddef.h>
#include <stdio.h>
#include <sys/time.h>

#include "mquickjs.h"
#include "mquickjs_priv.h"

/* provided by Rust */
JSValue mh_js_closure(JSContext *ctx, JSValue *this_val, int argc, JSValue *argv,
                      JSValue params);

static int64_t mh_now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

/* console.log / print: strings and dumped values go to stderr. */
JSValue js_print(JSContext *ctx, JSValue *this_val, int argc, JSValue *argv)
{
    int i;
    for (i = 0; i < argc; i++) {
        if (i != 0)
            fputc(' ', stderr);
        if (JS_IsString(ctx, argv[i])) {
            JSCStringBuf buf;
            const char *str;
            size_t len;
            str = JS_ToCStringLen(ctx, &len, argv[i], &buf);
            if (str)
                fwrite(str, 1, len, stderr);
        } else {
            JS_PrintValueF(ctx, argv[i], 0);
        }
    }
    fputc('\n', stderr);
    return JS_UNDEFINED;
}

JSValue js_gc(JSContext *ctx, JSValue *this_val, int argc, JSValue *argv)
{
    JS_GC(ctx);
    return JS_UNDEFINED;
}

JSValue js_load(JSContext *ctx, JSValue *this_val, int argc, JSValue *argv)
{
    return JS_ThrowTypeError(ctx, "load() is not available; use the read tool");
}

JSValue js_setTimeout(JSContext *ctx, JSValue *this_val, int argc, JSValue *argv)
{
    return JS_ThrowTypeError(ctx, "setTimeout() is not supported in PTC programs");
}

JSValue js_clearTimeout(JSContext *ctx, JSValue *this_val, int argc, JSValue *argv)
{
    return JS_ThrowTypeError(ctx, "clearTimeout() is not supported in PTC programs");
}

JSValue js_date_now(JSContext *ctx, JSValue *this_val, int argc, JSValue *argv)
{
    return JS_NewInt64(ctx, mh_now_ms());
}

JSValue js_performance_now(JSContext *ctx, JSValue *this_val, int argc, JSValue *argv)
{
    return JS_NewInt64(ctx, mh_now_ms());
}

/* Date constructor (upstream defines it in mqjs.c; reimplemented here
 * so the engine library is self-contained). */
JSValue js_date_constructor(JSContext *ctx, JSValue *this_val, int argc,
                            JSValue *argv)
{
    double val;
    argc &= ~FRAME_CF_CTOR;
    if (argc == 0) {
        val = (double)mh_now_ms();
    } else if (argc == 1 && JS_IsNumber(ctx, argv[0])) {
        if (JS_ToNumber(ctx, &val, argv[0]))
            return JS_EXCEPTION;
    } else {
        return JS_ThrowTypeError(ctx, "unsupported Date() parameter");
    }
    return JS_NewDate(ctx, val);
}

#include "mh_stdlib.h"

const JSSTDLibraryDef *mh_stdlib_def(void)
{
    return &js_stdlib;
}
