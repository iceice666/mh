# Vendored MicroQuickJS

Upstream: https://github.com/bellard/mquickjs
Pinned revision: `203d5bb79789bc47b74855d9207415dab71661a0` (see `flake.lock`)

Local modifications (keep minimal, documented here):

1. `mqjs_stdlib.c` — added a `CONFIG_MH` block in `js_c_function_decl[]`
   registering one extra closure entry:

   ```c
   #ifdef CONFIG_MH
   JS_CFUNC_SPECIAL_DEF("mh_closure", 2, generic_params, mh_js_closure ),
   #endif
   ```

   `mh_js_closure` is defined in Rust (`crates/mh-quickjs-sys`). It is the
   single C entry point through which every mh PTC host function
   (`tool`, `read`, `write`, `edit`, `glob`, `grep`, `exec`, `batch`,
   `call_tool`, `tools.*`, and ToolResult handle methods) is dispatched;
   the per-function identity is carried in the closure `params` value.

Build flow (performed by `crates/mh-quickjs-sys/build.rs`):

1. Compile `mquickjs_build.c` + `mh_stdlib.c` (which defines `CONFIG_MH`
   and includes `mqjs_stdlib.c`) as a host tool.
2. Run it twice to generate `mh_atom.h` (`-a`) and `mh_stdlib.h`
   (the serialized stdlib ROM table + `js_stdlib` `JSSTDLibraryDef`).
3. Compile `mquickjs.c`, `cutils.c`, `dtoa.c`, `libm.c`, and `mh_shim.c`
   against the generated headers into the static library linked by Rust.
