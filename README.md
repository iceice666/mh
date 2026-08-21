# mh

Minimal PTC-first coding-agent harness implemented in Rust with an embedded
MicroQuickJS runtime.

## Build

```sh
cargo build -p mh
```

The workspace vendors MicroQuickJS and builds it through
`crates/mh-quickjs-sys/build.rs`; no system JavaScript runtime is required.

## Configure

```sh
export MH_API_KEY=...
export MH_MODEL=gpt-4.1-mini
# Optional OpenAI-compatible Responses API root; defaults to https://api.openai.com/v1
export MH_BASE_URL=https://api.openai.com/v1
```

`OPENAI_API_KEY` is accepted as a fallback. `MH_MODEL_TIMEOUT_SECS` controls
the provider request timeout and defaults to 300 seconds.

`mh` uses `POST /v1/responses` with SSE streaming. The terminal prints
`[request]` as soon as the request is dispatched, then streams provider-supplied
reasoning summaries under `[reasoning]`, followed by response, tool, and PTC
status. Reasoning summaries are not the model's private raw chain-of-thought.

The model-facing Responses API exposes exactly one function: `ptc({ source })`.
Its `source` is executed inside bounded MicroQuickJS. Filesystem and process
operations are not separate provider functions; they are host calls available
only inside that PTC program.

## Use

```sh
mh                         # interactive REPL
mh "inspect and fix it"    # start a task in the current workspace
mh resume                  # resume the durable workspace session
mh sessions                # inspect session status and the last answer
```

Session state is stored under `.mh/` in the current workspace. Tool access is
confined to that workspace. Press Ctrl-C to cancel model, PTC, or process work.

## PTC runtime

The provider-facing entry point is:

```text
ptc({ source: "var files = glob(\"src/**/*.rs\"); return files;" })
```

Inside MicroQuickJS, the canonical host ABI is synchronous JavaScript:

```js
return tool("read", { path: "Cargo.toml" });
```

Convenience globals are also available inside PTC: `read`, `write`, `edit`,
`glob`, `grep`, `exec`, `call_tool`, and `tools.*`. Here `exec` starts an argv-
based OS subprocess; it does not execute the PTC program itself. `read(path)`
returns `{ path, content, totalLines, truncated }`; file text is in `.content`.
`glob(pattern)` returns a string array and `grep(args)` returns a match array.
Large tool output is stored as handle-backed results instead of being inserted
into model context directly.
