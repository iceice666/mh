# mh

Traceable, parallel, steerable PTC workflow runtime implemented in Rust with an
embedded MicroQuickJS runtime.

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
operations are synchronous host calls available only inside that PTC program;
host-side `batch()` supplies bounded parallelism without Promise or async JavaScript.

## Use

```sh
mh                         # interactive REPL
mh "inspect and fix it"    # start a task in the current workspace
mh resume                  # resume the durable workspace session
mh sessions                # inspect session status and the last answer
```

Session state, compact host-call traces, working state, and verification evidence
are stored under `.mh/` in the current workspace. Filesystem tools are confined
to that workspace, including symlink-safe writes. Subprocesses start in the
workspace with normal host OS capabilities; provider API keys are removed from
their environment. Press Ctrl-C to cancel model, PTC, batch, or process work.

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
`glob`, `grep`, `exec`, `batch`, `evidence`, `call_tool`, and `tools.*`. Here
`exec` starts an argv-based OS subprocess; it does not execute the PTC program
itself. `batch(name, args)` runs bounded parallel `read`, `grep`, `glob`, or
`exec` calls and preserves input ordering. `evidence(kind, ok, metadata)` records
verification against the current workspace mutation epoch.

`read(path)` returns `{ path, content, totalLines, truncated }`; file text is in
`.content`. `glob(pattern)` returns a string array and `grep(args)` returns a
match array; both arrays expose `.truncated`. Large subprocess output is stored
as handle-backed results instead of entering model context directly. Handles
provide `.read()`, `.head()`, `.tail()`, `.grep()`, `.json()`, plus `.id`,
`.length`, `.totalBytes`, `.truncated`, and `.kind` metadata.
