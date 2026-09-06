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
`glob`, `grep`, `exec`, `batch`, `evidence`, `checkpoint`, `restore`,
`delegate`, `delegate_batch`, `integrate`, `call_tool`, and `tools.*`. Here
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

## Prelude

Derived tools are defined at load time, not compiled in. `mh` evaluates a
prelude before every PTC program, so adjusting the available tool set costs a
file edit instead of a rebuild:

```js
// .mh/prelude.js
//! summarize(path) -> { path, lines } for one workspace file
//! cargoTest() -> exec result for the workspace test suite
function summarize(path) {
    var file = read(path);
    return { path: file.path, lines: file.totalLines };
}
function cargoTest() {
    return exec({ command: ["cargo", "test"] });
}
```

The first prelude found is used: `.mh/prelude.js` in the workspace, otherwise
`$XDG_CONFIG_HOME/mh/prelude.js` (falling back to `~/.config/mh/prelude.js`).
A missing prelude is normal; an unreadable or oversized one (over 256 KiB)
aborts the run rather than silently changing which tools exist. The active
prelude is reported as `[prelude] <path>` at startup.

`//!` lines are the tool documentation sent to the model, bounded to 4 KiB. The
prelude body itself is never sent. A prelude with no `//!` lines still loads and
still works for hand-written PTC programs, but the model is not told it exists
and will not call it; the startup line says so.

A prelude is not a plugin system. It defines no host primitive, and its
functions are ordinary PTC code: every call inside one is a normal host call,
counted against the same tool budget, confined to the same workspace sandbox,
and recorded in the same host-call trace. A prelude cannot replace a host
global — `read`, `exec`, and the rest are reasserted after it is evaluated, so
redefining one has no effect and cannot desynchronize the trace from what ran.
Delegated children evaluate the same prelude and receive the same documentation.

A prelude that fails to parse surfaces on first use as a PTC failure naming the
prelude path, not as an error inside the model's program.

### Prelude provenance

A prelude is execution configuration, not workspace content, so it is
deliberately **not** part of the workspace revision. Including it would make
the revision self-referential: the revision cache, session journal, and
checkpoints all live under `.mh`, so hashing that directory would mean every
recorded event changed the revision and no revision would ever converge.

Its identity is tracked separately instead, which is what provenance actually
needs:

- Loading a prelude appends a durable `prelude_loaded` event carrying the
  content hash, so the journal records which tool environment every later host
  call ran under.
- `evidence(...)` records the active prelude hash. Verification counts as fresh
  only when both the revision and the prelude still match; evidence recorded
  under a different prelude is reported as stale, because a changed prelude can
  change what a verification actually ran.
- `checkpoint()` records the active prelude. Restoring under a different
  prelude is refused with a prelude-mismatch error rather than silently
  reinstating content verified in another tool environment; the workspace is
  left untouched when that happens.

## Delegation

A PTC program can delegate a reasoning task to a bounded child agent. The child
runs its own model loop with its own context; only a structured result crosses
back into the parent. Child transcripts and child reasoning are never returned.

```js
var reports = delegate_batch([
    { task: "Explain how the parser represents precedence.", access: "read" },
    { task: "Identify the exact failing parser test behavior.", access: "read" }
]);
```

`task` is the child objective and `access` is required. An optional `context`
value is serialized into the child's compiled context, bounded to 8 KiB; the
parent's own conversation is never copied in.

`access` selects the child's capabilities:

- `"read"` runs the child directly in the parent workspace with writes and
  subprocesses disabled.
- `"isolated-write"` materializes a private Git-backed workspace under
  `.mh/workspaces/` at the parent's current revision. The child may write and
  run processes there; the parent workspace is never mutated until integration.

Every child is pinned to the parent's current tracked revision and refuses to
start if its workspace does not materialize at exactly that revision.
`"isolated-write"` additionally requires a Git-root workspace. Nested
delegation is rejected: `delegate`, `delegate_batch`, and `integrate` are
unavailable inside a child execution.

`delegate(options)` returns, and `delegate_batch(options[])` returns one entry
per input in input order:

```js
{
    taskId, executionId,        // child identities, allocated from the session
    ok,                         // true when the child produced a final answer
    summary,                    // child answer, or the failure reason
    baseRevision, finalRevision, changed,
    workspace,                  // isolated workspace id; absent for "read"
    evidence,                   // child evidence records, provenance preserved
    findings                    // last child PTC value, or a truncation marker
}
```

Delegation is bounded by `max_children` (8), `max_parallel_children` (4),
`max_child_turns` (16), and `max_findings_bytes` (16 KiB). Exceeding a budget
fails that child with `ok: false` instead of degrading the parent. Ctrl-C
cancels running children and records `DelegationCancelled`.

## Integration

An isolated child's delta is applied only when the parent asks for it:

```js
var merged = integrate(child.workspace);

if (merged.conflict) {
    return merged;    // parent diverged on merged.paths; nothing was applied
}
```

`integrate(workspace)` returns:

```js
{
    ok, conflict,
    previousRevision, parentRevision,   // present when applied
    childBase, parentCurrent,           // present on conflict
    paths,                              // applied paths, or conflicting paths
    error,
    requiresReverification              // true after a successful apply
}
```

Integration is refused as a conflict when the parent mutated any path the child
also changed; the parent workspace is left byte-identical in that case. A
successful integration creates a new parent revision and sets
`requiresReverification`, because evidence recorded before the merge no longer
matches the current workspace epoch. Re-run verification and record fresh
`evidence(...)` after integrating.

`.mh/workspaces/` entries are garbage collected when the owning agent finishes.
