# mh

Durable agent runtime for long-running coding tasks, implemented in Rust with
an embedded MicroQuickJS runtime.

A task is durable. Model calls, context windows, PTC executions, delegated
workers, and subprocesses are disposable execution mechanisms. A task survives
many context windows, keeps working while its workers run, owns long-lived
processes, and resumes after the process that started it exits.

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

## Task lifecycle

A task ends when the model explicitly calls `finish(...)` and the runtime
accepts it. Plain text is a progress message: it is shown to you and parks the
task as `waiting-user`, but it never completes it. This matters for long
horizons, because a model reports progress constantly while the objective is
still open.

`finish()` is validated against durable state and refused — as a value the
program can branch on, not an exception — while any of these hold:

- a delegated worker is unresolved
- a background process is still running
- the newest verification of some kind failed
- an isolated worker delta was never integrated or discarded
- the workspace changed after the last passing verification
- the model's own goal state still lists pending work or unmet criteria

`finish({ force: true })` waives only the last item: judgement about remaining
work. It never completes over a live worker, a running process, an unmerged
delta, or a failed verification, because those would hide execution you can no
longer see.

Statuses are `queued`, `running`, `waiting-agent`, `waiting-process`,
`waiting-user`, `blocked`, `verifying`, `completed`, `failed`, and `cancelled`.
They are derived from the journal, not from whether a model call happens to be
in flight, so an inspector and a restarted runtime always agree.

## Context windows

`max_turns_per_window` bounds one context window, not the task. When a window
fills up, or the compiled context crosses the soft limit, the runtime writes a
durable `ContextCheckpoint` — objective, decisions, completed and pending work,
blockers, findings, failed approaches, changed paths, verification, workers,
processes, next actions — and opens a fresh window from it. The old transcript
is not replayed.

Because the window is replaced while the task continues, the model records
durable state as it goes with `goal(...)`. Anything not recorded is lost at the
next rollover, so the prompt says exactly that.

## Use

```sh
mh                            # interactive REPL
mh "inspect and fix it"       # start a task in the current workspace
mh run "..." --detach         # start a task that outlives this command
mh tasks                      # list durable tasks
mh inspect <task-id>          # full durable state of one task
mh attach <task-id>           # show state, then continue it in the foreground
mh resume [<task-id>]         # continue a durable task
mh steer <task-id> "..."      # append durable steering
mh cancel [<task-id>]         # request durable cancellation
mh sessions                   # session status and the last answer
```

`steer` and `cancel` are journal appends, so they work against a task running
in another process: the running loop picks them up at its next safe point.
`tasks` and `inspect` are read-only and never mutate the journal of a live
task.

Session state, compact host-call traces, goal state, worker and process
lifecycles, and verification evidence are stored under `.mh/` in the current
workspace. Filesystem tools are confined to that workspace, including
symlink-safe writes. Subprocesses start in the workspace with normal host OS
capabilities; provider API keys are removed from their environment. Press
Ctrl-C to cancel model, PTC, batch, worker, or process work.

## PTC runtime

The provider-facing entry point is:

```text
ptc({ source: "var files = glob(\"src/**/*.rs\"); return files;" })
```

Inside MicroQuickJS, the canonical host ABI is synchronous JavaScript:

```js
return tool("read", { path: "Cargo.toml" });
```

Convenience globals are also available inside PTC, each with a `tools.*` alias:
`read`, `write`, `edit`, `glob`, `grep`, `exec`, `batch`, `evidence`,
`checkpoint`, `restore`, `goal`, `finish`, `agent_spawn`, `agent_poll`,
`agent_join`, `agent_cancel`, `agent_send`, `agent_list`, `delegate`,
`delegate_batch`, `integrate`, `discard`, `process_spawn`, `process_poll`,
`process_tail`, `process_wait`, `process_kill`, `process_write`,
`process_list`, and `call_tool`. Here `exec` starts an argv-based OS
subprocess; it does not execute the PTC program itself. `batch(name, args)`
runs bounded parallel `read`, `grep`, `glob`, or `exec` calls and preserves
input ordering. `evidence(kind, ok, metadata)` records verification against the
current workspace revision.

MicroQuickJS stays synchronous. Concurrency lives in the host runtime, so
nothing here needs a Promise: `agent_spawn` returns a handle immediately and
the Rust scheduler runs the worker on its own thread.

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

### First-load confirmation

A workspace prelude arrives with the repository and runs arbitrary code with
your own capabilities, so `mh` asks before running one for the first time:

```text
[prelude] /path/to/repo/.mh/prelude.js is not yet trusted.
  It runs before every PTC program with your own capabilities: it can
  read and write this workspace and start subprocesses.
  2 lines, sha256:80ac441f…
  Advertised tools:
    summarize(path) -> file info
  Trust this prelude? [y/N]
```

- The decision is recorded in `$XDG_CONFIG_HOME/mh/prelude-trust.json`
  (falling back to `~/.config/mh/`) — never in the workspace, because a
  workspace-local record would be shipped by the repository it authorizes.
- Trust is keyed by prelude **content**, so editing a trusted prelude asks
  again, and cloning the same content elsewhere stays trusted.
- Declining is remembered too; later runs report
  `previously declined for this exact content` instead of re-asking.
- Declining does not fail the run. The agent proceeds without the derived
  tools.
- The prompt appears only when stdin and stderr are both terminals. A piped or
  CI run refuses an unreviewed prelude rather than blocking on a prompt nobody
  can answer, and never mistakes task input for an answer. An
  already-trusted prelude loads normally in those runs.
- The user prelude in your config directory is never gated; it is your own
  configuration, not a repository payload.

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

## Workers

A PTC program delegates reasoning to a bounded worker agent. A worker is not a
special mini-agent: it is a normal agent execution with a parent, so it gets
its own durable task, journal, goal state, evidence, context rollover, resume,
cancellation, and process ownership. Only a structured result crosses back;
worker transcripts and worker reasoning are never returned.

Workers are asynchronous. `agent_spawn` returns a handle immediately and the
root agent keeps doing model and PTC work while they run:

```js
var a = agent_spawn({ task: "inspect the parser architecture", access: "read" });
var b = agent_spawn({ task: "implement the storage refactor", profile: "implement" });

// Root work happens here, while a and b run.

agent_send(a.agent, "also inspect the parser tests");
var status = agent_poll(a.agent);      // never blocks
var result = agent_join(a.agent);      // blocks until this one worker settles
agent_cancel(b.agent);                 // affects only b
```

`delegate(options)` and `delegate_batch(options[])` remain, implemented as
spawn-then-join over the same runtime; `delegate_batch` returns one entry per
input in input order.

`task` is the worker objective. `access` is `"read"` or `"isolated-write"`, and
may come from `profile` instead: `explore`, `implement`, `review`, or `test`,
each supplying an access mode, a turn budget, and a system instruction. An
explicit `access` overrides the profile default; an unknown profile name is an
error rather than a silent fallback, because a worker running with unintended
write access is not a detail. An optional `context` value is serialized into
the worker's compiled context, bounded to 8 KiB; the parent's own conversation
is never copied in.

`access` selects the worker's capabilities:

- `"read"` runs the worker directly in the parent workspace with writes and
  subprocesses disabled.
- `"isolated-write"` materializes a private Git-backed workspace under
  `.mh/workspaces/` at the parent's current revision. The worker may write and
  run processes there; the parent workspace is never mutated until integration.

Every worker is pinned to the parent's current tracked revision and refuses to
start if its workspace does not materialize at exactly that revision.
`"isolated-write"` additionally requires a Git-root workspace. Delegation depth
is one: `agent_spawn`, `delegate`, `delegate_batch`, `integrate`, and `discard`
refuse inside a worker, and the refusal names whichever primitive was called.

Worker states are `queued`, `running`, `waiting`, `completed`, `failed`, and
`cancelled`. `agent_poll` returns the state plus the worker's own window, turn
count, pending work, and next actions, so a parent can inspect progress without
joining. `agent_join` returns:

```js
{
    taskId, agent,              // worker identities, allocated from the session
    ok,                         // true when the worker produced a final answer
    summary,                    // worker answer, or the failure reason
    baseRevision, finalRevision, changed,
    workspace,                  // isolated workspace id; absent for "read"
    evidence,                   // worker evidence records, provenance preserved
    findings                    // last worker PTC value, or a truncation marker
}
```

Workers are bounded by `max_children` (8), `max_parallel_children` (4),
`max_child_turns` (16, per window), `max_child_windows` (4), and
`max_findings_bytes` (16 KiB). Exceeding a budget fails that worker with
`ok: false` instead of degrading the parent. Ctrl-C cancels running workers.

## Background processes

`exec` is for foreground commands and waits for exit. A dev server, watcher,
long build, or persistent test runner is a durable process resource instead:

```js
var p = process_spawn({ command: ["cargo", "watch", "-x", "test"] });

process_poll(p.id);                      // {state, pid, exitCode, ...}
process_tail(p.id, "stdout", 100);       // bounded tail, cheap while running
process_wait(p.id, 5000);                // optional timeout
process_write(p.id, "input\n");
process_kill(p.id);
process_list();
```

`process_spawn` returns immediately; output is captured incrementally to
`.mh/processes/<id>.{out,err}` with a bounded in-memory tail, so tailing never
loads an unbounded log. Process metadata is durable and visible through task
inspection, and a running process blocks `finish()`.

An OS process does not survive the runtime that started it. Rather than
pretend otherwise, recovery marks a process from a previous runtime
`orphaned`, records whether its recorded pid still appears to exist, and
reports it that way. Metadata is durable; the process is not.

## Integration

An isolated worker's delta is applied only when the parent asks for it, and
only once that worker has settled:

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

Integration is refused as a conflict when the parent mutated any path the
worker also changed; the parent workspace is left byte-identical in that case.
Integrating a workspace whose worker is still running is refused outright,
because merging a half-written tree is worse than waiting. A successful
integration creates a new parent revision and sets `requiresReverification`,
because evidence recorded before the merge no longer matches the current
workspace revision. Re-run verification and record fresh `evidence(...)` after
integrating.

A changed delta that is neither integrated nor discarded blocks `finish()`,
since silently dropping a worker's work is exactly the failure the check
exists to prevent. After a conflict the parent either resolves and re-runs the
work, or abandons the delta explicitly:

```js
discard(child.workspace, "parent diverged on tracked; redoing it here");
```

`discard` is journaled as `IntegrationDiscarded`: losing work is recorded as a
decision, not an omission.

`.mh/workspaces/` entries are garbage collected when the owning agent
finishes.
