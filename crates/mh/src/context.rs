//! Context compiler with priority-aware token budgeting and multi-window
//! rollover (spec v0.5 §2).
//!
//! A task outlives any single context window, so the compiler never renders a
//! whole transcript. It renders durable *semantic* state — goal, verification,
//! workers, processes, the previous window's checkpoint — plus the events of
//! the current window only.

use serde::{Deserialize, Serialize};

use crate::delegation::AgentAccess;
use crate::goal::{ContextCheckpoint, TaskStatus};
use crate::identity::TaskId;
use crate::session::{AgentRecord, EventRecord, Session, SessionEvent, TaskView};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Ephemeral,
    Working,
    Sticky,
    Hard,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextSource {
    System,
    Task,
    Goal,
    CurrentInstruction,
    WorkState,
    SessionEvent(u64),
    Compaction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    pub source: ContextSource,
    pub priority: Priority,
    pub estimated_tokens: usize,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledContext {
    pub items: Vec<ContextItem>,
    pub estimated_tokens: usize,
    pub omitted_items: usize,
}

impl CompiledContext {
    pub fn render(&self) -> String {
        self.items
            .iter()
            .map(|item| item.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    HardOverflow {
        required_tokens: usize,
        max_tokens: usize,
    },
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HardOverflow {
                required_tokens,
                max_tokens,
            } => write!(
                f,
                "hard context requires {required_tokens} tokens, exceeding maximum {max_tokens}"
            ),
        }
    }
}

impl std::error::Error for ContextError {}

/// Largest prelude description admitted into context. A prelude may be far
/// larger than its documentation; only the documentation is ever sent.
pub const MAX_PRELUDE_DESCRIPTION_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone)]
pub struct ContextCompiler {
    /// Hard ceiling: compiling above this fails rather than silently dropping
    /// required context.
    pub max_tokens: usize,
    /// Rollover threshold. Crossing it triggers compaction into a fresh
    /// window instead of quietly omitting older items.
    pub soft_limit_tokens: usize,
    pub system_prompt: String,
    /// Tool documentation advertised by the active PTC prelude, if any.
    pub prelude_tools: Option<String>,
}

impl Default for ContextCompiler {
    fn default() -> Self {
        Self {
            max_tokens: 32_000,
            soft_limit_tokens: 24_000,
            system_prompt: canonical_system_prompt().to_string(),
            prelude_tools: None,
        }
    }
}

impl ContextCompiler {
    /// Compiles the context for one task, root or worker.
    ///
    /// Root and worker differ only in the system framing and the parent-supplied
    /// context item: a worker is a normal agent execution with a parent, so it
    /// gets the same goal state, evidence, and window machinery.
    pub fn compile(
        &self,
        session: &Session,
        task_id: TaskId,
    ) -> Result<CompiledContext, ContextError> {
        let view = session.task_view(task_id);
        let worker = (!view.agent.is_root())
            .then(|| session.agent(view.agent))
            .flatten();
        self.compile_view(&view, &session.events(), worker.as_ref())
    }

    /// Compiles from an already-derived view; the agent runtime holds one.
    pub fn compile_view(
        &self,
        view: &TaskView,
        events: &[EventRecord],
        worker: Option<&AgentRecord>,
    ) -> Result<CompiledContext, ContextError> {
        let mut candidates = vec![item(
            ContextSource::System,
            Priority::Hard,
            self.system_item(worker),
        )];
        if let Some(tools) = self.prelude_item() {
            candidates.push(tools);
        }
        candidates.push(item(
            ContextSource::Task,
            Priority::Hard,
            format!(
                "[{}]\n{}",
                if worker.is_some() {
                    "delegated objective"
                } else {
                    "objective"
                },
                view.goal.objective
            ),
        ));
        if let Some(worker) = worker
            && !worker.context.is_null()
        {
            candidates.push(item(
                ContextSource::CurrentInstruction,
                Priority::Working,
                format!(
                    "[parent-provided context]\n{}",
                    bounded_json(&worker.context, 8 * 1024)
                ),
            ));
        }
        if let Some(instruction) = view.latest_user_message.as_deref() {
            candidates.push(item(
                ContextSource::CurrentInstruction,
                Priority::Hard,
                format!("[current user instruction]\n{instruction}"),
            ));
        }
        // Sticky, and rendered before the work state: after a rollover this is
        // the task's memory, and losing it would restart the task.
        if let Some(checkpoint) = view.checkpoint.as_ref() {
            candidates.push(item(
                ContextSource::Compaction,
                Priority::Sticky,
                checkpoint.render(),
            ));
        }
        candidates.push(item(
            ContextSource::Goal,
            Priority::Sticky,
            render_goal_state(view),
        ));
        candidates.push(item(
            ContextSource::WorkState,
            Priority::Sticky,
            render_work_state(view),
        ));

        // Only the current window's events. Earlier windows are represented by
        // their checkpoint, which is the point of rolling over.
        let window = events
            .iter()
            .filter(|record| {
                record.seq >= view.window_start_seq && record.event.task_id() == Some(view.task_id)
            })
            .collect::<Vec<_>>();
        let latest_assistant_seq = window.iter().rev().find_map(|record| {
            matches!(record.event, SessionEvent::AssistantMessage { .. }).then_some(record.seq)
        });
        let latest_user_seq = window.iter().rev().find_map(|record| {
            matches!(
                record.event,
                SessionEvent::UserMessage { .. } | SessionEvent::SteeringApplied { .. }
            )
            .then_some(record.seq)
        });
        let latest_completed_seq = window.iter().rev().find_map(|record| {
            matches!(record.event, SessionEvent::PtcCompleted { .. }).then_some(record.seq)
        });
        let latest_failed_seq = window.iter().rev().find_map(|record| {
            matches!(record.event, SessionEvent::PtcFailed { .. }).then_some(record.seq)
        });
        let dialogue_start = window
            .iter()
            .filter(|record| {
                matches!(
                    record.event,
                    SessionEvent::UserMessage { .. } | SessionEvent::AssistantMessage { .. }
                )
            })
            .rev()
            .nth(5)
            .map_or(0, |record| record.seq);
        for record in window {
            if let Some(context_item) = event_item(
                record,
                latest_user_seq,
                latest_assistant_seq,
                latest_completed_seq,
                latest_failed_seq,
                dialogue_start,
            ) {
                candidates.push(context_item);
            }
        }
        self.select(candidates)
    }

    /// Whether a compiled context has crossed the rollover threshold.
    pub fn needs_rollover(&self, context: &CompiledContext) -> bool {
        context.estimated_tokens >= self.soft_limit_tokens
    }

    fn system_item(&self, worker: Option<&AgentRecord>) -> String {
        let Some(worker) = worker else {
            return format!("[system]\n{}", self.system_prompt);
        };
        let profile = worker
            .profile
            .as_deref()
            .and_then(crate::delegation::profile);
        let capability = match worker.access {
            AgentAccess::Read => {
                "You may read the workspace. Writes and subprocesses are disabled."
            }
            AgentAccess::IsolatedWrite => {
                "You work in a private workspace forked from the parent revision. Your changes reach the parent only if the parent integrates them."
            }
        };
        format!(
            "[system]\n{}\n\nYou are a delegated worker with a parent. Delegation depth is one: agent_spawn, agent_join, delegate, and integrate are unavailable to you. {capability} Report a compact final result through finish(), not a transcript.{}",
            self.system_prompt,
            profile
                .map(|profile| format!("\nProfile {}: {}", profile.name, profile.instruction))
                .unwrap_or_default()
        )
    }

    /// Renders the prelude tool documentation as a hard context item. It is
    /// hard priority because a model unaware of a prelude tool will re-derive
    /// it by hand, which is exactly what defining the prelude avoided.
    fn prelude_item(&self) -> Option<ContextItem> {
        let tools = self.prelude_tools.as_deref()?.trim_end();
        if tools.is_empty() {
            return None;
        }
        Some(item(
            ContextSource::System,
            Priority::Hard,
            format!(
                "[workspace prelude tools]\nThese synchronous helpers are already defined for every PTC program in this workspace. Prefer them over reimplementing the same work inline.\n{}",
                bounded_text(tools, MAX_PRELUDE_DESCRIPTION_BYTES)
            ),
        ))
    }

    pub fn select(&self, candidates: Vec<ContextItem>) -> Result<CompiledContext, ContextError> {
        let required_tokens = candidates
            .iter()
            .filter(|item| item.priority == Priority::Hard)
            .fold(0usize, |total, item| {
                total.saturating_add(item.estimated_tokens)
            });
        if required_tokens > self.max_tokens {
            return Err(ContextError::HardOverflow {
                required_tokens,
                max_tokens: self.max_tokens,
            });
        }

        let mut used = required_tokens;
        let mut omitted = 0usize;
        let mut keep = vec![false; candidates.len()];
        for (index, candidate) in candidates.iter().enumerate() {
            if candidate.priority == Priority::Hard {
                keep[index] = true;
            }
        }

        let mut indexes: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, item)| (item.priority != Priority::Hard).then_some(index))
            .collect();
        indexes.sort_by_key(|index| (candidates[*index].priority, *index));
        indexes.reverse();
        for index in indexes {
            let candidate = &candidates[index];
            if used.saturating_add(candidate.estimated_tokens) <= self.max_tokens {
                used += candidate.estimated_tokens;
                keep[index] = true;
            } else {
                omitted += 1;
            }
        }
        let items = candidates
            .into_iter()
            .enumerate()
            .filter_map(|(index, item)| keep[index].then_some(item))
            .collect();
        Ok(CompiledContext {
            items,
            estimated_tokens: used,
            omitted_items: omitted,
        })
    }
}

fn event_item(
    record: &EventRecord,
    latest_user_seq: Option<u64>,
    latest_assistant_seq: Option<u64>,
    latest_completed_seq: Option<u64>,
    latest_failed_seq: Option<u64>,
    dialogue_start: u64,
) -> Option<ContextItem> {
    match &record.event {
        SessionEvent::UserMessage { content, .. }
            if Some(record.seq) != latest_user_seq && record.seq >= dialogue_start =>
        {
            Some(item(
                ContextSource::SessionEvent(record.seq),
                Priority::Ephemeral,
                format!("[recent conversation]\nUser: {content}"),
            ))
        }
        SessionEvent::AssistantMessage { content, .. }
            if Some(record.seq) == latest_assistant_seq || record.seq >= dialogue_start =>
        {
            Some(item(
                ContextSource::SessionEvent(record.seq),
                if Some(record.seq) == latest_assistant_seq {
                    Priority::Sticky
                } else {
                    Priority::Ephemeral
                },
                format!("[recent conversation]\nAssistant: {content}"),
            ))
        }
        SessionEvent::PtcCompleted { value, .. } if Some(record.seq) == latest_completed_seq => {
            Some(item(
                ContextSource::SessionEvent(record.seq),
                Priority::Working,
                format!(
                    "[latest relevant PTC result]\n{}",
                    bounded_json(value, 16 * 1024)
                ),
            ))
        }
        SessionEvent::PtcFailed { error, value, .. } if Some(record.seq) == latest_failed_seq => {
            Some(item(
                ContextSource::SessionEvent(record.seq),
                Priority::Working,
                format!(
                    "[latest PTC failure]\n{error}\n{}",
                    bounded_json(value, 8 * 1024)
                ),
            ))
        }
        SessionEvent::FinishProposed { verdict, .. } if !verdict.accepted => Some(item(
            ContextSource::SessionEvent(record.seq),
            Priority::Working,
            verdict.explain(),
        )),
        SessionEvent::RepeatedActionDetected { fingerprint, .. } => Some(item(
            ContextSource::SessionEvent(record.seq),
            Priority::Working,
            format!(
                "The previous action was repeated without changing the result. Choose a different approach instead of retrying the same PTC program. Fingerprint: {fingerprint}"
            ),
        )),
        _ => None,
    }
}

/// Renders durable goal state: the part of the task that outlives the window.
fn render_goal_state(view: &TaskView) -> String {
    let goal = &view.goal;
    let mut out = format!(
        "[goal state]\nstatus: {}\ncontext window: {}\nturns in this window: {}",
        goal.status.label(),
        view.window,
        view.turns_in_window
    );
    if !goal.acceptance_criteria.is_empty() {
        out.push_str("\n\nacceptance criteria:");
        for criterion in &goal.acceptance_criteria {
            out.push_str(&format!(
                "\n- [{}] {}",
                if criterion.met { "met" } else { "unmet" },
                criterion.description
            ));
        }
    }
    for (title, items) in [
        ("completed work", &goal.completed_work),
        ("pending work", &goal.pending_work),
    ] {
        if items.is_empty() {
            continue;
        }
        out.push_str(&format!("\n\n{title}:"));
        for entry in items {
            out.push_str(&format!("\n- {}", entry.title));
        }
    }
    if !goal.blockers.is_empty() {
        out.push_str("\n\nblockers:");
        for blocker in &goal.blockers {
            out.push_str(&format!("\n- {}", blocker.summary));
        }
    }
    if !goal.next_actions.is_empty() {
        out.push_str("\n\nnext actions:");
        for action in &goal.next_actions {
            out.push_str(&format!("\n- {action}"));
        }
    }
    if goal.status == TaskStatus::WaitingUser {
        out.push_str(
            "\n\nThis task is parked awaiting the user. It is NOT complete; plain text did not finish it.",
        );
    }
    out
}

fn render_work_state(view: &TaskView) -> String {
    let changes = if view.changed_paths.is_empty() {
        "- none".to_string()
    } else {
        view.changed_paths
            .iter()
            .map(|path| format!("- {}", path.display()))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let evidence = if view.evidence.is_empty() {
        "- none".to_string()
    } else {
        view.latest_evidence()
            .into_iter()
            .map(|record| {
                // Evidence is only fresh when both the workspace content and
                // the tool environment that produced it still match. A changed
                // prelude can change what a verification actually ran.
                let freshness = if record.revision != view.goal.current_revision {
                    "stale"
                } else if record.prelude == view.prelude {
                    "fresh"
                } else {
                    "stale — recorded under a different prelude"
                };
                format!(
                    "- {}: {} @ {} — {freshness}{}",
                    record.kind,
                    if record.ok { "PASS" } else { "FAIL" },
                    record.revision.0,
                    record
                        .note
                        .as_deref()
                        .map(|note| format!(" — {note}"))
                        .unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut out = format!(
        "[working state]\ntask: {}\nbase revision: {}\ncurrent revision: {}\n\nworkspace changes:\n{changes}\n\nverification:\n{evidence}",
        view.task_id.0,
        view.base_revision
            .as_ref()
            .map_or("none", |revision| revision.0.as_str()),
        view.goal.current_revision.0,
    );
    if !view.agents.is_empty() {
        out.push_str("\n\nworkers:");
        for record in &view.agents {
            out.push_str(&format!(
                "\n- agent {} [{}] {} ({}){}",
                record.agent.0,
                record.state.label(),
                record.objective,
                record.access.label(),
                record
                    .result
                    .as_ref()
                    .map(|result| format!(" — {}", bounded_text(&result.summary, 300)))
                    .unwrap_or_default()
            ));
        }
        out.push_str(
            "\nUnresolved workers block finish(): join or cancel each one before completing.",
        );
    }
    if !view.processes.is_empty() {
        out.push_str("\n\nbackground processes:");
        for record in &view.processes {
            out.push_str(&format!(
                "\n- process {} [{}] {}{}",
                record.process.0,
                record.state,
                record.argv.join(" "),
                record
                    .exit_code
                    .map(|code| format!(" exit {code}"))
                    .unwrap_or_default()
            ));
        }
    }
    out
}

fn item(source: ContextSource, priority: Priority, content: String) -> ContextItem {
    ContextItem {
        estimated_tokens: estimate_tokens(&content),
        source,
        priority,
        content,
    }
}

/// Conservative dependency-free approximation. ASCII technical text
/// averages near four bytes/token; non-ASCII bytes are charged higher.
pub fn estimate_tokens(text: &str) -> usize {
    text.bytes()
        .map(|byte| if byte.is_ascii() { 1usize } else { 2usize })
        .sum::<usize>()
        .div_ceil(4)
        .max(1)
}

fn bounded_text(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n… <truncated>", &text[..end])
}

fn bounded_json(value: &serde_json::Value, max_bytes: usize) -> String {
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string());
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n… <truncated>", &text[..end])
}

/// Renders a checkpoint for the durable `Compacted` event's prose field, which
/// older readers and `mh inspect` both use.
pub fn render_checkpoint(checkpoint: &ContextCheckpoint) -> String {
    checkpoint.render()
}

pub fn canonical_system_prompt() -> &'static str {
    r#"You are mh, a coding agent running as a durable task. Use the provider function `ptc` whenever work requires repository access.

The task is durable; your context window is not. Plain text is a progress message to the user and NEVER completes the task. Completion happens only when finish(...) is accepted. When you have nothing left to do, call finish(). When you are waiting on the user, say so in text and stop.

Because your context window will be replaced while the task continues, record durable state as you go with goal(...): pending work, completed work, decisions, findings, and failed approaches. Anything you do not record is lost at the next rollover.

PTC JavaScript uses var/function/return/if/for/while and these synchronous globals:
- tool(name, args) — canonical host ABI
- read(path) -> {path, content, totalLines, truncated}; use `.content` for file text
- write(path, content) -> {ok, path, bytes}; edit({path, old, new}) -> {ok, path}
- glob(pattern) -> string[] with `.truncated`
- grep({pattern, path|paths, max?}) -> match[] with `.truncated`
- exec({command: [program, ...args], timeout_ms?}) — foreground command; waits for exit
- batch(name, args[]) — bounded parallel read/grep/glob/exec; preserves input order

Durable task state:
- goal({objective?, acceptanceCriteria?, completed?, pending?, blockers?, decisions?, findings?, failedApproaches?, nextActions?}) — each field replaces that list
- finish({summary, unresolved?, evidence?, force?}) -> {accepted, objections, waived}. A refusal explains what to resolve; it is a value, not an error. force only waives judgement about remaining work, never a live worker, running process, unmerged delta, or failed verification.
- evidence(kind, ok, metadata?) — record verification against the current workspace revision

Asynchronous workers (the root agent keeps working while they run):
- agent_spawn({task, access|profile, context?}) -> status with `.agent`; returns immediately
- agent_poll(agent) -> status; agent_list() -> status[]
- agent_join(agent) -> result; agent_cancel(agent); agent_send(agent, message)
- access is "read" or "isolated-write"; profiles are explore, implement, review, test
- delegate(options) and delegate_batch(options[]) still exist: spawn plus join
- integrate(workspace) — explicitly apply an isolated worker delta or return a structured conflict

Long-lived processes (never use exec for a dev server, watcher, or long build):
- process_spawn({command: [...], cwd?, env?, label?}) -> snapshot with `.id`; returns immediately
- process_poll(id), process_tail(id, "stdout"|"stderr", lines), process_wait(id, timeout_ms?), process_kill(id), process_write(id, data), process_list()

Workspace state:
- checkpoint() -> {id, revision}; restore(checkpoint) — restoration is never automatic

Use one PTC program for related work. Use batch() for procedural independent host operations. Use workers only for independent model reasoning; they receive bounded context, not your conversation. isolated-write work never changes the parent until integrate(). After integration, re-run verification in the parent because worker evidence remains attributed to the worker revision. Treat verification as fresh only when its revision equals the current workspace revision. Do not emulate concurrency with Promise or async JavaScript. write/edit/restore are not batch-safe. exec runs an OS subprocess; it is not the provider PTC entry point.

exec returns {exitCode, durationMs, stdout, stderr}. stdout/stderr are handles with:
.read(offset?, limit?), .head(n), .tail(n), .grep(pattern, limit?), .json(),
.id, .length, .totalBytes, .truncated, .kind.
All filesystem access is confined to the workspace. Commands are argv arrays, never shell strings. Subprocesses use the workspace as cwd but otherwise have host OS capabilities; provider credentials are removed from their environment.
Do not use require, import, fetch, process, fs, async/await, Promise, let, or const. If a structured diagnostic reports unsupported lexical syntax, replace let/const with var.
Use a top-level return statement. Large raw tool output stays in handles; return only evidence needed for the next inference."#
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delegation::{AgentAccess, AgentState, DelegateResult};
    use crate::goal::{Finding, GoalUpdate, WorkItem};
    use crate::identity::{AgentId, ExecutionId, RevisionId};
    use crate::ptc::prelude::PreludeId;
    use crate::session::{EvidenceRecord, SessionEvent};
    use serde_json::Value;

    fn session() -> (Session, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let session = Session::open(dir.path()).unwrap();
        (session, dir)
    }

    #[test]
    fn hard_overflow_is_explicit() {
        let compiler = ContextCompiler {
            max_tokens: 1,
            soft_limit_tokens: 1,
            system_prompt: "system".to_string(),
            prelude_tools: None,
        };
        let error = compiler
            .select(vec![item(
                ContextSource::Task,
                Priority::Hard,
                "too large".to_string(),
            )])
            .unwrap_err();
        assert_eq!(
            error,
            ContextError::HardOverflow {
                required_tokens: estimate_tokens("too large"),
                max_tokens: 1
            }
        );
    }

    #[test]
    fn selection_never_exceeds_budget() {
        let compiler = ContextCompiler {
            max_tokens: 5,
            soft_limit_tokens: 4,
            system_prompt: String::new(),
            prelude_tools: None,
        };
        let compiled = compiler
            .select(vec![
                item(ContextSource::Task, Priority::Hard, "task".to_string()),
                item(
                    ContextSource::Compaction,
                    Priority::Sticky,
                    "sticky context".to_string(),
                ),
                item(
                    ContextSource::SessionEvent(3),
                    Priority::Working,
                    "working context".to_string(),
                ),
            ])
            .unwrap();
        assert!(compiled.estimated_tokens <= compiler.max_tokens);
    }

    #[test]
    fn a_rolled_over_window_carries_the_checkpoint_and_drops_the_transcript() {
        let (mut session, _dir) = session();
        let task = session.begin_task("refactor storage").unwrap();
        session
            .append(SessionEvent::AssistantMessage {
                task_id: task,
                content: "ANCIENT_TRANSCRIPT_LINE".to_string(),
            })
            .unwrap();
        session
            .append(SessionEvent::GoalUpdated {
                task_id: task,
                update: GoalUpdate {
                    findings: Some(vec![Finding {
                        summary: "EARLY_FINDING storage hides behind a trait".to_string(),
                        paths: vec![],
                    }]),
                    pending: Some(vec![WorkItem::new("update callers")]),
                    ..GoalUpdate::default()
                },
            })
            .unwrap();
        let checkpoint = crate::session::build_checkpoint(&session.task_view(task));
        session
            .append(SessionEvent::Compacted {
                task_id: task,
                agent: AgentId::ROOT,
                window: 0,
                summary: checkpoint.render(),
                checkpoint: Some(checkpoint),
            })
            .unwrap();

        let compiled = ContextCompiler::default().compile(&session, task).unwrap();
        let rendered = compiled.render();
        assert!(
            rendered.contains("EARLY_FINDING"),
            "an early finding must survive rollover:\n{rendered}"
        );
        assert!(
            rendered.contains("update callers"),
            "pending work must survive rollover"
        );
        assert!(
            !rendered.contains("ANCIENT_TRANSCRIPT_LINE"),
            "the pre-rollover transcript must not be replayed:\n{rendered}"
        );
        assert!(rendered.contains("context window: 1"));
    }

    #[test]
    fn worker_context_names_its_parent_and_capability_limits() {
        let (mut session, _dir) = session();
        let root = session.begin_task("root").unwrap();
        let (agent, child) = session.reserve_agent().unwrap();
        session
            .append(SessionEvent::AgentSpawned {
                parent_task_id: root,
                parent_agent: AgentId::ROOT,
                agent,
                task_id: child,
                objective: "inspect the parser".to_string(),
                access: AgentAccess::Read,
                profile: Some("explore".to_string()),
                base_revision: session.state().current_revision,
                context: serde_json::json!({ "hint": "start at parser.rs" }),
                isolated_workspace: None,
            })
            .unwrap();

        let rendered = ContextCompiler::default()
            .compile(&session, child)
            .unwrap()
            .render();
        assert!(rendered.contains("[delegated objective]\ninspect the parser"));
        assert!(rendered.contains("delegated worker with a parent"));
        assert!(rendered.contains("Writes and subprocesses are disabled"));
        assert!(rendered.contains("Profile explore"));
        assert!(rendered.contains("start at parser.rs"));
    }

    #[test]
    fn unresolved_workers_and_processes_appear_in_the_parent_work_state() {
        let (mut session, _dir) = session();
        let root = session.begin_task("orchestrate").unwrap();
        let revision = session.state().current_revision;
        let (agent, child) = session.reserve_agent().unwrap();
        session
            .append(SessionEvent::AgentSpawned {
                parent_task_id: root,
                parent_agent: AgentId::ROOT,
                agent,
                task_id: child,
                objective: "implement".to_string(),
                access: AgentAccess::IsolatedWrite,
                profile: None,
                base_revision: revision.clone(),
                context: Value::Null,
                isolated_workspace: None,
            })
            .unwrap();
        session
            .append(SessionEvent::AgentStateChanged {
                agent,
                task_id: child,
                state: AgentState::Running,
            })
            .unwrap();
        session
            .append(SessionEvent::ProcessSpawned {
                task_id: root,
                agent: AgentId::ROOT,
                process: session.reserve_process().unwrap(),
                argv: vec!["cargo".to_string(), "watch".to_string()],
                cwd: session.workspace_root(),
                label: Some("watch".to_string()),
                pid: Some(4242),
            })
            .unwrap();

        let rendered = ContextCompiler::default()
            .compile(&session, root)
            .unwrap()
            .render();
        assert!(rendered.contains("agent 1 [running] implement (isolated-write)"));
        assert!(rendered.contains("Unresolved workers block finish()"));
        assert!(rendered.contains("process 1 [running] cargo watch"));
    }

    #[test]
    fn a_refused_finish_is_surfaced_to_the_next_turn() {
        let (mut session, _dir) = session();
        let task = session.begin_task("verify").unwrap();
        let request = crate::goal::FinishRequest {
            summary: "done".to_string(),
            ..crate::goal::FinishRequest::default()
        };
        let verdict = crate::goal::FinishVerdict::evaluate(
            vec![crate::goal::FinishObjection::FailedVerification {
                kinds: vec!["tests".to_string()],
            }],
            false,
        );
        session
            .append(SessionEvent::FinishProposed {
                task_id: task,
                request,
                verdict,
            })
            .unwrap();
        let rendered = ContextCompiler::default()
            .compile(&session, task)
            .unwrap()
            .render();
        assert!(rendered.contains("finish() refused"));
        assert!(rendered.contains("tests"));
    }

    #[test]
    fn work_state_labels_revision_and_prelude_freshness() {
        let (mut session, _dir) = session();
        let task = session.begin_task("inspect").unwrap();
        let current = session.state().current_revision;
        let active = PreludeId("sha256:active".to_string());
        session
            .append(SessionEvent::PreludeLoaded {
                path: std::path::PathBuf::from(".mh/prelude.js"),
                prelude: active.clone(),
                described: true,
            })
            .unwrap();
        for (kind, revision, prelude, ts) in [
            ("tests", current.clone(), Some(active.clone()), 1u64),
            ("lint", RevisionId("base".to_string()), None, 2),
            (
                "bench",
                current.clone(),
                Some(PreludeId("sha256:old".to_string())),
                3,
            ),
        ] {
            session
                .append(SessionEvent::EvidenceRecorded {
                    evidence: EvidenceRecord {
                        kind: kind.to_string(),
                        ok: true,
                        revision,
                        result_ids: vec![],
                        note: None,
                        timestamp_ms: ts,
                        task_id: task,
                        execution_id: ExecutionId(1),
                        prelude,
                    },
                })
                .unwrap();
        }
        let view = session.task_view(task);
        let rendered = render_work_state(&view);
        assert!(rendered.contains(&format!("task: {}", task.0)));
        assert!(rendered.contains("tests: PASS @ "));
        assert!(rendered.contains("lint: PASS @ base — stale"));
        assert!(
            rendered.contains("bench: PASS @ ")
                && rendered.contains("stale — recorded under a different prelude"),
            "matching revision under a changed prelude is not fresh: {rendered}"
        );
    }

    #[test]
    fn a_parked_task_is_told_it_is_not_complete() {
        let (session, _dir) = session();
        let task = session.begin_task("park").unwrap();
        session
            .set_status(task, TaskStatus::WaitingUser, None)
            .unwrap();
        let rendered = render_goal_state(&session.task_view(task));
        assert!(rendered.contains("status: waiting-user"));
        assert!(rendered.contains("It is NOT complete"));
    }

    #[test]
    fn worker_results_are_bounded_in_the_parent_work_state() {
        let (mut session, _dir) = session();
        let root = session.begin_task("orchestrate").unwrap();
        let revision = session.state().current_revision;
        let (agent, child) = session.reserve_agent().unwrap();
        session
            .append(SessionEvent::AgentSpawned {
                parent_task_id: root,
                parent_agent: AgentId::ROOT,
                agent,
                task_id: child,
                objective: "inspect".to_string(),
                access: AgentAccess::Read,
                profile: None,
                base_revision: revision.clone(),
                context: Value::Null,
                isolated_workspace: None,
            })
            .unwrap();
        session
            .append(SessionEvent::AgentCompleted {
                agent,
                task_id: child,
                result: DelegateResult {
                    task_id: child,
                    agent,
                    ok: true,
                    summary: "x".repeat(5_000),
                    base_revision: revision.clone(),
                    final_revision: revision,
                    changed: false,
                    workspace: None,
                    evidence: vec![],
                    findings: Value::Null,
                },
            })
            .unwrap();
        let rendered = render_work_state(&session.task_view(root));
        assert!(rendered.contains("<truncated>"));
        assert!(rendered.len() < 2_000);
    }

    #[test]
    fn canonical_prompt_states_the_v05_completion_contract() {
        let prompt = canonical_system_prompt();
        assert!(prompt.contains("NEVER completes the task"));
        assert!(prompt.contains("finish("));
        assert!(prompt.contains("goal("));
        assert!(prompt.contains("agent_spawn("));
        assert!(prompt.contains("agent_join("));
        assert!(prompt.contains("process_spawn("));
        assert!(prompt.contains("process_tail("));
        assert!(prompt.contains("batch(name, args[])"));
        assert!(prompt.contains("evidence(kind, ok, metadata?)"));
        assert!(prompt.contains("checkpoint()"));
        assert!(prompt.contains("restore(checkpoint)"));
        assert!(prompt.contains(".totalBytes"));
        assert!(prompt.contains("write/edit/restore are not batch-safe"));
        assert!(prompt.contains("replace let/const with var"));
    }
}
