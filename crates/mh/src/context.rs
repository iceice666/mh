//! Context compiler with priority-aware token budgeting (spec §19–§20).

use serde::{Deserialize, Serialize};

use crate::session::{EventRecord, Session, SessionEvent, WorkState};

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

#[derive(Debug, Clone)]
pub struct ContextCompiler {
    pub max_tokens: usize,
    pub system_prompt: String,
}

impl Default for ContextCompiler {
    fn default() -> Self {
        Self {
            max_tokens: 32_000,
            system_prompt: canonical_system_prompt().to_string(),
        }
    }
}

impl ContextCompiler {
    pub fn compile(&self, session: &Session) -> Result<CompiledContext, ContextError> {
        let work = session.work_state();
        let mut candidates = vec![item(
            ContextSource::System,
            Priority::Hard,
            format!("[system]\n{}", self.system_prompt),
        )];
        if let Some(objective) = work.objective.as_deref() {
            candidates.push(item(
                ContextSource::Task,
                Priority::Hard,
                format!("[objective]\n{objective}"),
            ));
        }
        if let Some(instruction) = work.latest_user_message.as_deref() {
            candidates.push(item(
                ContextSource::CurrentInstruction,
                Priority::Hard,
                format!("[current user instruction]\n{instruction}"),
            ));
        }
        candidates.push(item(
            ContextSource::WorkState,
            Priority::Sticky,
            render_work_state(&work),
        ));

        let latest_user_seq = session.events().iter().rev().find_map(|record| {
            matches!(
                record.event,
                SessionEvent::UserMessage { .. } | SessionEvent::SteeringApplied { .. }
            )
            .then_some(record.seq)
        });
        let latest_assistant_seq = session.events().iter().rev().find_map(|record| {
            matches!(record.event, SessionEvent::AssistantMessage { .. }).then_some(record.seq)
        });
        let latest_completed_seq = session.events().iter().rev().find_map(|record| {
            matches!(
                record.event,
                SessionEvent::PtcCompleted { task_id, .. } if Some(task_id) == work.task_id
            )
            .then_some(record.seq)
        });
        let latest_failed_seq = session.events().iter().rev().find_map(|record| {
            matches!(
                record.event,
                SessionEvent::PtcFailed { task_id, .. } if Some(task_id) == work.task_id
            )
            .then_some(record.seq)
        });
        let dialogue_start = session
            .events()
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

        for record in session.events() {
            if let Some(context_item) = event_item(
                &record,
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

    pub fn compile_child(
        &self,
        objective: &str,
        parent_context: &serde_json::Value,
        workspace: &crate::workspace::WorkspaceRevision,
        access: crate::delegation::DelegationAccess,
    ) -> Result<CompiledContext, ContextError> {
        self.select(vec![
            item(
                ContextSource::System,
                Priority::Hard,
                format!(
                    "[system]\n{}\n\nYou are a delegated child execution. Delegation depth is one: delegate and integrate are unavailable. Return a compact final result, not a transcript. Access mode: {:?}.",
                    self.system_prompt, access
                ),
            ),
            item(
                ContextSource::Task,
                Priority::Hard,
                format!("[delegated objective]\n{objective}"),
            ),
            item(
                ContextSource::CurrentInstruction,
                Priority::Working,
                format!(
                    "[parent-provided context]\n{}",
                    bounded_json(parent_context, 8 * 1024)
                ),
            ),
            item(
                ContextSource::WorkState,
                Priority::Sticky,
                format!(
                    "[workspace state]\nrevision: {}\nchanged paths: none",
                    workspace.id.0
                ),
            ),
        ])
    }

    pub fn compile_child_followup(
        &self,
        objective: &str,
        parent_context: &serde_json::Value,
        workspace: &crate::workspace::WorkspaceRevision,
        access: crate::delegation::DelegationAccess,
        latest_result: &serde_json::Value,
        latest_error: Option<&str>,
    ) -> Result<CompiledContext, ContextError> {
        let mut context = self.compile_child(objective, parent_context, workspace, access)?;
        let content = latest_error.map_or_else(
            || {
                format!(
                    "[latest child PTC result]\n{}",
                    bounded_json(latest_result, 16 * 1024)
                )
            },
            |error| {
                format!(
                    "[latest child PTC failure]\n{error}\n{}",
                    bounded_json(latest_result, 8 * 1024)
                )
            },
        );
        context.items.push(item(
            ContextSource::SessionEvent(0),
            Priority::Working,
            content,
        ));
        self.select(context.items)
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
        SessionEvent::Compacted { summary, .. } => Some(item(
            ContextSource::Compaction,
            Priority::Sticky,
            format!("[older compacted state]\n{summary}"),
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

fn render_work_state(work: &WorkState) -> String {
    let changes = if work.changed_paths.is_empty() {
        "- none".to_string()
    } else {
        work.changed_paths
            .iter()
            .map(|path| format!("- {}", path.display()))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let evidence = if work.evidence.is_empty() {
        "- none".to_string()
    } else {
        work.evidence
            .iter()
            .map(|record| {
                let freshness = if record.revision == work.current_revision {
                    "fresh"
                } else {
                    "stale"
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
    format!(
        "[working state]\ntask: {}\nbase revision: {}\ncurrent revision: {}\n\nworkspace changes:\n{changes}\n\nverification:\n{evidence}",
        work.task_id
            .map_or_else(|| "none".to_string(), |id| id.0.to_string()),
        work.base_revision
            .as_ref()
            .map_or("none", |revision| revision.0.as_str()),
        work.current_revision.0,
    )
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

pub fn canonical_system_prompt() -> &'static str {
    r#"You are mh, a coding agent. Use the provider function `ptc` whenever work requires repository access.

Return a concise final answer when the latest PTC result provides enough evidence to satisfy the user. Call `ptc` exactly once only when additional repository work is still required; never repeat a successful operation already shown by a PTC result. Treat verification as fresh only when its revision equals the current workspace revision.

PTC JavaScript uses var/function/return/if/for/while and these synchronous globals:
- tool(name, args) — canonical host ABI
- read(path) -> {path, content, totalLines, truncated}; use `.content` for file text
- write(path, content) -> {ok, path, bytes}; edit({path, old, new}) -> {ok, path}
- glob(pattern) -> string[] with `.truncated`
- grep({pattern, path|paths, max?}) -> match[] with `.truncated`
- exec({command: [program, ...args], timeout_ms?})
- batch(name, args[]) — bounded parallel read/grep/glob/exec; preserves input order and isolates item errors
- delegate({task, access, context?}) — independent model reasoning with read or isolated-write workspace access
- delegate_batch(options[]) — bounded parallel independent reasoning; preserves input order
- integrate(workspace) — explicitly apply an isolated child delta or return a structured conflict
- evidence(kind, ok, metadata?) — record verification against the current workspace revision
- checkpoint() -> {id, revision} — explicitly capture the current workspace state
- restore(checkpoint) — explicitly restore a checkpoint; restoration is never automatic

The same checkpoint and restore operations are available through tool("checkpoint", {}) and tool("restore", {checkpoint}). Use one PTC program for related work. Use batch() for procedural independent host operations. Use delegate() only for independent model reasoning; children receive bounded context rather than the parent conversation. isolated-write work never changes the parent until integrate(). After integration, re-run verification in the parent because child evidence remains attributed to the child revision. Do not emulate concurrency with Promise or async JavaScript. write/edit/restore are not batch-safe. exec runs an OS subprocess; it is not the provider PTC entry point.

exec returns {exitCode, durationMs, stdout, stderr}. stdout/stderr are handles with:
.read(offset?, limit?), .head(n), .tail(n), .grep(pattern, limit?), .json(),
.id, .length, .totalBytes, .truncated, .kind.
All filesystem access is confined to the workspace. Commands are argv arrays, never shell strings. Subprocesses use the workspace as cwd but otherwise have host OS capabilities; provider credentials are removed from their environment.
Do not use require, import, fetch, process, fs, async/await, Promise, let, or const. If a structured diagnostic reports unsupported lexical syntax, replace let/const with var.
Use a top-level return statement. Large raw tool output stays in handles; return only evidence needed for the next inference."#
}

#[cfg(test)]
mod v03_prompt_tests {
    use super::canonical_system_prompt;

    #[test]
    fn canonical_prompt_exposes_v03_workflow_surface() {
        let prompt = canonical_system_prompt();
        assert!(prompt.contains("batch(name, args[])"));
        assert!(prompt.contains("evidence(kind, ok, metadata?)"));
        assert!(prompt.contains("checkpoint()"));
        assert!(prompt.contains("restore(checkpoint)"));
        assert!(prompt.contains(".totalBytes"));
        assert!(prompt.contains("with `.truncated`"));
        assert!(prompt.contains("write/edit/restore are not batch-safe"));
        assert!(prompt.contains("replace let/const with var"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ExecutionId, RevisionId, TaskId};
    use crate::session::EvidenceRecord;

    #[test]
    fn hard_overflow_is_explicit() {
        let compiler = ContextCompiler {
            max_tokens: 1,
            system_prompt: "system".to_string(),
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
            system_prompt: String::new(),
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
    fn work_state_labels_revision_freshness() {
        let current = RevisionId("current".to_string());
        let work = WorkState {
            task_id: Some(TaskId(7)),
            objective: Some("inspect".to_string()),
            base_revision: Some(RevisionId("base".to_string())),
            current_revision: current.clone(),
            latest_user_message: None,
            changed_paths: vec![std::path::PathBuf::from("src/lib.rs")],
            latest_ptc: None,
            latest_failure: None,
            evidence: vec![
                EvidenceRecord {
                    kind: "tests".to_string(),
                    ok: true,
                    revision: current,
                    result_ids: vec![],
                    note: None,
                    timestamp_ms: 1,
                    task_id: TaskId(7),
                    execution_id: ExecutionId(1),
                },
                EvidenceRecord {
                    kind: "lint".to_string(),
                    ok: true,
                    revision: RevisionId("base".to_string()),
                    result_ids: vec![],
                    note: None,
                    timestamp_ms: 2,
                    task_id: TaskId(7),
                    execution_id: ExecutionId(2),
                },
            ],
            legacy_evidence: vec![],
        };
        let rendered = render_work_state(&work);
        assert!(rendered.contains("task: 7"));
        assert!(rendered.contains("current revision: current"));
        assert!(rendered.contains("tests: PASS @ current — fresh"));
        assert!(rendered.contains("lint: PASS @ base — stale"));
    }

    #[test]
    fn system_prompt_routes_repository_work_through_ptc() {
        let prompt = canonical_system_prompt();
        assert!(prompt.contains("provider function `ptc`"));
        assert!(prompt.contains("checkpoint()"));
        assert!(prompt.contains("restore(checkpoint)"));
    }
}
