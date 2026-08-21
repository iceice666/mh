//! Context compiler with priority-aware token budgeting (spec §19–§20).

use serde::{Deserialize, Serialize};

use crate::session::{EventRecord, Session, SessionEvent};

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
    pub fn compile(&self, session: &Session) -> CompiledContext {
        let mut candidates = Vec::new();
        candidates.push(item(
            ContextSource::System,
            Priority::Hard,
            self.system_prompt.clone(),
        ));
        if let Some(task) = session.state().task.as_deref() {
            candidates.push(item(
                ContextSource::Task,
                Priority::Hard,
                format!("User task:\n{task}"),
            ));
        }
        for record in session.events() {
            if let Some(context_item) = event_item(record) {
                candidates.push(context_item);
            }
        }
        self.select(candidates)
    }

    pub fn select(&self, candidates: Vec<ContextItem>) -> CompiledContext {
        let mut selected = Vec::new();
        let mut used = 0usize;
        let mut omitted = 0usize;

        // Hard items are invariant and remain in chronological order.
        for item in candidates
            .iter()
            .filter(|item| item.priority == Priority::Hard)
        {
            used = used.saturating_add(item.estimated_tokens);
            selected.push(item.clone());
        }

        // For non-hard information, prefer high priority and recency.
        let mut indexes: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, item)| (item.priority != Priority::Hard).then_some(index))
            .collect();
        indexes.sort_by_key(|index| (candidates[*index].priority, *index));
        indexes.reverse();
        let mut keep = vec![false; candidates.len()];
        for index in indexes {
            let item = &candidates[index];
            if used.saturating_add(item.estimated_tokens) <= self.max_tokens {
                used += item.estimated_tokens;
                keep[index] = true;
            } else {
                omitted += 1;
            }
        }
        selected.extend(
            candidates
                .into_iter()
                .enumerate()
                .filter_map(|(index, item)| keep[index].then_some(item)),
        );

        CompiledContext {
            items: selected,
            estimated_tokens: used,
            omitted_items: omitted,
        }
    }
}

fn event_item(record: &EventRecord) -> Option<ContextItem> {
    match &record.event {
        SessionEvent::UserMessage { content } => Some(item(
            ContextSource::SessionEvent(record.seq),
            Priority::Hard,
            format!("User: {content}"),
        )),
        SessionEvent::AssistantMessage { content } => Some(item(
            ContextSource::SessionEvent(record.seq),
            Priority::Sticky,
            format!("Assistant: {content}"),
        )),
        SessionEvent::PtcCompleted { value, .. } => Some(item(
            ContextSource::SessionEvent(record.seq),
            Priority::Working,
            format!("PTC result:\n{}", bounded_json(value, 16 * 1024)),
        )),
        SessionEvent::PtcFailed { error, value, .. } => Some(item(
            ContextSource::SessionEvent(record.seq),
            Priority::Working,
            format!("PTC failed: {error}\n{}", bounded_json(value, 8 * 1024)),
        )),
        SessionEvent::Compacted { summary } => Some(item(
            ContextSource::Compaction,
            Priority::Sticky,
            format!("Session summary:\n{summary}"),
        )),
        SessionEvent::Interrupted { operation } => Some(item(
            ContextSource::SessionEvent(record.seq),
            Priority::Working,
            format!("Operation interrupted: {operation}"),
        )),
        SessionEvent::ModelStarted
        | SessionEvent::ModelCompleted { .. }
        | SessionEvent::PtcStarted { .. } => None,
    }
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

Return a concise final answer when the latest PTC result provides enough evidence to satisfy the user. Call `ptc` exactly once only when additional repository work is still required; never repeat a successful operation already shown by a PTC result.

PTC JavaScript uses var/function/return/if/for/while and these globals:
- tool(name, args) — canonical host ABI
- read(path) -> {path, content, totalLines, truncated}; use `.content` for file text
- write(path, content) -> {ok, path, bytes}; edit({path, old, new}) -> {ok, path}
- glob(pattern) -> string[]
- grep({pattern, path|paths, max?}) -> [{path, line, text}]
- exec({command: [program, ...args], timeout_ms?})

Use one PTC program to batch related reads, searches, edits, and commands instead of requesting host tools individually. exec runs an OS subprocess; it is not the provider PTC entry point.

exec returns {exitCode, stdout, stderr}. stdout/stderr are handles with:
.read(offset?, limit?), .head(n), .tail(n), .grep(pattern, limit?), .json(), .length.
All filesystem access is confined to the workspace. Commands are argv arrays, never shell strings.
Do not use require, import, fetch, process, fs, async/await, Promise, let, or const.
Use a top-level return statement. Large raw tool output stays in handles; return only evidence needed for the next inference."#
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionEvent;

    #[test]
    fn keeps_hard_and_recent_high_priority_items() {
        let compiler = ContextCompiler {
            max_tokens: 20,
            system_prompt: "system".to_string(),
        };
        let candidates = vec![
            item(ContextSource::Task, Priority::Hard, "task".to_string()),
            item(
                ContextSource::Compaction,
                Priority::Sticky,
                "old sticky information".to_string(),
            ),
            item(
                ContextSource::SessionEvent(3),
                Priority::Working,
                "new working information".to_string(),
            ),
        ];
        let compiled = compiler.select(candidates);
        assert!(compiled.items.iter().any(|item| item.content == "task"));
        assert!(compiled.estimated_tokens >= estimate_tokens("task"));
    }

    #[test]
    fn compile_uses_selected_ptc_result_not_blob_store() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = Session::open(dir.path()).unwrap();
        session.begin_task("inspect").unwrap();
        session
            .append(SessionEvent::PtcCompleted {
                value: serde_json::json!({"evidence": "small"}),
                tool_calls: 1,
                duration_ms: 2,
            })
            .unwrap();
        let compiled = ContextCompiler::default().compile(&session).render();
        assert!(compiled.contains("small"));
        assert!(!compiled.contains("results.jsonl"));
    }

    #[test]
    fn system_prompt_routes_repository_work_through_ptc() {
        let prompt = canonical_system_prompt();
        assert!(prompt.contains("provider function `ptc`"));
        assert!(prompt.contains("latest PTC result provides enough evidence"));
        assert!(prompt.contains("Call `ptc` exactly once"));
        assert!(prompt.contains("exec runs an OS subprocess"));
        assert!(prompt.contains("tool(name, args)"));
    }
}
