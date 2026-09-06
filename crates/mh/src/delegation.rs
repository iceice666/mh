//! Delegation contract: worker handles, access modes, and integration.
//!
//! v0.5 replaces delegation-as-RPC with handles. `agent_spawn` returns
//! immediately and the root agent keeps working; `agent_poll`, `agent_join`,
//! `agent_send`, and `agent_cancel` operate on the handle. The synchronous
//! `delegate` helpers remain, implemented as spawn-then-join over the same
//! runtime, so there is exactly one worker implementation.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::identity::{AgentId, IsolatedWorkspaceId, ProcessId, RevisionId, TaskId};
use crate::ptc::PtcExecution;
use crate::session::EvidenceRecord;

/// What a delegated worker may do to the workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AgentAccess {
    /// Runs in the parent workspace with writes and subprocesses disabled.
    Read,
    /// Runs in a private Git-backed workspace forked at the parent revision.
    IsolatedWrite,
}

impl AgentAccess {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::IsolatedWrite => "isolated-write",
        }
    }
}

/// Lifecycle of a delegated worker (spec v0.5 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    Queued,
    Running,
    /// Blocked on its own work: a background process or a model call it cannot
    /// proceed without.
    Waiting,
    Completed,
    Failed,
    Cancelled,
}

impl AgentState {
    /// Whether the worker has settled. A parent may only finish once every
    /// worker is terminal.
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Waiting => "waiting",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// A declarative worker profile (spec v0.5 §"Agent profiles").
///
/// Deliberately small: a name selecting access defaults, a turn budget, and an
/// extra system instruction. Anything larger would be a framework.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentProfile {
    pub name: &'static str,
    pub access: AgentAccess,
    pub max_turns_per_window: usize,
    pub instruction: &'static str,
}

/// Built-in profiles. An unknown profile name is an error, not a silent
/// fallback: a worker running with unintended write access is not a detail.
pub const PROFILES: &[AgentProfile] = &[
    AgentProfile {
        name: "explore",
        access: AgentAccess::Read,
        max_turns_per_window: 12,
        instruction: "Investigate and report. Do not attempt to change the workspace.",
    },
    AgentProfile {
        name: "implement",
        access: AgentAccess::IsolatedWrite,
        max_turns_per_window: 24,
        instruction: "Implement the objective in your isolated workspace and verify it there.",
    },
    AgentProfile {
        name: "review",
        access: AgentAccess::Read,
        max_turns_per_window: 12,
        instruction: "Review for correctness and risk. Report findings with exact paths.",
    },
    AgentProfile {
        name: "test",
        access: AgentAccess::IsolatedWrite,
        max_turns_per_window: 16,
        instruction: "Write and run tests in your isolated workspace; report failures precisely.",
    },
];

pub fn profile(name: &str) -> Option<&'static AgentProfile> {
    PROFILES.iter().find(|profile| profile.name == name)
}

/// Options for spawning a worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSpawnOptions {
    pub task: String,
    /// Optional when `profile` supplies the access mode.
    #[serde(default)]
    pub access: Option<AgentAccess>,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub context: Value,
}

impl AgentSpawnOptions {
    /// Resolves the effective access mode and profile.
    ///
    /// An explicit `access` wins over the profile default so a caller can run
    /// an `explore` profile against an isolated workspace deliberately.
    pub fn resolve(&self) -> Result<(AgentAccess, Option<&'static AgentProfile>), String> {
        let resolved = match self.profile.as_deref() {
            Some(name) => Some(profile(name).ok_or_else(|| {
                format!(
                    "unknown agent profile '{name}'; available: {}",
                    PROFILES
                        .iter()
                        .map(|profile| profile.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?),
            None => None,
        };
        let access = self
            .access
            .or(resolved.map(|profile| profile.access))
            .ok_or_else(|| "agent_spawn: access or profile is required".to_string())?;
        Ok((access, resolved))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationBudget {
    pub max_children: usize,
    pub max_parallel_children: usize,
    /// Model turns a worker may spend in one context window. A worker rolls
    /// over like the root agent, so this is not its lifetime.
    pub max_child_turns: usize,
    /// Context windows a worker may use before it is failed as non-converging.
    pub max_child_windows: u32,
    pub max_findings_bytes: usize,
}

impl Default for DelegationBudget {
    fn default() -> Self {
        Self {
            max_children: 8,
            max_parallel_children: 4,
            max_child_turns: 16,
            max_child_windows: 4,
            max_findings_bytes: 16 * 1024,
        }
    }
}

/// What a worker returns to its parent. Transcripts never cross back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DelegateResult {
    pub task_id: TaskId,
    pub agent: AgentId,
    pub ok: bool,
    pub summary: String,
    pub base_revision: RevisionId,
    pub final_revision: RevisionId,
    pub changed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace: Option<IsolatedWorkspaceId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRecord>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub findings: Value,
}

impl DelegateResult {
    pub fn failure(
        task_id: TaskId,
        agent: AgentId,
        revision: RevisionId,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            task_id,
            agent,
            ok: false,
            summary: summary.into(),
            base_revision: revision.clone(),
            final_revision: revision,
            changed: false,
            workspace: None,
            evidence: Vec::new(),
            findings: Value::Null,
        }
    }
}

/// Non-blocking status of a worker, as returned by `agent_poll`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentStatus {
    pub agent: AgentId,
    pub task_id: TaskId,
    pub state: AgentState,
    pub objective: String,
    pub access: AgentAccess,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    pub done: bool,
    /// Context window the worker is currently in.
    pub window: u32,
    pub turns_in_window: usize,
    /// Present once the worker settled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<DelegateResult>,
    /// The worker's own pending work, so a parent can inspect progress before
    /// joining.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub next_actions: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_revision: Option<RevisionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntegrationResponse {
    pub ok: bool,
    pub conflict: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_revision: Option<RevisionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_revision: Option<RevisionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_base: Option<RevisionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_current: Option<RevisionId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paths: Vec<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub requires_reverification: bool,
}

impl IntegrationResponse {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            conflict: false,
            previous_revision: None,
            parent_revision: None,
            child_base: None,
            parent_current: None,
            paths: Vec::new(),
            error: Some(message.into()),
            requires_reverification: false,
        }
    }
}

/// Host services a PTC program may reach for orchestration.
///
/// Implemented by the runtime, not by the PTC layer: MicroQuickJS stays
/// synchronous and the concurrency lives behind these calls.
pub trait DelegationHost: Send + Sync {
    /// Starts a worker and returns immediately.
    fn agent_spawn(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        options: AgentSpawnOptions,
    ) -> Result<AgentStatus, String>;
    /// Abandons an isolated worker's delta explicitly.
    ///
    /// Without this a conflicting delta could never be resolved, and the task
    /// could never legitimately finish. Discarding is journaled: losing work
    /// must be a decision, not an omission.
    fn discard(
        &self,
        parent: &PtcExecution,
        workspace: IsolatedWorkspaceId,
        reason: String,
    ) -> Result<Value, String>;
    fn agent_poll(&self, agent: AgentId) -> Result<AgentStatus, String>;
    /// Blocks the calling PTC program until the worker settles.
    fn agent_join(&self, agent: AgentId) -> Result<DelegateResult, String>;
    fn agent_cancel(&self, agent: AgentId) -> Result<AgentStatus, String>;
    fn agent_send(&self, agent: AgentId, message: String) -> Result<AgentStatus, String>;
    fn agent_list(&self) -> Vec<AgentStatus>;
    fn integrate(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        workspace: IsolatedWorkspaceId,
    ) -> IntegrationResponse;

    /// Records model-supplied goal state for the calling task.
    fn goal_update(
        &self,
        parent: &PtcExecution,
        update: crate::goal::GoalUpdate,
    ) -> Result<Value, String>;
    /// Requests explicit task completion; refusal is a value, not an error.
    fn finish(
        &self,
        parent: &PtcExecution,
        request: crate::goal::FinishRequest,
    ) -> Result<crate::goal::FinishVerdict, String>;

    fn process_spawn(
        &self,
        parent: &PtcExecution,
        spec: crate::process::ProcessSpec,
    ) -> Result<crate::process::ProcessSnapshot, String>;
    fn process_poll(&self, process: ProcessId) -> Result<crate::process::ProcessSnapshot, String>;
    fn process_tail(
        &self,
        process: ProcessId,
        stream: crate::process::ProcessStream,
        lines: usize,
    ) -> Result<crate::process::ProcessTail, String>;
    fn process_wait(
        &self,
        process: ProcessId,
        timeout_ms: Option<u64>,
    ) -> Result<crate::process::ProcessSnapshot, String>;
    fn process_kill(&self, process: ProcessId) -> Result<crate::process::ProcessSnapshot, String>;
    fn process_write(&self, process: ProcessId, data: &str) -> Result<Value, String>;
    fn process_list(&self) -> Vec<crate::process::ProcessSnapshot>;
}

/// Synchronous delegation kept for compatibility: spawn, then join.
pub fn delegate_sync(
    host: &dyn DelegationHost,
    parent: &PtcExecution,
    revision: &RevisionId,
    options: AgentSpawnOptions,
) -> Result<DelegateResult, String> {
    let status = host.agent_spawn(parent, revision, options)?;
    host.agent_join(status.agent)
}

/// Bounded parallel delegation: spawn every worker, then join in input order.
///
/// Parallelism comes from the scheduler underneath `agent_spawn`, so the JS
/// caller needs no promises and results stay in input order.
pub fn delegate_batch_sync(
    host: &dyn DelegationHost,
    parent: &PtcExecution,
    revision: &RevisionId,
    options: Vec<AgentSpawnOptions>,
) -> Vec<DelegateResult> {
    let spawned: Vec<Result<AgentStatus, String>> = options
        .into_iter()
        .map(|option| host.agent_spawn(parent, revision, option))
        .collect();
    spawned
        .into_iter()
        .map(|spawn| match spawn {
            Ok(status) => host.agent_join(status.agent).unwrap_or_else(|error| {
                DelegateResult::failure(status.task_id, status.agent, revision.clone(), error)
            }),
            Err(error) => DelegateResult::failure(TaskId(0), AgentId(0), revision.clone(), error),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_supplies_access_and_explicit_access_overrides_it() {
        let explore = AgentSpawnOptions {
            task: "inspect".to_string(),
            access: None,
            profile: Some("explore".to_string()),
            context: Value::Null,
        };
        let (access, resolved) = explore.resolve().unwrap();
        assert_eq!(access, AgentAccess::Read);
        assert_eq!(resolved.unwrap().name, "explore");

        let overridden = AgentSpawnOptions {
            access: Some(AgentAccess::IsolatedWrite),
            ..explore
        };
        assert_eq!(overridden.resolve().unwrap().0, AgentAccess::IsolatedWrite);
    }

    #[test]
    fn an_unknown_profile_is_an_error_not_a_silent_default() {
        let error = AgentSpawnOptions {
            task: "x".to_string(),
            access: None,
            profile: Some("wizard".to_string()),
            context: Value::Null,
        }
        .resolve()
        .unwrap_err();
        assert!(error.contains("unknown agent profile 'wizard'"));
        assert!(error.contains("explore"));
    }

    #[test]
    fn access_is_required_when_no_profile_is_given() {
        let error = AgentSpawnOptions {
            task: "x".to_string(),
            access: None,
            profile: None,
            context: Value::Null,
        }
        .resolve()
        .unwrap_err();
        assert!(error.contains("access or profile is required"));
    }

    #[test]
    fn terminal_states_are_exactly_the_settled_ones() {
        assert!(AgentState::Completed.is_terminal());
        assert!(AgentState::Failed.is_terminal());
        assert!(AgentState::Cancelled.is_terminal());
        assert!(!AgentState::Queued.is_terminal());
        assert!(!AgentState::Running.is_terminal());
        assert!(!AgentState::Waiting.is_terminal());
    }
}
