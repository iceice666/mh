//! Durable goal state and explicit completion (spec v0.5 §1).
//!
//! Before v0.5 a task ended when the model emitted plain text, which conflated
//! "the model said something" with "the objective is met". A long-running task
//! reports progress constantly, so text is now an ordinary assistant message
//! and completion is a separate, validated act: [`FinishRequest`] is checked
//! against durable runtime state before the task is allowed to complete.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::identity::{AgentId, ExecutionId, ProcessId, RevisionId, TaskId};

/// Lifecycle of a durable task (spec v0.5 §"Task state machine").
///
/// The status is derived from persisted events rather than from whether a model
/// call happens to be in flight, so an inspector and a restarted runtime agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskStatus {
    Queued,
    Running,
    WaitingAgent,
    WaitingProcess,
    WaitingUser,
    Blocked,
    Verifying,
    Completed,
    Failed,
    Cancelled,
}

impl TaskStatus {
    /// Whether the task still owns work. A terminal task is never resumed.
    pub const fn is_active(self) -> bool {
        !matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::WaitingAgent => "waiting-agent",
            Self::WaitingProcess => "waiting-process",
            Self::WaitingUser => "waiting-user",
            Self::Blocked => "blocked",
            Self::Verifying => "verifying",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// A unit of work the model committed to, tracked across context windows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkItem {
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl WorkItem {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            detail: None,
        }
    }
}

/// Something preventing progress that the model cannot resolve by retrying.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Blocker {
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub needs: Option<String>,
}

/// A condition the task must satisfy before `finish()` is honest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcceptanceCriterion {
    pub description: String,
    #[serde(default)]
    pub met: bool,
    /// Evidence kind that demonstrates this criterion, when one applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_kind: Option<String>,
}

/// A durable conclusion worth carrying across a context rollover.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    pub summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paths: Vec<PathBuf>,
}

/// An approach already proven not to work. Recorded so a fresh context window
/// does not spend its budget rediscovering the same dead end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Failure {
    pub approach: String,
    pub reason: String,
}

/// A decision that later windows must not silently reverse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Decision {
    pub decision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

/// Model-supplied goal state, updated through the PTC `goal()` primitive.
///
/// Only the model-authored fields live here. Revisions, evidence, workers, and
/// processes are derived from the journal instead of being restated, so goal
/// state can never disagree with what actually happened.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalUpdate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_criteria: Option<Vec<AcceptanceCriterion>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed: Option<Vec<WorkItem>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending: Option<Vec<WorkItem>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blockers: Option<Vec<Blocker>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decisions: Option<Vec<Decision>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub findings: Option<Vec<Finding>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_approaches: Option<Vec<Failure>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_actions: Option<Vec<String>>,
}

impl GoalUpdate {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// The durable goal of a task: what it is for, and how far it has got.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalState {
    pub task_id: TaskId,
    pub objective: String,
    pub status: TaskStatus,
    pub acceptance_criteria: Vec<AcceptanceCriterion>,
    pub completed_work: Vec<WorkItem>,
    pub pending_work: Vec<WorkItem>,
    pub blockers: Vec<Blocker>,
    pub decisions: Vec<Decision>,
    pub findings: Vec<Finding>,
    pub failed_approaches: Vec<Failure>,
    pub next_actions: Vec<String>,
    pub current_revision: RevisionId,
    /// Revision of the newest passing verification, when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_verified_revision: Option<RevisionId>,
}

impl GoalState {
    pub fn new(task_id: TaskId, objective: String, current_revision: RevisionId) -> Self {
        Self {
            task_id,
            objective,
            status: TaskStatus::Running,
            acceptance_criteria: Vec::new(),
            completed_work: Vec::new(),
            pending_work: Vec::new(),
            blockers: Vec::new(),
            decisions: Vec::new(),
            findings: Vec::new(),
            failed_approaches: Vec::new(),
            next_actions: Vec::new(),
            current_revision,
            last_verified_revision: None,
        }
    }

    /// Applies a model-supplied update. Every field is replace-if-present, so
    /// the model can amend one list without restating the rest.
    pub fn apply(&mut self, update: GoalUpdate) {
        if let Some(objective) = update.objective {
            self.objective = objective;
        }
        if let Some(criteria) = update.acceptance_criteria {
            self.acceptance_criteria = criteria;
        }
        if let Some(completed) = update.completed {
            self.completed_work = completed;
        }
        if let Some(pending) = update.pending {
            self.pending_work = pending;
        }
        if let Some(blockers) = update.blockers {
            self.blockers = blockers;
        }
        if let Some(decisions) = update.decisions {
            self.decisions = decisions;
        }
        if let Some(findings) = update.findings {
            self.findings = findings;
        }
        if let Some(failures) = update.failed_approaches {
            self.failed_approaches = failures;
        }
        if let Some(next) = update.next_actions {
            self.next_actions = next;
        }
    }

    /// Whether verification still describes the current workspace content.
    pub fn verification_is_current(&self) -> bool {
        self.last_verified_revision.as_ref() == Some(&self.current_revision)
    }
}

/// A model request to complete the task.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinishRequest {
    pub summary: String,
    #[serde(default)]
    pub unresolved: Vec<String>,
    #[serde(default)]
    pub evidence: Value,
    /// Set when the model knowingly finishes with work left over. It still
    /// cannot bypass runtime-verifiable objections such as a live worker.
    #[serde(default)]
    pub force: bool,
}

/// Why a `finish()` call was refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FinishObjection {
    /// Workers that have neither completed nor been cancelled.
    UnresolvedAgents { agents: Vec<AgentId> },
    /// Long-lived processes still running under this task.
    RunningProcesses { processes: Vec<ProcessId> },
    /// PTC executions admitted by a lost runner without a durable outcome.
    OutcomeUnknownExecutions { executions: Vec<ExecutionId> },
    /// The newest verification of this kind failed.
    FailedVerification { kinds: Vec<String> },
    /// The workspace changed after the last passing verification.
    StaleVerification {
        current: RevisionId,
        verified: Option<RevisionId>,
    },
    /// The model's own goal state still lists work.
    PendingWork { items: Vec<String> },
    /// Acceptance criteria the model has not marked met.
    UnmetCriteria { criteria: Vec<String> },
    /// An isolated child delta was never integrated or discarded.
    UnintegratedWorkspaces { workspaces: Vec<u64> },
    /// An integration conflict was reported and never resolved.
    OpenConflict { paths: Vec<String> },
}

impl FinishObjection {
    /// Whether `force: true` may override this objection.
    ///
    /// Runtime facts (a live worker, a running process, an unmerged delta) are
    /// not the model's to overrule: completing over them would leak execution
    /// the user can no longer see. Judgement calls about remaining work are.
    pub const fn is_overridable(&self) -> bool {
        matches!(
            self,
            Self::PendingWork { .. } | Self::UnmetCriteria { .. } | Self::StaleVerification { .. }
        )
    }

    pub fn explain(&self) -> String {
        match self {
            Self::UnresolvedAgents { agents } => format!(
                "{} delegated worker(s) are still unresolved: {}. Join or cancel them first.",
                agents.len(),
                join_ids(agents.iter().map(|agent| agent.0))
            ),
            Self::RunningProcesses { processes } => format!(
                "{} background process(es) are still running: {}. Wait for or kill them first.",
                processes.len(),
                join_ids(processes.iter().map(|process| process.0))
            ),
            Self::OutcomeUnknownExecutions { executions } => format!(
                "{} PTC execution(s) have unknown outcomes: {}. Inspect their effects before retrying or finishing.",
                executions.len(),
                join_ids(executions.iter().map(|execution| execution.0))
            ),
            Self::FailedVerification { kinds } => format!(
                "the newest verification failed for: {}. Fix and re-record evidence.",
                kinds.join(", ")
            ),
            Self::StaleVerification { current, verified } => format!(
                "the workspace is at revision {} but the last passing verification was {}. Re-verify.",
                current.0,
                verified
                    .as_ref()
                    .map_or("never recorded", |revision| revision.0.as_str())
            ),
            Self::PendingWork { items } => {
                format!("pending work remains: {}", items.join("; "))
            }
            Self::UnmetCriteria { criteria } => {
                format!("unmet acceptance criteria: {}", criteria.join("; "))
            }
            Self::UnintegratedWorkspaces { workspaces } => format!(
                "isolated child workspace(s) {} were never integrated or discarded.",
                join_ids(workspaces.iter().copied())
            ),
            Self::OpenConflict { paths } => format!(
                "an integration conflict on {} was never resolved.",
                paths.join(", ")
            ),
        }
    }
}

fn join_ids(ids: impl Iterator<Item = u64>) -> String {
    ids.map(|id| id.to_string()).collect::<Vec<_>>().join(", ")
}

/// Outcome of validating a `finish()` request against durable state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinishVerdict {
    pub accepted: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub objections: Vec<FinishObjection>,
    /// Objections the model waived with `force: true`, recorded rather than
    /// dropped so the durable journal shows what was knowingly skipped.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub waived: Vec<FinishObjection>,
}

impl FinishVerdict {
    pub fn accepted() -> Self {
        Self {
            accepted: true,
            objections: Vec::new(),
            waived: Vec::new(),
        }
    }

    /// Splits objections by whether `force` may override them.
    pub fn evaluate(objections: Vec<FinishObjection>, force: bool) -> Self {
        if !force {
            let accepted = objections.is_empty();
            return Self {
                accepted,
                objections,
                waived: Vec::new(),
            };
        }
        let (waived, blocking): (Vec<_>, Vec<_>) = objections
            .into_iter()
            .partition(FinishObjection::is_overridable);
        Self {
            accepted: blocking.is_empty(),
            objections: blocking,
            waived,
        }
    }

    /// Model-facing explanation of why completion was refused.
    pub fn explain(&self) -> String {
        if self.accepted {
            return "finish accepted".to_string();
        }
        let reasons = self
            .objections
            .iter()
            .map(|objection| format!("- {}", objection.explain()))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "finish() refused: the task is not complete.\n{reasons}\nResolve these, or call finish({{ force: true }}) if only pending-work judgement remains."
        )
    }
}

/// Durable semantic task state produced at a context rollover (spec v0.5 §2).
///
/// This is deliberately structured rather than a prose summary: a fresh window
/// needs to act on the state, and a summary of a transcript loses exactly the
/// parts (which revision, which evidence, which worker) that decide what to do
/// next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextCheckpoint {
    pub task_id: TaskId,
    pub window: u32,
    pub objective: String,
    pub status: TaskStatus,
    pub acceptance_criteria: Vec<AcceptanceCriterion>,
    pub decisions: Vec<Decision>,
    pub completed: Vec<WorkItem>,
    pub pending: Vec<WorkItem>,
    pub blockers: Vec<Blocker>,
    pub important_findings: Vec<Finding>,
    pub failed_approaches: Vec<Failure>,
    pub changed_paths: Vec<PathBuf>,
    pub verification: Vec<VerificationSummary>,
    pub active_agents: Vec<AgentSummary>,
    pub active_processes: Vec<ProcessSummary>,
    pub next_actions: Vec<String>,
    pub current_revision: RevisionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_verified_revision: Option<RevisionId>,
}

/// One verification, reduced to what a later window must know about it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationSummary {
    pub kind: String,
    pub ok: bool,
    pub revision: RevisionId,
    pub fresh: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A delegated worker as seen from the parent across a rollover.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSummary {
    pub agent: AgentId,
    pub task: String,
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// A background process as seen across a rollover.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessSummary {
    pub process: ProcessId,
    pub argv: Vec<String>,
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
}

impl ContextCheckpoint {
    /// Renders the checkpoint as the sticky context a fresh window receives.
    ///
    /// Rendered rather than serialized: the model reads this, and a labelled
    /// outline survives truncation far better than one JSON blob.
    pub fn render(&self) -> String {
        let mut out = format!(
            "[task state after context window {}]\nobjective: {}\nstatus: {}\ncurrent revision: {}\nlast verified revision: {}",
            self.window,
            self.objective,
            self.status.label(),
            self.current_revision.0,
            self.last_verified_revision
                .as_ref()
                .map_or("none", |revision| revision.0.as_str()),
        );
        section(
            &mut out,
            "acceptance criteria",
            &self.acceptance_criteria,
            |c| {
                format!(
                    "[{}] {}",
                    if c.met { "met" } else { "unmet" },
                    c.description
                )
            },
        );
        section(&mut out, "decisions", &self.decisions, |d| {
            d.rationale.as_ref().map_or_else(
                || d.decision.clone(),
                |why| format!("{} — {why}", d.decision),
            )
        });
        section(&mut out, "completed", &self.completed, |item| {
            item.title.clone()
        });
        section(&mut out, "pending", &self.pending, |item| {
            item.detail.as_ref().map_or_else(
                || item.title.clone(),
                |detail| format!("{} — {detail}", item.title),
            )
        });
        section(&mut out, "blockers", &self.blockers, |blocker| {
            blocker.needs.as_ref().map_or_else(
                || blocker.summary.clone(),
                |needs| format!("{} — needs {needs}", blocker.summary),
            )
        });
        section(&mut out, "findings", &self.important_findings, |finding| {
            if finding.paths.is_empty() {
                finding.summary.clone()
            } else {
                format!(
                    "{} ({})",
                    finding.summary,
                    finding
                        .paths
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        });
        section(
            &mut out,
            "failed approaches (do not retry)",
            &self.failed_approaches,
            |failure| format!("{}: {}", failure.approach, failure.reason),
        );
        section(&mut out, "changed paths", &self.changed_paths, |path| {
            path.display().to_string()
        });
        section(&mut out, "verification", &self.verification, |record| {
            format!(
                "{}: {} @ {} — {}{}",
                record.kind,
                if record.ok { "PASS" } else { "FAIL" },
                record.revision.0,
                if record.fresh { "fresh" } else { "stale" },
                record
                    .note
                    .as_ref()
                    .map(|note| format!(" — {note}"))
                    .unwrap_or_default()
            )
        });
        section(&mut out, "workers", &self.active_agents, |agent| {
            format!(
                "agent {} [{}] {}{}",
                agent.agent.0,
                agent.state,
                agent.task,
                agent
                    .summary
                    .as_ref()
                    .map(|summary| format!(" — {summary}"))
                    .unwrap_or_default()
            )
        });
        section(&mut out, "processes", &self.active_processes, |process| {
            format!(
                "process {} [{}] {}{}",
                process.process.0,
                process.state,
                process.argv.join(" "),
                process
                    .exit_code
                    .map(|code| format!(" exit {code}"))
                    .unwrap_or_default()
            )
        });
        section(&mut out, "next actions", &self.next_actions, Clone::clone);
        out
    }
}

fn section<T>(out: &mut String, title: &str, items: &[T], render: impl Fn(&T) -> String) {
    if items.is_empty() {
        return;
    }
    out.push_str(&format!("\n\n{title}:"));
    for item in items {
        out.push_str(&format!("\n- {}", render(item)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn revision(id: &str) -> RevisionId {
        RevisionId(id.to_string())
    }

    #[test]
    fn goal_update_replaces_only_supplied_fields() {
        let mut goal = GoalState::new(TaskId(1), "refactor".to_string(), revision("r1"));
        goal.apply(GoalUpdate {
            pending: Some(vec![WorkItem::new("update callers")]),
            ..GoalUpdate::default()
        });
        goal.apply(GoalUpdate {
            completed: Some(vec![WorkItem::new("moved module")]),
            ..GoalUpdate::default()
        });
        assert_eq!(goal.pending_work, vec![WorkItem::new("update callers")]);
        assert_eq!(goal.completed_work, vec![WorkItem::new("moved module")]);
        assert_eq!(goal.objective, "refactor", "objective was never restated");
    }

    #[test]
    fn force_waives_judgement_but_never_live_execution() {
        let verdict = FinishVerdict::evaluate(
            vec![
                FinishObjection::PendingWork {
                    items: vec!["tests".to_string()],
                },
                FinishObjection::UnresolvedAgents {
                    agents: vec![AgentId(3)],
                },
            ],
            true,
        );
        assert!(
            !verdict.accepted,
            "a live worker is not the model's to waive"
        );
        assert_eq!(
            verdict.objections,
            vec![FinishObjection::UnresolvedAgents {
                agents: vec![AgentId(3)]
            }]
        );
        assert_eq!(
            verdict.waived,
            vec![FinishObjection::PendingWork {
                items: vec!["tests".to_string()]
            }]
        );

        let waived_only = FinishVerdict::evaluate(
            vec![FinishObjection::PendingWork {
                items: vec!["docs".to_string()],
            }],
            true,
        );
        assert!(waived_only.accepted);
        assert_eq!(waived_only.waived.len(), 1);
    }

    #[test]
    fn unforced_finish_is_refused_with_an_actionable_explanation() {
        let verdict = FinishVerdict::evaluate(
            vec![FinishObjection::FailedVerification {
                kinds: vec!["tests".to_string()],
            }],
            false,
        );
        assert!(!verdict.accepted);
        let explanation = verdict.explain();
        assert!(explanation.contains("finish() refused"));
        assert!(explanation.contains("tests"));
    }

    #[test]
    fn checkpoint_renders_semantic_state_not_a_transcript() {
        let checkpoint = ContextCheckpoint {
            task_id: TaskId(4),
            window: 2,
            objective: "refactor storage".to_string(),
            status: TaskStatus::Running,
            acceptance_criteria: vec![AcceptanceCriterion {
                description: "tests pass".to_string(),
                met: false,
                evidence_kind: Some("tests".to_string()),
            }],
            decisions: vec![Decision {
                decision: "keep the trait object".to_string(),
                rationale: Some("callers are dynamic".to_string()),
            }],
            completed: vec![WorkItem::new("moved storage module")],
            pending: vec![WorkItem::new("update callers")],
            blockers: vec![],
            important_findings: vec![Finding {
                summary: "parser owns precedence".to_string(),
                paths: vec![PathBuf::from("src/parser.rs")],
            }],
            failed_approaches: vec![Failure {
                approach: "generic backend".to_string(),
                reason: "object safety".to_string(),
            }],
            changed_paths: vec![PathBuf::from("src/storage.rs")],
            verification: vec![VerificationSummary {
                kind: "tests".to_string(),
                ok: false,
                revision: revision("r2"),
                fresh: true,
                note: None,
            }],
            active_agents: vec![AgentSummary {
                agent: AgentId(2),
                task: "inspect callers".to_string(),
                state: "running".to_string(),
                summary: None,
            }],
            active_processes: vec![ProcessSummary {
                process: ProcessId(1),
                argv: vec!["cargo".to_string(), "watch".to_string()],
                state: "running".to_string(),
                exit_code: None,
            }],
            next_actions: vec!["join agent 2".to_string()],
            current_revision: revision("r2"),
            last_verified_revision: Some(revision("r1")),
        };
        let rendered = checkpoint.render();
        for expected in [
            "context window 2",
            "objective: refactor storage",
            "status: running",
            "[unmet] tests pass",
            "keep the trait object — callers are dynamic",
            "parser owns precedence (src/parser.rs)",
            "generic backend: object safety",
            "tests: FAIL @ r2 — fresh",
            "agent 2 [running] inspect callers",
            "process 1 [running] cargo watch",
            "join agent 2",
        ] {
            assert!(
                rendered.contains(expected),
                "missing {expected:?} in:\n{rendered}"
            );
        }
    }

    #[test]
    fn empty_sections_are_omitted_rather_than_rendered_as_none() {
        let checkpoint = ContextCheckpoint {
            task_id: TaskId(1),
            window: 1,
            objective: "inspect".to_string(),
            status: TaskStatus::Running,
            acceptance_criteria: vec![],
            decisions: vec![],
            completed: vec![],
            pending: vec![],
            blockers: vec![],
            important_findings: vec![],
            failed_approaches: vec![],
            changed_paths: vec![],
            verification: vec![],
            active_agents: vec![],
            active_processes: vec![],
            next_actions: vec![],
            current_revision: revision("r1"),
            last_verified_revision: None,
        };
        let rendered = checkpoint.render();
        assert!(!rendered.contains("pending:"));
        assert!(!rendered.contains("workers:"));
        assert!(rendered.contains("last verified revision: none"));
    }
}
