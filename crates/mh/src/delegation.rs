use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::identity::{ExecutionId, IsolatedWorkspaceId, RevisionId, TaskId};
use crate::ptc::PtcExecution;
use crate::session::EvidenceRecord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DelegationAccess {
    Read,
    IsolatedWrite,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DelegationOptions {
    pub task: String,
    pub access: DelegationAccess,
    #[serde(default)]
    pub context: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationBudget {
    pub max_children: usize,
    pub max_parallel_children: usize,
    pub max_child_turns: usize,
    pub max_findings_bytes: usize,
}

impl Default for DelegationBudget {
    fn default() -> Self {
        Self {
            max_children: 8,
            max_parallel_children: 4,
            max_child_turns: 16,
            max_findings_bytes: 16 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegatedTask {
    pub task_id: TaskId,
    pub parent_task_id: TaskId,
    pub execution_id: ExecutionId,
    pub parent_execution_id: ExecutionId,
    pub objective: String,
    pub access: DelegationAccess,
    pub base_revision: RevisionId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DelegateResult {
    pub task_id: TaskId,
    pub execution_id: ExecutionId,
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

pub trait DelegationHost: Send + Sync {
    fn delegate(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        options: DelegationOptions,
    ) -> DelegateResult;
    fn delegate_batch(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        options: Vec<DelegationOptions>,
    ) -> Vec<DelegateResult>;
    fn integrate(
        &self,
        parent: &PtcExecution,
        revision: &RevisionId,
        workspace: IsolatedWorkspaceId,
    ) -> IntegrationResponse;
}
