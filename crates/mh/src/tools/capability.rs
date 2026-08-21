//! Capability model (spec §16).
//!
//! Filesystem access is confined beneath a canonical workspace root.
//! Subprocess policy is explicit about its limited guarantee: workspace cwd
//! plus lifetime/output control, rather than an operating-system sandbox.

use std::path::{Path, PathBuf, absolute};

use serde::{Deserialize, Serialize};

/// Coarse side-effect class for a host tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolEffect {
    ReadOnly,
    WorkspaceMutation,
    Process,
    Meta,
}

/// Subprocess capability. `WorkspaceCwd` controls cwd and lifetime/output,
/// but is not an operating-system sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessPolicy {
    Disabled,
    WorkspaceCwd,
}

/// What side effects a PTC execution may perform.
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// Workspace root; all fs access is confined beneath it.
    pub workspace: PathBuf,
    /// Policy governing subprocess execution.
    pub process: ProcessPolicy,
}

/// Sandbox violation; the path the program tried to reach.
#[derive(Debug)]
pub struct SandboxError(pub String);

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "path escapes workspace: {}", self.0)
    }
}

impl std::error::Error for SandboxError {}

impl Capabilities {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        let workspace = workspace.into();
        Self {
            workspace: std::fs::canonicalize(&workspace).unwrap_or_else(|_| normalize(&workspace)),
            process: ProcessPolicy::WorkspaceCwd,
        }
    }

    /// Resolves an existing path for reading, following symlinks only when
    /// their canonical target remains beneath the canonical workspace root.
    pub fn resolve_existing_read(&self, path: &str) -> Result<PathBuf, SandboxError> {
        let lexical = self.resolve_lexical(path)?;
        let real = std::fs::canonicalize(&lexical)
            .map_err(|_| SandboxError(format!("{path}: not found")))?;
        self.ensure_inside(path, &real)?;
        Ok(real)
    }

    /// Resolves a write target without allowing an existing symlink, or a
    /// symlink in a not-yet-created target's ancestry, to escape the workspace.
    pub fn resolve_for_write(&self, path: &str) -> Result<PathBuf, SandboxError> {
        let lexical = self.resolve_lexical(path)?;

        if std::fs::symlink_metadata(&lexical).is_ok() {
            let real =
                std::fs::canonicalize(&lexical).map_err(|_| SandboxError(path.to_string()))?;
            self.ensure_inside(path, &real)?;
        }

        let mut ancestor = lexical.as_path();
        while std::fs::symlink_metadata(ancestor).is_err() {
            ancestor = ancestor
                .parent()
                .ok_or_else(|| SandboxError(path.to_string()))?;
        }
        let real_ancestor =
            std::fs::canonicalize(ancestor).map_err(|_| SandboxError(path.to_string()))?;
        self.ensure_inside(path, &real_ancestor)?;
        Ok(lexical)
    }

    fn resolve_lexical(&self, path: &str) -> Result<PathBuf, SandboxError> {
        let raw = Path::new(path);
        if raw
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
            || path.split(['/', '\\']).any(|segment| segment == "..")
        {
            return Err(SandboxError(path.to_string()));
        }
        let joined = if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            self.workspace.join(raw)
        };
        let lexical = normalize(&joined);
        if !lexical.starts_with(&self.workspace) {
            return Err(SandboxError(path.to_string()));
        }
        Ok(lexical)
    }

    fn ensure_inside(&self, path: &str, canonical: &Path) -> Result<(), SandboxError> {
        if canonical.starts_with(&self.workspace) {
            Ok(())
        } else {
            Err(SandboxError(path.to_string()))
        }
    }
}

/// Lexical normalization: collapses `.` and redundant separators
/// without touching the filesystem.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        use std::path::Component;
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                // `..` at the start cannot be resolved lexically; keep
                // it so the caller's starts_with check fails.
                out.push("..");
            }
            other => out.push(other.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    // absolute() is a no-op for absolute paths and makes relative
    // workspace roots comparable.
    match absolute(&out) {
        Ok(a) => a,
        Err(_) => out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> (Capabilities, tempfile::TempDir) {
        let workspace = tempfile::tempdir().unwrap();
        (Capabilities::new(workspace.path()), workspace)
    }

    #[test]
    fn allows_new_inside_paths_and_rejects_dotdot() {
        let workspace = tempfile::tempdir().unwrap();
        let caps = Capabilities::new(workspace.path());
        assert_eq!(
            caps.resolve_for_write("src/main.rs").unwrap(),
            std::fs::canonicalize(workspace.path())
                .unwrap()
                .join("src/main.rs")
        );
        assert!(caps.resolve_for_write("a/../b").is_err());
    }

    #[test]
    fn rejects_lexical_escape() {
        let (caps, _workspace) = caps();
        assert!(caps.resolve_for_write("../outside").is_err());
        assert!(caps.resolve_for_write("/etc/passwd").is_err());
        assert!(caps.resolve_for_write("sub/../../escape").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape_for_existing_and_new_targets() {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("existing"), "secret").unwrap();
        symlink(outside.path(), workspace.path().join("escape")).unwrap();
        let caps = Capabilities::new(workspace.path());

        assert!(caps.resolve_existing_read("escape/existing").is_err());
        assert!(caps.resolve_for_write("escape/existing").is_err());
        assert!(caps.resolve_for_write("escape/new/deep/file").is_err());
    }
}
