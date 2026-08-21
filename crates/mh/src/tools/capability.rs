//! Capability model (spec §16).
//!
//! v0.1 keeps the simple default policy: read/write inside the
//! workspace, exec from the workspace, network disabled. The
//! [`Capabilities`] struct exists so policy decisions live in one
//! place; `PathRule` lists are deliberately deferred until needed.

use std::path::Path;
use std::path::PathBuf;
use std::path::absolute;

/// What side effects a PTC execution may perform.
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// Workspace root; all fs access is confined beneath it.
    pub workspace: PathBuf,
    /// Allow running subprocesses via `exec`.
    pub exec: bool,
    /// Allow network access (no v0.1 tool uses it; always false).
    pub network: bool,
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
        Self {
            workspace: workspace.into(),
            exec: true,
            network: false,
        }
    }

    /// Resolves `path` (relative to the workspace when not absolute)
    /// and verifies it stays inside the workspace.
    ///
    /// Lexical `..` traversal is rejected rather than normalized away:
    /// model-generated paths that try to escape are a bug to surface,
    /// not to silently reinterpret.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, SandboxError> {
        let raw = Path::new(path);
        if path.split(['/', '\\']).any(|seg| seg == "..") {
            return Err(SandboxError(path.to_string()));
        }
        let joined = if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            self.workspace.join(raw)
        };
        let canonical_input = normalize(&joined);
        let root = normalize(&self.workspace);
        if !canonical_input.starts_with(&root) {
            return Err(SandboxError(path.to_string()));
        }
        Ok(canonical_input)
    }

    /// Same as [`resolve`] but additionally checks the path exists on
    /// disk with a canonical form inside the workspace (symlinks are
    /// resolved by the OS).
    pub fn resolve_existing(&self, path: &str) -> Result<PathBuf, SandboxError> {
        let lexical = self.resolve(path)?;
        match std::fs::canonicalize(&lexical) {
            Ok(real) => {
                // Compare against the canonicalized root too: on
                // macOS, /tmp is a symlink to /private/tmp, so the
                // lexical root never prefixes the canonical path.
                let root = match std::fs::canonicalize(&self.workspace) {
                    Ok(r) => r,
                    Err(_) => normalize(&self.workspace),
                };
                if !real.starts_with(&root) {
                    return Err(SandboxError(path.to_string()));
                }
                Ok(real)
            }
            Err(_) => Err(SandboxError(format!("{path}: not found"))),
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

    fn caps() -> Capabilities {
        Capabilities::new("/w/proj")
    }

    #[test]
    fn allows_inside_paths() {
        assert_eq!(
            caps().resolve("src/main.rs").unwrap(),
            PathBuf::from("/w/proj/src/main.rs")
        );
        assert!(
            caps().resolve("a/../b").is_err(),
            "dotdot segments are rejected outright"
        );
    }

    #[test]
    fn rejects_escape() {
        assert!(caps().resolve("../outside").is_err());
        assert!(caps().resolve("/etc/passwd").is_err());
        assert!(caps().resolve("sub/../../escape").is_err());
    }
}
