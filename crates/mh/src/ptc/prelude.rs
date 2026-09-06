//! Load-time PTC prelude discovery.
//!
//! A prelude is a plain PTC JavaScript file evaluated into the VM before the
//! model's program runs. It defines derived tools by composing the existing
//! host ABI (`read`, `write`, `exec`, `batch`, …), so adjusting the available
//! tool set costs a file edit rather than a rebuild.
//!
//! This is deliberately not a plugin system (spec v0.1 §4.5, §32): a prelude
//! adds no host primitive, declares no ABI, and gains no capability the calling
//! PTC program did not already have.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Content identity of a prelude: the tool environment a PTC program ran in.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PreludeId(pub String);

/// Workspace-relative prelude path; takes precedence over the user prelude.
pub const WORKSPACE_PRELUDE: &str = ".mh/prelude.js";

/// Largest prelude accepted. Preludes define derived tools; anything larger is
/// a program, and would silently consume the PTC heap budget.
pub const MAX_PRELUDE_BYTES: usize = 256 * 1024;

/// Prefix marking a line that describes a prelude tool to the model.
const DOC_PREFIX: &str = "//!";

#[derive(Debug)]
pub enum PreludeError {
    TooLarge {
        path: PathBuf,
        bytes: usize,
    },
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl fmt::Display for PreludeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { path, bytes } => write!(
                f,
                "prelude {} is {bytes} bytes, exceeding the {MAX_PRELUDE_BYTES} byte limit",
                path.display()
            ),
            Self::Io { path, source } => {
                write!(f, "prelude {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for PreludeError {}

/// Where a prelude came from. Only workspace preludes arrive with a
/// repository, so only those need a trust decision; the user prelude is the
/// operator's own configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreludeOrigin {
    Workspace,
    User,
}

impl PreludeOrigin {
    /// Whether loading this prelude requires an explicit trust decision.
    pub const fn requires_confirmation(self) -> bool {
        matches!(self, Self::Workspace)
    }
}

/// A discovered prelude: its source and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prelude {
    pub path: PathBuf,
    pub source: String,
    pub origin: PreludeOrigin,
}

impl Prelude {
    /// Wraps the prelude so a syntax or runtime error names the prelude rather
    /// than surfacing as a failure inside the model's own program.
    pub fn eval_name(&self) -> String {
        format!("prelude:{}", self.path.display())
    }

    /// Content identity of the active tool environment.
    ///
    /// This is deliberately *not* part of the workspace revision. A revision
    /// hashes workspace content; a prelude is execution configuration that
    /// decides how tools behave. Mixing them would make the revision
    /// self-referential, because the revision cache itself lives under `.mh`.
    /// Recording the identity separately still answers the question evidence
    /// actually needs: which tool environment produced this verification.
    pub fn identity(&self) -> PreludeId {
        let mut hash = Sha256::new();
        hash.update(b"mh-prelude-v1\0");
        hash.update(self.source.as_bytes());
        PreludeId(format!("sha256:{:x}", hash.finalize()))
    }

    /// Tool descriptions this prelude advertises to the model, taken from its
    /// `//!` lines. Extracting them from source keeps description and
    /// definition in one file, and costs no VM to read.
    ///
    /// A prelude with no `//!` lines contributes nothing to the context; its
    /// functions still work for hand-written PTC programs.
    pub fn description(&self) -> Option<String> {
        let doc = self
            .source
            .lines()
            .filter_map(|line| line.trim_start().strip_prefix(DOC_PREFIX))
            .map(str::trim_end)
            .collect::<Vec<_>>();
        if doc.iter().all(|line| line.trim().is_empty()) {
            return None;
        }
        Some(
            doc.iter()
                .map(|line| line.strip_prefix(' ').unwrap_or(line))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }
}

/// Resolves the active prelude: workspace first, then the user prelude.
///
/// A missing prelude is not an error; it yields `None`. An unreadable or
/// oversized prelude *is* an error, because silently running without a prelude
/// the operator intended would change which tools the model believes it has.
pub fn discover(workspace: &Path, user: Option<&Path>) -> Result<Option<Prelude>, PreludeError> {
    let workspace_prelude = workspace.join(WORKSPACE_PRELUDE);
    if let Some(prelude) = load(&workspace_prelude, PreludeOrigin::Workspace)? {
        return Ok(Some(prelude));
    }
    match user {
        Some(path) => load(path, PreludeOrigin::User),
        None => Ok(None),
    }
}

/// The conventional user prelude path, honouring `XDG_CONFIG_HOME`.
pub fn user_prelude_path() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        return Some(Path::new(&xdg).join("mh/prelude.js"));
    }
    std::env::var_os("HOME")
        .filter(|v| !v.is_empty())
        .map(|home| Path::new(&home).join(".config/mh/prelude.js"))
}

fn load(path: &Path, origin: PreludeOrigin) -> Result<Option<Prelude>, PreludeError> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(PreludeError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if !metadata.is_file() {
        return Ok(None);
    }
    let bytes = usize::try_from(metadata.len()).unwrap_or(usize::MAX);
    if bytes > MAX_PRELUDE_BYTES {
        return Err(PreludeError::TooLarge {
            path: path.to_path_buf(),
            bytes,
        });
    }
    let source = std::fs::read_to_string(path).map_err(|source| PreludeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(Some(Prelude {
        path: path.to_path_buf(),
        source,
        origin,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, content: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn workspace_prelude_wins_over_the_user_prelude() {
        let dir = tempfile::tempdir().unwrap();
        let user = dir.path().join("user/prelude.js");
        write(&dir.path().join(WORKSPACE_PRELUDE), "// workspace");
        write(&user, "// user");

        let found = discover(dir.path(), Some(&user)).unwrap().unwrap();
        assert_eq!(found.source, "// workspace");
        assert_eq!(found.path, dir.path().join(WORKSPACE_PRELUDE));
    }

    #[test]
    fn user_prelude_is_the_fallback_and_absence_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let user = dir.path().join("user/prelude.js");
        assert_eq!(discover(dir.path(), Some(&user)).unwrap(), None);
        assert_eq!(discover(dir.path(), None).unwrap(), None);

        write(&user, "// user");
        let found = discover(dir.path(), Some(&user)).unwrap().unwrap();
        assert_eq!(found.source, "// user");
    }

    #[test]
    fn an_oversized_prelude_fails_instead_of_being_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(WORKSPACE_PRELUDE);
        write(&path, &"x".repeat(MAX_PRELUDE_BYTES + 1));

        let error = discover(dir.path(), None).unwrap_err();
        assert!(
            matches!(error, PreludeError::TooLarge { bytes, .. } if bytes == MAX_PRELUDE_BYTES + 1),
            "expected a size error, got {error}"
        );
    }

    #[test]
    fn a_directory_at_the_prelude_path_is_not_a_prelude() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(WORKSPACE_PRELUDE)).unwrap();
        assert_eq!(discover(dir.path(), None).unwrap(), None);
    }

    #[test]
    fn descriptions_come_from_doc_lines_only() {
        let prelude = Prelude {
            path: PathBuf::from("p.js"),
            source: concat!(
                "//! cargoTest() -> exec result for the test suite\n",
                "// an ordinary comment is not a description\n",
                "  //! indented doc lines still count\n",
                "function cargoTest() { return exec({ command: [\"cargo\", \"test\"] }); }\n",
            )
            .to_string(),
            origin: PreludeOrigin::Workspace,
        };
        assert_eq!(
            prelude.description().unwrap(),
            "cargoTest() -> exec result for the test suite\nindented doc lines still count"
        );
    }

    #[test]
    fn a_prelude_without_doc_lines_contributes_no_description() {
        let prelude = Prelude {
            path: PathBuf::from("p.js"),
            source: "// just code\nfunction f() { return 1; }\n".to_string(),
            origin: PreludeOrigin::Workspace,
        };
        assert_eq!(prelude.description(), None);

        let blank = Prelude {
            path: PathBuf::from("p.js"),
            source: "//!\n//!   \n".to_string(),
            origin: PreludeOrigin::Workspace,
        };
        assert_eq!(blank.description(), None, "blank doc lines are not content");
    }
}
