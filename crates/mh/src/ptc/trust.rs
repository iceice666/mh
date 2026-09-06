//! Prelude trust decisions.
//!
//! A workspace prelude runs arbitrary code with the caller's own capabilities,
//! and `.mh/` is excluded from the workspace revision, so cloning a repository
//! would otherwise hand it silent code execution on the next PTC program.
//!
//! Trust is therefore confirmed once per prelude content, and the record lives
//! in the user's config directory — never in the workspace. A record inside the
//! workspace would be shipped by the same repository it is supposed to
//! authorize.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::prelude::{Prelude, PreludeId};

/// What the operator decided about one prelude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustDecision {
    Trusted,
    Rejected,
}

/// Whether a prelude may run, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trust {
    /// Already confirmed for exactly this content.
    Trusted,
    /// Never seen: the caller must confirm before the prelude runs.
    Unknown,
    /// Previously rejected for exactly this content.
    Rejected,
}

#[derive(Debug)]
pub enum TrustError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Corrupt {
        path: PathBuf,
        message: String,
    },
    /// No user config directory, so no tamper-proof place to record trust.
    NoConfigDirectory,
}

impl fmt::Display for TrustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "prelude trust store {}: {source}", path.display())
            }
            Self::Corrupt { path, message } => {
                write!(f, "invalid prelude trust store {}: {message}", path.display())
            }
            Self::NoConfigDirectory => f.write_str(
                "cannot record prelude trust: no user config directory (set XDG_CONFIG_HOME or HOME)",
            ),
        }
    }
}

impl std::error::Error for TrustError {}

#[derive(Debug, Default, Serialize, Deserialize)]
struct TrustFile {
    /// Keyed by prelude content hash, so editing a trusted prelude requires a
    /// fresh confirmation rather than inheriting the old decision.
    #[serde(default)]
    preludes: BTreeMap<String, TrustEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrustEntry {
    decision: TrustDecision,
    /// Where it was confirmed from. Recorded for auditing only; trust is keyed
    /// by content so moving a prelude does not change its decision.
    path: PathBuf,
    decided_at_ms: u64,
}

/// Records prelude trust decisions outside every workspace.
#[derive(Debug, Clone)]
pub struct TrustStore {
    path: PathBuf,
}

impl TrustStore {
    /// Opens the store at the conventional user location.
    pub fn user() -> Result<Self, TrustError> {
        Ok(Self {
            path: user_trust_path().ok_or(TrustError::NoConfigDirectory)?,
        })
    }

    /// Opens a store at an explicit path. Tests and non-default layouts use
    /// this; it must never point inside a workspace.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn status(&self, prelude: &Prelude) -> Result<Trust, TrustError> {
        let file = self.load()?;
        Ok(match file.preludes.get(&prelude.identity().0) {
            Some(entry) => match entry.decision {
                TrustDecision::Trusted => Trust::Trusted,
                TrustDecision::Rejected => Trust::Rejected,
            },
            None => Trust::Unknown,
        })
    }

    /// Persists a decision for exactly this prelude content.
    pub fn record(
        &self,
        prelude: &Prelude,
        decision: TrustDecision,
    ) -> Result<PreludeId, TrustError> {
        let id = prelude.identity();
        let mut file = self.load()?;
        file.preludes.insert(
            id.0.clone(),
            TrustEntry {
                decision,
                path: prelude.path.clone(),
                decided_at_ms: now_ms(),
            },
        );
        self.store(&file)?;
        Ok(id)
    }

    fn load(&self) -> Result<TrustFile, TrustError> {
        let bytes = match std::fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                return Ok(TrustFile::default());
            }
            Err(source) => {
                return Err(TrustError::Io {
                    path: self.path.clone(),
                    source,
                });
            }
        };
        serde_json::from_slice(&bytes).map_err(|error| TrustError::Corrupt {
            path: self.path.clone(),
            message: error.to_string(),
        })
    }

    fn store(&self, file: &TrustFile) -> Result<(), TrustError> {
        let parent = self.path.parent().ok_or(TrustError::NoConfigDirectory)?;
        std::fs::create_dir_all(parent).map_err(|source| TrustError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
        let bytes = serde_json::to_vec_pretty(file).map_err(|error| TrustError::Corrupt {
            path: self.path.clone(),
            message: error.to_string(),
        })?;
        // Write-then-rename so an interrupted write cannot leave a truncated
        // store that would silently drop existing decisions.
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, &bytes).map_err(|source| TrustError::Io {
            path: tmp.clone(),
            source,
        })?;
        std::fs::rename(&tmp, &self.path).map_err(|source| TrustError::Io {
            path: self.path.clone(),
            source,
        })
    }
}

/// The conventional trust store path, honouring `XDG_CONFIG_HOME`.
pub fn user_trust_path() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|value| !value.is_empty()) {
        return Some(Path::new(&xdg).join("mh/prelude-trust.json"));
    }
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(|home| Path::new(&home).join(".config/mh/prelude-trust.json"))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptc::prelude::PreludeOrigin;

    fn prelude(source: &str) -> Prelude {
        Prelude {
            path: PathBuf::from("/ws/.mh/prelude.js"),
            source: source.to_string(),
            origin: PreludeOrigin::Workspace,
        }
    }

    #[test]
    fn an_unseen_prelude_is_unknown_and_a_recorded_one_is_not() {
        let dir = tempfile::tempdir().unwrap();
        let store = TrustStore::at(dir.path().join("trust.json"));
        let prelude = prelude("function a() {}");

        assert_eq!(store.status(&prelude).unwrap(), Trust::Unknown);
        store.record(&prelude, TrustDecision::Trusted).unwrap();
        assert_eq!(store.status(&prelude).unwrap(), Trust::Trusted);

        // Reopening reads the same decision from disk.
        let reopened = TrustStore::at(store.path());
        assert_eq!(reopened.status(&prelude).unwrap(), Trust::Trusted);
    }

    #[test]
    fn editing_a_trusted_prelude_requires_fresh_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let store = TrustStore::at(dir.path().join("trust.json"));
        let original = prelude("function a() {}");
        store.record(&original, TrustDecision::Trusted).unwrap();

        let edited = prelude("function a() {} function b() {}");
        assert_eq!(
            store.status(&edited).unwrap(),
            Trust::Unknown,
            "trust is keyed by content, so an edit is a new decision"
        );
        assert_eq!(store.status(&original).unwrap(), Trust::Trusted);
    }

    #[test]
    fn a_rejection_is_remembered_and_not_re_prompted() {
        let dir = tempfile::tempdir().unwrap();
        let store = TrustStore::at(dir.path().join("trust.json"));
        let prelude = prelude("function evil() {}");

        store.record(&prelude, TrustDecision::Rejected).unwrap();
        assert_eq!(store.status(&prelude).unwrap(), Trust::Rejected);
    }

    #[test]
    fn trust_follows_content_not_location() {
        let dir = tempfile::tempdir().unwrap();
        let store = TrustStore::at(dir.path().join("trust.json"));
        let here = Prelude {
            path: PathBuf::from("/a/.mh/prelude.js"),
            source: "function a() {}".to_string(),
            origin: PreludeOrigin::Workspace,
        };
        let moved = Prelude {
            path: PathBuf::from("/b/.mh/prelude.js"),
            source: "function a() {}".to_string(),
            origin: PreludeOrigin::Workspace,
        };
        store.record(&here, TrustDecision::Trusted).unwrap();

        assert_eq!(store.status(&moved).unwrap(), Trust::Trusted);
    }

    #[test]
    fn recording_preserves_other_decisions() {
        let dir = tempfile::tempdir().unwrap();
        let store = TrustStore::at(dir.path().join("trust.json"));
        let first = prelude("function a() {}");
        let second = prelude("function b() {}");

        store.record(&first, TrustDecision::Trusted).unwrap();
        store.record(&second, TrustDecision::Rejected).unwrap();

        assert_eq!(store.status(&first).unwrap(), Trust::Trusted);
        assert_eq!(store.status(&second).unwrap(), Trust::Rejected);
    }

    #[test]
    fn a_corrupt_store_is_reported_rather_than_treated_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trust.json");
        std::fs::write(&path, b"{ not json").unwrap();
        let store = TrustStore::at(&path);

        let error = store.status(&prelude("function a() {}")).unwrap_err();
        assert!(
            matches!(error, TrustError::Corrupt { .. }),
            "a damaged store must not silently re-trust everything, got {error}"
        );
    }

    #[test]
    fn the_trust_store_never_lives_inside_a_workspace() {
        let dir = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("XDG_CONFIG_HOME", dir.path()) };
        let path = user_trust_path().expect("XDG path resolves");
        unsafe { std::env::remove_var("XDG_CONFIG_HOME") };

        assert_eq!(path, dir.path().join("mh/prelude-trust.json"));
        assert!(
            !path
                .components()
                .any(|component| component.as_os_str() == ".mh"),
            "a workspace-local record would be shipped by the repository it authorizes"
        );
    }
}
