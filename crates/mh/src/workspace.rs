use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::identity::RevisionId;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceRevision {
    pub id: RevisionId,
    pub sequence: u64,
    pub parent: Option<RevisionId>,
    pub changed_paths: Vec<PathBuf>,
    pub detected_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceDelta {
    pub from: RevisionId,
    pub to: RevisionId,
    pub added: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
    pub deleted: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevisionSource {
    Tool,
    Process,
    External,
    Restore,
}

#[derive(Debug)]
pub enum WorkspaceError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Git {
        operation: &'static str,
        message: String,
    },
    InvalidWorkspace(PathBuf),
    UnknownRevision(RevisionId),
    CorruptCache {
        path: PathBuf,
        message: String,
    },
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "workspace I/O error at {}: {source}", path.display())
            }
            Self::Git { operation, message } => write!(f, "git {operation} failed: {message}"),
            Self::InvalidWorkspace(path) => {
                write!(f, "workspace is not a directory: {}", path.display())
            }
            Self::UnknownRevision(id) => write!(f, "workspace revision is not cached: {}", id.0),
            Self::CorruptCache { path, message } => {
                write!(f, "invalid workspace cache {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for WorkspaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub trait WorkspaceTracker: Send + Sync {
    fn current_revision(&self) -> Result<WorkspaceRevision, WorkspaceError>;
    fn delta(&self, from: &RevisionId, to: &RevisionId) -> Result<WorkspaceDelta, WorkspaceError>;
}

#[derive(Clone)]
pub struct WorkspaceTrackerImpl {
    inner: Arc<TrackerInner>,
}

struct TrackerInner {
    workspace: PathBuf,
    cache_path: PathBuf,
    backend: Backend,
    state: Mutex<RevisionCache>,
}

#[derive(Clone)]
pub(crate) enum Backend {
    Git,
    Filesystem,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ManifestEntry {
    pub path: PathBuf,
    pub kind: EntryKind,
    pub executable: bool,
    pub size: u64,
    pub content_hash: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) enum EntryKind {
    File,
    Symlink,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    head: Option<String>,
    entries: Vec<ManifestEntry>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct RevisionCache {
    sequence: u64,
    current: Option<RevisionId>,
    manifests: BTreeMap<String, Manifest>,
}

impl WorkspaceTrackerImpl {
    pub fn open(workspace: impl AsRef<Path>) -> Result<Self, WorkspaceError> {
        let supplied = workspace.as_ref();
        if !supplied.is_dir() {
            return Err(WorkspaceError::InvalidWorkspace(supplied.to_path_buf()));
        }
        let workspace = supplied
            .canonicalize()
            .map_err(|source| WorkspaceError::Io {
                path: supplied.to_path_buf(),
                source,
            })?;
        let cache_path = workspace.join(".mh/workspace/revision-cache.json");
        let backend = detect_backend(&workspace)?;
        let state = if cache_path.exists() {
            let bytes = fs::read(&cache_path).map_err(|source| WorkspaceError::Io {
                path: cache_path.clone(),
                source,
            })?;
            serde_json::from_slice(&bytes).map_err(|e| WorkspaceError::CorruptCache {
                path: cache_path.clone(),
                message: e.to_string(),
            })?
        } else {
            RevisionCache::default()
        };
        Ok(Self {
            inner: Arc::new(TrackerInner {
                workspace,
                cache_path,
                backend,
                state: Mutex::new(state),
            }),
        })
    }

    pub fn workspace(&self) -> &Path {
        &self.inner.workspace
    }
}

impl WorkspaceTracker for WorkspaceTrackerImpl {
    fn current_revision(&self) -> Result<WorkspaceRevision, WorkspaceError> {
        let manifest = capture_manifest(&self.inner.workspace, &self.inner.backend)?;
        let id = manifest_id(&manifest);
        let mut state = self
            .inner
            .state
            .lock()
            .expect("workspace tracker mutex poisoned");
        let parent = state.current.clone();
        let changed_paths = match parent.as_ref() {
            Some(old) if old != &id => {
                manifest_delta(state.manifests.get(&old.0), Some(&manifest)).all_paths()
            }
            None => manifest
                .entries
                .iter()
                .map(|entry| entry.path.clone())
                .collect(),
            _ => Vec::new(),
        };
        if state.current.as_ref() != Some(&id) {
            state.sequence = state.sequence.saturating_add(1);
            state.manifests.insert(id.0.clone(), manifest);
            state.current = Some(id.clone());
            persist_cache(&self.inner.cache_path, &state)?;
        } else if !state.manifests.contains_key(&id.0) {
            state.manifests.insert(id.0.clone(), manifest);
            persist_cache(&self.inner.cache_path, &state)?;
        }
        Ok(WorkspaceRevision {
            id,
            sequence: state.sequence,
            parent: parent.filter(|p| p != state.current.as_ref().unwrap()),
            changed_paths,
            detected_at_ms: now_ms(),
        })
    }

    fn delta(&self, from: &RevisionId, to: &RevisionId) -> Result<WorkspaceDelta, WorkspaceError> {
        let state = self
            .inner
            .state
            .lock()
            .expect("workspace tracker mutex poisoned");
        let from_manifest = state
            .manifests
            .get(&from.0)
            .ok_or_else(|| WorkspaceError::UnknownRevision(from.clone()))?;
        let to_manifest = state
            .manifests
            .get(&to.0)
            .ok_or_else(|| WorkspaceError::UnknownRevision(to.clone()))?;
        let paths = manifest_delta(Some(from_manifest), Some(to_manifest));
        Ok(WorkspaceDelta {
            from: from.clone(),
            to: to.clone(),
            added: paths.added,
            modified: paths.modified,
            deleted: paths.deleted,
        })
    }
}

struct DeltaPaths {
    added: Vec<PathBuf>,
    modified: Vec<PathBuf>,
    deleted: Vec<PathBuf>,
}
impl DeltaPaths {
    fn all_paths(&self) -> Vec<PathBuf> {
        let mut paths = self
            .added
            .iter()
            .chain(&self.modified)
            .chain(&self.deleted)
            .cloned()
            .collect::<Vec<_>>();
        paths.sort();
        paths
    }
}

fn manifest_delta(from: Option<&Manifest>, to: Option<&Manifest>) -> DeltaPaths {
    let old: BTreeMap<_, _> = from
        .into_iter()
        .flat_map(|m| &m.entries)
        .map(|e| (&e.path, e))
        .collect();
    let new: BTreeMap<_, _> = to
        .into_iter()
        .flat_map(|m| &m.entries)
        .map(|e| (&e.path, e))
        .collect();
    let added = new
        .keys()
        .filter(|p| !old.contains_key(*p))
        .map(|p| (*p).clone())
        .collect();
    let deleted = old
        .keys()
        .filter(|p| !new.contains_key(*p))
        .map(|p| (*p).clone())
        .collect();
    let modified = new
        .iter()
        .filter(|(p, entry)| old.get(*p).is_some_and(|old_entry| old_entry != *entry))
        .map(|(p, _)| (*p).clone())
        .collect();
    DeltaPaths {
        added,
        modified,
        deleted,
    }
}

fn persist_cache(path: &Path, state: &RevisionCache) -> Result<(), WorkspaceError> {
    let parent = path.parent().expect("cache path has parent");
    fs::create_dir_all(parent).map_err(|source| WorkspaceError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let bytes = serde_json::to_vec(state).map_err(|e| WorkspaceError::CorruptCache {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    let tmp = path.with_extension("json.tmp");
    let mut file = fs::File::create(&tmp).map_err(|source| WorkspaceError::Io {
        path: tmp.clone(),
        source,
    })?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| WorkspaceError::Io {
            path: tmp.clone(),
            source,
        })?;
    fs::rename(&tmp, path).map_err(|source| WorkspaceError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn detect_backend(workspace: &Path) -> Result<Backend, WorkspaceError> {
    let output = git_command(workspace)
        .args(["rev-parse", "--show-toplevel"])
        .output();
    let Ok(output) = output else {
        return Ok(Backend::Filesystem);
    };
    if !output.status.success() {
        return Ok(Backend::Filesystem);
    }
    let root = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let root = root
        .canonicalize()
        .map_err(|source| WorkspaceError::Io { path: root, source })?;
    if root != workspace {
        return Ok(Backend::Filesystem);
    }
    Ok(Backend::Git)
}

fn git_head(workspace: &Path) -> Result<String, WorkspaceError> {
    let output = git_command(workspace)
        .args(["rev-parse", "--verify", "HEAD"])
        .output()
        .map_err(|e| WorkspaceError::Git {
            operation: "rev-parse HEAD",
            message: e.to_string(),
        })?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    } else {
        Ok("unborn".to_owned())
    }
}

fn capture_manifest(workspace: &Path, backend: &Backend) -> Result<Manifest, WorkspaceError> {
    let (head, paths) = match backend {
        Backend::Git => {
            let head = git_head(workspace)?;
            let tracked = git_head_paths(workspace)?;
            let all = walk_paths(workspace, false)?;
            let mut selected = tracked;
            for path in all {
                if selected.contains(&path) || !git_ignored(workspace, &path)? {
                    selected.insert(path);
                }
            }
            (Some(head), selected)
        }
        Backend::Filesystem => (None, walk_paths(workspace, true)?),
    };
    let mut entries = Vec::with_capacity(paths.len());
    for path in paths {
        let absolute = workspace.join(&path);
        let Ok(metadata) = fs::symlink_metadata(&absolute) else {
            continue;
        };
        // A path can disappear between the directory walk and this read: an
        // inspector capturing a revision runs concurrently with the task that
        // is rewriting the workspace. A vanished path is simply not part of
        // this revision, so skip it rather than failing a read-only caller.
        let (kind, bytes) = if metadata.file_type().is_symlink() {
            match fs::read_link(&absolute) {
                Ok(target) => (EntryKind::Symlink, path_bytes(&target)),
                Err(source) if is_vanished(&source) => continue,
                Err(source) => {
                    return Err(WorkspaceError::Io {
                        path: absolute,
                        source,
                    });
                }
            }
        } else if metadata.is_file() {
            match fs::read(&absolute) {
                Ok(bytes) => (EntryKind::File, bytes),
                Err(source) if is_vanished(&source) => continue,
                Err(source) => {
                    return Err(WorkspaceError::Io {
                        path: absolute,
                        source,
                    });
                }
            }
        } else {
            continue;
        };
        let executable = is_executable(&metadata);
        let content_hash = format!("{:x}", Sha256::digest(&bytes));
        entries.push(ManifestEntry {
            path,
            kind,
            executable,
            size: bytes.len() as u64,
            content_hash,
        });
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Manifest { head, entries })
}

pub(crate) fn capture_entries(workspace: &Path) -> Result<Vec<ManifestEntry>, WorkspaceError> {
    let backend = detect_backend(workspace)?;
    Ok(capture_manifest(workspace, &backend)?.entries)
}

fn manifest_id(manifest: &Manifest) -> RevisionId {
    let mut hash = Sha256::new();
    hash.update(b"mh-workspace-v1\0");
    if let Some(head) = &manifest.head {
        hash.update(head.as_bytes());
    }
    hash.update([0]);
    for entry in &manifest.entries {
        hash.update(path_bytes(&entry.path));
        hash.update([0, entry.kind as u8, u8::from(entry.executable)]);
        hash.update(entry.size.to_le_bytes());
        hash.update(entry.content_hash.as_bytes());
        hash.update([0]);
    }
    RevisionId(format!("sha256:{:x}", hash.finalize()))
}

fn git_head_paths(workspace: &Path) -> Result<BTreeSet<PathBuf>, WorkspaceError> {
    let output = git_command(workspace)
        .args(["ls-tree", "-r", "-z", "--name-only", "HEAD"])
        .output()
        .map_err(|e| WorkspaceError::Git {
            operation: "ls-tree",
            message: e.to_string(),
        })?;
    if !output.status.success() {
        return Ok(BTreeSet::new());
    }
    Ok(output
        .stdout
        .split(|b| *b == 0)
        .filter(|b| !b.is_empty())
        .map(bytes_path)
        .filter(|p| !excluded_always(p))
        .collect())
}

fn git_ignored(workspace: &Path, path: &Path) -> Result<bool, WorkspaceError> {
    let status = git_command(workspace)
        .args(["check-ignore", "--no-index", "-q", "--"])
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| WorkspaceError::Git {
            operation: "check-ignore",
            message: e.to_string(),
        })?;
    match status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(WorkspaceError::Git {
            operation: "check-ignore",
            message: format!("exit status {status}"),
        }),
    }
}

fn git_command(workspace: &Path) -> Command {
    let mut command = Command::new("git");
    command
        .current_dir(workspace)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE");
    command
}

fn walk_paths(
    workspace: &Path,
    fallback_excludes: bool,
) -> Result<BTreeSet<PathBuf>, WorkspaceError> {
    fn visit(
        root: &Path,
        relative: &Path,
        fallback: bool,
        out: &mut BTreeSet<PathBuf>,
    ) -> Result<(), WorkspaceError> {
        let dir = root.join(relative);
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            // The task under inspection may remove a directory between the
            // parent listing and this descent.
            Err(source) if is_vanished(&source) => return Ok(()),
            Err(source) => return Err(WorkspaceError::Io { path: dir, source }),
        };
        for item in entries {
            let item = match item {
                Ok(item) => item,
                Err(source) if is_vanished(&source) => continue,
                Err(source) => {
                    return Err(WorkspaceError::Io {
                        path: dir.clone(),
                        source,
                    });
                }
            };
            let path = relative.join(item.file_name());
            if excluded_always(&path) || (fallback && excluded_fallback(&path)) {
                continue;
            }
            let Ok(ty) = item.file_type() else {
                continue;
            };
            if ty.is_dir() {
                visit(root, &path, fallback, out)?;
            } else if ty.is_file() || ty.is_symlink() {
                out.insert(path);
            }
        }
        Ok(())
    }
    let mut out = BTreeSet::new();
    visit(workspace, Path::new(""), fallback_excludes, &mut out)?;
    Ok(out)
}

/// Whether an I/O error means the path is simply no longer there.
///
/// Revision capture races the task it observes by design: `mh inspect` and a
/// running agent look at the same tree. A path that disappeared is not part of
/// the revision, which is different from a workspace we cannot read.
fn is_vanished(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
    )
}

fn excluded_always(path: &Path) -> bool {
    path.components()
        .next()
        .is_some_and(|c| c.as_os_str() == ".mh" || c.as_os_str() == ".git")
}
fn excluded_fallback(path: &Path) -> bool {
    path.components()
        .next()
        .is_some_and(|c| c.as_os_str() == "target" || c.as_os_str() == "node_modules")
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(unix)]
fn is_executable(metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o111 != 0
}
#[cfg(not(unix))]
fn is_executable(_: &fs::Metadata) -> bool {
    false
}
#[cfg(unix)]
fn path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes().to_vec()
}
#[cfg(not(unix))]
fn path_bytes(path: &Path) -> Vec<u8> {
    path.to_string_lossy().as_bytes().to_vec()
}
#[cfg(unix)]
fn bytes_path(bytes: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStrExt;
    PathBuf::from(std::ffi::OsStr::from_bytes(bytes))
}
#[cfg(not(unix))]
fn bytes_path(bytes: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(bytes).as_ref())
}
