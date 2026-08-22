use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::identity::{ExecutionId, IsolatedWorkspaceId, RevisionId, TaskId};
use crate::workspace::{
    EntryKind, ManifestEntry, WorkspaceError, WorkspaceTracker, WorkspaceTrackerImpl,
    capture_entries,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IsolatedWorkspaceRecord {
    pub id: IsolatedWorkspaceId,
    pub owner_task_id: TaskId,
    pub owner_execution_id: ExecutionId,
    pub path: PathBuf,
    pub base_revision: RevisionId,
    pub final_revision: RevisionId,
    pub changed_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationResult {
    Applied {
        previous_revision: RevisionId,
        current_revision: RevisionId,
        changed_paths: Vec<PathBuf>,
    },
    Conflict {
        child_base: RevisionId,
        parent_current: RevisionId,
        paths: Vec<PathBuf>,
    },
}

#[derive(Debug)]
pub enum IsolationError {
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    Git {
        operation: &'static str,
        path: PathBuf,
        message: String,
    },
    Workspace(WorkspaceError),
    UnsupportedWorkspace {
        path: PathBuf,
        operation: &'static str,
    },
    NotFound(IsolatedWorkspaceId),
    AlreadyExists(IsolatedWorkspaceId),
    RevisionMismatch {
        path: PathBuf,
        expected: RevisionId,
        actual: RevisionId,
    },
    InvalidRecord {
        path: PathBuf,
        message: String,
    },
}

impl fmt::Display for IsolationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(f, "{operation} failed at {}: {source}", path.display()),
            Self::Git {
                operation,
                path,
                message,
            } => write!(f, "git {operation} failed in {}: {message}", path.display()),
            Self::Workspace(error) => write!(f, "workspace operation failed: {error}"),
            Self::UnsupportedWorkspace { path, operation } => write!(
                f,
                "isolated write operation {operation} requires a Git-root workspace: {}",
                path.display()
            ),
            Self::NotFound(id) => write!(f, "isolated workspace {} was not found", id.0),
            Self::AlreadyExists(id) => write!(f, "isolated workspace {} already exists", id.0),
            Self::RevisionMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "workspace revision mismatch at {}: expected {}, found {}",
                path.display(),
                expected.0,
                actual.0
            ),
            Self::InvalidRecord { path, message } => {
                write!(
                    f,
                    "invalid isolated workspace record {}: {message}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for IsolationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Workspace(source) => Some(source),
            _ => None,
        }
    }
}

impl From<WorkspaceError> for IsolationError {
    fn from(value: WorkspaceError) -> Self {
        Self::Workspace(value)
    }
}

#[derive(Clone)]
pub struct IsolationStore {
    inner: Arc<StoreInner>,
}

struct StoreInner {
    parent: PathBuf,
    root: PathBuf,
    lock: Mutex<()>,
}
#[derive(Clone)]
struct EntryData {
    manifest: ManifestEntry,
    bytes: Vec<u8>,
}

impl IsolationStore {
    pub fn open(parent: impl AsRef<Path>) -> Result<Self, IsolationError> {
        let supplied = parent.as_ref();
        if !supplied.is_dir() {
            return Err(IsolationError::Io {
                operation: "open workspace",
                path: supplied.to_path_buf(),
                source: io::Error::new(io::ErrorKind::NotFound, "workspace is not a directory"),
            });
        }
        let parent = supplied
            .canonicalize()
            .map_err(|source| IsolationError::Io {
                operation: "canonicalize workspace",
                path: supplied.to_path_buf(),
                source,
            })?;
        require_git_root(&parent)?;
        let root = parent.join(".mh/workspaces");
        fs::create_dir_all(&root).map_err(|source| IsolationError::Io {
            operation: "create workspace store",
            path: root.clone(),
            source,
        })?;
        Ok(Self {
            inner: Arc::new(StoreInner {
                parent,
                root,
                lock: Mutex::new(()),
            }),
        })
    }

    pub fn create(
        &self,
        id: IsolatedWorkspaceId,
        owner_task_id: TaskId,
        owner_execution_id: ExecutionId,
        base_revision: RevisionId,
    ) -> Result<IsolatedWorkspaceRecord, IsolationError> {
        let _guard = self.inner.lock.lock().expect("isolation mutex poisoned");
        let dir = self.workspace_dir(id);
        if dir.exists() {
            return Err(IsolationError::AlreadyExists(id));
        }

        let parent_tracker = WorkspaceTrackerImpl::open(&self.inner.parent)?;
        let current = parent_tracker.current_revision()?;
        if current.id != base_revision {
            return Err(IsolationError::RevisionMismatch {
                path: self.inner.parent.clone(),
                expected: base_revision,
                actual: current.id,
            });
        }
        let source_entries = read_entries(&self.inner.parent)?;

        fs::create_dir_all(&dir).map_err(|source| IsolationError::Io {
            operation: "create isolated workspace directory",
            path: dir.clone(),
            source,
        })?;
        let path = self.worktree_path(id);
        if let Err(error) = git_success(
            &self.inner.parent,
            "worktree add",
            Command::new("git")
                .current_dir(&self.inner.parent)
                .env_remove("GIT_DIR")
                .env_remove("GIT_WORK_TREE")
                .env_remove("GIT_INDEX_FILE")
                .args(["worktree", "add", "--detach", "--force"])
                .arg(&path)
                .arg("HEAD"),
        ) {
            let _ = fs::remove_dir_all(&dir);
            return Err(error);
        }

        let result = (|| {
            replace_workspace_entries(&path, &source_entries)?;
            let child_tracker = WorkspaceTrackerImpl::open(&path)?;
            let child_revision = child_tracker.current_revision()?;
            if child_revision.id != current.id {
                return Err(IsolationError::RevisionMismatch {
                    path: path.clone(),
                    expected: current.id,
                    actual: child_revision.id,
                });
            }
            let record = IsolatedWorkspaceRecord {
                id,
                owner_task_id,
                owner_execution_id,
                path: path.clone(),
                base_revision: child_revision.id.clone(),
                final_revision: child_revision.id,
                changed_paths: Vec::new(),
            };
            self.persist(&record)?;
            Ok(record)
        })();
        if result.is_err() {
            let _ = remove_worktree(&self.inner.parent, &path);
            let _ = fs::remove_dir_all(&dir);
        }
        result
    }

    pub fn finalize(
        &self,
        id: IsolatedWorkspaceId,
    ) -> Result<IsolatedWorkspaceRecord, IsolationError> {
        let _guard = self.inner.lock.lock().expect("isolation mutex poisoned");
        self.finalize_locked(id)
    }

    pub fn load(&self, id: IsolatedWorkspaceId) -> Result<IsolatedWorkspaceRecord, IsolationError> {
        let _guard = self.inner.lock.lock().expect("isolation mutex poisoned");
        self.load_locked(id)
    }

    pub fn integrate(&self, id: IsolatedWorkspaceId) -> Result<IntegrationResult, IsolationError> {
        let _guard = self.inner.lock.lock().expect("isolation mutex poisoned");
        let record = self.load_locked(id)?;
        let final_revision = record.final_revision.clone();

        let parent_tracker = WorkspaceTrackerImpl::open(&self.inner.parent)?;
        let parent_current = parent_tracker.current_revision()?;
        let parent_delta = parent_tracker.delta(&record.base_revision, &parent_current.id)?;
        let mut parent_paths = parent_delta.added;
        parent_paths.extend(parent_delta.modified);
        parent_paths.extend(parent_delta.deleted);
        parent_paths.sort();
        parent_paths.dedup();

        let conflicts = conflicting_child_paths(&record.changed_paths, &parent_paths);
        if !conflicts.is_empty() {
            return Ok(IntegrationResult::Conflict {
                child_base: record.base_revision,
                parent_current: parent_current.id,
                paths: conflicts,
            });
        }

        let child_tracker = WorkspaceTrackerImpl::open(&record.path)?;
        let actual_final = child_tracker.current_revision()?;
        if actual_final.id != final_revision {
            return Err(IsolationError::RevisionMismatch {
                path: record.path,
                expected: final_revision,
                actual: actual_final.id,
            });
        }

        let child_entries = read_entries(&record.path)?;
        validate_destinations(&self.inner.parent, &record.changed_paths)?;
        let parent_backup = read_overlapping_entries(&self.inner.parent, &record.changed_paths)?;
        let apply_result = apply_delta(&self.inner.parent, &record.changed_paths, &child_entries);
        if let Err(error) = apply_result {
            let _ = restore_delta(&self.inner.parent, &record.changed_paths, &parent_backup);
            return Err(error);
        }

        let integrated = match parent_tracker.current_revision() {
            Ok(revision) => revision,
            Err(error) => {
                let _ = restore_delta(&self.inner.parent, &record.changed_paths, &parent_backup);
                return Err(error.into());
            }
        };
        if let Err(error) = self.discard_locked(id, &record.path) {
            let rollback = restore_delta(&self.inner.parent, &record.changed_paths, &parent_backup);
            return match rollback {
                Ok(()) => Err(error),
                Err(rollback_error) => Err(IsolationError::InvalidRecord {
                    path: self.inner.parent.clone(),
                    message: format!(
                        "workspace cleanup failed ({error}); rollback also failed ({rollback_error})"
                    ),
                }),
            };
        }
        Ok(IntegrationResult::Applied {
            previous_revision: parent_current.id,
            current_revision: integrated.id,
            changed_paths: record.changed_paths,
        })
    }

    pub fn discard(&self, id: IsolatedWorkspaceId) -> Result<(), IsolationError> {
        let _guard = self.inner.lock.lock().expect("isolation mutex poisoned");
        let record = self.load_locked(id)?;
        self.discard_locked(id, &record.path)
    }

    pub fn cleanup_owner(&self, task_id: TaskId) -> Result<(), IsolationError> {
        let _guard = self.inner.lock.lock().expect("isolation mutex poisoned");
        let mut owned = Vec::new();
        for item in fs::read_dir(&self.inner.root).map_err(|source| IsolationError::Io {
            operation: "list isolated workspaces",
            path: self.inner.root.clone(),
            source,
        })? {
            let item = item.map_err(|source| IsolationError::Io {
                operation: "read isolated workspace entry",
                path: self.inner.root.clone(),
                source,
            })?;
            if !item
                .file_type()
                .map_err(|source| IsolationError::Io {
                    operation: "inspect isolated workspace entry",
                    path: item.path(),
                    source,
                })?
                .is_dir()
            {
                continue;
            }
            let Some(id) = item
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u64>().ok())
            else {
                continue;
            };
            match self.load_locked(IsolatedWorkspaceId(id)) {
                Ok(record) if record.owner_task_id == task_id => owned.push(record),
                Ok(_) => {}
                Err(IsolationError::NotFound(_)) => {}
                Err(error) => return Err(error),
            }
        }
        for record in owned {
            self.discard_locked(record.id, &record.path)?;
        }
        Ok(())
    }

    fn finalize_locked(
        &self,
        id: IsolatedWorkspaceId,
    ) -> Result<IsolatedWorkspaceRecord, IsolationError> {
        let mut record = self.load_locked(id)?;
        let tracker = WorkspaceTrackerImpl::open(&record.path)?;
        let revision = tracker.current_revision()?;
        let delta = tracker.delta(&record.base_revision, &revision.id)?;
        let mut changed_paths = delta.added;
        changed_paths.extend(delta.modified);
        changed_paths.extend(delta.deleted);
        changed_paths.sort();
        changed_paths.dedup();
        record.final_revision = revision.id;
        record.changed_paths = changed_paths;
        self.persist(&record)?;
        Ok(record)
    }

    fn load_locked(
        &self,
        id: IsolatedWorkspaceId,
    ) -> Result<IsolatedWorkspaceRecord, IsolationError> {
        let path = self.record_path(id);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(IsolationError::NotFound(id));
            }
            Err(source) => {
                return Err(IsolationError::Io {
                    operation: "read isolated workspace record",
                    path,
                    source,
                });
            }
        };
        let record: IsolatedWorkspaceRecord =
            serde_json::from_slice(&bytes).map_err(|error| IsolationError::InvalidRecord {
                path: path.clone(),
                message: error.to_string(),
            })?;
        if record.id != id || record.path != self.worktree_path(id) {
            return Err(IsolationError::InvalidRecord {
                path,
                message: "record identity or workspace path does not match its location".to_owned(),
            });
        }
        Ok(record)
    }

    fn persist(&self, record: &IsolatedWorkspaceRecord) -> Result<(), IsolationError> {
        let path = self.record_path(record.id);
        let bytes = serde_json::to_vec(record).map_err(|error| IsolationError::InvalidRecord {
            path: path.clone(),
            message: error.to_string(),
        })?;
        write_durable(&path, &bytes)
    }

    fn discard_locked(&self, id: IsolatedWorkspaceId, path: &Path) -> Result<(), IsolationError> {
        remove_worktree(&self.inner.parent, path)?;
        let dir = self.workspace_dir(id);
        match fs::remove_dir_all(&dir) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(source) => Err(IsolationError::Io {
                operation: "remove isolated workspace metadata",
                path: dir,
                source,
            }),
        }
    }

    fn workspace_dir(&self, id: IsolatedWorkspaceId) -> PathBuf {
        self.inner.root.join(id.0.to_string())
    }

    fn record_path(&self, id: IsolatedWorkspaceId) -> PathBuf {
        self.workspace_dir(id).join("record.json")
    }

    fn worktree_path(&self, id: IsolatedWorkspaceId) -> PathBuf {
        self.workspace_dir(id).join("worktree")
    }
}

fn require_git_root(path: &Path) -> Result<(), IsolationError> {
    let output = Command::new("git")
        .current_dir(path)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|source| IsolationError::Io {
            operation: "run git rev-parse",
            path: path.to_path_buf(),
            source,
        })?;
    if !output.status.success() {
        return Err(IsolationError::UnsupportedWorkspace {
            path: path.to_path_buf(),
            operation: "open",
        });
    }
    let reported = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let root = reported
        .canonicalize()
        .map_err(|source| IsolationError::Io {
            operation: "canonicalize Git root",
            path: reported,
            source,
        })?;
    if root != path {
        return Err(IsolationError::UnsupportedWorkspace {
            path: path.to_path_buf(),
            operation: "open",
        });
    }
    Ok(())
}

fn read_entries(workspace: &Path) -> Result<Vec<EntryData>, IsolationError> {
    capture_entries(workspace)?
        .into_iter()
        .map(|manifest| {
            let path = workspace.join(&manifest.path);
            let bytes = match manifest.kind {
                EntryKind::File => fs::read(&path).map_err(|source| IsolationError::Io {
                    operation: "read workspace file",
                    path: path.clone(),
                    source,
                })?,
                EntryKind::Symlink => {
                    path_bytes(&fs::read_link(&path).map_err(|source| IsolationError::Io {
                        operation: "read workspace symlink",
                        path: path.clone(),
                        source,
                    })?)
                }
            };
            Ok(EntryData { manifest, bytes })
        })
        .collect()
}

fn read_overlapping_entries(
    workspace: &Path,
    changed_paths: &[PathBuf],
) -> Result<Vec<EntryData>, IsolationError> {
    Ok(read_entries(workspace)?
        .into_iter()
        .filter(|entry| {
            changed_paths
                .iter()
                .any(|path| paths_overlap(&entry.manifest.path, path))
        })
        .collect())
}

fn replace_workspace_entries(
    workspace: &Path,
    desired: &[EntryData],
) -> Result<(), IsolationError> {
    let desired_paths: BTreeSet<_> = desired
        .iter()
        .map(|entry| entry.manifest.path.clone())
        .collect();
    let mut current = capture_entries(workspace)?;
    current.sort_by(|a, b| {
        path_depth(&b.path)
            .cmp(&path_depth(&a.path))
            .then_with(|| b.path.cmp(&a.path))
    });
    for entry in current {
        if !desired_paths.contains(&entry.path) {
            remove_if_exists(&workspace.join(entry.path))?;
        }
    }
    for entry in desired {
        restore_entry(workspace, entry)?;
    }
    Ok(())
}

fn apply_delta(
    workspace: &Path,
    changed_paths: &[PathBuf],
    child_entries: &[EntryData],
) -> Result<(), IsolationError> {
    let child: BTreeMap<_, _> = child_entries
        .iter()
        .map(|entry| (entry.manifest.path.clone(), entry))
        .collect();
    let mut deletions = changed_paths
        .iter()
        .filter(|path| !child.contains_key(*path))
        .cloned()
        .collect::<Vec<_>>();
    deletions.sort_by(|a, b| path_depth(b).cmp(&path_depth(a)).then_with(|| b.cmp(a)));
    for path in deletions {
        remove_if_exists(&workspace.join(path))?;
    }
    for path in changed_paths {
        if let Some(entry) = child.get(path) {
            restore_entry(workspace, entry)?;
        }
    }
    Ok(())
}

fn restore_delta(
    workspace: &Path,
    changed_paths: &[PathBuf],
    backup: &[EntryData],
) -> Result<(), IsolationError> {
    let mut paths = changed_paths.to_vec();
    paths.sort_by(|a, b| path_depth(b).cmp(&path_depth(a)).then_with(|| b.cmp(a)));
    for path in paths {
        remove_if_exists(&workspace.join(path))?;
    }
    let mut entries = backup.to_vec();
    entries.sort_by(|a, b| a.manifest.path.cmp(&b.manifest.path));
    for entry in &entries {
        restore_entry(workspace, entry)?;
    }
    Ok(())
}

fn restore_entry(workspace: &Path, entry: &EntryData) -> Result<(), IsolationError> {
    if !safe_relative(&entry.manifest.path) {
        return Err(IsolationError::InvalidRecord {
            path: entry.manifest.path.clone(),
            message: "unsafe workspace path".to_owned(),
        });
    }
    create_safe_parents(workspace, &entry.manifest.path)?;
    let destination = workspace.join(&entry.manifest.path);
    remove_if_exists(&destination)?;
    match entry.manifest.kind {
        EntryKind::File => {
            fs::write(&destination, &entry.bytes).map_err(|source| IsolationError::Io {
                operation: "write workspace file",
                path: destination.clone(),
                source,
            })?;
            set_executable(&destination, entry.manifest.executable)?;
        }
        EntryKind::Symlink => create_symlink(&bytes_path(&entry.bytes), &destination)?,
    }
    Ok(())
}

fn validate_destinations(workspace: &Path, paths: &[PathBuf]) -> Result<(), IsolationError> {
    for path in paths {
        if !safe_relative(path) {
            return Err(IsolationError::InvalidRecord {
                path: path.clone(),
                message: "unsafe changed path".to_owned(),
            });
        }
        let mut current = workspace.to_path_buf();
        let mut components = path.components().peekable();
        while let Some(component) = components.next() {
            if components.peek().is_none() {
                break;
            }
            current.push(component.as_os_str());
            match fs::symlink_metadata(&current) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(IsolationError::InvalidRecord {
                        path: current,
                        message: "destination ancestor is a symlink".to_owned(),
                    });
                }
                Ok(metadata) if metadata.is_dir() => {}
                Ok(_) => break,
                Err(error) if error.kind() == io::ErrorKind::NotFound => break,
                Err(source) => {
                    return Err(IsolationError::Io {
                        operation: "inspect destination ancestor",
                        path: current,
                        source,
                    });
                }
            }
        }
    }
    Ok(())
}

fn create_safe_parents(workspace: &Path, relative: &Path) -> Result<(), IsolationError> {
    let Some(parent) = relative.parent() else {
        return Ok(());
    };
    let mut current = workspace.to_path_buf();
    for component in parent.components() {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
            Ok(_) => {
                return Err(IsolationError::InvalidRecord {
                    path: current,
                    message: "workspace path ancestor is not a real directory".to_owned(),
                });
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                fs::create_dir(&current).map_err(|source| IsolationError::Io {
                    operation: "create workspace directory",
                    path: current.clone(),
                    source,
                })?;
            }
            Err(source) => {
                return Err(IsolationError::Io {
                    operation: "inspect workspace path ancestor",
                    path: current,
                    source,
                });
            }
        }
    }
    Ok(())
}

fn remove_if_exists(path: &Path) -> Result<(), IsolationError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            fs::remove_dir_all(path).map_err(|source| IsolationError::Io {
                operation: "remove workspace directory",
                path: path.to_path_buf(),
                source,
            })
        }
        Ok(_) => fs::remove_file(path).map_err(|source| IsolationError::Io {
            operation: "remove workspace entry",
            path: path.to_path_buf(),
            source,
        }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(IsolationError::Io {
            operation: "inspect workspace entry",
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn conflicting_child_paths(child: &[PathBuf], parent: &[PathBuf]) -> Vec<PathBuf> {
    child
        .iter()
        .filter(|child_path| {
            parent
                .iter()
                .any(|parent_path| paths_overlap(child_path, parent_path))
        })
        .cloned()
        .collect()
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
}

fn safe_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        && !path.starts_with(".mh")
        && !path.starts_with(".git")
}

fn path_depth(path: &Path) -> usize {
    path.components().count()
}

fn write_durable(path: &Path, bytes: &[u8]) -> Result<(), IsolationError> {
    let parent = path.parent().expect("metadata path has parent");
    fs::create_dir_all(parent).map_err(|source| IsolationError::Io {
        operation: "create metadata directory",
        path: parent.to_path_buf(),
        source,
    })?;
    let temp = path.with_extension("json.tmp");
    let mut file = fs::File::create(&temp).map_err(|source| IsolationError::Io {
        operation: "create metadata file",
        path: temp.clone(),
        source,
    })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| IsolationError::Io {
            operation: "write metadata file",
            path: temp.clone(),
            source,
        })?;
    fs::rename(&temp, path).map_err(|source| IsolationError::Io {
        operation: "replace metadata file",
        path: path.to_path_buf(),
        source,
    })
}

fn remove_worktree(parent: &Path, path: &Path) -> Result<(), IsolationError> {
    if !path.exists() {
        return Ok(());
    }
    git_success(
        parent,
        "worktree remove",
        Command::new("git")
            .current_dir(parent)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .args(["worktree", "remove", "--force"])
            .arg(path),
    )
}

fn git_success(
    path: &Path,
    operation: &'static str,
    command: &mut Command,
) -> Result<(), IsolationError> {
    let output = command.output().map_err(|source| IsolationError::Io {
        operation: "run git",
        path: path.to_path_buf(),
        source,
    })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(IsolationError::Git {
            operation,
            path: path.to_path_buf(),
            message: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        })
    }
}

#[cfg(unix)]
fn set_executable(path: &Path, executable: bool) -> Result<(), IsolationError> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = fs::metadata(path)
        .map_err(|source| IsolationError::Io {
            operation: "read workspace file permissions",
            path: path.to_path_buf(),
            source,
        })?
        .permissions();
    let mut mode = permissions.mode();
    if executable {
        mode |= 0o111;
    } else {
        mode &= !0o111;
    }
    permissions.set_mode(mode);
    fs::set_permissions(path, permissions).map_err(|source| IsolationError::Io {
        operation: "set workspace file permissions",
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(not(unix))]
fn set_executable(_: &Path, _: bool) -> Result<(), IsolationError> {
    Ok(())
}

#[cfg(unix)]
fn create_symlink(target: &Path, destination: &Path) -> Result<(), IsolationError> {
    std::os::unix::fs::symlink(target, destination).map_err(|source| IsolationError::Io {
        operation: "create workspace symlink",
        path: destination.to_path_buf(),
        source,
    })
}

#[cfg(windows)]
fn create_symlink(target: &Path, destination: &Path) -> Result<(), IsolationError> {
    std::os::windows::fs::symlink_file(target, destination).map_err(|source| IsolationError::Io {
        operation: "create workspace symlink",
        path: destination.to_path_buf(),
        source,
    })
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
