use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::identity::{CheckpointId, RevisionId, TaskId};
use crate::workspace::{
    EntryKind, ManifestEntry, WorkspaceError, WorkspaceRevision, WorkspaceTracker, capture_entries,
};

const DEFAULT_BYTE_CAP: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: CheckpointId,
    pub revision: RevisionId,
    pub backend: CheckpointBackend,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointBackend {
    Git { tree: String },
    Filesystem,
}

#[derive(Debug)]
pub enum CheckpointError {
    Workspace(WorkspaceError),
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Git {
        operation: &'static str,
        message: String,
    },
    SizeLimitExceeded {
        limit: u64,
        actual: u64,
    },
    NotFound {
        task_id: TaskId,
        checkpoint_id: CheckpointId,
    },
    TaskMismatch {
        expected: TaskId,
        actual: TaskId,
    },
    InvalidCheckpoint {
        path: PathBuf,
        message: String,
    },
    RevisionMismatch {
        expected: RevisionId,
        actual: RevisionId,
    },
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workspace(error) => write!(f, "workspace error: {error}"),
            Self::Io { path, source } => {
                write!(f, "checkpoint I/O error at {}: {source}", path.display())
            }
            Self::Git { operation, message } => write!(f, "git {operation} failed: {message}"),
            Self::SizeLimitExceeded { limit, actual } => {
                write!(f, "checkpoint size {actual} exceeds byte cap {limit}")
            }
            Self::NotFound {
                task_id,
                checkpoint_id,
            } => write!(
                f,
                "checkpoint {} for task {} was not found",
                checkpoint_id.0, task_id.0
            ),
            Self::TaskMismatch { expected, actual } => write!(
                f,
                "checkpoint belongs to task {}, not task {}",
                actual.0, expected.0
            ),
            Self::InvalidCheckpoint { path, message } => {
                write!(f, "invalid checkpoint {}: {message}", path.display())
            }
            Self::RevisionMismatch { expected, actual } => write!(
                f,
                "restored revision {} does not match checkpoint revision {}",
                actual.0, expected.0
            ),
        }
    }
}

impl std::error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Workspace(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
impl From<WorkspaceError> for CheckpointError {
    fn from(value: WorkspaceError) -> Self {
        Self::Workspace(value)
    }
}

#[derive(Clone)]
pub struct CheckpointStore {
    inner: Arc<StoreInner>,
}

struct StoreInner {
    workspace: PathBuf,
    root: PathBuf,
    tracker: Arc<dyn WorkspaceTracker>,
    byte_cap: u64,
    lock: Mutex<()>,
}

#[derive(Serialize, Deserialize)]
struct StoredCheckpoint {
    checkpoint: Checkpoint,
    task_id: TaskId,
    head: Option<String>,
    entries: Vec<StoredEntry>,
}

#[derive(Serialize, Deserialize)]
struct StoredEntry {
    manifest: ManifestEntry,
    blob: String,
}

impl CheckpointStore {
    pub fn open(
        workspace: impl AsRef<Path>,
        tracker: Arc<dyn WorkspaceTracker>,
    ) -> Result<Self, CheckpointError> {
        let supplied = workspace.as_ref();
        if !supplied.is_dir() {
            return Err(CheckpointError::Workspace(
                WorkspaceError::InvalidWorkspace(supplied.to_path_buf()),
            ));
        }
        let workspace = supplied
            .canonicalize()
            .map_err(|source| CheckpointError::Io {
                path: supplied.to_path_buf(),
                source,
            })?;
        let root = workspace.join(".mh/checkpoints");
        fs::create_dir_all(&root).map_err(|source| CheckpointError::Io {
            path: root.clone(),
            source,
        })?;
        Ok(Self {
            inner: Arc::new(StoreInner {
                workspace,
                root,
                tracker,
                byte_cap: DEFAULT_BYTE_CAP,
                lock: Mutex::new(()),
            }),
        })
    }

    pub fn create(&self, task_id: TaskId) -> Result<Checkpoint, CheckpointError> {
        let _guard = self.inner.lock.lock().expect("checkpoint mutex poisoned");
        let revision = self.inner.tracker.current_revision()?;
        let entries = capture_entries(&self.inner.workspace)?;
        let actual = entries.iter().try_fold(0_u64, |total, entry| {
            total
                .checked_add(entry.size)
                .ok_or(CheckpointError::SizeLimitExceeded {
                    limit: self.inner.byte_cap,
                    actual: u64::MAX,
                })
        })?;
        if actual > self.inner.byte_cap {
            return Err(CheckpointError::SizeLimitExceeded {
                limit: self.inner.byte_cap,
                actual,
            });
        }
        let id = self.next_id(task_id)?;
        let dir = self.checkpoint_dir(task_id, id);
        let blobs = dir.join("blobs");
        fs::create_dir_all(&blobs).map_err(|source| CheckpointError::Io {
            path: blobs.clone(),
            source,
        })?;
        let mut stored_entries = Vec::with_capacity(entries.len());
        for (index, entry) in entries.into_iter().enumerate() {
            let bytes = read_entry_bytes(&self.inner.workspace.join(&entry.path), entry.kind)?;
            let blob = format!("{index:016x}");
            write_durable(&blobs.join(&blob), &bytes)?;
            stored_entries.push(StoredEntry {
                manifest: entry,
                blob,
            });
        }
        let head = git_head_if_root(&self.inner.workspace)?;
        let backend = match head.as_ref() {
            Some(_) => CheckpointBackend::Git {
                tree: git_write_tree(&self.inner.workspace, &dir)?,
            },
            None => CheckpointBackend::Filesystem,
        };
        let checkpoint = Checkpoint {
            id,
            revision: revision.id,
            backend,
        };
        let metadata = StoredCheckpoint {
            checkpoint: checkpoint.clone(),
            task_id,
            head,
            entries: stored_entries,
        };
        let bytes =
            serde_json::to_vec(&metadata).map_err(|e| CheckpointError::InvalidCheckpoint {
                path: dir.join("checkpoint.json"),
                message: e.to_string(),
            })?;
        write_durable(&dir.join("checkpoint.json"), &bytes)?;
        Ok(checkpoint)
    }

    pub fn restore(
        &self,
        task_id: TaskId,
        checkpoint: &Checkpoint,
    ) -> Result<WorkspaceRevision, CheckpointError> {
        let _guard = self.inner.lock.lock().expect("checkpoint mutex poisoned");
        let dir = self.checkpoint_dir(task_id, checkpoint.id);
        let metadata_path = dir.join("checkpoint.json");
        if !metadata_path.exists() {
            return Err(CheckpointError::NotFound {
                task_id,
                checkpoint_id: checkpoint.id,
            });
        }
        let bytes = fs::read(&metadata_path).map_err(|source| CheckpointError::Io {
            path: metadata_path.clone(),
            source,
        })?;
        let stored: StoredCheckpoint =
            serde_json::from_slice(&bytes).map_err(|e| CheckpointError::InvalidCheckpoint {
                path: metadata_path.clone(),
                message: e.to_string(),
            })?;
        if stored.task_id != task_id {
            return Err(CheckpointError::TaskMismatch {
                expected: task_id,
                actual: stored.task_id,
            });
        }
        if &stored.checkpoint != checkpoint {
            return Err(CheckpointError::InvalidCheckpoint {
                path: metadata_path,
                message: "checkpoint metadata does not match supplied handle".to_owned(),
            });
        }
        if let Some(expected_head) = &stored.head {
            let actual = git_head(&self.inner.workspace)?;
            if &actual != expected_head {
                return Err(CheckpointError::InvalidCheckpoint {
                    path: dir,
                    message: format!("Git HEAD changed from {expected_head} to {actual}"),
                });
            }
        }
        validate_blobs(&dir, &stored.entries, self.inner.byte_cap)?;
        validate_restore_ancestors(&self.inner.workspace, &stored.entries)?;
        let desired = stored
            .entries
            .iter()
            .map(|e| e.manifest.path.clone())
            .collect::<std::collections::BTreeSet<_>>();
        for current in capture_entries(&self.inner.workspace)? {
            if !desired.contains(&current.path) {
                remove_entry(&self.inner.workspace.join(current.path))?;
            }
        }
        for entry in &stored.entries {
            let destination = self.inner.workspace.join(&entry.manifest.path);
            create_safe_parents(&self.inner.workspace, &entry.manifest.path)?;
            remove_if_exists(&destination)?;
            let bytes = fs::read(dir.join("blobs").join(&entry.blob)).map_err(|source| {
                CheckpointError::Io {
                    path: dir.join("blobs").join(&entry.blob),
                    source,
                }
            })?;
            restore_entry(
                &destination,
                entry.manifest.kind,
                entry.manifest.executable,
                &bytes,
            )?;
        }
        let restored = self.inner.tracker.current_revision()?;
        if restored.id != checkpoint.revision {
            return Err(CheckpointError::RevisionMismatch {
                expected: checkpoint.revision.clone(),
                actual: restored.id,
            });
        }
        Ok(restored)
    }

    fn next_id(&self, task_id: TaskId) -> Result<CheckpointId, CheckpointError> {
        let task_dir = self.inner.root.join(task_id.0.to_string());
        fs::create_dir_all(&task_dir).map_err(|source| CheckpointError::Io {
            path: task_dir.clone(),
            source,
        })?;
        let counter = task_dir.join("next-id");
        let next = match fs::read_to_string(&counter) {
            Ok(value) => {
                value
                    .trim()
                    .parse::<u64>()
                    .map_err(|e| CheckpointError::InvalidCheckpoint {
                        path: counter.clone(),
                        message: e.to_string(),
                    })?
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => 1,
            Err(source) => {
                return Err(CheckpointError::Io {
                    path: counter,
                    source,
                });
            }
        };
        write_durable(&counter, next.saturating_add(1).to_string().as_bytes())?;
        Ok(CheckpointId(next))
    }

    fn checkpoint_dir(&self, task_id: TaskId, id: CheckpointId) -> PathBuf {
        self.inner
            .root
            .join(task_id.0.to_string())
            .join(id.0.to_string())
    }
}

fn validate_blobs(dir: &Path, entries: &[StoredEntry], limit: u64) -> Result<(), CheckpointError> {
    let mut total = 0_u64;
    for entry in entries {
        if !safe_relative(&entry.manifest.path) {
            return Err(CheckpointError::InvalidCheckpoint {
                path: dir.to_path_buf(),
                message: format!("unsafe path {}", entry.manifest.path.display()),
            });
        }
        let path = dir.join("blobs").join(&entry.blob);
        let bytes = fs::read(&path).map_err(|source| CheckpointError::Io {
            path: path.clone(),
            source,
        })?;
        total =
            total
                .checked_add(bytes.len() as u64)
                .ok_or(CheckpointError::SizeLimitExceeded {
                    limit,
                    actual: u64::MAX,
                })?;
        if total > limit {
            return Err(CheckpointError::SizeLimitExceeded {
                limit,
                actual: total,
            });
        }
        let hash = format!("{:x}", Sha256::digest(&bytes));
        if hash != entry.manifest.content_hash || bytes.len() as u64 != entry.manifest.size {
            return Err(CheckpointError::InvalidCheckpoint {
                path,
                message: "blob content hash or size mismatch".to_owned(),
            });
        }
    }
    Ok(())
}

fn safe_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
        && !path.starts_with(".mh")
        && !path.starts_with(".git")
}

fn validate_restore_ancestors(
    workspace: &Path,
    entries: &[StoredEntry],
) -> Result<(), CheckpointError> {
    for entry in entries {
        let mut current = workspace.to_path_buf();
        let mut components = entry.manifest.path.components().peekable();
        while let Some(component) = components.next() {
            if components.peek().is_none() {
                break;
            }
            current.push(component.as_os_str());
            match fs::symlink_metadata(&current) {
                Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                    return Err(CheckpointError::InvalidCheckpoint {
                        path: current,
                        message: "restore path ancestor is not a real directory".to_owned(),
                    });
                }
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => break,
                Err(source) => {
                    return Err(CheckpointError::Io {
                        path: current,
                        source,
                    });
                }
            }
        }
    }
    Ok(())
}

fn create_safe_parents(workspace: &Path, relative: &Path) -> Result<(), CheckpointError> {
    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let mut current = workspace.to_path_buf();
    for component in parent.components() {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(CheckpointError::InvalidCheckpoint {
                    path: current,
                    message: "restore path ancestor is not a real directory".to_owned(),
                });
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => fs::create_dir(&current)
                .map_err(|source| CheckpointError::Io {
                    path: current.clone(),
                    source,
                })?,
            Err(source) => {
                return Err(CheckpointError::Io {
                    path: current,
                    source,
                });
            }
        }
    }
    Ok(())
}

fn git_write_tree(workspace: &Path, checkpoint_dir: &Path) -> Result<String, CheckpointError> {
    let index = checkpoint_dir.join("private-index");
    let read_tree = git_command(workspace)
        .args(["read-tree", "--empty"])
        .env("GIT_INDEX_FILE", &index)
        .output()
        .map_err(|e| CheckpointError::Git {
            operation: "read-tree with private index",
            message: e.to_string(),
        })?;
    if !read_tree.status.success() {
        return Err(CheckpointError::Git {
            operation: "read-tree with private index",
            message: String::from_utf8_lossy(&read_tree.stderr).trim().to_owned(),
        });
    }
    let add = git_command(workspace)
        .args(["add", "-A", "--", ".", ":(exclude).mh", ":(exclude).git"])
        .env("GIT_INDEX_FILE", &index)
        .output()
        .map_err(|e| CheckpointError::Git {
            operation: "add with private index",
            message: e.to_string(),
        })?;
    if !add.status.success() {
        return Err(CheckpointError::Git {
            operation: "add with private index",
            message: String::from_utf8_lossy(&add.stderr).trim().to_owned(),
        });
    }
    let tree = git_command(workspace)
        .args(["write-tree"])
        .env("GIT_INDEX_FILE", &index)
        .output()
        .map_err(|e| CheckpointError::Git {
            operation: "write-tree",
            message: e.to_string(),
        })?;
    if !tree.status.success() {
        return Err(CheckpointError::Git {
            operation: "write-tree",
            message: String::from_utf8_lossy(&tree.stderr).trim().to_owned(),
        });
    }
    Ok(String::from_utf8_lossy(&tree.stdout).trim().to_owned())
}

fn git_head_if_root(workspace: &Path) -> Result<Option<String>, CheckpointError> {
    let output = match git_command(workspace)
        .args(["rev-parse", "--show-toplevel"])
        .output()
    {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    if !output.status.success() {
        return Ok(None);
    }
    let root = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim())
        .canonicalize()
        .map_err(|source| CheckpointError::Io {
            path: workspace.to_path_buf(),
            source,
        })?;
    if root != workspace {
        return Ok(None);
    }
    Ok(Some(git_head(workspace)?))
}

fn git_head(workspace: &Path) -> Result<String, CheckpointError> {
    let output = git_command(workspace)
        .args(["rev-parse", "--verify", "HEAD"])
        .output()
        .map_err(|e| CheckpointError::Git {
            operation: "rev-parse HEAD",
            message: e.to_string(),
        })?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    } else {
        Ok("unborn".to_owned())
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

fn read_entry_bytes(path: &Path, kind: EntryKind) -> Result<Vec<u8>, CheckpointError> {
    match kind {
        EntryKind::File => fs::read(path).map_err(|source| CheckpointError::Io {
            path: path.to_path_buf(),
            source,
        }),
        EntryKind::Symlink => fs::read_link(path)
            .map(|p| path_bytes(&p))
            .map_err(|source| CheckpointError::Io {
                path: path.to_path_buf(),
                source,
            }),
    }
}

fn restore_entry(
    path: &Path,
    kind: EntryKind,
    executable: bool,
    bytes: &[u8],
) -> Result<(), CheckpointError> {
    match kind {
        EntryKind::File => {
            write_durable(path, bytes)?;
            set_executable(path, executable)?;
            Ok(())
        }
        EntryKind::Symlink => create_symlink(bytes_path(bytes), path),
    }
}

fn write_durable(path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    let mut file = fs::File::create(path).map_err(|source| CheckpointError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| CheckpointError::Io {
            path: path.to_path_buf(),
            source,
        })
}
fn remove_entry(path: &Path) -> Result<(), CheckpointError> {
    remove_if_exists(path)
}
fn remove_if_exists(path: &Path) -> Result<(), CheckpointError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(value) => value,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(CheckpointError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
    .map_err(|source| CheckpointError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(unix)]
fn set_executable(path: &Path, executable: bool) -> Result<(), CheckpointError> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = fs::metadata(path)
        .map_err(|source| CheckpointError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .permissions();
    let mode = permissions.mode();
    permissions.set_mode(if executable {
        mode | 0o111
    } else {
        mode & !0o111
    });
    fs::set_permissions(path, permissions).map_err(|source| CheckpointError::Io {
        path: path.to_path_buf(),
        source,
    })
}
#[cfg(not(unix))]
fn set_executable(_: &Path, _: bool) -> Result<(), CheckpointError> {
    Ok(())
}
#[cfg(unix)]
fn create_symlink(target: PathBuf, path: &Path) -> Result<(), CheckpointError> {
    std::os::unix::fs::symlink(target, path).map_err(|source| CheckpointError::Io {
        path: path.to_path_buf(),
        source,
    })
}
#[cfg(windows)]
fn create_symlink(target: PathBuf, path: &Path) -> Result<(), CheckpointError> {
    std::os::windows::fs::symlink_file(target, path).map_err(|source| CheckpointError::Io {
        path: path.to_path_buf(),
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
