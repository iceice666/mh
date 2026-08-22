use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use mh::identity::{ExecutionId, IsolatedWorkspaceId, TaskId};
use mh::isolation::{IntegrationResult, IsolationError, IsolationStore};
use mh::workspace::{WorkspaceTracker, WorkspaceTrackerImpl};

fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn repository() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    git(dir.path(), &["init", "-q"]);
    git(dir.path(), &["config", "user.name", "mh test"]);
    git(dir.path(), &["config", "user.email", "mh@example.invalid"]);
    fs::write(dir.path().join("tracked"), "base").unwrap();
    fs::write(dir.path().join("parent-only"), "stable").unwrap();
    git(dir.path(), &["add", "."]);
    git(dir.path(), &["commit", "-qm", "base"]);
    dir
}

fn revision(dir: &Path) -> mh::identity::RevisionId {
    WorkspaceTrackerImpl::open(dir)
        .unwrap()
        .current_revision()
        .unwrap()
        .id
}

#[cfg(unix)]
#[test]
fn create_materializes_exact_dirty_base_without_changing_parent() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let dir = repository();
    fs::write(dir.path().join("tracked"), [0, 1, 255]).unwrap();
    fs::write(dir.path().join("untracked"), [4, 0, 5]).unwrap();
    fs::write(dir.path().join("script"), "#!/bin/sh\n").unwrap();
    fs::set_permissions(dir.path().join("script"), fs::Permissions::from_mode(0o755)).unwrap();
    symlink("tracked", dir.path().join("link")).unwrap();
    fs::create_dir_all(dir.path().join(".mh/private")).unwrap();
    fs::write(dir.path().join(".mh/private/state"), "parent metadata").unwrap();

    let base = revision(dir.path());
    let parent_head = git(dir.path(), &["rev-parse", "HEAD"]);
    let store = IsolationStore::open(dir.path()).unwrap();
    let record = store
        .create(
            IsolatedWorkspaceId(1),
            TaskId(7),
            ExecutionId(9),
            base.clone(),
        )
        .unwrap();

    assert_eq!(record.base_revision, base);
    assert_eq!(record.final_revision, record.base_revision);
    assert_eq!(revision(&record.path), record.base_revision);
    assert_eq!(fs::read(record.path.join("tracked")).unwrap(), [0, 1, 255]);
    assert_eq!(fs::read(record.path.join("untracked")).unwrap(), [4, 0, 5]);
    assert_eq!(
        fs::read_link(record.path.join("link")).unwrap(),
        Path::new("tracked")
    );
    assert_ne!(
        fs::metadata(record.path.join("script"))
            .unwrap()
            .permissions()
            .mode()
            & 0o111,
        0
    );
    assert!(!record.path.join(".mh/private/state").exists());
    assert_eq!(fs::read(dir.path().join("tracked")).unwrap(), [0, 1, 255]);
    assert_eq!(git(dir.path(), &["rev-parse", "HEAD"]), parent_head);

    fs::write(record.path.join("tracked"), "child edit").unwrap();
    fs::remove_file(record.path.join("untracked")).unwrap();
    assert_eq!(fs::read(dir.path().join("tracked")).unwrap(), [0, 1, 255]);
    assert!(dir.path().join("untracked").exists());
}

#[test]
fn finalize_persists_the_complete_child_delta() {
    let dir = repository();
    let base = revision(dir.path());
    let store = IsolationStore::open(dir.path()).unwrap();
    let created = store
        .create(
            IsolatedWorkspaceId(2),
            TaskId(1),
            ExecutionId(2),
            base.clone(),
        )
        .unwrap();

    fs::write(created.path.join("tracked"), "child").unwrap();
    fs::remove_file(created.path.join("parent-only")).unwrap();
    fs::write(created.path.join("added"), "new").unwrap();
    let finalized = store.finalize(created.id).unwrap();

    assert_eq!(finalized.base_revision, base);
    assert_eq!(finalized.final_revision, revision(&created.path));
    assert_eq!(
        finalized.changed_paths,
        vec![
            PathBuf::from("added"),
            PathBuf::from("parent-only"),
            PathBuf::from("tracked")
        ]
    );
    drop(store);
    assert_eq!(
        IsolationStore::open(dir.path())
            .unwrap()
            .load(created.id)
            .unwrap(),
        finalized
    );
}

#[test]
fn disjoint_parent_and_child_changes_integrate_and_cleanup() {
    let dir = repository();
    let base = revision(dir.path());
    let store = IsolationStore::open(dir.path()).unwrap();
    let child = store
        .create(IsolatedWorkspaceId(3), TaskId(1), ExecutionId(1), base)
        .unwrap();
    fs::write(child.path.join("tracked"), "child").unwrap();
    fs::write(child.path.join("child-added"), "added").unwrap();
    store.finalize(child.id).unwrap();

    fs::write(dir.path().join("parent-only"), "parent").unwrap();
    let previous_revision = revision(dir.path());
    let result = store.integrate(child.id).unwrap();
    let current_revision = revision(dir.path());

    assert_eq!(
        result,
        IntegrationResult::Applied {
            previous_revision,
            current_revision,
            changed_paths: vec![PathBuf::from("child-added"), PathBuf::from("tracked")],
        }
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("tracked")).unwrap(),
        "child"
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("parent-only")).unwrap(),
        "parent"
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("child-added")).unwrap(),
        "added"
    );
    assert!(!child.path.exists());
    assert!(matches!(
        store.load(child.id),
        Err(IsolationError::NotFound(_))
    ));
}

#[test]
fn overlapping_changes_conflict_without_overwriting_parent() {
    let dir = repository();
    let base = revision(dir.path());
    let store = IsolationStore::open(dir.path()).unwrap();
    let child = store
        .create(
            IsolatedWorkspaceId(4),
            TaskId(1),
            ExecutionId(1),
            base.clone(),
        )
        .unwrap();
    fs::write(child.path.join("tracked"), "child").unwrap();
    store.finalize(child.id).unwrap();

    fs::write(dir.path().join("tracked"), "parent").unwrap();
    let parent_current = revision(dir.path());
    let result = store.integrate(child.id).unwrap();

    assert_eq!(
        result,
        IntegrationResult::Conflict {
            child_base: base,
            parent_current: parent_current.clone(),
            paths: vec![PathBuf::from("tracked")],
        }
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("tracked")).unwrap(),
        "parent"
    );
    assert_eq!(revision(dir.path()), parent_current);
    assert!(child.path.exists());
    assert_eq!(store.load(child.id).unwrap().id, child.id);
}

#[test]
fn discard_and_owner_cleanup_remove_only_selected_workspaces() {
    let dir = repository();
    let base = revision(dir.path());
    let store = IsolationStore::open(dir.path()).unwrap();
    let discarded = store
        .create(
            IsolatedWorkspaceId(5),
            TaskId(10),
            ExecutionId(1),
            base.clone(),
        )
        .unwrap();
    let owner_a = store
        .create(
            IsolatedWorkspaceId(6),
            TaskId(10),
            ExecutionId(2),
            base.clone(),
        )
        .unwrap();
    let owner_b = store
        .create(IsolatedWorkspaceId(7), TaskId(11), ExecutionId(3), base)
        .unwrap();

    store.discard(discarded.id).unwrap();
    assert!(!discarded.path.exists());
    store.cleanup_owner(TaskId(10)).unwrap();
    assert!(!owner_a.path.exists());
    assert!(owner_b.path.exists());
    assert_eq!(store.load(owner_b.id).unwrap().owner_task_id, TaskId(11));
}

#[test]
fn non_git_workspace_reports_structured_unsupported_error() {
    let dir = tempfile::tempdir().unwrap();
    let error = match IsolationStore::open(dir.path()) {
        Ok(_) => panic!("filesystem-only workspace unexpectedly supported isolation"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        IsolationError::UnsupportedWorkspace {
            operation: "open",
            ..
        }
    ));
}
