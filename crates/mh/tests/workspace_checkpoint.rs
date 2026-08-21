use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use mh::checkpoint::CheckpointStore;
use mh::identity::TaskId;
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

fn init_git(dir: &Path) {
    git(dir, &["init", "-q"]);
    git(dir, &["config", "user.name", "mh test"]);
    git(dir, &["config", "user.email", "mh@example.invalid"]);
}

#[test]
fn filesystem_revisions_are_deterministic_and_report_deltas() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("binary"), [0, 255, 1]).unwrap();
    fs::create_dir(dir.path().join("target")).unwrap();
    fs::write(dir.path().join("target/ignored"), "one").unwrap();
    let tracker = WorkspaceTrackerImpl::open(dir.path()).unwrap();
    let first = tracker.current_revision().unwrap();
    let repeated = tracker.current_revision().unwrap();
    assert_eq!(first.id, repeated.id);
    assert_eq!(first.sequence, repeated.sequence);
    fs::write(dir.path().join("binary"), [0, 254, 1]).unwrap();
    fs::write(dir.path().join("added"), "new").unwrap();
    fs::write(dir.path().join("target/ignored"), "two").unwrap();
    let second = tracker.current_revision().unwrap();
    assert_ne!(first.id, second.id);
    let delta = tracker.delta(&first.id, &second.id).unwrap();
    assert_eq!(delta.added, vec![std::path::PathBuf::from("added")]);
    assert_eq!(delta.modified, vec![std::path::PathBuf::from("binary")]);
    assert!(delta.deleted.is_empty());

    fs::remove_file(dir.path().join("added")).unwrap();
    let third = tracker.current_revision().unwrap();
    assert_eq!(
        tracker.delta(&second.id, &third.id).unwrap().deleted,
        vec![std::path::PathBuf::from("added")]
    );

    let reopened = WorkspaceTrackerImpl::open(dir.path()).unwrap();
    assert_eq!(reopened.delta(&first.id, &second.id).unwrap(), delta);
}

#[test]
fn git_identity_ignores_index_but_tracks_head_worktree_and_untracked() {
    let dir = tempfile::tempdir().unwrap();
    init_git(dir.path());
    fs::write(dir.path().join("tracked"), "base").unwrap();
    git(dir.path(), &["add", "tracked"]);
    git(dir.path(), &["commit", "-qm", "base"]);
    let tracker = WorkspaceTrackerImpl::open(dir.path()).unwrap();
    let base = tracker.current_revision().unwrap();

    fs::write(dir.path().join("tracked"), "changed").unwrap();
    let changed = tracker.current_revision().unwrap();
    assert_ne!(base.id, changed.id);
    git(dir.path(), &["add", "tracked"]);
    assert_eq!(changed.id, tracker.current_revision().unwrap().id);

    fs::write(dir.path().join("untracked"), [7, 0, 9]).unwrap();
    let untracked = tracker.current_revision().unwrap();
    assert_ne!(changed.id, untracked.id);
    git(dir.path(), &["commit", "-qm", "changed"]);
    assert_ne!(untracked.id, tracker.current_revision().unwrap().id);
}

#[cfg(unix)]
#[test]
fn checkpoint_roundtrip_preserves_git_state_bytes_links_and_modes() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let dir = tempfile::tempdir().unwrap();
    init_git(dir.path());
    fs::write(dir.path().join("tracked"), [0, 1, 255]).unwrap();
    fs::write(dir.path().join("staged-only"), "index state").unwrap();
    fs::write(dir.path().join("script"), "#!/bin/sh\nexit 0\n").unwrap();
    fs::set_permissions(dir.path().join("script"), fs::Permissions::from_mode(0o755)).unwrap();
    symlink("tracked", dir.path().join("link")).unwrap();
    git(dir.path(), &["add", "tracked", "script", "link"]);
    git(dir.path(), &["commit", "-qm", "base"]);
    git(dir.path(), &["add", "staged-only"]);
    fs::write(dir.path().join("untracked"), [4, 0, 5]).unwrap();
    fs::create_dir_all(dir.path().join(".mh/private")).unwrap();
    fs::write(dir.path().join(".mh/private/keep"), "untouched").unwrap();
    fs::write(dir.path().join("ignored"), "untouched").unwrap();
    fs::write(dir.path().join(".gitignore"), "ignored\n").unwrap();

    let head_before = git(dir.path(), &["rev-parse", "HEAD"]);
    let index_before = fs::read(dir.path().join(".git/index")).unwrap();
    let tracker = WorkspaceTrackerImpl::open(dir.path()).unwrap();
    let store = CheckpointStore::open(dir.path(), Arc::new(tracker.clone())).unwrap();
    let checkpoint = store.create(TaskId(7)).unwrap();

    fs::write(dir.path().join("tracked"), "wrong").unwrap();
    fs::remove_file(dir.path().join("untracked")).unwrap();
    fs::remove_file(dir.path().join("link")).unwrap();
    fs::set_permissions(dir.path().join("script"), fs::Permissions::from_mode(0o644)).unwrap();
    fs::write(dir.path().join("extra"), "delete me").unwrap();
    fs::write(dir.path().join("ignored"), "still untouched").unwrap();
    fs::write(dir.path().join(".mh/private/keep"), "still untouched").unwrap();

    let restored = store.restore(TaskId(7), &checkpoint).unwrap();
    assert_eq!(restored.id, checkpoint.revision);
    assert_eq!(fs::read(dir.path().join("tracked")).unwrap(), [0, 1, 255]);
    assert_eq!(fs::read(dir.path().join("untracked")).unwrap(), [4, 0, 5]);
    assert_eq!(
        fs::read_link(dir.path().join("link")).unwrap(),
        Path::new("tracked")
    );
    assert_ne!(
        fs::metadata(dir.path().join("script"))
            .unwrap()
            .permissions()
            .mode()
            & 0o111,
        0
    );
    assert!(!dir.path().join("extra").exists());
    assert_eq!(
        fs::read_to_string(dir.path().join("ignored")).unwrap(),
        "still untouched"
    );
    assert_eq!(
        fs::read_to_string(dir.path().join(".mh/private/keep")).unwrap(),
        "still untouched"
    );
    assert_eq!(git(dir.path(), &["rev-parse", "HEAD"]), head_before);
    assert_eq!(
        fs::read(dir.path().join(".git/index")).unwrap(),
        index_before
    );
}

#[cfg(unix)]
#[test]
fn restore_rejects_symlink_ancestor_before_mutating_workspace() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("nested")).unwrap();
    fs::write(dir.path().join("nested/file"), "checkpoint").unwrap();
    fs::write(dir.path().join("stable"), "before").unwrap();
    let tracker = WorkspaceTrackerImpl::open(dir.path()).unwrap();
    let store = CheckpointStore::open(dir.path(), Arc::new(tracker)).unwrap();
    let checkpoint = store.create(TaskId(1)).unwrap();

    fs::remove_dir_all(dir.path().join("nested")).unwrap();
    let outside = tempfile::tempdir().unwrap();
    symlink(outside.path(), dir.path().join("nested")).unwrap();
    fs::write(dir.path().join("stable"), "after").unwrap();

    assert!(store.restore(TaskId(1), &checkpoint).is_err());
    assert_eq!(
        fs::read_to_string(dir.path().join("stable")).unwrap(),
        "after"
    );
    assert!(!outside.path().join("file").exists());
}
