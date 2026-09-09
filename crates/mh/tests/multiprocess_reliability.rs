use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::thread;
use std::time::{Duration, Instant};

use mh::session::{Session, SessionError, SessionEvent};

const CHILD_ENV: &str = "MH_RELIABILITY_CHILD";
const WORKSPACE_ENV: &str = "MH_RELIABILITY_WORKSPACE";
const READY_ENV: &str = "MH_RELIABILITY_READY";
const GO_ENV: &str = "MH_RELIABILITY_GO";
const COUNT_ENV: &str = "MH_RELIABILITY_COUNT";
const HOLD_ENV: &str = "MH_RELIABILITY_HOLD";
const DESCENDANT_ENV: &str = "MH_RELIABILITY_DESCENDANT";

#[test]
fn child_entry() {
    let Ok(role) = std::env::var(CHILD_ENV) else {
        return;
    };
    let workspace = PathBuf::from(std::env::var(WORKSPACE_ENV).unwrap());
    let ready = PathBuf::from(std::env::var(READY_ENV).unwrap());
    let go = PathBuf::from(std::env::var(GO_ENV).unwrap());
    fs::write(&ready, b"ready").unwrap();
    wait_exists(&go, Duration::from_secs(10));
    match role.as_str() {
        "append" => {
            let session = Session::resume(&workspace).unwrap();
            let count: usize = std::env::var(COUNT_ENV).unwrap().parse().unwrap();
            for index in 0..count {
                session
                    .register_queued_task(&format!(
                        "writer {} — 大型記錄 {}",
                        std::process::id(),
                        index
                    ))
                    .unwrap();
            }
        }
        "open" => {
            Session::open(&workspace).unwrap();
        }
        "hold_runner" => {
            let session = Session::resume(&workspace).unwrap();
            let task = session.root_task();
            let _guard = session.acquire_runner(task).unwrap();
            fs::write(std::env::var(HOLD_ENV).unwrap(), b"held").unwrap();
            thread::sleep(Duration::from_secs(2));
        }
        "hold_runner_with_descendant" => {
            let session = Session::resume(&workspace).unwrap();
            let task = session.root_task();
            let _guard = session.acquire_runner(task).unwrap();
            let mut descendant = Command::new("sleep").arg("30").spawn().unwrap();
            fs::write(
                std::env::var(DESCENDANT_ENV).unwrap(),
                descendant.id().to_string(),
            )
            .unwrap();
            fs::write(std::env::var(HOLD_ENV).unwrap(), b"held").unwrap();
            thread::sleep(Duration::from_secs(30));
            let _ = descendant.kill();
            let _ = descendant.wait();
        }
        "try_runner" => {
            let session = Session::resume(&workspace).unwrap();
            let result = session.acquire_runner(session.root_task());
            assert!(matches!(result, Err(SessionError::Busy(_))));
        }
        other => panic!("unknown child role {other}"),
    }
}

#[test]
fn real_process_writers_keep_physical_sequence_and_ids_unique() {
    let dir = tempfile::tempdir().unwrap();
    Session::open(dir.path()).unwrap();
    let gate = tempfile::tempdir().unwrap();
    let go = gate.path().join("go");
    let count = 24;
    let mut children = Vec::new();
    for index in 0..4 {
        let ready = gate.path().join(format!("ready-{index}"));
        children.push(spawn_child(
            "append",
            dir.path(),
            &ready,
            &go,
            &[(COUNT_ENV, count.to_string())],
        ));
        wait_exists(&ready, Duration::from_secs(10));
    }
    fs::write(&go, b"go").unwrap();
    wait_children(children);

    let session = Session::inspect(dir.path()).unwrap();
    let events = session.events();
    assert_eq!(events.len(), 1 + 4 * count);
    assert!(events.windows(2).all(|pair| pair[1].seq == pair[0].seq + 1));
    let task_ids = events
        .iter()
        .filter_map(|record| match record.event {
            SessionEvent::TaskStarted { task_id, .. } => Some(task_id.0),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        task_ids.len(),
        task_ids.iter().copied().collect::<BTreeSet<_>>().len()
    );
    let raw = fs::read(dir.path().join(".mh/session.jsonl")).unwrap();
    assert!(raw.ends_with(b"\n"));
    assert_eq!(
        raw.split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .count(),
        events.len()
    );
}

#[test]
fn canonical_workspace_allows_only_one_root_runner() {
    let dir = tempfile::tempdir().unwrap();
    let session = Session::open(dir.path()).unwrap();
    session.begin_task("owned").unwrap();
    let alias_root = tempfile::tempdir().unwrap();
    let alias = alias_root.path().join("alias");
    #[cfg(unix)]
    std::os::unix::fs::symlink(dir.path(), &alias).unwrap();

    let gate = tempfile::tempdir().unwrap();
    let go = gate.path().join("go");
    let held = gate.path().join("held");
    let ready_owner = gate.path().join("ready-owner");
    let owner = spawn_child(
        "hold_runner",
        dir.path(),
        &ready_owner,
        &go,
        &[(HOLD_ENV, held.display().to_string())],
    );
    wait_exists(&ready_owner, Duration::from_secs(10));
    fs::write(&go, b"go").unwrap();
    wait_exists(&held, Duration::from_secs(10));

    let ready_contender = gate.path().join("ready-contender");
    let contender = spawn_child("try_runner", &alias, &ready_contender, &go, &[]);
    wait_children(vec![contender]);
    wait_children(vec![owner]);

    let session = Session::resume(dir.path()).unwrap();
    let guard = session.acquire_runner(session.root_task()).unwrap();
    drop(guard);
}

#[cfg(unix)]
#[test]
fn killed_owner_descendant_does_not_inherit_runner_lock() {
    let dir = tempfile::tempdir().unwrap();
    let session = Session::open(dir.path()).unwrap();
    let task = session.begin_task("owned").unwrap();
    let gate = tempfile::tempdir().unwrap();
    let go = gate.path().join("go");
    let ready = gate.path().join("ready");
    let held = gate.path().join("held");
    let descendant = gate.path().join("descendant");
    let mut owner = spawn_child(
        "hold_runner_with_descendant",
        dir.path(),
        &ready,
        &go,
        &[
            (HOLD_ENV, held.display().to_string()),
            (DESCENDANT_ENV, descendant.display().to_string()),
        ],
    );
    wait_exists(&ready, Duration::from_secs(10));
    fs::write(&go, b"go").unwrap();
    wait_exists(&held, Duration::from_secs(10));
    wait_exists(&descendant, Duration::from_secs(10));
    owner.kill().unwrap();
    owner.wait().unwrap();

    let resumed = Session::resume(dir.path()).unwrap();
    let guard = resumed.acquire_runner(Some(task)).unwrap();
    drop(guard);

    let pid = fs::read_to_string(&descendant).unwrap();
    let _ = Command::new("kill").arg(pid.trim()).status();
}

#[test]
fn inspection_reads_torn_prefix_without_rewriting_and_writer_repairs_only_tail() {
    let dir = tempfile::tempdir().unwrap();
    let session = Session::open(dir.path()).unwrap();
    session.begin_task("stable").unwrap();
    let journal = dir.path().join(".mh/session.jsonl");
    let mut file = OpenOptions::new().append(true).open(&journal).unwrap();
    file.write_all(b"{\"seq\":999,\"type\":\"task_started\"")
        .unwrap();
    file.sync_all().unwrap();
    let before = fs::read(&journal).unwrap();

    let inspected = Session::inspect(dir.path()).unwrap();
    assert!(!inspected.warnings().is_empty());
    assert_eq!(fs::read(&journal).unwrap(), before);

    let writer = Session::resume(dir.path()).unwrap();
    writer.register_queued_task("after repair").unwrap();
    let repaired = fs::read(&journal).unwrap();
    assert!(repaired.ends_with(b"\n"));
    assert!(
        !repaired
            .windows(b"999".len())
            .any(|window| window == b"999")
    );
    assert!(
        fs::read_dir(dir.path().join(".mh"))
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| entry
                .file_name()
                .to_string_lossy()
                .starts_with("session.jsonl.torn"))
    );
}

#[test]
fn complete_corruption_and_duplicate_sequences_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let session = Session::open(dir.path()).unwrap();
    session.begin_task("stable").unwrap();
    let journal = dir.path().join(".mh/session.jsonl");
    let valid = fs::read_to_string(&journal).unwrap();
    let duplicate = valid.lines().last().unwrap().to_string();
    let mut file = OpenOptions::new().append(true).open(&journal).unwrap();
    writeln!(file, "{duplicate}").unwrap();
    file.sync_all().unwrap();
    assert!(matches!(
        Session::inspect(dir.path()),
        Err(SessionError::CorruptJournal { .. })
    ));

    let dir = tempfile::tempdir().unwrap();
    let session = Session::open(dir.path()).unwrap();
    session.begin_task("stable").unwrap();
    let journal = dir.path().join(".mh/session.jsonl");
    let next = session.state().last_event_seq + 1;
    let mut file = OpenOptions::new().append(true).open(&journal).unwrap();
    writeln!(
        file,
        "{{\"seq\":{next},\"timestamp_ms\":1,\"type\":\"task_started\"}}"
    )
    .unwrap();
    file.sync_all().unwrap();
    assert!(matches!(
        Session::inspect(dir.path()),
        Err(SessionError::CorruptJournal { .. })
    ));
}

#[test]
fn missing_or_stale_state_cache_is_rebuilt_from_the_journal() {
    let dir = tempfile::tempdir().unwrap();
    let session = Session::open(dir.path()).unwrap();
    let task = session.begin_task("durable").unwrap();
    let state = dir.path().join(".mh/state.json");
    fs::remove_file(&state).unwrap();
    let resumed = Session::resume(dir.path()).unwrap();
    assert_eq!(resumed.root_task(), Some(task));
    assert!(state.exists());

    fs::write(&state, br#"{"id":"stale","workspace":"/wrong","active_task":null,"current_revision":"bad","last_event_seq":0,"created_at_ms":0,"updated_at_ms":0}"#).unwrap();
    let resumed = Session::resume(dir.path()).unwrap();
    assert_eq!(resumed.root_task(), Some(task));
    assert_eq!(
        resumed.state().last_event_seq,
        resumed.events().last().unwrap().seq
    );
}

#[test]
fn concurrent_open_initializes_one_session_header() {
    let dir = tempfile::tempdir().unwrap();
    let gate = tempfile::tempdir().unwrap();
    let go = gate.path().join("go");
    let mut children = Vec::new();
    for index in 0..4 {
        let ready = gate.path().join(format!("open-ready-{index}"));
        children.push(spawn_child("open", dir.path(), &ready, &go, &[]));
        wait_exists(&ready, Duration::from_secs(10));
    }
    fs::write(&go, b"go").unwrap();
    wait_children(children);
    let session = Session::inspect(dir.path()).unwrap();
    assert_eq!(
        session
            .events()
            .iter()
            .filter(|record| matches!(record.event, SessionEvent::SessionInitialized { .. }))
            .count(),
        1
    );
}

#[test]
fn healthy_v4_journal_migrates_once_and_preserves_identity() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join(".mh")).unwrap();
    let canonical = dir.path().canonicalize().unwrap();
    let state = serde_json::json!({
        "id": "legacy-session",
        "workspace": canonical,
        "active_task": 1,
        "current_revision": "legacy-revision",
        "last_event_seq": 2,
        "created_at_ms": 123,
        "updated_at_ms": 456
    });
    fs::write(
        dir.path().join(".mh/state.json"),
        serde_json::to_vec_pretty(&state).unwrap(),
    )
    .unwrap();
    let journal = dir.path().join(".mh/session.jsonl");
    fs::write(
        &journal,
        concat!(
            "{\"seq\":1,\"timestamp_ms\":124,\"type\":\"task_started\",\"task_id\":1,\"objective\":\"legacy task\",\"base_revision\":\"legacy-revision\"}\n",
            "{\"seq\":2,\"timestamp_ms\":125,\"type\":\"user_message\",\"task_id\":1,\"content\":\"legacy task\"}\n"
        ),
    )
    .unwrap();

    assert!(Session::inspect(dir.path()).is_err());
    let resumed = Session::resume(dir.path()).unwrap();
    assert_eq!(resumed.state().id, "legacy-session");
    assert_eq!(resumed.root_task(), Some(mh::identity::TaskId(1)));
    assert_eq!(resumed.events().first().unwrap().seq, 0);
    assert!(matches!(
        resumed.events().last().unwrap().event,
        SessionEvent::SessionMigrated {
            from_version: 4,
            ..
        }
    ));
    let once = fs::read(&journal).unwrap();
    Session::resume(dir.path()).unwrap();
    assert_eq!(fs::read(&journal).unwrap(), once);
}

#[test]
fn cli_observation_is_byte_for_byte_read_only() {
    let dir = tempfile::tempdir().unwrap();
    let session = Session::open(dir.path()).unwrap();
    let task = session.register_queued_task("observe me").unwrap();
    let journal = dir.path().join(".mh/session.jsonl");
    let state = dir.path().join(".mh/state.json");
    let before_journal = fs::read(&journal).unwrap();
    let before_state = fs::read(&state).unwrap();

    for args in [
        vec!["tasks".to_string()],
        vec!["inspect".to_string(), task.0.to_string()],
        vec!["attach".to_string(), task.0.to_string()],
        vec!["sessions".to_string()],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_mh"))
            .args(args)
            .current_dir(dir.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert_eq!(fs::read(&journal).unwrap(), before_journal);
    assert_eq!(fs::read(&state).unwrap(), before_state);
}

#[test]
fn detached_cli_reports_busy_instead_of_started() {
    let dir = tempfile::tempdir().unwrap();
    let session = Session::open(dir.path()).unwrap();
    let root = session.begin_task("live").unwrap();
    let _guard = session.acquire_runner(Some(root)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mh"))
        .args(["run", "queued while busy", "--detach"])
        .current_dir(dir.path())
        .env("MH_API_KEY", "unused")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("task 2 queued but not admitted"),
        "{stderr}"
    );
    assert!(stderr.contains("busy"), "{stderr}");
    assert!(!stderr.contains("task 2 admitted"), "{stderr}");
    let snapshot = Session::inspect(dir.path()).unwrap();
    assert_eq!(snapshot.state().active_task, Some(root));
    assert_eq!(
        snapshot.task_view(mh::identity::TaskId(2)).status().label(),
        "queued"
    );
}
#[test]
fn detached_cli_persists_pre_admission_failure() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mh"))
        .args(["run", "bad configuration", "--detach"])
        .current_dir(dir.path())
        .env_remove("MH_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("task 1 admitted"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("task 1 queued but not admitted"));
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let view = Session::inspect(dir.path())
            .unwrap()
            .task_view(mh::identity::TaskId(1));
        if view.status().label() == "blocked" {
            assert!(
                view.status_note
                    .as_deref()
                    .is_some_and(|note| note.contains("MH_API_KEY")),
                "missing durable configuration failure: {:?}",
                view.status_note
            );
            break;
        }
        assert!(
            Instant::now() < deadline,
            "detached failure was not persisted"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn detached_cli_reports_only_confirmed_admission() {
    let dir = tempfile::tempdir().unwrap();
    let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = server.local_addr().unwrap();
    let provider = thread::spawn(move || {
        let (mut stream, _) = server.accept().unwrap();
        let mut request = Vec::new();
        let mut buffer = [0u8; 4096];
        let header_end = loop {
            let read = std::io::Read::read(&mut stream, &mut buffer).unwrap();
            assert_ne!(read, 0, "provider request ended before headers");
            request.extend_from_slice(&buffer[..read]);
            if let Some(offset) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break offset + 4;
            }
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.strip_prefix("Content-Length: ")
                    .or_else(|| line.strip_prefix("content-length: "))
            })
            .unwrap()
            .parse::<usize>()
            .unwrap();
        while request.len() - header_end < content_length {
            let read = std::io::Read::read(&mut stream, &mut buffer).unwrap();
            assert_ne!(read, 0, "provider request body ended early");
            request.extend_from_slice(&buffer[..read]);
        }
        let body = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"waiting\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_cli\",\"output\":[]}}\n\n"
        );
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .unwrap();
    });
    let output = Command::new(env!("CARGO_BIN_EXE_mh"))
        .args(["run", "detached success", "--detach"])
        .current_dir(dir.path())
        .env("MH_API_KEY", "test")
        .env("MH_BASE_URL", format!("http://{address}/v1"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("task 1 admitted"));
    provider.join().unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let view = Session::inspect(dir.path())
            .unwrap()
            .task_view(mh::identity::TaskId(1));
        if view.status().label() == "waiting-user" {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "detached child did not settle: {}",
            fs::read_to_string(dir.path().join(".mh/tasks/1.log")).unwrap_or_default()
        );
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn recovery_cli_resolves_an_audited_unknown_execution() {
    let dir = tempfile::tempdir().unwrap();
    let session = Session::open(dir.path()).unwrap();
    let task = session.begin_task("recover from crash").unwrap();
    let execution = session.reserve_execution().unwrap();
    session
        .append_shared_event(SessionEvent::RunnerAdmitted {
            info: mh::session::RunnerInfo {
                generation: 1,
                instance_id: "lost".to_string(),
                pid: 42,
                task_id: Some(task),
                started_at_ms: 1,
            },
        })
        .unwrap();
    session
        .append_shared_event(SessionEvent::PtcStarted {
            task_id: task,
            execution_id: execution,
            source: "write()".to_string(),
            start_revision: session.state().current_revision,
        })
        .unwrap();
    session.recover_previous_owner(2).unwrap();
    assert_eq!(session.task_view(task).status().label(), "blocked");

    let inspected = Command::new(env!("CARGO_BIN_EXE_mh"))
        .args(["inspect", &task.0.to_string()])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(inspected.status.success());
    assert!(String::from_utf8_lossy(&inspected.stdout).contains("outcome-unknown executions"));

    let recovered = Command::new(env!("CARGO_BIN_EXE_mh"))
        .args([
            "recover",
            &task.0.to_string(),
            "execution",
            &execution.0.to_string(),
            "verified",
            "externally",
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        recovered.status.success(),
        "{}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    assert_eq!(
        Session::inspect(dir.path())
            .unwrap()
            .task_view(task)
            .status()
            .label(),
        "running"
    );
}

#[test]
fn queued_registration_does_not_replace_a_live_root_pointer() {
    let dir = tempfile::tempdir().unwrap();
    let session = Session::open(dir.path()).unwrap();
    let root = session.begin_task("live root").unwrap();
    let queued = session.register_queued_task("later").unwrap();
    assert_ne!(root, queued);
    assert_eq!(session.state().active_task, Some(root));
    assert_eq!(session.task_view(queued).status().label(), "queued");
}

fn spawn_child(
    role: &str,
    workspace: &Path,
    ready: &Path,
    go: &Path,
    extra: &[(&str, String)],
) -> Child {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .arg("--exact")
        .arg("child_entry")
        .arg("--nocapture")
        .env(CHILD_ENV, role)
        .env(WORKSPACE_ENV, workspace)
        .env(READY_ENV, ready)
        .env(GO_ENV, go);
    for (key, value) in extra {
        command.env(key, value);
    }
    command.spawn().unwrap()
}

fn wait_children(children: Vec<Child>) {
    for mut child in children {
        let status = child.wait().unwrap();
        assert!(status.success(), "child exited {status}");
    }
}

fn wait_exists(path: &Path, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        thread::sleep(Duration::from_millis(5));
    }
}
