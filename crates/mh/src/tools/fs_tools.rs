//! Core tool set (spec §9): read, write, edit, glob, grep, exec.
//!
//! Every tool takes JSON args and returns a JSON-serializable value
//! (structured results; large blobs go through the [`ResultStore`]
//! as handles).

use std::collections::VecDeque;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::capability::{Capabilities, ProcessPolicy, SandboxError};
use super::store::{ResultId, ResultStore};

/// Upper bound on a single read() result entering JS/context.
const READ_CAP: usize = 256 * 1024;
/// Upper bound on stored exec output per stream.
const EXEC_CAP: usize = 4 << 20;
/// Default exec timeout.
const DEFAULT_TIMEOUT_MS: u64 = 120_000;

/// Tool failures are values, not exceptions: the model sees them and
/// can branch on them inside the same PTC program.
pub type ToolOutput = Value;

#[derive(Debug)]
pub enum ToolError {
    Sandbox(String),
    Invalid(String),
    Io(String),
    TimedOut { program: String, timeout_ms: u64 },
    Cancelled,
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::Sandbox(m) => write!(f, "{m}"),
            ToolError::Invalid(m) => write!(f, "invalid arguments: {m}"),
            ToolError::Io(m) => write!(f, "io error: {m}"),
            ToolError::TimedOut {
                program,
                timeout_ms,
            } => {
                write!(f, "timed out after {timeout_ms} ms: {program}")
            }
            ToolError::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl From<SandboxError> for ToolError {
    fn from(e: SandboxError) -> Self {
        ToolError::Sandbox(e.to_string())
    }
}

fn err_json(e: &ToolError) -> Value {
    json!({ "error": e.to_string() })
}

/// Converts a sandbox error into the error-value convention used by
/// every tool.
impl SandboxError {
    pub fn to_json(&self) -> Value {
        json!({ "error": self.to_string() })
    }
}

// ---------------------------------------------------------------------------
// read
// ---------------------------------------------------------------------------

/// `read { path, offset?, limit? }` — line-oriented bounded read.
/// Returns `{ path, content, totalLines, truncated }`.
pub fn read(caps: &Capabilities, args: &Value) -> ToolOutput {
    let Some(path) = args.get("path").and_then(Value::as_str) else {
        return json!({ "error": "read: missing 'path'" });
    };
    let abs = match caps.resolve_existing_read(path) {
        Ok(p) => p,
        Err(e) => return e.to_json(),
    };
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(10_000) as usize;

    let bytes = match std::fs::read(&abs) {
        Ok(b) => b,
        Err(e) => return json!({ "error": format!("read {path}: {e}") }),
    };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let total = lines.len();
    let end = offset.saturating_add(limit).min(total);
    let slice = if offset < total {
        &lines[offset..end]
    } else {
        &[][..]
    };
    let mut content = String::new();
    let mut included = 0usize;
    for line in slice {
        if content.len().saturating_add(line.len()) > READ_CAP {
            break;
        }
        content.push_str(line);
        included += 1;
    }
    let truncated = end < total || included < slice.len();
    json!({
        "path": path,
        "content": content,
        "totalLines": total,
        "truncated": truncated,
    })
}

// ---------------------------------------------------------------------------
// write
// ---------------------------------------------------------------------------

/// `write { path, content }` — full overwrite. Parent directories are
/// created (model ergonomics outweigh strictness here; the sandbox
/// still confines the target).
pub fn write(caps: &Capabilities, args: &Value) -> ToolOutput {
    let Some(path) = args.get("path").and_then(Value::as_str) else {
        return json!({ "error": "write: missing 'path'" });
    };
    let Some(content) = args.get("content").and_then(Value::as_str) else {
        return json!({ "error": "write: missing 'content'" });
    };
    let abs = match caps.resolve_for_write(path) {
        Ok(p) => p,
        Err(e) => return e.to_json(),
    };
    if let Some(parent) = abs.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return json!({ "error": format!("write {path}: {e}") });
    }
    match std::fs::write(&abs, content) {
        Ok(()) => json!({ "ok": true, "path": path, "bytes": content.len() }),
        Err(e) => json!({ "error": format!("write {path}: {e}") }),
    }
}

// ---------------------------------------------------------------------------
// edit
// ---------------------------------------------------------------------------

/// `edit { path, old, new }` — deterministic single-occurrence replace
/// (spec §9.3). Errors if `old` is absent or ambiguous.
pub fn edit(caps: &Capabilities, args: &Value) -> ToolOutput {
    let Some(path) = args.get("path").and_then(Value::as_str) else {
        return json!({ "error": "edit: missing 'path'" });
    };
    let Some(old) = args.get("old").and_then(Value::as_str) else {
        return json!({ "error": "edit: missing 'old'" });
    };
    let Some(new) = args.get("new").and_then(Value::as_str) else {
        return json!({ "error": "edit: missing 'new'" });
    };
    let abs = match caps.resolve_for_write(path) {
        Ok(p) => p,
        Err(e) => return e.to_json(),
    };
    let text = match std::fs::read_to_string(&abs) {
        Ok(t) => t,
        Err(e) => return json!({ "error": format!("edit {path}: {e}") }),
    };
    let first = text.find(old);
    let Some(first) = first else {
        return json!({ "error": format!("edit {path}: 'old' not found") });
    };
    if text[first + old.len()..].contains(old) {
        return json!({ "error": format!("edit {path}: 'old' matches {} times; make it unique", 2) });
    }
    let mut out = String::with_capacity(text.len() - old.len() + new.len());
    out.push_str(&text[..first]);
    out.push_str(new);
    out.push_str(&text[first + old.len()..]);
    match std::fs::write(&abs, out) {
        Ok(()) => json!({ "ok": true, "path": path }),
        Err(e) => json!({ "error": format!("edit {path}: {e}") }),
    }
}

// ---------------------------------------------------------------------------
// glob
// ---------------------------------------------------------------------------

/// `glob { pattern }` — shell-style matching (`**` supported) under
/// the workspace. Returns `{ files: [...], truncated }`, sorted, capped.
pub fn glob(caps: &Capabilities, args: &Value) -> ToolOutput {
    let Some(pattern) = args.get("pattern").and_then(Value::as_str) else {
        return json!({ "error": "glob: missing 'pattern'" });
    };
    if pattern.split('/').any(|seg| seg == "..") {
        return SandboxError(pattern.to_string()).to_json();
    }
    let root = Path::new(&caps.workspace);
    let matcher = GlobPattern::new(pattern);
    let mut hits = Vec::new();
    let mut truncated = false;
    let mut queue = VecDeque::new();
    queue.push_back(root.to_path_buf());
    while let Some(dir) = queue.pop_front() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            let Ok(ft) = entry.file_type() else { continue };
            let rel = p
                .strip_prefix(root)
                .unwrap_or(&p)
                .to_string_lossy()
                .into_owned();
            if ft.is_dir() {
                if matcher.matches_dir(&rel) {
                    queue.push_back(p);
                }
            } else if matcher.matches(&rel) {
                hits.push(rel);
            }
            if hits.len() > 10_000 {
                truncated = true;
                break;
            }
        }
        if truncated {
            break;
        }
    }
    hits.sort();
    hits.truncate(10_000);
    json!({ "files": hits, "truncated": truncated })
}

/// Minimal `**`/`*`/`?` glob matcher over `/`-separated paths.
struct GlobPattern {
    /// One segment list; a segment of `**` matches any number of dirs.
    segments: Vec<String>,
}

impl GlobPattern {
    fn new(pattern: &str) -> Self {
        Self {
            segments: pattern
                .trim_start_matches('/')
                .split('/')
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
        }
    }

    /// Would a directory with this relative path possibly contain
    /// matches below it? (Prune walk.)
    fn matches_dir(&self, rel: &str) -> bool {
        let walked = rel.split('/').filter(|s| !s.is_empty()).count();
        // Prune hidden dirs (cheap, avoids .git).
        if rel.split('/').any(|s| s.starts_with('.')) {
            return false;
        }
        // If all consumed segments so far matched the pattern prefix
        // (allowing **), descend.
        self.prefix_ok(rel, walked)
    }

    fn prefix_ok(&self, rel: &str, depth: usize) -> bool {
        // Segment-by-segment match of the first `depth` pattern
        // segments, with `**` gobbling.
        fn rec(pats: &[String], segs: &[&str]) -> bool {
            match (pats.split_first(), segs.split_first()) {
                (None, _) => true,
                (Some((p, rest)), None) => p == "**" && rec(rest, &[]),
                (Some((p, rest)), Some((s, srest))) => {
                    if p == "**" {
                        rec(rest, segs) || rec(pats, srest)
                    } else if seg_match(p, s) {
                        rec(rest, srest)
                    } else {
                        false
                    }
                }
            }
        }
        let segs: Vec<&str> = rel.split('/').filter(|s| !s.is_empty()).collect();
        rec(&self.segments[..depth.min(self.segments.len())], &segs)
    }

    fn matches(&self, rel: &str) -> bool {
        fn rec(pats: &[String], segs: &[&str]) -> bool {
            match (pats.split_first(), segs.split_first()) {
                (None, None) => true,
                (None, Some(_)) | (Some(_), None) => false,
                (Some((p, rest)), Some((s, srest))) => {
                    if p == "**" {
                        rec(rest, segs) || rec(pats, srest)
                    } else if seg_match(p, s) {
                        rec(rest, srest)
                    } else {
                        false
                    }
                }
            }
        }
        let segs: Vec<&str> = rel.split('/').filter(|s| !s.is_empty()).collect();
        rec(&self.segments, &segs)
    }
}

fn seg_match(pat: &str, s: &str) -> bool {
    // Classic wildcard DP over one path segment.
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = s.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = pi;
            mark = ti;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

// ---------------------------------------------------------------------------
// grep
// ---------------------------------------------------------------------------

/// `grep { pattern, path? | paths?, max? }` — literal-substring search
/// across files (host-side, spec §9.5). Returns
/// `{ matches: [{ path, line, text }], truncated }`.
pub fn grep(caps: &Capabilities, args: &Value) -> ToolOutput {
    let Some(pattern) = args.get("pattern").and_then(Value::as_str) else {
        return json!({ "error": "grep: missing 'pattern'" });
    };
    let max = args.get("max").and_then(Value::as_u64).unwrap_or(200) as usize;

    // Target selection: `paths` (array) or single `path`; default:
    // whole workspace.
    let mut files: Vec<String> = Vec::new();
    let collect = |p: &str, files: &mut Vec<String>| {
        match caps.resolve_existing_read(p) {
            Ok(abs) => {
                if abs.is_dir() {
                    walk_files(&abs, files);
                } else {
                    files.push(p.to_string());
                }
            }
            Err(_) => files.push(p.to_string()), // read error surfaces later
        }
    };
    if let Some(paths) = args.get("paths").and_then(Value::as_array) {
        for p in paths {
            if let Some(p) = p.as_str() {
                collect(p, &mut files);
            }
        }
    } else if let Some(p) = args.get("path").and_then(Value::as_str) {
        collect(p, &mut files);
    } else {
        walk_files(&caps.workspace, &mut files);
    }
    files.sort();

    let mut matches = Vec::new();
    let mut truncated = false;
    'outer: for f in &files {
        let Ok(abs) = caps.resolve_existing_read(f) else {
            continue;
        };
        let Ok(bytes) = std::fs::read(&abs) else {
            continue;
        };
        let text = String::from_utf8_lossy(&bytes);
        for (i, line) in text.lines().enumerate() {
            if line.contains(pattern) {
                if matches.len() >= max {
                    truncated = true;
                    break 'outer;
                }
                matches.push(json!({ "path": f, "line": i + 1, "text": line }));
            }
        }
    }
    json!({ "matches": matches, "truncated": truncated })
}

fn walk_files(dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            subdirs.push(p);
        } else if file_type.is_file() {
            out.push(p.to_string_lossy().into_owned());
        }
    }
    subdirs.sort();
    for d in subdirs {
        walk_files(&d, out);
    }
}

// ---------------------------------------------------------------------------
// exec
// ---------------------------------------------------------------------------

/// `exec { command: [program, args...], timeout_ms? }` — argv-based
/// subprocess (spec §9.6; no shell interpretation). Output goes to the
/// result store; the JS-visible value carries exit code plus handles.
pub fn exec(
    caps: &Capabilities,
    store: &ResultStore,
    args: &Value,
    result_cap: usize,
    cancel: &dyn Fn() -> bool,
) -> ToolOutput {
    let Some(cmd) = args.get("command").and_then(Value::as_array) else {
        return json!({ "error": "exec: missing 'command' array" });
    };
    let mut argv: Vec<String> = Vec::with_capacity(cmd.len());
    for a in cmd {
        match a {
            Value::String(s) => argv.push(s.clone()),
            Value::Number(n) => argv.push(n.to_string()),
            _ => return json!({ "error": "exec: 'command' entries must be strings" }),
        }
    }
    if argv.is_empty() {
        return json!({ "error": "exec: empty 'command'" });
    }
    if caps.process == ProcessPolicy::Disabled {
        return json!({ "error": "exec: disabled by policy" });
    }
    let timeout_ms = args
        .get("timeout_ms")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_TIMEOUT_MS);
    let timeout = Duration::from_millis(timeout_ms.max(1));

    let mut command = Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .current_dir(&caps.workspace)
        .env_remove("MH_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => return json!({ "error": format!("exec {}: {e}", argv[0]) }),
    };

    let capture_cap = result_cap.min(EXEC_CAP);
    let stdout_reader = child
        .stdout
        .take()
        .map(|mut pipe| std::thread::spawn(move || read_bounded_pipe(&mut pipe, capture_cap)));
    let stderr_reader = child
        .stderr
        .take()
        .map(|mut pipe| std::thread::spawn(move || read_bounded_pipe(&mut pipe, capture_cap)));
    let start = Instant::now();
    let status = loop {
        if cancel() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.map(std::thread::JoinHandle::join);
            let _ = stderr_reader.map(std::thread::JoinHandle::join);
            return json!({ "error": "exec: cancelled" });
        }
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if start.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.map(std::thread::JoinHandle::join);
                let _ = stderr_reader.map(std::thread::JoinHandle::join);
                return json!({
                    "error": ToolError::TimedOut {
                        program: argv[0].clone(),
                        timeout_ms,
                    }
                    .to_string()
                });
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(_) => break None,
        }
    };
    let (stdout, stdout_total) = stdout_reader
        .and_then(|reader| reader.join().ok())
        .unwrap_or_default();
    let (stderr, stderr_total) = stderr_reader
        .and_then(|reader| reader.join().ok())
        .unwrap_or_default();

    let duration_ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let exit_code = status
        .and_then(|status| status.code())
        .map(i64::from)
        .unwrap_or(-1);
    let out_res = store.put_with_total(stdout, stdout_total, capture_cap);
    let err_res = store.put_with_total(stderr, stderr_total, capture_cap);
    json!({
        "exitCode": exit_code,
        "durationMs": duration_ms,
        "stdoutId": out_res.id.0,
        "stderrId": err_res.id.0,
        "stdoutTruncated": out_res.truncated,
        "stderrTruncated": err_res.truncated,
    })
}

fn read_bounded_pipe(reader: &mut impl std::io::Read, cap: usize) -> (Vec<u8>, u64) {
    let mut tail = VecDeque::with_capacity(cap.min(64 * 1024));
    let mut buffer = [0u8; 8192];
    let mut total = 0u64;
    while let Ok(read) = reader.read(&mut buffer) {
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if cap == 0 {
            continue;
        }
        for byte in &buffer[..read] {
            if tail.len() == cap {
                tail.pop_front();
            }
            tail.push_back(*byte);
        }
    }
    (tail.into_iter().collect(), total)
}

/// Reads a range of lines from a stored result.
pub fn handle_read(store: &ResultStore, id: u64, offset: usize, limit: usize) -> Value {
    let Some(res) = store.get(ResultId(id)) else {
        return json!({ "error": format!("unknown result handle {id}") });
    };
    let text = String::from_utf8_lossy(&res.data);
    let lines: Vec<&str> = text.lines().collect();
    let end = offset.saturating_add(limit).min(lines.len());
    let slice = if offset < lines.len() {
        &lines[offset..end]
    } else {
        &[][..]
    };
    json!({
        "content": slice.join("\n"),
        "totalLines": lines.len(),
        "truncated": end < lines.len(),
    })
}

pub fn handle_tail(store: &ResultStore, id: u64, limit: usize) -> Value {
    let Some(res) = store.get(ResultId(id)) else {
        return json!({ "error": format!("unknown result handle {id}") });
    };
    let text = String::from_utf8_lossy(&res.data);
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(limit);
    json!({
        "content": lines[start..].join("\n"),
        "totalLines": lines.len(),
        "truncated": start > 0,
    })
}

pub fn handle_grep(store: &ResultStore, id: u64, pattern: &str, limit: usize) -> Value {
    let Some(res) = store.get(ResultId(id)) else {
        return json!({ "error": format!("unknown result handle {id}") });
    };
    let text = String::from_utf8_lossy(&res.data);
    let mut matches = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.contains(pattern) {
            if matches.len() >= limit {
                return json!({ "matches": matches, "truncated": true });
            }
            matches.push(json!({ "line": i + 1, "text": line }));
        }
    }
    json!({ "matches": matches, "truncated": false })
}

pub fn handle_json(store: &ResultStore, id: u64) -> Value {
    let Some(res) = store.get(ResultId(id)) else {
        return json!({ "error": format!("unknown result handle {id}") });
    };
    serde_json::from_slice(&res.data)
        .unwrap_or_else(|e| json!({ "error": format!("invalid JSON: {e}") }))
}

pub fn handle_length(store: &ResultStore, id: u64) -> Option<usize> {
    store.metadata(ResultId(id)).map(|metadata| metadata.length)
}

pub fn handle_total_bytes(store: &ResultStore, id: u64) -> Option<u64> {
    store
        .metadata(ResultId(id))
        .map(|metadata| metadata.total_bytes)
}

pub fn handle_truncated(store: &ResultStore, id: u64) -> Option<bool> {
    store
        .metadata(ResultId(id))
        .map(|metadata| metadata.truncated)
}

#[allow(dead_code)]
fn _unused(_: &ToolError) -> Value {
    err_json(&ToolError::Cancelled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn caps() -> (Capabilities, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (Capabilities::new(dir.path().to_path_buf()), dir)
    }

    #[test]
    fn write_read_edit_roundtrip() {
        let (caps, _d) = caps();
        let w = write(
            &caps,
            &json!({"path": "a.txt", "content": "hello\nworld\n"}),
        );
        assert_eq!(w["ok"], true);
        let r = read(&caps, &json!({"path": "a.txt"}));
        assert_eq!(r["content"], "hello\nworld\n");
        assert_eq!(r["totalLines"], 2);
        let r2 = read(&caps, &json!({"path": "a.txt", "offset": 1, "limit": 1}));
        assert_eq!(r2["content"], "world\n");
        let e = edit(
            &caps,
            &json!({"path": "a.txt", "old": "world", "new": "mh"}),
        );
        assert_eq!(e["ok"], true);
        let r3 = read(&caps, &json!({"path": "a.txt"}));
        assert_eq!(r3["content"], "hello\nmh\n");
    }

    #[test]
    fn edit_rejects_ambiguous_match() {
        let (caps, _d) = caps();
        write(&caps, &json!({"path": "b.txt", "content": "x x"}));
        let e = edit(&caps, &json!({"path": "b.txt", "old": "x", "new": "y"}));
        assert!(e["error"].as_str().unwrap().contains("times"));
    }

    #[test]
    fn glob_reports_truncation_envelope() {
        let (caps, _d) = caps();
        for index in 0..=10_000 {
            std::fs::write(caps.workspace.join(format!("{index:05}.txt")), "x").unwrap();
        }
        let result = glob(&caps, &json!({"pattern": "*.txt"}));
        assert_eq!(result["files"].as_array().unwrap().len(), 10_000);
        assert_eq!(result["truncated"], true);
    }

    #[test]
    fn glob_finds_nested_files() {
        let (caps, _d) = caps();
        std::fs::create_dir_all(caps.workspace.join("src/deep")).unwrap();
        std::fs::write(caps.workspace.join("src/a.rs"), "fn a(){}").unwrap();
        std::fs::write(caps.workspace.join("src/deep/b.rs"), "fn b(){}").unwrap();
        std::fs::write(caps.workspace.join("src/c.txt"), "no").unwrap();
        let g = glob(&caps, &json!({"pattern": "src/**/*.rs"}));
        let files = g["files"].as_array().unwrap();
        assert!(files.contains(&json!("src/a.rs")));
        assert!(files.contains(&json!("src/deep/b.rs")));
        assert!(!files.contains(&json!("src/c.txt")));
        assert_eq!(g["truncated"], false);
    }

    #[test]
    fn grep_reports_line_numbers() {
        let (caps, _d) = caps();
        write(
            &caps,
            &json!({"path": "g.txt", "content": "alpha\nbeta unsafe\ngamma\nunsafe again"}),
        );
        let g = grep(&caps, &json!({"pattern": "unsafe", "path": "g.txt"}));
        let m = g["matches"].as_array().unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0]["line"], 2);
        assert_eq!(m[1]["line"], 4);
    }

    #[test]
    fn exec_runs_argv() {
        let (caps, _d) = caps();
        let store = ResultStore::new();
        let no_cancel = || false;
        let r = exec(
            &caps,
            &store,
            &json!({"command": ["/bin/sh", "-c", "echo out; echo err 1>&2; exit 3"]}),
            EXEC_CAP,
            &no_cancel,
        );
        assert_eq!(r["exitCode"], 3);
        assert!(r["durationMs"].as_u64().is_some());
        let out = store
            .text(ResultId(r["stdoutId"].as_u64().unwrap()))
            .unwrap();
        assert_eq!(out.trim(), "out");
        let err = store
            .text(ResultId(r["stderrId"].as_u64().unwrap()))
            .unwrap();
        assert_eq!(err.trim(), "err");
    }

    #[test]
    fn exec_enforces_policy_scrubs_secrets_and_reports_metadata() {
        const CHILD_MARKER: &str = "MH_EXEC_SCRUB_TEST_CHILD";
        if std::env::var_os(CHILD_MARKER).is_some() {
            let (caps, _d) = caps();
            let store = ResultStore::new();
            let result = exec(
                &caps,
                &store,
                &json!({"command": ["/bin/sh", "-c", "printf 0123456789; test -z \"$MH_API_KEY\" && test -z \"$OPENAI_API_KEY\" && printf unset:unset err >&2"]}),
                4,
                &|| false,
            );
            assert_eq!(result["exitCode"], 0);
            assert!(result["durationMs"].as_u64().is_some());
            let stdout = store
                .get(ResultId(result["stdoutId"].as_u64().unwrap()))
                .unwrap();
            assert_eq!(stdout.data, b"6789");
            assert_eq!(stdout.total_bytes, 10);
            assert!(stdout.truncated);
            let stderr = store
                .get(ResultId(result["stderrId"].as_u64().unwrap()))
                .unwrap();
            assert!(String::from_utf8_lossy(&stderr.data).ends_with("nset"));
            assert_eq!(stderr.total_bytes, 11);
            assert!(stderr.truncated);
            return;
        }

        let (mut caps, _d) = caps();
        caps.process = ProcessPolicy::Disabled;
        let disabled = exec(
            &caps,
            &ResultStore::new(),
            &json!({"command": ["/bin/echo", "no"]}),
            EXEC_CAP,
            &|| false,
        );
        assert!(disabled["error"].as_str().unwrap().contains("disabled"));

        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tools::fs_tools::tests::exec_enforces_policy_scrubs_secrets_and_reports_metadata",
            ])
            .env(CHILD_MARKER, "1")
            .env("MH_API_KEY", "must-not-leak")
            .env("OPENAI_API_KEY", "must-not-leak")
            .status()
            .unwrap();
        assert!(status.success());
    }
    #[test]
    fn exec_drains_large_output_without_deadlock_and_caps_tail() {
        let (caps, _d) = caps();
        let store = ResultStore::new();
        let no_cancel = || false;
        let result = exec(
            &caps,
            &store,
            &json!({"command": ["/usr/bin/yes", "x"], "timeout_ms": 40}),
            1024,
            &no_cancel,
        );
        assert!(result["error"].as_str().unwrap().contains("timed out"));
    }
    #[test]
    fn exec_rejects_shell_string() {
        let (caps, _d) = caps();
        let store = ResultStore::new();
        let no_cancel = || false;
        let r = exec(
            &caps,
            &store,
            &json!({"command": "echo hi"}),
            EXEC_CAP,
            &no_cancel,
        );
        assert!(r["error"].as_str().unwrap().contains("command"));
    }
}
