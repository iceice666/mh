//! Result Store (spec §10, §11).
//!
//! Large tool outputs live here, not in model context. A store may be
//! execution-local (`new`) or session-persistent (`persistent`).

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

/// Handle to a stored result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResultId(pub u64);

/// One stored blob (usually exec stdout/stderr or a file snapshot).
#[derive(Debug, Clone)]
pub struct StoredResult {
    pub id: ResultId,
    pub data: Vec<u8>,
    /// Truncated by the tool if it exceeded the byte cap.
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedResult {
    id: u64,
    hash: String,
    truncated: bool,
}

#[derive(Debug, Default)]
struct Inner {
    next: u64,
    map: HashMap<u64, StoredResult>,
    persistent_dir: Option<PathBuf>,
}

/// Session-scoped store. Cheap to clone (`Arc`).
#[derive(Debug, Clone, Default)]
pub struct ResultStore {
    inner: Arc<Mutex<Inner>>,
}

impl ResultStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                next: 1,
                ..Inner::default()
            })),
        }
    }

    /// Opens a session-persistent store under `.mh/blobs`.
    pub fn persistent(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let mut inner = Inner {
            next: 1,
            map: HashMap::new(),
            persistent_dir: Some(dir.clone()),
        };
        let index = dir.join("results.jsonl");
        if let Ok(text) = std::fs::read_to_string(index) {
            for line in text.lines() {
                let Ok(meta) = serde_json::from_str::<PersistedResult>(line) else {
                    break;
                };
                let Ok(data) = std::fs::read(dir.join(&meta.hash)) else {
                    continue;
                };
                inner.next = inner.next.max(meta.id.saturating_add(1));
                inner.map.insert(
                    meta.id,
                    StoredResult {
                        id: ResultId(meta.id),
                        data,
                        truncated: meta.truncated,
                    },
                );
            }
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Stores `data`; returns the handle. Data larger than `cap` keeps
    /// its tail, which usually contains the failure evidence.
    pub fn put(&self, data: Vec<u8>, cap: usize) -> StoredResult {
        let truncated = data.len() > cap;
        let data = if truncated {
            data[data.len().saturating_sub(cap)..].to_vec()
        } else {
            data
        };
        let mut guard = self.inner.lock().expect("result store poisoned");
        let id = guard.next.max(1);
        guard.next = id.saturating_add(1);
        let result = StoredResult {
            id: ResultId(id),
            data,
            truncated,
        };
        if let Some(dir) = guard.persistent_dir.as_deref() {
            let _ = persist(dir, &result);
        }
        guard.map.insert(id, result.clone());
        result
    }

    pub fn get(&self, id: ResultId) -> Option<StoredResult> {
        self.inner
            .lock()
            .expect("result store poisoned")
            .map
            .get(&id.0)
            .cloned()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("result store poisoned").map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn text(&self, id: ResultId) -> Option<String> {
        self.get(id)
            .map(|r| String::from_utf8_lossy(&r.data).into_owned())
    }
}

fn persist(dir: &Path, result: &StoredResult) -> std::io::Result<()> {
    let hash = fnv1a_hex(&result.data);
    let blob = dir.join(&hash);
    if !blob.exists() {
        let temp = dir.join(format!(".{hash}.tmp"));
        std::fs::write(&temp, &result.data)?;
        std::fs::rename(temp, blob)?;
    }
    let meta = PersistedResult {
        id: result.id.0,
        hash,
        truncated: result.truncated,
    };
    let mut index = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("results.jsonl"))?;
    serde_json::to_writer(&mut index, &meta)?;
    index.write_all(b"\n")?;
    index.sync_data()
}

fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_to_cap_keeping_tail() {
        let store = ResultStore::new();
        let result = store.put(b"0123456789abcdef".to_vec(), 8);
        assert!(result.truncated);
        assert_eq!(result.data, b"89abcdef");
        assert_eq!(store.text(result.id).unwrap(), "89abcdef");
    }

    #[test]
    fn roundtrips_small_results() {
        let store = ResultStore::new();
        let result = store.put(b"hello".to_vec(), 1024);
        assert!(!result.truncated);
        assert_eq!(store.text(result.id).unwrap(), "hello");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn persistent_store_recovers_handles() {
        let dir = tempfile::tempdir().unwrap();
        let id = {
            let store = ResultStore::persistent(dir.path()).unwrap();
            store.put(b"durable".to_vec(), 1024).id
        };
        let reopened = ResultStore::persistent(dir.path()).unwrap();
        assert_eq!(reopened.text(id).unwrap(), "durable");
    }
}
