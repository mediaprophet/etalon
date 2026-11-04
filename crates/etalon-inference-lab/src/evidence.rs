use crate::experiment::ExperimentResult;
use crate::runner::ArtifactSink;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
pub const DEFAULT_ARTIFACT_BUDGET: u64 = 64 * 1024 * 1024;
pub const MAX_LOG_RECORDS: usize = 100_000;
static APPEND_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
pub(crate) struct AppendOwner {
    _serial: std::sync::MutexGuard<'static, ()>,
    marker: std::path::PathBuf,
}
impl Drop for AppendOwner {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.marker);
    }
}
pub(crate) fn own_append(path: &Path) -> Result<AppendOwner, String> {
    let serial = APPEND_MUTEX.lock().map_err(|_| "append owner poisoned")?;
    let mut marker = path.as_os_str().to_owned();
    marker.push(".etalon-append-owner");
    let marker = PathBuf::from(marker);
    if let Some(parent) = marker.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut file=std::fs::OpenOptions::new().create_new(true).write(true).open(&marker).map_err(|e|format!("artifact already has an append owner (interrupted owners require explicit recovery): {e}"))?;
    let owner = AppendOwner {
        _serial: serial,
        marker,
    };
    file.write_all(b"etalon-inference-lab append owner v1\n")
        .map_err(|e| e.to_string())?;
    Ok(owner)
}

pub fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    if f.metadata().map_err(|e| e.to_string())?.len() > limit {
        return Err("artifact byte budget exceeded".into());
    }
    let mut bytes = Vec::new();
    f.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("artifact byte budget exceeded".into());
    }
    Ok(bytes)
}
pub fn atomic_write(path: &Path, bytes: &[u8], limit: u64) -> Result<(), String> {
    if bytes.len() as u64 > limit {
        return Err("artifact byte budget exceeded".into());
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut staged = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    staged
        .write_all(bytes)
        .and_then(|_| staged.as_file().sync_all())
        .map_err(|e| e.to_string())?;
    staged.persist(path).map_err(|e| e.to_string())?;
    Ok(())
}
pub struct BoundedBytes {
    pub bytes: Vec<u8>,
    pub limit: u64,
}
impl Write for BoundedBytes {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        if (self.bytes.len() as u64).saturating_add(data.len() as u64) > self.limit {
            return Err(std::io::Error::other("serialization byte budget exhausted"));
        }
        self.bytes.extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
pub fn serialize_bounded<T: serde::Serialize>(value: &T, limit: u64) -> Result<Vec<u8>, String> {
    let mut writer = BoundedBytes {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut writer, value).map_err(|e| e.to_string())?;
    Ok(writer.bytes)
}
pub fn append_experiment_jsonl(path: &Path, result: &ExperimentResult) -> Result<(), String> {
    append_experiment_jsonl_with_budget(path, result, DEFAULT_ARTIFACT_BUDGET)
}
pub fn append_experiment_jsonl_with_budget(
    path: &Path,
    result: &ExperimentResult,
    limit: u64,
) -> Result<(), String> {
    let mut line = serialize_bounded(result, limit.saturating_sub(1))?;
    line.push(b'\n');
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let _owner = own_append(path)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| e.to_string())?;

    if file
        .metadata()
        .map_err(|e| e.to_string())?
        .len()
        .saturating_add(line.len() as u64)
        > limit
    {
        return Err("experiment log byte budget exceeded".into());
    }
    file.write_all(&line)
        .and_then(|_| file.sync_data())
        .map_err(|e| e.to_string())
}
pub fn load_experiment_log(path: &Path) -> Result<Vec<ExperimentResult>, String> {
    load_experiment_log_with_budget(path, DEFAULT_ARTIFACT_BUDGET, MAX_LOG_RECORDS)
}
pub fn load_experiment_log_with_budget(
    path: &Path,
    limit: u64,
    max_records: usize,
) -> Result<Vec<ExperimentResult>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = read_bounded(path, limit)?;
    let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    let mut records = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if records.len() >= max_records {
            return Err("experiment record budget exceeded".into());
        }
        records.push(
            serde_json::from_str(line)
                .map_err(|e| format!("incomplete experiment log line {}: {e}", i + 1))?,
        );
    }
    Ok(records)
}
#[derive(Debug)]
pub struct DirectoryArtifactSink {
    root: PathBuf,
    remaining: u64,
}
impl DirectoryArtifactSink {
    pub fn new(root: PathBuf, budget: u64) -> Self {
        Self {
            root,
            remaining: budget,
        }
    }
}
impl ArtifactSink for DirectoryArtifactSink {
    fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), String> {
        let relative = Path::new(name);
        if relative
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            return Err("artifact name must be relative normal components".into());
        }
        if bytes.len() as u64 > self.remaining {
            return Err("artifact budget exhausted".into());
        }
        atomic_write(&self.root.join(relative), bytes, self.remaining)?;
        self.remaining -= bytes.len() as u64;
        Ok(())
    }
}
