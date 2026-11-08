use crate::runner::ArtifactSink;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub struct BoundedArtifacts {
    directory: tempfile::TempDir,
    budget: u64,
    written: u64,
}
impl BoundedArtifacts {
    pub fn new(parent: &Path, byte_budget: u64) -> Result<Self, String> {
        if byte_budget < 64 {
            return Err("artifact byte budget too small".into());
        }
        let directory = tempfile::Builder::new()
            .prefix("etalon-p64-lab-")
            .tempdir_in(parent)
            .map_err(|e| e.to_string())?;
        let mut sink = Self {
            directory,
            budget: byte_budget,
            written: 0,
        };
        sink.write("ETALON-LAB-RUN", b"etalon-p64-lab-v1\n")?;
        Ok(sink)
    }
    pub fn path(&self) -> &Path {
        self.directory.path()
    }
    pub fn written_bytes(&self) -> u64 {
        self.written
    }
    pub fn write_json(&mut self, name: &str, value: &impl serde::Serialize) -> Result<(), String> {
        self.write_with(name, |writer| {
            serde_json::to_writer(writer, value).map_err(io::Error::other)
        })
    }
    fn write_with(
        &mut self,
        name: &str,
        write: impl FnOnce(&mut dyn Write) -> io::Result<()>,
    ) -> Result<(), String> {
        if name.is_empty()
            || name.len() > 160
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.')
        {
            return Err("invalid artifact name".into());
        }
        let path = self.directory.path().join(name);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| e.to_string())?;
        let mut bounded = LimitedWriter {
            file: &mut file,
            remaining: self.budget - self.written,
            written: 0,
        };
        let outcome = write(&mut bounded);
        let written = bounded.written;
        drop(bounded);
        if let Err(error) = outcome.and_then(|_| file.sync_all()) {
            drop(file);
            std::fs::remove_file(path)
                .map_err(|e| format!("{error}; partial artifact cleanup failed: {e}"))?;
            return Err(error.to_string());
        }
        self.written += written;
        Ok(())
    }
    /// Retain only into a new caller-selected destination on the same filesystem.
    pub fn retain(self, destination: &Path) -> Result<PathBuf, String> {
        if destination.exists() {
            return Err("artifact destination already exists".into());
        }
        std::fs::rename(self.directory.path(), destination).map_err(|e| e.to_string())?;
        Ok(destination.to_owned())
    }
}
impl ArtifactSink for BoundedArtifacts {
    fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), String> {
        self.write_with(name, |writer| writer.write_all(bytes))
    }
}
struct LimitedWriter<'a> {
    file: &'a mut std::fs::File,
    remaining: u64,
    written: u64,
}
impl Write for LimitedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() as u64 > self.remaining {
            return Err(io::Error::other("artifact byte budget exhausted"));
        }
        let count = self.file.write(bytes)?;
        self.remaining -= count as u64;
        self.written += count as u64;
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}
