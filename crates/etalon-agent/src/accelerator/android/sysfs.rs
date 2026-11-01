use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use super::super::{NpuProbeReport, ProbeStatus, MAX_PROPERTY_BYTES};

pub(super) const MAX_ENTRIES: usize = 256;
pub(super) const MAX_BYTES: usize = 256 * 1024;
const MAX_PATH_BYTES: usize = 4096;

pub(super) struct Scan {
    pub report: NpuProbeReport,
    root: PathBuf,
    entries_left: usize,
    bytes_left: usize,
    pub directories_read: usize,
}

impl Scan {
    pub fn new(root: &Path) -> Self {
        let mut report =
            NpuProbeReport::new("android", "sysfs-accel+platform+edgetpu-class+dev-metadata");
        report.status = ProbeStatus::Complete;
        let root = match root.canonicalize() {
            Ok(root) => root,
            Err(error) => {
                report.status = ProbeStatus::Partial;
                report.issue("sysfs_root", error.to_string());
                root.to_path_buf()
            }
        };
        Self {
            report,
            root,
            entries_left: MAX_ENTRIES,
            bytes_left: MAX_BYTES,
            directories_read: 0,
        }
    }

    pub fn issue(&mut self, operation: &'static str, message: impl AsRef<str>) {
        self.report.status = ProbeStatus::Partial;
        self.report.issue(operation, message);
    }

    fn charge(&mut self, bytes: usize) -> bool {
        if bytes > self.bytes_left {
            self.issue("byte_limit", "Android discovery byte budget exhausted");
            false
        } else {
            self.bytes_left -= bytes;
            true
        }
    }

    pub fn entries(&mut self, path: &Path) -> Vec<PathBuf> {
        let Some(path) = self.canonical(path) else {
            return Vec::new();
        };
        let entries = match fs::read_dir(&path) {
            Ok(entries) => {
                self.directories_read += 1;
                entries
            }
            Err(error) => {
                self.issue("read_dir", format!("{}: {error}", path.display()));
                return Vec::new();
            }
        };
        let mut result = Vec::new();
        for entry in entries {
            if self.entries_left == 0 {
                self.issue(
                    "entry_limit",
                    "Android discovery directory entry budget exhausted",
                );
                break;
            }
            self.entries_left -= 1;
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    self.issue("read_dir_entry", error.to_string());
                    continue;
                }
            };
            let path = entry.path();
            let bytes = path.as_os_str().len();
            if bytes > MAX_PATH_BYTES || !self.charge(bytes) {
                self.issue("path_limit", "Android discovery path exceeds byte budget");
                break;
            }
            result.push(path);
        }
        result.sort();
        result
    }

    pub fn canonical(&mut self, path: &Path) -> Option<PathBuf> {
        let resolved = match path.canonicalize() {
            Ok(path) => path,
            Err(error) => {
                self.issue("device_path", format!("{}: {error}", path.display()));
                return None;
            }
        };
        if !resolved.starts_with(&self.root) {
            self.issue(
                "sysfs_boundary",
                "discovery link leaves the configured sysfs root",
            );
            return None;
        }
        let size = resolved.as_os_str().len();
        if size > MAX_PATH_BYTES || !self.charge(size) {
            self.issue("path_limit", "resolved sysfs path exceeds byte budget");
            return None;
        }
        Some(resolved)
    }

    pub fn text(&mut self, path: &Path) -> Option<String> {
        // Missing optional properties are normal; unreadable or oversized properties are not.
        match fs::symlink_metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
            Err(error) => {
                self.issue("property_metadata", error.to_string());
                return None;
            }
            Ok(_) => {}
        }
        let path = self.canonical(path)?;
        if !self.charge(MAX_PROPERTY_BYTES + 1) {
            return None;
        }
        let read = || -> std::io::Result<Vec<u8>> {
            let mut bytes = Vec::with_capacity(MAX_PROPERTY_BYTES + 1);
            fs::File::open(path)?
                .take((MAX_PROPERTY_BYTES + 1) as u64)
                .read_to_end(&mut bytes)?;
            Ok(bytes)
        };
        match read() {
            Ok(bytes) if bytes.len() <= MAX_PROPERTY_BYTES => match String::from_utf8(bytes) {
                Ok(text) => Some(text.trim().to_owned()),
                Err(error) => {
                    self.issue("property_utf8", error.to_string());
                    None
                }
            },
            Ok(_) => {
                self.issue("property_limit", "sysfs property exceeds 4096 bytes");
                None
            }
            Err(error) => {
                self.issue("property_read", error.to_string());
                None
            }
        }
    }

    pub fn driver(&mut self, path: &Path) -> Option<String> {
        match fs::read_link(path.join("driver")) {
            Ok(link) => {
                if !self.charge(link.as_os_str().len()) {
                    return None;
                }
                self.canonical(&path.join("driver"))?;
                link.file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_owned)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => self
                .text(&path.join("uevent"))?
                .lines()
                .find_map(|line| line.strip_prefix("DRIVER=").map(str::to_owned)),
            Err(error) => {
                self.issue("driver_link", error.to_string());
                None
            }
        }
    }
}
