use std::{fs::{File, OpenOptions}, path::{Component, Path}};
use crate::DeploymentError;

pub(crate) fn admit_path(path: &Path) -> Result<(), DeploymentError> {
    if !path.is_absolute() || path.as_os_str().len() > 4096
        || path.components().any(|part| matches!(part, Component::ParentDir)) {
        return Err(DeploymentError::InvalidPath);
    }
    #[cfg(windows)] {
        use std::path::Prefix;
        if !matches!(path.components().next(), Some(Component::Prefix(prefix))
            if matches!(prefix.kind(), Prefix::Disk(_))) { return Err(DeploymentError::InvalidPath); }
        for component in path.components() {
            if let Component::Normal(name) = component {
                let name = name.to_str().ok_or(DeploymentError::InvalidPath)?;
                if name.contains(':') || name.ends_with(' ') || name.ends_with('.') { return Err(DeploymentError::InvalidPath); }
                let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
                if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CLOCK$" | "CONIN$" | "CONOUT$")
                    || matches!(stem.strip_prefix("COM").or_else(|| stem.strip_prefix("LPT")),
                        Some("0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³")) {
                    return Err(DeploymentError::InvalidPath);
                }
            }
        }
    }
    #[cfg(unix)] {
        if ["/dev", "/proc", "/sys"].iter().any(|root| path.starts_with(root)) {
            return Err(DeploymentError::InvalidPath);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_relative_and_parent_paths() {
        assert!(admit_path(Path::new("runtime.dll")).is_err());
        assert!(admit_path(&std::env::temp_dir().join("../runtime.dll")).is_err());
    }
    #[cfg(windows)]
    #[test]
    fn rejects_remote_and_reserved_names_without_probe() {
        for path in [r"\\server\share\runtime.dll", r"\\?\UNC\server\runtime.dll",
            r"C:\local\NUL.dll", r"C:\local\COM¹.dll", r"C:\local\file:stream"] {
            assert!(admit_path(Path::new(path)).is_err());
        }
        assert!(admit_path(Path::new(r"C:\local\runtime.dll")).is_ok());
    }
    #[cfg(unix)]
    #[test]
    fn rejects_common_device_roots() {
        assert!(admit_path(Path::new("/dev/zero")).is_err());
        assert!(admit_path(Path::new("/proc/self/mem")).is_err());
    }
}

pub(crate) fn open_regular(path: &Path) -> Result<File, DeploymentError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)] {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    }
    #[cfg(windows)] {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x0020_0000);
    }
    #[cfg(not(any(unix, windows)))]
    return Err(DeploymentError::UnsupportedPlatform);
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() { return Err(DeploymentError::NotRegular); }
    #[cfg(windows)] {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 { return Err(DeploymentError::InvalidPath); }
    }
    Ok(file)
}
