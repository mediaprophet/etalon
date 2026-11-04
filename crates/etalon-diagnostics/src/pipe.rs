use std::io::Read;

pub(crate) trait ReadyPipe: Read {
    fn available(&self) -> Result<Option<usize>, String>;
}

pub(crate) fn drain(
    pipe: &mut impl ReadyPipe,
    bytes: &mut Vec<u8>,
    limit: usize,
) -> Result<bool, String> {
    let mut chunk = [0; 4096];
    for _ in 0..64 {
        let Some(available) = pipe.available()? else {
            return Ok(true);
        };
        if available == 0 {
            return Ok(false);
        }
        let count = pipe
            .read(&mut chunk[..available.min(4096)])
            .map_err(|e| e.to_string())?;
        if count == 0 {
            return Ok(true);
        }
        if count > limit.saturating_sub(bytes.len()) {
            return Err("worker output budget exceeded".into());
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    Ok(false)
}

#[cfg(windows)]
fn available(pipe: &impl std::os::windows::io::AsRawHandle) -> Result<Option<usize>, String> {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn PeekNamedPipe(
            pipe: *mut c_void,
            buffer: *mut c_void,
            size: u32,
            read: *mut u32,
            available: *mut u32,
            left: *mut u32,
        ) -> i32;
    }
    let mut count = 0;
    // SAFETY: The owned anonymous read pipe stays live, and only this thread reads it.
    let ok = unsafe {
        PeekNamedPipe(
            pipe.as_raw_handle(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            &mut count,
            std::ptr::null_mut(),
        )
    };
    if ok != 0 {
        return Ok(Some(count as usize));
    }
    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(109 | 233) => Ok(None),
        _ => Err(format!("inspect worker pipe: {error}")),
    }
}

#[cfg(unix)]
fn available(pipe: &impl std::os::fd::AsRawFd) -> Result<Option<usize>, String> {
    use std::os::raw::{c_int, c_short};
    #[repr(C)]
    struct PollFd {
        fd: c_int,
        events: c_short,
        revents: c_short,
    }
    #[cfg(any(target_os = "linux", target_os = "solaris", target_os = "illumos"))]
    type Nfds = std::os::raw::c_ulong;
    #[cfg(not(any(target_os = "linux", target_os = "solaris", target_os = "illumos")))]
    type Nfds = std::os::raw::c_uint;
    unsafe extern "C" {
        fn poll(fds: *mut PollFd, count: Nfds, timeout: c_int) -> c_int;
    }
    let mut descriptor = PollFd {
        fd: pipe.as_raw_fd(),
        events: 1,
        revents: 0,
    };
    // SAFETY: Exclusive ownership prevents another reader consuming readiness before read().
    let ready = unsafe { poll(&mut descriptor, 1, 0) };
    if ready < 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::Interrupted {
            return Ok(Some(0));
        }
        return Err(format!("inspect worker pipe: {error}"));
    }
    if descriptor.revents & 1 != 0 {
        return Ok(Some(4096));
    }
    if descriptor.revents & 0x10 != 0 {
        return Ok(None);
    }
    if descriptor.revents & (0x8 | 0x20) != 0 {
        return Err("worker pipe failed".into());
    }
    Ok(Some(0))
}

#[cfg(not(any(windows, unix)))]
fn available(_pipe: &impl Read) -> Result<Option<usize>, String> {
    Err("nonblocking worker pipes unsupported on this platform".into())
}

impl ReadyPipe for std::process::ChildStdout {
    fn available(&self) -> Result<Option<usize>, String> {
        available(self)
    }
}
impl ReadyPipe for std::process::ChildStderr {
    fn available(&self) -> Result<Option<usize>, String> {
        available(self)
    }
}
