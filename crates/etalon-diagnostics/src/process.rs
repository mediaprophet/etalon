use crate::pipe::drain;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct WorkerLimits {
    pub timeout: Duration,
    pub stdout_bytes: usize,
    pub stderr_bytes: usize,
}
impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            stdout_bytes: 1024 * 1024,
            stderr_bytes: 64 * 1024,
        }
    }
}
pub struct WorkerOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub status: ExitStatus,
}
struct OwnedChild(Child);
impl OwnedChild {
    fn terminate(&mut self) -> Result<(), String> {
        if matches!(self.0.try_wait(), Ok(Some(_))) {
            return Ok(());
        }
        if let Err(error) = self.0.kill() {
            if matches!(self.0.try_wait(), Ok(Some(_))) {
                return Ok(());
            }
            return Err(format!("kill worker failed; child not reaped: {error}"));
        }
        let started = Instant::now();
        loop {
            match self.0.try_wait() {
                Ok(Some(_)) => return Ok(()),
                Err(error) => return Err(format!("reap worker: {error}")),
                Ok(None) => {}
            }
            if started.elapsed() >= Duration::from_secs(1) {
                return Err("worker not reaped within cleanup grace period".into());
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if let Err(error) = self.terminate() {
            eprintln!("diagnostic worker cleanup failed: {error}");
        }
    }
}

/// Owns both pipes directly; no reader threads survive a deadline or unwind.
pub fn run_worker(
    command: &mut Command,
    limits: &WorkerLimits,
    cancelled: &dyn Fn() -> bool,
    check_artifacts: &dyn Fn() -> Result<(), String>,
) -> Result<WorkerOutput, String> {
    if limits.timeout.is_zero() || limits.stdout_bytes == 0 || limits.stderr_bytes == 0 {
        return Err("invalid worker limits".into());
    }
    if cancelled() {
        return Err("worker cancelled before launch".into());
    }
    check_artifacts()?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut out = Vec::new();
    let mut err = Vec::new();
    out.try_reserve_exact(limits.stdout_bytes)
        .map_err(|e| format!("stdout byte budget: {e}"))?;
    err.try_reserve_exact(limits.stderr_bytes)
        .map_err(|e| format!("stderr byte budget: {e}"))?;
    let started = Instant::now();
    let mut child = OwnedChild(command.spawn().map_err(|e| format!("launch worker: {e}"))?);
    let result = (|| {
        let mut stdout = child.0.stdout.take().ok_or("worker stdout missing")?;
        let mut stderr = child.0.stderr.take().ok_or("worker stderr missing")?;
        let mut status = None;
        let (mut out_done, mut err_done) = (false, false);
        loop {
            if cancelled() {
                return Err("worker cancelled".into());
            }
            if started.elapsed() >= limits.timeout {
                return Err("worker deadline exceeded".into());
            }
            check_artifacts()?;
            if !out_done {
                out_done = drain(&mut stdout, &mut out, limits.stdout_bytes)?;
            }
            if !err_done {
                err_done = drain(&mut stderr, &mut err, limits.stderr_bytes)?;
            }
            if status.is_none() {
                status = child
                    .0
                    .try_wait()
                    .map_err(|e| format!("wait for worker: {e}"))?;
            }
            if let Some(exit) = status {
                if out_done && err_done {
                    if !exit.success() {
                        return Err(format!(
                            "worker exited {exit}: {}",
                            String::from_utf8_lossy(&err)
                        ));
                    }
                    return Ok(WorkerOutput {
                        stdout: out,
                        stderr: err,
                        status: exit,
                    });
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    })();
    if let Err(error) = result {
        return match child.terminate() {
            Ok(()) => Err(error),
            Err(cleanup) => Err(format!("{error}; cleanup failed: {cleanup}")),
        };
    }
    result
}
