use core::time::Duration;

// glow accepts signed nanoseconds; preserve the caller deadline across chunks.
pub(super) fn wait_with_deadline<E>(
    timeout: Option<Duration>,
    mut elapsed: impl FnMut() -> Duration,
    mut wait: impl FnMut(i32) -> Result<bool, E>,
) -> Result<bool, E> {
    let mut first = true;
    loop {
        let remaining = timeout.map(|limit| limit.saturating_sub(elapsed()));
        if !first && remaining == Some(Duration::ZERO) {
            return Ok(false);
        }
        let chunk = remaining.map_or(i32::MAX, |left| {
            left.as_nanos().min(i32::MAX as u128) as i32
        });
        if wait(chunk)? {
            return Ok(true);
        }
        if remaining == Some(Duration::ZERO) {
            return Ok(false);
        }
        first = false;
    }
}

#[cfg(test)]
#[path = "fence_wait_tests.rs"]
mod tests;
