use core::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContextError { Busy, Stale, Open }

/// One resident-module context. Native handles are numbers, never pointers.
pub(crate) struct NativeContext {
    lock: AtomicU32,
    highest_generation: AtomicU32,
    active_generation: AtomicU32,
    highest_request_low: AtomicU32,
    highest_request_high: AtomicU32,
}

impl NativeContext {
    pub(crate) const fn new() -> Self {
        Self { lock: AtomicU32::new(0), highest_generation: AtomicU32::new(0),
            active_generation: AtomicU32::new(0), highest_request_low: AtomicU32::new(0),
            highest_request_high: AtomicU32::new(0) }
    }

    fn acquire(&self) -> Result<Lease<'_>, ContextError> {
        self.lock.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| ContextError::Busy)?;
        Ok(Lease(&self.lock))
    }

    pub(crate) fn open(&self, generation: u32) -> Result<u64, ContextError> {
        let _lease = self.acquire()?;
        if self.active_generation.load(Ordering::Relaxed) != 0 { return Err(ContextError::Open); }
        if generation == 0 || generation <= self.highest_generation.load(Ordering::Relaxed) {
            return Err(ContextError::Stale);
        }
        self.highest_generation.store(generation, Ordering::Relaxed);
        self.active_generation.store(generation, Ordering::Relaxed);
        self.highest_request_low.store(0, Ordering::Relaxed);
        self.highest_request_high.store(0, Ordering::Relaxed);
        Ok(u64::from(generation))
    }

    pub(crate) fn close(&self, handle: u64) -> Result<(), ContextError> {
        let _lease = self.acquire()?;
        self.validate(handle)?;
        self.active_generation.store(0, Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn with_active<T>(&self, handle: u64, operation: impl FnOnce(&mut ActiveContext<'_>) -> T)
        -> Result<T, ContextError> {
        let _lease = self.acquire()?;
        let generation = self.validate(handle)?;
        let mut active = ActiveContext { generation, request_low: &self.highest_request_low,
            request_high: &self.highest_request_high };
        Ok(operation(&mut active))
    }

    fn validate(&self, handle: u64) -> Result<u32, ContextError> {
        let generation = self.active_generation.load(Ordering::Relaxed);
        if generation == 0 || handle != u64::from(generation) { return Err(ContextError::Stale); }
        Ok(generation)
    }
}

/// References cannot escape the synchronous lifecycle lease.
pub(crate) struct ActiveContext<'a> {
    generation: u32,
    request_low: &'a AtomicU32,
    request_high: &'a AtomicU32,
}

impl ActiveContext<'_> {
    pub(crate) fn generation(&self) -> u32 { self.generation }

    /// Call only after validating the complete request and before kernel work.
    pub(crate) fn consume_request(&mut self, request_id: u64) -> Result<(), ContextError> {
        let highest = (u64::from(self.request_high.load(Ordering::Relaxed)) << 32)
            | u64::from(self.request_low.load(Ordering::Relaxed));
        if request_id == 0 || request_id <= highest { return Err(ContextError::Stale); }
        self.request_low.store(request_id as u32, Ordering::Relaxed);
        self.request_high.store((request_id >> 32) as u32, Ordering::Relaxed);
        Ok(())
    }
}

struct Lease<'a>(&'a AtomicU32);
impl Drop for Lease<'_> {
    fn drop(&mut self) { self.0.store(0, Ordering::Release); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifetime_tokens_and_high_water_never_reuse() {
        let context = NativeContext::new();
        let first = context.open(7).unwrap();
        assert_eq!(context.open(8), Err(ContextError::Open));
        assert_eq!(context.close(first | (1u64 << 32)), Err(ContextError::Stale));
        assert_eq!(context.with_active(first, |active| active.generation()), Ok(7));
        context.close(first).unwrap();
        assert_eq!(context.open(7), Err(ContextError::Stale));
        assert_eq!(context.open(6), Err(ContextError::Stale));
        let second = context.open(8).unwrap();
        assert_eq!(context.with_active(first, |_| ()), Err(ContextError::Stale));
        assert_eq!(context.with_active(second, |_| context.close(second)), Ok(Err(ContextError::Busy)));
        context.close(second).unwrap();
    }

    #[test]
    fn exhausted_generation_fails_instead_of_wrapping() {
        let context = NativeContext::new();
        let handle = context.open(u32::MAX).unwrap();
        context.close(handle).unwrap();
        assert_eq!(context.open(u32::MAX), Err(ContextError::Stale));
        assert_eq!(context.open(1), Err(ContextError::Stale));
    }

    #[test]
    fn request_high_water_rejects_duplicates_and_resets_only_on_fresh_open() {
        let context = NativeContext::new();
        let handle = context.open(1).unwrap();
        assert_eq!(context.with_active(handle, |active| active.consume_request(0)), Ok(Err(ContextError::Stale)));
        assert_eq!(context.with_active(handle, |active| active.consume_request(9)), Ok(Ok(())));
        for stale in [1, 8, 9] {
            assert_eq!(context.with_active(handle, |active| active.consume_request(stale)), Ok(Err(ContextError::Stale)));
        }
        // Kernel failure after admission still consumes its request ID.
        assert_eq!(context.with_active(handle, |active| {
            active.consume_request(10).unwrap();
            Err::<(), i32>(-7)
        }), Ok(Err(-7)));
        assert_eq!(context.with_active(handle, |active| active.consume_request(10)), Ok(Err(ContextError::Stale)));
        assert_eq!(context.with_active(handle, |active| active.consume_request(u64::MAX)), Ok(Ok(())));
        assert_eq!(context.with_active(handle, |active| active.consume_request(u64::MAX)), Ok(Err(ContextError::Stale)));
        context.close(handle).unwrap();
        let fresh = context.open(2).unwrap();
        assert_eq!(context.with_active(fresh, |active| active.consume_request(1)), Ok(Ok(())));
    }
}
