use super::{AdmissionLimits, ReservationError};
use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct ReservationPool<const CLASSES: usize> {
    total: AtomicU64,
    classes: [AtomicU64; CLASSES],
}

impl<const CLASSES: usize> Default for ReservationPool<CLASSES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CLASSES: usize> ReservationPool<CLASSES> {
    pub const fn new() -> Self {
        Self {
            total: AtomicU64::new(0),
            classes: [const { AtomicU64::new(0) }; CLASSES],
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.total.load(Ordering::Acquire)
    }

    pub fn class_bytes(&self, class: usize) -> Option<u64> {
        self.classes
            .get(class)
            .map(|bytes| bytes.load(Ordering::Acquire))
    }

    /// Zero bytes consumes no capacity but still requires a valid class and a
    /// total within the supplied limits. Retain the guard for the allocation
    /// lifetime; it accounts for bytes, not hardware completion.
    pub fn try_reserve(
        &self,
        class: usize,
        bytes: u64,
        limits: AdmissionLimits,
    ) -> Result<Reservation<'_, CLASSES>, ReservationError> {
        let counter = self
            .classes
            .get(class)
            .ok_or(ReservationError::InvalidClass {
                class,
                classes: CLASSES,
            })?;
        let capacity = limits.remaining_capacity();
        let mut current = self.total.load(Ordering::Acquire);
        loop {
            let available_bytes = capacity.saturating_sub(current);
            let next = current
                .checked_add(bytes)
                .ok_or(ReservationError::ArithmeticOverflow {
                    requested_bytes: bytes,
                    available_bytes,
                })?;
            if next > capacity {
                return Err(ReservationError::BudgetExceeded {
                    requested_bytes: bytes,
                    available_bytes,
                });
            }
            match self.total.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Aggregate admission bounds each class; release decrements its
                    // class before the aggregate, so this addition cannot overflow.
                    counter.fetch_add(bytes, Ordering::Relaxed);
                    return Ok(Reservation {
                        pool: self,
                        class,
                        bytes,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

#[must_use = "dropping the guard releases its byte reservation"]
#[derive(Debug)]
pub struct Reservation<'a, const CLASSES: usize> {
    pool: &'a ReservationPool<CLASSES>,
    class: usize,
    bytes: u64,
}

impl<const CLASSES: usize> Reservation<'_, CLASSES> {
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    pub const fn class(&self) -> usize {
        self.class
    }
}

impl<const CLASSES: usize> Drop for Reservation<'_, CLASSES> {
    fn drop(&mut self) {
        self.pool.classes[self.class].fetch_sub(self.bytes, Ordering::AcqRel);
        self.pool.total.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
