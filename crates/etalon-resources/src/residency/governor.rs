use super::{ResidencyError, ResidencyPoolCaps, ResidencySlot};

/// Fixed-capacity Tier-1 zero-heap governor for CPU and GPU VRAM residency.
#[derive(Debug, Clone, Copy)]
pub struct ResidencyGovernor<const MAX_SLOTS: usize> {
    caps: ResidencyPoolCaps,
    current_cpu_bytes: u64,
    current_gpu_vram_bytes: u64,
    slots: [ResidencySlot; MAX_SLOTS],
}

impl<const MAX_SLOTS: usize> ResidencyGovernor<MAX_SLOTS> {
    /// Creates a new governor with specified capacity caps and an empty slot array.
    pub const fn new(caps: ResidencyPoolCaps) -> Self {
        Self {
            caps,
            current_cpu_bytes: 0,
            current_gpu_vram_bytes: 0,
            slots: [ResidencySlot {
                resource_hash: 0,
                cpu_bytes: 0,
                gpu_vram_bytes: 0,
                last_used_tick: 0,
                pinned: false,
                active: false,
            }; MAX_SLOTS],
        }
    }

    #[inline]
    pub const fn caps(&self) -> ResidencyPoolCaps {
        self.caps
    }

    #[inline]
    pub const fn current_cpu_bytes(&self) -> u64 {
        self.current_cpu_bytes
    }

    #[inline]
    pub const fn current_gpu_vram_bytes(&self) -> u64 {
        self.current_gpu_vram_bytes
    }

    #[inline]
    pub fn occupied_slots(&self) -> usize {
        let mut count = 0;
        for s in &self.slots {
            if s.active {
                count += 1;
            }
        }
        count
    }

    #[inline]
    pub fn pinned_slots(&self) -> usize {
        let mut count = 0;
        for s in &self.slots {
            if s.active && s.pinned {
                count += 1;
            }
        }
        count
    }

    #[inline]
    fn find_slot(&self, resource_hash: u64) -> Option<usize> {
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.active && slot.resource_hash == resource_hash {
                return Some(i);
            }
        }
        None
    }

    #[inline]
    pub fn contains(&self, resource_hash: u64) -> bool {
        self.find_slot(resource_hash).is_some()
    }

    #[inline]
    pub fn get_slot(&self, resource_hash: u64) -> Option<&ResidencySlot> {
        self.find_slot(resource_hash).map(|i| &self.slots[i])
    }

    /// Touches a resident resource to refresh its LRU recency tick.
    pub fn touch(&mut self, resource_hash: u64, current_tick: u64) -> bool {
        if let Some(idx) = self.find_slot(resource_hash) {
            self.slots[idx].last_used_tick = current_tick;
            true
        } else {
            false
        }
    }

    /// Tier-1 zero-heap admission with deterministic LRU eviction.
    ///
    /// Fails closed with `ResidencyBudgetExceeded` if requested bytes exceed pool caps
    /// or if existing pinned allocations leave insufficient headroom.
    pub fn admit_or_evict_lru_zero_heap(
        &mut self,
        resource_hash: u64,
        cpu_bytes: u64,
        gpu_vram_bytes: u64,
        pinned: bool,
        current_tick: u64,
    ) -> Result<usize, ResidencyError> {
        if cpu_bytes > self.caps.max_cpu_bytes || gpu_vram_bytes > self.caps.max_gpu_vram_bytes {
            return Err(self.budget_error(
                cpu_bytes,
                gpu_vram_bytes,
                self.current_cpu_bytes,
                self.current_gpu_vram_bytes,
            ));
        }
        let previous = self.find_slot(resource_hash);
        if let Some(index) = previous {
            let slot = &mut self.slots[index];
            if slot.cpu_bytes == cpu_bytes && slot.gpu_vram_bytes == gpu_vram_bytes {
                slot.last_used_tick = current_tick;
                slot.pinned |= pinned;
                return Ok(index);
            }
        }

        // Exclude the replacement target from feasibility, but keep all state intact
        // until every refusal condition is resolved.
        let mut pinned_cpu = 0u64;
        let mut pinned_gpu = 0u64;
        let mut pinned_count = 0usize;
        let mut active_count = 0usize;
        for (index, slot) in self.slots.iter().enumerate() {
            if !slot.active || previous == Some(index) {
                continue;
            }
            active_count += 1;
            if slot.pinned {
                pinned_cpu = pinned_cpu.checked_add(slot.cpu_bytes).ok_or_else(|| {
                    self.budget_error(cpu_bytes, gpu_vram_bytes, u64::MAX, u64::MAX)
                })?;
                pinned_gpu = pinned_gpu.checked_add(slot.gpu_vram_bytes).ok_or_else(|| {
                    self.budget_error(cpu_bytes, gpu_vram_bytes, u64::MAX, u64::MAX)
                })?;
                pinned_count += 1;
            }
        }
        if cpu_bytes > self.caps.max_cpu_bytes.saturating_sub(pinned_cpu)
            || gpu_vram_bytes > self.caps.max_gpu_vram_bytes.saturating_sub(pinned_gpu)
        {
            return Err(self.budget_error(cpu_bytes, gpu_vram_bytes, pinned_cpu, pinned_gpu));
        }
        if pinned_count >= MAX_SLOTS {
            return Err(ResidencyError::SlotCapacityFull);
        }

        if let Some(index) = previous {
            self.remove_slot(index);
        }
        while self
            .current_cpu_bytes
            .checked_add(cpu_bytes)
            .is_none_or(|total| total > self.caps.max_cpu_bytes)
            || self
                .current_gpu_vram_bytes
                .checked_add(gpu_vram_bytes)
                .is_none_or(|total| total > self.caps.max_gpu_vram_bytes)
            || active_count >= MAX_SLOTS
        {
            let mut oldest: Option<usize> = None;
            for (index, slot) in self.slots.iter().enumerate() {
                if slot.active
                    && !slot.pinned
                    && oldest.is_none_or(|old| slot.last_used_tick < self.slots[old].last_used_tick)
                {
                    oldest = Some(index);
                }
            }
            // Feasibility above proves that an unpinned victim exists until it fits.
            let index = oldest.expect("preflight guarantees an eviction candidate");
            self.remove_slot(index);
            active_count -= 1;
        }
        let index = self
            .slots
            .iter()
            .position(|slot| !slot.active)
            .expect("preflight guarantees a free slot");
        self.slots[index] = ResidencySlot {
            resource_hash,
            cpu_bytes,
            gpu_vram_bytes,
            last_used_tick: current_tick,
            pinned,
            active: true,
        };
        self.current_cpu_bytes = self
            .current_cpu_bytes
            .checked_add(cpu_bytes)
            .expect("admission total was checked");
        self.current_gpu_vram_bytes = self
            .current_gpu_vram_bytes
            .checked_add(gpu_vram_bytes)
            .expect("admission total was checked");
        Ok(index)
    }

    fn budget_error(
        &self,
        requested_cpu: u64,
        requested_gpu: u64,
        occupied_cpu: u64,
        occupied_gpu: u64,
    ) -> ResidencyError {
        ResidencyError::ResidencyBudgetExceeded {
            requested_cpu,
            requested_gpu,
            available_cpu: self.caps.max_cpu_bytes.saturating_sub(occupied_cpu),
            available_gpu: self.caps.max_gpu_vram_bytes.saturating_sub(occupied_gpu),
        }
    }

    fn remove_slot(&mut self, index: usize) {
        let slot = &mut self.slots[index];
        self.current_cpu_bytes -= slot.cpu_bytes;
        self.current_gpu_vram_bytes -= slot.gpu_vram_bytes;
        slot.active = false;
    }

    /// Releases a resident resource and returns its byte allocations to the pool.
    pub fn release_resource(&mut self, resource_hash: u64) -> Result<(), ResidencyError> {
        for slot in self.slots.iter_mut() {
            if slot.active && slot.resource_hash == resource_hash {
                self.current_cpu_bytes = self.current_cpu_bytes.saturating_sub(slot.cpu_bytes);
                self.current_gpu_vram_bytes = self
                    .current_gpu_vram_bytes
                    .saturating_sub(slot.gpu_vram_bytes);
                slot.active = false;
                slot.pinned = false;
                return Ok(());
            }
        }
        Err(ResidencyError::ResourceNotFound(resource_hash))
    }

    /// Sets or clears the pinned status for a resident resource.
    pub fn pin_resource(&mut self, resource_hash: u64, pinned: bool) -> Result<(), ResidencyError> {
        for slot in self.slots.iter_mut() {
            if slot.active && slot.resource_hash == resource_hash {
                slot.pinned = pinned;
                return Ok(());
            }
        }
        Err(ResidencyError::ResourceNotFound(resource_hash))
    }

    /// Cancels a pending or in-flight hydration, releasing its slot immediately.
    #[inline]
    pub fn cancel_pending_hydration(&mut self, resource_hash: u64) -> Result<(), ResidencyError> {
        self.release_resource(resource_hash)
    }

    /// Evicts all unpinned resources (e.g. on stream abort or camera scene jump).
    pub fn cancel_all_unpinned(&mut self) -> usize {
        let mut evicted = 0;
        for slot in self.slots.iter_mut() {
            if slot.active && !slot.pinned {
                self.current_cpu_bytes = self.current_cpu_bytes.saturating_sub(slot.cpu_bytes);
                self.current_gpu_vram_bytes = self
                    .current_gpu_vram_bytes
                    .saturating_sub(slot.gpu_vram_bytes);
                slot.active = false;
                evicted += 1;
            }
        }
        evicted
    }
}
