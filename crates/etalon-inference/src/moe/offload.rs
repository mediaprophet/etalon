//! Heterogeneous MoE expert offload manager and streaming cache (Work Package F10).
//!
//! Manages compressed host-resident expert weights and bounded GPU VRAM staging slots,
//! with bandwidth-adaptive CPU/GPU hybrid partitioning for the RTX A2000 12GB budget.

use super::placement::ExpertPlacementPolicy;

/// Maximum number of active expert slots in GPU VRAM simultaneously.
pub const DEFAULT_GPU_EXPERT_SLOTS: usize = 32;

/// Device capability tier for personal computing local AI across Windows, Linux, and macOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersonalHardwareTier {
    /// Workstation / Unified Studio (>= 32 GiB): All 256 experts resident in VRAM / unified memory (zero PCIe paging).
    WorkstationResident,
    /// Personal Discrete GPU (8–24 GiB VRAM, e.g. RTX A2000, 3060/4060, Radeon on Vulkan/CUDA): Dynamic LRU staging ring.
    PersonalGpuStaging,
    /// Apple Silicon Unified Memory (macOS Metal 16–36 GiB): Zero-copy shared pool with direct compute shader dispatch.
    AppleUnifiedMemory,
    /// Constrained Edge (< 8 GiB VRAM or CPU-only): On-demand hybrid CPU/GPU streaming.
    ConstrainedEdge,
}

/// Cold-path memory observation bound to a model activation.
///
/// `available_accelerator_bytes` is the measured free budget after other processes and the
/// runtime safety reserve; it is deliberately not a nominal adapter-VRAM claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertResidencyProfile {
    pub available_accelerator_bytes: u64,
    pub backbone_bytes: u64,
    pub kv_cache_bytes: u64,
    pub bytes_per_expert: usize,
    pub total_experts: usize,
    pub is_unified_memory: bool,
}

impl ExpertResidencyProfile {
    /// Placement and hardware tier.
    pub fn placement(self) -> (usize, PersonalHardwareTier) {
        compute_principled_slot_capacity(
            self.available_accelerator_bytes,
            self.backbone_bytes,
            self.kv_cache_bytes,
            self.bytes_per_expert,
            self.total_experts,
            self.is_unified_memory,
        )
    }
}

/// Compute principled slot capacity given detected system memory and model footprint.
///
/// This principle-based calculation works across Vulkan/wgpu, CUDA, DirectML, and Apple Metal:
/// - Determines the available accelerator headroom after subtracting backbone and KV cache budgets.
/// - Slices headroom into slots of `bytes_per_expert`.
/// - Selects the appropriate `PersonalHardwareTier` without hardcoding specific GPU models.
pub fn compute_principled_slot_capacity(
    available_accelerator_bytes: u64,
    backbone_bytes: u64,
    kv_cache_bytes: u64,
    bytes_per_expert: usize,
    total_experts: usize,
    is_unified_memory: bool,
) -> (usize, PersonalHardwareTier) {
    if is_unified_memory {
        let total_required = backbone_bytes
            .saturating_add((total_experts * bytes_per_expert) as u64)
            .saturating_add(kv_cache_bytes);
        if available_accelerator_bytes >= total_required {
            return (total_experts, PersonalHardwareTier::AppleUnifiedMemory);
        }
    }

    let reserved = backbone_bytes.saturating_add(kv_cache_bytes);
    let expert_headroom = available_accelerator_bytes.saturating_sub(reserved);
    let computed_slots = if bytes_per_expert > 0 {
        (expert_headroom / bytes_per_expert as u64) as usize
    } else {
        0
    };

    if computed_slots >= total_experts && total_experts > 0 {
        (total_experts, PersonalHardwareTier::WorkstationResident)
    } else if computed_slots >= 8 {
        (
            computed_slots.min(total_experts),
            PersonalHardwareTier::PersonalGpuStaging,
        )
    } else {
        (
            computed_slots.max(4).min(total_experts),
            PersonalHardwareTier::ConstrainedEdge,
        )
    }
}

/// Telemetry and allocation status for dynamic VRAM tiering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicVramStatus {
    pub available_bytes: u64,
    pub hardware_tier: PersonalHardwareTier,
    pub total_slots: usize,
    pub pinned_slots: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

/// Query the current dynamic hardware tier given accelerator budget and workload requirements.
pub fn query_dynamic_hardware_tier(
    available_bytes: u64,
    backbone_bytes: u64,
    kv_cache_bytes: u64,
    bytes_per_expert: usize,
    total_experts: usize,
    is_unified: bool,
) -> (usize, PersonalHardwareTier) {
    compute_principled_slot_capacity(
        available_bytes,
        backbone_bytes,
        kv_cache_bytes,
        bytes_per_expert,
        total_experts,
        is_unified,
    )
}

/// Outcome of accessing an expert in the offload cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotAccessOutcome {
    /// Expert already resident in GPU slot `slot_id`.
    Hit { slot_id: usize },
    /// Expert missing from GPU; should evict `evict_slot_id` (resident: `evicted_expert_id`) to fetch `expert_id`.
    MissFetch {
        evict_slot_id: usize,
        evicted_expert_id: Option<u16>,
    },
    /// Expert missing and transfer latency exceeds compute; assigned to host CPU execution.
    MissComputeCpu,
}

/// Telemetry counters for MoE offload and residency.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertCacheTelemetry {
    pub hits: u64,
    pub misses: u64,
    pub gpu_fetches: u64,
    pub cpu_evaluations: u64,
    pub bytes_transferred: u64,
}

/// Metadata for a single GPU expert staging slot.
#[derive(Debug, Clone, Copy, Default)]
struct SlotEntry {
    expert_id: Option<u16>,
    last_access_tick: u64,
    is_pinned: bool,
}

/// Manager for heterogeneous GPU expert staging and host-RAM offload.
#[derive(Debug)]
pub struct MoeOffloadManager {
    slots: Vec<SlotEntry>,
    current_tick: u64,
    bytes_per_expert: usize,
    prefer_cpu_hybrid_threshold: f32,
    placement_policy: ExpertPlacementPolicy,
    hardware_tier: PersonalHardwareTier,
    pub telemetry: ExpertCacheTelemetry,
}

impl MoeOffloadManager {
    /// Create a new offload manager with `num_gpu_slots` staging capacity.
    pub fn new(num_gpu_slots: usize, bytes_per_expert: usize) -> Self {
        Self::with_tier(
            num_gpu_slots,
            bytes_per_expert,
            PersonalHardwareTier::PersonalGpuStaging,
        )
    }

    /// Create a new offload manager with explicit hardware tier and staging capacity.
    pub fn with_tier(
        num_gpu_slots: usize,
        bytes_per_expert: usize,
        hardware_tier: PersonalHardwareTier,
    ) -> Self {
        let capacity = if num_gpu_slots > 0 {
            num_gpu_slots
        } else {
            DEFAULT_GPU_EXPERT_SLOTS
        };
        Self {
            slots: vec![SlotEntry::default(); capacity],
            current_tick: 0,
            bytes_per_expert,
            prefer_cpu_hybrid_threshold: 0.5,
            placement_policy: ExpertPlacementPolicy::default(),
            hardware_tier,
            telemetry: ExpertCacheTelemetry::default(),
        }
    }

    /// Construct the staging cache directly from a measured activation memory profile.
    ///
    /// This is intentionally cold-path only: adapter memory probing and KV planning must finish
    /// before any decode request is admitted.  The resulting fixed slot vector never grows in
    /// the hot path.
    pub fn from_residency_profile(profile: ExpertResidencyProfile) -> (Self, PersonalHardwareTier) {
        let (slots, tier) = profile.placement();
        (Self::with_tier(slots, profile.bytes_per_expert, tier), tier)
    }

    /// Active hardware tier of this offload manager.
    #[inline]
    pub fn hardware_tier(&self) -> PersonalHardwareTier {
        self.hardware_tier
    }

    /// Query current dynamic VRAM status and cache efficiency.
    pub fn query_vram_status(&self, available_accelerator_bytes: u64) -> DynamicVramStatus {
        let pinned = self.slots.iter().filter(|s| s.is_pinned).count();
        let total_requests = self.telemetry.hits + self.telemetry.misses;
        let hit_rate = if total_requests > 0 {
            self.telemetry.hits as f64 / total_requests as f64
        } else {
            0.0
        };
        DynamicVramStatus {
            available_bytes: available_accelerator_bytes,
            hardware_tier: self.hardware_tier,
            total_slots: self.slots.len(),
            pinned_slots: pinned,
            hits: self.telemetry.hits,
            misses: self.telemetry.misses,
            hit_rate,
        }
    }

    /// Access or schedule an expert for evaluation.
    ///
    /// Checks GPU residency. If resident, returns `Hit`.
    /// If not resident and `allow_cpu_hybrid` is enabled and transfer queue is contested, returns `MissComputeCpu`.
    /// Otherwise returns `MissFetch` identifying the least-recently-used slot for replacement.
    pub fn resolve_expert(
        &mut self,
        expert_id: u16,
        allow_cpu_hybrid: bool,
        estimated_pcie_queue_depth: usize,
    ) -> SlotAccessOutcome {
        self.current_tick = self.current_tick.wrapping_add(1);

        // 1. Check for residency hit
        for (slot_idx, slot) in self.slots.iter_mut().enumerate() {
            if slot.expert_id == Some(expert_id) {
                slot.last_access_tick = self.current_tick;
                self.telemetry.hits += 1;
                return SlotAccessOutcome::Hit { slot_id: slot_idx };
            }
        }

        self.telemetry.misses += 1;

        // 2. Prefer the frozen measured placement policy.  Keep the legacy queue-ratio
        // fallback only until a profile has been calibrated for this host/model/shape.
        if self.placement_policy.prefer_cpu(
            self.bytes_per_expert,
            estimated_pcie_queue_depth,
            allow_cpu_hybrid,
        ) {
            self.telemetry.cpu_evaluations += 1;
            return SlotAccessOutcome::MissComputeCpu;
        }

        // 3. Conservative pre-calibration fallback.
        let queue_ratio = estimated_pcie_queue_depth as f32 / self.slots.len().max(1) as f32;
        if allow_cpu_hybrid && queue_ratio >= self.prefer_cpu_hybrid_threshold {
            self.telemetry.cpu_evaluations += 1;
            return SlotAccessOutcome::MissComputeCpu;
        }

        // 4. Find LRU slot to evict
        let mut lru_slot_idx = 0;
        let mut min_tick = u64::MAX;

        for (idx, slot) in self.slots.iter().enumerate() {
            if !slot.is_pinned && slot.last_access_tick < min_tick {
                min_tick = slot.last_access_tick;
                lru_slot_idx = idx;
                if slot.expert_id.is_none() {
                    // Empty slot preferred immediately
                    break;
                }
            }
        }

        let evicted_expert_id = self.slots[lru_slot_idx].expert_id;
        self.slots[lru_slot_idx].expert_id = Some(expert_id);
        self.slots[lru_slot_idx].last_access_tick = self.current_tick;

        self.telemetry.gpu_fetches += 1;
        self.telemetry.bytes_transferred += self.bytes_per_expert as u64;

        SlotAccessOutcome::MissFetch {
            evict_slot_id: lru_slot_idx,
            evicted_expert_id,
        }
    }

    /// Pin an expert into GPU VRAM (e.g. shared expert that must never be evicted).
    pub fn pin_expert(&mut self, slot_id: usize, expert_id: u16) {
        if slot_id < self.slots.len() {
            self.slots[slot_id].expert_id = Some(expert_id);
            self.slots[slot_id].is_pinned = true;
            self.slots[slot_id].last_access_tick = u64::MAX;
        }
    }

    /// Number of available staging slots in GPU memory.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// CPU hybrid dispatch threshold (queue depth / slot capacity).
    pub fn prefer_cpu_hybrid_threshold(&self) -> f32 {
        self.prefer_cpu_hybrid_threshold
    }

    /// Set the CPU hybrid dispatch threshold.
    pub fn set_prefer_cpu_hybrid_threshold(&mut self, threshold: f32) {
        self.prefer_cpu_hybrid_threshold = threshold;
    }

    /// Freeze the cold-path calibration for all subsequent cache-miss decisions in this run.
    pub fn set_placement_policy(&mut self, policy: ExpertPlacementPolicy) {
        self.placement_policy = policy;
    }

    /// `MoeOffloadManager::placement_policy`.
    pub fn placement_policy(&self) -> ExpertPlacementPolicy {
        self.placement_policy
    }
}
