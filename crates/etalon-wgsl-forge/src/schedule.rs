//! Dispatch schedule geometry, adapter constraints, and workgroup tile configuration.

use serde::{Deserialize, Serialize};

use crate::ir::{ForgeError, IntrinsicClass, KernelSpec};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkgroupSize {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub const fn flat_threads(&self) -> u32 {
        self.x * self.y * self.z
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileConfig {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl TileConfig {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Schedule {
    pub workgroup_size: u32,
    pub items_per_invocation: u32,
    pub vector_width: u32,
    pub workgroup: WorkgroupSize,
    pub tile: TileConfig,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            workgroup_size: 64,
            items_per_invocation: 1,
            vector_width: 1,
            workgroup: WorkgroupSize::new(64, 1, 1),
            tile: TileConfig::new(16, 16, 16),
        }
    }
}

impl Schedule {
    pub const fn new(workgroup: WorkgroupSize, tile: TileConfig, vector_width: u32) -> Self {
        Self {
            workgroup_size: workgroup.x,
            items_per_invocation: 1,
            vector_width,
            workgroup,
            tile,
        }
    }

    pub const fn from_1d(workgroup_size: u32, items_per_invocation: u32, vector_width: u32) -> Self {
        Self {
            workgroup_size,
            items_per_invocation,
            vector_width,
            workgroup: WorkgroupSize::new(workgroup_size, 1, 1),
            tile: TileConfig::new(16, 16, 16),
        }
    }

    pub const fn effective_workgroup_size(&self) -> u32 {
        if self.workgroup_size > 0 {
            self.workgroup_size
        } else {
            self.workgroup.x
        }
    }

    pub fn validate(
        self,
        kernel: &KernelSpec,
        constraints: &AdapterConstraints,
    ) -> Result<(), ForgeError> {
        kernel.validate()?;
        let wg = self.effective_workgroup_size();
        if !matches!(wg, 1..=1024) || !wg.is_power_of_two() {
            return Err(ForgeError::InvalidSchedule(
                "workgroup size must be a power of two in 1..=1024".to_string(),
            ));
        }
        if wg > constraints.max_workgroup_size_x
            || wg > constraints.max_invocations_per_workgroup
        {
            return Err(ForgeError::InvalidSchedule(format!(
                "workgroup {} exceeds adapter limit {}",
                wg,
                constraints
                    .max_workgroup_size_x
                    .min(constraints.max_invocations_per_workgroup)
            )));
        }
        if !matches!(self.items_per_invocation, 1 | 2 | 4 | 8) {
            return Err(ForgeError::InvalidSchedule(
                "items per invocation must be one of 1, 2, 4, 8".to_string(),
            ));
        }
        if !matches!(self.vector_width, 1 | 2 | 4) {
            return Err(ForgeError::InvalidSchedule(
                "vector width must be one of 1, 2, 4".to_string(),
            ));
        }
        Ok(())
    }

    pub const fn elements_per_workgroup(self) -> u32 {
        self.effective_workgroup_size() * self.items_per_invocation * self.vector_width
    }

    pub fn dispatch_workgroups(self, element_count: usize) -> u32 {
        let width = self.elements_per_workgroup().max(1) as usize;
        element_count.div_ceil(width).max(1) as u32
    }

    pub const fn sort_key(self) -> (u32, u32, u32) {
        (
            self.effective_workgroup_size(),
            self.items_per_invocation,
            self.vector_width,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterConstraints {
    pub max_workgroup_size_x: u32,
    pub max_invocations_per_workgroup: u32,
    pub max_workgroups_per_dimension: u32,
    pub supports_subgroups: bool,
    pub supports_coopmat: bool,
    pub supports_rt_cores: bool,
    pub warp_size: u32,
}

impl AdapterConstraints {
    pub const fn portable() -> Self {
        Self {
            max_workgroup_size_x: 256,
            max_invocations_per_workgroup: 256,
            max_workgroups_per_dimension: 65_535,
            supports_subgroups: false,
            supports_coopmat: false,
            supports_rt_cores: false,
            warp_size: 32,
        }
    }

    pub fn supports_kernel(&self, kernel: &KernelSpec) -> Result<(), ForgeError> {
        for intrinsic in kernel.required_intrinsics() {
            match intrinsic.class() {
                IntrinsicClass::CooperativeMatrix if !self.supports_coopmat => {
                    return Err(ForgeError::InvalidSchedule(
                        "kernel requires cooperative-matrix (tensor cores) unavailable on this adapter".to_string(),
                    ));
                }
                IntrinsicClass::RayTracing if !self.supports_rt_cores => {
                    return Err(ForgeError::InvalidSchedule(
                        "kernel requires ray-query (RT cores) unavailable on this adapter"
                            .to_string(),
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleSpace {
    pub workgroup_sizes: Vec<u32>,
    pub items_per_invocation: Vec<u32>,
    pub vector_widths: Vec<u32>,
}

impl Default for ScheduleSpace {
    fn default() -> Self {
        Self {
            workgroup_sizes: vec![32, 64, 128, 256],
            items_per_invocation: vec![1, 2, 4, 8],
            vector_widths: vec![1, 2, 4],
        }
    }
}

impl ScheduleSpace {
    pub fn candidates(
        &self,
        kernel: &KernelSpec,
        constraints: &AdapterConstraints,
    ) -> Vec<Schedule> {
        let mut candidates = Vec::new();
        for &workgroup_size in &self.workgroup_sizes {
            for &items_per_invocation in &self.items_per_invocation {
                for &vector_width in &self.vector_widths {
                    let schedule = Schedule::from_1d(
                        workgroup_size,
                        items_per_invocation,
                        vector_width,
                    );
                    let warp_aligned =
                        constraints.warp_size <= 1 || workgroup_size % constraints.warp_size == 0;
                    if warp_aligned && schedule.validate(kernel, constraints).is_ok() {
                        candidates.push(schedule);
                    }
                }
            }
        }
        candidates.sort_by_key(|schedule| schedule.sort_key());
        candidates.dedup();
        candidates
    }
}
