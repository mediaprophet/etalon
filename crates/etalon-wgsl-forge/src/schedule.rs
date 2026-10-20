//! Dispatch schedule geometry and workgroup tile configuration.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub workgroup: WorkgroupSize,
    pub tile: TileConfig,
    pub vector_width: u32,
}

impl Schedule {
    pub const fn new(workgroup: WorkgroupSize, tile: TileConfig, vector_width: u32) -> Self {
        Self {
            workgroup,
            tile,
            vector_width,
        }
    }
}
