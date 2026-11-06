//! Spacetime dimensions (x, y, z, t) and hierarchical multi-scale coordinate system
//! for semantic topology, volumetric rendering, and temporal ledgers.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// 4D Spacetime coordinate `(x, y, z, t)` in single-precision floats (16 bytes, `#[repr(C)]`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct SpacetimeCoord {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub t: f32,
}

impl Default for SpacetimeCoord {
    #[inline]
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            t: 0.0,
        }
    }
}

impl SpacetimeCoord {
    /// Creates a new `SpacetimeCoord`.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, t: f32) -> Self {
        Self { x, y, z, t }
    }

    /// Euclidean spatial distance in 3D `(x, y, z)`.
    #[inline]
    pub fn spatial_distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Hierarchical multi-scale spacetime coordinate (32 bytes, `#[repr(C)]`, Pod).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct HierarchicalSpacetimeCoord {
    /// Coarse sector ID (e.g., 64-bit Hilbert / HEALPix / Astro-Grid cell)
    pub sector_id: u64,
    /// Scale tier exponent S in [0..7] -> step = 10^(3*S - 9) meters
    pub scale_tier: u8,
    pub _pad: [u8; 7],
    /// Fine local offset within sector in f32
    pub local_xyz: [f32; 3],
    /// Coordinate time t (seconds relative to sector epoch)
    pub local_t: f32,
}

const _: () = assert!(core::mem::size_of::<SpacetimeCoord>() == 16);
const _: () = assert!(core::mem::size_of::<HierarchicalSpacetimeCoord>() == 32);

impl Default for HierarchicalSpacetimeCoord {
    #[inline]
    fn default() -> Self {
        Self {
            sector_id: 0,
            scale_tier: 3, // meter default (10^0 m)
            _pad: [0; 7],
            local_xyz: [0.0; 3],
            local_t: 0.0,
        }
    }
}

impl HierarchicalSpacetimeCoord {
    /// Computes metric step in meters for scale tier S in [0..7]: 10^(3*S - 9).
    #[inline]
    pub fn scale_step_meters(scale_tier: u8) -> f64 {
        10f64.powi(3 * (scale_tier.min(7) as i32) - 9)
    }

    /// Convert local offset to meters in double-precision f64.
    #[inline]
    pub fn to_meters_f64(&self) -> [f64; 3] {
        let step = Self::scale_step_meters(self.scale_tier);
        [
            self.local_xyz[0] as f64 * step,
            self.local_xyz[1] as f64 * step,
            self.local_xyz[2] as f64 * step,
        ]
    }

    /// Construct from metric position in meters at a given scale tier.
    #[inline]
    pub fn from_meters_f64(
        sector_id: u64,
        scale_tier: u8,
        xyz_meters: [f64; 3],
        local_t: f32,
    ) -> Self {
        let step = Self::scale_step_meters(scale_tier);
        let inv_step = 1.0 / step;
        Self {
            sector_id,
            scale_tier: scale_tier.min(7),
            _pad: [0; 7],
            local_xyz: [
                (xyz_meters[0] * inv_step) as f32,
                (xyz_meters[1] * inv_step) as f32,
                (xyz_meters[2] * inv_step) as f32,
            ],
            local_t,
        }
    }

    /// Calculate camera-relative fine offset in meters.
    #[inline]
    pub fn camera_relative_offset(&self, cam: &Self, sector_span_meters: f64) -> [f32; 3] {
        let self_m = self.to_meters_f64();
        let cam_m = cam.to_meters_f64();
        if self.sector_id == cam.sector_id || sector_span_meters == 0.0 {
            [
                (self_m[0] - cam_m[0]) as f32,
                (self_m[1] - cam_m[1]) as f32,
                (self_m[2] - cam_m[2]) as f32,
            ]
        } else {
            let (sx1, sy1, sz1) = unpack_sector_xyz(self.sector_id);
            let (sx2, sy2, sz2) = unpack_sector_xyz(cam.sector_id);
            let dsx = (sx1 - sx2) as f64 * sector_span_meters;
            let dsy = (sy1 - sy2) as f64 * sector_span_meters;
            let dsz = (sz1 - sz2) as f64 * sector_span_meters;
            [
                ((self_m[0] + dsx) - cam_m[0]) as f32,
                ((self_m[1] + dsy) - cam_m[1]) as f32,
                ((self_m[2] + dsz) - cam_m[2]) as f32,
            ]
        }
    }
}

#[inline]
const fn sign_extend_21(v: u32) -> i32 {
    let v = v & 0x1F_FFFF;
    if v & 0x10_0000 != 0 {
        (v | !0x1F_FFFF) as i32
    } else {
        v as i32
    }
}

/// Pack 3 signed 21-bit sector coordinates into 64-bit sector ID.
#[inline]
pub const fn pack_sector_xyz(sx: i32, sy: i32, sz: i32) -> u64 {
    let ux = (sx as i64 & 0x1F_FFFF) as u64;
    let uy = (sy as i64 & 0x1F_FFFF) as u64;
    let uz = (sz as i64 & 0x1F_FFFF) as u64;
    (ux << 42) | (uy << 21) | uz
}

/// Unpack 64-bit sector ID into 3 signed 21-bit sector coordinates.
#[inline]
pub const fn unpack_sector_xyz(sector_id: u64) -> (i32, i32, i32) {
    let sx = sign_extend_21(((sector_id >> 42) & 0x1F_FFFF) as u32);
    let sy = sign_extend_21(((sector_id >> 21) & 0x1F_FFFF) as u32);
    let sz = sign_extend_21((sector_id & 0x1F_FFFF) as u32);
    (sx, sy, sz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_step_meters_tiers() {
        let expected_steps = [
            (0, 1e-9), // sub-nanometer / nanometer
            (1, 1e-6), // micrometer
            (2, 1e-3), // millimeter
            (3, 1.0),  // meter
            (4, 1e3),  // kilometer
            (5, 1e6),  // megameter
            (6, 1e9),  // gigameter / astronomical
            (7, 1e12), // parsec tier
        ];

        for (tier, expected) in expected_steps {
            let step = HierarchicalSpacetimeCoord::scale_step_meters(tier);
            assert!(
                (step - expected).abs() / expected < 1e-12,
                "Tier {tier} expected {expected}, got {step}"
            );
        }
    }

    #[test]
    fn test_from_and_to_meters_round_trip() {
        for s in 0..=7 {
            let step = HierarchicalSpacetimeCoord::scale_step_meters(s);
            let test_xyz = [10.5 * step, -5.25 * step, 0.125 * step];
            let coord = HierarchicalSpacetimeCoord::from_meters_f64(100, s, test_xyz, 0.0);
            assert_eq!(coord.scale_tier, s);
            let recovered = coord.to_meters_f64();
            for i in 0..3 {
                assert!(
                    (recovered[i] - test_xyz[i]).abs() / step < 1e-4,
                    "Failed at tier {s} axis {i}"
                );
            }
        }
    }

    #[test]
    fn test_camera_relative_offset() {
        let cam = HierarchicalSpacetimeCoord::from_meters_f64(0, 3, [100.0, 200.0, 300.0], 0.0);
        let obj = HierarchicalSpacetimeCoord::from_meters_f64(0, 3, [110.0, 200.0, 295.0], 0.0);
        let delta = obj.camera_relative_offset(&cam, 1000.0);
        assert!((delta[0] - 10.0).abs() < 1e-4);
        assert!(delta[1].abs() < 1e-4);
        assert!((delta[2] - (-5.0)).abs() < 1e-4);

        // Cross-sector offset
        let s_cam = pack_sector_xyz(0, 0, 0);
        let s_obj = pack_sector_xyz(1, 0, 0);
        let cam_sec = HierarchicalSpacetimeCoord::from_meters_f64(s_cam, 3, [0.0, 0.0, 0.0], 0.0);
        let obj_sec = HierarchicalSpacetimeCoord::from_meters_f64(s_obj, 3, [0.0, 0.0, 0.0], 0.0);
        let delta_sec = obj_sec.camera_relative_offset(&cam_sec, 5000.0);
        assert!((delta_sec[0] - 5000.0).abs() < 1e-3);
    }
}
