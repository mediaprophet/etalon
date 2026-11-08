//! Deterministic 64-Byte UDP Multicast Frame-Lock & Conical Frustum Slicing Protocol.
//!
//! Powers industrial multi-projector full-domes (e.g. 600x 4K displays) with microsecond
//! synchronization and zero network geometry transmission.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Compile-time 256-entry CRC-32C (Castagnoli, reflected polynomial `0x82F6_3B78`) lookup table.
const CRC32C_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = (crc >> 1) ^ (0x82F6_3B78 & 0u32.wrapping_sub(crc & 1));
            bit += 1;
        }
        t[i] = crc;
        i += 1;
    }
    t
};

/// Zero-heap CRC-32C (Castagnoli, reflected) over `bytes`, bit-identical to
/// `qualia-core-db::container_10d::crc32c::crc32c`.
#[inline]
fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in bytes {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32C_TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

/// Frame-lock magic ASCII identifier: `FMLK`
pub const FRAME_LOCK_MAGIC: [u8; 4] = *b"FMLK";

/// Protocol version 1
pub const FRAME_LOCK_VERSION: u16 = 1;

/// Fixed 64-Byte Aligned Frame-Lock Packet broadcast over UDP multicast at 60/120 Hz.
#[repr(C, align(64))]
#[derive(Debug, Copy, Clone, PartialEq, Pod, Zeroable)]
pub struct FrameLockPacket {
    /// 0x00..0x04: ASCII Magic `FMLK`
    pub magic: [u8; 4],
    /// 0x04..0x06: Protocol version (1)
    pub protocol_version: u16,
    /// 0x06..0x08: Cluster Identifier
    pub cluster_id: u16,
    /// 0x08..0x10: Monotonic Lamport simulation clock tick
    pub lamport_clock: u64,
    /// 0x10..0x1C: Observer camera focal position (x, y, z) in meters relative to cluster datum (12 bytes)
    pub observer_pos: [f32; 3],
    /// 0x1C..0x20: Solar radiant irradiance & zenith factor (α)
    pub solar_irradiance_alpha: f32,
    /// 0x20..0x30: Camera orientation quaternion (qx, qy, qz, qw) (16 bytes)
    pub observer_rot: [f32; 4],
    /// 0x30..0x34: Active simulation quantum branch q (0 = ground truth)
    pub quantum_branch_q: u32,
    /// 0x34..0x38: Environmental PRNG seed (wind, rain turbulence, wildlife)
    pub environmental_seed: u32,
    /// 0x38..0x3C: Reserved for network flags
    pub network_flags: u32,
    /// 0x3C..0x40: CRC-32C checksum of bytes 0..60
    pub packet_crc32c: u32,
}

impl Default for FrameLockPacket {
    fn default() -> Self {
        Self {
            magic: FRAME_LOCK_MAGIC,
            protocol_version: FRAME_LOCK_VERSION,
            cluster_id: 1,
            lamport_clock: 0,
            observer_pos: [0.0, 0.0, 0.0],
            solar_irradiance_alpha: 1.0,
            observer_rot: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            quantum_branch_q: 0,
            environmental_seed: 42,
            network_flags: 0,
            packet_crc32c: 0,
        }
    }
}

impl FrameLockPacket {
    /// Updates the CRC-32C checksum over the 60-byte payload.
    pub fn update_crc(&mut self) {
        self.packet_crc32c = 0;
        let bytes: &[u8; 64] = bytemuck::cast_ref(self);
        self.packet_crc32c = crc32c(&bytes[..60]);
    }

    /// Validates the frame packet magic, version, and CRC-32C checksum.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.magic != FRAME_LOCK_MAGIC {
            return Err("Invalid frame-lock magic");
        }
        if self.protocol_version != FRAME_LOCK_VERSION {
            return Err("Unsupported frame-lock protocol version");
        }
        let bytes: &[u8; 64] = bytemuck::cast_ref(self);
        let expected_crc = crc32c(&bytes[..60]);
        if self.packet_crc32c != expected_crc {
            return Err("Frame-lock CRC-32C checksum mismatch");
        }
        Ok(())
    }
}

/// Conical solid-angle wedge assigned to an individual projector node in a dome cluster.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Pod, Zeroable)]
pub struct ConicalDomeWedge {
    pub node_index: u32,
    pub theta_center_rad: f32,
    pub theta_extent_rad: f32,
    pub phi_center_rad: f32,
    pub phi_extent_rad: f32,
    pub edge_blend_overlap: f32,
    pub _reserved: [u8; 8],
}

impl ConicalDomeWedge {
    /// Tests if a normalized ray direction (x, y, z) falls within this conical display wedge.
    #[inline]
    pub fn contains_ray(&self, ray_dir: [f32; 3]) -> bool {
        let x = ray_dir[0];
        let y = ray_dir[1];
        let z = ray_dir[2];

        let r_horizontal = (x * x + y * y).sqrt();
        let theta = y.atan2(x);
        let phi = z.atan2(r_horizontal);

        let d_theta = (theta - self.theta_center_rad).abs();
        let d_phi = (phi - self.phi_center_rad).abs();

        d_theta <= self.theta_extent_rad * 0.5 && d_phi <= self.phi_extent_rad * 0.5
    }
}

/// Functional role played by a device in the user's multi-screen ecosystem.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompanionDeviceRole {
    /// 100% immersive, zero-HUD natural environmental viewport (Smart TV / Main Monitor)
    PrimaryVista,
    /// Tactical GIS map, subterranean strata core-slicer, and 3D Specimen Loupe (Tablet)
    TacticalTablet,
    /// Tactile remote: vehicle throttle, winch controls, expedition journal (Mobile Phone)
    MobileRemote,
    /// Networked participant in a different physical location (Social Peer)
    SocialPeer,
}

/// Messages exchanged across local WebRTC DataChannels / WebSockets between companion devices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompanionSyncMessage {
    /// Primary vista informs tablet that user approached an ecological entity
    ProximityWakeSpecimen {
        taxon_id_hash: u64,
        entity_name: String,
        distance_meters: f32,
    },
    /// Tablet informs simulation of user interaction in specimen loupe (e.g. growth scrubber)
    SpecimenGrowthScrub {
        taxon_id_hash: u64,
        growth_scale_factor: f32,
        season_month: f32,
    },
    /// Mobile phone sends tactile vehicle steering or winch inputs
    VehicleControlInput {
        steering_delta: f32,
        throttle_normalized: f32,
        winch_speed: f32,
    },
    /// User excavated or modified terrain (committed as Merkle-CRDT mutation)
    TerrainMutation {
        chunk_index: u64,
        local_pos: [f32; 3],
        volume_delta_m3: f32,
        material_type: u8,
        lamport_clock: u64,
    },
    /// Telemetry status update from host governor to companion dashboard
    GovernorTelemetryUpdate {
        vram_pressure_pct: u8,
        llm_decode_toks: f32,
        gpu_temperature_c: u8,
    },
}

/// Conical solid-angle wedge assigned to an individual projector or display node.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrustumWedge {
    /// Unique index of this display node in the cluster [0..N-1]
    pub node_index: u32,
    /// Center azimuth angle θ in radians [-π..π]
    pub theta_center_rad: f32,
    /// Azimuth angular spread Δθ in radians
    pub theta_extent_rad: f32,
    /// Center elevation angle φ in radians [0..π/2] (0 = horizon, π/2 = zenith)
    pub phi_center_rad: f32,
    /// Elevation angular spread Δφ in radians
    pub phi_extent_rad: f32,
    /// Projector optical edge-blend margin ratio (e.g. 0.08 = 8% overlap)
    pub edge_blend_overlap: f32,
}

impl FrustumWedge {
    /// Generates a full-dome cluster decomposition for N display nodes.
    /// Distributes nodes across elevation rings according to solid angle area.
    pub fn generate_dome_cluster(node_count: u32) -> Vec<Self> {
        let mut wedges = Vec::with_capacity(node_count as usize);
        if node_count == 0 {
            return wedges;
        }

        // Estimate number of elevation rings: ~ sqrt(N / 2)
        let ring_count = ((node_count as f32 / 2.0).sqrt().round() as u32).max(1);
        let phi_step = (std::f32::consts::FRAC_PI_2) / ring_count as f32;

        let mut current_node = 0u32;
        for ring in 0..ring_count {
            let phi_center = (ring as f32 + 0.5) * phi_step;
            // Circumference proportional to cos(phi)
            let ring_weight = phi_center.cos().max(0.1);
            let ring_nodes = ((node_count as f32 / ring_count as f32) * ring_weight * 1.5)
                .round() as u32;
            let ring_nodes = ring_nodes.max(1);

            let theta_step = (std::f32::consts::TAU) / ring_nodes as f32;

            for n in 0..ring_nodes {
                if current_node >= node_count {
                    break;
                }
                let theta_center = -std::f32::consts::PI + (n as f32 + 0.5) * theta_step;
                wedges.push(FrustumWedge {
                    node_index: current_node,
                    theta_center_rad: theta_center,
                    theta_extent_rad: theta_step * 1.08, // 8% blend overlap
                    phi_center_rad: phi_center,
                    phi_extent_rad: phi_step * 1.08,
                    edge_blend_overlap: 0.08,
                });
                current_node += 1;
            }
        }

        // Fill remaining nodes if rounding didn't reach target
        while current_node < node_count {
            wedges.push(FrustumWedge {
                node_index: current_node,
                theta_center_rad: 0.0,
                theta_extent_rad: 0.2,
                phi_center_rad: std::f32::consts::FRAC_PI_2, // Zenith
                phi_extent_rad: 0.2,
                edge_blend_overlap: 0.08,
            });
            current_node += 1;
        }

        wedges
    }

    /// Evaluates whether a given 3D unit ray from the dome center falls within this wedge.
    #[inline]
    pub fn contains_ray(&self, ray: [f32; 3]) -> bool {
        // ray: [x, y, z] where z is up
        let r_horizontal = (ray[0] * ray[0] + ray[1] * ray[1]).sqrt();
        let phi = ray[2].atan2(r_horizontal);
        let theta = ray[1].atan2(ray[0]);

        let half_phi = self.phi_extent_rad * 0.5;
        let half_theta = self.theta_extent_rad * 0.5;

        (phi - self.phi_center_rad).abs() <= half_phi
            && (theta - self.theta_center_rad).abs() <= half_theta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32c_check_value_123456789_is_e3069283() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(&[]), 0x0000_0000);
    }

    #[test]
    fn test_frame_lock_packet_size_and_alignment() {
        assert_eq!(std::mem::size_of::<FrameLockPacket>(), 64);
        assert_eq!(std::mem::align_of::<FrameLockPacket>(), 64);
    }

    #[test]
    fn test_frame_lock_crc() {
        let mut pkt = FrameLockPacket::default();
        pkt.lamport_clock = 1024;
        pkt.observer_pos = [120.5, 450.2, 540.0];
        pkt.update_crc();

        assert!(pkt.validate().is_ok());

        // Alter position -> checksum fail
        pkt.observer_pos[0] += 0.1;
        assert!(pkt.validate().is_err());
    }

    #[test]
    fn test_conical_dome_wedge_ray_test() {
        let wedge = ConicalDomeWedge {
            node_index: 0,
            theta_center_rad: 0.0,
            theta_extent_rad: 0.5,
            phi_center_rad: 0.0,
            phi_extent_rad: 0.5,
            edge_blend_overlap: 0.08,
            _reserved: [0; 8],
        };

        // Ray directly forward (1, 0, 0)
        assert!(wedge.contains_ray([1.0, 0.0, 0.0]));
        // Ray pointing opposite (-1, 0, 0)
        assert!(!wedge.contains_ray([-1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_600_screen_dome_cluster_generation() {
        let cluster = FrustumWedge::generate_dome_cluster(600);
        assert_eq!(cluster.len(), 600);
        assert_eq!(cluster[0].node_index, 0);
        assert_eq!(cluster[599].node_index, 599);
    }
}
