//! Topological classes (`v`) for geometric physics rules and 10D manifold metric evaluation.

use serde::{Deserialize, Serialize};

/// Topological class variants (§2.1).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopologyClass {
    /// Euclidean (flat semantic proximity, standard distance)
    Euclidean = 0,
    /// Cyclic / Toroidal (feedback loops, circadian rhythms, periodic states)
    Cyclic = 1,
    /// Hyperbolic / Tree (hierarchies, family trees, taxonomies)
    Hyperbolic = 2,
    /// Self-contained Boundary Cliques / Community Classes
    BoundaryCliques = 3,
    /// Spherical / HEALPix (2-sphere S^2 / celestial dome & planetary shell)
    SphericalHealpix = 4,
    /// Lorentzian Spacetime (3+1D pseudo-Riemannian metric ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2)
    LorentzianSpacetime = 5,
    /// Multiply-Connected Portal (glued chart transitions across portal boundaries)
    MultiplyConnectedPortal = 6,
}

impl Default for TopologyClass {
    #[inline]
    fn default() -> Self {
        TopologyClass::Euclidean
    }
}

impl TopologyClass {
    /// Construct from continuous `v` value.
    #[inline]
    pub fn from_v_value(v: f32) -> Self {
        match v.round() as u32 {
            0 => TopologyClass::Euclidean,
            1 => TopologyClass::Cyclic,
            2 => TopologyClass::Hyperbolic,
            3 => TopologyClass::BoundaryCliques,
            4 => TopologyClass::SphericalHealpix,
            5 => TopologyClass::LorentzianSpacetime,
            6 => TopologyClass::MultiplyConnectedPortal,
            _ => TopologyClass::Euclidean,
        }
    }

    /// Convert to canonical continuous `v` value.
    #[inline]
    pub const fn to_v_value(self) -> f32 {
        self as u32 as f32
    }

    /// Calculate distance between two points based on topology class.
    #[inline]
    pub fn calculate_distance(&self, p1: (f32, f32, f32), p2: (f32, f32, f32)) -> f32 {
        match self {
            TopologyClass::Euclidean => {
                let dx = p1.0 - p2.0;
                let dy = p1.1 - p2.1;
                let dz = p1.2 - p2.2;
                (dx * dx + dy * dy + dz * dz).sqrt()
            }
            TopologyClass::Cyclic => {
                let dx = (p1.0 - p2.0).abs().min(1.0 - (p1.0 - p2.0).abs());
                let dy = (p1.1 - p2.1).abs().min(1.0 - (p1.1 - p2.1).abs());
                let dz = (p1.2 - p2.2).abs().min(1.0 - (p1.2 - p2.2).abs());
                (dx * dx + dy * dy + dz * dz).sqrt()
            }
            TopologyClass::Hyperbolic => {
                let dx = (p1.0 - p2.0).abs();
                let dy = (p1.1 - p2.1).abs();
                let dz = (p1.2 - p2.2).abs();
                (dx.exp() + dy.exp() + dz.exp()).ln()
            }
            TopologyClass::BoundaryCliques => {
                if p1 == p2 {
                    0.0
                } else {
                    1.0
                }
            }
            TopologyClass::SphericalHealpix => {
                let r1 = (p1.0 * p1.0 + p1.1 * p1.1 + p1.2 * p1.2).sqrt();
                let r2 = (p2.0 * p2.0 + p2.1 * p2.1 + p2.2 * p2.2).sqrt();
                if r1 > 1e-6 && r2 > 1e-6 {
                    let dot = ((p1.0 * p2.0 + p1.1 * p2.1 + p1.2 * p2.2) / (r1 * r2))
                        .clamp(-1.0, 1.0);
                    let delta_theta = dot.acos();
                    let r_mean = 0.5 * (r1 + r2);
                    let arc = r_mean * delta_theta;
                    let dr = r1 - r2;
                    (arc * arc + dr * dr).sqrt()
                } else {
                    let dx = p1.0 - p2.0;
                    let dy = p1.1 - p2.1;
                    let dz = p1.2 - p2.2;
                    (dx * dx + dy * dy + dz * dz).sqrt()
                }
            }
            TopologyClass::LorentzianSpacetime => {
                let dx = p1.0 - p2.0;
                let dy = p1.1 - p2.1;
                let dz = p1.2 - p2.2;
                (dx * dx + dy * dy + dz * dz).sqrt()
            }
            TopologyClass::MultiplyConnectedPortal => {
                let wrap_delta = |a: f32, b: f32| -> f32 {
                    let d = (a - b).abs();
                    d.min((1.0 - d).abs())
                };
                let dx = wrap_delta(p1.0, p2.0);
                let dy = wrap_delta(p1.1, p2.1);
                let dz = wrap_delta(p1.2, p2.2);
                (dx * dx + dy * dy + dz * dz).sqrt()
            }
        }
    }

    /// Calculate 4D pseudo-Riemannian spacetime interval `ds`:
    /// `ds^2 = -(dt)^2 + dx^2 + dy^2 + dz^2`, returning `sgn(ds^2) * sqrt(|ds^2|)`.
    #[inline]
    pub fn calculate_spacetime_interval(
        &self,
        p1: (f32, f32, f32, f32),
        p2: (f32, f32, f32, f32),
    ) -> f32 {
        let dx = p1.0 - p2.0;
        let dy = p1.1 - p2.1;
        let dz = p1.2 - p2.2;
        let dt = p1.3 - p2.3;
        let ds2 = -(dt * dt) + (dx * dx + dy * dy + dz * dz);
        let sgn = if ds2 < 0.0 { -1.0 } else { 1.0 };
        sgn * ds2.abs().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_seven_topology_classes_round_trip() {
        let variants = [
            TopologyClass::Euclidean,
            TopologyClass::Cyclic,
            TopologyClass::Hyperbolic,
            TopologyClass::BoundaryCliques,
            TopologyClass::SphericalHealpix,
            TopologyClass::LorentzianSpacetime,
            TopologyClass::MultiplyConnectedPortal,
        ];

        for (expected_idx, variant) in variants.iter().enumerate() {
            assert_eq!(*variant as u32, expected_idx as u32);
            assert_eq!(variant.to_v_value(), expected_idx as f32);
            assert_eq!(
                TopologyClass::from_v_value(expected_idx as f32),
                *variant
            );
        }
    }

    #[test]
    fn test_spherical_healpix_distance() {
        let topo = TopologyClass::SphericalHealpix;
        let p1 = (1.0, 0.0, 0.0);
        let p2 = (0.0, 1.0, 0.0);
        let d = topo.calculate_distance(p1, p2);
        let expected = core::f32::consts::FRAC_PI_2;
        assert!((d - expected).abs() < 1e-4);

        let p_origin = (0.0, 0.0, 0.0);
        let p_near = (0.0, 0.0, 0.5);
        let d_origin = topo.calculate_distance(p_origin, p_near);
        assert!((d_origin - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_lorentzian_spacetime_interval() {
        let topo = TopologyClass::LorentzianSpacetime;
        let p1 = (0.0, 0.0, 0.0, 0.0);
        let p2 = (3.0, 0.0, 0.0, 0.0);
        assert_eq!(topo.calculate_spacetime_interval(p1, p2), 3.0);

        let p3 = (0.0, 0.0, 0.0, 4.0);
        assert_eq!(topo.calculate_spacetime_interval(p1, p3), -4.0);

        let p4 = (3.0, 4.0, 0.0, 5.0);
        assert_eq!(topo.calculate_spacetime_interval(p1, p4), 0.0);
    }

    #[test]
    fn test_multiply_connected_portal_wrap() {
        let topo = TopologyClass::MultiplyConnectedPortal;
        let p1 = (0.05, 0.1, 0.2);
        let p2 = (0.95, 0.1, 0.2);
        let d = topo.calculate_distance(p1, p2);
        assert!((d - 0.1).abs() < 1e-5);
    }
}
