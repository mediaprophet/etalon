//! Zero-Heap Gerstner Wave Spectrum & Nearshore Attenuation Engine.
//!
//! Provides an 8-harmonic analytical Gerstner wave spectrum evaluated on CPU
//! without allocations, with smooth shoreline boundary damping.

/// Maximum number of analytical wave components evaluated simultaneously.
pub const MAX_GERSTNER_HARMONICS: usize = 8;
/// Standard water density in kg/m³ (freshwater = 1000.0, ocean = 1025.0).
pub const WATER_DENSITY_KG_M3: f32 = 1025.0;
/// Gravity constant in m/s².
pub const GRAVITY_M_S2: f32 = 9.80665;

/// Smoothstep hermite interpolation $3t^2 - 2t^3$.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Individual Gerstner wave harmonic component.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GerstnerWave {
    /// Normalized horizontal propagation direction `[dx, dz]`.
    pub direction: [f32; 2],
    /// Wave amplitude in meters (crest height above calm surface).
    pub amplitude: f32,
    /// Wavelength in meters.
    pub wavelength: f32,
    /// Wave steepness $Q \in [0.0, 1.0]$ where $Q=0$ is sinusoidal and $Q=1$ is sharp trochoidal.
    pub steepness: f32,
    /// Phase offset in radians.
    pub phase: f32,
}

impl Default for GerstnerWave {
    fn default() -> Self {
        Self {
            direction: [1.0, 0.0],
            amplitude: 0.0,
            wavelength: 10.0,
            steepness: 0.0,
            phase: 0.0,
        }
    }
}

/// 8-harmonic Gerstner wave spectrum for zero-allocation CPU water queries.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimdGerstnerSpectrum {
    pub waves: [GerstnerWave; MAX_GERSTNER_HARMONICS],
    pub active_waves: usize,
    /// Global calm water surface elevation along the Y-axis.
    pub sea_level: f32,
}

impl Default for SimdGerstnerSpectrum {
    fn default() -> Self {
        Self {
            waves: [GerstnerWave::default(); MAX_GERSTNER_HARMONICS],
            active_waves: 0,
            sea_level: 0.0,
        }
    }
}

impl SimdGerstnerSpectrum {
    /// Constructs a standard calm coastal water profile.
    pub const fn coastal_preset(sea_level: f32) -> Self {
        Self {
            waves: [
                // Primary ocean swell: 35m wavelength, 0.6m amplitude
                GerstnerWave {
                    direction: [0.92, 0.38],
                    amplitude: 0.6,
                    wavelength: 35.0,
                    steepness: 0.45,
                    phase: 0.0,
                },
                // Secondary cross-swell: 18m wavelength, 0.35m amplitude
                GerstnerWave {
                    direction: [0.707, -0.707],
                    amplitude: 0.35,
                    wavelength: 18.0,
                    steepness: 0.35,
                    phase: 1.2,
                },
                // Wind chop: 7m wavelength, 0.15m amplitude
                GerstnerWave {
                    direction: [0.38, 0.92],
                    amplitude: 0.15,
                    wavelength: 7.0,
                    steepness: 0.25,
                    phase: 2.4,
                },
                // High-frequency capillary ripples: 2.5m wavelength, 0.05m amplitude
                GerstnerWave {
                    direction: [0.0, 1.0],
                    amplitude: 0.05,
                    wavelength: 2.5,
                    steepness: 0.15,
                    phase: 0.5,
                },
                GerstnerWave {
                    direction: [1.0, 0.0],
                    amplitude: 0.0,
                    wavelength: 10.0,
                    steepness: 0.0,
                    phase: 0.0,
                },
                GerstnerWave {
                    direction: [1.0, 0.0],
                    amplitude: 0.0,
                    wavelength: 10.0,
                    steepness: 0.0,
                    phase: 0.0,
                },
                GerstnerWave {
                    direction: [1.0, 0.0],
                    amplitude: 0.0,
                    wavelength: 10.0,
                    steepness: 0.0,
                    phase: 0.0,
                },
                GerstnerWave {
                    direction: [1.0, 0.0],
                    amplitude: 0.0,
                    wavelength: 10.0,
                    steepness: 0.0,
                    phase: 0.0,
                },
            ],
            active_waves: 4,
            sea_level,
        }
    }

    /// Evaluates the water surface elevation and normal at world coordinate `(x, z)` at time `time_sec`.
    /// Zero heap allocations, pure stack evaluation.
    pub fn sample_surface(&self, x: f32, z: f32, time_sec: f32) -> (f32, [f32; 3]) {
        let mut height = self.sea_level;
        let mut normal_x = 0.0f32;
        let mut normal_z = 0.0f32;

        let count = if self.active_waves < MAX_GERSTNER_HARMONICS {
            self.active_waves
        } else {
            MAX_GERSTNER_HARMONICS
        };

        let mut i = 0;
        while i < count {
            let w = &self.waves[i];
            if w.amplitude <= 1e-5 || w.wavelength <= 1e-4 {
                i += 1;
                continue;
            }

            // Wave number k = 2pi / L
            let k = core::f32::consts::PI * 2.0 / w.wavelength;
            // Dispersion relation: omega = sqrt(g * k)
            let omega = (GRAVITY_M_S2 * k).sqrt();
            let dot = w.direction[0] * x + w.direction[1] * z;
            let theta = k * dot - omega * time_sec + w.phase;

            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            // Vertical displacement: y = A * sin(theta)
            height += w.amplitude * sin_theta;

            // Surface derivatives for normal vector calculation
            let factor = k * w.amplitude * cos_theta;
            normal_x -= w.direction[0] * factor;
            normal_z -= w.direction[1] * factor;

            i += 1;
        }

        // Un-normalized normal = [-normal_x, 1.0, -normal_z]
        let len_sq = normal_x * normal_x + 1.0 + normal_z * normal_z;
        let inv_len = 1.0 / len_sq.sqrt();
        let normal = [normal_x * inv_len, 1.0 * inv_len, normal_z * inv_len];

        (height, normal)
    }

    /// Evaluates water surface with smooth shoreline attenuation.
    ///
    /// `signed_shore_distance_m`: Positive inside open water, negative inland.
    /// `attenuation_band_m`: Distance over which waves fade from full amplitude to 0 at the shore.
    pub fn sample_nearshore_surface(
        &self,
        x: f32,
        z: f32,
        t: f32,
        signed_shore_distance_m: f32,
        attenuation_band_m: f32,
    ) -> (f32, [f32; 3]) {
        let factor = smoothstep(0.0, attenuation_band_m, signed_shore_distance_m.max(0.0));
        if factor <= 0.0 {
            return (self.sea_level, [0.0, 1.0, 0.0]);
        }

        let (raw_height, raw_normal) = self.sample_surface(x, z, t);
        let height = self.sea_level + (raw_height - self.sea_level) * factor;

        let nx = raw_normal[0] * factor;
        let ny = raw_normal[1];
        let nz = raw_normal[2] * factor;
        let len_sq = nx * nx + ny * ny + nz * nz;
        let inv_len = if len_sq > 1e-8 {
            1.0 / len_sq.sqrt()
        } else {
            1.0
        };

        (height, [nx * inv_len, ny * inv_len, nz * inv_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calm_water_elevation_and_normal() {
        let spectrum = SimdGerstnerSpectrum::default();
        let (height, normal) = spectrum.sample_surface(12.34, 56.78, 10.0);
        assert!((height - 0.0).abs() < 1e-6);
        assert!((normal[0] - 0.0).abs() < 1e-6);
        assert!((normal[1] - 1.0).abs() < 1e-6);
        assert!((normal[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gerstner_wave_elevation_bounds() {
        let spectrum = SimdGerstnerSpectrum::coastal_preset(0.0);
        let (height, normal) = spectrum.sample_surface(0.0, 0.0, 0.0);
        assert!(height >= -1.2 && height <= 1.2);
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_nearshore_attenuation() {
        let spectrum = SimdGerstnerSpectrum::coastal_preset(10.0);
        // Onshore: completely flattened
        let (h_onshore, n_onshore) = spectrum.sample_nearshore_surface(5.0, 5.0, 1.0, -2.0, 10.0);
        assert_eq!(h_onshore, 10.0);
        assert_eq!(n_onshore, [0.0, 1.0, 0.0]);

        // Far offshore: full amplitude matches raw sample
        let (raw_h, raw_n) = spectrum.sample_surface(5.0, 5.0, 1.0);
        let (h_offshore, n_offshore) = spectrum.sample_nearshore_surface(5.0, 5.0, 1.0, 15.0, 10.0);
        assert!((h_offshore - raw_h).abs() < 1e-5);
        assert!((n_offshore[0] - raw_n[0]).abs() < 1e-5);
        assert!((n_offshore[1] - raw_n[1]).abs() < 1e-5);
        assert!((n_offshore[2] - raw_n[2]).abs() < 1e-5);
    }
}
