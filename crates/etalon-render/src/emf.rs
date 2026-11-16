//! Electromagnetic field and signal wave-propagation visualization pipeline.
//!
//! Provides the complete GPU rendering pipeline for drawing 2D slices of 4D EMF field grids
//! (x * y * z * t) with 10D manifold records and 3 Stokes polarization parameters preserved,
//! along with CPU reference evaluators matching `shaders/emf_volumetric.wgsl`.

use bytemuck::{Pod, Zeroable};

/// Raw WGSL shader source for the EMF volumetric visualizer.
pub const EMF_VOLUMETRIC_WGSL: &str = include_str!("shaders/emf_volumetric.wgsl");

const TWO_PI_F32: f32 = 6.283_185_307;

/// Per-cell field data uploaded to the GPU storage buffer.
///
/// Layout matches the WGSL `EmfFieldCell` struct (64 bytes = 3 field f32 + 10 manifold f32 + 3 Stokes f32).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default, PartialEq)]
pub struct EmfFieldCell {
    pub amplitude: f32,
    pub phase: f32,
    pub frequency: f32,
    // 10D manifold coordinate:
    pub scale: f32,
    pub attention_depth: f32,
    pub epistemic_weight: f32,
    pub topological_spin: f32,
    pub temporal_decay: f32,
    pub entropy_bias: f32,
    pub spatial_phase: f32,
    pub recurrence_frequency: f32,
    pub density_threshold: f32,
    pub manifold_curvature: f32,
    // Stokes polarization parameters:
    pub stokes_q: f32,
    pub stokes_u: f32,
    pub stokes_v: f32,
}

const _: () = {
    assert!(core::mem::size_of::<EmfFieldCell>() == 64);
    assert!(core::mem::align_of::<EmfFieldCell>() == 4);
    assert!(core::mem::offset_of!(EmfFieldCell, amplitude) == 0);
    assert!(core::mem::offset_of!(EmfFieldCell, phase) == 4);
    assert!(core::mem::offset_of!(EmfFieldCell, frequency) == 8);
    assert!(core::mem::offset_of!(EmfFieldCell, scale) == 12);
    assert!(core::mem::offset_of!(EmfFieldCell, attention_depth) == 16);
    assert!(core::mem::offset_of!(EmfFieldCell, epistemic_weight) == 20);
    assert!(core::mem::offset_of!(EmfFieldCell, topological_spin) == 24);
    assert!(core::mem::offset_of!(EmfFieldCell, temporal_decay) == 28);
    assert!(core::mem::offset_of!(EmfFieldCell, entropy_bias) == 32);
    assert!(core::mem::offset_of!(EmfFieldCell, spatial_phase) == 36);
    assert!(core::mem::offset_of!(EmfFieldCell, recurrence_frequency) == 40);
    assert!(core::mem::offset_of!(EmfFieldCell, density_threshold) == 44);
    assert!(core::mem::offset_of!(EmfFieldCell, manifold_curvature) == 48);
    assert!(core::mem::offset_of!(EmfFieldCell, stokes_q) == 52);
    assert!(core::mem::offset_of!(EmfFieldCell, stokes_u) == 56);
    assert!(core::mem::offset_of!(EmfFieldCell, stokes_v) == 60);
};

impl EmfFieldCell {
    /// Migrate a legacy 52-byte (`13 × f32`) v1 cell to the 64-byte (`16 × f32`) v2 layout,
    /// initializing the 3 Stokes polarization parameters `(stokes_q, stokes_u, stokes_v)` to `0.0`.
    #[inline]
    pub const fn from_v1_13f32(v1: [f32; 13]) -> Self {
        Self {
            amplitude: v1[0],
            phase: v1[1],
            frequency: v1[2],
            scale: v1[3],
            attention_depth: v1[4],
            epistemic_weight: v1[5],
            topological_spin: v1[6],
            temporal_decay: v1[7],
            entropy_bias: v1[8],
            spatial_phase: v1[9],
            recurrence_frequency: v1[10],
            density_threshold: v1[11],
            manifold_curvature: v1[12],
            stokes_q: 0.0,
            stokes_u: 0.0,
            stokes_v: 0.0,
        }
    }

    /// Computes degree of polarization $\Pi \in [0, 1]$ from Stokes $(Q, U, V)$ and amplitude.
    #[inline]
    pub fn degree_of_polarization(&self) -> f32 {
        let pol_mag = (self.stokes_q * self.stokes_q
            + self.stokes_u * self.stokes_u
            + self.stokes_v * self.stokes_v)
            .sqrt();
        (pol_mag / (self.amplitude.abs() + 1.0e-5)).clamp(0.0, 1.0)
    }
}

/// Slice parameter uniform (64 bytes, std140-aligned).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct EmfSliceUniform {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub nt: u32,
    pub slice_z: u32,
    pub slice_t: u32,
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z_min: f32,
    pub z_max: f32,
    pub amplitude_scale: f32,
    pub phase_offset: f32,
    pub manifold_gain: f32,
    pub projection_mode: u32,
}

const _: () = {
    assert!(core::mem::size_of::<EmfSliceUniform>() == 64);
    assert!(core::mem::align_of::<EmfSliceUniform>() == 16);
    assert!(core::mem::offset_of!(EmfSliceUniform, nx) == 0);
    assert!(core::mem::offset_of!(EmfSliceUniform, slice_z) == 16);
    assert!(core::mem::offset_of!(EmfSliceUniform, x_min) == 24);
    assert!(core::mem::offset_of!(EmfSliceUniform, amplitude_scale) == 48);
    assert!(core::mem::offset_of!(EmfSliceUniform, projection_mode) == 60);
};

impl Default for EmfSliceUniform {
    fn default() -> Self {
        Self {
            nx: 1,
            ny: 1,
            nz: 1,
            nt: 1,
            slice_z: 0,
            slice_t: 0,
            x_min: -1.0,
            x_max: 1.0,
            y_min: -1.0,
            y_max: 1.0,
            z_min: -1.0,
            z_max: 1.0,
            amplitude_scale: 1.0,
            phase_offset: 0.0,
            manifold_gain: 1.0,
            projection_mode: 0,
        }
    }
}

#[inline]
fn wgsl_fract(x: f32) -> f32 {
    x - x.floor()
}

#[inline]
fn wgsl_mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

/// CPU reference matching WGSL `phase_to_sigma(phase)`.
#[inline]
pub fn phase_to_sigma_cpu(phase: f32) -> f32 {
    wgsl_fract(phase / TWO_PI_F32)
}

/// CPU reference matching WGSL `sigma_to_cie_xyz(sigma)`.
#[inline]
pub fn sigma_to_cie_xyz_cpu(sigma: f32) -> [f32; 3] {
    let s = wgsl_fract(sigma);
    let lambda = 380.0 + (s * 370.0);
    let x1 = 1.056 * (-0.5 * ((lambda - 599.8) / 43.2).powi(2)).exp();
    let x2 = 0.362 * (-0.5 * ((lambda - 442.0) / 32.0).powi(2)).exp();
    let x3 = -0.065 * (-0.5 * ((lambda - 501.1) / 20.4).powi(2)).exp();
    let x = x1 + x2 + x3;
    let y1 = 0.821 * (-0.5 * ((lambda - 568.8) / 46.9).powi(2)).exp();
    let y2 = 0.286 * (-0.5 * ((lambda - 530.9) / 16.3).powi(2)).exp();
    let y = y1 + y2;
    let z1 = 1.217 * (-0.5 * ((lambda - 437.0) / 11.8).powi(2)).exp();
    let z2 = 0.681 * (-0.5 * ((lambda - 459.0) / 26.0).powi(2)).exp();
    let z = z1 + z2;
    [x, y, z]
}

/// CPU reference matching WGSL `xyz_to_linear_srgb(xyz)`.
#[inline]
pub fn xyz_to_linear_srgb_cpu(xyz: [f32; 3]) -> [f32; 3] {
    let r = 3.240_454_2 * xyz[0] - 1.537_138_5 * xyz[1] - 0.498_531_4 * xyz[2];
    let g = -0.969_266_0 * xyz[0] + 1.876_010_8 * xyz[1] + 0.041_556_0 * xyz[2];
    let b = 0.055_643_4 * xyz[0] - 0.204_025_9 * xyz[1] + 1.057_225_2 * xyz[2];
    [r.max(0.0), g.max(0.0), b.max(0.0)]
}

/// CPU reference matching WGSL `sigma_to_linear_rgb(sigma)` across all 6 full-spectrum bands `[0.0, 6.0)`.
pub fn sigma_to_linear_rgb_cpu(sigma: f32) -> [f32; 3] {
    let band = sigma.floor() as i32;
    let s = wgsl_fract(sigma);
    if band == 0 {
        xyz_to_linear_srgb_cpu(sigma_to_cie_xyz_cpu(s))
    } else if band == 1 {
        [
            wgsl_mix(0.35, 1.0, s),
            wgsl_mix(0.02, 0.55, s * s),
            wgsl_mix(0.05, 0.08, s),
        ]
    } else if band == 2 {
        let r = (s * 1.6).clamp(0.0, 1.0);
        let g = ((s - 0.35) * 1.55).clamp(0.0, 1.0);
        let b = ((s * 3.141_592_7).sin() * 0.75 + ((s - 0.75) * 4.0).max(0.0)).clamp(0.0, 1.0);
        [r, g, b]
    } else if band == 3 {
        [
            wgsl_mix(0.42, 0.05, s),
            wgsl_mix(0.05, 0.92, s),
            wgsl_mix(0.75, 1.0, s),
        ]
    } else if band == 4 {
        let c = s.powf(0.85);
        [c * 0.75, c * 0.95, (c * 1.15).clamp(0.0, 1.0)]
    } else {
        [wgsl_mix(0.15, 0.95, s * s), wgsl_mix(0.35, 0.98, s), 1.0]
    }
}

/// CPU reference matching WGSL `sample_field_bilinear(uv)`.
pub fn sample_field_bilinear_cpu(
    params: &EmfSliceUniform,
    field: &[EmfFieldCell],
    uv: [f32; 2],
) -> EmfFieldCell {
    let sample_at = |ix: u32, iy: u32| -> EmfFieldCell {
        let idx = (((params.slice_t * params.nz + params.slice_z) * params.ny + iy) * params.nx
            + ix) as usize;
        field[idx]
    };
    let fx = uv[0].clamp(0.0, 0.999) * (params.nx as f32);
    let fy = uv[1].clamp(0.0, 0.999) * (params.ny as f32);
    let ix0 = fx.floor() as u32;
    let iy0 = fy.floor() as u32;
    let ix1 = (ix0 + 1).min(params.nx - 1);
    let iy1 = (iy0 + 1).min(params.ny - 1);
    let tx = fx - (ix0 as f32);
    let ty = fy - (iy0 as f32);

    let c00 = sample_at(ix0, iy0);
    let c10 = sample_at(ix1, iy0);
    let c01 = sample_at(ix0, iy1);
    let c11 = sample_at(ix1, iy1);
    let bilerp =
        |v00: f32, v10: f32, v01: f32, v11: f32| wgsl_mix(wgsl_mix(v00, v10, tx), wgsl_mix(v01, v11, tx), ty);

    EmfFieldCell {
        amplitude: bilerp(c00.amplitude, c10.amplitude, c01.amplitude, c11.amplitude),
        phase: bilerp(c00.phase, c10.phase, c01.phase, c11.phase),
        frequency: bilerp(c00.frequency, c10.frequency, c01.frequency, c11.frequency),
        scale: bilerp(c00.scale, c10.scale, c01.scale, c11.scale),
        attention_depth: bilerp(
            c00.attention_depth,
            c10.attention_depth,
            c01.attention_depth,
            c11.attention_depth,
        ),
        epistemic_weight: bilerp(
            c00.epistemic_weight,
            c10.epistemic_weight,
            c01.epistemic_weight,
            c11.epistemic_weight,
        ),
        topological_spin: bilerp(
            c00.topological_spin,
            c10.topological_spin,
            c01.topological_spin,
            c11.topological_spin,
        ),
        temporal_decay: bilerp(
            c00.temporal_decay,
            c10.temporal_decay,
            c01.temporal_decay,
            c11.temporal_decay,
        ),
        entropy_bias: bilerp(
            c00.entropy_bias,
            c10.entropy_bias,
            c01.entropy_bias,
            c11.entropy_bias,
        ),
        spatial_phase: bilerp(
            c00.spatial_phase,
            c10.spatial_phase,
            c01.spatial_phase,
            c11.spatial_phase,
        ),
        recurrence_frequency: bilerp(
            c00.recurrence_frequency,
            c10.recurrence_frequency,
            c01.recurrence_frequency,
            c11.recurrence_frequency,
        ),
        density_threshold: bilerp(
            c00.density_threshold,
            c10.density_threshold,
            c01.density_threshold,
            c11.density_threshold,
        ),
        manifold_curvature: bilerp(
            c00.manifold_curvature,
            c10.manifold_curvature,
            c01.manifold_curvature,
            c11.manifold_curvature,
        ),
        stokes_q: bilerp(c00.stokes_q, c10.stokes_q, c01.stokes_q, c11.stokes_q),
        stokes_u: bilerp(c00.stokes_u, c10.stokes_u, c01.stokes_u, c11.stokes_u),
        stokes_v: bilerp(c00.stokes_v, c10.stokes_v, c01.stokes_v, c11.stokes_v),
    }
}

/// CPU reference matching WGSL `fragment_main(input)` in `shaders/emf_volumetric.wgsl`.
///
/// Honors `projection_mode == 1u`, masks out invalid/missing cells with `epistemic_weight <= 0.0`
/// in `projection_mode == 0u`, and selects `cell.recurrence_frequency` (`spectral_sigma` in `[0.0, 6.0)`)
/// when `cell.recurrence_frequency > 0.0`.
pub fn evaluate_emf_fragment_cpu(
    params: &EmfSliceUniform,
    field: &[EmfFieldCell],
    uv: [f32; 2],
) -> [f32; 4] {
    if params.projection_mode == 1 {
        let ix = ((uv[0].clamp(0.0, 1.0) * (params.nx as f32)) as u32).min(params.nx - 1);
        let iy = ((uv[1].clamp(0.0, 1.0) * (params.ny as f32)) as u32).min(params.ny - 1);
        let idx = (((params.slice_t * params.nz + params.slice_z) * params.ny + iy) * params.nx
            + ix) as usize;
        let display = field[idx];
        let brightness = (display.amplitude * params.amplitude_scale).clamp(0.0, 1.0);
        return [brightness, brightness, brightness, display.epistemic_weight];
    }

    let cell = sample_field_bilinear_cpu(params, field, uv);
    if cell.epistemic_weight <= 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let amp = cell.amplitude.abs() * params.amplitude_scale;
    let brightness = amp.clamp(0.0, 4.0);

    let sigma = if cell.recurrence_frequency > 0.0 {
        cell.recurrence_frequency
    } else {
        phase_to_sigma_cpu(cell.phase + params.phase_offset)
    };
    let base_rgb = sigma_to_linear_rgb_cpu(sigma);

    let manifold_gain = params.manifold_gain;
    let saturation = (cell.scale * manifold_gain).clamp(0.0, 1.5);
    let luminance = 0.2126 * base_rgb[0] + 0.7152 * base_rgb[1] + 0.0722 * base_rgb[2];
    let saturated_rgb = [
        wgsl_mix(luminance, base_rgb[0], saturation),
        wgsl_mix(luminance, base_rgb[1], saturation),
        wgsl_mix(luminance, base_rgb[2], saturation),
    ];

    let degree_of_polarization = cell.degree_of_polarization();
    let pol_highlight = 1.0 + 0.35 * degree_of_polarization;

    let curvature_glow = (cell.manifold_curvature.abs() * manifold_gain).clamp(0.0, 1.0);
    let glow_tint = sigma_to_linear_rgb_cpu(phase_to_sigma_cpu(cell.spatial_phase));
    let glow_scale = curvature_glow * (0.3 + 0.2 * degree_of_polarization);

    let rgb = [
        saturated_rgb[0] * brightness * pol_highlight + glow_tint[0] * glow_scale,
        saturated_rgb[1] * brightness * pol_highlight + glow_tint[1] * glow_scale,
        saturated_rgb[2] * brightness * pol_highlight + glow_tint[2] * glow_scale,
    ];
    let alpha = (brightness * 0.5 + 0.1).clamp(0.0, 1.0);

    [rgb[0], rgb[1], rgb[2], alpha]
}

/// Bounded calculation of field cell memory requirements.
pub fn checked_grid_cell_count(nx: u32, ny: u32, nz: u32, nt: u32) -> Option<u64> {
    u64::from(nx)
        .checked_mul(u64::from(ny))?
        .checked_mul(u64::from(nz))?
        .checked_mul(u64::from(nt))
}

/// Bounded byte size calculation for a given cell count.
pub fn checked_field_bytes(cells: u64) -> Option<u64> {
    cells.checked_mul(core::mem::size_of::<EmfFieldCell>() as u64)
}

/// High-level descriptor for constructing the EMF visualization pipeline.
#[derive(Clone, Debug)]
pub struct EmfPipelineDesc {
    pub color_format: etalon::TextureFormat,
    pub max_cells: u64,
}

/// Complete, decoupled GPU state for EMF wave slice visualization.
pub struct EmfPipeline {
    pub param_buf: etalon::Buffer,
    pub params_layout: etalon::BindGroupLayout,
    pub field_layout: etalon::BindGroupLayout,
    pub params_bind_group: etalon::BindGroup,
    pub pipeline: etalon::RenderPipeline,
}

impl EmfPipeline {
    pub fn new(device: &etalon::Device, color_format: etalon::TextureFormat) -> Self {
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-emf-volumetric-shader"),
            source: etalon::ShaderSource::Wgsl(EMF_VOLUMETRIC_WGSL.into()),
        });

        let param_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-emf-slice-params"),
            size: core::mem::size_of::<EmfSliceUniform>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("etalon-emf-params-layout"),
            entries: &[etalon::BindGroupLayoutEntry {
                binding: 0,
                visibility: etalon::ShaderStages::VERTEX | etalon::ShaderStages::FRAGMENT,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: core::num::NonZeroU64::new(
                        core::mem::size_of::<EmfSliceUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let field_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("etalon-emf-field-layout"),
            entries: &[etalon::BindGroupLayoutEntry {
                binding: 0,
                visibility: etalon::ShaderStages::FRAGMENT,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: core::num::NonZeroU64::new(
                        core::mem::size_of::<EmfFieldCell>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let params_bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("etalon-emf-params-bind-group"),
            layout: &params_layout,
            entries: &[etalon::BindGroupEntry {
                binding: 0,
                resource: param_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("etalon-emf-pipeline-layout"),
            bind_group_layouts: &[Some(&params_layout), Some(&field_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("etalon-emf-render-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(etalon::ColorTargetState {
                    format: color_format,
                    blend: Some(etalon::BlendState::ALPHA_BLENDING),
                    write_mask: etalon::ColorWrites::ALL,
                })],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: etalon::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: etalon::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            param_buf,
            params_layout,
            field_layout,
            params_bind_group,
            pipeline,
        }
    }

    /// Update slice uniform parameters in caller buffer queue.
    pub fn update_params(&self, queue: &etalon::Queue, params: &EmfSliceUniform) {
        queue.write_buffer(&self.param_buf, 0, bytemuck::bytes_of(params));
    }

    /// Record a full-screen EMF slice draw pass into the active render pass.
    pub fn render_slice<'a>(
        &'a self,
        rpass: &mut etalon::RenderPass<'a>,
        field_bind_group: &'a etalon::BindGroup,
    ) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.params_bind_group, &[]);
        rpass.set_bind_group(1, field_bind_group, &[]);
        rpass.draw(0..6, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emf_field_cell_v2_layout_and_v1_migration() {
        assert_eq!(core::mem::size_of::<EmfFieldCell>(), 64);
        assert_eq!(core::mem::size_of::<EmfSliceUniform>(), 64);
        let v1 = [
            1.0, 0.5, 2.4e9, 0.8, 0.3, 0.9, 0.1, 0.05, 0.2, 0.4, 0.6, 0.7, 0.15,
        ];
        let cell = EmfFieldCell::from_v1_13f32(v1);
        assert_eq!(cell.amplitude, 1.0);
        assert_eq!(cell.manifold_curvature, 0.15);
        assert_eq!(cell.stokes_q, 0.0);
        assert_eq!(cell.stokes_u, 0.0);
        assert_eq!(cell.stokes_v, 0.0);
        assert_eq!(cell.degree_of_polarization(), 0.0);
    }

    #[test]
    fn emf_volumetric_wgsl_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(EMF_VOLUMETRIC_WGSL)
            .expect("emf_volumetric.wgsl must parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        );
        validator
            .validate(&module)
            .expect("emf_volumetric.wgsl must validate");
    }

    #[test]
    fn emf_fragment_masks_zero_epistemic_weight_and_selects_full_spectrum_sigma() {
        let params = EmfSliceUniform::default();
        let mut cell = EmfFieldCell {
            amplitude: 1.5,
            phase: 1.0,
            frequency: 5.0e14,
            scale: 1.0,
            epistemic_weight: 0.0,
            recurrence_frequency: 2.5,
            ..Default::default()
        };

        // Missing/invalid cell with epistemic_weight == 0.0 must be masked to transparent zero
        let masked = evaluate_emf_fragment_cpu(&params, &[cell], [0.5, 0.5]);
        assert_eq!(masked, [0.0, 0.0, 0.0, 0.0]);

        // Valid cell with recurrence_frequency = 2.5 (Band 2 Infrared/Thermal) vs 4.5 (Band 4 X-Ray)
        cell.epistemic_weight = 1.0;
        let ir_rgba = evaluate_emf_fragment_cpu(&params, &[cell], [0.5, 0.5]);
        assert!(ir_rgba[3] > 0.0);

        cell.recurrence_frequency = 4.5;
        let xray_rgba = evaluate_emf_fragment_cpu(&params, &[cell], [0.5, 0.5]);
        assert_ne!(ir_rgba, xray_rgba);

        // Fallback to phase_to_sigma when recurrence_frequency <= 0.0
        cell.recurrence_frequency = 0.0;
        let phase_rgba = evaluate_emf_fragment_cpu(&params, &[cell], [0.5, 0.5]);
        assert!(phase_rgba[3] > 0.0);
        assert_ne!(phase_rgba, xray_rgba);
    }
}

#[cfg(test)]
#[path = "emf_gain_tests.rs"]
mod gain_tests;
