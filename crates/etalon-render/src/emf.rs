//! Electromagnetic field and signal wave-propagation visualization pipeline.
//!
//! Provides the complete GPU rendering pipeline for drawing 2D slices of 4D EMF field grids
//! (x * y * z * t) with 10D manifold records preserved.

use bytemuck::{Pod, Zeroable};

/// Raw WGSL shader source for the EMF volumetric visualizer.
pub const EMF_VOLUMETRIC_WGSL: &str = r#"// EMF 5D volumetric visualizer — renders 2D slices of the 4D EMF field grid
// (x×y×z×t) with 10D manifold tags mapped to color.
//
// Fragment shader samples a storage buffer containing the flat field grid
// (indexed [t][z][y][x]) and maps:
//   amplitude  → brightness (HDR value)
//   phase      → hue (via σ → CIE XYZ → linear sRGB)
//   manifold.scale + attention_depth → saturation gain
//   manifold.manifold_curvature → rim glow / contour enhancement
//
// @group(0) camera + slice params · @group(1) field grid storage

struct EmfSliceUniform {
    // Grid dimensions.
    nx: u32,
    ny: u32,
    nz: u32,
    nt: u32,
    // Current slice indices.
    slice_z: u32,
    slice_t: u32,
    // Grid bounds: [x_min, x_max, y_min, y_max, z_min, z_max].
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    z_min: f32,
    z_max: f32,
    // Display controls.
    amplitude_scale: f32,
    phase_offset: f32,
    manifold_gain: f32,
    _pad: f32,
};

// Per-cell field data: amplitude, phase, frequency + 10D manifold coords.
// Matches the host-side EmfFieldCell struct (48 bytes = 3 f32 + 10 f32 + 1 pad).
struct EmfFieldCell {
    amplitude: f32,
    phase: f32,
    frequency: f32,
    // 10D manifold coordinate.
    scale: f32,
    attention_depth: f32,
    epistemic_weight: f32,
    topological_spin: f32,
    temporal_decay: f32,
    entropy_bias: f32,
    spatial_phase: f32,
    recurrence_frequency: f32,
    density_threshold: f32,
    manifold_curvature: f32,
};

@group(0) @binding(0) var<uniform> params: EmfSliceUniform;
@group(1) @binding(0) var<storage, read> field: array<EmfFieldCell>;

const TWO_PI: f32 = 6.283185307;

// σ → CIE 1931 XYZ → linear sRGB (shared spectral mapping).
fn sigma_to_cie_xyz(sigma: f32) -> vec3<f32> {
    let s = fract(sigma);
    let lambda = 400.0 + (s * 300.0);
    let x1 = 1.056 * exp(-0.5 * pow((lambda - 599.8) / 43.2, 2.0));
    let x2 = 0.362 * exp(-0.5 * pow((lambda - 442.0) / 32.0, 2.0));
    let x3 = -0.065 * exp(-0.5 * pow((lambda - 501.1) / 20.4, 2.0));
    let X = x1 + x2 + x3;
    let y1 = 0.821 * exp(-0.5 * pow((lambda - 568.8) / 46.9, 2.0));
    let y2 = 0.286 * exp(-0.5 * pow((lambda - 530.9) / 16.3, 2.0));
    let Y = y1 + y2;
    let z1 = 1.217 * exp(-0.5 * pow((lambda - 437.0) / 11.8, 2.0));
    let z2 = 0.681 * exp(-0.5 * pow((lambda - 459.0) / 26.0, 2.0));
    let Z = z1 + z2;
    return vec3<f32>(X, Y, Z);
}

fn xyz_to_linear_srgb(xyz: vec3<f32>) -> vec3<f32> {
    let R = 3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z;
    let G = -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z;
    let B = 0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z;
    return max(vec3<f32>(R, G, B), vec3<f32>(0.0));
}

fn sigma_to_linear_rgb(sigma: f32) -> vec3<f32> {
    return xyz_to_linear_srgb(sigma_to_cie_xyz(sigma));
}

// Map phase (radians) to σ in [0,1) for spectral coloring.
fn phase_to_sigma(phase: f32) -> f32 {
    return fract(phase / TWO_PI);
}

// Sample the field grid at integer (ix, iy) for the current (slice_z, slice_t).
fn sample_field(ix: u32, iy: u32) -> EmfFieldCell {
    let idx = ((params.slice_t * params.nz + params.slice_z) * params.ny + iy) * params.nx + ix;
    return field[idx];
}

// Bilinear sample at normalized UV in [0,1].
fn sample_field_bilinear(uv: vec2<f32>) -> EmfFieldCell {
    let fx = clamp(uv.x, 0.0, 0.999) * f32(params.nx);
    let fy = clamp(uv.y, 0.0, 0.999) * f32(params.ny);
    let ix0 = u32(floor(fx));
    let iy0 = u32(floor(fy));
    let ix1 = min(ix0 + 1u, params.nx - 1u);
    let iy1 = min(iy0 + 1u, params.ny - 1u);
    let tx = fx - f32(ix0);
    let ty = fy - f32(iy0);

    let c00 = sample_field(ix0, iy0);
    let c10 = sample_field(ix1, iy0);
    let c01 = sample_field(ix0, iy1);
    let c11 = sample_field(ix1, iy1);

    var result: EmfFieldCell;
    result.amplitude = mix(mix(c00.amplitude, c10.amplitude, tx), mix(c01.amplitude, c11.amplitude, tx), ty);
    result.phase = mix(mix(c00.phase, c10.phase, tx), mix(c01.phase, c11.phase, tx), ty);
    result.frequency = mix(mix(c00.frequency, c10.frequency, tx), mix(c01.frequency, c11.frequency, tx), ty);
    result.scale = mix(mix(c00.scale, c10.scale, tx), mix(c01.scale, c11.scale, tx), ty);
    result.attention_depth = mix(mix(c00.attention_depth, c10.attention_depth, tx), mix(c01.attention_depth, c11.attention_depth, tx), ty);
    result.epistemic_weight = mix(mix(c00.epistemic_weight, c10.epistemic_weight, tx), mix(c01.epistemic_weight, c11.epistemic_weight, tx), ty);
    result.topological_spin = mix(mix(c00.topological_spin, c10.topological_spin, tx), mix(c01.topological_spin, c11.topological_spin, tx), ty);
    result.temporal_decay = mix(mix(c00.temporal_decay, c10.temporal_decay, tx), mix(c01.temporal_decay, c11.temporal_decay, tx), ty);
    result.entropy_bias = mix(mix(c00.entropy_bias, c10.entropy_bias, tx), mix(c01.entropy_bias, c11.entropy_bias, tx), ty);
    result.spatial_phase = mix(mix(c00.spatial_phase, c10.spatial_phase, tx), mix(c01.spatial_phase, c11.spatial_phase, tx), ty);
    result.recurrence_frequency = mix(mix(c00.recurrence_frequency, c10.recurrence_frequency, tx), mix(c01.recurrence_frequency, c11.recurrence_frequency, tx), ty);
    result.density_threshold = mix(mix(c00.density_threshold, c10.density_threshold, tx), mix(c01.density_threshold, c11.density_threshold, tx), ty);
    result.manifold_curvature = mix(mix(c00.manifold_curvature, c10.manifold_curvature, tx), mix(c01.manifold_curvature, c11.manifold_curvature, tx), ty);
    return result;
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    // Full-screen triangle strip (2 triangles covering [-1,1]²).
    let positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    let uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );
    var out: VertexOutput;
    let p = positions[vid];
    out.clip_position = vec4<f32>(p, 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let cell = sample_field_bilinear(input.uv);

    // Amplitude → brightness (HDR, scaled).
    let amp = abs(cell.amplitude) * params.amplitude_scale;
    let brightness = clamp(amp, 0.0, 4.0);

    // Phase → hue via spectral mapping.
    let sigma = phase_to_sigma(cell.phase + params.phase_offset);
    let base_rgb = sigma_to_linear_rgb(sigma);

    // Manifold fields modulate saturation and tint.
    let manifold_gain = params.manifold_gain;
    let saturation = clamp(cell.scale * manifold_gain, 0.0, 1.5);
    let luminance = dot(base_rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let saturated_rgb = mix(vec3<f32>(luminance), base_rgb, saturation);

    // Curvature → rim glow / contour enhancement.
    let curvature_glow = clamp(abs(cell.manifold_curvature) * manifold_gain, 0.0, 1.0);
    let glow_tint = sigma_to_linear_rgb(phase_to_sigma(cell.spatial_phase));
    let rgb = saturated_rgb * brightness + glow_tint * curvature_glow * 0.3;

    // Alpha: amplitude-based opacity, fades to transparent at zero amplitude.
    let alpha = clamp(brightness * 0.5 + 0.1, 0.0, 1.0);

    return vec4<f32>(rgb, alpha);
}
"#;

/// Per-cell field data uploaded to the GPU storage buffer.
///
/// Layout matches the WGSL `EmfFieldCell` struct (52 bytes = 3 field f32 + 10 manifold f32).
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
}

const _: () = {
    assert!(core::mem::size_of::<EmfFieldCell>() == 52);
};

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
    pub _pad: f32,
}

const _: () = {
    assert!(core::mem::size_of::<EmfSliceUniform>() == 64);
    assert!(core::mem::align_of::<EmfSliceUniform>() == 16);
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
            _pad: 0.0,
        }
    }
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
            bind_group_layouts: &[&params_layout, &field_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("etalon-emf-render-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
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
            multiview: None,
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
