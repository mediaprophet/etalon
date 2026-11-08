//! Volumetric ray-marcher and holographic projection pass.
//!
//! Evaluates procedural Signed Distance Fields (SDFs), constructive wave interference,
//! logarithmic depth ($10^{-4}\text{ m}$ to $10^9\text{ m}$), ADM 3+1 gravitational lensing
//! geodesics, non-Euclidean portal ray transport, and the Epistemic Honesty Lens.

use bytemuck::{Pod, Zeroable};

/// Raw WGSL shader source for the procedural holographic volumetric engine.
pub const HOLOGRAPHIC_WGSL: &str = include_str!("shaders/holographic_standalone.wgsl");

/// Logarithmic depth encoding (§4.2.1):
/// $$z_{\text{log}} = \frac{\log_2(\max(10^{-6}, 1.0 + C \cdot z_{\text{view}}))}{\log_2(1.0 + C \cdot z_{\text{far}})}$$
#[inline]
pub fn compute_log_depth(z_view: f32, z_far: f32, c_scale: f32) -> f32 {
    let c = c_scale.max(1.0e-6);
    let numer = (1.0 + c * z_view).max(1.0e-6).log2();
    let denom = (1.0 + c * z_far.max(1.0e-6)).log2().max(1.0e-6);
    (numer / denom).clamp(0.0, 1.0)
}

/// Exact inverse of [`compute_log_depth`], recovering view-space depth $z_{\text{view}}$ from $z_{\text{log}} \in [0, 1]$.
#[inline]
pub fn invert_log_depth(z_log: f32, z_far: f32, c_scale: f32) -> f32 {
    let c = c_scale.max(1.0e-6);
    let denom = (1.0 + c * z_far.max(1.0e-6)).log2().max(1.0e-6);
    ((z_log * denom).exp2() - 1.0) / c
}

/// ADM 3+1 weak-to-moderate field Schwarzschild null geodesic step (§4.2.2).
///
/// Advances `ray_pos` by `step_ds` and deflects `ray_dir` toward `mass_center` with
/// Einstein gravitational deflection acceleration $\mathbf{a}_{\perp} = -\frac{2 r_s}{r^3}\mathbf{r}_{\perp}$.
#[inline]
pub fn adm_geodesic_deflect_ray(
    ray_pos: [f32; 3],
    ray_dir: [f32; 3],
    mass_center: [f32; 3],
    schwarzschild_radius: f32,
    step_ds: f32,
) -> ([f32; 3], [f32; 3]) {
    let dir = normalize3(ray_dir);
    if schwarzschild_radius <= 0.0 || step_ds <= 0.0 {
        let ds = step_ds.max(0.0);
        return (
            [
                ray_pos[0] + dir[0] * ds,
                ray_pos[1] + dir[1] * ds,
                ray_pos[2] + dir[2] * ds,
            ],
            dir,
        );
    }
    let rel = [
        ray_pos[0] - mass_center[0],
        ray_pos[1] - mass_center[1],
        ray_pos[2] - mass_center[2],
    ];
    let r = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2])
        .sqrt()
        .max(schwarzschild_radius * 1.5)
        .max(1.0e-4);
    let r3 = r * r * r;
    let r_dot_d = rel[0] * dir[0] + rel[1] * dir[1] + rel[2] * dir[2];
    let rel_perp = [
        rel[0] - dir[0] * r_dot_d,
        rel[1] - dir[1] * r_dot_d,
        rel[2] - dir[2] * r_dot_d,
    ];
    let factor = -2.0 * schwarzschild_radius * step_ds / r3;
    let next_dir = normalize3([
        dir[0] + rel_perp[0] * factor,
        dir[1] + rel_perp[1] * factor,
        dir[2] + rel_perp[2] * factor,
    ]);
    let next_pos = [
        ray_pos[0] + next_dir[0] * step_ds,
        ray_pos[1] + next_dir[1] * step_ds,
        ray_pos[2] + next_dir[2] * step_ds,
    ];
    (next_pos, next_dir)
}

/// Non-Euclidean portal ray transport (§4.2.2).
///
/// Transforms `(ray_pos, ray_dir)` from the entrance portal's world space through `portal_in_inv`
/// into local portal coordinates and out through `portal_out` into destination world coordinates.
/// Supports both column-major (WGSL/wgpu standard) and row-major $4\times 4$ affine matrices.
#[inline]
pub fn portal_transport_ray(
    ray_pos: [f32; 3],
    ray_dir: [f32; 3],
    portal_in_inv: &[[f32; 4]; 4],
    portal_out: &[[f32; 4]; 4],
) -> ([f32; 3], [f32; 3]) {
    let local_pos = mat4_transform_point(portal_in_inv, ray_pos);
    let out_pos = mat4_transform_point(portal_out, local_pos);
    let local_dir = mat4_transform_vec(portal_in_inv, ray_dir);
    let out_dir = normalize3(mat4_transform_vec(portal_out, local_dir));
    (out_pos, out_dir)
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1.0e-8 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[inline]
fn is_row_major_affine(m: &[[f32; 4]; 4]) -> bool {
    let col3_trans = m[0][3].abs() + m[1][3].abs() + m[2][3].abs();
    let row3_trans = m[3][0].abs() + m[3][1].abs() + m[3][2].abs();
    col3_trans > 0.0 && row3_trans == 0.0
}

#[inline]
fn mat4_transform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    if is_row_major_affine(m) {
        [
            m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
        ]
    } else {
        [
            m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
            m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
            m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
        ]
    }
}

#[inline]
fn mat4_transform_vec(m: &[[f32; 4]; 4], v: [f32; 3]) -> [f32; 3] {
    if is_row_major_affine(m) {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    } else {
        [
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
        ]
    }
}

/// 96-byte std140/std430 aligned GPU representation of a semantic holographic entity.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct HolographicNodeGpu {
    /// World space position xyz, bounding sphere / proxy scale w
    pub position_scale: [f32; 4],
    /// Half extents xyz, primitive type w (0: Sphere, 1: RoundedBox, 2: Capsule, 3: Metaball, 4: InterferenceBridge)
    pub half_extents_primitive: [f32; 4],
    /// Base frequency x, amplitude y, phase z, manifold curvature w
    pub frequency_amplitude: [f32; 4],
    /// 10D Manifold parameters: epistemic_q x, topological_spin y, temporal_decay z, density_threshold w
    pub manifold_tensor: [f32; 4],
    /// Target endpoint xyz (for relational bridges), coherence weight w
    pub connection_endpoint: [f32; 4],
    /// Emissive tint rgba
    pub emissive_color: [f32; 4],
}

const _: () = {
    assert!(core::mem::size_of::<HolographicNodeGpu>() == 96);
    assert!(core::mem::align_of::<HolographicNodeGpu>() == 16);
};

impl Default for HolographicNodeGpu {
    fn default() -> Self {
        Self {
            position_scale: [0.0, 0.0, 0.0, 1.0],
            half_extents_primitive: [0.5, 0.5, 0.5, 0.0],
            frequency_amplitude: [1.0, 1.0, 0.0, 0.0],
            manifold_tensor: [1.0, 0.0, 0.0, 0.5],
            connection_endpoint: [0.0, 0.0, 0.0, 0.0],
            emissive_color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Camera uniform (128 bytes, std140-aligned).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct CameraUniform {
    pub view_projection: [[f32; 4]; 4],
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub tensor_mode: u32,
    pub _padding0: [f32; 4],
    pub _padding1: [f32; 4],
    pub _padding2: [f32; 4],
}

const _: () = {
    assert!(core::mem::size_of::<CameraUniform>() == 128);
    assert!(core::mem::align_of::<CameraUniform>() == 16);
};

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_projection: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            yaw: 0.0,
            pitch: 0.0,
            zoom: 1.0,
            tensor_mode: 0,
            _padding0: [0.0; 4],
            _padding1: [0.0; 4],
            _padding2: [0.0; 4],
        }
    }
}

/// Observer standpoint uniform (128 bytes, std140-aligned).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct ObserverStandpointUniform {
    pub standpoint_hash: [u32; 2],
    pub session_nonce: [u32; 2],
    pub epistemic_q: f32,
    pub t_slice: f32,
    pub t_window: f32,
    pub deontic_lane: u32,
    pub standpoint_class: u32,
    pub fabric_gate: u32,
    pub _padding0: [f32; 2],
    pub _padding1: [f32; 4],
    pub _padding2: [f32; 4],
    pub _padding3: [f32; 4],
    pub _padding4: [f32; 4],
    pub _padding5: [f32; 4],
}

const _: () = {
    assert!(core::mem::size_of::<ObserverStandpointUniform>() == 128);
    assert!(core::mem::align_of::<ObserverStandpointUniform>() == 16);
};

impl Default for ObserverStandpointUniform {
    fn default() -> Self {
        Self {
            standpoint_hash: [0, 0],
            session_nonce: [0, 0],
            epistemic_q: 1.0,
            t_slice: 0.0,
            t_window: 1.0,
            deontic_lane: 0,
            standpoint_class: 0,
            fabric_gate: 0,
            _padding0: [0.0; 2],
            _padding1: [0.0; 4],
            _padding2: [0.0; 4],
            _padding3: [0.0; 4],
            _padding4: [0.0; 4],
            _padding5: [0.0; 4],
        }
    }
}

/// Complete GPU volumetric ray-marching pipeline.
pub struct VolumetricPipeline {
    pub camera_buf: etalon::Buffer,
    pub observer_buf: etalon::Buffer,
    pub uniforms_layout: etalon::BindGroupLayout,
    pub nodes_layout: etalon::BindGroupLayout,
    pub uniforms_bind_group: etalon::BindGroup,
    pub pipeline: etalon::RenderPipeline,
}

impl VolumetricPipeline {
    pub fn new(
        device: &etalon::Device,
        color_format: etalon::TextureFormat,
        depth_format: Option<etalon::TextureFormat>,
    ) -> Self {
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-holographic-shader"),
            source: etalon::ShaderSource::Wgsl(HOLOGRAPHIC_WGSL.into()),
        });

        let camera_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-holographic-camera"),
            size: core::mem::size_of::<CameraUniform>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let observer_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-holographic-observer"),
            size: core::mem::size_of::<ObserverStandpointUniform>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("etalon-holographic-uniforms-layout"),
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::VERTEX | etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: core::num::NonZeroU64::new(core::mem::size_of::<
                            CameraUniform,
                        >()
                            as u64),
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::VERTEX | etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: core::num::NonZeroU64::new(core::mem::size_of::<
                            ObserverStandpointUniform,
                        >()
                            as u64),
                    },
                    count: None,
                },
            ],
        });

        let nodes_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("etalon-holographic-nodes-layout"),
            entries: &[etalon::BindGroupLayoutEntry {
                binding: 0,
                visibility: etalon::ShaderStages::VERTEX | etalon::ShaderStages::FRAGMENT,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: core::num::NonZeroU64::new(core::mem::size_of::<
                        HolographicNodeGpu,
                    >() as u64),
                },
                count: None,
            }],
        });

        let uniforms_bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("etalon-holographic-uniforms-bind-group"),
            layout: &uniforms_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: observer_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("etalon-holographic-pipeline-layout"),
            bind_group_layouts: &[Some(&uniforms_layout), Some(&nodes_layout)],
            immediate_size: 0,
        });

        let depth_stencil = depth_format.map(|format| etalon::DepthStencilState {
            format,
            depth_write_enabled: Some(false),
            depth_compare: Some(etalon::CompareFunction::LessEqual),
            stencil: etalon::StencilState::default(),
            bias: etalon::DepthBiasState::default(),
        });

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("etalon-holographic-render-pipeline"),
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
                cull_mode: Some(etalon::Face::Back),
                unclipped_depth: false,
                polygon_mode: etalon::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            camera_buf,
            observer_buf,
            uniforms_layout,
            nodes_layout,
            uniforms_bind_group,
            pipeline,
        }
    }

    pub fn update_camera(&self, queue: &etalon::Queue, camera: &CameraUniform) {
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(camera));
    }

    pub fn update_observer(&self, queue: &etalon::Queue, observer: &ObserverStandpointUniform) {
        queue.write_buffer(&self.observer_buf, 0, bytemuck::bytes_of(observer));
    }

    /// Draw instanced bounding cuboids for `node_count` entities.
    pub fn render_nodes<'a>(
        &'a self,
        rpass: &mut etalon::RenderPass<'a>,
        nodes_bind_group: &'a etalon::BindGroup,
        node_count: u32,
    ) {
        if node_count == 0 {
            return;
        }
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.uniforms_bind_group, &[]);
        rpass.set_bind_group(1, nodes_bind_group, &[]);
        // 36 indices = 12 triangles for unit cuboid proxy
        rpass.draw(0..36, 0..node_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_depth_monotonicity_and_inversion_across_13_orders_of_magnitude() {
        let z_far = 1.0e9_f32;
        let c_scale = 1.0_f32;
        let samples = [
            1.0e-4_f32, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 1.0e3, 1.0e6, 1.0e9,
        ];
        let mut prev_log = -1.0_f32;
        for &z in &samples {
            let z_log = compute_log_depth(z, z_far, c_scale);
            assert!(
                z_log > prev_log,
                "log depth must be strictly monotonic at z={z}: {z_log} <= {prev_log}"
            );
            prev_log = z_log;

            let recovered = invert_log_depth(z_log, z_far, c_scale);
            let rel_err = ((recovered - z) / z).abs();
            assert!(
                rel_err < 2.0e-3,
                "log depth inversion drift at z={z}: recovered={recovered}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn adm_geodesic_bends_ray_toward_schwarzschild_mass() {
        // Ray starts at (-10, 2, 0) traveling in +X past a mass at (0, 0, 0).
        let (pos, dir) = adm_geodesic_deflect_ray(
            [-10.0, 2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.5,
            1.0,
        );
        assert!(pos[0] > -10.0);
        // Ray at +Y above the mass must bend downward (-Y) toward the mass center.
        assert!(
            dir[1] < 0.0,
            "ADM geodesic must deflect ray toward mass center, got dir={dir:?}"
        );
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        assert!((norm - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn portal_transport_ray_maps_position_and_rotates_direction() {
        // Entrance portal at origin (identity inverse), exit portal translated by (10, 0, 5)
        // and rotated +90 deg around Y (so +Z maps to +X).
        let portal_in_inv = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        // Column-major exit portal: col0=(0,0,-1), col1=(0,1,0), col2=(1,0,0), col3=(10,0,5,1)
        let portal_out = [
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 5.0, 1.0],
        ];
        let (out_pos, out_dir) = portal_transport_ray(
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            &portal_in_inv,
            &portal_out,
        );
        assert!((out_pos[0] - 10.0).abs() < 1.0e-5);
        assert!((out_pos[1] - 2.0).abs() < 1.0e-5);
        assert!((out_pos[2] - 5.0).abs() < 1.0e-5);
        assert!((out_dir[0] - 1.0).abs() < 1.0e-5);
        assert!(out_dir[1].abs() < 1.0e-5);
        assert!(out_dir[2].abs() < 1.0e-5);
    }
}
