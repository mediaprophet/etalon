//! Two-Phase Meshlet Cluster DAG Frustum/Cone/Occlusion Culling & Monotonic Cut Selection (§7.3.1).
//!
//! Provides the certified WGSL compute kernel [`CLUSTER_DAG_CULL_AND_LOD_WGSL`], the
//! zero-heap CPU reference oracle [`cluster_dag_cull_and_lod_cpu`], and GPU execution
//! harness [`cluster_dag_cull_and_lod_gpu`].
//!
//! Flat cluster layout (`CLUSTER_NODE_STRIDE = 12` `f32` scalars per cluster node):
//! - `[0..4]`: bounding sphere `[center_x, center_y, center_z, radius]`
//! - `[4..8]`: normal cone `[cone_axis_x, cone_axis_y, cone_axis_z, cone_cutoff]`
//! - `[8..12]`: `[err_self, err_parent, min_depth, hzb_occluder_depth]`

use crate::numerical::ForgeError;

/// WGSL compute kernel source for Two-Phase Meshlet Cluster DAG Culling & Monotonic Cut Selection.
pub const CLUSTER_DAG_CULL_AND_LOD_WGSL: &str = r#"// Two-Phase Meshlet Cluster DAG Culling & Monotonic Cut Selection.
//
// Each cluster node occupies 12 contiguous f32 scalars in `clusters`:
//   [0..4]  sphere_center.xyz, sphere_radius
//   [4..8]  cone_axis.xyz, cone_cutoff
//   [8..12] err_self, err_parent, min_depth, hzb_occluder_depth
//
// `params` (28 f32 scalars):
//   [0..24]  6 frustum planes (nx, ny, nz, d) with inward-pointing normals
//   [24..27] camera_eye (ex, ey, ez)
//   [27]     tau_px (monotonic LOD pixel error threshold)
//
// `selected_out[i]` receives 1.0 if cluster `i` passes frustum, normal cone,
// occlusion, and lies on the active monotonic LOD cut (`err_self <= tau_px < err_parent`),
// else 0.0.

@group(0) @binding(0) var<storage, read> clusters: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> selected_out: array<f32>;

@compute @workgroup_size(64)
fn cluster_dag_cull_and_lod(@builtin(global_invocation_id) gid: vec3<u32>) {
    let stride = 12u;
    let count = arrayLength(&clusters) / stride;
    let i = gid.x;
    if (i >= count) {
        return;
    }

    let base = i * stride;
    let center = vec3<f32>(clusters[base + 0u], clusters[base + 1u], clusters[base + 2u]);
    let radius = clusters[base + 3u];
    let cone_axis = vec3<f32>(clusters[base + 4u], clusters[base + 5u], clusters[base + 6u]);
    let cone_cutoff = clusters[base + 7u];
    let err_self = clusters[base + 8u];
    let err_parent = clusters[base + 9u];
    let min_depth = clusters[base + 10u];
    let hzb_occluder_depth = clusters[base + 11u];

    let camera_eye = vec3<f32>(params[24u], params[25u], params[26u]);
    let tau_px = params[27u];

    // 1. Monotonic LOD cut condition: err_self <= tau_px < err_parent
    if (!(err_self <= tau_px && tau_px < err_parent)) {
        selected_out[i] = 0.0;
        return;
    }

    // 2. Bounding sphere 6-plane frustum test
    for (var p: u32 = 0u; p < 6u; p = p + 1u) {
        let pb = p * 4u;
        let normal = vec3<f32>(params[pb + 0u], params[pb + 1u], params[pb + 2u]);
        let d = params[pb + 3u];
        if (dot(normal, center) + d < -radius) {
            selected_out[i] = 0.0;
            return;
        }
    }

    // 3. Normal cone backface test: dot(view_dir, cone_axis) < cone_cutoff -> culled
    let to_eye = camera_eye - center;
    let dist = max(length(to_eye), 1e-6);
    let view_dir = to_eye / dist;
    if (dot(view_dir, cone_axis) < cone_cutoff) {
        selected_out[i] = 0.0;
        return;
    }

    // 4. Two-phase HZB occlusion test
    if (min_depth > hzb_occluder_depth) {
        selected_out[i] = 0.0;
        return;
    }

    selected_out[i] = 1.0;
}
"#;

/// Entry-point name of [`CLUSTER_DAG_CULL_AND_LOD_WGSL`].
pub const CLUSTER_DAG_CULL_AND_LOD_ENTRY: &str = "cluster_dag_cull_and_lod";
/// Scalars per cluster node in the flat input buffer.
pub const CLUSTER_NODE_STRIDE: usize = 12;

/// Descriptor for a single cluster node in the meshlet LOD DAG.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterDagNode {
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
    pub err_self: f32,
    pub err_parent: f32,
    pub min_depth: f32,
    pub hzb_occluder_depth: f32,
}

impl ClusterDagNode {
    /// Serializes this node into 12 contiguous `f32` scalars.
    #[inline]
    pub const fn to_f32_array(&self) -> [f32; CLUSTER_NODE_STRIDE] {
        [
            self.center[0],
            self.center[1],
            self.center[2],
            self.radius,
            self.cone_axis[0],
            self.cone_axis[1],
            self.cone_axis[2],
            self.cone_cutoff,
            self.err_self,
            self.err_parent,
            self.min_depth,
            self.hzb_occluder_depth,
        ]
    }
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Evaluates whether a single [`ClusterDagNode`] is visible and lies on the active monotonic cut.
#[inline]
pub fn evaluate_cluster_visibility_and_cut(
    node: &ClusterDagNode,
    frustum_planes: &[[f32; 4]; 6],
    camera_eye: [f32; 3],
    tau_px: f32,
) -> bool {
    // 1. Monotonic LOD cut condition: err_self <= tau_px < err_parent
    if !(node.err_self <= tau_px && tau_px < node.err_parent) {
        return false;
    }

    // 2. Bounding sphere 6-plane frustum test
    for plane in frustum_planes {
        let dist = plane[0] * node.center[0]
            + plane[1] * node.center[1]
            + plane[2] * node.center[2]
            + plane[3];
        if dist < -node.radius {
            return false;
        }
    }

    // 3. Normal cone backface test: dot(view_dir, cone_axis) < cone_cutoff
    let to_eye = [
        camera_eye[0] - node.center[0],
        camera_eye[1] - node.center[1],
        camera_eye[2] - node.center[2],
    ];
    let dist = dot3(to_eye, to_eye).sqrt().max(1.0e-6);
    let view_dir = [to_eye[0] / dist, to_eye[1] / dist, to_eye[2] / dist];
    if dot3(view_dir, node.cone_axis) < node.cone_cutoff {
        return false;
    }

    // 4. Two-phase HZB occlusion test
    if node.min_depth > node.hzb_occluder_depth {
        return false;
    }

    true
}

/// Zero-heap CPU reference oracle for Two-Phase Meshlet Cluster DAG Culling & Monotonic Cut Selection.
pub fn cluster_dag_cull_and_lod_cpu(
    clusters: &[f32],
    frustum_planes: &[[f32; 4]; 6],
    camera_eye: [f32; 3],
    tau_px: f32,
    out_selected: &mut [f32],
) {
    let count = (clusters.len() / CLUSTER_NODE_STRIDE).min(out_selected.len());
    for i in 0..count {
        let b = i * CLUSTER_NODE_STRIDE;
        let node = ClusterDagNode {
            center: [clusters[b], clusters[b + 1], clusters[b + 2]],
            radius: clusters[b + 3],
            cone_axis: [clusters[b + 4], clusters[b + 5], clusters[b + 6]],
            cone_cutoff: clusters[b + 7],
            err_self: clusters[b + 8],
            err_parent: clusters[b + 9],
            min_depth: clusters[b + 10],
            hzb_occluder_depth: clusters[b + 11],
        };
        out_selected[i] = if evaluate_cluster_visibility_and_cut(
            &node,
            frustum_planes,
            camera_eye,
            tau_px,
        ) {
            1.0
        } else {
            0.0
        };
    }
}

/// Dispatches Two-Phase Meshlet Cluster DAG Culling & Monotonic Cut Selection on the GPU.
pub fn cluster_dag_cull_and_lod_gpu(
    clusters: &[f32],
    frustum_planes: &[[f32; 4]; 6],
    camera_eye: [f32; 3],
    tau_px: f32,
) -> Result<Vec<f32>, ForgeError> {
    use crate::numerical::execute::{
        BindingUsage, QualiaCompute, WgpuComputeContext, WgpuPipeline,
    };
    use crate::numerical::Schedule;

    if clusters.is_empty() || clusters.len() % CLUSTER_NODE_STRIDE != 0 {
        return Err(ForgeError::GpuValidation(format!(
            "cluster_dag_cull_and_lod_gpu: length {} is not a non-zero multiple of {CLUSTER_NODE_STRIDE}",
            clusters.len()
        )));
    }
    let count = clusters.len() / CLUSTER_NODE_STRIDE;
    let capacity = (clusters.len() * 8).max(4 << 20);
    let mut ctx = WgpuComputeContext::new(capacity)?;

    let view_clusters = ctx.allocate_and_write(
        bytemuck::cast_slice(clusters),
        0,
        0,
        BindingUsage::StorageRead,
    )?;

    let mut params = [0.0f32; 28];
    for (p, plane) in frustum_planes.iter().enumerate() {
        params[p * 4] = plane[0];
        params[p * 4 + 1] = plane[1];
        params[p * 4 + 2] = plane[2];
        params[p * 4 + 3] = plane[3];
    }
    params[24] = camera_eye[0];
    params[25] = camera_eye[1];
    params[26] = camera_eye[2];
    params[27] = tau_px;

    let view_params = ctx.allocate_and_write(
        bytemuck::cast_slice(&params),
        1,
        0,
        BindingUsage::StorageRead,
    )?;

    let zeros = vec![0.0f32; count];
    let view_out = ctx.allocate_and_write(
        bytemuck::cast_slice(&zeros),
        2,
        0,
        BindingUsage::StorageReadWrite,
    )?;

    let buffers = vec![view_clusters, view_params, view_out];
    let pipeline = WgpuPipeline::compile(
        &ctx,
        CLUSTER_DAG_CULL_AND_LOD_WGSL,
        CLUSTER_DAG_CULL_AND_LOD_ENTRY,
    )?;
    let schedule = Schedule {
        workgroup_size: 64,
        ..Default::default()
    };
    pipeline.dispatch(&buffers, &schedule, count)?;
    let mut out = ctx.read_buffer_f32(&view_out)?;
    out.truncate(count);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wide_open_frustum() -> [[f32; 4]; 6] {
        [
            [1.0, 0.0, 0.0, 1000.0],
            [-1.0, 0.0, 0.0, 1000.0],
            [0.0, 1.0, 0.0, 1000.0],
            [0.0, -1.0, 0.0, 1000.0],
            [0.0, 0.0, 1.0, 1000.0],
            [0.0, 0.0, -1.0, 1000.0],
        ]
    }

    #[test]
    fn cluster_dag_cull_and_lod_wgsl_validates() {
        let report = crate::validate_wgsl(CLUSTER_DAG_CULL_AND_LOD_WGSL)
            .expect("cluster DAG cull and LOD WGSL must naga-validate");
        assert!(
            report
                .entry_points
                .iter()
                .any(|e| e == CLUSTER_DAG_CULL_AND_LOD_ENTRY),
            "expected entry point {CLUSTER_DAG_CULL_AND_LOD_ENTRY}, got {:?}",
            report.entry_points
        );
        assert_eq!(report.binding_count, 3);
    }

    #[test]
    fn monotonic_cut_selection_completeness_no_cracks_or_overlaps() {
        // 4-level LOD chain from root (LOD 3: error [8.0, inf)) down to leaf (LOD 0: error [0.0, 1.0)).
        // For every threshold tau_px >= 0, exactly one level in the chain must be selected:
        // neither 0 (crack) nor >=2 (overlap).
        let chain = [
            // LOD 0 (finest leaf): err_self = 0.0, err_parent = 1.0
            ClusterDagNode {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
                cone_axis: [0.0, 0.0, 1.0],
                cone_cutoff: -0.5,
                err_self: 0.0,
                err_parent: 1.0,
                min_depth: 0.2,
                hzb_occluder_depth: 0.9,
            },
            // LOD 1: err_self = 1.0, err_parent = 3.0
            ClusterDagNode {
                center: [0.0, 0.0, 0.0],
                radius: 2.0,
                cone_axis: [0.0, 0.0, 1.0],
                cone_cutoff: -0.5,
                err_self: 1.0,
                err_parent: 3.0,
                min_depth: 0.2,
                hzb_occluder_depth: 0.9,
            },
            // LOD 2: err_self = 3.0, err_parent = 8.0
            ClusterDagNode {
                center: [0.0, 0.0, 0.0],
                radius: 4.0,
                cone_axis: [0.0, 0.0, 1.0],
                cone_cutoff: -0.5,
                err_self: 3.0,
                err_parent: 8.0,
                min_depth: 0.2,
                hzb_occluder_depth: 0.9,
            },
            // LOD 3 (coarsest root): err_self = 8.0, err_parent = 1e30
            ClusterDagNode {
                center: [0.0, 0.0, 0.0],
                radius: 8.0,
                cone_axis: [0.0, 0.0, 1.0],
                cone_cutoff: -0.5,
                err_self: 8.0,
                err_parent: 1.0e30,
                min_depth: 0.2,
                hzb_occluder_depth: 0.9,
            },
        ];

        let mut flat = [0.0f32; 4 * CLUSTER_NODE_STRIDE];
        for (i, node) in chain.iter().enumerate() {
            flat[i * CLUSTER_NODE_STRIDE..(i + 1) * CLUSTER_NODE_STRIDE]
                .copy_from_slice(&node.to_f32_array());
        }

        let frustum = wide_open_frustum();
        let eye = [0.0, 0.0, 10.0];
        for &tau_px in &[0.0f32, 0.5, 1.0, 2.2, 3.0, 5.5, 8.0, 25.0] {
            let mut selected = [0.0f32; 4];
            cluster_dag_cull_and_lod_cpu(&flat, &frustum, eye, tau_px, &mut selected);
            let active_count: usize = selected.iter().filter(|&&v| v == 1.0).count();
            assert_eq!(
                active_count, 1,
                "cut at tau_px={tau_px} must select exactly 1 level (no cracks/overlaps), got {selected:?}"
            );
        }
    }

    #[test]
    fn frustum_backface_cone_and_hzb_occlusion_culling() {
        let frustum = wide_open_frustum();
        let eye = [0.0, 0.0, 10.0];

        // Backfacing cluster: cone_axis points -Z (away from camera at +Z), cone_cutoff = 0.0
        let backfacing = ClusterDagNode {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            cone_axis: [0.0, 0.0, -1.0],
            cone_cutoff: 0.0,
            err_self: 0.5,
            err_parent: 2.0,
            min_depth: 0.2,
            hzb_occluder_depth: 0.9,
        };
        assert!(!evaluate_cluster_visibility_and_cut(&backfacing, &frustum, eye, 1.0));

        // Outside frustum: x = -2000 outside plane x >= -1000
        let outside_frustum = ClusterDagNode {
            center: [-2000.0, 0.0, 0.0],
            radius: 1.0,
            cone_axis: [0.0, 0.0, 1.0],
            cone_cutoff: -1.0,
            err_self: 0.5,
            err_parent: 2.0,
            min_depth: 0.2,
            hzb_occluder_depth: 0.9,
        };
        assert!(!evaluate_cluster_visibility_and_cut(&outside_frustum, &frustum, eye, 1.0));

        // HZB occluded: min_depth (0.85) > hzb_occluder_depth (0.40)
        let occluded = ClusterDagNode {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            cone_axis: [0.0, 0.0, 1.0],
            cone_cutoff: -1.0,
            err_self: 0.5,
            err_parent: 2.0,
            min_depth: 0.85,
            hzb_occluder_depth: 0.40,
        };
        assert!(!evaluate_cluster_visibility_and_cut(&occluded, &frustum, eye, 1.0));
    }
}
