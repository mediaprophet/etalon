use super::*;
/// Returns `(total_texels_written, mip_slices, mip_count)`.
pub fn build_hzb_mip_chain_zero_heap(
    mip0: &[f32],
    base_width: u32,
    base_height: u32,
    out_texels: &mut [f32],
) -> (usize, [(usize, u32, u32); 8], usize) {
    let mut slices = [(0usize, 0u32, 0u32); 8];
    let mip0_len = (base_width as usize).saturating_mul(base_height as usize);
    if mip0_len == 0 || mip0.len() < mip0_len || out_texels.len() < mip0_len {
        return (0, slices, 0);
    }

    out_texels[..mip0_len].copy_from_slice(&mip0[..mip0_len]);
    slices[0] = (0, base_width, base_height);
    let mut cursor = mip0_len;
    let mut mip_count = 1usize;
    let mut prev_w = base_width;
    let mut prev_h = base_height;
    let mut prev_offset = 0usize;

    while mip_count < 8 && (prev_w > 1 || prev_h > 1) {
        let next_w = (prev_w / 2).max(1);
        let next_h = (prev_h / 2).max(1);
        let next_len = (next_w as usize) * (next_h as usize);
        if cursor + next_len > out_texels.len() {
            break;
        }

        for y in 0..next_h {
            for x in 0..next_w {
                let sx0 = (x * 2).min(prev_w - 1);
                let sy0 = (y * 2).min(prev_h - 1);
                let sx1 = (x * 2 + 1).min(prev_w - 1);
                let sy1 = (y * 2 + 1).min(prev_h - 1);

                let d00 = out_texels[prev_offset + (sy0 * prev_w + sx0) as usize];
                let d10 = out_texels[prev_offset + (sy0 * prev_w + sx1) as usize];
                let d01 = out_texels[prev_offset + (sy1 * prev_w + sx0) as usize];
                let d11 = out_texels[prev_offset + (sy1 * prev_w + sx1) as usize];

                out_texels[cursor + (y * next_w + x) as usize] = d00.max(d10).max(d01).max(d11);
            }
        }

        slices[mip_count] = (cursor, next_w, next_h);
        prev_offset = cursor;
        cursor += next_len;
        prev_w = next_w;
        prev_h = next_h;
        mip_count += 1;
    }

    (cursor, slices, mip_count)
}

/// Selects the 4-tier LOD index (`0..=3`) from an epistemic-weighted screen-space projected radius
/// in pixels, applying a fractional hysteresis deadband around threshold transitions.
#[inline]
pub fn select_lod_tier_with_hysteresis(
    proj_radius_px: f32,
    prev_lod: u8,
    thresholds_px: [f32; 3],
    hysteresis: f32,
) -> u8 {
    let t0 = thresholds_px[0];
    let t1 = thresholds_px[1];
    let t2 = thresholds_px[2];
    let h = hysteresis.clamp(0.0, 0.45);

    let raw_tier = if proj_radius_px >= t0 {
        0u8
    } else if proj_radius_px >= t1 {
        1u8
    } else if proj_radius_px >= t2 {
        2u8
    } else {
        3u8
    };

    if prev_lod == 0 && raw_tier == 1 && proj_radius_px >= t0 * (1.0 - h) {
        return 0;
    }
    if prev_lod == 1 && raw_tier == 0 && proj_radius_px < t0 * (1.0 + h) {
        return 1;
    }
    if prev_lod == 1 && raw_tier == 2 && proj_radius_px >= t1 * (1.0 - h) {
        return 1;
    }
    if prev_lod == 2 && raw_tier == 1 && proj_radius_px < t1 * (1.0 + h) {
        return 2;
    }
    if prev_lod == 2 && raw_tier == 3 && proj_radius_px >= t2 * (1.0 - h) {
        return 2;
    }
    if prev_lod == 3 && raw_tier == 2 && proj_radius_px < t2 * (1.0 + h) {
        return 3;
    }
    raw_tier
}

#[inline]
fn sphere_passes_frustum_planes(center: [f32; 3], radius: f32, planes: &[FrustumPlane; 6]) -> bool {
    let mut i = 0usize;
    while i < 6 {
        let p = &planes[i];
        let dist =
            p.normal[0] * center[0] + p.normal[1] * center[1] + p.normal[2] * center[2] + p.d;
        if dist < -radius {
            return false;
        }
        i += 1;
    }
    true
}

#[inline]
fn mul_mat4_vec4_col_major(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
        m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3],
    ]
}

#[inline]
fn sphere_passes_hzb_cpu(
    world_center: [f32; 3],
    radius: f32,
    camera_pos: [f32; 3],
    focal_px: f32,
    view_proj: &[[f32; 4]; 4],
    hzb: &HzbPyramidView<'_>,
) -> bool {
    let dx = camera_pos[0] - world_center[0];
    let dy = camera_pos[1] - world_center[1];
    let dz = camera_pos[2] - world_center[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    if dist <= radius + 0.25 {
        return true;
    }

    let clip = mul_mat4_vec4_col_major(
        view_proj,
        [world_center[0], world_center[1], world_center[2], 1.0],
    );
    if clip[3] <= 1e-4 {
        return true;
    }
    let ndc_x = clip[0] / clip[3];
    let ndc_y = clip[1] / clip[3];
    let uv_x = ndc_x * 0.5 + 0.5;
    let uv_y = 0.5 - ndc_y * 0.5;
    if !(0.0..=1.0).contains(&uv_x) || !(0.0..=1.0).contains(&uv_y) {
        return true;
    }

    let proj_radius_px = (radius * focal_px) / dist.max(0.1);
    let footprint_px = (proj_radius_px * 2.0).max(1.0);
    let mip_level =
        (footprint_px.log2().ceil().max(0.0) as usize).min(hzb.mip_count.saturating_sub(1));
    let hzb_farthest = hzb.sample_farthest_depth([uv_x, uv_y], mip_level);

    let inv_dist = 1.0 / dist;
    let nearest_world = [
        world_center[0] + dx * inv_dist * radius,
        world_center[1] + dy * inv_dist * radius,
        world_center[2] + dz * inv_dist * radius,
        1.0,
    ];
    let nearest_clip = mul_mat4_vec4_col_major(view_proj, nearest_world);
    let nearest_depth = (nearest_clip[2] / nearest_clip[3].max(1e-5)).clamp(0.0, 1.0);

    nearest_depth <= (hzb_farthest + 1e-4)
}

/// Zero-heap CPU/GPU parity execution of 6-plane frustum culling, 2-phase HZB depth-pyramid
/// occlusion culling, 4-tier hybrid LOD selection with hysteresis, and multi-modal stream routing.
///
/// Must make **zero heap allocations** (`HotZeroHeap`).
#[allow(clippy::too_many_arguments)]
pub fn cull_and_select_lod_indirect_zero_heap(
    instances: &[InstanceCullDescriptor],
    assets: &[ResidentAssetEntry],
    view_proj: &[[f32; 4]; 4],
    camera_pos: [f32; 3],
    focal_length_px: f32,
    hzb_pyramid: Option<&HzbPyramidView<'_>>,
    out_draw_commands: &mut [DrawIndexedIndirectCommand],
    out_visible_mesh_indices: &mut [u32],
    out_visible_holo_indices: &mut [u32],
    out_visible_surfel_indices: &mut [u32],
    out_selected_lods: Option<&mut [u8]>,
) -> IndirectCullStreamCounts {
    let planes = extract_frustum_planes(view_proj);
    let mut counts = IndirectCullStreamCounts::default();
    let mut selected_lods = out_selected_lods;

    for (idx, inst) in instances.iter().enumerate() {
        let world_center = inst.world_center();
        let world_radius = inst.world_radius();

        // 1. 6-plane frustum culling
        if !sphere_passes_frustum_planes(world_center, world_radius, &planes) {
            continue;
        }

        // 2. Screen-space projected radius modulated by epistemic salience & spectral uncertainty
        let dx = world_center[0] - camera_pos[0];
        let dy = world_center[1] - camera_pos[1];
        let dz = world_center[2] - camera_pos[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
        let raw_proj_px = (world_radius * focal_length_px) / dist;

        let alpha = inst.alpha_salience.clamp(0.05, 4.0);
        let sigma = inst.sigma_spectral.max(0.0);
        let epistemic_weight = alpha.sqrt() / (1.0 + 0.35 * sigma).max(0.5);
        let effective_proj_px = raw_proj_px * epistemic_weight;

        if effective_proj_px < DEFAULT_MIN_CULL_RADIUS_PX {
            continue;
        }

        // 3. 2-Phase HZB depth-pyramid occlusion culling
        if let Some(hzb) = hzb_pyramid {
            if !sphere_passes_hzb_cpu(
                world_center,
                world_radius,
                camera_pos,
                focal_length_px,
                view_proj,
                hzb,
            ) {
                counts.occluded_count += 1;
                continue;
            }
        }

        // 4. 4-Tier LOD selection with 10% hysteresis
        let prev_lod = (inst.prev_lod_tier.max(0.0) as u8).min(3);
        let lod_tier = select_lod_tier_with_hysteresis(
            effective_proj_px,
            prev_lod,
            DEFAULT_LOD_THRESHOLDS_PX,
            DEFAULT_LOD_HYSTERESIS,
        );
        if let Some(ref mut lods) = selected_lods {
            if idx < lods.len() {
                lods[idx] = lod_tier;
            }
        }

        let policy = RepresentationPolicy::from_u16(inst.rep_policy);
        let tensor_q = inst.tensor_q();

        // 5. Route to Holographic SDF Proxy Stream if q > 0 or policy == HolographicSdfProxy
        if tensor_q > 0 || policy == RepresentationPolicy::HolographicSdfProxy {
            let slot = counts.holo_proxy_count as usize;
            if slot < out_visible_holo_indices.len() {
                out_visible_holo_indices[slot] = idx as u32;
                counts.holo_proxy_count += 1;
            }
            continue;
        }

        // 6. Route to Surfel / Octahedral Impostor Stream if LOD3 (unless ExplicitMeshLod) or policy == Surfel/Impostor
        if (lod_tier >= 3 && policy != RepresentationPolicy::ExplicitMeshLod)
            || policy == RepresentationPolicy::GaussianSurfelCloud
            || policy == RepresentationPolicy::OctahedralImpostor
        {
            let slot = counts.surfel_impostor_count as usize;
            if slot < out_visible_surfel_indices.len() {
                out_visible_surfel_indices[slot] = idx as u32;
                counts.surfel_impostor_count += 1;
            }
            continue;
        }

        // 7. Route to Mesh Indirect Stream (LOD0..LOD2)
        let asset_idx = inst.asset_id as usize;
        if asset_idx >= assets.len() {
            continue;
        }
        let asset = &assets[asset_idx];
        let mut slice: SubmeshSlice = asset.lod_slices[(lod_tier as usize).min(3)];
        if slice.index_count == 0 {
            slice = asset.lod_slices[0];
        }
        if slice.index_count == 0 {
            continue;
        }

        let cmd_slot = counts.mesh_draw_count as usize;
        if cmd_slot < out_draw_commands.len() && cmd_slot < out_visible_mesh_indices.len() {
            out_visible_mesh_indices[cmd_slot] = idx as u32;
            out_draw_commands[cmd_slot] = DrawIndexedIndirectCommand {
                index_count: slice.index_count,
                instance_count: 1,
                first_index: slice.first_index,
                base_vertex: slice.base_vertex,
                first_instance: idx as u32,
            };
            counts.mesh_draw_count += 1;
        }
    }

    counts
}
