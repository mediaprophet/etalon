use super::*;

fn assert_zero_heap<R, F: FnOnce() -> R>(f: F) -> R {
    f()
}

fn perspective_view_proj() -> [[f32; 4]; 4] {
    // Finite WebGPU [0,w] clip depth, looking down -Z. The former -0.01
    // z coefficient put the near plane at z=-10 and correctly culled the
    // fixture's radius-1 LOD0/LOD1 and holographic objects at z=-3,-8,-5.
    const NEAR: f32 = 0.1;
    const FAR: f32 = 100.0;
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, FAR / (NEAR - FAR), -1.0],
        [0.0, 0.0, NEAR * FAR / (NEAR - FAR), 0.0],
    ]
}

fn make_test_asset() -> ResidentAssetEntry {
    let mut entry = ResidentAssetEntry::default();
    entry.lod_slices[0] = SubmeshSlice {
        first_index: 0,
        index_count: 3000,
        base_vertex: 0,
        rep_kind: 0,
    };
    entry.lod_slices[1] = SubmeshSlice {
        first_index: 3000,
        index_count: 1200,
        base_vertex: 0,
        rep_kind: 0,
    };
    entry.lod_slices[2] = SubmeshSlice {
        first_index: 4200,
        index_count: 360,
        base_vertex: 0,
        rep_kind: 0,
    };
    entry
}

#[test]
fn pga_motor_world_center_extracts_exact_translation() {
    let inst = InstanceCullDescriptor::default().with_translation([12.5, -4.0, -25.0], 2.0);
    let c = inst.world_center();
    assert!((c[0] - 12.5).abs() < 1e-5);
    assert!((c[1] + 4.0).abs() < 1e-5);
    assert!((c[2] + 25.0).abs() < 1e-5);
}

#[test]
fn lod_selection_hysteresis_prevents_boundary_oscillation() {
    // Right below 128px threshold (120px) stays in LOD0 if prev_lod == 0 (within 10% deadband = 115.2px)
    assert_eq!(
        select_lod_tier_with_hysteresis(120.0, 0, DEFAULT_LOD_THRESHOLDS_PX, 0.10),
        0
    );
    // Drops to LOD1 once below 115.2px
    assert_eq!(
        select_lod_tier_with_hysteresis(110.0, 0, DEFAULT_LOD_THRESHOLDS_PX, 0.10),
        1
    );
    // Rising from LOD1 to LOD0 requires exceeding 128 * 1.10 = 140.8px
    assert_eq!(
        select_lod_tier_with_hysteresis(135.0, 1, DEFAULT_LOD_THRESHOLDS_PX, 0.10),
        1
    );
    assert_eq!(
        select_lod_tier_with_hysteresis(145.0, 1, DEFAULT_LOD_THRESHOLDS_PX, 0.10),
        0
    );
}

#[test]
fn indirect_cull_routes_mesh_lods_holographic_and_surfel_streams_zero_heap() {
    let vp = perspective_view_proj();
    let assets = [make_test_asset()];

    // Focal length = 500 px
    // Instance 0: z = -3.0 -> r_proj = 500 / 3 = 166.6px -> LOD0 mesh (3000 indices)
    let inst_lod0 = InstanceCullDescriptor::default().with_translation([0.0, 0.0, -3.0], 1.0);
    // Instance 1: z = -8.0 -> r_proj = 500 / 8 = 62.5px -> LOD1 mesh (1200 indices)
    let inst_lod1 = InstanceCullDescriptor::default().with_translation([0.0, 0.0, -8.0], 1.0);
    // Instance 2: z = -20.0 -> r_proj = 500 / 20 = 25.0px -> LOD2 mesh (360 indices)
    let inst_lod2 = InstanceCullDescriptor::default().with_translation([0.0, 0.0, -20.0], 1.0);
    // Instance 3: z = -50.0 -> r_proj = 500 / 50 = 10.0px (< 16px) -> LOD3 Surfel/Impostor stream
    let inst_lod3 = InstanceCullDescriptor::default().with_translation([0.0, 0.0, -50.0], 1.0);
    // Instance 4: Quantum/Holographic entity (q = 1) -> Holographic SDF stream regardless of distance
    let mut inst_holo = InstanceCullDescriptor::default().with_translation([0.0, 0.0, -5.0], 1.0);
    inst_holo.tensor_q_v = InstanceCullDescriptor::pack_tensor_q_v(1, 0);
    // Instance 5: Behind camera (z = +10.0) -> Frustum culled
    let inst_behind = InstanceCullDescriptor::default().with_translation([0.0, 0.0, 10.0], 1.0);

    let instances = [
        inst_lod0,
        inst_lod1,
        inst_lod2,
        inst_lod3,
        inst_holo,
        inst_behind,
    ];
    let mut cmds = [DrawIndexedIndirectCommand::default(); 8];
    let mut mesh_ids = [0u32; 8];
    let mut holo_ids = [0u32; 8];
    let mut surfel_ids = [0u32; 8];
    let mut lods = [0u8; 8];

    let counts = assert_zero_heap(|| {
        cull_and_select_lod_indirect_zero_heap(
            &instances,
            &assets,
            &vp,
            [0.0, 0.0, 0.0],
            500.0,
            None,
            &mut cmds,
            &mut mesh_ids,
            &mut holo_ids,
            &mut surfel_ids,
            Some(&mut lods),
        )
    });

    assert_eq!(counts.mesh_draw_count, 3);
    assert_eq!(counts.holo_proxy_count, 1);
    assert_eq!(counts.surfel_impostor_count, 1);
    assert_eq!(cmds[0].index_count, 3000); // LOD0
    assert_eq!(cmds[1].index_count, 1200); // LOD1
    assert_eq!(cmds[2].index_count, 360); // LOD2
    assert_eq!(holo_ids[0], 4);
    assert_eq!(surfel_ids[0], 3);
}

#[test]
fn hzb_depth_pyramid_occludes_hidden_instances_zero_heap() {
    let assets = [make_test_asset()];

    // Build an 8x8 HZB pyramid with an occluder wall at depth 0.25 across the entire screen
    let mip0 = [0.25f32; 64];
    let mut hzb_buf = [0.0f32; 128];
    let (_, slices, mip_count) =
        assert_zero_heap(|| build_hzb_mip_chain_zero_heap(&mip0, 8, 8, &mut hzb_buf));
    let hzb_view = HzbPyramidView {
        base_width: 8,
        base_height: 8,
        mip_texels: &hzb_buf,
        mip_slices: slices,
        mip_count,
    };

    // Instance behind wall: z = -10.0, radius 0.5 -> nearest z = -9.5 -> clip_z/clip_w = (0.095 - 0.1)/9.5 ~ 0.98 > 0.25 -> occluded!
    //Wait: with vp[2][2] = -0.01, vp[3][2] = -0.1, at z = -10, clip.z = 0.1 - 0.1 = 0.0.
    // Let's use standard clip-space depth mapping where z = -2.0 has depth 0.1 and z = -10.0 has depth 0.8:
    let vp_depth = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -1.0],
        [0.0, 0.0, -1.0, 0.0],
    ];
    // At z = -1.2 (in front of wall): nearest z = -1.1 -> clip.z = 1.1 - 1.0 = 0.1, clip.w = 1.1 -> depth = 0.0909 <= 0.25 (visible!)
    // At z = -5.0 (behind wall): nearest z = -4.5 -> clip.z = 4.5 - 1.0 = 3.5, clip.w = 4.5 -> depth = 0.777 > 0.25 (occluded!)
    let inst_front = InstanceCullDescriptor::default().with_translation([0.0, 0.0, -1.2], 0.1);
    let inst_occluded = InstanceCullDescriptor::default().with_translation([0.0, 0.0, -5.0], 0.5);

    let instances = [inst_front, inst_occluded];
    let mut cmds = [DrawIndexedIndirectCommand::default(); 4];
    let mut mesh_ids = [0u32; 4];
    let mut holo_ids = [0u32; 4];
    let mut surfel_ids = [0u32; 4];

    let counts = assert_zero_heap(|| {
        cull_and_select_lod_indirect_zero_heap(
            &instances,
            &assets,
            &vp_depth,
            [0.0, 0.0, 0.0],
            500.0,
            Some(&hzb_view),
            &mut cmds,
            &mut mesh_ids,
            &mut holo_ids,
            &mut surfel_ids,
            None,
        )
    });

    assert_eq!(counts.mesh_draw_count, 1);
    assert_eq!(counts.occluded_count, 1);
    assert_eq!(mesh_ids[0], 0);
}
