use super::{
    aabb_outside_shadow_clip, camera_csm_4_sun_view_projections, camera_range_sun_view_projections,
    compute_csm_4_splits_zero_heap, csm_cascade_blend_factor, stabilized_sun_view_projection,
    ShadowAtlasAllocator, CSM_4_CASCADE_COUNT, SHADOW_CASCADE_COUNT, SHADOW_RESOLUTION_CANDIDATES,
};

fn assert_zero_heap<R, F: FnOnce() -> R>(f: F) -> R {
    f()
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

#[test]
fn cascade_aabb_culling_is_conservative_at_clip_edges() {
    assert!(!aabb_outside_shadow_clip(
        [-0.5, -0.5, 0.25],
        [0.5, 0.5, 0.75],
        IDENTITY,
        IDENTITY,
    ));
    assert!(aabb_outside_shadow_clip(
        [1.1, -0.25, 0.25],
        [2.0, 0.25, 0.75],
        IDENTITY,
        IDENTITY,
    ));
    assert!(aabb_outside_shadow_clip(
        [-0.25, -0.25, 1.1],
        [0.25, 0.25, 2.0],
        IDENTITY,
        IDENTITY,
    ));
    assert!(!aabb_outside_shadow_clip(
        [-1.5, -0.25, 0.25],
        [0.25, 0.25, 0.75],
        IDENTITY,
        IDENTITY,
    ));
    assert!(!aabb_outside_shadow_clip(
        [f32::NAN, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        IDENTITY,
        IDENTITY,
    ));
    let mut translated_model = IDENTITY;
    translated_model[3][0] = 4.0;
    assert!(aabb_outside_shadow_clip(
        [-0.25, -0.25, 0.25],
        [0.25, 0.25, 0.75],
        IDENTITY,
        translated_model,
    ));
}

#[test]
fn shadow_resolution_fallback_is_monotonic_and_bounded() {
    assert_eq!(SHADOW_RESOLUTION_CANDIDATES, [1024, 512, 256]);
    assert!(SHADOW_RESOLUTION_CANDIDATES
        .windows(2)
        .all(|pair| pair[0] > pair[1]));
}

#[test]
fn light_projection_is_finite_for_degenerate_direction_and_extent() {
    let matrix = stabilized_sun_view_projection([0.0; 3], [0.0; 3], [0.0; 3], 1024);
    assert!(matrix.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn light_facing_surface_has_nearer_zero_to_one_depth() {
    let matrix = stabilized_sun_view_projection([0.0, 0.0, 1.0], [0.0; 3], [2.0; 3], 1024);
    let depth = |z: f32| matrix[2][2] * z + matrix[3][2];
    assert!(
        depth(0.5) < depth(-0.5),
        "surface closer to the directional light must win Less depth: near={}, far={}",
        depth(0.5),
        depth(-0.5)
    );
    assert!((0.0..1.0).contains(&depth(0.5)));
    assert!((0.0..1.0).contains(&depth(-0.5)));
}

#[test]
fn shadow_projection_is_stable_within_a_texel_cell() {
    let extent = [2.0, 2.0, 2.0];
    let a = stabilized_sun_view_projection([0.2, 0.9, 0.3], [0.0; 3], extent, 1024);
    let b = stabilized_sun_view_projection([0.2, 0.9, 0.3], [0.0, 0.0, 0.0001], extent, 1024);
    for column in 0..4 {
        assert_eq!(a[column][0], b[column][0]);
        assert_eq!(a[column][1], b[column][1]);
    }
}

#[test]
fn camera_range_cascades_are_finite_and_cover_distinct_ranges() {
    let matrices = camera_range_sun_view_projections(
        [0.2, 0.9, 0.3],
        [0.0, 0.0, 3.5],
        [0.0, 0.0, -1.0],
        [0.0; 3],
        [220.0; 3],
        16.0 / 9.0,
        24.0,
        200.0,
        1024,
    );
    assert_eq!(matrices.len(), SHADOW_CASCADE_COUNT);
    assert!(matrices.iter().flatten().flatten().all(|v| v.is_finite()));
    assert_ne!(matrices[0], matrices[1]);
    let distant_draw = ([-0.5, -0.5, -100.5], [0.5, 0.5, -99.5]);
    assert!(aabb_outside_shadow_clip(
        distant_draw.0,
        distant_draw.1,
        matrices[0],
        IDENTITY,
    ));
    assert!(!aabb_outside_shadow_clip(
        distant_draw.0,
        distant_draw.1,
        matrices[1],
        IDENTITY,
    ));
}

#[test]
fn camera_range_cascades_survive_degenerate_camera_inputs() {
    let matrices = camera_range_sun_view_projections(
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        0.0,
        f32::NAN,
        f32::INFINITY,
        0,
    );
    assert!(matrices.iter().flatten().flatten().all(|v| v.is_finite()));
}

#[test]
fn csm_4_splits_and_cascade_blend_are_monotonic_and_zero_heap() {
    let splits = assert_zero_heap(|| compute_csm_4_splits_zero_heap(0.1, 200.0, 0.75));
    assert_eq!(splits.len(), CSM_4_CASCADE_COUNT);
    assert!(splits[0] > 0.1 && splits[0] < splits[1]);
    assert!(splits[1] < splits[2] && splits[2] < splits[3]);
    assert!((splits[3] - 200.0).abs() < 1e-3);

    // Inside cascade 0 well before the overlap band -> blend == 0
    let (c0, c1, blend) = csm_cascade_blend_factor(splits[0] * 0.5, 0.1, &splits, 0.15);
    assert_eq!((c0, c1), (0, 0));
    assert_eq!(blend, 0.0);

    // Near the end of cascade 0 -> smoothly blends into cascade 1
    let (c0_b, c1_b, blend_b) = csm_cascade_blend_factor(splits[0] * 0.96, 0.1, &splits, 0.15);
    assert_eq!((c0_b, c1_b), (0, 1));
    assert!(blend_b > 0.0 && blend_b <= 1.0);

    let (matrices, out_splits) = assert_zero_heap(|| {
        camera_csm_4_sun_view_projections(
            [0.3, 0.85, 0.4],
            [0.0, 2.0, 5.0],
            [0.0, 0.0, -1.0],
            [0.0; 3],
            [250.0; 3],
            16.0 / 9.0,
            0.1,
            200.0,
            0.75,
            1024,
        )
    });
    assert_eq!(out_splits, splits);
    assert!(matrices.iter().flatten().flatten().all(|v| v.is_finite()));
}

#[test]
fn shadow_atlas_allocator_tiles_non_overlapping_uv_rects_zero_heap() {
    let mut alloc = ShadowAtlasAllocator::<16>::new(2048, 4);
    let s0 = assert_zero_heap(|| alloc.allocate_slot(IDENTITY, 0, 0, 0.001)).unwrap();
    let s1 = assert_zero_heap(|| alloc.allocate_slot(IDENTITY, 1, 0, 0.001)).unwrap();
    assert_eq!(s0.atlas_uv_rect, [0.0, 0.0, 0.25, 0.25]);
    assert_eq!(s1.atlas_uv_rect, [0.25, 0.0, 0.25, 0.25]);
}
