use super::{
    compact_visible_packed_records, compact_visible_records, extract_clip_planes,
    provably_outside_clip, select_visible_instances, select_visible_instances_packed,
    select_visible_mesh_instances, BoundedInstance, GpuInstanceRecord, VisibilityListError,
};
use bytemuck::Zeroable;

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn perspective(near: f32, far: f32) -> [[f32; 4]; 4] {
    let focal = 1.0 / (45.0_f32.to_radians() * 0.5).tan();
    [
        [focal, 0.0, 0.0, 0.0],
        [0.0, focal, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

fn instance(min: [f32; 3], max: [f32; 3]) -> BoundedInstance {
    BoundedInstance {
        bounds_min: min,
        bounds_max: max,
        world_from_local: IDENTITY,
    }
}

#[test]
fn selection_is_stable_and_keeps_intersecting_bounds() {
    let instances = [
        instance([-0.5; 3], [0.5; 3]),
        instance([2.0, -0.2, 0.2], [3.0, 0.2, 0.8]),
        instance([0.9, -0.1, 0.2], [1.1, 0.1, 0.8]),
    ];
    let mut visible = [u32::MAX; 3];
    let count = select_visible_instances(&instances, IDENTITY, &mut visible).unwrap();
    assert_eq!(&visible[..count], &[0, 2]);
}

#[test]
fn mesh_instance_selection_composes_actor_and_instance_transforms() {
    let translate = |x: f32| {
        let mut matrix = IDENTITY;
        matrix[3][0] = x;
        matrix
    };
    let records = [
        GpuInstanceRecord::new(IDENTITY, 0x10),
        GpuInstanceRecord::new(translate(2.0), 0x20),
    ];
    let mut visible = [u32::MAX; 2];
    let count = select_visible_mesh_instances(
        [-0.2; 3],
        [0.2; 3],
        translate(-2.0),
        IDENTITY,
        &records,
        &mut visible,
    )
    .unwrap();
    assert_eq!(&visible[..count], &[1]);
    assert_eq!(records[visible[0] as usize].semantic_id(), 0x20);
}

#[test]
fn mesh_instance_selection_preflights_capacity_and_fails_open_on_invalid_matrix() {
    let record = GpuInstanceRecord::new(IDENTITY, 7);
    let mut short = [0xfeed_u32; 0];
    assert_eq!(
        select_visible_mesh_instances(
            [-0.5; 3],
            [0.5; 3],
            IDENTITY,
            IDENTITY,
            &[record],
            &mut short,
        ),
        Err(VisibilityListError::OutputTooSmall {
            required: 1,
            available: 0,
        })
    );

    let mut malformed = IDENTITY;
    malformed[0][0] = f32::NAN;
    let mut visible = [u32::MAX; 1];
    assert_eq!(
        select_visible_mesh_instances(
            [-0.5; 3],
            [0.5; 3],
            malformed,
            IDENTITY,
            &[record],
            &mut visible,
        ),
        Ok(1)
    );
    assert_eq!(visible, [0]);
}

#[test]
fn packed_selection_matches_struct_api_and_keeps_output_atomic_on_bad_shape() {
    let bounds_min = [-0.5, -0.5, -0.5, 2.0, -0.2, 0.2, 0.9, -0.1, 0.2];
    let bounds_max = [0.5, 0.5, 0.5, 3.0, 0.2, 0.8, 1.1, 0.1, 0.8];
    let mut transforms = [0.0; 48];
    for instance in 0..3 {
        for axis in 0..4 {
            transforms[instance * 16 + axis * 4 + axis] = 1.0;
        }
    }
    let view_projection: [f32; 16] = std::array::from_fn(|index| IDENTITY[index / 4][index % 4]);
    let mut visible = [u32::MAX; 3];
    assert_eq!(
        select_visible_instances_packed(
            &bounds_min,
            &bounds_max,
            &transforms,
            &view_projection,
            &mut visible,
        ),
        Ok(2)
    );
    assert_eq!(&visible[..2], &[0, 2]);
    let mut unchanged = [17_u32; 3];
    assert_eq!(
        select_visible_instances_packed(
            &bounds_min[..8],
            &bounds_max,
            &transforms,
            &view_projection,
            &mut unchanged,
        ),
        Err(VisibilityListError::PackedInputShapeInvalid)
    );
    assert_eq!(unchanged, [17; 3]);
}

#[test]
fn rejects_each_fully_outside_webgpu_clip_plane() {
    let cases = [
        instance([-3.0, -0.1, 0.2], [-2.0, 0.1, 0.8]),
        instance([2.0, -0.1, 0.2], [3.0, 0.1, 0.8]),
        instance([-0.1, -3.0, 0.2], [0.1, -2.0, 0.8]),
        instance([-0.1, 2.0, 0.2], [0.1, 3.0, 0.8]),
        instance([-0.1, -0.1, -2.0], [0.1, 0.1, -1.0]),
        instance([-0.1, -0.1, 2.0], [0.1, 0.1, 3.0]),
    ];
    let mut visible = [u32::MAX; 6];
    assert_eq!(
        select_visible_instances(&cases, IDENTITY, &mut visible),
        Ok(0)
    );
}

#[test]
fn transformed_bounds_are_tested_in_world_space() {
    let mut translated = instance([-0.25; 3], [0.25; 3]);
    translated.world_from_local[3][0] = 4.0;
    let mut visible = [u32::MAX; 1];
    assert_eq!(
        select_visible_instances(&[translated], IDENTITY, &mut visible),
        Ok(0)
    );
}

#[test]
fn perspective_projection_checks_depth_and_fails_open_for_camera_crossing_bounds() {
    let instances = [
        instance([-0.1, -0.1, -2.1], [0.1, 0.1, -1.9]),
        instance([-0.1, -0.1, -0.08], [0.1, 0.1, -0.04]),
        instance([-0.1, -0.1, -120.0], [0.1, 0.1, -110.0]),
        instance([-0.1, -0.1, 1.0], [0.1, 0.1, 2.0]),
        instance([-0.2, -0.2, -1.0], [0.2, 0.2, 1.0]),
    ];
    let mut visible = [u32::MAX; 5];
    let count =
        select_visible_instances(&instances, perspective(0.1, 100.0), &mut visible).unwrap();
    assert_eq!(&visible[..count], &[0, 4]);
}

#[test]
fn ten_thousand_repeated_instances_fit_the_caller_buffer_without_reordering() {
    let repeated = instance([-0.1, -0.1, 0.2], [0.1, 0.1, 0.8]);
    let instances = vec![repeated; 10_000];
    let mut visible = vec![u32::MAX; instances.len()];
    let count = select_visible_instances(&instances, IDENTITY, &mut visible).unwrap();
    assert_eq!(count, 10_000);
    assert_eq!(visible[0], 0);
    assert_eq!(visible[9_999], 9_999);
}

#[test]
fn support_radius_matches_corner_oracle_for_deterministic_affine_cases() {
    let view_projection = perspective(0.1, 80.0);
    let mut seed = 0x91E1_0DA5_u32;
    let mut instances = Vec::with_capacity(512);
    for _ in 0..512 {
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let min = [next() * 4.0, next() * 4.0, -next().abs() * 60.0 - 0.2];
        let extent = [next().abs() * 1.5, next().abs() * 1.5, next().abs() * 1.5];
        let angle = next() * std::f32::consts::PI;
        let (sine, cosine) = angle.sin_cos();
        let translation = [next() * 8.0, next() * 8.0, next() * 60.0 - 30.0];
        let mut transform = IDENTITY;
        transform[0] = [cosine, sine, 0.0, 0.0];
        transform[1] = [-sine, cosine, 0.0, 0.0];
        transform[3] = [translation[0], translation[1], translation[2], 1.0];
        instances.push(BoundedInstance {
            bounds_min: min,
            bounds_max: [min[0] + extent[0], min[1] + extent[1], min[2] + extent[2]],
            world_from_local: transform,
        });
    }
    let planes = extract_clip_planes(view_projection);
    for instance in instances {
        assert_eq!(
            provably_outside_clip(&instance, planes),
            corner_oracle_outside(&instance, planes),
            "instance {instance:?}"
        );
    }
}

#[test]
fn compact_upload_stream_preserves_stable_semantic_ids_and_order() {
    let source = [
        GpuInstanceRecord::new(IDENTITY, 0x0000_0001_0000_0002),
        GpuInstanceRecord::new(IDENTITY, 0xAABB_CCDD_EEFF_0011),
        GpuInstanceRecord::new(IDENTITY, 0xFEDC_BA98_7654_3210),
    ];
    assert_eq!(std::mem::size_of::<GpuInstanceRecord>(), 128);
    assert_eq!(
        std::mem::offset_of!(GpuInstanceRecord, semantic_id_words),
        112
    );
    assert_eq!(
        std::mem::offset_of!(GpuInstanceRecord, orientation_sign),
        120
    );
    let indices = [2, 0];
    let mut compacted = [GpuInstanceRecord::zeroed(); 2];
    assert_eq!(
        compact_visible_records(&source, &indices, &mut compacted),
        Ok(2)
    );
    assert_eq!(compacted[0].semantic_id(), source[2].semantic_id());
    assert_eq!(compacted[1].semantic_id(), source[0].semantic_id());
}

#[test]
fn packed_compaction_is_alignment_independent_and_preserves_identity() {
    let source = [
        GpuInstanceRecord::new(IDENTITY, 11),
        GpuInstanceRecord::new(IDENTITY, 22),
        GpuInstanceRecord::new(IDENTITY, 33),
    ];
    let source_bytes = bytemuck::cast_slice(&source);
    let mut output = [0_u8; 256];
    assert_eq!(
        compact_visible_packed_records(source_bytes, &[2, 0], &mut output),
        Ok(2)
    );
    let first = bytemuck::pod_read_unaligned::<GpuInstanceRecord>(&output[..128]);
    let second = bytemuck::pod_read_unaligned::<GpuInstanceRecord>(&output[128..]);
    assert_eq!([first.semantic_id(), second.semantic_id()], [33, 11]);
}

#[test]
fn compact_stream_errors_are_all_or_nothing() {
    let source = [GpuInstanceRecord::new(IDENTITY, 7)];
    let sentinel = GpuInstanceRecord::new(IDENTITY, 99);
    let mut output = [sentinel; 2];
    assert_eq!(
        compact_visible_records(&source, &[0, 0], &mut output[..1]),
        Err(VisibilityListError::CompactedOutputTooSmall {
            required: 2,
            available: 1
        })
    );
    assert_eq!(output, [sentinel; 2]);
    assert_eq!(
        compact_visible_records(&source, &[0, 4], &mut output),
        Err(VisibilityListError::SourceIndexOutOfRange {
            visible_offset: 1,
            source_index: 4,
            source_count: 1
        })
    );
    assert_eq!(output, [sentinel; 2]);
}

fn corner_oracle_outside(instance: &BoundedInstance, planes: [[f32; 4]; 6]) -> bool {
    if instance
        .bounds_min
        .iter()
        .chain(instance.bounds_max.iter())
        .chain(instance.world_from_local.iter().flatten())
        .chain(planes.iter().flatten())
        .any(|value| !value.is_finite())
        || (0..3).any(|axis| instance.bounds_min[axis] > instance.bounds_max[axis])
    {
        return false;
    }
    for plane in planes {
        let mut all_outside = true;
        for corner in 0..8 {
            let point = [
                if corner & 1 == 0 {
                    instance.bounds_min[0]
                } else {
                    instance.bounds_max[0]
                },
                if corner & 2 == 0 {
                    instance.bounds_min[1]
                } else {
                    instance.bounds_max[1]
                },
                if corner & 4 == 0 {
                    instance.bounds_min[2]
                } else {
                    instance.bounds_max[2]
                },
                1.0,
            ];
            let world = std::array::from_fn::<_, 4, _>(|row| {
                (0..4)
                    .map(|column| instance.world_from_local[column][row] * point[column])
                    .sum::<f32>()
            });
            let distance = plane.iter().zip(world).map(|(a, b)| a * b).sum::<f32>();
            let epsilon = 1.0e-6 * (distance.abs() + 1.0);
            all_outside &= distance < -epsilon;
        }
        if all_outside {
            return true;
        }
    }
    false
}

#[test]
fn invalid_or_inverted_bounds_fail_open() {
    let cases = [
        instance([f32::NAN, 0.0, 0.0], [f32::NAN, 0.0, 0.0]),
        instance([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
    ];
    let mut visible = [u32::MAX; 2];
    assert_eq!(
        select_visible_instances(&cases, IDENTITY, &mut visible),
        Ok(2)
    );
    assert_eq!(visible, [0, 1]);
}

#[test]
fn insufficient_output_capacity_fails_before_writing() {
    let cases = [instance([-0.5; 3], [0.5; 3]), instance([2.0; 3], [3.0; 3])];
    let mut visible = [77; 1];
    assert_eq!(
        select_visible_instances(&cases, IDENTITY, &mut visible),
        Err(VisibilityListError::OutputTooSmall {
            required: 2,
            available: 1
        })
    );
    assert_eq!(visible, [77]);
}
