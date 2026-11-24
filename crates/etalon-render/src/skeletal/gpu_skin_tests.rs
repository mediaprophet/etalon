//! Unit tests for `ik_motor` and `gpu_skin`.

use super::gpu_skin::*;
use super::ik_motor::{
    dot3, len3, normalize3_or, solve_constrained_fabrik_motor_ik_zero_heap,
    solve_two_bone_motor_ik_zero_heap, sub3, JointConstraint,
};
use super::skin::{skin_vertices_into, SkinWeight};
use super::{assert_zero_alloc, Motor};

#[test]
fn analytical_two_bone_motor_ik_reaches_target_and_aligns_pole_zero_heap() {
    let root = Motor::from_translation(0.0, 2.0, 0.0);
    let mid = Motor::from_translation(0.0, 1.0, 0.0);
    let end = Motor::from_translation(0.0, 0.0, 0.0);

    let target = [0.0f32, 0.8, 0.6];
    let pole = [0.0f32, 1.0, 2.0];

    let mut res = None;
    assert_zero_alloc("two_bone_motor_ik", || {
        res = Some(
            solve_two_bone_motor_ik_zero_heap(
                root,
                mid,
                end,
                1.0,
                1.0,
                target,
                pole,
                0.1,
                std::f32::consts::PI,
            )
            .unwrap(),
        );
    });
    let res = res.unwrap();
    assert!(res.target_error < 1e-4, "target_error={}", res.target_error);

    let (_, mid_t) = res.world_motors[1].to_rotation_translation();
    assert!(mid_t[2] > 0.4, "knee must bend toward +Z pole: {mid_t:?}");

    let composed_mid = res.local_motors[0].mul(&res.local_motors[1]).normalize();
    let composed_end = composed_mid.mul(&res.local_motors[2]).normalize();
    let (_, end_t) = composed_end.to_rotation_translation();
    for axis in 0..3 {
        assert!((end_t[axis] - target[axis] as f64).abs() < 1e-4);
    }
}

#[test]
fn constrained_fabrik_motor_ik_respects_cone_and_hinge_limits_zero_heap() {
    let chain = [
        Motor::from_translation(0.0, 0.0, 0.0),
        Motor::from_translation(0.0, 1.0, 0.0),
        Motor::from_translation(0.0, 2.0, 0.0),
        Motor::from_translation(0.0, 3.0, 0.0),
    ];
    let constraints = [
        JointConstraint::ConeLimit {
            max_half_angle_rad: 0.5,
        },
        JointConstraint::HingeLimit {
            axis: [1.0, 0.0, 0.0],
            min_angle_rad: -0.8,
            max_angle_rad: 0.8,
        },
        JointConstraint::Unconstrained,
        JointConstraint::Unconstrained,
    ];
    let mut world_out = [Motor::identity(); 4];
    let mut local_out = [Motor::identity(); 4];

    assert_zero_alloc("constrained_fabrik_motor_ik", || {
        let err = solve_constrained_fabrik_motor_ik_zero_heap(
            &chain,
            &constraints,
            [0.0, 2.2, 1.2],
            1e-3,
            16,
            &mut world_out,
            &mut local_out,
        )
        .unwrap();
        assert!(err.is_finite());
    });

    let (_, p0) = world_out[0].to_rotation_translation();
    assert!(len3(p0) < 1e-6);

    for i in 0..3 {
        let (_, pa) = world_out[i].to_rotation_translation();
        let (_, pb) = world_out[i + 1].to_rotation_translation();
        let seg_len = len3(sub3(pb, pa));
        assert!((seg_len - 1.0).abs() < 1e-4, "seg {i} len={seg_len}");
    }

    let (_, p1) = world_out[1].to_rotation_translation();
    let dir0 = normalize3_or(p1, [0.0, 1.0, 0.0]);
    let angle0 = dot3(dir0, [0.0, 1.0, 0.0]).clamp(-1.0, 1.0).acos();
    assert!(angle0 <= 0.5 + 1e-3, "cone limit violated: {angle0}");
}

#[test]
fn gpu_shader_skin_matches_cpu_skin_vertices_into_oracle() {
    let inv_bind = [
        Motor::from_translation(0.0, 0.0, 0.0),
        Motor::from_translation(0.0, -1.0, 0.0),
    ];
    let half = std::f64::consts::FRAC_PI_4;
    let world_prev = [
        Motor::from_translation(1.0, 0.0, 0.0),
        Motor::from_translation(1.0, 1.0, 0.0),
    ];
    let world_curr = [
        Motor::from_translation(2.0, 0.0, 0.0),
        Motor::from_rotation_translation([half.cos(), 0.0, 0.0, half.sin()], [2.0, 1.0, 0.0]),
    ];

    let mut prev_skin = [Motor::identity(); 2];
    let mut curr_skin = [Motor::identity(); 2];
    let mut gpu_palette = [GpuJointMotorPair::identity(); 2];

    evaluate_skinning_motor_palette_zero_heap(
        &inv_bind,
        &world_prev,
        &[Motor::identity(); 2],
        &mut gpu_palette,
        &mut prev_skin,
    );
    let n_joints = evaluate_skinning_motor_palette_zero_heap(
        &inv_bind,
        &world_curr,
        &prev_skin,
        &mut gpu_palette,
        &mut curr_skin,
    );
    assert_eq!(n_joints, 2);

    let bind_pos = [[0.0f32, 2.0, 0.0], [1.0, 1.0, 0.0]];
    let bind_nrm = [[0.0f32, 1.0, 0.0], [1.0, 0.0, 0.0]];
    let bind_tan = [[1.0f32, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, -1.0]];
    let cpu_weights = [
        SkinWeight {
            joints: [1, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        },
        SkinWeight {
            joints: [0, 1, 0, 0],
            weights: [0.5, 0.5, 0.0, 0.0],
        },
    ];
    let gpu_attrs = [
        GpuSkinnedVertex {
            joint_indices: [1, 0, 0, 0],
            joint_weights: [1.0, 0.0, 0.0, 0.0],
        },
        GpuSkinnedVertex {
            joint_indices: [0, 1, 0, 0],
            joint_weights: [0.5, 0.5, 0.0, 0.0],
        },
    ];

    let mut cpu_pos = [[0.0f32; 3]; 2];
    let mut cpu_nrm = [[0.0f32; 3]; 2];
    skin_vertices_into(
        &bind_pos,
        &bind_nrm,
        &cpu_weights,
        &curr_skin,
        &mut cpu_pos,
        &mut cpu_nrm,
    )
    .unwrap();

    let (gpu_p0, gpu_n0, gpu_t0, mv0) = gpu_shader_skin_vertex_reference(
        bind_pos[0],
        bind_nrm[0],
        bind_tan[0],
        &gpu_attrs[0],
        &gpu_palette,
    );
    for axis in 0..3 {
        assert!(
            (gpu_p0[axis] - cpu_pos[0][axis]).abs() < 1e-5,
            "pos mismatch axis {axis}: gpu={gpu_p0:?} cpu={:?}",
            cpu_pos[0]
        );
        assert!(
            (gpu_n0[axis] - cpu_nrm[0][axis]).abs() < 1e-5,
            "nrm mismatch axis {axis}: gpu={gpu_n0:?} cpu={:?}",
            cpu_nrm[0]
        );
    }
    assert_eq!(gpu_t0[3], 1.0);
    assert!(mv0.iter().map(|x| x * x).sum::<f32>() > 0.1);
}

#[test]
fn antipodal_hemisphere_alignment_prevents_dual_quaternion_collapse() {
    let m_pos = Motor::from_translation(0.0, 2.0, 0.0);
    let m_neg = Motor {
        r_w: -m_pos.r_w,
        r_x: -m_pos.r_x,
        r_y: -m_pos.r_y,
        r_z: -m_pos.r_z,
        d_w: -m_pos.d_w,
        d_x: -m_pos.d_x,
        d_y: -m_pos.d_y,
        d_z: -m_pos.d_z,
    };
    let aligned = hemisphere_align_motor(&m_pos, &m_neg);
    assert!((aligned.r_w - m_pos.r_w).abs() < 1e-6);

    let palette = [
        GpuJointMotorPair::from_motors(&m_pos, &m_pos),
        GpuJointMotorPair {
            current_r: [
                m_neg.r_w as f32,
                m_neg.r_x as f32,
                m_neg.r_y as f32,
                m_neg.r_z as f32,
            ],
            current_d: [
                m_neg.d_w as f32,
                m_neg.d_x as f32,
                m_neg.d_y as f32,
                m_neg.d_z as f32,
            ],
            prev_r: [
                m_neg.r_w as f32,
                m_neg.r_x as f32,
                m_neg.r_y as f32,
                m_neg.r_z as f32,
            ],
            prev_d: [
                m_neg.d_w as f32,
                m_neg.d_x as f32,
                m_neg.d_y as f32,
                m_neg.d_z as f32,
            ],
        },
    ];
    let attr = GpuSkinnedVertex {
        joint_indices: [0, 1, 0, 0],
        joint_weights: [0.5, 0.5, 0.0, 0.0],
    };
    let (p, n, _, mv) = gpu_shader_skin_vertex_reference(
        [1.0, 3.0, -2.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        &attr,
        &palette,
    );
    assert!((p[0] - 1.0).abs() < 1e-5);
    assert!((p[1] - 5.0).abs() < 1e-5);
    assert!((p[2] - -2.0).abs() < 1e-5);
    assert!((n[1] - 1.0).abs() < 1e-5);
    assert!(mv.iter().all(|v| v.abs() < 1e-6));
}

#[test]
fn gpu_skinning_palette_and_vertex_reference_are_zero_heap() {
    let inv_bind = [Motor::identity(); 4];
    let world = [Motor::from_translation(1.0, 2.0, 3.0); 4];
    let prev = [Motor::identity(); 4];
    let mut gpu_palette = [GpuJointMotorPair::identity(); 4];
    let mut curr_skin = [Motor::identity(); 4];
    let attr = GpuSkinnedVertex {
        joint_indices: [0, 1, 2, 3],
        joint_weights: [0.25, 0.25, 0.25, 0.25],
    };

    assert_zero_alloc("gpu_skin_palette_and_vertex", || {
        let n = evaluate_skinning_motor_palette_zero_heap(
            &inv_bind,
            &world,
            &prev,
            &mut gpu_palette,
            &mut curr_skin,
        );
        assert_eq!(n, 4);
        let (p, _, _, _) = gpu_shader_skin_vertex_reference(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            &attr,
            &gpu_palette,
        );
        assert!((p[0] - 1.0).abs() < 1e-5);
    });
}
