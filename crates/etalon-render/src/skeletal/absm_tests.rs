//! Unit tests for the multi-layer Animation Blending State Machine (`absm`).

use super::absm::*;
use super::{assert_zero_alloc, Motor};

#[test]
fn bone_layer_mask_from_subtree_and_multi_layer_absm_zero_heap() {
    let parents = [-1, 0, 1];
    let upper_mask = BoneLayerMask::from_joint_subtree(&parents, 1);
    assert!(!upper_mask.contains(0));
    assert!(upper_mask.contains(1));
    assert!(upper_mask.contains(2));

    let rest = [Motor::identity(); 3];
    let clip_idle = [
        Motor::from_translation(0.0, 0.0, 0.0),
        Motor::identity(),
        Motor::identity(),
    ];
    let clip_walk = [
        Motor::from_translation(0.0, 0.0, 2.0),
        Motor::identity(),
        Motor::identity(),
    ];
    let clip_wave = [
        Motor::identity(),
        Motor::identity(),
        Motor::from_translation(0.0, 1.0, 0.0),
    ];
    let clips: [&[Motor]; 3] = [&clip_idle, &clip_walk, &clip_wave];

    let mut machine = AbsmMachine::<4, 8, 8>::new();
    machine.root_motion = Some(RootMotionExtractor::new(
        0,
        Motor::identity(),
        Motor::from_translation(0.0, 0.0, 2.0),
    ));
    machine.eont_posture = EontPostureConfig {
        enabled: true,
        spine_mask: BoneLayerMask(1u128 << 1),
        stage: 3, // Senescent
        energy_permill: 400,
        damage_permill: 300,
        hydration_permill: 700,
        max_droop_rad: 0.3,
    };

    let mut base_layer = AbsmLayer::new(BoneLayerMask::ALL, LayerBlendMode::Override, 1.0);
    let s_idle = base_layer
        .add_state(AbsmStateNode {
            source: AbsmStateSource::SingleClip { clip_index: 0 },
            duration: 1.0,
            speed: 1.0,
            looping: true,
            ..Default::default()
        })
        .unwrap();
    let mut walk_node = AbsmStateNode {
        source: AbsmStateSource::SingleClip { clip_index: 1 },
        duration: 1.0,
        speed: 1.0,
        looping: true,
        ..Default::default()
    };
    walk_node.triggers[0] = SignalTrigger {
        active: true,
        signal_id: 42,
        normalized_time: 0.5,
        sonic_archetype: 2,
        payload_f32: 0.8,
    };
    let s_walk = base_layer.add_state(walk_node).unwrap();
    base_layer.add_transition(
        s_idle,
        s_walk,
        0.0,
        TransitionCondition::ParamGt {
            param_idx: 0,
            threshold: 0.5,
        },
    );
    machine.add_layer(base_layer).unwrap();

    let mut upper_layer = AbsmLayer::new(upper_mask, LayerBlendMode::Additive, 1.0);
    upper_layer
        .add_state(AbsmStateNode {
            source: AbsmStateSource::SingleClip { clip_index: 2 },
            duration: 1.0,
            speed: 1.0,
            looping: true,
            ..Default::default()
        })
        .unwrap();
    machine.add_layer(upper_layer).unwrap();

    machine.set_parameter(0, AbsmParameter::Float(1.0));

    let mut out_pose = [Motor::identity(); 3];
    let mut signals = SignalQueue::<16>::new();

    assert_zero_alloc("absm_step_and_evaluate", || {
        let rm1 = machine
            .step_and_evaluate_zero_heap(0.1, &rest, &clips, &mut out_pose, &mut signals)
            .unwrap();
        assert!(rm1.delta_translation[2] >= 0.0);
        let (_, root_t) = out_pose[0].to_rotation_translation();
        assert!(root_t[2].abs() < 1e-5);

        let _ = machine
            .step_and_evaluate_zero_heap(1.1, &rest, &clips, &mut out_pose, &mut signals)
            .unwrap();
    });

    let sig = signals.pop().expect("footstep signal must fire");
    assert_eq!(sig.signal_id, 42);
    assert_eq!(sig.sonic_archetype, 2);

    let (_, arm_t) = out_pose[2].to_rotation_translation();
    assert!((arm_t[1] - 1.0).abs() < 1e-5);
    let (spine_r, _) = out_pose[1].to_rotation_translation();
    assert!(spine_r[1].abs() > 0.01);
}
