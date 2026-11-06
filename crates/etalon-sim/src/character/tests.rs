//! Unit Tests for Kinematic Character Controller.

use super::config::*;
use crate::time::FIXED_DELTA_SECONDS;
use etalon_render::water_surface::SubmersionState;

#[test]
fn test_character_gravity_and_grounding() {
    let config = CharacterConfig::default();
    let mut controller = CharacterController::new(config, [0.0, 2.0, 0.0]);

    // Probe that returns flat ground at y = 0
    let flat_ground = |_x: f32, _z: f32| (0.0, [0.0, 1.0, 0.0]);

    assert!(!controller.state.is_grounded);

    // Fall under gravity for 1 second (64 steps)
    for _ in 0..64 {
        controller.step([0.0, 0.0], false, false, false, FIXED_DELTA_SECONDS as f32, flat_ground);
    }

    // Must have hit ground and grounded state set
    assert!(controller.state.is_grounded);
    assert!((controller.state.position[1] - 0.0).abs() < 1e-4);
    assert_eq!(controller.state.velocity[1], 0.0);
}

#[test]
fn test_character_walk_and_jump() {
    let config = CharacterConfig::default();
    let mut controller = CharacterController::new(config, [0.0, 0.0, 0.0]);
    let flat_ground = |_x: f32, _z: f32| (0.0, [0.0, 1.0, 0.0]);

    // Push forward
    for _ in 0..32 {
        controller.step([1.0, 0.0], false, false, false, FIXED_DELTA_SECONDS as f32, flat_ground);
    }

    assert!(controller.state.velocity[2] > 0.0);
    assert!(controller.state.position[2] > 0.0);

    // Trigger jump
    controller.step([1.0, 0.0], false, true, false, FIXED_DELTA_SECONDS as f32, flat_ground);
    assert!(!controller.state.is_grounded);
    assert!(controller.state.velocity[1] > 0.0);
}

#[test]
fn test_character_step_into_water_transition() {
    let config = CharacterConfig::default();
    let mut controller = CharacterController::new(config, [0.0, 0.0, 0.0]);
    let flat_ground = |_x: f32, _z: f32| (0.0, [0.0, 1.0, 0.0]);

    // Water surface at y = 0.35m (wading depth of 0.35m)
    let water_surface = |_x: f32, _z: f32| 0.35;

    let trans = controller.step_with_water(
        [1.0, 0.0],
        false,
        false,
        false,
        FIXED_DELTA_SECONDS as f32,
        flat_ground,
        water_surface,
    );

    assert_eq!(trans, Some(SubmersionState::Wading));
    assert_eq!(controller.state.submersion, SubmersionState::Wading);
    assert!(controller.state.wetness > 0.0);
    assert!((controller.state.water_depth - 0.35).abs() < 1e-4);
}

#[test]
fn test_character_3d_capsule_sweep_and_platform_inheritance() {
    use crate::physics3d::{BroadphaseBvh, ColliderShape3D};
    use etalon_render::spatial_math::pga::motor_translate;

    let config = CharacterConfig::default();
    let mut controller = CharacterController::new(config, [0.0, 0.5, 0.0]);

    let shapes = [ColliderShape3D::Cuboid {
        half_extents: [10.0, 0.5, 10.0],
    }];
    let motors = [motor_translate([0.0, -0.5, 0.0])];
    let platform_vels = [[2.0, 0.0, 0.0]];

    let mut bvh = BroadphaseBvh::<16>::new(0.1);
    bvh.insert(0, shapes[0].world_aabb(&motors[0])).unwrap();

    for _ in 0..32 {
        controller.step_character_3d(
            [0.0, 0.0],
            false,
            false,
            false,
            FIXED_DELTA_SECONDS as f32,
            &bvh,
            &shapes,
            &motors,
            Some(&platform_vels),
        );
    }

    assert!(controller.state.is_grounded);
    assert!((controller.state.position[1] - 0.0).abs() < 0.05);
    assert!(controller.state.position[0] > 0.2);
}
