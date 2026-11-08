//! Etalon Deterministic Simulation & Continuum World Engine (`etalon-sim`).
//!
//! Unifies:
//! - `time`: 64 Hz `FixedTimestepAccumulator`, 20 Hz lockstep `FixedTickWorld` (`WorldState`),
//!   `DeterministicPrng`, and `MultiRateSimClock`
//! - `ecs`: Zero-heap SOA `SimTable`, `Tick`, and `Added`/`Changed` reactive queries
//! - `input`: 256-bit `ZeroHeapButtonInput` with single-frame edge transitions and `MultiTouchTracker`
//! - `navigation`: 2D grid octile A* + boids crowd separation (`grid2d`) and 2.5D convex polygon
//!   `ZeroHeapNavMesh` funnel string-pulling (`navmesh`)
//! - `physics3d`: 64 Hz XPBD rigid-body solver, GJK/EPA + analytical SDF colliders, broadphase BVH,
//!   PGA motor joints, CSG stress cleavage fracture, active/passive ragdoll, continuum bridges,
//!   and 4D RK4 driven spring/thermal ODE responses
//! - `character` & `kinematics`: 64 Hz kinematic capsule character controller (ground, water, 3D sweep)
//!   and 4-wheel Ackermann suspension `RoverKinematics`
//! - `environment`: Atmospheric boundary layer & aeroelasticity (`atmosphere`), heliophysical space
//!   weather (`space_weather`), Rothermel/Byram fire ecology (`fire_ecology`), Mohr-Coulomb/McKyes-Ali
//!   soil geomechanics & excavation (`geomechanics`), and `evaluate_heliophysical_fire_weather` coupling

pub mod character;
pub mod continuum_3d;
pub mod ecs;
pub mod environment;
pub mod input;
pub mod kinematics;
pub mod navigation;
pub mod physics3d;
pub mod time;

// Module-level aliases preserving 100% path compatibility with `qualia-sim`
pub use environment::{atmosphere, coupling, excavation, fire_ecology, geomechanics, space_weather};
pub use navigation::navmesh;

// Re-export rendering-coupled simulation modules consolidated in `etalon-render`
pub mod animation {
    pub use etalon_render::skeletal::inertial_skeleton::*;
}
pub mod camera {
    pub use etalon_render::spatial_math::camera::*;
}
pub mod differential_codec {
    pub use etalon_render::frame_quality::disocclusion::*;
}
pub mod flora_archetype {
    pub use etalon_render::vegetation::flora_archetype::*;
}
pub mod ground_cover {
    pub use etalon_render::vegetation::ground_cover::*;
}
pub mod hydro {
    pub use etalon_render::water_surface::*;
}
pub mod leaf_biophysics {
    pub use etalon_render::vegetation::leaf_biophysics::*;
}
pub mod metamer_fspm {
    pub use etalon_render::vegetation::metamer_fspm::*;
}
pub mod optics {
    pub use etalon_render::water_surface::optics::*;
}
pub mod ripple {
    pub use etalon_render::water_surface::ripple::*;
}

pub use animation::{
    AnimationStateMachine, Transform3D, TransitionMode, ZeroHeapSkeleton, MAX_BONES,
};
pub use atmosphere::{
    roughness, AeroelasticBranch, BarometricState, BoundaryLayerProfile, LeafFlutterState,
    GAS_CONSTANT_DRY_AIR, GAS_CONSTANT_WATER_VAPOR, STANDARD_MSLP_HPA, VON_KARMAN_KAPPA,
};
pub use camera::{unproject_screen_to_ray, FlyCamera, OrbitCamera, Ray3D, SpringChaseCamera};
pub use character::{step_character_3d, CharacterConfig, CharacterController, CharacterState};
pub use continuum_3d::*;
pub use coupling::{evaluate_heliophysical_fire_weather, HeliophysicalFireWeatherReport};
pub use differential_codec::{
    DisocclusionForecaster, EntitySpatialBounds, KinematicExtrapolator,
    ProgressiveMonteCarloGovernor, SaliencePerceptualGovernor, MAX_PREDICTED_DISOCCLUSIONS,
};
pub use ecs::{Added, Changed, SimTable, SimTableError, Tick, INVALID_ROW};
pub use fire_ecology::{
    evaluate_dead_fine_fuel_moisture, evaluate_fireline_intensity, FirelineRegime, FirelineVerdict,
    ForestFuelStratum, FuelMoistureState, DEAD_FUEL_EXTINCTION_MOISTURE, EUCALYPT_HEAT_YIELD_KJ_KG,
};
pub use flora_archetype::{
    FloraArchetypeConfig, FloraBiomechanicsEvaluator, RaunkiaerLifeForm, StemSupportRegime,
};
pub use geomechanics::{
    critical_trench_depth_m, evaluate_excavation_pass, evaluate_soil_compaction,
    ExcavationVerdict, SoilMechanicsProfile, ToolEffector, ToolKind,
};
pub use ground_cover::{
    spatial_hash_2d, EcotopePatch, GroundCoverLod, ToroidalGroundCoverClipmap, TOROIDAL_GRID_DIM,
    TOROIDAL_TOTAL_CELLS,
};
pub use hydro::{
    ArchimedesProxy, BuoyancyResult, BuoyancySphere, GerstnerWave, HydrologicalHysteresisGate,
    SimdGerstnerSpectrum, SubmersionState, GRAVITY_M_S2, MAX_BUOYANCY_SPHERES,
    MAX_GERSTNER_HARMONICS, WATER_DENSITY_KG_M3,
};
pub use input::{
    MultiTouchTracker, TouchContact, VirtualTouchStick, ZeroHeapButtonInput, BUTTON_MIDDLE,
    BUTTON_PRIMARY, BUTTON_SECONDARY, INPUT_WORDS_4, KEY_A, KEY_ARROW_DOWN, KEY_ARROW_LEFT,
    KEY_ARROW_RIGHT, KEY_ARROW_UP, KEY_D, KEY_E, KEY_ESCAPE, KEY_Q, KEY_S, KEY_SHIFT, KEY_SPACE,
    KEY_TAB, KEY_W, MAX_TOUCH_POINTS,
};
pub use kinematics::{RoverKinematics, VehicleChassisConfig, VehicleState, WheelConfig};
pub use leaf_biophysics::{
    LeafEconomicsProfile, LeafVascularSkeleton, VeinEdge, MAX_VEIN_EDGES, NO_PARENT_VEIN,
};
pub use metamer_fspm::{
    Metamer, MetamerOrgan, MetamerPlantGraph, ThermalPhenology, MAX_METAMERS, NO_PARENT_METAMER,
};
pub use navigation::{
    compute_boids_separation, project_grid_path_to_waypoints, NavPolygon, NavigationGrid,
    PathResult, Point2D, ZeroHeapNavMesh, MAX_NAVMESH_NODES, MAX_PATH_LENGTH, MAX_PATH_WAYPOINTS,
    NAV_GRID_MAX_CELLS, NAV_GRID_MAX_DIM,
};
pub use optics::{
    BioGeoOpticalConstituents, DielectricVolumeIop, PURE_WATER_ABSORPTION_RGB,
    PURE_WATER_RAYLEIGH_RGB, WAVELENGTH_BLUE_NM, WAVELENGTH_GREEN_NM, WAVELENGTH_RED_NM,
};
pub use physics3d::{
    cast_capsule, cast_ray, cast_sphere, compute_narrowphase_contact,
    couple_rigid_body_with_botanical_stem, couple_rigid_body_with_hydro,
    couple_rigid_body_with_soil, evaluate_and_fracture_body, gjk_intersect, overlap_aabb,
    step_spring, step_thermal, step_xpbd_world, Aabb3D, ArticulatedRagdoll, BallMotorJoint,
    BroadphaseBvh, ColliderShape3D, ContactManifold3D, ContactPoint3D, DrivenSpring,
    FixedMotorJoint, FractureOutcome3D, MotorJoint3D, NarrowphaseContact, RagdollMode, RayHit3D,
    RevoluteMotorJoint, RigidBody3D, RigidBodyType3D, SdfPrimitive, ShapeCastHit3D,
    SpringState, SpringSuspensionJoint, ThermalTargets,
};
pub use ripple::{InteractiveRippleBuffer, RIPPLE_GRID_DIM, RIPPLE_TOTAL_CELLS};
pub use space_weather::{
    evaluate_solar_cycle_phase, evaluate_space_weather_coupling, AtmosphericElectricCircuit,
    GeomagneticStormState, HaleMagneticCycle, SchwabeSolarCycle, SolarWindState,
    BASELINE_TSI_W_M2, HALE_CYCLE_YEARS, SCHWABE_CYCLE_YEARS,
};
pub use time::{
    AgentState, CommandPayload, CommandReceipt, CommandReceiptStatus, DeterministicPrng,
    EntityState, FixedTickWorld, FixedTimestepAccumulator, FixedVec2, MultiRateSimClock,
    RejectionReason, SimCommand, SimulationCommand, WorldState, DEFAULT_MAX_SUBSTEPS,
    FIXED_DELTA_NANOS, FIXED_DELTA_SECONDS, FIXED_HZ, MAX_COMMANDS_PER_TICK, MAX_ENTITIES,
    SIMULATION_TICK_MS, SIMULATION_TICK_RATE_HZ,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sim_pipeline_integration() {
        let mut sim = FixedTimestepAccumulator::new();
        let mut input = ZeroHeapButtonInput::<4>::new();
        let mut cam = SpringChaseCamera::default();

        let substeps = sim.advance(0.0166);
        assert_eq!(substeps, 1);

        input.press(KEY_W);
        assert!(input.is_pressed(KEY_W));

        for _ in 0..substeps {
            cam.update([0.0, 1.0, 0.0], [0.0, 5.0, 0.0], 0.0, FIXED_DELTA_SECONDS);
        }

        let alpha = sim.interpolation_alpha();
        assert!((0.0..=1.0).contains(&alpha));
    }

    #[test]
    fn test_hydrological_character_ripple_pipeline() {
        let mut character = CharacterController::new(CharacterConfig::default(), [0.0, 0.0, 0.0]);
        let mut hysteresis = HydrologicalHysteresisGate::default();
        let mut ripple = InteractiveRippleBuffer::new(0.0, 0.0, 16.0, 2.5, 0.985);
        let baro = BarometricState::default();
        let abl = BoundaryLayerProfile::default();

        let dt = FIXED_DELTA_SECONDS as f32;
        let waves = SimdGerstnerSpectrum::default();
        let _surface_h = waves.sample_surface(0.0, 0.0, 0.0);

        let sub = character.step_with_water(
            [0.0, 1.0],
            false,
            false,
            false,
            dt,
            |_x, _z| (0.0, [0.0, 1.0, 0.0]),
            |_x, _z| 0.6,
        );
        assert_eq!(sub, Some(SubmersionState::Wading));

        let _transition = hysteresis.update(0.6, 1, dt);
        assert!(hysteresis.wetness_ratio > 0.0);

        ripple.inject_impulse(character.state.position[0], character.state.position[2], 0.15, 0.3);
        ripple.step(dt);

        let wave_h = ripple.sample_height(character.state.position[0], character.state.position[2]);
        assert!(wave_h > 0.0);

        let rho = baro.air_density(0.0);
        assert!(rho > 1.15 && rho < 1.30);

        let wind_at_char = abl.sample_wind_speed(character.state.position[1] + 1.0);
        assert!(wind_at_char > 0.0);
    }

    #[test]
    fn test_flora_biophysics_and_ground_cover_pipeline() {
        let leaf_profile = LeafEconomicsProfile::eucalyptus_sclerophyll();
        let leaf = LeafVascularSkeleton::new(leaf_profile, 1.2);
        assert!(leaf.edges[0].radius_mm > leaf.edges[4].radius_mm);

        let phenology = ThermalPhenology::default();
        let mut tree = MetamerPlantGraph::new(phenology);
        tree.solve_growth_cycle(1.0, 0.8);
        assert!(tree.radius_m(0) > tree.radius_m(tree.count - 1));

        let mut herb = FloraBiomechanicsEvaluator::new(FloraArchetypeConfig::culinary_herb(), 0.5);
        herb.step_environment(-5.0, 0.5, 0.5);
        assert!(herb.is_dieback_active);
        assert!(herb.crown_height_m <= 0.05);

        let mut clipmap = ToroidalGroundCoverClipmap::default();
        clipmap.inject_trample(10.0, 10.0, 1.0, 0.0, 0.9);
        let cell_idx = clipmap.cell_index_at(5, 5);
        let fiber = clipmap.evaluate_sggx_fiber_direction(&clipmap.cells[cell_idx]);
        assert!(fiber[0] > 0.8);
        assert!(fiber[1] < 0.1);
    }
}
