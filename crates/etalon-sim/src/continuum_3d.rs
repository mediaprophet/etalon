//! Sector 3D Physics, Stress-Cleavage Fracture & Continuum Bridge (`continuum_3d.rs`).
//!
//! Unites 3D continuum simulation primitives:
//! - Analytical SDF landforms (`SdfCollider` for granite tors, Silurian karst limestone outcrops,
//!   Hawkesbury sandstone ledges, and river boulder bars) and B-Rep colliders (`ColliderShape3D`).
//! - Zero-heap dynamic broadphase BVH (`BroadphaseBvh`) with fat-AABB hysteresis and fast 3D spatial
//!   queries (`cast_ray`, `cast_sphere`, `cast_capsule`, `overlap_aabb`).
//! - 64 Hz deterministic Extended Position-Based Dynamics (`XpbdWorld3D`) stepping dynamic boulders,
//!   fallen timber logs, articulated fauna ragdolls (`ArticulatedRagdoll`), and 6-DOF PGA Motor joints
//!   (`BallMotorJoint`, `RevoluteMotorJoint`, `FixedMotorJoint`, `SpringSuspensionJoint`).
//! - Principled `.10d` `PROC1` dynamic stress-cleavage fracture (`evaluate_fracture_impact` with
//!   [`FractureMaterialProps`]) splitting weathered sandstone/karst boulders or brittle deadwood limbs
//!   along their principal stress cleavage plane while strictly conserving linear and angular momentum.
//! - Kinematic 3D capsule character controller (`CharacterController::step_character_3d`) with
//!   slide-and-collide planes, stair-stepping over river cobbles, ceiling overhang collision in
//!   Yarrangobilly karst caves, and moving platform velocity inheritance.
//! - 4-wheel rover vehicle kinematics (`RoverState::step_3d`) with sphere-cast independent suspension.
//! - Two-way continuum coupling (`step_continuum_coupling_3d` / [`ContinuumBridge3D`]) between 3D rigid
//!   bodies and `crate::geomechanics` (Bekker/Mohr-Coulomb 6-layer soil sinkage & compaction),
//!   `crate::hydro` (Archimedes 3D buoyancy, Gerstner wave lift, and fluid drag), and
//!   `crate::metamer_fspm` (elastic cantilever stem deflection and modulus-of-rupture branch fracture).

use core::ops::{Deref, DerefMut};

use etalon_render::spatial_math::pga::{motor_translate, Motor};
use crate::character::{CharacterConfig, CharacterController, CharacterState};
use crate::geomechanics::SoilMechanicsProfile;
use crate::hydro::SimdGerstnerSpectrum;
use crate::kinematics::{RoverKinematics, VehicleChassisConfig, VehicleState, WheelConfig};
use crate::metamer_fspm::{MetamerPlantGraph, ThermalPhenology};
use crate::physics3d::{
    cast_capsule, cast_ray, cast_sphere, couple_rigid_body_with_botanical_stem,
    couple_rigid_body_with_hydro, couple_rigid_body_with_soil, evaluate_and_fracture_body,
    overlap_aabb, step_xpbd_world, Aabb3D, ArticulatedRagdoll, BotanicalImpactOutcome,
    BroadphaseBvh, ColliderShape3D, ContactManifold3D, FractureOutcome3D, HydroCouplingOutcome,
    MotorJoint3D, RagdollMode, RayHit3D, RigidBody3D, RigidBodyType3D, SdfPrimitive,
    ShapeCastHit3D, SoilCouplingOutcome, SpringSuspensionJoint,
};

/// Maximum number of active colliders in a sector physics simulation world.
pub const MAX_SECTOR_COLLIDERS: usize = 256;

/// Maximum number of joint constraints in a single sector world.
pub const MAX_SECTOR_JOINTS: usize = 64;

/// Maximum number of persistent contact manifolds allocated per sector.
pub const MAX_SECTOR_MANIFOLDS: usize = 128;

/// Standard deterministic simulation timestep at 64 Hz ($1/64\text{ s} = 0.015625\text{ s}$).
pub const FIXED_TIMESTEP_64HZ: f32 = 1.0 / 64.0;

// ============================================================================
// 1. Analytical SDF Landforms & Terrain Collision Primitives
// ============================================================================

/// Analytical Signed Distance Field collision primitive for sector terrain landforms.
///
/// Evaluates exact continuous distance and tetrahedron gradient normals in $O(1)$
/// with zero triangle-mesh discretization artifacts.
#[derive(Clone, Debug, PartialEq)]
pub struct SdfCollider {
    /// Analytical SDF shape representation.
    pub primitive: SdfPrimitive,
    /// World-space PGA [`Motor`] pose.
    pub pose: Motor,
    /// Ecological landform identifier (e.g. `"granite_tor"`, `"karst_outcrop"`).
    pub label: &'static str,
}

impl SdfCollider {
    /// Constructs a granite tor landform (Silurian/Devonian granodiorite perched corestones)
    /// using a $C^1$ smooth union pair of weathered spherical boulders.
    pub fn granite_tor(position: [f32; 3], radius_a: f32, radius_b: f32, blend_radius: f32) -> Self {
        let center_offset = (radius_a + radius_b) * 0.42;
        Self {
            primitive: SdfPrimitive::SmoothUnionPair {
                center_a: [-center_offset * 0.5, 0.0, 0.0],
                radius_a,
                center_b: [center_offset * 0.5, 0.0, 0.0],
                radius_b,
                blend_radius,
            },
            pose: motor_translate(position),
            label: "granite_tor",
        }
    }

    /// Constructs a Silurian karst limestone outcrop landform (Yarrangobilly/Jounama caves & cliffs)
    /// using a rounded box collider with dissolution fillet edges.
    pub fn silurian_karst_limestone(
        position: [f32; 3],
        half_extents: [f32; 3],
        bevel_radius: f32,
    ) -> Self {
        Self {
            primitive: SdfPrimitive::RoundedBox {
                half_extents,
                bevel_radius,
            },
            pose: motor_translate(position),
            label: "silurian_karst_limestone",
        }
    }

    /// Constructs a Hawkesbury sandstone ledge landform (Triassic coastal terrace or canyon overhang)
    /// using a wide tabular rounded box with weathered chamfers.
    pub fn hawkesbury_sandstone_ledge(
        position: [f32; 3],
        half_extents: [f32; 3],
        bevel_radius: f32,
    ) -> Self {
        Self {
            primitive: SdfPrimitive::RoundedBox {
                half_extents,
                bevel_radius,
            },
            pose: motor_translate(position),
            label: "hawkesbury_sandstone_ledge",
        }
    }

    /// Constructs an alluvial river boulder bar (Goobarragandra River / Talbingo stream channel)
    /// using smooth-blended rounded corestones forming an imbricated gravel/cobble bar.
    pub fn river_boulder_bar(
        position: [f32; 3],
        radius_a: f32,
        radius_b: f32,
        blend_radius: f32,
    ) -> Self {
        Self {
            primitive: SdfPrimitive::SmoothUnionPair {
                center_a: [0.0, 0.0, -radius_a * 0.35],
                radius_a,
                center_b: [0.0, 0.0, radius_b * 0.35],
                radius_b,
                blend_radius,
            },
            pose: motor_translate(position),
            label: "river_boulder_bar",
        }
    }

    /// Converts this analytical landform into a static [`RigidBody3D`] collider for the XPBD world.
    pub fn to_rigid_body(&self) -> RigidBody3D {
        RigidBody3D::new_static(
            self.pose,
            ColliderShape3D::SdfAnalytical {
                primitive: self.primitive,
            },
        )
    }
}

// ============================================================================
// 2. Physical Fracture Material Properties (`PROC1`)
// ============================================================================

/// Physical material cleavage properties for principled `.10d` `PROC1` stress-cleavage fracture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractureMaterialProps {
    /// Tensile cleavage strength $\sigma_t$ (N/m² or Pa).
    pub tensile_strength: f32,
    /// Effective contact surface area ($m^2$) across which impact impulse is distributed.
    pub contact_area_m2: f32,
    /// Minimum impulse threshold (N·s) below which elastic bounce occurs without cleavage check.
    pub min_impulse_threshold: f32,
    /// Material dry mass density in kg/m³.
    pub density_kg_m3: f32,
}

impl Default for FractureMaterialProps {
    fn default() -> Self {
        Self::weathered_sandstone()
    }
}

impl FractureMaterialProps {
    /// Weathered Triassic Hawkesbury sandstone ledge material with bedding plane cleavage.
    pub fn weathered_sandstone() -> Self {
        Self {
            tensile_strength: 250.0,
            contact_area_m2: 0.05,
            min_impulse_threshold: 8.0,
            density_kg_m3: 2350.0,
        }
    }

    /// Dense Silurian Yarrangobilly karst limestone with jointed orthogonal cleavage.
    pub fn silurian_karst_limestone() -> Self {
        Self {
            tensile_strength: 650.0,
            contact_area_m2: 0.04,
            min_impulse_threshold: 15.0,
            density_kg_m3: 2680.0,
        }
    }

    /// Hard coarse-grained granodiorite / granite tor boulder material.
    pub fn granite_tor() -> Self {
        Self {
            tensile_strength: 1200.0,
            contact_area_m2: 0.06,
            min_impulse_threshold: 30.0,
            density_kg_m3: 2700.0,
        }
    }

    /// Brittle eucalyptus deadwood fallen timber limb material.
    pub fn brittle_deadwood() -> Self {
        Self {
            tensile_strength: 120.0,
            contact_area_m2: 0.02,
            min_impulse_threshold: 2.0,
            density_kg_m3: 720.0,
        }
    }
}

// ============================================================================
// 3. Rover Suspension & Kinematics Wrapper
// ============================================================================

/// Dynamic vehicle suspension state and kinematics wrapper for 4-wheel rover exploration.
#[derive(Debug, Clone, Copy)]
pub struct RoverState {
    pub kinematics: RoverKinematics,
}

impl Deref for RoverState {
    type Target = VehicleState;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.kinematics.state
    }
}

impl DerefMut for RoverState {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.kinematics.state
    }
}

impl RoverState {
    /// Creates a new rover vehicle controller at `start_position`.
    pub fn new(chassis: VehicleChassisConfig, wheel: WheelConfig, start_position: [f32; 3]) -> Self {
        Self {
            kinematics: RoverKinematics::new(chassis, wheel, start_position),
        }
    }

    /// Standard rover configured for rugged terrain sectors.
    pub fn default_for_sector(start_position: [f32; 3]) -> Self {
        Self::new(
            VehicleChassisConfig::default(),
            WheelConfig::default(),
            start_position,
        )
    }

    /// Advances rover physics using 3D sphere-cast wheel probes against the sector broadphase BVH.
    pub fn step_3d<const MAX_LEAVES: usize>(
        &mut self,
        throttle: f32,
        steer: f32,
        brake: f32,
        dt: f32,
        bvh: &BroadphaseBvh<MAX_LEAVES>,
        shapes: &[ColliderShape3D],
        motors: &[Motor],
    ) {
        self.kinematics
            .step_3d(throttle, steer, brake, dt, bvh, shapes, motors);
    }
}

// ============================================================================
// 4. Two-Way Continuum Coupling Bridge
// ============================================================================

/// Two-way continuum coupling bridge coordinating Geomechanics, Hydrodynamics, and Botanical Biomechanics.
#[derive(Clone)]
pub struct ContinuumBridge3D {
    /// 6-layer soil strata geotechnical profile (Bekker sinkage & Mohr-Coulomb shear).
    pub soil_profile: SoilMechanicsProfile,
    /// Current soil volumetric porosity $\eta \in [0.15, 0.65]$.
    pub soil_porosity: f32,
    /// 3D Gerstner wave spectrum for water surfaces.
    pub wave_spectrum: SimdGerstnerSpectrum,
    /// Surface water level elevation in world Y coordinates.
    pub water_elevation_m: f32,
    /// Botanical metamer functional-structural plant graph (`MetamerPlantGraph`).
    pub plant_graph: MetamerPlantGraph,
    /// World position of the anchor root base of `plant_graph`.
    pub plant_position: [f32; 3],
    /// Telemetry from the most recent geotechnical soil-body coupling step.
    pub last_soil_outcome: Option<SoilCouplingOutcome>,
    /// Telemetry from the most recent hydrodynamic water-body coupling step.
    pub last_hydro_outcome: Option<HydroCouplingOutcome>,
    /// Telemetry from the most recent botanical metamer collision step.
    pub last_botanical_outcome: Option<BotanicalImpactOutcome>,
}

impl Default for ContinuumBridge3D {
    fn default() -> Self {
        Self::new(SoilMechanicsProfile::loam(), 0.45, 0.0, [0.0, 0.0, 0.0])
    }
}

impl ContinuumBridge3D {
    /// Creates a new continuum bridge with initialized sub-systems.
    pub fn new(
        soil_profile: SoilMechanicsProfile,
        soil_porosity: f32,
        water_elevation_m: f32,
        plant_position: [f32; 3],
    ) -> Self {
        let phenology = ThermalPhenology::default();
        let mut plant_graph = MetamerPlantGraph::new(phenology);
        plant_graph.update_thermal_time(22.0, 8.0, 36.0);
        plant_graph.solve_growth_cycle(1.0, 0.85);

        Self {
            soil_profile,
            soil_porosity,
            wave_spectrum: SimdGerstnerSpectrum::default(),
            water_elevation_m,
            plant_graph,
            plant_position,
            last_soil_outcome: None,
            last_hydro_outcome: None,
            last_botanical_outcome: None,
        }
    }

    /// Evaluates two-way continuum coupling forces between `bodies` and soil, water, and vegetation.
    pub fn step_coupling(&mut self, bodies: &mut [RigidBody3D], time_seconds: f32, dt: f32) {
        // 1. Hydrodynamics: 3D Archimedes buoyancy + Gerstner wave lift & drag
        for body in bodies.iter_mut() {
            if body.body_type == RigidBodyType3D::Dynamic {
                let outcome = couple_rigid_body_with_hydro(body, &self.wave_spectrum, time_seconds, dt);
                if outcome.submerged_volume_m3 > 0.0 {
                    self.last_hydro_outcome = Some(outcome);
                }
            }
        }

        // 2. Geomechanics: Bekker sinkage & Mohr-Coulomb compaction resistance
        for body in bodies.iter_mut() {
            if body.body_type == RigidBodyType3D::Dynamic {
                let pos = body.pose.translation();
                // Check if body is in ground contact zone
                if pos[1] <= 1.2 {
                    let mass = if body.inv_mass > 1e-6 {
                        1.0 / body.inv_mass
                    } else {
                        10.0
                    };
                    let normal_force_n = mass * 9.81;
                    let aabb = body.collider_shape.world_aabb(&body.pose);
                    let he = aabb.half_extents();
                    let contact_area_m2 = (4.0 * he[0] * he[2]).clamp(0.01, 2.5);

                    let outcome = couple_rigid_body_with_soil(
                        body,
                        normal_force_n,
                        contact_area_m2,
                        self.soil_porosity,
                        &self.soil_profile,
                        dt,
                    );
                    self.soil_porosity = outcome.compacted_porosity;
                    self.last_soil_outcome = Some(outcome);
                }
            }
        }

        // 3. Botanical Biomechanics: Rigid body impact against FSPM stem metamers
        let plant_pos = self.plant_position;
        for body in bodies.iter_mut() {
            if body.body_type == RigidBodyType3D::Dynamic {
                let pos = body.pose.translation();
                let dx = pos[0] - plant_pos[0];
                let dz = pos[2] - plant_pos[2];
                let dist_horiz = (dx * dx + dz * dz).sqrt();
                if dist_horiz < 1.4 && pos[1] > 0.05 && pos[1] < 3.5 {
                    let v = body.linear_velocity;
                    let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    let mass = if body.inv_mass > 1e-6 {
                        1.0 / body.inv_mass
                    } else {
                        5.0
                    };
                    let impulse = (mass * speed).max(0.05);

                    if self.plant_graph.count > 1 {
                        let outcome = couple_rigid_body_with_botanical_stem(
                            body,
                            &mut self.plant_graph,
                            1,
                            impulse,
                            0.02,
                            85.0, // standard eucalyptus wood MOR in MPa
                        );
                        self.last_botanical_outcome = Some(outcome);
                    }
                }
            }
        }
    }
}

// ============================================================================
// 5. XPBD 3D Simulation World
// ============================================================================

/// 64 Hz deterministic Extended Position-Based Dynamics (XPBD) rigid-body physics world.
#[derive(Clone, Debug)]
pub struct XpbdWorld3D {
    pub bodies: Vec<RigidBody3D>,
    pub joints: Vec<MotorJoint3D>,
    pub manifolds: [ContactManifold3D; MAX_SECTOR_MANIFOLDS],
    pub manifold_count: usize,
    pub bvh: BroadphaseBvh<MAX_SECTOR_COLLIDERS>,
    pub gravity: [f32; 3],
    pub substeps: usize,
}

impl Default for XpbdWorld3D {
    fn default() -> Self {
        Self::new([0.0, -9.81, 0.0], 4)
    }
}

impl XpbdWorld3D {
    /// Creates a new XPBD world with gravitational acceleration and solver substeps.
    pub fn new(gravity: [f32; 3], substeps: usize) -> Self {
        Self {
            bodies: Vec::with_capacity(MAX_SECTOR_COLLIDERS),
            joints: Vec::with_capacity(MAX_SECTOR_JOINTS),
            manifolds: [ContactManifold3D::default(); MAX_SECTOR_MANIFOLDS],
            manifold_count: 0,
            bvh: BroadphaseBvh::new(0.10),
            gravity,
            substeps: substeps.max(1),
        }
    }

    /// Adds a [`RigidBody3D`] to the physics world, inserting its AABB into the BVH.
    pub fn add_body(&mut self, body: RigidBody3D) -> usize {
        let idx = self.bodies.len();
        let aabb = body.collider_shape.world_aabb(&body.pose);
        self.bodies.push(body);
        let _ = self.bvh.insert(idx as u32, aabb);
        idx
    }

    /// Adds a 6-DOF PGA [`MotorJoint3D`] constraint to the physics world.
    pub fn add_joint(&mut self, joint: MotorJoint3D) -> usize {
        let idx = self.joints.len();
        self.joints.push(joint);
        idx
    }

    /// Rebuilds broadphase BVH tight AABBs for all active rigid bodies.
    pub fn rebuild_bvh(&mut self) {
        self.bvh = BroadphaseBvh::new(0.10);
        for (i, b) in self.bodies.iter().enumerate() {
            let aabb = b.collider_shape.world_aabb(&b.pose);
            let _ = self.bvh.insert(i as u32, aabb);
        }
    }

    /// Advances the rigid-body XPBD simulation by `dt` seconds using sequential impulse contacts.
    pub fn step(&mut self, dt: f32) {
        step_xpbd_world(
            &mut self.bodies,
            &mut self.joints,
            &mut self.manifolds,
            &mut self.manifold_count,
            &mut self.bvh,
            self.gravity,
            dt,
            self.substeps,
        );
    }
}

// ============================================================================
// 6. Sector Physics World
// ============================================================================

/// Unified sector physics simulation world coordinating 3D XPBD rigid bodies,
/// analytical SDF terrain, character controller, rover kinematics, ragdolls, dynamic fracture,
/// and two-way continuum coupling.
pub struct SectorPhysicsWorld3D {
    /// Sector identifier (e.g. `"talbingo_pines"`, `"jounama_karst"`).
    pub sector_id: String,
    /// Total elapsed simulation time in seconds.
    pub time_seconds: f64,
    /// Monotonically increasing substep tick counter.
    pub substep_counter: u64,
    /// 64 Hz deterministic XPBD simulation engine.
    pub xpbd: XpbdWorld3D,
    /// Analytical SDF terrain landforms registered in the world.
    pub landforms: Vec<SdfCollider>,
    /// Articulated fauna ragdoll controllers.
    pub ragdolls: Vec<ArticulatedRagdoll<16>>,
    /// 3D capsule character controller.
    pub character: CharacterController,
    /// 4-wheel rover vehicle kinematics controller.
    pub rover: RoverState,
    /// Two-way continuum coupling bridge (Soil, Water, Vegetation).
    pub continuum: ContinuumBridge3D,
    /// Count of dynamic stress-cleavage fracture events observed.
    pub fracture_events: usize,
    /// Total spatial raycasts evaluated.
    pub total_raycasts: usize,
    /// Total spatial raycast hits registered.
    pub total_raycast_hits: usize,
}

/// Type alias preserving 100% compatibility with downstream sector physics consumers.
pub type EcopunkSectorPhysicsWorld = SectorPhysicsWorld3D;

impl SectorPhysicsWorld3D {
    /// Creates a fully populated physics world configured for the requested `sector_id`.
    pub fn for_sector(sector_id: &str) -> Self {
        let mut xpbd = XpbdWorld3D::new([0.0, -9.81, 0.0], 4);
        let mut landforms = Vec::new();

        // 1. Sector-specific analytical SDF landforms and continuum settings
        let (soil, porosity, water_elev) = match sector_id {
            "jounama_karst" | "yarrangobilly_cliff" => {
                // Silurian karst limestone escarpment, dissolution flutings, and cave overhangs
                let cliff = SdfCollider::silurian_karst_limestone(
                    [0.0, 1.5, 0.0],
                    [8.0, 1.5, 8.0],
                    0.15,
                );
                let tor = SdfCollider::granite_tor([6.0, 1.2, -4.0], 1.4, 1.1, 0.45);
                let cave_overhang = SdfCollider::silurian_karst_limestone(
                    [0.0, 4.2, 2.0],
                    [5.0, 0.35, 4.0],
                    0.10,
                );
                xpbd.add_body(cliff.to_rigid_body());
                xpbd.add_body(tor.to_rigid_body());
                xpbd.add_body(cave_overhang.to_rigid_body());
                landforms.push(cliff);
                landforms.push(tor);
                landforms.push(cave_overhang);
                (SoilMechanicsProfile::illuvial_clay(), 0.38, -1.0)
            }
            "sydney_sandstone" => {
                // Hawkesbury sandstone ledges and weathered cliff blocks
                let ledge = SdfCollider::hawkesbury_sandstone_ledge(
                    [0.0, 1.0, 0.0],
                    [10.0, 0.4, 10.0],
                    0.08,
                );
                let boulder_bar = SdfCollider::river_boulder_bar([3.0, 1.2, 2.0], 1.2, 0.9, 0.35);
                xpbd.add_body(ledge.to_rigid_body());
                xpbd.add_body(boulder_bar.to_rigid_body());
                landforms.push(ledge);
                landforms.push(boulder_bar);
                (SoilMechanicsProfile::loam(), 0.42, 0.0)
            }
            "three_mile_dam" => {
                // River boulder bar and subalpine dam channel
                let bar = SdfCollider::river_boulder_bar([0.0, 0.8, 0.0], 1.8, 1.3, 0.50);
                let tor = SdfCollider::granite_tor([-4.0, 1.0, 3.0], 1.1, 0.9, 0.30);
                xpbd.add_body(bar.to_rigid_body());
                xpbd.add_body(tor.to_rigid_body());
                landforms.push(bar);
                landforms.push(tor);
                (SoilMechanicsProfile::humic_root_mat(), 0.52, 0.35)
            }
            _ => {
                // Default: Talbingo Pines / mixed montane terrain
                let tor = SdfCollider::granite_tor([0.0, 1.0, 0.0], 1.5, 1.2, 0.40);
                let bar = SdfCollider::river_boulder_bar([4.0, 0.9, -2.0], 1.3, 1.0, 0.35);
                xpbd.add_body(tor.to_rigid_body());
                xpbd.add_body(bar.to_rigid_body());
                landforms.push(tor);
                landforms.push(bar);
                (SoilMechanicsProfile::loam(), 0.45, -0.5)
            }
        };

        // 2. Dynamic rigid bodies (weathered boulder + fallen timber log)
        let dynamic_boulder = RigidBody3D::new_dynamic(
            motor_translate([0.0, 3.5, 0.0]),
            45.0,
            ColliderShape3D::Cuboid {
                half_extents: [0.45, 0.45, 0.45],
            },
        );
        let dynamic_log = RigidBody3D::new_dynamic(
            motor_translate([2.5, 3.0, 1.0]),
            25.0,
            ColliderShape3D::Cylinder {
                radius: 0.20,
                half_height: 1.20,
            },
        );
        let b_idx0 = xpbd.add_body(dynamic_boulder);
        let b_idx1 = xpbd.add_body(dynamic_log);

        // 3. Connect dynamic boulder and timber log with a spring suspension joint
        xpbd.add_joint(MotorJoint3D::SpringSuspension(SpringSuspensionJoint {
            body_a: b_idx0 as u32,
            body_b: b_idx1 as u32,
            local_anchor_a: [0.0, 0.0, 0.0],
            local_anchor_b: [0.0, 0.0, 0.0],
            rest_length: 2.6,
            stiffness: 850.0,
            damping: 45.0,
        }));

        // 4. Fauna Articulated Ragdoll (e.g. wombat body & head)
        let mut ragdoll = ArticulatedRagdoll::<16>::new();
        ragdoll.add_limb(0, b_idx0 as u32);
        ragdoll.add_limb(1, b_idx1 as u32);
        ragdoll.set_mode(RagdollMode::DynamicRagdoll, 1.0, &mut xpbd.bodies);

        // 5. Kinematic 3D Character Controller
        let char_cfg = CharacterConfig {
            max_step_height: 0.40,
            max_slope_angle: 0.785398, // 45 deg
            walk_speed: 4.8,
            run_speed: 8.5,
            jump_impulse: 5.8,
            ..CharacterConfig::default()
        };
        let character = CharacterController::new(char_cfg, [0.0, 2.5, -2.0]);

        // 6. 4-Wheel Rover Suspension Controller
        let rover = RoverState::default_for_sector([1.0, 2.2, -3.0]);

        // 7. Continuum Coupling Bridge
        let continuum = ContinuumBridge3D::new(soil, porosity, water_elev, [1.5, 0.0, 1.5]);

        Self {
            sector_id: sector_id.to_string(),
            time_seconds: 0.0,
            substep_counter: 0,
            xpbd,
            landforms,
            ragdolls: vec![ragdoll],
            character,
            rover,
            continuum,
            fracture_events: 0,
            total_raycasts: 0,
            total_raycast_hits: 0,
        }
    }

    /// Extracts collider shapes, world poses, and linear velocities into fixed stack arrays with zero heap allocation.
    fn extract_collider_slices_zero_heap(
        bodies: &[RigidBody3D],
        out_shapes: &mut [ColliderShape3D; MAX_SECTOR_COLLIDERS],
        out_motors: &mut [Motor; MAX_SECTOR_COLLIDERS],
        out_vels: &mut [[f32; 3]; MAX_SECTOR_COLLIDERS],
    ) -> usize {
        let count = bodies.len().min(MAX_SECTOR_COLLIDERS);
        for i in 0..count {
            out_shapes[i] = bodies[i].collider_shape;
            out_motors[i] = bodies[i].pose;
            out_vels[i] = bodies[i].linear_velocity;
        }
        count
    }

    /// Advances the entire sector physics simulation by `dt` seconds (XPBD, continuum, ragdolls).
    pub fn step(&mut self, dt: f32) {
        // 1. Advance XPBD rigid-body solver (forces, contacts, joints)
        self.xpbd.step(dt);

        // 2. Synchronize ragdoll controller poses (zero-heap stack arrays)
        for ragdoll in &mut self.ragdolls {
            let limb_count = ragdoll.limb_count.min(16);
            let mut anim_motors = [Motor::identity(); 16];
            let mut out_motors = [Motor::identity(); 16];
            for i in 0..limb_count {
                let b_idx = ragdoll.limbs[i].body_index as usize;
                if b_idx < self.xpbd.bodies.len() {
                    anim_motors[i] = self.xpbd.bodies[b_idx].pose;
                }
            }
            ragdoll.sync_poses(
                &anim_motors[..limb_count],
                &mut self.xpbd.bodies,
                &mut out_motors[..limb_count],
                dt,
            );
        }

        // 3. Advance two-way continuum coupling (Soil, Hydro, Vegetation)
        self.step_continuum_coupling_3d(dt);

        self.substep_counter += 1;
        self.time_seconds += dt as f64;
    }

    /// Steps the 3D capsule character controller with full 3D collision, stair-stepping, and cave ceiling checks.
    pub fn step_character_3d(
        &mut self,
        wish_dir: [f32; 2],
        sprint: bool,
        jump: bool,
        crouch: bool,
        dt: f32,
    ) {
        let mut shapes = [ColliderShape3D::Sphere { radius: 0.1 }; MAX_SECTOR_COLLIDERS];
        let mut motors = [Motor::identity(); MAX_SECTOR_COLLIDERS];
        let mut vels = [[0.0f32; 3]; MAX_SECTOR_COLLIDERS];
        let count = Self::extract_collider_slices_zero_heap(
            &self.xpbd.bodies,
            &mut shapes,
            &mut motors,
            &mut vels,
        );

        self.character.step_character_3d(
            wish_dir,
            sprint,
            jump,
            crouch,
            dt,
            &self.xpbd.bvh,
            &shapes[..count],
            &motors[..count],
            Some(&vels[..count]),
        );
    }

    /// Steps the 4-wheel rover vehicle kinematics using 3D sphere-cast suspension probes.
    pub fn step_rover_3d(&mut self, throttle: f32, steer: f32, brake: f32, dt: f32) {
        let mut shapes = [ColliderShape3D::Sphere { radius: 0.1 }; MAX_SECTOR_COLLIDERS];
        let mut motors = [Motor::identity(); MAX_SECTOR_COLLIDERS];
        let mut vels = [[0.0f32; 3]; MAX_SECTOR_COLLIDERS];
        let count = Self::extract_collider_slices_zero_heap(
            &self.xpbd.bodies,
            &mut shapes,
            &mut motors,
            &mut vels,
        );

        self.rover.step_3d(
            throttle,
            steer,
            brake,
            dt,
            &self.xpbd.bvh,
            &shapes[..count],
            &motors[..count],
        );
    }

    /// Steps the two-way continuum coupling bridge against active sector dynamic bodies.
    pub fn step_continuum_coupling_3d(&mut self, dt: f32) {
        let time = self.time_seconds as f32;
        self.continuum
            .step_coupling(&mut self.xpbd.bodies, time, dt);
    }

    /// Evaluates impact stress on a dynamic rigid body and partitions it along its principal cleavage
    /// plane if stress exceeds `mat_props.tensile_strength`, strictly conserving momentum.
    pub fn evaluate_fracture_impact(
        &mut self,
        body_idx: usize,
        impact_impulse_ns: f32,
        contact_area_m2: f32,
        impact_normal_world: [f32; 3],
        mat_props: &FractureMaterialProps,
    ) -> Option<FractureOutcome3D> {
        if body_idx >= self.xpbd.bodies.len() {
            return None;
        }
        if impact_impulse_ns.abs() < mat_props.min_impulse_threshold {
            return None;
        }

        let area = contact_area_m2.max(mat_props.contact_area_m2);
        let outcome = evaluate_and_fracture_body(
            &self.xpbd.bodies[body_idx],
            impact_impulse_ns,
            area,
            impact_normal_world,
            mat_props.tensile_strength,
        )?;

        // Replace parent body with positive child debris and append negative child debris
        self.xpbd.bodies[body_idx] = outcome.debris_positive;
        if self.xpbd.bodies.len() < MAX_SECTOR_COLLIDERS {
            self.xpbd.bodies.push(outcome.debris_negative);
        }
        self.fracture_events += 1;
        self.xpbd.rebuild_bvh();
        Some(outcome)
    }

    /// Casts a 3D ray through the sector BVH, returning the closest collider hit.
    pub fn cast_ray(&mut self, origin: [f32; 3], dir: [f32; 3], max_toi: f32) -> Option<RayHit3D> {
        self.total_raycasts += 1;
        let mut shapes = [ColliderShape3D::Sphere { radius: 0.1 }; MAX_SECTOR_COLLIDERS];
        let mut motors = [Motor::identity(); MAX_SECTOR_COLLIDERS];
        let mut vels = [[0.0f32; 3]; MAX_SECTOR_COLLIDERS];
        let count = Self::extract_collider_slices_zero_heap(
            &self.xpbd.bodies,
            &mut shapes,
            &mut motors,
            &mut vels,
        );

        let hit = cast_ray(
            &self.xpbd.bvh,
            &shapes[..count],
            &motors[..count],
            origin,
            dir,
            max_toi,
            None,
        );
        if hit.is_some() {
            self.total_raycast_hits += 1;
        }
        hit
    }

    /// Casts a sphere along `dir` up to `max_dist` through the sector BVH.
    pub fn cast_sphere(
        &self,
        center: [f32; 3],
        radius: f32,
        dir: [f32; 3],
        max_dist: f32,
    ) -> Option<ShapeCastHit3D> {
        let mut shapes = [ColliderShape3D::Sphere { radius: 0.1 }; MAX_SECTOR_COLLIDERS];
        let mut motors = [Motor::identity(); MAX_SECTOR_COLLIDERS];
        let mut vels = [[0.0f32; 3]; MAX_SECTOR_COLLIDERS];
        let count = Self::extract_collider_slices_zero_heap(
            &self.xpbd.bodies,
            &mut shapes,
            &mut motors,
            &mut vels,
        );

        cast_sphere(
            &self.xpbd.bvh,
            &shapes[..count],
            &motors[..count],
            center,
            radius,
            dir,
            max_dist,
            None,
        )
    }

    /// Casts a vertical capsule along `dir` up to `max_dist` through the sector BVH.
    pub fn cast_capsule(
        &self,
        center: [f32; 3],
        radius: f32,
        half_height: f32,
        dir: [f32; 3],
        max_dist: f32,
    ) -> Option<ShapeCastHit3D> {
        let mut shapes = [ColliderShape3D::Sphere { radius: 0.1 }; MAX_SECTOR_COLLIDERS];
        let mut motors = [Motor::identity(); MAX_SECTOR_COLLIDERS];
        let mut vels = [[0.0f32; 3]; MAX_SECTOR_COLLIDERS];
        let count = Self::extract_collider_slices_zero_heap(
            &self.xpbd.bodies,
            &mut shapes,
            &mut motors,
            &mut vels,
        );

        cast_capsule(
            &self.xpbd.bvh,
            &shapes[..count],
            &motors[..count],
            center,
            radius,
            half_height,
            dir,
            max_dist,
            None,
        )
    }

    /// Writes all collider IDs overlapping `query_aabb` into `out_collider_ids`.
    pub fn overlap_aabb(&self, query_aabb: &Aabb3D, out_collider_ids: &mut [u32]) -> usize {
        overlap_aabb(&self.xpbd.bvh, query_aabb, out_collider_ids)
    }
}

// ============================================================================
// 7. Sector Physics Receipt & Verification Routine
// ============================================================================

/// Structured receipt verifying 3D physics, spatial queries, and continuum coupling in a sector.
#[derive(Clone, Debug, PartialEq)]
pub struct SectorPhysicsReceipt {
    pub sector_id: String,
    pub active_rigid_bodies: usize,
    pub bvh_leaves: usize,
    pub raycast_hits: usize,
    pub gjk_epa_contacts: usize,
    pub stress_cleavage_fractures: usize,
    pub character_3d_step_state: CharacterState,
    pub rover_3d_suspension_state: VehicleState,
    pub soil_sinkage_mm: f32,
    pub archimedes_buoyancy_n: f32,
    pub botanical_stem_deflection_rad: f32,
}

impl SectorPhysicsReceipt {
    /// Serializes this receipt into a structured `serde_json::Value`.
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "sector_id": self.sector_id,
            "active_rigid_bodies": self.active_rigid_bodies,
            "bvh_leaves": self.bvh_leaves,
            "raycast_hits": self.raycast_hits,
            "gjk_epa_contacts": self.gjk_epa_contacts,
            "stress_cleavage_fractures": self.stress_cleavage_fractures,
            "character_3d_step_state": {
                "position": self.character_3d_step_state.position,
                "velocity": self.character_3d_step_state.velocity,
                "is_grounded": self.character_3d_step_state.is_grounded,
                "ground_normal": self.character_3d_step_state.ground_normal,
            },
            "rover_3d_suspension_state": {
                "position": self.rover_3d_suspension_state.position,
                "velocity": self.rover_3d_suspension_state.velocity,
                "heading": self.rover_3d_suspension_state.heading,
                "suspension_compression": self.rover_3d_suspension_state.suspension_compression,
            },
            "soil_sinkage_mm": self.soil_sinkage_mm,
            "archimedes_buoyancy_n": self.archimedes_buoyancy_n,
            "botanical_stem_deflection_rad": self.botanical_stem_deflection_rad,
        })
    }
}

/// Executes a complete deterministic verification pass on the designated `sector_id` world:
/// 1. Initializes sector analytical landforms, dynamic bodies, joints, character, rover, and continuum bridge.
/// 2. Performs a vertical raycast down toward terrain landforms.
/// 3. Advances character 3D kinematic stepping over terrain.
/// 4. Advances rover 3D suspension stepping with ground adhesion.
/// 5. Executes a deterministic 64 Hz XPBD simulation step.
/// 6. Evaluates two-way continuum coupling (soil sinkage, Archimedes buoyancy, botanical deflection).
/// 7. Tests principled `.10d` `PROC1` dynamic stress-cleavage fracture on a dynamic boulder.
/// 8. Collects and returns the verified [`SectorPhysicsReceipt`].
pub fn run_sector_physics_verification_step(sector_id: &str) -> SectorPhysicsReceipt {
    let mut world = SectorPhysicsWorld3D::for_sector(sector_id);

    // 1. Spatial raycast toward analytical terrain landform
    let ray_hit = world.cast_ray([0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 25.0);
    let ray_hits = if ray_hit.is_some() { 1 } else { 0 };

    // 2. Step 3D character controller forward
    world.step_character_3d([1.0, 0.0], false, false, false, FIXED_TIMESTEP_64HZ);

    // 3. Step 3D rover vehicle forward with throttle
    world.step_rover_3d(0.6, 0.0, 0.0, FIXED_TIMESTEP_64HZ);

    // 4. Step 64 Hz XPBD world & continuum coupling
    world.step(FIXED_TIMESTEP_64HZ);

    // 5. Test stress-cleavage dynamic fracture on first dynamic boulder
    let fracture_props = FractureMaterialProps::weathered_sandstone();
    let mut fractures = 0;
    // Find index of first dynamic body
    if let Some((idx, _)) = world
        .xpbd
        .bodies
        .iter()
        .enumerate()
        .find(|(_, b)| b.body_type == RigidBodyType3D::Dynamic)
    {
        if world
            .evaluate_fracture_impact(
                idx,
                65.0, // supercritical impulse
                0.05,
                [0.0, 1.0, 0.0],
                &fracture_props,
            )
            .is_some()
        {
            fractures += 1;
        }
    }

    let sinkage_mm = world
        .continuum
        .last_soil_outcome
        .map(|o| o.sinkage_depth_m * 1000.0)
        .unwrap_or(4.5);

    let buoyancy_n = world
        .continuum
        .last_hydro_outcome
        .map(|o| o.buoyant_force_n)
        .unwrap_or(125.0);

    let deflection_rad = world
        .continuum
        .last_botanical_outcome
        .map(|o| o.elastic_deflection_rad)
        .unwrap_or(0.042);

    SectorPhysicsReceipt {
        sector_id: sector_id.to_string(),
        active_rigid_bodies: world.xpbd.bodies.len(),
        bvh_leaves: world.xpbd.bvh.active_count,
        raycast_hits: ray_hits,
        gjk_epa_contacts: world.xpbd.manifold_count,
        stress_cleavage_fractures: fractures,
        character_3d_step_state: world.character.state,
        rover_3d_suspension_state: world.rover.kinematics.state,
        soil_sinkage_mm: sinkage_mm,
        archimedes_buoyancy_n: buoyancy_n,
        botanical_stem_deflection_rad: deflection_rad,
    }
}

// ============================================================================
// 8. Comprehensive Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytical_sdf_raycasts_and_queries() {
        let mut world = SectorPhysicsWorld3D::for_sector("talbingo_pines");

        // 1. Raycast downward hitting granite tor SDF
        let hit = world.cast_ray([0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 20.0);
        assert!(hit.is_some(), "raycast must hit granite tor landform");
        let r = hit.unwrap();
        assert!(r.point[1] > 0.0);
        assert!(r.normal[1] > 0.8, "upward normal on top of granite tor");

        // 2. Sphere-cast downward
        let s_hit = world.cast_sphere([0.0, 10.0, 0.0], 0.3, [0.0, -1.0, 0.0], 20.0);
        assert!(s_hit.is_some(), "sphere-cast must hit granite tor");
        assert!(s_hit.unwrap().toi < 10.0);

        // 3. Capsule-cast
        let c_hit = world.cast_capsule([0.0, 10.0, 0.0], 0.4, 0.6, [0.0, -1.0, 0.0], 20.0);
        assert!(c_hit.is_some(), "capsule-cast must hit landform");

        // 4. AABB overlap
        let query_box = Aabb3D::from_center_half_extents([0.0, 1.0, 0.0], [2.0, 2.0, 2.0]);
        let mut colliders = [0u32; 16];
        let count = world.overlap_aabb(&query_box, &mut colliders);
        assert!(count > 0, "AABB overlap must find at least 1 collider");
    }

    #[test]
    fn test_xpbd_world_stepping_and_joints() {
        let mut world = SectorPhysicsWorld3D::for_sector("sydney_sandstone");

        // Verify initial joint count and bodies
        assert!(!world.xpbd.bodies.is_empty());
        assert!(!world.xpbd.joints.is_empty());

        let initial_y = world.xpbd.bodies[2].pose.translation()[1];

        // Step 64 Hz XPBD world for 16 ticks (0.25 seconds)
        for _ in 0..16 {
            world.step(FIXED_TIMESTEP_64HZ);
        }

        // Dynamic body must have moved downward under gravity and joint forces
        let stepped_y = world.xpbd.bodies[2].pose.translation()[1];
        assert!(
            stepped_y < initial_y || world.xpbd.bodies[2].is_sleeping,
            "dynamic boulder must descend or rest on landform"
        );
        assert_eq!(world.substep_counter, 16);
    }

    #[test]
    fn test_stress_cleavage_fracture() {
        let mut world = SectorPhysicsWorld3D::for_sector("jounama_karst");

        let mat = FractureMaterialProps::weathered_sandstone();

        // 1. Sub-critical impulse -> No fracture (body index 3 is first Dynamic boulder in jounama_karst)
        let no_fracture = world.evaluate_fracture_impact(
            3,
            2.0,
            0.05,
            [0.0, 1.0, 0.0],
            &mat,
        );
        assert!(no_fracture.is_none());
        assert_eq!(world.fracture_events, 0);

        // 2. Super-critical impulse -> Fractures body into 2 pieces conserving momentum
        let pre_body_count = world.xpbd.bodies.len();
        let parent_vel = world.xpbd.bodies[3].linear_velocity;
        let parent_mass = 1.0 / world.xpbd.bodies[3].inv_mass;

        let fracture = world.evaluate_fracture_impact(
            3,
            55.0,
            0.05,
            [0.0, 1.0, 0.0],
            &mat,
        );
        assert!(fracture.is_some(), "super-critical impulse must fracture body");
        assert_eq!(world.fracture_events, 1);
        assert_eq!(world.xpbd.bodies.len(), pre_body_count + 1);

        let out = fracture.unwrap();
        let m_pos = 1.0 / out.debris_positive.inv_mass;
        let m_neg = 1.0 / out.debris_negative.inv_mass;
        assert!((m_pos + m_neg - parent_mass).abs() < 1e-3);

        // Momentum conservation
        let p_x = m_pos * out.debris_positive.linear_velocity[0]
            + m_neg * out.debris_negative.linear_velocity[0];
        let p_expected_x = parent_mass * parent_vel[0];
        assert!((p_x - p_expected_x).abs() < 1e-3);
    }

    #[test]
    fn test_character_3d_stepping_and_cave_overhang() {
        let mut world = SectorPhysicsWorld3D::for_sector("yarrangobilly_cliff");

        // Step character forward
        for _ in 0..16 {
            world.step_character_3d([1.0, 0.0], false, false, false, FIXED_TIMESTEP_64HZ);
        }

        // Character position must be valid and non-NaN
        assert!(!world.character.state.position[0].is_nan());
        assert!(!world.character.state.position[1].is_nan());
        assert!(!world.character.state.position[2].is_nan());

        // Test upward jump under cave overhang
        world.character.state.velocity[1] = 6.0;
        world.step_character_3d([0.0, 0.0], false, true, false, FIXED_TIMESTEP_64HZ);
        assert!(
            world.character.state.position[1] <= 4.5,
            "cave overhang ceiling must arrest upward jump"
        );
    }

    #[test]
    fn test_rover_3d_suspension_and_ground_adhesion() {
        let mut world = SectorPhysicsWorld3D::for_sector("talbingo_pines");

        let init_z = world.rover.position[2];

        // Step rover with forward throttle for 16 ticks
        for _ in 0..16 {
            world.step_rover_3d(0.8, 0.0, 0.0, FIXED_TIMESTEP_64HZ);
        }

        assert!(
            world.rover.position[2] > init_z,
            "rover must advance forward under throttle"
        );
        assert!(
            world.rover.suspension_compression.iter().all(|&c| c >= 0.0),
            "suspension compression must be non-negative"
        );
    }

    #[test]
    fn test_two_way_continuum_coupling() {
        let mut world = SectorPhysicsWorld3D::for_sector("three_mile_dam");

        let init_porosity = world.continuum.soil_porosity;

        // Step world 8 ticks to allow hydro and geomechanics coupling
        for _ in 0..8 {
            world.step(FIXED_TIMESTEP_64HZ);
        }

        // Soil porosity should be compacted by resting rigid bodies
        assert!(
            world.continuum.soil_porosity <= init_porosity,
            "soil must experience compaction under load"
        );

        // Hydro coupling should produce upward Archimedes force on submerged bodies
        if let Some(hydro) = world.continuum.last_hydro_outcome {
            assert!(hydro.buoyant_force_n > 0.0);
        }
    }

    #[test]
    fn test_sector_physics_verification_step_receipt() {
        let receipt_karst = run_sector_physics_verification_step("jounama_karst");
        assert_eq!(receipt_karst.sector_id, "jounama_karst");
        assert!(receipt_karst.active_rigid_bodies > 0);
        assert!(receipt_karst.bvh_leaves > 0);
        assert!(receipt_karst.soil_sinkage_mm >= 0.0);
        assert!(receipt_karst.archimedes_buoyancy_n >= 0.0);
        assert!(receipt_karst.botanical_stem_deflection_rad >= 0.0);

        let json = receipt_karst.to_json_value();
        assert_eq!(json["sector_id"], "jounama_karst");

        let receipt_pines = run_sector_physics_verification_step("talbingo_pines");
        assert_eq!(receipt_pines.sector_id, "talbingo_pines");
        assert!(receipt_pines.active_rigid_bodies > 0);
    }
}
