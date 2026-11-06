//! Cold resident allocation and allocation-free visual playback lifecycle.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerError {
    InvalidInput,
    BudgetExceeded,
    UnknownClip,
    Kernel(SkeletalError),
    Tangents,
}

impl From<SkeletalError> for PlayerError {
    fn from(e: SkeletalError) -> Self {
        Self::Kernel(e)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackReceipt {
    pub clip_id: u64,
    pub time: f64,
    pub vertices: usize,
    pub joints: usize,
    pub reduced_motion: bool,
}

/// Metadata record describing an animation clip span in caller-owned track/key tables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipRecord {
    pub id: u64,
    pub duration: f64,
    pub first_track: u32,
    pub track_count: u32,
    pub looping: bool,
}

/// Build a runtime [`Clip`] view after loading the section into caller-owned arrays.
pub fn clip_view<'a>(record: ClipRecord, tracks: &'a [Track], keys: &'a [Key]) -> Option<Clip<'a>> {
    let end = (record.first_track as usize).checked_add(record.track_count as usize)?;
    Some(Clip {
        duration: record.duration,
        looping: record.looping,
        tracks: tracks.get(record.first_track as usize..end)?,
        keys,
    })
}

/// Validated element counts and encoded payload size for skeletal admission budgeting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkeletalCounts {
    pub joints: usize,
    pub vertices: usize,
    pub clips: usize,
    pub tracks: usize,
    pub keys: usize,
    pub encoded_bytes: Option<usize>,
}

/// Zero-copy section source abstraction so [`SkeletalPlayer`] can be constructed from
/// `.10d` `SkeletalSection` payloads or direct in-memory slice descriptors without
/// introducing reverse crate dependencies.
pub trait SkeletalSectionSource {
    /// Returns the element counts and encoded byte size for budget admission.
    fn section_counts(&self) -> SkeletalCounts;

    /// Copies the validated rig, skin weights, clips, tracks, and keys into caller-owned slices.
    fn decode_section_into(
        &self,
        joints: &mut [Joint],
        joint_ids: &mut [u64],
        skin: &mut [SkinWeight],
        clips: &mut [ClipRecord],
        tracks: &mut [Track],
        keys: &mut [Key],
    ) -> Result<(), PlayerError>;
}

/// Direct borrowed rig descriptor implementing [`SkeletalSectionSource`].
#[derive(Debug, Clone, Copy)]
pub struct SkeletalRigDescriptor<'a> {
    pub joints: &'a [Joint],
    pub joint_ids: &'a [u64],
    pub skin: &'a [SkinWeight],
    pub clips: &'a [ClipRecord],
    pub tracks: &'a [Track],
    pub keys: &'a [Key],
}

impl SkeletalSectionSource for SkeletalRigDescriptor<'_> {
    fn section_counts(&self) -> SkeletalCounts {
        let valid_counts = !self.joints.is_empty()
            && self.joints.len() <= MAX_JOINTS
            && self.joint_ids.len() == self.joints.len()
            && !self.skin.is_empty()
            && self.skin.len() <= MAX_VERTICES
            && !self.clips.is_empty()
            && self.clips.len() <= 32
            && !self.tracks.is_empty()
            && self.tracks.len() <= 4096
            && !self.keys.is_empty()
            && self.keys.len() <= MAX_KEYS;
        let encoded_bytes = valid_counts.then_some(
            32 + self.joints.len() * 144
                + self.clips.len() * 32
                + self.tracks.len() * 8
                + self.keys.len() * 72
                + self.skin.len() * 24,
        );
        SkeletalCounts {
            joints: self.joints.len(),
            vertices: self.skin.len(),
            clips: self.clips.len(),
            tracks: self.tracks.len(),
            keys: self.keys.len(),
            encoded_bytes,
        }
    }

    fn decode_section_into(
        &self,
        joints: &mut [Joint],
        joint_ids: &mut [u64],
        skin: &mut [SkinWeight],
        clips: &mut [ClipRecord],
        tracks: &mut [Track],
        keys: &mut [Key],
    ) -> Result<(), PlayerError> {
        if joints.len() < self.joints.len()
            || joint_ids.len() < self.joint_ids.len()
            || self.joint_ids.len() != self.joints.len()
            || skin.len() < self.skin.len()
            || clips.len() < self.clips.len()
            || tracks.len() < self.tracks.len()
            || keys.len() < self.keys.len()
        {
            return Err(PlayerError::InvalidInput);
        }
        validate_joints(self.joints).map_err(|_| PlayerError::InvalidInput)?;
        joints[..self.joints.len()].copy_from_slice(self.joints);
        joint_ids[..self.joint_ids.len()].copy_from_slice(self.joint_ids);
        skin[..self.skin.len()].copy_from_slice(self.skin);
        clips[..self.clips.len()].copy_from_slice(self.clips);
        tracks[..self.tracks.len()].copy_from_slice(self.tracks);
        keys[..self.keys.len()].copy_from_slice(self.keys);
        Ok(())
    }
}

/// Neutral body-fit specification applied in bind/CCF space before orbit-frame normalisation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyFitSpec {
    pub stature_scale: f32,
    pub torso_scale_y: f32,
    pub leg_scale_y: f32,
    pub arm_span_scale_x: f32,
    pub shoulder_scale_x: f32,
    pub chest_radial: f32,
    pub waist_radial: f32,
    pub hip_radial: f32,
    pub pregnancy_abdomen: f32,
    pub pelvis_y_norm: f32,
    pub waist_y_norm: f32,
    pub chest_y_norm: f32,
    pub shoulder_y_norm: f32,
    pub hidden_keys_count: usize,
    pub hidden_keys_bytes: usize,
    pub identity: bool,
}

impl Default for BodyFitSpec {
    fn default() -> Self {
        Self {
            stature_scale: 1.0,
            torso_scale_y: 1.0,
            leg_scale_y: 1.0,
            arm_span_scale_x: 1.0,
            shoulder_scale_x: 1.0,
            chest_radial: 1.0,
            waist_radial: 1.0,
            hip_radial: 1.0,
            pregnancy_abdomen: 0.0,
            pelvis_y_norm: 0.42,
            waist_y_norm: 0.52,
            chest_y_norm: 0.68,
            shoulder_y_norm: 0.78,
            hidden_keys_count: 0,
            hidden_keys_bytes: 0,
            identity: true,
        }
    }
}

impl BodyFitSpec {
    /// Transforms a single point by the anatomical fit inside `[gmin, gmax]`.
    pub fn transform_point(&self, p: [f32; 3], gmin: [f32; 3], gmax: [f32; 3]) -> [f32; 3] {
        let span_y = (gmax[1] - gmin[1]).max(1e-6);
        let mid_x = (gmin[0] + gmax[0]) * 0.5;
        let mid_z = (gmin[2] + gmax[2]) * 0.5;
        let y_norm = ((p[1] - gmin[1]) / span_y).clamp(0.0, 1.0);

        let mut x = p[0] - mid_x;
        let mut y = p[1];
        let mut z = p[2] - mid_z;

        let y_seg = if y_norm < self.pelvis_y_norm {
            self.leg_scale_y
        } else {
            self.torso_scale_y
        };
        y = gmin[1] + (y - gmin[1]) * y_seg;

        let radial = if y_norm < self.pelvis_y_norm {
            lerp(
                1.0,
                self.hip_radial,
                smoothstep(0.20, self.pelvis_y_norm, y_norm),
            )
        } else if y_norm < self.waist_y_norm {
            lerp(
                self.hip_radial,
                self.waist_radial,
                smoothstep(self.pelvis_y_norm, self.waist_y_norm, y_norm),
            )
        } else if y_norm < self.chest_y_norm {
            lerp(
                self.waist_radial,
                self.chest_radial,
                smoothstep(self.waist_y_norm, self.chest_y_norm, y_norm),
            )
        } else {
            lerp(
                self.chest_radial,
                self.shoulder_scale_x,
                smoothstep(self.chest_y_norm, self.shoulder_y_norm, y_norm),
            )
        };
        x *= radial * self.arm_span_scale_x;
        z *= radial;

        if self.pregnancy_abdomen > 0.0 {
            let band = bump(y_norm, 0.48, 0.10);
            z += self.pregnancy_abdomen * band * span_y * 0.18;
            x *= 1.0 + self.pregnancy_abdomen * band * 0.12;
        }

        [
            mid_x + x * self.stature_scale,
            gmin[1] + (y - gmin[1]) * self.stature_scale,
            mid_z + z * self.stature_scale,
        ]
    }

    /// Applies the fit deformation in place to `positions`.
    pub fn apply_in_place(&self, positions: &mut [[f32; 3]], gmin: [f32; 3], gmax: [f32; 3]) {
        if self.identity {
            return;
        }
        for p in positions.iter_mut() {
            *p = self.transform_point(*p, gmin, gmax);
        }
    }
}

/// Conversion trait allowing host body-fit descriptors (e.g. `AnatomyBodyFit`) to configure
/// [`SkeletalPlayer::set_body_fit`] without reverse crate dependencies.
pub trait BodyFitSource {
    /// Extracts the neutral numeric fit parameters and hidden-key budget metadata.
    fn to_body_fit_spec(&self) -> BodyFitSpec;
}

impl BodyFitSource for BodyFitSpec {
    fn to_body_fit_spec(&self) -> BodyFitSpec {
        *self
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn bump(x: f32, center: f32, width: f32) -> f32 {
    let t = ((x - center) / width).abs();
    if t >= 1.0 {
        0.0
    } else {
        1.0 - t * t
    }
}

/// Buffers are allocated once on validated scene admission, never on advance.
/// Geometry stays in the authored bind frame; output can receive a uniform
/// orbit transform without altering the rig or losing attachment alignment.
pub struct SkeletalPlayer {
    joints: Vec<Joint>,
    joint_ids: Vec<u64>,
    skin: Vec<SkinWeight>,
    clips: Vec<ClipRecord>,
    tracks: Vec<Track>,
    keys: Vec<Key>,
    bind_positions: Vec<[f32; 3]>,
    bind_normals: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    indices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tangents: Vec<[f32; 4]>,
    tangent_scratch: Vec<[f32; 6]>,
    local: Vec<Motor>,
    previous: Vec<Motor>,
    globals: Vec<Motor>,
    palette: Vec<Motor>,
    prev_skin_motors: Vec<Motor>,
    gpu_palette: Vec<super::gpu_skin::GpuJointMotorPair>,
    joint_deltas: Vec<Motor>,
    controlled_local: Vec<Motor>,
    clip: usize,
    time: f64,
    transition: f64,
    transition_duration: f64,
    root: Motor,
    origin: [f32; 3],
    scale: f32,
    allocated_bytes: usize,
    body_fit: Option<(BodyFitSpec, [f32; 3], [f32; 3])>,
    fit_bytes: usize,
}

impl SkeletalPlayer {
    /// Builds a player from a skeletal section + mesh attributes with the default 42 MiB resident budget; validates vertex/index/normal inputs and fails closed (`InvalidInput`/`BudgetExceeded`).
    pub fn from_section<S: SkeletalSectionSource + ?Sized>(
        section: &S,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        uv: &[[f32; 2]],
        indices: &[u32],
    ) -> Result<Self, PlayerError> {
        Self::from_section_with_budget(section, positions, normals, uv, indices, 42 * 1024 * 1024)
    }

    /// As `from_section` with an explicit caller byte budget — the admission pass counts borrowed inputs plus encoded section size and refuses before allocating beyond either bound.
    pub fn from_section_with_budget<S: SkeletalSectionSource + ?Sized>(
        section: &S,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        uv: &[[f32; 2]],
        indices: &[u32],
        resident_budget: usize,
    ) -> Result<Self, PlayerError> {
        let c = section.section_counts();
        let n = c.vertices;
        if positions.len() != n
            || normals.len() != n
            || uv.len() != n
            || indices.is_empty()
            || indices.len() % 3 != 0
            || indices.len() > MAX_VERTICES * 6
            || indices.iter().any(|i| *i as usize >= n)
            || positions
                .iter()
                .flatten()
                .chain(normals.iter().flatten())
                .chain(uv.iter().flatten())
                .any(|v| !v.is_finite() || v.abs() > 1e8)
            || normals
                .iter()
                .any(|v| v.iter().map(|x| x * x).sum::<f32>() < 1e-12)
        {
            return Err(PlayerError::InvalidInput);
        }
        let allocated_bytes = std::mem::size_of::<Self>()
            + n * 120
            + indices.len() * 4
            + c.joints * (std::mem::size_of::<Joint>() + 8 + 8 * 64)
            + c.clips * std::mem::size_of::<ClipRecord>()
            + c.tracks * std::mem::size_of::<Track>()
            + c.keys * std::mem::size_of::<Key>();
        // Include borrowed authoring inputs and the encoded section during the
        // admission pass; refuse before allocating beyond the execution budget.
        if allocated_bytes > resident_budget
            || allocated_bytes
                + n * 32
                + indices.len() * 4
                + c.encoded_bytes.unwrap_or(usize::MAX)
                > 42 * 1024 * 1024
        {
            return Err(PlayerError::BudgetExceeded);
        }
        let identity = Motor::identity();
        let mut result = Self {
            joints: vec![
                Joint {
                    parent: None,
                    rest_local: identity,
                    inverse_bind: identity
                };
                c.joints
            ],
            joint_ids: vec![0; c.joints],
            skin: vec![
                SkinWeight {
                    joints: [0; 4],
                    weights: [1.0, 0.0, 0.0, 0.0]
                };
                n
            ],
            clips: vec![
                ClipRecord {
                    id: 0,
                    duration: 1.0,
                    first_track: 0,
                    track_count: 0,
                    looping: false
                };
                c.clips
            ],
            tracks: vec![
                Track {
                    joint: 0,
                    first_key: 0,
                    key_count: 0
                };
                c.tracks
            ],
            keys: vec![
                Key {
                    time: 0.0,
                    local: identity
                };
                c.keys
            ],
            bind_positions: positions.to_vec(),
            bind_normals: normals.to_vec(),
            uv: uv.to_vec(),
            indices: indices.to_vec(),
            positions: vec![[0.0; 3]; n],
            normals: vec![[0.0; 3]; n],
            tangents: vec![[0.0; 4]; n],
            tangent_scratch: vec![[0.0; 6]; n],
            local: vec![identity; c.joints],
            previous: vec![identity; c.joints],
            globals: vec![identity; c.joints],
            palette: vec![identity; c.joints],
            prev_skin_motors: vec![identity; c.joints],
            gpu_palette: vec![super::gpu_skin::GpuJointMotorPair::identity(); c.joints],
            joint_deltas: vec![identity; c.joints],
            controlled_local: vec![identity; c.joints],
            clip: 0,
            time: 0.0,
            transition: 0.0,
            transition_duration: 0.0,
            root: identity,
            origin: [0.0; 3],
            scale: 1.0,
            allocated_bytes,
            body_fit: None,
            fit_bytes: 0,
        };
        section.decode_section_into(
            &mut result.joints,
            &mut result.joint_ids,
            &mut result.skin,
            &mut result.clips,
            &mut result.tracks,
            &mut result.keys,
        )?;
        result.advance(0.0, false)?;
        Ok(result)
    }

    /// Estimated resident bytes for joints, clips, tracks, keys, fit data — tracked against the 42 MiB budget.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }
    /// Vertex positions slice.
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }
    /// Vertex normals slice.
    pub fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }
    /// Vertex tangents slice.
    pub fn tangents(&self) -> &[[f32; 4]] {
        &self.tangents
    }
    /// `true` when a clip with `id` is registered.
    pub fn has_clip(&self, id: u64) -> bool {
        self.clips.iter().any(|c| c.id == id)
    }
    /// Applies a [`BodyFitSource`] inside `[min,max]` bounds; validates all scale fields are finite (≤100) and hidden-key payloads bounded (128 keys / 4 KiB); identity fit clears the override. Fails closed on `BudgetExceeded`.
    pub fn set_body_fit<F: BodyFitSource + ?Sized>(
        &mut self,
        fit: &F,
        min: [f32; 3],
        max: [f32; 3],
    ) -> Result<(), PlayerError> {
        let spec = fit.to_body_fit_spec();
        if min.iter().chain(max.iter()).any(|v| !v.is_finite())
            || (0..3).any(|a| min[a] > max[a])
            || spec.hidden_keys_count > 128
            || spec.hidden_keys_bytes > 4096
            || [
                spec.stature_scale,
                spec.torso_scale_y,
                spec.leg_scale_y,
                spec.arm_span_scale_x,
                spec.shoulder_scale_x,
                spec.chest_radial,
                spec.waist_radial,
                spec.hip_radial,
                spec.pregnancy_abdomen,
                spec.pelvis_y_norm,
                spec.waist_y_norm,
                spec.chest_y_norm,
                spec.shoulder_y_norm,
            ]
            .iter()
            .any(|v| !v.is_finite() || v.abs() > 100.0)
        {
            return Err(PlayerError::InvalidInput);
        }
        let bytes = if spec.identity {
            0
        } else {
            spec.hidden_keys_count * std::mem::size_of::<String>() + spec.hidden_keys_bytes
        };
        let allocated = self.allocated_bytes - self.fit_bytes + bytes;
        if allocated > 42 * 1024 * 1024 {
            return Err(PlayerError::BudgetExceeded);
        }
        self.body_fit = if spec.identity {
            None
        } else {
            Some((spec, min, max))
        };
        self.allocated_bytes = allocated;
        self.fit_bytes = bytes;
        Ok(())
    }
    /// Sets world `origin`/`scale`; rejects non-finite or out-of-range values.
    pub fn set_output_frame(&mut self, origin: [f32; 3], scale: f32) -> Result<(), PlayerError> {
        if origin.iter().any(|v| !v.is_finite() || v.abs() > 1e8)
            || !scale.is_finite()
            || scale <= 0.0
            || scale > 1e8
        {
            return Err(PlayerError::InvalidInput);
        }
        self.origin = origin;
        self.scale = scale;
        Ok(())
    }
    /// Sets the root `Motor` transform; `InvalidInput` on a non-unit/invalid motor.
    pub fn set_root(&mut self, root: Motor) -> Result<(), PlayerError> {
        if !valid_motor(&root) {
            return Err(PlayerError::InvalidInput);
        }
        self.root = root;
        Ok(())
    }
    /// `true` when `id` is a known joint.
    pub fn has_joint(&self, id: u64) -> bool {
        self.joint_ids.contains(&id)
    }
    /// Persistent local control delta, composed after the sampled clip pose.
    /// Wheels, steering and authored constraints use the same skinning path.
    pub fn set_joint_delta(&mut self, id: u64, delta: Motor) -> Result<(), PlayerError> {
        if !valid_motor(&delta) {
            return Err(PlayerError::InvalidInput);
        }
        let joint = self
            .joint_ids
            .iter()
            .position(|key| *key == id)
            .ok_or(PlayerError::InvalidInput)?;
        self.joint_deltas[joint] = delta;
        Ok(())
    }
    /// Switches to clip `id` with `fade_seconds` blend; no-op when already active (still validates fade range); delegates to `restart_clip` otherwise.
    pub fn select_clip(&mut self, id: u64, fade_seconds: f64) -> Result<(), PlayerError> {
        if self.clips[self.clip].id == id {
            if !fade_seconds.is_finite() || !(0.0..=10.0).contains(&fade_seconds) {
                return Err(PlayerError::InvalidInput);
            }
            return Ok(());
        }
        self.restart_clip(id, fade_seconds)
    }
    /// Restarts clip `id` at t=0, snapshotting current pose for the `fade_seconds` transition (≤10 s); `UnknownClip` when `id` is not registered.
    pub fn restart_clip(&mut self, id: u64, fade_seconds: f64) -> Result<(), PlayerError> {
        if !fade_seconds.is_finite() || !(0.0..=10.0).contains(&fade_seconds) {
            return Err(PlayerError::InvalidInput);
        }
        let clip = self
            .clips
            .iter()
            .position(|c| c.id == id)
            .ok_or(PlayerError::UnknownClip)?;
        self.previous.copy_from_slice(&self.local);
        self.clip = clip;
        self.time = 0.0;
        self.transition = 0.0;
        self.transition_duration = fade_seconds;
        Ok(())
    }
    /// Deterministic fixture/replay positioning, without a wall-clock source.
    pub fn seek(&mut self, time: f64) -> Result<(), PlayerError> {
        if !time.is_finite() || time.abs() > 1e12 {
            return Err(PlayerError::InvalidInput);
        }
        self.time = time;
        self.transition_duration = 0.0;
        Ok(())
    }
    /// Evaluates hierarchical joint `Motor`s and updates the GPU dual-quaternion
    /// skinning palette (`current` + `prev` for TAA motion vectors) with zero heap
    /// allocation and without executing the CPU per-vertex skinning loop.
    pub fn advance_pose_only(&mut self, dt_seconds: f64) -> Result<PlaybackReceipt, PlayerError> {
        self.advance_pose_only_with_mode(dt_seconds, false)
    }

    /// As [`Self::advance_pose_only`], with an explicit `reduced_motion` flag.
    pub fn advance_pose_only_with_mode(
        &mut self,
        dt_seconds: f64,
        reduced_motion: bool,
    ) -> Result<PlaybackReceipt, PlayerError> {
        self.advance_pose_only_with_policy(dt_seconds, reduced_motion, PoseTimePolicy::AuthoredLooping)
    }
    /// Explicit closed-interval sampling avoids wrapping a requested final
    /// contact boundary to the first pose. Stored clip metadata is unchanged.
    pub fn advance_pose_only_with_policy(
        &mut self,
        dt_seconds: f64,
        reduced_motion: bool,
        policy: PoseTimePolicy,
    ) -> Result<PlaybackReceipt, PlayerError> {
        if !dt_seconds.is_finite() || !(0.0..=1.0).contains(&dt_seconds) {
            return Err(PlayerError::InvalidInput);
        }
        self.prev_skin_motors.copy_from_slice(&self.palette);
        self.time += dt_seconds;
        self.transition += dt_seconds;
        let clip = self.clips[self.clip];
        if clip.looping && policy == PoseTimePolicy::AuthoredLooping {
            self.time = self.time.rem_euclid(clip.duration);
        } else {
            self.time = self.time.clamp(0.0, clip.duration);
        }
        let blend = if reduced_motion || self.transition_duration == 0.0 {
            1.0
        } else {
            (self.transition / self.transition_duration).clamp(0.0, 1.0)
        };
        let previous = (blend < 1.0).then_some(self.previous.as_slice());
        let mut sampled_clip =
            clip_view(clip, &self.tracks, &self.keys).ok_or(PlayerError::InvalidInput)?;
        // This borrowed invocation view controls sampling only. The admitted
        // ClipRecord and its loop-compatible source remain untouched.
        if policy == PoseTimePolicy::ExactClipInterval {
            sampled_clip.looping = false;
        }
        sample_pose_into(
            &self.joints,
            sampled_clip,
            if reduced_motion { 0.0 } else { self.time },
            previous,
            blend,
            &mut self.local,
        )?;
        for ((controlled, local), delta) in self
            .controlled_local
            .iter_mut()
            .zip(&self.local)
            .zip(&self.joint_deltas)
        {
            *controlled = local.mul(delta);
        }
        compose_palette_into(
            &self.joints,
            &self.controlled_local,
            self.root,
            &mut self.globals,
            &mut self.palette,
        )?;
        for ((gpu_slot, curr), prev) in self
            .gpu_palette
            .iter_mut()
            .zip(&self.palette)
            .zip(&self.prev_skin_motors)
        {
            *gpu_slot = super::gpu_skin::GpuJointMotorPair::from_motors(curr, prev);
        }
        Ok(PlaybackReceipt {
            clip_id: clip.id,
            time: self.time,
            vertices: self.positions.len(),
            joints: self.joints.len(),
            reduced_motion,
        })
    }

    /// Samples and deforms from immutable bind data, avoiding accumulated drift.
    /// Reduced motion selects the authored first pose and disables transitions.
    pub fn advance(
        &mut self,
        dt_seconds: f64,
        reduced_motion: bool,
    ) -> Result<PlaybackReceipt, PlayerError> {
        self.advance_with_policy(dt_seconds, reduced_motion, PoseTimePolicy::AuthoredLooping)
    }
    /// The same policy is used for pose evaluation and skinning. Hosts with a
    /// separate contact pass must select it consistently for both phases.
    pub fn advance_with_policy(
        &mut self,
        dt_seconds: f64,
        reduced_motion: bool,
        policy: PoseTimePolicy,
    ) -> Result<PlaybackReceipt, PlayerError> {
        let receipt = self.advance_pose_only_with_policy(dt_seconds, reduced_motion, policy)?;
        skin_vertices_into(
            &self.bind_positions,
            &self.bind_normals,
            &self.skin,
            &self.palette,
            &mut self.positions,
            &mut self.normals,
        )?;
        if let Some((fit, min, max)) = &self.body_fit {
            fit.apply_in_place(&mut self.positions, *min, *max);
            crate::surface_frames::surface::vertex_normals_from_indices_into(
                &self.positions,
                &self.indices,
                &mut self.normals,
            )
            .map_err(|_| PlayerError::InvalidInput)?;
        }
        for point in &mut self.positions {
            for (axis, v) in point.iter_mut().enumerate() {
                *v = (*v - self.origin[axis]) * self.scale;
            }
        }
        crate::surface_frames::tangent::tangents_from_uvs_into(
            &self.positions,
            &self.uv,
            &self.normals,
            &self.indices,
            &mut self.tangents,
            &mut self.tangent_scratch,
        )
        .map_err(|_| PlayerError::Tangents)?;
        Ok(receipt)
    }

    /// Current GPU dual-quaternion skinning motor palette (`current_r/d` + `prev_r/d`).
    pub fn gpu_motor_palette(&self) -> &[super::gpu_skin::GpuJointMotorPair] {
        &self.gpu_palette
    }

    /// Current world-space joint `Motor`s.
    pub fn world_motors(&self) -> &[Motor] {
        &self.globals
    }

    /// Current skinning `Motor` palette ($M_{\text{world}} \circ M_{\text{inv\_bind}}$).
    pub fn skinning_palette(&self) -> &[Motor] {
        &self.palette
    }

    /// Previous frame's skinning `Motor` palette.
    pub fn prev_skinning_palette(&self) -> &[Motor] {
        &self.prev_skin_motors
    }
    /// Attachment globals are in authored world space, as are rig roots.
    pub fn attachment(&self, joint_id: u64) -> Option<Motor> {
        self.joint_ids
            .iter()
            .position(|id| *id == joint_id)
            .map(|i| self.globals[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeletal::assert_zero_alloc;

    #[test]
    fn standalone_player_advances_pose_and_applies_body_fit_zero_heap() {
        let identity = Motor::identity();
        let joints = [Joint {
            parent: None,
            rest_local: identity,
            inverse_bind: identity,
        }];
        let joint_ids = [101u64];
        let skin = [SkinWeight {
            joints: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
        }; 3];
        let clips = [ClipRecord {
            id: 202,
            duration: 1.0,
            first_track: 0,
            track_count: 1,
            looping: false,
        }];
        let tracks = [Track {
            joint: 0,
            first_key: 0,
            key_count: 2,
        }];
        let keys = [
            Key {
                time: 0.0,
                local: identity,
            },
            Key {
                time: 1.0,
                local: Motor::from_translation(1.0, 0.0, 0.0),
            },
        ];
        let rig = SkeletalRigDescriptor {
            joints: &joints,
            joint_ids: &joint_ids,
            skin: &skin,
            clips: &clips,
            tracks: &tracks,
            keys: &keys,
        };
        let mut player = SkeletalPlayer::from_section(
            &rig,
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.8, 0.0]],
            &[[0.0, 0.0, 1.0]; 3],
            &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            &[0, 1, 2],
        )
        .unwrap();

        let fit = BodyFitSpec {
            stature_scale: 1.2,
            identity: false,
            ..BodyFitSpec::default()
        };
        player
            .set_body_fit(&fit, [0.0, 0.0, 0.0], [1.0, 1.8, 0.3])
            .unwrap();

        assert_zero_alloc("etalon_skeletal_player_advance", || {
            let receipt = player.advance(0.5, false).unwrap();
            assert_eq!(receipt.clip_id, 202);
            assert_eq!(receipt.vertices, 3);
            assert_eq!(receipt.joints, 1);
        });
        assert!((player.positions()[2][1] - 2.16).abs() < 1e-3);
    }
}
