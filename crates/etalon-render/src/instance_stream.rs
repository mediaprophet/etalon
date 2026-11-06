//! Bounded per-instance transform storage for indexed mesh batches.

use crate::instance_culling::GpuInstanceRecord;
use etalon::util::DeviceExt;

pub const MAX_MESH_INSTANCES: usize = crate::instance_culling::MAX_GPU_MESH_INSTANCES;
pub const MAX_COALESCED_DIRTY_RANGES: usize = 32;
const DIRTY_RANGE_SCRATCH_CAP: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirtyInstanceRange {
    pub start: u32,
    pub count: u32,
}

impl DirtyInstanceRange {
    /// Creates a dirty instance range starting at `start` covering `count` records.
    #[inline]
    pub const fn new(start: u32, count: u32) -> Self {
        Self { start, count }
    }

    /// Checked exclusive end index (`start + count`).
    #[inline]
    pub const fn end(self) -> Option<u32> {
        self.start.checked_add(self.count)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyUploadPolicy {
    /// Merge adjacent dirty ranges separated by at most `max_gap_records` clean records.
    pub max_gap_records: u32,
    /// When coalesced dirty records * 100 >= total_records * full_upload_percent, emit a single full upload.
    pub full_upload_percent: u8,
}

impl Default for DirtyUploadPolicy {
    fn default() -> Self {
        Self {
            max_gap_records: 4,
            full_upload_percent: 65,
        }
    }
}

/// Validates and coalesces `ranges` into `out` in deterministic ascending order without heap allocation.
///
/// - Rejects `total_records > MAX_MESH_INSTANCES` with [`MeshInstanceUploadError::TooManyInstances`].
/// - Rejects any range where `start + count > total_records` (or `u32` overflow) with
///   [`MeshInstanceUploadError::BufferSizeOverflow`].
/// - Ignores empty ranges (`count == 0`).
/// - Sorts and merges overlapping or nearby ranges (`next.start <= cur_end + policy.max_gap_records`)
///   on a fixed stack buffer.
/// - Collapses into a single `[0..total_records]` upload when the coalesced range count exceeds
///   [`MAX_COALESCED_DIRTY_RANGES`] or when `covered_records * 100 >= total_records * policy.full_upload_percent`.
pub fn coalesce_dirty_ranges(
    ranges: &[DirtyInstanceRange],
    total_records: u32,
    policy: DirtyUploadPolicy,
    out: &mut [DirtyInstanceRange; MAX_COALESCED_DIRTY_RANGES],
) -> Result<usize, MeshInstanceUploadError> {
    if (total_records as usize) > MAX_MESH_INSTANCES {
        return Err(MeshInstanceUploadError::TooManyInstances {
            count: total_records as usize,
            maximum: MAX_MESH_INSTANCES,
        });
    }
    for range in ranges {
        let end = range
            .end()
            .ok_or(MeshInstanceUploadError::BufferSizeOverflow)?;
        if end > total_records {
            return Err(MeshInstanceUploadError::BufferSizeOverflow);
        }
    }
    if total_records == 0 {
        return Ok(0);
    }

    let mut scratch = [DirtyInstanceRange::new(0, 0); DIRTY_RANGE_SCRATCH_CAP];
    let mut len = 0usize;
    let mut overflowed_scratch = false;
    let gap = policy.max_gap_records as u64;

    for &range in ranges {
        if range.count == 0 {
            continue;
        }
        if len == DIRTY_RANGE_SCRATCH_CAP {
            overflowed_scratch = true;
            break;
        }
        // Binary search insertion point by `start` to maintain sorted order on the stack.
        let mut lo = 0usize;
        let mut hi = len;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if scratch[mid].start <= range.start {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut idx = len;
        while idx > lo {
            scratch[idx] = scratch[idx - 1];
            idx -= 1;
        }
        scratch[lo] = range;
        len += 1;

        // Coalesce overlapping and gap-adjacent intervals in-place.
        let mut write = 0usize;
        for read in 1..len {
            let cur = scratch[write];
            let next = scratch[read];
            let cur_end = cur.start as u64 + cur.count as u64;
            let next_start = next.start as u64;
            let next_end = next_start + next.count as u64;
            if next_start <= cur_end.saturating_add(gap) {
                let merged_end = cur_end.max(next_end) as u32;
                scratch[write] = DirtyInstanceRange {
                    start: cur.start,
                    count: merged_end - cur.start,
                };
            } else {
                write += 1;
                scratch[write] = next;
            }
        }
        len = write + 1;
    }

    if !overflowed_scratch && len == 0 {
        return Ok(0);
    }
    if overflowed_scratch || len > MAX_COALESCED_DIRTY_RANGES {
        out[0] = DirtyInstanceRange {
            start: 0,
            count: total_records,
        };
        return Ok(1);
    }

    let covered_records: u64 = scratch[..len].iter().map(|r| r.count as u64).sum();
    if covered_records.saturating_mul(100)
        >= (total_records as u64).saturating_mul(policy.full_upload_percent as u64)
    {
        out[0] = DirtyInstanceRange {
            start: 0,
            count: total_records,
        };
        return Ok(1);
    }

    out[..len].copy_from_slice(&scratch[..len]);
    Ok(len)
}

/// Validates [`supports_affine_transform`] only for records inside `coalesced_ranges`.
pub fn validate_dirty_ranges_packed(
    packed_records: &[u8],
    coalesced_ranges: &[DirtyInstanceRange],
) -> Result<(), MeshInstanceUploadError> {
    let record_size = std::mem::size_of::<GpuInstanceRecord>();
    if packed_records.len() % record_size != 0 {
        return Err(MeshInstanceUploadError::InvalidTransform { index: 0 });
    }
    let records = packed_records.len() / record_size;
    if records > MAX_MESH_INSTANCES {
        return Err(MeshInstanceUploadError::TooManyInstances {
            count: records,
            maximum: MAX_MESH_INSTANCES,
        });
    }
    let total_records =
        u32::try_from(records).map_err(|_| MeshInstanceUploadError::BufferSizeOverflow)?;
    for range in coalesced_ranges {
        let end = range
            .end()
            .ok_or(MeshInstanceUploadError::BufferSizeOverflow)?;
        if end > total_records {
            return Err(MeshInstanceUploadError::BufferSizeOverflow);
        }
        for index in (range.start as usize)..(end as usize) {
            let byte_offset = index * record_size;
            let bytes = &packed_records[byte_offset..byte_offset + record_size];
            let record = bytemuck::pod_read_unaligned::<GpuInstanceRecord>(bytes);
            if !supports_affine_transform(&record) {
                return Err(MeshInstanceUploadError::InvalidTransform { index });
            }
        }
    }
    Ok(())
}
const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
const IDENTITY3: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];
const DEFAULT_INSTANCE: GpuInstanceRecord = GpuInstanceRecord {
    world_from_local: IDENTITY,
    normal_from_local: IDENTITY3,
    semantic_id_words: [0; 2],
    orientation_sign: 1.0,
    _padding: 0,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshInstanceUploadError {
    TooManyInstances { count: usize, maximum: usize },
    InvalidTransform { index: usize },
    TransparentInstancesRequireSortedSubmission,
    ReservationRefused,
    BufferSizeOverflow,
}

pub struct GpuInstanceStream<R> {
    buffer: etalon::Buffer,
    bind_group: etalon::BindGroup,
    _reservation: R,
    capacity: usize,
    count: u32,
}

impl<R> GpuInstanceStream<R> {
    /// Creates with the given state.
    pub fn new(
        device: &etalon::Device,
        layout: &etalon::BindGroupLayout,
        reserve: impl FnOnce(u64) -> Option<R>,
    ) -> Result<Self, MeshInstanceUploadError> {
        let bytes = std::mem::size_of::<GpuInstanceRecord>() as u64;
        let reservation = reserve(bytes).ok_or(MeshInstanceUploadError::ReservationRefused)?;
        let buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("portal-mesh-instances"),
            contents: bytemuck::bytes_of(&DEFAULT_INSTANCE),
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
        });
        let bind_group = bind_group(device, layout, &buffer);
        Ok(Self {
            buffer,
            bind_group,
            _reservation: reservation,
            capacity: 1,
            count: 1,
        })
    }

    /// `replace_packed`.
    pub fn replace_packed(
        &mut self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        layout: &etalon::BindGroupLayout,
        packed_records: &[u8],
        reserve: impl FnOnce(u64) -> Option<R>,
    ) -> Result<(), MeshInstanceUploadError> {
        let record_size = std::mem::size_of::<GpuInstanceRecord>();
        if packed_records.len() % record_size != 0 {
            return Err(MeshInstanceUploadError::InvalidTransform { index: 0 });
        }
        let records = packed_records.len() / record_size;
        if records > MAX_MESH_INSTANCES {
            return Err(MeshInstanceUploadError::TooManyInstances {
                count: records,
                maximum: MAX_MESH_INSTANCES,
            });
        }
        for (index, bytes) in packed_records.chunks_exact(record_size).enumerate() {
            let record = bytemuck::pod_read_unaligned::<GpuInstanceRecord>(bytes);
            if !supports_affine_transform(&record) {
                return Err(MeshInstanceUploadError::InvalidTransform { index });
            }
        }

        let (upload, count) = if records == 0 {
            (bytemuck::bytes_of(&DEFAULT_INSTANCE), 1_u32)
        } else {
            (
                packed_records,
                u32::try_from(records).map_err(|_| MeshInstanceUploadError::BufferSizeOverflow)?,
            )
        };
        let required = upload.len() / record_size;
        if required <= self.capacity {
            queue.write_buffer(&self.buffer, 0, upload);
            self.count = count;
            return Ok(());
        }

        // Grow geometrically but never beyond the declared 10k-instance workload. Reserve the
        // replacement while the old stream remains valid; if admission fails, current draws stay
        // intact and the caller receives a deterministic refusal.
        let capacity = required
            .checked_next_power_of_two()
            .unwrap_or(MAX_MESH_INSTANCES)
            .min(MAX_MESH_INSTANCES);
        let buffer_bytes = (capacity as u64)
            .checked_mul(record_size as u64)
            .ok_or(MeshInstanceUploadError::BufferSizeOverflow)?;
        let reservation =
            reserve(buffer_bytes).ok_or(MeshInstanceUploadError::ReservationRefused)?;
        let buffer = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("portal-mesh-instances-grown"),
            size: buffer_bytes,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, upload);
        let bind_group = bind_group(device, layout, &buffer);
        self.buffer = buffer;
        self.bind_group = bind_group;
        self._reservation = reservation;
        self.capacity = capacity;
        self.count = count;
        Ok(())
    }

    /// Replace the camera-visible stream. Unlike the source stream, zero visible records means
    /// no indexed instances should be drawn (an empty source stream still means legacy identity).
    pub fn replace_culled_packed(
        &mut self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        layout: &etalon::BindGroupLayout,
        packed_records: &[u8],
        reserve: impl FnOnce(u64) -> Option<R>,
    ) -> Result<(), MeshInstanceUploadError> {
        if packed_records.is_empty() {
            self.count = 0;
            return Ok(());
        }
        self.replace_packed(device, queue, layout, packed_records, reserve)
    }

    /// Updates only the coalesced dirty subranges of an already-sized instance stream.
    ///
    /// When `records > self.capacity` or `self.count != records as u32`, delegates to
    /// [`Self::replace_packed`] and returns `Ok(1)`. Otherwise, coalesces `dirty_ranges` via
    /// [`coalesce_dirty_ranges`], validates [`supports_affine_transform`] only for records inside
    /// the coalesced ranges, and issues one `queue.write_buffer` per coalesced range, returning
    /// the number of `write_buffer` calls issued.
    pub fn update_dirty_ranges_packed(
        &mut self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        layout: &etalon::BindGroupLayout,
        packed_records: &[u8],
        dirty_ranges: &[DirtyInstanceRange],
        policy: DirtyUploadPolicy,
        reserve: impl FnOnce(u64) -> Option<R>,
    ) -> Result<usize, MeshInstanceUploadError> {
        let record_size = std::mem::size_of::<GpuInstanceRecord>();
        if packed_records.len() % record_size != 0 {
            return Err(MeshInstanceUploadError::InvalidTransform { index: 0 });
        }
        let records = packed_records.len() / record_size;
        if records > MAX_MESH_INSTANCES {
            return Err(MeshInstanceUploadError::TooManyInstances {
                count: records,
                maximum: MAX_MESH_INSTANCES,
            });
        }
        let total_records =
            u32::try_from(records).map_err(|_| MeshInstanceUploadError::BufferSizeOverflow)?;

        if records > self.capacity || self.count != total_records {
            for range in dirty_ranges {
                let end = range
                    .end()
                    .ok_or(MeshInstanceUploadError::BufferSizeOverflow)?;
                if end > total_records {
                    return Err(MeshInstanceUploadError::BufferSizeOverflow);
                }
            }
            self.replace_packed(device, queue, layout, packed_records, reserve)?;
            return Ok(1);
        }

        let mut coalesced = [DirtyInstanceRange::new(0, 0); MAX_COALESCED_DIRTY_RANGES];
        let range_count =
            coalesce_dirty_ranges(dirty_ranges, total_records, policy, &mut coalesced)?;
        validate_dirty_ranges_packed(packed_records, &coalesced[..range_count])?;

        for range in &coalesced[..range_count] {
            let byte_offset = (range.start as usize)
                .checked_mul(record_size)
                .ok_or(MeshInstanceUploadError::BufferSizeOverflow)?;
            let byte_len = (range.count as usize)
                .checked_mul(record_size)
                .ok_or(MeshInstanceUploadError::BufferSizeOverflow)?;
            let span_bytes = &packed_records[byte_offset..byte_offset + byte_len];
            queue.write_buffer(&self.buffer, byte_offset as u64, span_bytes);
        }
        Ok(range_count)
    }

    /// `bind_group`.
    pub fn bind_group(&self) -> &etalon::BindGroup {
        &self.bind_group
    }

    /// `count`.
    pub fn count(&self) -> u32 {
        self.count
    }
}

fn bind_group(
    device: &etalon::Device,
    layout: &etalon::BindGroupLayout,
    buffer: &etalon::Buffer,
) -> etalon::BindGroup {
    device.create_bind_group(&etalon::BindGroupDescriptor {
        label: Some("portal-mesh-instance-bind"),
        layout,
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}

/// Reject transforms that cannot produce a stable, matching inverse-transpose normal matrix.
fn supports_affine_transform(record: &GpuInstanceRecord) -> bool {
    let matrix = record.world_from_local;
    if matrix.iter().flatten().any(|value| !value.is_finite())
        || matrix[0][3].abs() > 1.0e-5
        || matrix[1][3].abs() > 1.0e-5
        || matrix[2][3].abs() > 1.0e-5
        || (matrix[3][3] - 1.0).abs() > 1.0e-5
    {
        return false;
    }
    let columns = [
        [matrix[0][0], matrix[0][1], matrix[0][2]],
        [matrix[1][0], matrix[1][1], matrix[1][2]],
        [matrix[2][0], matrix[2][1], matrix[2][2]],
    ];
    let length = |column: [f32; 3]| {
        (column[0] * column[0] + column[1] * column[1] + column[2] * column[2]).sqrt()
    };
    let volume_scale = length(columns[0]) * length(columns[1]) * length(columns[2]);
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let cofactors = [
        cross(columns[1], columns[2]),
        cross(columns[2], columns[0]),
        cross(columns[0], columns[1]),
    ];
    let determinant = columns[0][0] * cofactors[0][0]
        + columns[0][1] * cofactors[0][1]
        + columns[0][2] * cofactors[0][2];
    if !determinant.is_finite()
        || !volume_scale.is_finite()
        || volume_scale == 0.0
        || determinant.abs() <= 1.0e-6 * volume_scale
    {
        return false;
    }
    let expected = GpuInstanceRecord::new(matrix, record.semantic_id());
    if !record
        .normal_from_local
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        || !record.orientation_sign.is_finite()
        || (record.orientation_sign - expected.orientation_sign).abs() > 1.0e-5
    {
        return false;
    }
    record
        .normal_from_local
        .iter()
        .flatten()
        .zip(expected.normal_from_local.iter().flatten())
        .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-4 * (1.0 + expected.abs()))
}

#[cfg(test)]
mod tests {
    use super::{supports_affine_transform, GpuInstanceRecord};

    #[test]
    fn accepts_rigid_and_nonuniform_affine_scale() {
        let mut transform = [
            [0.0, 2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [3.0, -4.0, 5.0, 1.0],
        ];
        assert!(supports_affine_transform(&GpuInstanceRecord::new(
            transform, 1
        )));
        transform[0][0] = f32::NAN;
        assert!(!supports_affine_transform(&GpuInstanceRecord::new(
            transform, 1
        )));
    }

    #[test]
    fn accepts_reflection_and_rejects_singular_scale() {
        let mut nonuniform = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert!(supports_affine_transform(&GpuInstanceRecord::new(
            nonuniform, 1
        )));
        nonuniform[0][0] = -1.0;
        nonuniform[1][1] = 1.0;
        assert!(supports_affine_transform(&GpuInstanceRecord::new(
            nonuniform, 1
        )));
        nonuniform[1][1] = 0.0;
        assert!(!supports_affine_transform(&GpuInstanceRecord::new(
            nonuniform, 1
        )));
    }

    #[test]
    fn rejects_a_forged_normal_frame() {
        let matrix = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut record = GpuInstanceRecord::new(matrix, 8);
        assert!((record.normal_from_local[0][0] - 0.5).abs() < 1.0e-6);
        assert!((record.normal_from_local[1][1] - 1.0 / 3.0).abs() < 1.0e-6);
        assert!((record.normal_from_local[2][2] - 0.25).abs() < 1.0e-6);
        assert!(supports_affine_transform(&record));
        record.normal_from_local[0][0] = 1.0;
        assert!(!supports_affine_transform(&record));
    }

    #[test]
    fn determinant_guard_is_scale_relative_for_small_instances() {
        let transform = [
            [1.0e-10, 0.0, 0.0, 0.0],
            [0.0, 1.0e-10, 0.0, 0.0],
            [0.0, 0.0, 1.0e-10, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let record = GpuInstanceRecord::new(transform, 9);
        assert!(supports_affine_transform(&record));
        assert!((record.normal_from_local[0][0] - 1.0e10).abs() < 2048.0);
    }

    #[test]
    fn coalesces_overlapping_and_gap_adjacent_dirty_ranges() {
        use super::{
            coalesce_dirty_ranges, DirtyInstanceRange, DirtyUploadPolicy,
            MAX_COALESCED_DIRTY_RANGES,
        };
        let mut out = [DirtyInstanceRange::new(0, 0); MAX_COALESCED_DIRTY_RANGES];
        let policy = DirtyUploadPolicy::default(); // max_gap_records = 4, full_upload_percent = 65
        let ranges = [
            DirtyInstanceRange::new(20, 4), // [20..24]
            DirtyInstanceRange::new(5, 0),  // empty, ignored
            DirtyInstanceRange::new(2, 5),  // [2..7]
            DirtyInstanceRange::new(6, 4),  // [6..10], overlaps [2..7] -> [2..10]
            DirtyInstanceRange::new(14, 2), // [14..16], gap from 10 is 4 <= 4 -> [2..16]
            DirtyInstanceRange::new(40, 3), // [40..43], gap from 24 is 16 > 4 -> separate
        ];
        // Note: [2..16] and [20..24] have gap 20 - 16 = 4 <= 4 -> merges into [2..24] (count 22)!
        let count = coalesce_dirty_ranges(&ranges, 100, policy, &mut out).unwrap();
        assert_eq!(count, 2);
        assert_eq!(out[0], DirtyInstanceRange::new(2, 22));
        assert_eq!(out[1], DirtyInstanceRange::new(40, 3));
    }

    #[test]
    fn rejects_out_of_bounds_and_overflowing_dirty_ranges() {
        use super::{
            coalesce_dirty_ranges, DirtyInstanceRange, DirtyUploadPolicy, MeshInstanceUploadError,
            MAX_COALESCED_DIRTY_RANGES, MAX_MESH_INSTANCES,
        };
        let mut out = [DirtyInstanceRange::new(0, 0); MAX_COALESCED_DIRTY_RANGES];
        let policy = DirtyUploadPolicy::default();
        assert_eq!(
            coalesce_dirty_ranges(&[DirtyInstanceRange::new(96, 5)], 100, policy, &mut out),
            Err(MeshInstanceUploadError::BufferSizeOverflow)
        );
        assert_eq!(
            coalesce_dirty_ranges(
                &[DirtyInstanceRange::new(u32::MAX, 2)],
                100,
                policy,
                &mut out
            ),
            Err(MeshInstanceUploadError::BufferSizeOverflow)
        );
        assert_eq!(
            coalesce_dirty_ranges(
                &[],
                (MAX_MESH_INSTANCES + 1) as u32,
                policy,
                &mut out
            ),
            Err(MeshInstanceUploadError::TooManyInstances {
                count: MAX_MESH_INSTANCES + 1,
                maximum: MAX_MESH_INSTANCES,
            })
        );
    }

    #[test]
    fn collapses_to_full_upload_on_threshold_or_range_count_exceeded() {
        use super::{
            coalesce_dirty_ranges, DirtyInstanceRange, DirtyUploadPolicy,
            MAX_COALESCED_DIRTY_RANGES,
        };
        let mut out = [DirtyInstanceRange::new(0, 0); MAX_COALESCED_DIRTY_RANGES];
        let policy = DirtyUploadPolicy::default(); // 65%
        // 65 / 100 == 65% -> collapses to [0..100]
        let count = coalesce_dirty_ranges(
            &[
                DirtyInstanceRange::new(10, 30),
                DirtyInstanceRange::new(50, 35),
            ],
            100,
            policy,
            &mut out,
        )
        .unwrap();
        assert_eq!(count, 1);
        assert_eq!(out[0], DirtyInstanceRange::new(0, 100));

        // 33 disjoint 1-record ranges with gap 5 > 4 -> exceeds MAX_COALESCED_DIRTY_RANGES (32),
        // collapses to [0..500] even though covered records (33/500 = 6.6%) < 65%.
        let mut many = [DirtyInstanceRange::new(0, 0); 33];
        for (i, slot) in many.iter_mut().enumerate() {
            *slot = DirtyInstanceRange::new((i as u32) * 10, 1);
        }
        let count = coalesce_dirty_ranges(&many, 500, policy, &mut out).unwrap();
        assert_eq!(count, 1);
        assert_eq!(out[0], DirtyInstanceRange::new(0, 500));
    }

    #[test]
    fn validates_transforms_only_within_coalesced_dirty_subranges() {
        use super::{
            coalesce_dirty_ranges, validate_dirty_ranges_packed, DirtyInstanceRange,
            DirtyUploadPolicy, MeshInstanceUploadError, DEFAULT_INSTANCE,
            MAX_COALESCED_DIRTY_RANGES,
        };
        let mut records = [DEFAULT_INSTANCE; 10];
        // Corrupt record 7 (outside dirty range [1..3])
        records[7].world_from_local[0][0] = 0.0;
        let packed: &[u8] = bytemuck::cast_slice(&records);

        let mut coalesced = [DirtyInstanceRange::new(0, 0); MAX_COALESCED_DIRTY_RANGES];
        let n = coalesce_dirty_ranges(
            &[DirtyInstanceRange::new(1, 2)],
            10,
            DirtyUploadPolicy::default(),
            &mut coalesced,
        )
        .unwrap();
        assert_eq!(n, 1);
        assert_eq!(
            validate_dirty_ranges_packed(packed, &coalesced[..n]),
            Ok(())
        );

        // Including record 7 in a dirty subrange catches the invalid transform at index 7.
        let n = coalesce_dirty_ranges(
            &[
                DirtyInstanceRange::new(1, 2),
                DirtyInstanceRange::new(6, 3),
            ],
            10,
            DirtyUploadPolicy::default(),
            &mut coalesced,
        )
        .unwrap();
        assert_eq!(
            validate_dirty_ranges_packed(packed, &coalesced[..n]),
            Err(MeshInstanceUploadError::InvalidTransform { index: 7 })
        );
    }
}
