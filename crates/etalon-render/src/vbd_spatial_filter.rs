//! GPU swept surface broad phase with resident face and edge BVHs.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use etalon::{util::DeviceExt, BufferUsages};

const WORKGROUP_SIZE: usize = 64;
const MAX_PRIMITIVES: usize = 32_768;
const MAX_CANDIDATES_PER_QUERY: usize = 256;
const MAX_TRAVERSAL_VISITS: usize = 64 * 1024 * 1024;
const AABB_BYTES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VbdContactKind {
    VertexFace,
    EdgeEdge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VbdContactCandidate {
    pub kind: VbdContactKind,
    pub source: u32,
    pub target: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VbdCandidateCounts {
    pub vertex_face: usize,
    pub edge_edge: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum VbdSpatialFilterError {
    #[error("invalid VBD surface topology, motion, margin, or staging buffer")]
    InvalidInput,
    #[error("VBD spatial filter exceeds its traversal, candidate, or byte budget")]
    BudgetExceeded,
    #[error("Etalon device limits do not admit the VBD spatial filter")]
    CapabilityUnavailable,
    #[error("Etalon did not start the VBD spatial-filter submission")]
    SubmissionRejected,
    #[error("VBD candidate row overflow for {kind:?} source {source_id}")]
    CandidateOverflow {
        kind: VbdContactKind,
        source_id: u32,
    },
    #[error("invalid VBD GPU candidate readback")]
    InvalidReadback,
    #[error("VBD candidate output requires {needed} entries")]
    OutputTooSmall { needed: usize },
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DispatchParams {
    vertex_count: u32,
    face_count: u32,
    edge_count: u32,
    face_capacity: u32,
    edge_capacity: u32,
    row_capacity: u32,
    level: u32,
    margin_m: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VertexMotion {
    before: [f32; 4],
    after: [f32; 4],
}

const _: [(); 32] = [(); std::mem::size_of::<DispatchParams>()];
const _: [(); 32] = [(); std::mem::size_of::<VertexMotion>()];

pub struct VbdSpatialFilter {
    device: etalon::Device,
    queue: etalon::Queue,
    leaves: etalon::ComputePipeline,
    refit: etalon::ComputePipeline,
    queries: etalon::ComputePipeline,
    group: etalon::BindGroup,
    uniform: etalon::Buffer,
    positions: etalon::Buffer,
    candidates: etalon::Buffer,
    params: DispatchParams,
    uniform_stride: usize,
    depth: usize,
    row_bytes: usize,
    readback_bytes: usize,
    motion_upload: Vec<VertexMotion>,
    uniform_upload: Vec<u8>,
}

impl VbdSpatialFilter {
    /// Build the resident topology from a reference pose. Candidate IDs retain
    /// the caller's vertex, face and edge ordering after spatial permutation.
    pub fn new(
        device: &etalon::Device,
        queue: &etalon::Queue,
        reference: &[[f32; 3]],
        faces: &[[u32; 3]],
        edges: &[[u32; 2]],
        max_candidates_per_query: usize,
        byte_budget: usize,
    ) -> Result<Self, VbdSpatialFilterError> {
        use VbdSpatialFilterError as Error;
        if reference.is_empty()
            || (faces.is_empty() && edges.is_empty())
            || max_candidates_per_query == 0
        {
            return Err(Error::InvalidInput);
        }
        if [reference.len(), faces.len(), edges.len()]
            .into_iter()
            .any(|count| count > MAX_PRIMITIVES)
            || max_candidates_per_query > MAX_CANDIDATES_PER_QUERY
        {
            return Err(Error::BudgetExceeded);
        }
        if !valid_positions(reference)
            || faces.iter().any(|&[a, b, c]| {
                a == b
                    || b == c
                    || a == c
                    || [a, b, c]
                        .into_iter()
                        .any(|index| index as usize >= reference.len())
            })
            || edges.iter().any(|&[a, b]| {
                a == b
                    || [a, b]
                        .into_iter()
                        .any(|index| index as usize >= reference.len())
            })
        {
            return Err(Error::InvalidInput);
        }
        let face_capacity = faces.len().max(1).next_power_of_two();
        let edge_capacity = edges.len().max(1).next_power_of_two();
        let vertices = reference.len();
        let query_count = vertices + edges.len();
        let leaves_count = face_capacity + edge_capacity;
        let depth = face_capacity.max(edge_capacity).ilog2() as usize;
        let visits = vertices * (2 * face_capacity - 1) + edges.len() * (2 * edge_capacity - 1);
        if visits > MAX_TRAVERSAL_VISITS {
            return Err(Error::BudgetExceeded);
        }

        let limits = device.limits();
        let alignment = limits.min_uniform_buffer_offset_alignment.max(1) as usize;
        let uniform_bytes = std::mem::size_of::<DispatchParams>();
        let uniform_stride = uniform_bytes.div_ceil(alignment) * alignment;
        let uniform_size = (depth + 1)
            .checked_mul(uniform_stride)
            .ok_or(Error::CapabilityUnavailable)?;
        let row_bytes = (max_candidates_per_query + 1) * std::mem::size_of::<u32>();
        let readback_bytes = query_count * row_bytes;
        let motion_bytes = vertices * std::mem::size_of::<VertexMotion>();
        let storage = [
            motion_bytes,
            (faces.len() + edges.len()) * 16,
            (faces.len() + edges.len()) * 4,
            2 * leaves_count * AABB_BYTES,
            readback_bytes,
        ];
        let topology_scratch = storage[1] + 3 * storage[2] + edges.len() * 8;
        let bytes_used = (2usize)
            .checked_mul(uniform_size)
            .and_then(|params| {
                params.checked_add(
                    storage.into_iter().sum::<usize>()
                        + readback_bytes
                        + motion_bytes
                        + topology_scratch,
                )
            })
            .ok_or(Error::BudgetExceeded)?;
        if bytes_used > byte_budget {
            return Err(Error::BudgetExceeded);
        }
        if limits.max_storage_buffers_per_shader_stage < 5
            || limits.max_bindings_per_bind_group < 6
            || limits.max_bind_groups < 1
            || limits.max_dynamic_uniform_buffers_per_pipeline_layout < 1
            || (limits.max_uniform_buffer_binding_size as usize) < uniform_bytes
            || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE as u32
            || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE as u32
            || limits.max_compute_workgroups_per_dimension < groups(query_count.max(leaves_count))
            || uniform_size as u64 > limits.max_buffer_size
            || uniform_size as u64 > u64::from(u32::MAX)
            || storage.into_iter().any(|size| {
                size as u64 > limits.max_buffer_size
                    || size as u64 > limits.max_storage_buffer_binding_size
            })
        {
            return Err(Error::CapabilityUnavailable);
        }

        // Multiple identical edges assign duplicate physical contact IDs.
        let mut unique_edges: Vec<_> = edges.iter().map(|&[a, b]| [a.min(b), a.max(b)]).collect();
        unique_edges.sort_unstable();
        if unique_edges.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidInput);
        }
        let topology: Vec<_> = faces
            .iter()
            .map(|&[a, b, c]| [a, b, c, 0])
            .chain(edges.iter().map(|&[a, b]| [a, b, 0, 0]))
            .collect();
        let order = leaf_order(reference, faces, edges);
        let uniform = buffer(
            device,
            "vbd-spatial-params",
            uniform_size,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let positions = buffer(
            device,
            "vbd-swept-positions",
            storage[0],
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let nodes = buffer(device, "vbd-dynamic-bvh", storage[3], BufferUsages::STORAGE);
        let candidates = buffer(
            device,
            "vbd-contact-candidates",
            storage[4],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let topology = initialized(
            device,
            "vbd-surface-topology",
            bytemuck::cast_slice(&topology),
        );
        let order = initialized(device, "vbd-bvh-leaf-order", bytemuck::cast_slice(&order));
        let uniform_minimum = NonZeroU64::new(uniform_bytes as u64);
        let layouts = [
            etalon::BindGroupLayoutEntry {
                binding: 0,
                visibility: etalon::ShaderStages::COMPUTE,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: uniform_minimum,
                },
                count: None,
            },
            storage_layout(1, true),
            storage_layout(2, true),
            storage_layout(3, true),
            storage_layout(4, false),
            storage_layout(5, false),
        ];
        let layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("vbd-spatial-layout"),
            entries: &layouts,
        });
        let group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("vbd-spatial-bindings"),
            layout: &layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: etalon::BindingResource::Buffer(etalon::BufferBinding {
                        buffer: &uniform,
                        offset: 0,
                        size: uniform_minimum,
                    }),
                },
                binding(1, &positions),
                binding(2, &topology),
                binding(3, &order),
                binding(4, &nodes),
                binding(5, &candidates),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("vbd-spatial-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("vbd-spatial-shader"),
            source: etalon::ShaderSource::Wgsl(
                include_str!("shaders/vbd_spatial_filter.wgsl").into(),
            ),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            leaves: pipeline("refit_leaves"),
            refit: pipeline("refit_level"),
            queries: pipeline("collect_contacts"),
            group,
            uniform,
            positions,
            candidates,
            params: DispatchParams {
                vertex_count: vertices as u32,
                face_count: faces.len() as u32,
                edge_count: edges.len() as u32,
                face_capacity: face_capacity as u32,
                edge_capacity: edge_capacity as u32,
                row_capacity: max_candidates_per_query as u32,
                level: 0,
                margin_m: 0.0,
            },
            uniform_stride,
            depth,
            row_bytes,
            readback_bytes,
            motion_upload: vec![VertexMotion::zeroed(); vertices],
            uniform_upload: vec![0; uniform_size],
        })
    }

    pub const fn readback_bytes(&self) -> usize {
        self.readback_bytes
    }

    /// GPU storage rows for the latest submission. Vertex rows precede edge
    /// rows. Each row has a count and `capacity` authored target IDs. Ignore
    /// unused IDs and treat `capacity + 1` as an overflow marker.
    pub fn candidate_rows(&self) -> &etalon::Buffer {
        &self.candidates
    }

    /// Vertices, edges, and target capacity per row for the storage view.
    pub fn candidate_dimensions(&self) -> [u32; 3] {
        [
            self.params.vertex_count,
            self.params.edge_count,
            self.params.row_capacity,
        ]
    }

    pub fn create_readback_buffer(&self) -> etalon::Buffer {
        buffer(
            &self.device,
            "vbd-candidate-readback",
            self.readback_bytes,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )
    }

    /// Submit in queue order so each refit consumes its own positions and
    /// parameters before a later frame can update the resident buffers.
    pub fn refit_and_dispatch(
        &mut self,
        previous: &[[f32; 3]],
        current: &[[f32; 3]],
        margin_m: f32,
        staging: Option<&etalon::Buffer>,
    ) -> Result<etalon::SubmissionIndex, VbdSpatialFilterError> {
        if previous.len() != self.motion_upload.len()
            || current.len() != self.motion_upload.len()
            || !valid_positions(previous)
            || !valid_positions(current)
            || !margin_m.is_finite()
            || !(0.0..=1e6).contains(&margin_m)
            || staging.is_some_and(|buffer| {
                buffer.size() < self.readback_bytes as u64
                    || !buffer.usage().contains(BufferUsages::COPY_DST)
            })
        {
            return Err(VbdSpatialFilterError::InvalidInput);
        }
        for ((entry, &[x, y, z]), &[a, b, c]) in
            self.motion_upload.iter_mut().zip(previous).zip(current)
        {
            *entry = VertexMotion {
                before: [x, y, z, 0.0],
                after: [a, b, c, 0.0],
            };
        }
        for (level, dst) in self
            .uniform_upload
            .chunks_exact_mut(self.uniform_stride)
            .enumerate()
        {
            let params = DispatchParams {
                level: level as u32,
                margin_m,
                ..self.params
            };
            let bytes = bytemuck::bytes_of(&params);
            dst[..bytes.len()].copy_from_slice(bytes);
        }
        self.queue.write_buffer(
            &self.positions,
            0,
            bytemuck::cast_slice(&self.motion_upload),
        );
        self.queue
            .write_buffer(&self.uniform, 0, &self.uniform_upload);
        let mut encoder = self
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor {
                label: Some("vbd-spatial-filter"),
            });
        let faces = self.params.face_capacity as usize;
        let edges = self.params.edge_capacity as usize;
        dispatch(&mut encoder, &self.group, &self.leaves, 0, faces + edges);
        for level in 1..=self.depth {
            let nodes = (faces >> level) + (edges >> level);
            dispatch(
                &mut encoder,
                &self.group,
                &self.refit,
                (self.uniform_stride * level) as u32,
                nodes,
            );
        }
        let queries = self.params.vertex_count as usize + self.params.edge_count as usize;
        dispatch(&mut encoder, &self.group, &self.queries, 0, queries);
        if let Some(staging) = staging {
            encoder.copy_buffer_to_buffer(
                &self.candidates,
                0,
                staging,
                0,
                self.readback_bytes as u64,
            );
        }
        self.queue
            .submit([encoder.finish()])
            .ok_or(VbdSpatialFilterError::SubmissionRejected)
    }

    /// Validate the complete fixed-stride readback before publishing any pairs.
    pub fn decode_candidates_into(
        &self,
        mapped: &[u8],
        out: &mut [VbdContactCandidate],
    ) -> Result<VbdCandidateCounts, VbdSpatialFilterError> {
        use VbdSpatialFilterError as Error;
        if mapped.len() != self.readback_bytes {
            return Err(Error::InvalidReadback);
        }
        let mut counts = VbdCandidateCounts::default();
        for (index, row) in mapped.chunks_exact(self.row_bytes).enumerate() {
            let (kind, source_id, target_count) = self.query_identity(index as u32);
            let count = read_word(row) as usize;
            let capacity = self.params.row_capacity as usize;
            if count == capacity + 1 {
                return Err(Error::CandidateOverflow { kind, source_id });
            }
            if count > capacity
                || row[4..4 + count * 4]
                    .chunks_exact(4)
                    .map(read_word)
                    .any(|target| {
                        target >= target_count
                            || (kind == VbdContactKind::EdgeEdge && target <= source_id)
                    })
            {
                return Err(Error::InvalidReadback);
            }
            match kind {
                VbdContactKind::VertexFace => counts.vertex_face += count,
                VbdContactKind::EdgeEdge => counts.edge_edge += count,
            }
        }
        let needed = counts.vertex_face + counts.edge_edge;
        if out.len() < needed {
            return Err(Error::OutputTooSmall { needed });
        }
        let mut write = 0;
        for (index, row) in mapped.chunks_exact(self.row_bytes).enumerate() {
            let (kind, source, _) = self.query_identity(index as u32);
            let count = read_word(row) as usize;
            for target in row[4..4 + count * 4].chunks_exact(4).map(read_word) {
                out[write] = VbdContactCandidate {
                    kind,
                    source,
                    target,
                };
                write += 1;
            }
        }
        Ok(counts)
    }

    fn query_identity(&self, index: u32) -> (VbdContactKind, u32, u32) {
        if index < self.params.vertex_count {
            (VbdContactKind::VertexFace, index, self.params.face_count)
        } else {
            (
                VbdContactKind::EdgeEdge,
                index - self.params.vertex_count,
                self.params.edge_count,
            )
        }
    }
}

fn valid_positions(positions: &[[f32; 3]]) -> bool {
    positions
        .iter()
        .flatten()
        .all(|v| v.is_finite() && v.abs() <= 1e8)
}

fn leaf_order(reference: &[[f32; 3]], faces: &[[u32; 3]], edges: &[[u32; 2]]) -> Vec<u32> {
    let (min, max) = reference
        .iter()
        .fold((reference[0], reference[0]), |(min, max), p| {
            (
                std::array::from_fn(|lane| min[lane].min(p[lane])),
                std::array::from_fn(|lane| max[lane].max(p[lane])),
            )
        });
    let mut keyed = Vec::with_capacity(faces.len() + edges.len());
    let point = |id: u32, indices: &[u32]| {
        let centre: [f32; 3] = std::array::from_fn(|axis| {
            indices
                .iter()
                .map(|&vertex| reference[vertex as usize][axis])
                .sum::<f32>()
                / indices.len() as f32
        });
        let [x, y, z] = std::array::from_fn(|axis| {
            let span = max[axis] - min[axis];
            let ratio = if span == 0.0 {
                0.0
            } else {
                (centre[axis] - min[axis]) / span
            };
            spread((ratio.clamp(0.0, 1.0) * 1023.0) as u32)
        });
        (x | (y << 1) | (z << 2), id)
    };
    keyed.extend(
        faces
            .iter()
            .enumerate()
            .map(|(id, face)| point(id as u32, face)),
    );
    keyed.extend(
        edges
            .iter()
            .enumerate()
            .map(|(id, edge)| point(id as u32, edge)),
    );
    keyed[..faces.len()].sort_unstable();
    keyed[faces.len()..].sort_unstable();
    keyed.into_iter().map(|(_, id)| id).collect()
}

fn spread(mut value: u32) -> u32 {
    value = (value | (value << 16)) & 0x0300_00ff;
    value = (value | (value << 8)) & 0x0300_f00f;
    value = (value | (value << 4)) & 0x030c_30c3;
    (value | (value << 2)) & 0x0924_9249
}

fn read_word(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn buffer(
    device: &etalon::Device,
    label: &str,
    bytes: usize,
    usage: BufferUsages,
) -> etalon::Buffer {
    device.create_buffer(&etalon::BufferDescriptor {
        label: Some(label),
        size: bytes as u64,
        usage,
        mapped_at_creation: false,
    })
}

fn initialized(device: &etalon::Device, label: &str, bytes: &[u8]) -> etalon::Buffer {
    device.create_buffer_init(&etalon::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: BufferUsages::STORAGE,
    })
}

fn storage_layout(binding: u32, read_only: bool) -> etalon::BindGroupLayoutEntry {
    etalon::BindGroupLayoutEntry {
        binding,
        visibility: etalon::ShaderStages::COMPUTE,
        ty: etalon::BindingType::Buffer {
            ty: etalon::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn binding(binding: u32, buffer: &etalon::Buffer) -> etalon::BindGroupEntry<'_> {
    etalon::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn groups(items: usize) -> u32 {
    items.div_ceil(WORKGROUP_SIZE) as u32
}

fn dispatch(
    encoder: &mut etalon::CommandEncoder,
    group: &etalon::BindGroup,
    pipeline: &etalon::ComputePipeline,
    offset: u32,
    lanes: usize,
) {
    let mut pass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
        label: Some("vbd-spatial-stage"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[offset]);
    pass.dispatch_workgroups(groups(lanes), 1, 1);
}
