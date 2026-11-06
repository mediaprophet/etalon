use super::*;
/// Unified GPU mesh & multi-modal `.10d` asset residency arena.
pub struct MeshSlabArenaGpu<R, P: Pod + Zeroable + Copy + Default> {
    pub vertex_buffer: etalon::Buffer,
    pub index_buffer: etalon::Buffer,
    pub skin_attr_buffer: etalon::Buffer,
    pub proc_headers_buffer: etalon::Buffer,
    pub asset_table_buffer: etalon::Buffer,
    pub indirect_args_buffer: etalon::Buffer,
    pub vertex_allocator: SlabRangeAllocator<512>,
    pub index_allocator: SlabRangeAllocator<512>,
    pub assets: [ResidentAssetEntry; MAX_SLAB_RESIDENT_ASSETS],
    pub proc_headers: [P; MAX_SLAB_PROC_HEADERS],
    pub asset_count: u32,
    pub proc_header_count: u32,
    pub descriptor: MeshSlabArenaDescriptor,
    _vram_reservation: R,
}

impl<R, P: Pod + Zeroable + Copy + Default> MeshSlabArenaGpu<R, P> {
    /// Computes the total VRAM bytes required for a `MeshSlabArenaDescriptor`.
    pub fn required_vram_bytes(desc: MeshSlabArenaDescriptor) -> Option<u64> {
        let v_bytes = u64::from(desc.max_vertices.max(1))
            .checked_mul(std::mem::size_of::<MeshSlabVertex>() as u64)?;
        let i_bytes =
            u64::from(desc.max_indices.max(1)).checked_mul(std::mem::size_of::<u32>() as u64)?;
        let s_bytes = u64::from(desc.max_vertices.max(1))
            .checked_mul(std::mem::size_of::<SkinAttrVertex>() as u64)?;
        let p_bytes =
            u64::from(desc.max_proc_headers.max(1)).checked_mul(std::mem::size_of::<P>() as u64)?;
        let a_bytes = u64::from(desc.max_assets.max(1))
            .checked_mul(std::mem::size_of::<ResidentAssetEntry>() as u64)?;
        let cmd_bytes = u64::from(desc.max_indirect_commands.max(1)).checked_mul(20)?;
        v_bytes
            .checked_add(i_bytes)?
            .checked_add(s_bytes)?
            .checked_add(p_bytes)?
            .checked_add(a_bytes)?
            .checked_add(cmd_bytes)
    }

    /// Allocates the unified GPU slab buffers under `VramLedger` governance.
    pub fn try_new(
        device: &etalon::Device,
        desc: MeshSlabArenaDescriptor,
        reserve: impl FnOnce(u64) -> Option<R>,
    ) -> Option<Self> {
        let total_bytes = Self::required_vram_bytes(desc)?;
        let reservation = reserve(total_bytes)?;

        let vertex_buffer = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia-mesh-slab-vertices"),
            size: u64::from(desc.max_vertices.max(1))
                * std::mem::size_of::<MeshSlabVertex>() as u64,
            usage: etalon::BufferUsages::VERTEX
                | etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia-mesh-slab-indices"),
            size: u64::from(desc.max_indices.max(1)) * std::mem::size_of::<u32>() as u64,
            usage: etalon::BufferUsages::INDEX
                | etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let skin_attr_buffer = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia-mesh-slab-skin-attrs"),
            size: u64::from(desc.max_vertices.max(1))
                * std::mem::size_of::<SkinAttrVertex>() as u64,
            usage: etalon::BufferUsages::VERTEX
                | etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let proc_headers_buffer = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia-mesh-slab-proc-headers"),
            size: u64::from(desc.max_proc_headers.max(1)) * std::mem::size_of::<P>() as u64,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let asset_table_buffer = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia-mesh-slab-asset-table"),
            size: u64::from(desc.max_assets.max(1))
                * std::mem::size_of::<ResidentAssetEntry>() as u64,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indirect_args_buffer = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia-mesh-slab-indirect-args"),
            size: u64::from(desc.max_indirect_commands.max(1)) * 20,
            usage: etalon::BufferUsages::INDIRECT
                | etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_DST
                | etalon::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Some(Self {
            vertex_buffer,
            index_buffer,
            skin_attr_buffer,
            proc_headers_buffer,
            asset_table_buffer,
            indirect_args_buffer,
            vertex_allocator: SlabRangeAllocator::new(desc.max_vertices),
            index_allocator: SlabRangeAllocator::new(desc.max_indices),
            assets: [ResidentAssetEntry::default(); MAX_SLAB_RESIDENT_ASSETS],
            proc_headers: [P::default(); MAX_SLAB_PROC_HEADERS],
            asset_count: 0,
            proc_header_count: 0,
            descriptor: desc,
            _vram_reservation: reservation,
        })
    }

    /// Uploads a `.10d` `PROC1` header into `proc_headers_buffer` and returns its slot index.
    pub fn register_proc_header(&mut self, queue: &etalon::Queue, header: &P) -> Option<u16> {
        let slot = self.proc_header_count as usize;
        if slot >= MAX_SLAB_PROC_HEADERS || slot >= self.descriptor.max_proc_headers as usize {
            return None;
        }
        let gpu_header = *header;
        self.proc_headers[slot] = gpu_header;
        self.proc_header_count += 1;
        let byte_offset = (slot * std::mem::size_of::<P>()) as u64;
        queue.write_buffer(
            &self.proc_headers_buffer,
            byte_offset,
            bytemuck::bytes_of(&gpu_header),
        );
        Some(slot as u16)
    }

    /// Allocates vertex/index slab space and uploads a multi-LOD asset entry into the unified arena.
    pub fn register_mesh_lod_asset(
        &mut self,
        queue: &etalon::Queue,
        asset_id: u32,
        vertices: &[MeshSlabVertex],
        skin_attrs: Option<&[SkinAttrVertex]>,
        lod_indices: [&[u32]; 4],
        lod_rep_kinds: [u32; 4],
        bounding_sphere: [f32; 4],
        proc_header_index: u16,
        eont_header_index: u16,
    ) -> Option<u32> {
        let slot = self.asset_count as usize;
        if slot >= MAX_SLAB_RESIDENT_ASSETS || slot >= self.descriptor.max_assets as usize {
            return None;
        }
        let v_count = u32::try_from(vertices.len()).ok()?;
        let base_vertex = self.vertex_allocator.allocate(v_count, 1)?;

        let total_indices: u32 = lod_indices.iter().try_fold(0u32, |acc, slice| {
            acc.checked_add(u32::try_from(slice.len()).ok()?)
        })?;
        let Some(first_index_base) = self.index_allocator.allocate(total_indices, 1) else {
            self.vertex_allocator.free(base_vertex, v_count);
            return None;
        };

        if !vertices.is_empty() {
            let v_byte_offset =
                u64::from(base_vertex) * std::mem::size_of::<MeshSlabVertex>() as u64;
            queue.write_buffer(
                &self.vertex_buffer,
                v_byte_offset,
                bytemuck::cast_slice(vertices),
            );
            if let Some(skin) = skin_attrs {
                let len = skin.len().min(vertices.len());
                let s_byte_offset =
                    u64::from(base_vertex) * std::mem::size_of::<SkinAttrVertex>() as u64;
                queue.write_buffer(
                    &self.skin_attr_buffer,
                    s_byte_offset,
                    bytemuck::cast_slice(&skin[..len]),
                );
            }
        }

        let mut lod_slices = [SubmeshSlice::default(); 4];
        let mut cursor = first_index_base;
        for lod in 0..4 {
            let idx_slice = lod_indices[lod];
            let count = idx_slice.len() as u32;
            if count > 0 {
                let i_byte_offset = u64::from(cursor) * std::mem::size_of::<u32>() as u64;
                queue.write_buffer(
                    &self.index_buffer,
                    i_byte_offset,
                    bytemuck::cast_slice(idx_slice),
                );
            }
            lod_slices[lod] = SubmeshSlice {
                first_index: cursor,
                index_count: count,
                base_vertex: base_vertex as i32,
                rep_kind: lod_rep_kinds[lod],
            };
            cursor += count;
        }

        let entry = ResidentAssetEntry {
            asset_id,
            proc_header_index,
            eont_header_index,
            surfel_slice_offset: 0,
            surfel_count: 0,
            impostor_atlas_layer: 0,
            flags: 1 | if skin_attrs.is_some() { 4 } else { 0 },
            total_vertex_offset: base_vertex,
            total_vertex_count: v_count,
            reserved_u32: 0,
            bounding_sphere,
            lod_slices,
        };
        self.assets[slot] = entry;
        self.asset_count += 1;
        let entry_offset = (slot * std::mem::size_of::<ResidentAssetEntry>()) as u64;
        queue.write_buffer(
            &self.asset_table_buffer,
            entry_offset,
            bytemuck::bytes_of(&entry),
        );
        Some(slot as u32)
    }
}
