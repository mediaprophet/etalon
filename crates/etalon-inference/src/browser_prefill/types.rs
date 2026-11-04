pub const KV_ATTENTION_MASK_WORDS: usize = 32;
pub const MC8_UNIFORM_ALIGN: usize = 256;
pub const MC8_MAX_GEMM_UNIFORM_SLOTS: usize = 8;
pub const MC8_MAX_ELEM_UNIFORM_LAYER_SLOTS: usize = 8;
pub const MC8_MAX_ATTN_UNIFORM_SLOTS: usize = 8;
pub struct Mc8ChunkUniformCursors {
    pub attn: usize,
    pub elem: usize,
    pub gemm: usize,
}
impl Mc8ChunkUniformCursors {
    /// Resets the per-chunk uniform cursor state.
    pub fn reset(&mut self) {
        self.attn = 0;
        self.elem = 0;
        self.gemm = 0;
    }

    /// Byte offset of the attention uniform region.
    pub fn attn_base_byte(&self) -> u32 {
        (self.attn * MC8_UNIFORM_ALIGN) as u32
    }

    /// Byte offset of the elementwise uniform region.
    pub fn elem_base_byte(&self) -> u32 {
        (self.elem * MC8_UNIFORM_ALIGN) as u32
    }

    /// Byte offset of the GEMM uniform region.
    pub fn gemm_base_byte(&self) -> u32 {
        (self.gemm * MC8_UNIFORM_ALIGN) as u32
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mc8WeightRole {
    AttnK,
    AttnV,
    AttnQ,
    OProj,
    Gate,
    Up,
    Down,
}
impl Mc8WeightRole {
    /// Stable index into the per-role resident stride table (`mc8_weight_role_stride`).
    #[inline]
    pub fn idx(self) -> usize {
        match self {
            Mc8WeightRole::AttnK => 0,
            Mc8WeightRole::AttnV => 1,
            Mc8WeightRole::AttnQ => 2,
            Mc8WeightRole::OProj => 3,
            Mc8WeightRole::Gate => 4,
            Mc8WeightRole::Up => 5,
            Mc8WeightRole::Down => 6,
        }
    }
}
pub struct Mc8PrefillLayerUniforms {
    pub k_off: u32,
    pub v_off: u32,
    pub q_off: u32,
    pub attn_norm_elem_off: Option<u32>,
    pub off_o: Option<u32>,
    pub off_attn_res: u32,
    pub off_ffn_norm: Option<u32>,
    pub off_gate: u32,
    pub off_up: u32,
    pub off_silu: u32,
    pub off_down: u32,
    pub off_ffn_res: u32,
    /// Phase 5.5: dynamic offsets for the Q/K/V projection GEMMs (parallel kernel).
    pub off_q_gemm: u32,
    pub off_k_gemm: u32,
    pub off_v_gemm: u32,
}
pub struct Mc8PrefillLayerGeom {
    pub row_stride: usize,
    pub row_stride_u32: u32,
    pub batch_in_bytes: wgpu::BufferAddress,
    pub work_span_bytes: wgpu::BufferAddress,
    pub emb_bytes: wgpu::BufferAddress,
    pub n_embd_bytes: wgpu::BufferAddress,
    pub slot_o: wgpu::BufferAddress,
    pub slot_gate: wgpu::BufferAddress,
    pub slot_up: wgpu::BufferAddress,
    pub slot_save: wgpu::BufferAddress,
    pub slot_scratch_half: wgpu::BufferAddress,
    pub slot_o_f: u32,
    pub slot_save_f: u32,
    pub slot_gate_f: u32,
    pub slot_up_f: u32,
    pub slot_scratch_half_f: u32,
}
pub struct Mc8UniformArena {
    pub bytes: [u8; MC8_MAX_GEMM_UNIFORM_SLOTS * MC8_UNIFORM_ALIGN],
    pub slots: usize,
}
impl Mc8UniformArena {
    /// Appends `value` to the GEMM uniform arena; returns the assigned slot.
    pub fn push<T: bytemuck::Pod>(&mut self, value: &T) -> u32 {
        debug_assert!(std::mem::size_of::<T>() <= MC8_UNIFORM_ALIGN);
        debug_assert!(self.slots < MC8_MAX_GEMM_UNIFORM_SLOTS);
        let byte_off = self.slots * MC8_UNIFORM_ALIGN;
        self.slots += 1;
        let dst = &mut self.bytes[byte_off..byte_off + std::mem::size_of::<T>()];
        dst.copy_from_slice(bytemuck::bytes_of(value));
        byte_off as u32
    }

    /// Uploads the GEMM arena to `buf`.
    pub fn upload(&self, queue: &wgpu::Queue, buf: &wgpu::Buffer) {
        if self.slots == 0 {
            return;
        }
        queue.write_buffer(buf, 0, &self.bytes[..self.slots * MC8_UNIFORM_ALIGN]);
    }

    /// Uploads the GEMM arena starting at `base_slot`.
    pub fn upload_at(&self, queue: &wgpu::Queue, buf: &wgpu::Buffer, base_slot: usize) {
        if self.slots == 0 {
            return;
        }
        let byte_off = (base_slot * MC8_UNIFORM_ALIGN) as wgpu::BufferAddress;
        queue.write_buffer(buf, byte_off, &self.bytes[..self.slots * MC8_UNIFORM_ALIGN]);
    }
}
pub struct Mc8ElemUniformArena {
    pub bytes: [u8; MC8_MAX_ELEM_UNIFORM_LAYER_SLOTS * MC8_UNIFORM_ALIGN],
    pub slots: usize,
}
impl Mc8ElemUniformArena {
    /// Appends `value` to the elementwise uniform arena; returns the assigned slot.
    pub fn push<T: bytemuck::Pod>(&mut self, value: &T) -> u32 {
        debug_assert!(std::mem::size_of::<T>() <= MC8_UNIFORM_ALIGN);
        debug_assert!(self.slots < MC8_MAX_ELEM_UNIFORM_LAYER_SLOTS);
        let byte_off = self.slots * MC8_UNIFORM_ALIGN;
        self.slots += 1;
        self.bytes[byte_off..byte_off + std::mem::size_of::<T>()]
            .copy_from_slice(bytemuck::bytes_of(value));
        byte_off as u32
    }

    /// Uploads the whole arena contents to `buf` via `queue`.
    pub fn upload(&self, queue: &wgpu::Queue, buf: &wgpu::Buffer) {
        if self.slots == 0 {
            return;
        }
        queue.write_buffer(buf, 0, &self.bytes[..self.slots * MC8_UNIFORM_ALIGN]);
    }

    /// Uploads arena contents starting at `base_slot`.
    pub fn upload_at(&self, queue: &wgpu::Queue, buf: &wgpu::Buffer, base_slot: usize) {
        if self.slots == 0 {
            return;
        }
        if base_slot == 0 {
            self.upload(queue, buf);
            return;
        }
        let byte_off = (base_slot * MC8_UNIFORM_ALIGN) as wgpu::BufferAddress;
        queue.write_buffer(buf, byte_off, &self.bytes[..self.slots * MC8_UNIFORM_ALIGN]);
    }
}
pub struct Mc8AttnUniformArena {
    pub bytes: [u8; MC8_MAX_ATTN_UNIFORM_SLOTS * MC8_UNIFORM_ALIGN],
    pub slots: usize,
}
impl Mc8AttnUniformArena {
    /// Appends `value` to the attention uniform arena; returns the assigned slot.
    pub fn push<T: bytemuck::Pod>(&mut self, value: &T) -> u32 {
        debug_assert!(std::mem::size_of::<T>() <= MC8_UNIFORM_ALIGN);
        debug_assert!(self.slots < MC8_MAX_ATTN_UNIFORM_SLOTS);
        let byte_off = self.slots * MC8_UNIFORM_ALIGN;
        self.slots += 1;
        self.bytes[byte_off..byte_off + std::mem::size_of::<T>()]
            .copy_from_slice(bytemuck::bytes_of(value));
        byte_off as u32
    }

    /// Uploads the attention arena to `buf`.
    pub fn upload(&self, queue: &wgpu::Queue, buf: &wgpu::Buffer) {
        if self.slots == 0 {
            return;
        }
        queue.write_buffer(buf, 0, &self.bytes[..self.slots * MC8_UNIFORM_ALIGN]);
    }

    /// Uploads the attention arena starting at `base_slot`.
    pub fn upload_at(&self, queue: &wgpu::Queue, buf: &wgpu::Buffer, base_slot: usize) {
        if self.slots == 0 {
            return;
        }
        if base_slot == 0 {
            self.upload(queue, buf);
            return;
        }
        let byte_off = (base_slot * MC8_UNIFORM_ALIGN) as wgpu::BufferAddress;
        queue.write_buffer(buf, byte_off, &self.bytes[..self.slots * MC8_UNIFORM_ALIGN]);
    }
}
