//! High-throughput ring-slab allocator with zero allocation in steady state.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTopology {
    Unified { zero_copy: bool },
    Discrete { staging_required: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingUsage {
    StorageRead,
    StorageReadWrite,
    Uniform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferView {
    pub offset: usize,
    pub length_bytes: usize,
    pub binding: u32,
    pub group: u32,
    pub usage: BindingUsage,
}

pub const DEFAULT_BINDING_ALIGNMENT: usize = 256;

#[derive(Debug)]
pub struct QualiaSlabAllocator {
    pub topology: MemoryTopology,
    pub capacity_bytes: u64,
    pub alignment: u64,
    pub write_count: u64,
    pub read_count: u64,
}

impl QualiaSlabAllocator {
    pub fn new(capacity_bytes: usize, alignment: usize) -> Self {
        Self {
            topology: MemoryTopology::Discrete { staging_required: false },
            capacity_bytes: capacity_bytes as u64,
            alignment: alignment as u64,
            write_count: 0,
            read_count: 0,
        }
    }

    pub fn allocate(
        &mut self,
        size_bytes: usize,
        binding: u32,
        group: u32,
        usage: BindingUsage,
    ) -> Result<BufferView, &'static str> {
        let size_u64 = size_bytes as u64;
        if size_u64 > self.capacity_bytes {
            return Err("Allocation exceeds slab capacity");
        }

        let aligned_write = align_up(self.write_count, self.alignment);
        let mut padding = aligned_write - self.write_count;
        let mut offset = aligned_write % self.capacity_bytes;

        if offset + size_u64 > self.capacity_bytes {
            padding += self.capacity_bytes - offset;
            offset = 0;
        }

        if self.write_count + padding + size_u64 - self.read_count > self.capacity_bytes {
            return Err("Write head lapped read head in slab allocator");
        }

        self.write_count += padding + size_u64;

        Ok(BufferView {
            offset: offset as usize,
            length_bytes: size_bytes,
            binding,
            group,
            usage,
        })
    }

    pub fn advance_read_head(&mut self, new_head_offset: usize) {
        let current_offset = (self.read_count % self.capacity_bytes) as usize;
        let advance = if new_head_offset >= current_offset {
            new_head_offset - current_offset
        } else {
            (self.capacity_bytes as usize) - current_offset + new_head_offset
        };
        self.read_count += advance as u64;
    }

    pub fn clear(&mut self) {
        self.read_count = self.write_count;
    }
}

#[inline(always)]
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}
