//! Key-Value cache layouts and paged block tables.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheLayout {
    Fp32,
    Fp16,
    Int8Quantized,
    PagedBlock,
}

#[derive(Clone, Debug)]
pub struct PagedKvBlockTable {
    pub block_size: u32,
    pub num_layers: u32,
    pub max_blocks: u32,
    pub block_entries: Vec<u32>,
}

impl PagedKvBlockTable {
    pub fn new(block_size: u32, num_layers: u32, max_blocks: u32) -> Self {
        Self {
            block_size,
            num_layers,
            max_blocks,
            block_entries: Vec::new(),
        }
    }
}
