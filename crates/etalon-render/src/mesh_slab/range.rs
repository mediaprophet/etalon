use bytemuck::{Pod, Zeroable};
/// Contiguous free span `[offset .. offset + length)` inside a GPU buffer slab.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct SlabRange {
    pub offset: u32,
    pub length: u32,
}

/// Zero-heap first-fit range sub-allocator with immediate bidirectional coalescing.
#[derive(Clone, Debug)]
pub struct SlabRangeAllocator<const MAX_RANGES: usize = 512> {
    free_ranges: [SlabRange; MAX_RANGES],
    free_count: usize,
    capacity: u32,
    used: u32,
}

impl<const MAX_RANGES: usize> SlabRangeAllocator<MAX_RANGES> {
    /// Creates a new zero-heap range allocator managing `[0 .. capacity)`.
    #[inline]
    pub const fn new(capacity: u32) -> Self {
        let mut free_ranges = [SlabRange {
            offset: 0,
            length: 0,
        }; MAX_RANGES];
        let free_count = if capacity > 0 && MAX_RANGES > 0 {
            free_ranges[0] = SlabRange {
                offset: 0,
                length: capacity,
            };
            1
        } else {
            0
        };
        Self {
            free_ranges,
            free_count,
            capacity,
            used: 0,
        }
    }

    /// Total element capacity of the slab.
    #[inline]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Currently allocated element count (including alignment padding).
    #[inline]
    pub const fn used(&self) -> u32 {
        self.used
    }

    /// Remaining free element capacity.
    #[inline]
    pub const fn free_remaining(&self) -> u32 {
        self.capacity.saturating_sub(self.used)
    }

    /// Number of coalesced free spans currently tracked.
    #[inline]
    pub const fn free_range_count(&self) -> usize {
        self.free_count
    }

    /// Allocates `count` contiguous elements aligned to `alignment` (power of two or `>= 1`).
    /// Performs zero heap allocations.
    pub fn allocate(&mut self, count: u32, alignment: u32) -> Option<u32> {
        if count == 0 {
            return Some(0);
        }
        let align = alignment.max(1);
        for idx in 0..self.free_count {
            let range = self.free_ranges[idx];
            let rem = range.offset % align;
            let pad = if rem == 0 { 0 } else { align - rem };
            let total_needed = count.checked_add(pad)?;
            if range.length >= total_needed {
                let aligned_offset = range.offset + pad;
                let tail_start = aligned_offset + count;
                let range_end = range.offset + range.length;
                let tail_len = range_end - tail_start;

                if pad == 0 && tail_len == 0 {
                    // Exact match: remove free slot by shifting left
                    for k in idx..self.free_count - 1 {
                        self.free_ranges[k] = self.free_ranges[k + 1];
                    }
                    self.free_count -= 1;
                } else if pad == 0 {
                    // Trim front of current range
                    self.free_ranges[idx] = SlabRange {
                        offset: tail_start,
                        length: tail_len,
                    };
                } else if tail_len == 0 {
                    // Keep only front alignment padding
                    self.free_ranges[idx] = SlabRange {
                        offset: range.offset,
                        length: pad,
                    };
                } else {
                    // Split into front pad and tail remainder if capacity permits
                    if self.free_count >= MAX_RANGES {
                        continue;
                    }
                    for k in (idx + 1..=self.free_count).rev() {
                        self.free_ranges[k] = self.free_ranges[k - 1];
                    }
                    self.free_ranges[idx] = SlabRange {
                        offset: range.offset,
                        length: pad,
                    };
                    self.free_ranges[idx + 1] = SlabRange {
                        offset: tail_start,
                        length: tail_len,
                    };
                    self.free_count += 1;
                }
                self.used = self.used.saturating_add(count);
                return Some(aligned_offset);
            }
        }
        None
    }

    /// Frees `[offset .. offset + count)` and coalesces with adjacent free spans in-place.
    pub fn free(&mut self, offset: u32, count: u32) -> bool {
        if count == 0 {
            return true;
        }
        let Some(end) = offset.checked_add(count) else {
            return false;
        };
        if end > self.capacity {
            return false;
        }

        // Find sorted insertion point by offset
        let mut insert_idx = 0;
        while insert_idx < self.free_count && self.free_ranges[insert_idx].offset < offset {
            insert_idx += 1;
        }

        // Reject overlapping frees
        if insert_idx > 0 {
            let prev = self.free_ranges[insert_idx - 1];
            if prev.offset + prev.length > offset {
                return false;
            }
        }
        if insert_idx < self.free_count {
            let next = self.free_ranges[insert_idx];
            if end > next.offset {
                return false;
            }
        }

        let merge_prev = insert_idx > 0 && {
            let prev = self.free_ranges[insert_idx - 1];
            prev.offset + prev.length == offset
        };
        let merge_next = insert_idx < self.free_count && {
            let next = self.free_ranges[insert_idx];
            end == next.offset
        };

        match (merge_prev, merge_next) {
            (true, true) => {
                let next_len = self.free_ranges[insert_idx].length;
                self.free_ranges[insert_idx - 1].length += count + next_len;
                for k in insert_idx..self.free_count - 1 {
                    self.free_ranges[k] = self.free_ranges[k + 1];
                }
                self.free_count -= 1;
            }
            (true, false) => {
                self.free_ranges[insert_idx - 1].length += count;
            }
            (false, true) => {
                self.free_ranges[insert_idx].offset = offset;
                self.free_ranges[insert_idx].length += count;
            }
            (false, false) => {
                if self.free_count >= MAX_RANGES {
                    return false;
                }
                for k in (insert_idx + 1..=self.free_count).rev() {
                    self.free_ranges[k] = self.free_ranges[k - 1];
                }
                self.free_ranges[insert_idx] = SlabRange {
                    offset,
                    length: count,
                };
                self.free_count += 1;
            }
        }
        self.used = self.used.saturating_sub(count);
        true
    }
}
