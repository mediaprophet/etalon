use crate::{quantity::SampleEncoding, ContractError};

/// A generation-tagged identity for an allocation managed by its owning runtime.
///
/// This token identifies a lifetime; it does not grant access or prove that the
/// allocation is still live. It is not a native pointer or serialized ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferToken {
    allocation_id: u64,
    generation: u32,
}

impl BufferToken {
    pub const fn new(allocation_id: u64, generation: u32) -> Result<Self, ContractError> {
        if allocation_id == 0 || generation == 0 {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self {
            allocation_id,
            generation,
        })
    }

    pub const fn allocation_id(self) -> u64 {
        self.allocation_id
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// A fixed-rank byte-strided tensor view. Axes are ordered as described by the
/// caller; arbitrary axis permutations and padding are supported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleLayout {
    rank: u8,
    extents: [u32; 4],
    strides_bytes: [u64; 4],
    encoding: SampleEncoding,
}

impl SampleLayout {
    pub fn new(
        rank: u8,
        extents: [u32; 4],
        strides_bytes: [u64; 4],
        encoding: SampleEncoding,
    ) -> Result<Self, ContractError> {
        if !(1..=4).contains(&rank) {
            return Err(ContractError::InvalidExtent);
        }
        let active = usize::from(rank);
        if extents[..active].contains(&0)
            || extents[active..].iter().any(|&extent| extent != 0)
            || strides_bytes[active..].iter().any(|&stride| stride != 0)
        {
            return Err(ContractError::InvalidExtent);
        }

        let element_bytes = u64::from(encoding.element_bytes());
        let scalar_alignment = encoding.alignment_bytes();
        if element_bytes == 0 || scalar_alignment == 0 || !scalar_alignment.is_power_of_two() {
            return Err(ContractError::IncompatibleEncoding);
        }

        for axis in 0..active {
            let extent = extents[axis];
            let stride = strides_bytes[axis];
            if extent > 1 && stride == 0 {
                return Err(ContractError::InvalidStride);
            }
            if extent > 1 && stride % u64::from(scalar_alignment) != 0 {
                return Err(ContractError::InvalidStride);
            }
        }

        let layout = Self {
            rank,
            extents,
            strides_bytes,
            encoding,
        };
        layout.required_byte_span()?;
        Ok(layout)
    }

    pub const fn rank(&self) -> u8 {
        self.rank
    }

    pub const fn extents(&self) -> &[u32; 4] {
        &self.extents
    }

    pub const fn strides_bytes(&self) -> &[u64; 4] {
        &self.strides_bytes
    }

    pub const fn encoding(&self) -> &SampleEncoding {
        &self.encoding
    }

    /// Byte span from the first element's start through the last element's end.
    pub fn required_byte_span(&self) -> Result<u64, ContractError> {
        let mut occupied_span = u64::from(self.encoding.element_bytes());
        let mut axes = [(0u64, 0u32); 4];
        let mut count = 0usize;
        for axis in 0..usize::from(self.rank) {
            if self.extents[axis] > 1 {
                axes[count] = (self.strides_bytes[axis], self.extents[axis]);
                count += 1;
            }
        }
        axes[..count].sort_unstable_by_key(|&(stride, _)| stride);

        for &(stride, extent) in &axes[..count] {
            // Requiring each next stride to cover the preceding span is a
            // conservative non-aliasing rule; some valid exotic layouts fail it.
            if stride < occupied_span {
                return Err(ContractError::InvalidStride);
            }
            let axis_span = u64::from(extent - 1)
                .checked_mul(stride)
                .ok_or(ContractError::ArithmeticOverflow)?;
            occupied_span = occupied_span
                .checked_add(axis_span)
                .ok_or(ContractError::ArithmeticOverflow)?;
        }

        Ok(occupied_span)
    }
}

/// A checked subregion of a generation-tagged allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleRegion {
    token: BufferToken,
    offset_bytes: u64,
    length_bytes: u64,
    allocation_capacity_bytes: u64,
    alignment_bytes: u32,
}

impl SampleRegion {
    pub fn new(
        token: BufferToken,
        offset_bytes: u64,
        length_bytes: u64,
        allocation_capacity_bytes: u64,
        alignment_bytes: u32,
    ) -> Result<Self, ContractError> {
        if alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(ContractError::InvalidAlignment);
        }
        if offset_bytes % u64::from(alignment_bytes) != 0 {
            return Err(ContractError::InvalidAlignment);
        }
        if length_bytes == 0 {
            return Err(ContractError::InvalidExtent);
        }
        let end = offset_bytes
            .checked_add(length_bytes)
            .ok_or(ContractError::ArithmeticOverflow)?;
        if end > allocation_capacity_bytes {
            return Err(ContractError::OutOfBounds);
        }
        Ok(Self {
            token,
            offset_bytes,
            length_bytes,
            allocation_capacity_bytes,
            alignment_bytes,
        })
    }

    pub const fn token(&self) -> BufferToken {
        self.token
    }

    pub const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    pub const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }

    pub const fn alignment_bytes(&self) -> u32 {
        self.alignment_bytes
    }

    pub fn validate_layout(&self, layout: &SampleLayout) -> Result<(), ContractError> {
        let scalar_alignment = layout.encoding.alignment_bytes();
        if self.alignment_bytes < scalar_alignment
            || !self.alignment_bytes.is_multiple_of(scalar_alignment)
            || !self
                .offset_bytes
                .is_multiple_of(u64::from(scalar_alignment))
        {
            return Err(ContractError::InvalidAlignment);
        }
        let span = layout.required_byte_span()?;
        if span > self.length_bytes {
            return Err(ContractError::OutOfBounds);
        }
        let end = self
            .offset_bytes
            .checked_add(span)
            .ok_or(ContractError::ArithmeticOverflow)?;
        if end > self.allocation_capacity_bytes {
            return Err(ContractError::OutOfBounds);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "buffer_tests.rs"]
mod tests;
