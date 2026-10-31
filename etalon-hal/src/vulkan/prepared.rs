//! Private-device, fixed-resource reusable compute execution.

mod record;

use alloc::sync::Arc;
use core::{fmt, ptr::NonNull, time::Duration};

use ash::vk;
use wgt::PreparedComputeError as Error;

use super::{BindGroup, Buffer, ComputePipeline, Device, DeviceShared, PipelineLayout, Queue};

#[derive(Debug)]
pub struct PreparedComputeDescriptor<'a> {
    pub input: &'a Buffer,
    pub parameters: &'a Buffer,
    pub output: &'a Buffer,
    pub upload: &'a Buffer,
    pub readback: &'a Buffer,
    pub pipeline: &'a ComputePipeline,
    pub bind_group: &'a BindGroup,
    pub pipeline_layout: &'a PipelineLayout,
    pub input_size: u64,
    pub parameter_size: u64,
    pub output_size: u64,
    pub workgroups: [u32; 3],
}

struct Mapping {
    pointer: NonNull<u8>,
    range: Option<vk::MappedMemoryRange<'static>>,
}

pub struct PreparedCompute {
    device: Arc<DeviceShared>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    upload: Mapping,
    readback: Mapping,
    input_size: usize,
    parameter_size: usize,
    output_size: usize,
    pending: bool,
    poisoned: bool,
}

impl fmt::Debug for PreparedCompute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedCompute")
            .field("input_size", &self.input_size)
            .field("parameter_size", &self.parameter_size)
            .field("output_size", &self.output_size)
            .field("pending", &self.pending)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

// Private mapped allocations remain live across moves; exclusive access serializes all use.
unsafe impl Send for PreparedCompute {}

impl PreparedCompute {
    /// # Safety
    ///
    /// The device and queue must be private to this owner, idle, and never used
    /// through ordinary submission or polling while this object exists. All
    /// resources must belong to that device, satisfy the descriptor's sizes and
    /// usages, and outlive successful retirement and this object's destruction.
    /// The validated pipeline, layout, bind group, dispatch and copies must match
    /// this descriptor. No other code may access or destroy these resources.
    pub unsafe fn new(
        device: &Device,
        queue: &Queue,
        descriptor: &PreparedComputeDescriptor<'_>,
    ) -> Result<Self, Error> {
        if !Arc::ptr_eq(&device.shared, &queue.device)
            || descriptor.workgroups.contains(&0)
            || [
                descriptor.input_size,
                descriptor.parameter_size,
                descriptor.output_size,
            ]
            .iter()
            .any(|size| *size == 0 || size % 4 != 0)
        {
            return Err(Error::InvalidDescriptor);
        }
        let upload_size = descriptor
            .input_size
            .checked_add(descriptor.parameter_size)
            .filter(|size| *size <= isize::MAX as u64)
            .ok_or(Error::Capacity)?;
        if descriptor.output_size > isize::MAX as u64 {
            return Err(Error::Capacity);
        }
        let upload = mapping(device, descriptor.upload, upload_size)?;
        let readback = mapping(device, descriptor.readback, descriptor.output_size)?;
        let mut prepared = Self {
            device: device.shared.clone(),
            queue: queue.raw,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            upload,
            readback,
            input_size: descriptor.input_size as usize,
            parameter_size: descriptor.parameter_size as usize,
            output_size: descriptor.output_size as usize,
            pending: false,
            poisoned: false,
        };
        prepared.pool = unsafe {
            prepared.device.raw.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(queue.family_index),
                None,
            )
        }
        .map_err(map_error)?;
        let commands = unsafe {
            prepared.device.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(prepared.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(map_error)?;
        prepared.command = commands[0];
        prepared.fence = unsafe {
            prepared
                .device
                .raw
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(map_error)?;
        unsafe { prepared.record(descriptor) }?;
        Ok(prepared)
    }

    /// Runs only against resources retained under the constructor's contract.
    ///
    /// # Safety
    /// The constructor's exclusive ownership and resource-lifetime requirements
    /// must still hold. No command or map callbacks may access this private queue.
    pub unsafe fn execute(
        &mut self,
        input: &[u8],
        parameters: &[u8],
        output: &mut [u8],
        timeout: Duration,
    ) -> Result<(), Error> {
        output.fill(0);
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        if input.len() > self.input_size
            || parameters.len() != self.parameter_size
            || output.len() > self.output_size
        {
            return Err(Error::Capacity);
        }
        let upload = unsafe {
            core::slice::from_raw_parts_mut(
                self.upload.pointer.as_ptr(),
                self.input_size + self.parameter_size,
            )
        };
        upload[..self.input_size].fill(0);
        upload[..input.len()].copy_from_slice(input);
        upload[self.input_size..].copy_from_slice(parameters);
        if let Some(range) = self.upload.range {
            let result = unsafe { self.device.raw.flush_mapped_memory_ranges(&[range]) };
            if let Err(error) = result {
                return Err(self.fail(error));
            }
        }
        let result = unsafe { self.device.raw.reset_fences(&[self.fence]) };
        if let Err(error) = result {
            return Err(self.fail(error));
        }
        let commands = [self.command];
        let submissions = [vk::SubmitInfo::default().command_buffers(&commands)];
        self.pending = true;
        let result = unsafe {
            self.device
                .raw
                .queue_submit(self.queue, &submissions, self.fence)
        };
        if let Err(error) = result {
            return Err(self.fail(error));
        }
        let timeout = timeout.as_nanos().min((u64::MAX - 1) as u128) as u64;
        let result = unsafe {
            self.device
                .raw
                .wait_for_fences(&[self.fence], true, timeout)
        };
        if let Err(error) = result {
            return Err(self.fail(error));
        }
        self.pending = false;
        if let Some(range) = self.readback.range {
            let result = unsafe { self.device.raw.invalidate_mapped_memory_ranges(&[range]) };
            if let Err(error) = result {
                return Err(self.fail(error));
            }
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.readback.pointer.as_ptr(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok(())
    }

    /// Drains timed-out or uncertain work before any referenced resources drop.
    /// Unexpected failures require the caller to retain the entire resource owner.
    pub fn retire(&mut self) -> Result<(), Error> {
        if !self.pending {
            return Ok(());
        }
        match unsafe { self.device.raw.device_wait_idle() } {
            Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => {
                self.pending = false;
                Ok(())
            }
            Err(error) => Err(self.fail(error)),
        }
    }

    fn fail(&mut self, error: vk::Result) -> Error {
        self.poisoned = true;
        map_error(error)
    }
}

impl Drop for PreparedCompute {
    fn drop(&mut self) {
        if self.retire().is_err() {
            return;
        }
        unsafe {
            if self.pool != vk::CommandPool::null() {
                self.device.raw.destroy_command_pool(self.pool, None);
            }
            if self.fence != vk::Fence::null() {
                self.device.raw.destroy_fence(self.fence, None);
            }
        }
    }
}

fn mapping(device: &Device, buffer: &Buffer, size: u64) -> Result<Mapping, Error> {
    let super::BufferOwnership::Managed(backing) = &buffer.ownership else {
        return Err(Error::UnsupportedConfiguration);
    };
    let (pointer, coherent, allocation_offset, allocation_end) = {
        let backing = backing.lock();
        let super::BufferMemoryBacking::Managed(allocation) = &*backing else {
            return Err(Error::UnsupportedConfiguration);
        };
        if size > allocation.size() {
            return Err(Error::Capacity);
        }
        (
            allocation
                .mapped_ptr()
                .ok_or(Error::UnsupportedConfiguration)?
                .cast::<u8>(),
            allocation
                .memory_properties()
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT),
            allocation.offset(),
            allocation
                .offset()
                .checked_add(allocation.size())
                .ok_or(Error::Capacity)?,
        )
    };
    let range = if coherent {
        None
    } else {
        let mut ranges = device
            .shared
            .make_memory_ranges(buffer, core::iter::once(0..size))
            .ok_or(Error::UnsupportedConfiguration)?;
        let range = ranges.next().ok_or(Error::UnsupportedConfiguration)?;
        if range.offset < allocation_offset
            || range
                .offset
                .checked_add(range.size)
                .ok_or(Error::Capacity)?
                > allocation_end
        {
            return Err(Error::UnsupportedConfiguration);
        }
        Some(
            vk::MappedMemoryRange::default()
                .memory(range.memory)
                .offset(range.offset)
                .size(range.size),
        )
    };
    Ok(Mapping { pointer, range })
}

pub(super) fn map_error(error: vk::Result) -> Error {
    match error {
        vk::Result::TIMEOUT => Error::Timeout,
        vk::Result::ERROR_DEVICE_LOST => Error::DeviceLost,
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            Error::OutOfMemory
        }
        _ => Error::Unexpected,
    }
}
