use ash::vk;
use wgt::PreparedComputeError;

use super::{map_error, PreparedCompute, PreparedComputeDescriptor};

impl PreparedCompute {
    pub(super) unsafe fn record(
        &self,
        descriptor: &PreparedComputeDescriptor<'_>,
    ) -> Result<(), PreparedComputeError> {
        let raw = &self.device.raw;
        let command = self.command;
        unsafe {
            raw.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .map_err(map_error)?;
            let start = [vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::MEMORY_READ
                        | vk::AccessFlags::MEMORY_WRITE
                        | vk::AccessFlags::HOST_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)];
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &start,
                &[],
                &[],
            );
            raw.cmd_copy_buffer(
                command,
                descriptor.upload.raw,
                descriptor.input.raw,
                &[vk::BufferCopy::default().size(descriptor.input_size)],
            );
            raw.cmd_copy_buffer(
                command,
                descriptor.upload.raw,
                descriptor.parameters.raw,
                &[vk::BufferCopy::default()
                    .src_offset(descriptor.input_size)
                    .size(descriptor.parameter_size)],
            );
            raw.cmd_fill_buffer(command, descriptor.output.raw, 0, descriptor.output_size, 0);
            let to_compute = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::UNIFORM_READ,
                )];
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &to_compute,
                &[],
                &[],
            );
            raw.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                descriptor.pipeline.raw,
            );
            raw.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                descriptor.pipeline_layout.raw,
                0,
                &[descriptor.bind_group.set.raw()],
                &[],
            );
            raw.cmd_dispatch(
                command,
                descriptor.workgroups[0],
                descriptor.workgroups[1],
                descriptor.workgroups[2],
            );
            let to_readback = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &to_readback,
                &[],
                &[],
            );
            raw.cmd_copy_buffer(
                command,
                descriptor.output.raw,
                descriptor.readback.raw,
                &[vk::BufferCopy::default().size(descriptor.output_size)],
            );
            let to_host = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(descriptor.readback.raw)
                .offset(0)
                .size(descriptor.output_size)];
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &to_host,
                &[],
            );
            raw.end_command_buffer(command).map_err(map_error)?;
        }
        Ok(())
    }
}
