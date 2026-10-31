use alloc::{borrow::Cow, string::ToString, sync::Arc};

use wgt::{PrepareComputeError, PreparedComputeDescriptor, PreparedComputeError};

use crate::{
    binding_model as binding,
    command::ComputePassDescriptor,
    device::{queue::Queue, Device},
    instance::Adapter,
    pipeline,
    resource::{Buffer, RawResourceAccess},
};

pub(super) struct Resources {
    buffers: [Arc<Buffer>; 5],
    pipeline: Arc<pipeline::ComputePipeline>,
    bind_group: Arc<binding::BindGroup>,
    layout: Arc<binding::PipelineLayout>,
    queue: Arc<Queue>,
    pub(super) device: Arc<Device>,
}

fn diagnostic(error: impl core::fmt::Display) -> PrepareComputeError {
    PrepareComputeError {
        reason: PreparedComputeError::InvalidDescriptor,
        message: error.to_string(),
    }
}

impl Resources {
    pub(super) fn new(
        adapter: &Arc<Adapter>,
        desc: &PreparedComputeDescriptor<'_>,
    ) -> Result<Self, PrepareComputeError> {
        let (device, queue) = adapter
            .request_prepared_device(&crate::device::DeviceDescriptor::default())
            .map_err(diagnostic)?;
        desc.validate(&device.limits)?;
        let create_buffer = |size, usage| {
            device
                .create_buffer_inner(&wgt::BufferDescriptor {
                    label: None,
                    size,
                    usage,
                    mapped_at_creation: false,
                })
                .map_err(diagnostic)
        };
        use wgt::BufferUsages as U;
        let buffers = [
            create_buffer(desc.input_size, U::STORAGE | U::COPY_DST)?,
            create_buffer(desc.parameter_size, U::UNIFORM | U::COPY_DST)?,
            create_buffer(desc.output_size, U::STORAGE | U::COPY_SRC | U::COPY_DST)?,
            create_buffer(desc.input_size + desc.parameter_size, U::MAP_WRITE | U::COPY_SRC)?,
            create_buffer(desc.output_size, U::MAP_READ | U::COPY_DST)?,
        ];
        let binding_entry = |binding, ty| wgt::BindGroupLayoutEntry {
            binding,
            visibility: wgt::ShaderStages::COMPUTE,
            ty: wgt::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let entries = [
            binding_entry(0, wgt::BufferBindingType::Storage { read_only: true }),
            binding_entry(1, wgt::BufferBindingType::Storage { read_only: false }),
            binding_entry(2, wgt::BufferBindingType::Uniform),
        ];
        let bgl = device
            .create_bind_group_layout_inner(&binding::BindGroupLayoutDescriptor {
                label: None,
                entries: Cow::Borrowed(&entries),
            })
            .map_err(diagnostic)?;
        let layout = device
            .create_pipeline_layout_impl(
                &binding::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: Cow::Borrowed(&[Some(bgl.clone())]),
                    immediate_size: 0,
                },
                false,
            )
            .map_err(diagnostic)?;
        let shader = device
            .create_shader_module_inner(
                &pipeline::ShaderModuleDescriptor {
                    label: None,
                    runtime_checks: wgt::ShaderRuntimeChecks::default(),
                },
                pipeline::ShaderModuleSource::Wgsl(Cow::Borrowed(desc.shader)),
            )
            .map_err(diagnostic)?;
        let pipeline = device
            .create_compute_pipeline_or_error_inner(pipeline::ComputePipelineDescriptor {
                label: None,
                layout: Some(layout.clone()),
                stage: pipeline::ProgrammableStageDescriptor {
                    module: shader,
                    entry_point: Some(Cow::Borrowed(desc.entry_point)),
                    constants: Default::default(),
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            })
            .map_err(diagnostic)?;
        let buffer_entry = |binding, index: usize, size| binding::BindGroupEntry {
            binding,
            resource: binding::BindingResource::Buffer(binding::BufferBinding {
                buffer: buffers[index].clone(),
                offset: 0,
                size: Some(size),
            }),
        };
        let entries = [
            buffer_entry(0, 0, desc.input_size),
            buffer_entry(1, 2, desc.output_size),
            buffer_entry(2, 1, desc.parameter_size),
        ];
        let bind_group = device
            .create_bind_group_inner(&binding::BindGroupDescriptor {
                label: None,
                layout: bgl,
                entries: Cow::Borrowed(&entries),
            })
            .map_err(diagnostic)?;
        let resources = Self { buffers, pipeline, bind_group, layout, queue, device };
        resources.validate_commands(desc)?;
        if let Some(index) = resources.queue.flush_pending_writes().map_err(diagnostic)? {
            resources.device.poll(wgt::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            }).map_err(diagnostic)?;
        }
        resources.device.check_is_valid().map_err(diagnostic)?;
        Ok(resources)
    }

    fn validate_commands(&self, desc: &PreparedComputeDescriptor<'_>) -> Result<(), PrepareComputeError> {
        let encoder = self.device.create_command_encoder_inner(&None).map_err(diagnostic)?;
        encoder.copy_buffer_to_buffer(self.buffers[3].clone(), 0, self.buffers[0].clone(), 0, Some(desc.input_size));
        encoder.copy_buffer_to_buffer(self.buffers[3].clone(), desc.input_size, self.buffers[1].clone(), 0, Some(desc.parameter_size));
        encoder.clear_buffer(self.buffers[2].clone(), 0, Some(desc.output_size));
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(self.pipeline.clone());
        pass.set_bind_group(0, Some(self.bind_group.clone()), &[]);
        pass.dispatch_workgroups(desc.workgroups[0], desc.workgroups[1], desc.workgroups[2]);
        pass.end();
        encoder.copy_buffer_to_buffer(self.buffers[2].clone(), 0, self.buffers[4].clone(), 0, Some(desc.output_size));
        let (commands, error) = encoder.finish_inner(&wgt::CommandBufferDescriptor { label: None });
        if let Some(error) = error {
            return Err(diagnostic(error));
        }
        drop(commands.take_finished().map_err(diagnostic)?);
        Ok(())
    }

    pub(super) fn prepare_hal(
        &self,
        desc: &PreparedComputeDescriptor<'_>,
    ) -> Result<hal::vulkan::prepared::PreparedCompute, PrepareComputeError> {
        let guard = self.device.snatchable_lock.read();
        let buffer = |index: usize| {
            self.buffers[index].try_raw(&guard).map_err(diagnostic)?
                .as_any().downcast_ref::<hal::vulkan::Buffer>()
                .ok_or_else(|| PreparedComputeError::UnsupportedBackend.into())
        };
        let device = self.device.raw().as_any().downcast_ref::<hal::vulkan::Device>()
            .ok_or(PreparedComputeError::UnsupportedBackend)?;
        let queue = self.queue.raw().as_any().downcast_ref::<hal::vulkan::Queue>()
            .ok_or(PreparedComputeError::UnsupportedBackend)?;
        let pipeline = self.pipeline.raw().map_err(diagnostic)?.as_any()
            .downcast_ref::<hal::vulkan::ComputePipeline>()
            .ok_or(PreparedComputeError::UnsupportedBackend)?;
        let bind_group = self.bind_group.try_raw(&guard).map_err(diagnostic)?.as_any()
            .downcast_ref::<hal::vulkan::BindGroup>()
            .ok_or(PreparedComputeError::UnsupportedBackend)?;
        let layout = self.layout.raw().map_err(diagnostic)?.as_any()
            .downcast_ref::<hal::vulkan::PipelineLayout>()
            .ok_or(PreparedComputeError::UnsupportedBackend)?;
        // Normal core creation and command validation establish all HAL preconditions;
        // this owner seals the resources and private queue until retirement.
        unsafe {
            hal::vulkan::prepared::PreparedCompute::new(device, queue, &hal::vulkan::prepared::PreparedComputeDescriptor {
                input: buffer(0)?,
                parameters: buffer(1)?,
                output: buffer(2)?,
                upload: buffer(3)?,
                readback: buffer(4)?,
                pipeline,
                bind_group,
                pipeline_layout: layout,
                input_size: desc.input_size,
                parameter_size: desc.parameter_size,
                output_size: desc.output_size,
                workgroups: desc.workgroups,
            }).map_err(Into::into)
        }
    }
}
