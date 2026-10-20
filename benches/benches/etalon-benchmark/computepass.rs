use std::{
    num::{NonZeroU32, NonZeroU64},
    time::{Duration, Instant},
};

use nanorand::{Rng, WyRand};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use etalon_benchmark::{iter_auto, iter_many, BenchmarkContext, LoopControl};

use crate::DeviceState;

fn dispatch_count(ctx: &BenchmarkContext) -> usize {
    // When testing we only want to run a very lightweight version of the benchmark
    // to ensure that it does not break.
    if ctx.is_test() {
        8
    } else {
        10_000
    }
}

// Currently bindless is _much_ slower than with regularly resources,
// since etalon needs to issues barriers for all resources between each dispatch for all read/write textures & buffers.
// This is in fact so slow that it makes the benchmark unusable when we use the same amount of
// resources as the regular benchmark.
// For details see https://github.com/mediaprophet/etalon/issues/5766
fn dispatch_count_bindless(ctx: &BenchmarkContext) -> usize {
    // On CI we only want to run a very lightweight version of the benchmark
    // to ensure that it does not break.
    if ctx.is_test() {
        8
    } else {
        1_000
    }
}

fn thread_count_list(ctx: &BenchmarkContext) -> &'static [usize] {
    if ctx.is_test() {
        &[2]
    } else {
        &[2, 4, 8]
    }
}

// Must match the number of textures in the computepass.wgsl shader
const TEXTURES_PER_DISPATCH: usize = 2;
const STORAGE_TEXTURES_PER_DISPATCH: usize = 2;
const STORAGE_BUFFERS_PER_DISPATCH: usize = 2;

const BUFFER_SIZE: u64 = 16;

struct ComputepassState {
    device_state: DeviceState,
    pipeline: etalon::ComputePipeline,
    bind_groups: Vec<etalon::BindGroup>,

    // Bindless resources
    bindless_bind_group: Option<etalon::BindGroup>,
    bindless_pipeline: Option<etalon::ComputePipeline>,
}

impl ComputepassState {
    /// Create and prepare all the resources needed for the computepass benchmark.
    fn new(ctx: &BenchmarkContext) -> Self {
        let device_state = DeviceState::new();

        let dispatch_count = dispatch_count(ctx);
        let dispatch_count_bindless = dispatch_count_bindless(ctx);
        let texture_count = dispatch_count * TEXTURES_PER_DISPATCH;
        let storage_buffer_count = dispatch_count * STORAGE_BUFFERS_PER_DISPATCH;
        let storage_texture_count = dispatch_count * STORAGE_TEXTURES_PER_DISPATCH;

        let supports_bindless = device_state.device.features().contains(
            etalon::Features::BUFFER_BINDING_ARRAY
                | etalon::Features::TEXTURE_BINDING_ARRAY
                | etalon::Features::STORAGE_RESOURCE_BINDING_ARRAY
                | etalon::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        )
        // TODO: as of writing llvmpipe segfaults the bindless benchmark on ci
        && device_state.adapter_info.driver != "llvmpipe"
        // TODO(https://github.com/mediaprophet/etalon/issues/9849): Currently broken on Metal
        && device_state.adapter_info.backend != etalon::Backend::Metal;

        // Performance gets considerably worse if the resources are shuffled.
        //
        // This more closely matches the real-world use case where resources have no
        // well defined usage order.
        let mut random = WyRand::new_seed(0x8BADF00D);

        let mut bind_group_layout_entries = Vec::with_capacity(TEXTURES_PER_DISPATCH);
        for i in 0..TEXTURES_PER_DISPATCH {
            bind_group_layout_entries.push(etalon::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: etalon::ShaderStages::COMPUTE,
                ty: etalon::BindingType::Texture {
                    sample_type: etalon::TextureSampleType::Float { filterable: true },
                    view_dimension: etalon::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        for i in 0..STORAGE_TEXTURES_PER_DISPATCH {
            bind_group_layout_entries.push(etalon::BindGroupLayoutEntry {
                binding: (TEXTURES_PER_DISPATCH + i) as u32,
                visibility: etalon::ShaderStages::COMPUTE,
                ty: etalon::BindingType::StorageTexture {
                    access: etalon::StorageTextureAccess::ReadWrite,
                    format: etalon::TextureFormat::R32Float,
                    view_dimension: etalon::TextureViewDimension::D2,
                },
                count: None,
            });
        }
        for i in 0..STORAGE_BUFFERS_PER_DISPATCH {
            bind_group_layout_entries.push(etalon::BindGroupLayoutEntry {
                binding: (TEXTURES_PER_DISPATCH + STORAGE_BUFFERS_PER_DISPATCH + i) as u32,
                visibility: etalon::ShaderStages::COMPUTE,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(BUFFER_SIZE),
                },
                count: None,
            });
        }

        let bind_group_layout =
            device_state
                .device
                .create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &bind_group_layout_entries,
                });

        let mut texture_views = Vec::with_capacity(texture_count);
        for i in 0..texture_count {
            let texture = device_state
                .device
                .create_texture(&etalon::TextureDescriptor {
                    label: Some(&format!("Texture {i}")),
                    size: etalon::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: etalon::TextureDimension::D2,
                    format: etalon::TextureFormat::Rgba8UnormSrgb,
                    usage: etalon::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
            texture_views.push(texture.create_view(&etalon::TextureViewDescriptor {
                label: Some(&format!("Texture View {i}")),
                ..Default::default()
            }));
        }
        random.shuffle(&mut texture_views);
        let texture_view_refs: Vec<_> = texture_views.iter().collect();

        let mut storage_texture_views = Vec::with_capacity(storage_texture_count);
        for i in 0..storage_texture_count {
            let texture = device_state
                .device
                .create_texture(&etalon::TextureDescriptor {
                    label: Some(&format!("StorageTexture {i}")),
                    size: etalon::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: etalon::TextureDimension::D2,
                    format: etalon::TextureFormat::R32Float,
                    usage: etalon::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                });
            storage_texture_views.push(texture.create_view(&etalon::TextureViewDescriptor {
                label: Some(&format!("StorageTexture View {i}")),
                ..Default::default()
            }));
        }
        random.shuffle(&mut storage_texture_views);
        let storage_texture_view_refs: Vec<_> = storage_texture_views.iter().collect();

        let mut storage_buffers = Vec::with_capacity(storage_buffer_count);
        for i in 0..storage_buffer_count {
            storage_buffers.push(device_state.device.create_buffer(&etalon::BufferDescriptor {
                label: Some(&format!("Buffer {i}")),
                size: BUFFER_SIZE,
                usage: etalon::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }
        random.shuffle(&mut storage_buffers);
        let storage_buffer_bindings: Vec<_> = storage_buffers
            .iter()
            .map(|b| b.as_entire_buffer_binding())
            .collect();

        let mut bind_groups = Vec::with_capacity(dispatch_count);
        for dispatch_idx in 0..dispatch_count {
            let mut entries = Vec::with_capacity(TEXTURES_PER_DISPATCH);
            for tex_idx in 0..TEXTURES_PER_DISPATCH {
                entries.push(etalon::BindGroupEntry {
                    binding: tex_idx as u32,
                    resource: etalon::BindingResource::TextureView(
                        &texture_views[dispatch_idx * TEXTURES_PER_DISPATCH + tex_idx],
                    ),
                });
            }
            for tex_idx in 0..STORAGE_TEXTURES_PER_DISPATCH {
                entries.push(etalon::BindGroupEntry {
                    binding: (TEXTURES_PER_DISPATCH + tex_idx) as u32,
                    resource: etalon::BindingResource::TextureView(
                        &storage_texture_views
                            [dispatch_idx * STORAGE_TEXTURES_PER_DISPATCH + tex_idx],
                    ),
                });
            }
            for buffer_idx in 0..STORAGE_BUFFERS_PER_DISPATCH {
                entries.push(etalon::BindGroupEntry {
                    binding: (TEXTURES_PER_DISPATCH + STORAGE_BUFFERS_PER_DISPATCH + buffer_idx)
                        as u32,
                    resource: etalon::BindingResource::Buffer(
                        storage_buffers[dispatch_idx * STORAGE_BUFFERS_PER_DISPATCH + buffer_idx]
                            .as_entire_buffer_binding(),
                    ),
                });
            }

            bind_groups.push(
                device_state
                    .device
                    .create_bind_group(&etalon::BindGroupDescriptor {
                        label: None,
                        layout: &bind_group_layout,
                        entries: &entries,
                    }),
            );
        }
        random.shuffle(&mut bind_groups);

        let sm = device_state
            .device
            .create_shader_module(etalon::include_wgsl!("computepass.wgsl"));

        let pipeline_layout =
            device_state
                .device
                .create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let pipeline =
            device_state
                .device
                .create_compute_pipeline(&etalon::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &sm,
                    entry_point: Some("cs_main"),
                    compilation_options: etalon::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let (bindless_bind_group, bindless_pipeline) = if supports_bindless {
            let bindless_bind_group_layout =
                device_state
                    .device
                    .create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            etalon::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: etalon::ShaderStages::COMPUTE,
                                ty: etalon::BindingType::Texture {
                                    sample_type: etalon::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: etalon::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: Some(NonZeroU32::new(texture_count as u32).unwrap()),
                            },
                            etalon::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: etalon::ShaderStages::COMPUTE,
                                ty: etalon::BindingType::StorageTexture {
                                    access: etalon::StorageTextureAccess::ReadWrite,
                                    format: etalon::TextureFormat::R32Float,
                                    view_dimension: etalon::TextureViewDimension::D2,
                                },
                                count: Some(NonZeroU32::new(storage_texture_count as u32).unwrap()),
                            },
                            etalon::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: etalon::ShaderStages::COMPUTE,
                                ty: etalon::BindingType::Buffer {
                                    ty: etalon::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: std::num::NonZeroU64::new(BUFFER_SIZE),
                                },
                                count: Some(NonZeroU32::new(storage_buffer_count as u32).unwrap()),
                            },
                        ],
                    });

            let bindless_bind_group =
                device_state
                    .device
                    .create_bind_group(&etalon::BindGroupDescriptor {
                        label: None,
                        layout: &bindless_bind_group_layout,
                        entries: &[
                            etalon::BindGroupEntry {
                                binding: 0,
                                resource: etalon::BindingResource::TextureViewArray(
                                    &texture_view_refs[..dispatch_count_bindless],
                                ),
                            },
                            etalon::BindGroupEntry {
                                binding: 1,
                                resource: etalon::BindingResource::TextureViewArray(
                                    &storage_texture_view_refs[..dispatch_count_bindless],
                                ),
                            },
                            etalon::BindGroupEntry {
                                binding: 2,
                                resource: etalon::BindingResource::BufferArray(
                                    &storage_buffer_bindings[..dispatch_count_bindless],
                                ),
                            },
                        ],
                    });

            let bindless_sm = device_state
                .device
                .create_shader_module(etalon::include_wgsl!("computepass-bindless.wgsl"));

            let bindless_pipeline_layout =
                device_state
                    .device
                    .create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&bindless_bind_group_layout)],
                        immediate_size: 0,
                    });

            let bindless_pipeline =
                device_state
                    .device
                    .create_compute_pipeline(&etalon::ComputePipelineDescriptor {
                        label: Some("Compute Pipeline bindless"),
                        layout: Some(&bindless_pipeline_layout),
                        module: &bindless_sm,
                        entry_point: Some("cs_main"),
                        compilation_options: etalon::PipelineCompilationOptions::default(),
                        cache: None,
                    });

            (Some(bindless_bind_group), Some(bindless_pipeline))
        } else {
            (None, None)
        };

        Self {
            device_state,
            pipeline,
            bind_groups,

            bindless_bind_group,
            bindless_pipeline,
        }
    }

    fn run_subpass(
        &self,
        ctx: &BenchmarkContext,
        pass_number: usize,
        total_passes: usize,
    ) -> etalon::CommandBuffer {
        profiling::scope!("Computepass", &format!("Pass {pass_number}/{total_passes}"));

        let dispatch_count = dispatch_count(ctx);
        let dispatch_per_pass = dispatch_count / total_passes;

        let mut encoder = self
            .device_state
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        let mut compute_pass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        let start_idx = pass_number * dispatch_per_pass;
        let end_idx = start_idx + dispatch_per_pass;
        for dispatch_idx in start_idx..end_idx {
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_groups[dispatch_idx], &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        drop(compute_pass);

        encoder.finish()
    }

    fn run_bindless_pass(&self, dispatch_count_bindless: usize) -> etalon::CommandBuffer {
        profiling::scope!("Bindless Computepass");

        let mut encoder = self
            .device_state
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        let mut compute_pass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(self.bindless_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, Some(self.bindless_bind_group.as_ref().unwrap()), &[]);
        for _ in 0..dispatch_count_bindless {
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        drop(compute_pass);

        encoder.finish()
    }
}

pub fn run_bench(mut ctx: BenchmarkContext) -> anyhow::Result<Vec<etalon_benchmark::SubBenchResult>> {
    let state = ComputepassState::new(&ctx);

    ctx.default_iterations = LoopControl::Time(Duration::from_secs(3));

    // This benchmark hangs on Apple Paravirtualized GPUs. No idea why.
    if state.device_state.adapter_info.name.contains("Paravirtual") {
        anyhow::bail!("Benchmark unsupported on Paravirtualized GPUs");
    }

    let dispatch_count = dispatch_count(&ctx);
    let dispatch_count_bindless = dispatch_count_bindless(&ctx);

    let mut results = Vec::new();

    // Test 10k dispatch calls split up into 1, 2, 4, and 8 computepasses
    for &cpasses in thread_count_list(&ctx) {
        let labels = vec![
            format!("Encoding ({cpasses} passes)"),
            format!("Submit ({cpasses} passes)"),
        ];

        results.extend(iter_many(
            &ctx,
            labels,
            "dispatches",
            dispatch_count as _,
            || {
                let mut buffers: Vec<etalon::CommandBuffer> = Vec::with_capacity(cpasses);
                let encoding_start = Instant::now();
                for i in 0..cpasses {
                    buffers.push(state.run_subpass(&ctx, i, cpasses));
                }
                let encoding_duration = encoding_start.elapsed();

                let submit_start = Instant::now();
                state.device_state.queue.submit(buffers);
                let submit_duration = submit_start.elapsed();

                state
                    .device_state
                    .device
                    .poll(etalon::PollType::wait_indefinitely())
                    .unwrap();

                vec![encoding_duration, submit_duration]
            },
        ));
    }

    // Test 10k dispatch calls split up over 2, 4, and 8 threads.
    for &threads in thread_count_list(&ctx) {
        let labels = vec![
            format!("Encoding ({threads} threads)"),
            format!("Submit ({threads} threads)"),
        ];

        results.extend(iter_many(
            &ctx,
            labels,
            "dispatches",
            dispatch_count as _,
            || {
                let encoding_start = Instant::now();
                let buffers = (0..threads)
                    .into_par_iter()
                    .map(|i| state.run_subpass(&ctx, i, threads))
                    .collect::<Vec<_>>();
                let encoding_duration = encoding_start.elapsed();

                let submit_start = Instant::now();
                state.device_state.queue.submit(buffers);
                let submit_duration = submit_start.elapsed();

                state
                    .device_state
                    .device
                    .poll(etalon::PollType::wait_indefinitely())
                    .unwrap();

                vec![encoding_duration, submit_duration]
            },
        ));
    }

    // Test 10k dispatch calls with bindless rendering.
    if state.bindless_bind_group.is_some() {
        let labels = vec![
            "Encoding (bindless)".to_string(),
            "Submit (bindless)".to_string(),
        ];

        results.extend(iter_many(
            &ctx,
            labels,
            "dispatches",
            dispatch_count_bindless as _,
            || {
                let encoding_start = Instant::now();
                let buffer = state.run_bindless_pass(dispatch_count_bindless);
                let encoding_duration = encoding_start.elapsed();

                let submit_start = Instant::now();
                state.device_state.queue.submit([buffer]);
                let submit_duration = submit_start.elapsed();

                state
                    .device_state
                    .device
                    .poll(etalon::PollType::wait_indefinitely())
                    .unwrap();

                vec![encoding_duration, submit_duration]
            },
        ));
    }

    // Test empty submit overhead with all resources
    let texture_count = dispatch_count * TEXTURES_PER_DISPATCH;
    let storage_buffer_count = dispatch_count * STORAGE_BUFFERS_PER_DISPATCH;
    let storage_texture_count = dispatch_count * STORAGE_TEXTURES_PER_DISPATCH;

    results.push(iter_auto(
        &ctx,
        &format!(
            "Empty Submit with {} Resources",
            texture_count + storage_texture_count + storage_buffer_count
        ),
        "submits",
        1,
        || {
            state.device_state.queue.submit([]);
        },
    ));

    Ok(results)
}
