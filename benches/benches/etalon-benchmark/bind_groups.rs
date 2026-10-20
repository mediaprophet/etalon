use std::{num::NonZeroU32, time::Instant};

use nanorand::{Rng, WyRand};
use etalon_benchmark::{iter, BenchmarkContext, SubBenchResult};

use crate::DeviceState;

struct Params {
    max_texture_count: u32,
    texture_counts: &'static [u32],
}

// Creating 50_000 textures takes a considerable amount of time with syncval enabled.
//
// We greatly reduce the number of textures for the test case to keep the runtime
// reasonable for testing.
const BENCHMARK_PARAMS: Params = Params {
    max_texture_count: 50_000,
    texture_counts: &[5, 50, 500, 5_000, 50_000],
};

const TEST_PARAMS: Params = Params {
    max_texture_count: 5,
    texture_counts: &[5],
};

pub fn run_bench(ctx: BenchmarkContext) -> anyhow::Result<Vec<SubBenchResult>> {
    let device_state = DeviceState::new();

    if !device_state
        .device
        .features()
        .contains(etalon::Features::TEXTURE_BINDING_ARRAY)
    {
        anyhow::bail!("Device does not support required feature TEXTURE_BINDING_ARRAY");
    }

    let params = if ctx.is_test() {
        TEST_PARAMS
    } else {
        BENCHMARK_PARAMS
    };

    // Performance gets considerably worse if the resources are shuffled.
    //
    // This more closely matches the real-world use case where resources have no
    // well defined usage order.
    let mut random = WyRand::new_seed(0x8BADF00D);

    let mut texture_views = Vec::with_capacity(params.max_texture_count as usize);
    for i in 0..params.max_texture_count {
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

    let mut results = Vec::new();

    for &count in params.texture_counts {
        let bind_group_layout =
            device_state
                .device
                .create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[etalon::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: etalon::ShaderStages::FRAGMENT,
                        ty: etalon::BindingType::Texture {
                            sample_type: etalon::TextureSampleType::Float { filterable: true },
                            view_dimension: etalon::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: Some(NonZeroU32::new(count).unwrap()),
                    }],
                });

        let texture_view_refs: Vec<_> = texture_views.iter().take(count as usize).collect();

        let name = format!("{count} Textures");

        let res = iter(&ctx, &name, "bindings", count, || {
            let start = Instant::now();
            let bind_group = device_state
                .device
                .create_bind_group(&etalon::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[etalon::BindGroupEntry {
                        binding: 0,
                        resource: etalon::BindingResource::TextureViewArray(&texture_view_refs),
                    }],
                    label: None,
                });

            let time = start.elapsed();

            drop(bind_group);
            device_state
                .device
                .poll(etalon::PollType::wait_indefinitely())
                .unwrap();

            time
        });

        results.push(res);
    }

    Ok(results)
}
