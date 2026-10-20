use std::sync::Arc;

use etalon_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([TEST_SINGLE_WRITE, TEST_SCATTER]);
}

#[apply(gpu_test!)]
static TEST_SINGLE_WRITE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| async move { run_test(ctx, false).await });

#[apply(gpu_test!)]
static TEST_SCATTER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            // See https://github.com/mediaprophet/etalon/issues/6827
            .expect_fail(FailureCase::backend_adapter(
                etalon::Backends::METAL,
                "Apple M", // M1,M2 etc
            ))
            .expect_fail(FailureCase::backend_adapter(
                etalon::Backends::METAL,
                "Apple Paravirtual device", // CI on M1
            )),
    )
    .run_async(|ctx| async move { run_test(ctx, true).await });

async fn run_test(ctx: TestingContext, use_many_writes: bool) {
    let device = ctx.device;
    let queue = ctx.queue;

    // Deliberately distinct in every dimension so that an axis swap (width vs.
    // height vs. depth) in the copy or the readback would be caught.
    let size = etalon::Extent3d {
        width: 4,
        height: 5,
        depth_or_array_layers: 6,
    };
    let texture = {
        device.create_texture(&etalon::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D3,
            format: etalon::TextureFormat::Rgba8Uint,
            view_formats: &[],
            usage: etalon::TextureUsages::TEXTURE_BINDING
                | etalon::TextureUsages::COPY_DST
                | etalon::TextureUsages::COPY_SRC,
            label: None,
        })
    };

    if use_many_writes {
        many_writes(&texture, &device, &queue);
    } else {
        single_write(&texture, &queue);
    }

    let light_texels: Vec<[u8; 4]> = {
        let tc = TextureCopyParameters::from_texture(&texture);
        let temp_buffer = tc.copy_texture_to_new_buffer(&device, &queue, &texture);

        let result_cell =
            Arc::new(std::sync::OnceLock::<Result<(), etalon::BufferAsyncError>>::new());
        temp_buffer.slice(..).map_async(etalon::MapMode::Read, {
            let result_cell = result_cell.clone();
            move |result| result_cell.set(result).unwrap()
        });
        device.poll(etalon::PollType::wait_indefinitely()).unwrap();
        result_cell
            .get()
            .as_ref()
            .expect("cell not set")
            .as_ref()
            .expect("mapping failed");

        tc.copy_mapped_to_vec(1, &temp_buffer)
    };

    let mut wrong_texels = Vec::new();
    for (zyx_index, cube) in texel_iter(&texture).enumerate() {
        #[allow(clippy::cast_possible_wrap)]
        let expected = texel_for_cube(cube);
        let actual = light_texels[zyx_index];
        if expected != actual {
            println!("{:?}", (cube, expected, actual));
            wrong_texels.push((cube, expected, actual));
        }
    }

    let volume = size.width * size.height * size.depth_or_array_layers;
    assert!(
        wrong_texels.is_empty(),
        "out of {volume}, {len} were wrong",
        len = wrong_texels.len(),
    );
}

// -------------------------------------------------------------------------------------------------

type Texel = [u8; COMPONENTS];

pub fn texel_for_cube(point: [u32; 3]) -> [u8; 4] {
    [
        10 + point[0] as u8,
        10 + point[1] as u8,
        10 + point[2] as u8,
        10,
    ]
}

const COMPONENTS: usize = 4;

fn texel_iter(texture: &etalon::Texture) -> impl Iterator<Item = [u32; 3]> {
    itertools::iproduct!(
        0..texture.depth_or_array_layers(),
        0..texture.height(),
        0..texture.width()
    )
    .map(|(z, y, x)| [x, y, z])
}

fn compute_data(texture: &etalon::Texture) -> Vec<Texel> {
    let mut data = Vec::new();
    for point in texel_iter(texture) {
        data.push(texel_for_cube(point));
    }
    data
}

pub fn single_write(texture: &etalon::Texture, queue: &etalon::Queue) {
    let data = compute_data(texture);

    queue.write_texture(
        etalon::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect: etalon::TextureAspect::All,
        },
        data.as_flattened(),
        etalon::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(texture.width() * COMPONENTS as u32),
            rows_per_image: Some(texture.height()),
        },
        texture.size(),
    )
}

pub fn many_writes(texture: &etalon::Texture, device: &etalon::Device, queue: &etalon::Queue) {
    let data: Vec<Texel> = compute_data(texture);

    let copy_buffer_2 = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: u64::try_from(data.len() * COMPONENTS).unwrap(),
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());

    for (index, cube) in texel_iter(texture).enumerate() {
        encoder.copy_buffer_to_texture(
            etalon::TexelCopyBufferInfo {
                buffer: &copy_buffer_2,
                layout: etalon::TexelCopyBufferLayout {
                    offset: (index * COMPONENTS) as u64,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            etalon::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: etalon::Origin3d {
                    x: cube[0],
                    y: cube[1],
                    z: cube[2],
                },
                aspect: etalon::TextureAspect::All,
            },
            etalon::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    queue.write_buffer(&copy_buffer_2, 0, data.as_flattened());
    queue.submit([encoder.finish()]);
}

// -------------------------------------------------------------------------------------------------

/// Elements of GPU-to-CPU copying
#[derive(Clone, Copy, Debug)]
struct TextureCopyParameters {
    pub size: etalon::Extent3d,
    pub byte_size_of_texel: u32,
}
impl TextureCopyParameters {
    pub fn from_texture(texture: &etalon::Texture) -> Self {
        let format = texture.format();
        assert_eq!(
            format.block_dimensions(),
            (1, 1),
            "compressed texture format {format:?} not supported",
        );

        Self {
            size: texture.size(),
            byte_size_of_texel: format
                .block_copy_size(None)
                .expect("non-color texture format {format:} not supported"),
        }
    }

    pub fn dense_bytes_per_row(&self) -> u32 {
        self.size.width * self.byte_size_of_texel
    }

    pub fn padded_bytes_per_row(&self) -> u32 {
        self.dense_bytes_per_row()
            .div_ceil(etalon::COPY_BYTES_PER_ROW_ALIGNMENT)
            * etalon::COPY_BYTES_PER_ROW_ALIGNMENT
    }

    #[track_caller]
    pub fn copy_texture_to_new_buffer(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        texture: &etalon::Texture,
    ) -> etalon::Buffer {
        let padded_bytes_per_row = self.padded_bytes_per_row();

        let temp_buffer = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("GPU-to-CPU image copy buffer"),
            size: u64::from(padded_bytes_per_row)
                * u64::from(self.size.height)
                * u64::from(self.size.depth_or_array_layers),
            usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        {
            let mut encoder =
                device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                etalon::TexelCopyBufferInfo {
                    buffer: &temp_buffer,
                    layout: etalon::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row),
                        rows_per_image: Some(self.size.height),
                    },
                },
                texture.size(),
            );
            queue.submit(Some(encoder.finish()));
        }

        temp_buffer
    }

    /// Given a mapped buffer, make a [`Vec<C>`] of it.
    ///
    /// `size_of::<C>() * components` must be equal to the byte size of a texel.
    pub fn copy_mapped_to_vec<C>(&self, components: usize, buffer: &etalon::Buffer) -> Vec<C>
    where
        C: bytemuck::AnyBitPattern,
    {
        assert_eq!(
            u32::try_from(components * size_of::<C>()).ok(),
            Some(self.byte_size_of_texel),
            "Texture format does not match requested format",
        );

        // Copy the mapped buffer data into a Rust vector, removing row padding if present
        // by copying it one row at a time.
        let mut texel_vector: Vec<C> = Vec::new();
        {
            let mapped: &[u8] = &buffer.slice(..).get_mapped_range().unwrap();
            for row in 0..self.row_count() {
                let byte_start_of_row = (self.padded_bytes_per_row()) as usize * row;
                texel_vector.extend(bytemuck::cast_slice::<u8, C>(
                    &mapped[byte_start_of_row..][..self.dense_bytes_per_row() as usize],
                ));
            }
        }

        texel_vector
    }

    fn row_count(&self) -> usize {
        self.size.height as usize * self.size.depth_or_array_layers as usize
    }
}
