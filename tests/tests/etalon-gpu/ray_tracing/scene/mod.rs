use std::{iter, mem};

use etalon_test::{apply, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

use etalon::util::DeviceExt;

use crate::ray_tracing::acceleration_structure_limits;
use glam::{Affine3A, Quat, Vec3};

mod mesh_gen;

pub fn all_tests(tests: &mut Vec<etalon_test::GpuTestInitializer>) {
    tests.extend([
        ACCELERATION_STRUCTURE_BUILD_NO_INDEX,
        ACCELERATION_STRUCTURE_BUILD_WITH_INDEX,
    ]);
}

fn acceleration_structure_build(ctx: &TestingContext, use_index_buffer: bool) {
    let max_instances = 1000;
    let device = &ctx.device;

    let (vertex_data, index_data) = mesh_gen::create_vertices();

    let vertex_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: etalon::BufferUsages::VERTEX | etalon::BufferUsages::BLAS_INPUT,
    });

    let index_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&index_data),
        usage: etalon::BufferUsages::INDEX | etalon::BufferUsages::BLAS_INPUT,
    });

    let index_format = etalon::IndexFormat::Uint16;
    let index_count = index_data.len() as u32;

    let blas_geo_size_desc = etalon::BlasTriangleGeometrySizeDescriptor {
        vertex_format: etalon::VertexFormat::Float32x3,
        vertex_count: vertex_data.len() as u32,
        index_format: use_index_buffer.then_some(index_format),
        index_count: use_index_buffer.then_some(index_count),
        flags: etalon::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = device.create_blas(
        &etalon::CreateBlasDescriptor {
            label: None,
            flags: etalon::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: etalon::AccelerationStructureUpdateMode::Build,
        },
        etalon::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_geo_size_desc.clone()],
        },
    );

    let mut tlas = device.create_tlas(&etalon::CreateTlasDescriptor {
        label: None,
        flags: etalon::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: etalon::AccelerationStructureUpdateMode::Build,
        max_instances,
    });

    for j in 0..max_instances {
        tlas[j as usize] = Some(etalon::TlasInstance::new(
            &blas,
            mesh_gen::affine_to_rows(&Affine3A::from_rotation_translation(
                Quat::from_rotation_y(45.9_f32.to_radians()),
                Vec3 {
                    x: j as f32,
                    y: 0.0,
                    z: 0.0,
                },
            )),
            0,
            0xff,
        ));
    }

    let mut encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

    encoder.build_acceleration_structures(
        iter::once(&etalon::BlasBuildEntry {
            blas: &blas,
            geometry: etalon::BlasGeometries::TriangleGeometries(vec![etalon::BlasTriangleGeometry {
                size: &blas_geo_size_desc,
                vertex_buffer: &vertex_buf,
                first_vertex: 0,
                vertex_stride: mem::size_of::<mesh_gen::Vertex>() as u64,
                index_buffer: use_index_buffer.then_some(&index_buffer),
                first_index: use_index_buffer.then_some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        iter::once(&tlas),
    );

    ctx.queue.submit(Some(encoder.finish()));

    ctx.device
        .poll(etalon::PollType::wait_indefinitely())
        .unwrap();
}

#[apply(gpu_test!)]
static ACCELERATION_STRUCTURE_BUILD_NO_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(etalon::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        acceleration_structure_build(&ctx, false);
    });

#[apply(gpu_test!)]
static ACCELERATION_STRUCTURE_BUILD_WITH_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(etalon::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        acceleration_structure_build(&ctx, true);
    });
