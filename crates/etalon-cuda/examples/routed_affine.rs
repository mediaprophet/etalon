#[cfg(not(target_family = "wasm"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use etalon_cuda::{CudaContext, CudaError, CudaKernels};
    use etalon_wgsl_forge::{
        emit_routed_shader, BuiltinKernel, RouteCapabilities, RoutePreference, Schedule,
        TargetBackend,
    };

    let preference = match std::env::args().nth(1).as_deref() {
        None | Some("auto") => RoutePreference::PreferCuda,
        Some("cuda") => RoutePreference::RequireCuda,
        Some("vulkan") => RoutePreference::Portable,
        _ => return Err("usage: routed_affine [auto|cuda|vulkan]".into()),
    };
    let ordinal = match std::env::var("ETALON_CUDA_DEVICE") {
        Ok(value) => value.parse::<i32>()?,
        Err(std::env::VarError::NotPresent) => 0,
        Err(error) => return Err(error.into()),
    };
    let instance = etalon::Instance::new(etalon::InstanceDescriptor {
        backends: etalon::Backends::VULKAN,
        flags: etalon::InstanceFlags::empty(),
        ..etalon::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(etalon::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))?;
    let mut capabilities = RouteCapabilities::default();
    let mut cuda = None;
    if preference != RoutePreference::Portable {
        match CudaContext::new(ordinal) {
            Ok(context) => {
                let selected = adapter.get_info().device_pci_bus_id;
                capabilities.cuda.driver_available = true;
                capabilities.cuda.compute_capability =
                    Some(context.device_info().compute_capability);
                capabilities.cuda.selected_device_matches = !selected.is_empty()
                    && context
                        .device_info()
                        .pci_bus_id
                        .eq_ignore_ascii_case(&selected);
                if capabilities.cuda.selected_device_matches {
                    match CudaKernels::new(context) {
                        Ok(kernels) => {
                            capabilities.cuda.compiler_available = true;
                            cuda = Some(kernels);
                        }
                        Err(error @ CudaError::Compilation(_))
                            if preference == RoutePreference::PreferCuda =>
                        {
                            eprintln!(
                                "CUDA compiler unavailable, selecting portable execution: {error}"
                            );
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            Err(error) if preference == RoutePreference::RequireCuda => return Err(error.into()),
            Err(error) => {
                eprintln!("CUDA driver/device unavailable, selecting portable execution: {error}");
            }
        }
    }
    let route = emit_routed_shader(
        &BuiltinKernel::AffineF32.spec(),
        Schedule::from_1d(256, 1, 1),
        preference,
        capabilities,
    )?;
    println!(
        "Forge source: {:?}; route: {:?}",
        route.target, route.reason
    );
    let input: Vec<f32> = (0..513).map(|value| value as f32 / 16.0).collect();
    let output = if route.target == TargetBackend::CudaC {
        let kernels = cuda
            .as_ref()
            .ok_or("selected CUDA without an execution owner")?;
        println!("Executed on {:?}", kernels.context().device_info());
        kernels.affine_f32(&input, 1.25, -0.75)?
    } else {
        println!("Executed on {:?}", adapter.get_info());
        let bytes = (input.len() * std::mem::size_of::<f32>()) as u64;
        let mut prepared = adapter.prepare_compute(&etalon::PreparedComputeDescriptor {
            shader: &route.shader.source,
            entry_point: "affine_f32",
            input_size: bytes,
            parameter_size: 16,
            output_size: bytes,
            workgroups: [(input.len() as u32).div_ceil(256), 1, 1],
            buffer_budget: 2 * (bytes * 2 + 16),
        })?;
        let mut output = vec![0.0f32; input.len()];
        let parameters = [
            input.len() as u32,
            1.25f32.to_bits(),
            (-0.75f32).to_bits(),
            0,
        ];
        prepared.execute(
            bytemuck::cast_slice(&input),
            bytemuck::cast_slice(&parameters),
            bytemuck::cast_slice_mut(&mut output),
            std::time::Duration::from_secs(10),
        )?;
        prepared.retire()?;
        output
    };
    for (actual, input) in output.iter().zip(&input) {
        assert!((actual - (input * 1.25 - 0.75)).abs() < 1e-5);
    }
    println!(
        "Verified {} results against the CPU reference",
        output.len()
    );
    Ok(())
}

#[cfg(target_family = "wasm")]
fn main() {}
