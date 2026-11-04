use crate::gpu::{bench_gpu_gemv, bench_upload_gbps};
use crate::measurement::gflops;
use crate::{CircuitBench, CircuitKind, DeviceBenchmarkRequest};

pub fn backend_name(backend: etalon::Backend) -> &'static str {
    match backend {
        etalon::Backend::Vulkan => "vulkan",
        etalon::Backend::Dx12 => "dx12",
        etalon::Backend::Metal => "metal",
        etalon::Backend::Gl => "gl",
        etalon::Backend::BrowserWebGpu => "browser-webgpu",
        etalon::Backend::Noop => "noop",
    }
}

fn backend_from_name(name: &str) -> Option<(etalon::Backend, etalon::Backends)> {
    match name {
        "vulkan" => Some((etalon::Backend::Vulkan, etalon::Backends::VULKAN)),
        "dx12" => Some((etalon::Backend::Dx12, etalon::Backends::DX12)),
        "metal" => Some((etalon::Backend::Metal, etalon::Backends::METAL)),
        "gl" => Some((etalon::Backend::Gl, etalon::Backends::GL)),
        _ => None,
    }
}

pub fn benchmark_one(request: &DeviceBenchmarkRequest) -> Result<CircuitBench, String> {
    super::measurement::validate_shape(request.gemv_n, super::measurement::DEFAULT_BENCH_BYTES)?;
    let (expected_backend, backends) = backend_from_name(&request.backend)
        .ok_or_else(|| format!("unsupported backend {}", request.backend))?;
    let mut descriptor = etalon::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    let instance = etalon::Instance::new(descriptor);
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    let adapter = adapters
        .into_iter()
        .find(|adapter| {
            let info = adapter.get_info();
            info.backend == expected_backend
                && info.vendor == request.vendor
                && info.device == request.device
        })
        .ok_or_else(|| {
            format!(
                "adapter {:04x}:{:04x}/{} unavailable",
                request.vendor, request.device, request.backend
            )
        })?;
    let info = adapter.get_info();
    if info.device_type == etalon::DeviceType::Cpu || info.backend == etalon::Backend::Noop {
        return Err("software adapter cannot supply a hardware GPU benchmark".into());
    }
    let (device, queue) =
        pollster::block_on(adapter.request_device(&etalon::DeviceDescriptor::default()))
            .map_err(|e| format!("request_device: {e}"))?;
    let ms = bench_gpu_gemv(&device, &queue, request.gemv_n)?;
    let upload_gbps = bench_upload_gbps(&device, &queue, 16 * 1024 * 1024)?;
    Ok(CircuitBench {
        label: format!("{} ({:?})", info.name, info.backend),
        kind: CircuitKind::from_wgpu(info.device_type),
        backend: format!("{:?}", info.backend),
        ms_per_gemv: ms,
        gflops: gflops(request.gemv_n, ms),
        upload_gbps,
        rel_score: 1.0,
        decode_proxy_tok_s: None,
    })
}
