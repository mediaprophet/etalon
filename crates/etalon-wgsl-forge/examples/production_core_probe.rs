#[cfg(feature = "runtime")]
mod probe {
    use etalon_wgsl_forge::numerical::execute::{BindingUsage, OracleContext, WgpuComputeContext};
    use etalon_wgsl_forge::numerical::{
        certify_builtin, generate_builtin, validate_wgsl, AffineParams, BuiltinKernel, ForgeError,
        Schedule, TargetBackend, ValidationLevel,
    };
    use serde_json::{json, Value};
    use std::{
        fs,
        io::{Read, Write},
        path::Path,
        time::Instant,
    };

    fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        file.write_all(bytes)
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    fn write_json(dir: &Path, name: &str, value: &Value) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
        write_new(&dir.join(name), &bytes)
    }

    fn elapsed_ns(start: Instant) -> u64 {
        start.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }

    fn asset_phase(
        context: &mut WgpuComputeContext,
        schedule: Schedule,
        dir: &Path,
        vertices_path: &str,
        asset_sha256: &str,
    ) -> Result<Value, String> {
        let stage = Instant::now();
        let mut bytes = Vec::new();
        fs::File::open(vertices_path)
            .map_err(|e| e.to_string())?
            .take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > 4 * 1024 * 1024 {
            return Err("vertices JSON exceeds 4 MiB".into());
        }
        let positions: Vec<[f32; 3]> = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let count = positions
            .len()
            .checked_mul(3)
            .ok_or("scalar count overflow")?;
        if count == 0 || count > 65536 || positions.iter().flatten().any(|x| !x.is_finite()) {
            return Err(
                "vertices must contain 1..=65536 finite scalar values in xyz triples".into(),
            );
        }
        let input: Vec<f32> = positions.iter().flatten().copied().collect();
        let params = AffineParams {
            length: count as u32,
            scale: 2.0,
            bias: 0.25,
            _pad: 0,
        };
        let kernel = BuiltinKernel::AffineF32.spec();
        schedule
            .validate(&kernel, &context.constraints)
            .map_err(|e| e.to_string())?;
        context.clear_transient_allocations();
        let input_view = context
            .allocate_and_write(
                bytemuck::cast_slice(&input),
                0,
                0,
                BindingUsage::StorageRead,
            )
            .map_err(|e| e.to_string())?;
        let output_view = context
            .allocate_transient(count * 4, 1, 0, BindingUsage::StorageReadWrite)
            .map_err(|e| e.to_string())?;
        let params_view = context
            .allocate_and_write(bytemuck::bytes_of(&params), 2, 0, BindingUsage::Uniform)
            .map_err(|e| e.to_string())?;
        let samples = context
            .run_kernel(
                &kernel,
                &schedule,
                &[input_view, output_view, params_view],
                count,
                0,
                1,
            )
            .map_err(|e| e.to_string())?;
        let output = context
            .read_buffer_f32(&output_view)
            .map_err(|e| e.to_string())?;
        if output.len() != count {
            return Err("asset output length mismatch".into());
        }
        let mut max_absolute_error = 0.0f64;
        let mut mismatches = 0usize;
        for (&x, &actual) in input.iter().zip(&output) {
            let expected = (x as f64) * 2.0 + 0.25;
            let absolute = (expected - actual as f64).abs();
            let relative = absolute
                / expected
                    .abs()
                    .max((actual as f64).abs())
                    .max(f64::MIN_POSITIVE);
            max_absolute_error = max_absolute_error.max(absolute);
            if !actual.is_finite() || !(absolute <= 1.0e-6 || relative <= 1.0e-5) {
                mismatches += 1;
            }
        }
        let derived: Vec<[f32; 3]> = output.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        let result = json!({
            "outcome":if mismatches == 0 {"passed"} else {"failed"},
            "source_asset_sha256":asset_sha256,
            "source_asset_binding":"caller supplied; source-to-decoded-vertices join must be independently verified",
            "vertices_json_blake3":blake3::hash(&bytes).to_hex().to_string(),
            "decoded_f32_blake3":blake3::hash(bytemuck::cast_slice(&input)).to_hex().to_string(),
            "output_f32_blake3":blake3::hash(bytemuck::cast_slice(&output)).to_hex().to_string(),
            "vertex_count":positions.len(), "compared":count, "mismatch_count":mismatches,
            "scale":2.0, "bias":0.25, "reference":"independent scalar f64 multiply then addition",
            "absolute_tolerance":1.0e-6, "relative_tolerance":1.0e-5,
            "maximum_absolute_error":max_absolute_error,
            "dispatch_sample_ns":samples, "wall_ns":elapsed_ns(stage),
            "qualification":"one dispatch differential tracer; seeded kernel certification is separate"
        });
        if mismatches == 0 {
            write_json(dir, "asset-affine-positions.json", &json!(derived))?;
        }
        write_json(dir, "asset-affine.json", &result)?;
        context.clear_transient_allocations();
        if mismatches != 0 {
            return Err(format!("asset affine has {mismatches} mismatches"));
        }
        Ok(result)
    }
    fn execute(
        dir: &Path,
        expected_adapter: Option<&str>,
        vertices_path: Option<&str>,
        asset_sha256: Option<&str>,
        report: &mut Value,
    ) -> Result<(), (bool, String)> {
        let schedule = Schedule {
            workgroup_size: 64,
            items_per_invocation: 2,
            vector_width: 4,
        };
        let stage = Instant::now();
        let generated = generate_builtin(BuiltinKernel::AffineF32, schedule, TargetBackend::Wgsl)
            .map_err(|e| (false, e.to_string()))?;
        report["stages"]
            .as_array_mut()
            .unwrap()
            .push(json!({"stage":"generated", "wall_ns":elapsed_ns(stage)}));
        report["source_hash"] = json!(generated.source_hash);
        report["semantic_hash"] = json!(generated.semantic_hash);
        write_new(&dir.join("affine.wgsl"), generated.source.as_bytes()).map_err(|e| (false, e))?;

        let stage = Instant::now();
        let validation = validate_wgsl(&generated.source).map_err(|e| (false, e.to_string()))?;
        if !validation.naga_validated || validation.source_hash != generated.source_hash {
            return Err((
                false,
                "Naga validation evidence does not bind generated source".into(),
            ));
        }
        report["stages"]
            .as_array_mut()
            .unwrap()
            .push(json!({"stage":"naga_validated", "wall_ns":elapsed_ns(stage)}));
        write_json(dir, "validation.json", &json!(validation)).map_err(|e| (false, e))?;

        let stage = Instant::now();
        let mut context = WgpuComputeContext::new(1024 * 1024).map_err(|e| {
            let unsupported = matches!(e, ForgeError::GpuUnavailable(_));
            (unsupported, e.to_string())
        })?;
        report["device_admission_wall_ns"] = json!(elapsed_ns(stage));
        report["adapter"] = json!(context.adapter);
        write_json(dir, "hardware.json", &json!(context.profile)).map_err(|e| (false, e))?;
        if let Some(expected) = expected_adapter {
            if !context
                .adapter
                .name
                .to_lowercase()
                .contains(&expected.to_lowercase())
            {
                return Err((
                    true,
                    format!(
                        "selected adapter {:?} does not match {:?}",
                        context.adapter.name, expected
                    ),
                ));
            }
        }

        let stage = Instant::now();
        let pipeline = context
            .compile_pipeline(
                &generated.source,
                &BuiltinKernel::AffineF32.spec().entry_point,
            )
            .map_err(|e| (false, e.to_string()))?;
        report["stages"]
            .as_array_mut()
            .unwrap()
            .push(json!({"stage":"pipeline_created", "wall_ns":elapsed_ns(stage)}));
        drop(pipeline);

        let stage = Instant::now();
        let manifest = certify_builtin(
            &mut context,
            BuiltinKernel::AffineF32,
            schedule,
            4099,
            5,
            30,
        )
        .map_err(|e| (false, e.to_string()))?;
        if manifest.validation_level != ValidationLevel::Certified
            || manifest.source_hash != generated.source_hash
            || !manifest
                .oracle
                .as_ref()
                .is_some_and(|o| o.passed() && o.compared == 4099)
            || !manifest
                .timing
                .as_ref()
                .is_some_and(|t| t.sample_count == 30)
        {
            return Err((
                false,
                "certification evidence is incomplete or inconsistent".into(),
            ));
        }
        report["stages"].as_array_mut().unwrap().push(json!({
            "stage":"certified", "wall_ns":elapsed_ns(stage),
            "includes":["oracle_verified", "profiled"],
            "note":"oracle and profiling share the existing certification procedure; separate wall times are not measured"
        }));
        report["oracle"] = json!(manifest.oracle);
        report["timing"] = json!(manifest.timing);
        write_new(
            &dir.join("manifest.json"),
            manifest
                .to_pretty_json()
                .map_err(|e| (false, e.to_string()))?
                .as_bytes(),
        )
        .map_err(|e| (false, e))?;
        if let (Some(vertices), Some(asset_hash)) = (vertices_path, asset_sha256) {
            report["asset_phase"] = asset_phase(&mut context, schedule, dir, vertices, asset_hash)
                .map_err(|e| (false, e))?;
        }
        Ok(())
    }

    pub fn run() -> Result<bool, String> {
        let mut args = std::env::args().skip(1);
        let output = args
            .next()
            .ok_or("usage: production_core_probe OUTPUT_DIR [EXPECTED_ADAPTER_SUBSTRING]")?;
        let mut expected = None;
        let mut vertices_path = None;
        let mut asset_sha256 = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--vertices-json" if vertices_path.is_none() => {
                    vertices_path = Some(args.next().ok_or("missing vertices path")?)
                }
                "--asset-sha256" if asset_sha256.is_none() => {
                    asset_sha256 = Some(args.next().ok_or("missing asset SHA256")?)
                }
                _ if !arg.starts_with('-') && expected.is_none() => expected = Some(arg),
                _ => return Err(format!("unexpected argument {arg}")),
            }
        }
        if vertices_path.is_some() != asset_sha256.is_some() {
            return Err("--vertices-json and --asset-sha256 must be supplied together".into());
        }
        if let Some(hash) = &asset_sha256 {
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("asset SHA256 must be 64 hexadecimal characters".into());
            }
        }
        let dir = Path::new(&output);
        fs::create_dir(dir).map_err(|e| format!("create-only output {}: {e}", dir.display()))?;
        let started = Instant::now();
        let mut report = json!({
            "schema_version":1, "probe":"production_core_affine_baseline", "outcome":"running",
            "length":4099, "warmups":5, "retained_samples":30,
            "requested_backend":std::env::var("ETALON_WGPU_BACKEND").ok(),
            "expected_adapter_substring":expected,
            "slab_capacity_bytes":1048576,
            "limitations":"fixed FP32 affine kernel only; not application, browser, cross-vendor, geometry, or WebGPU CTS qualification; three device slabs plus staging and pipeline allocations are not a total-memory bound",
            "probe_source_blake3":blake3::hash(include_str!("production_core_probe.rs").as_bytes()).to_hex().to_string(),
            "stages":[]
        });
        let result = execute(
            dir,
            expected.as_deref(),
            vertices_path.as_deref(),
            asset_sha256.as_deref(),
            &mut report,
        );
        report["total_wall_ns"] = json!(elapsed_ns(started));
        match &result {
            Ok(()) => report["outcome"] = json!("passed"),
            Err((unsupported, error)) => {
                report["outcome"] = json!(if *unsupported {
                    "unsupported"
                } else {
                    "failed"
                });
                report["error"] = json!(error);
            }
        }
        write_json(dir, "probe.json", &report)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
        match result {
            Ok(()) => Ok(true),
            Err((true, _)) => Ok(false),
            Err((false, error)) => Err(error),
        }
    }
}

fn main() {
    #[cfg(feature = "runtime")]
    match probe::run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(2),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "runtime"))]
    {
        eprintln!("production_core_probe requires --features gpu-native-dx12 on Windows (or runtime with an enabled native backend)");
        std::process::exit(2);
    }
}
