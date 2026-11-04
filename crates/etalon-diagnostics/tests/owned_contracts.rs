use etalon_diagnostics::{
    ipc::{
        decode_matching_response, decode_response, encode_response, WorkerArtifacts, RESPONSE_BYTES,
    },
    process::{run_worker, WorkerLimits},
    CircuitBench, CircuitKind, DeviceBenchmarkRequest, DeviceBenchmarkResponse,
};
use std::{
    cell::Cell,
    fs::File,
    io::Write,
    process::Command,
    time::{Duration, Instant},
};

fn response() -> DeviceBenchmarkResponse {
    DeviceBenchmarkResponse {
        request: DeviceBenchmarkRequest {
            backend: "vulkan".into(),
            vendor: 1,
            device: 2,
            gemv_n: 16,
        },
        bench: Some(CircuitBench {
            label: "fixture".into(),
            kind: CircuitKind::DiscreteGpu,
            backend: "Vulkan".into(),
            ms_per_gemv: 1.0,
            gflops: 0.5,
            upload_gbps: 0.1,
            rel_score: 1.0,
            decode_proxy_tok_s: None,
        }),
        error: None,
    }
}

#[test]
fn ipc_roundtrip_is_identity_bound_and_does_not_overwrite_existing_output() {
    let owner = WorkerArtifacts::new().unwrap();
    let path = owner.response_path();
    let value = response();
    encode_response(&path, &value).unwrap();
    owner.check_budget().unwrap();
    assert_eq!(
        decode_matching_response(&path, &value.request)
            .unwrap()
            .label,
        "fixture"
    );
    let saved = std::fs::read(&path).unwrap();
    assert!(encode_response(&path, &value).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), saved);
    for field in 0..4 {
        let mut wrong = value.request.clone();
        match field {
            0 => wrong.backend = "dx12".into(),
            1 => wrong.vendor += 1,
            2 => wrong.device += 1,
            _ => wrong.gemv_n += 1,
        }
        assert!(decode_matching_response(&path, &wrong)
            .unwrap_err()
            .contains("identity"));
    }
}

#[test]
fn ipc_rejects_truncation_trailing_data_zero_lengths_and_oversized_artifacts() {
    let owner = WorkerArtifacts::new().unwrap();
    let path = owner.response_path();
    encode_response(&path, &response()).unwrap();
    let valid = std::fs::read(&path).unwrap();
    for length in [0, 7, valid.len() - 1] {
        std::fs::write(&path, &valid[..length]).unwrap();
        assert!(decode_response(&path).is_err());
    }
    let mut trailing = valid.clone();
    trailing.push(0);
    trailing[..8].copy_from_slice(&((valid.len() - 7) as u64).to_le_bytes());
    std::fs::write(&path, trailing).unwrap();
    assert!(decode_response(&path).unwrap_err().contains("trailing"));
    std::fs::write(&path, 0u64.to_le_bytes()).unwrap();
    assert!(decode_response(&path).is_err());
    File::create(&path)
        .unwrap()
        .set_len((RESPONSE_BYTES + 9) as u64)
        .unwrap();
    assert!(decode_response(&path).is_err());
    assert!(owner.check_budget().is_err());
}

#[test]
fn ipc_encoding_budget_fails_before_creating_output_and_metrics_fail_closed() {
    let owner = WorkerArtifacts::new().unwrap();
    let path = owner.response_path();
    let mut huge = response();
    huge.error = Some("x".repeat(RESPONSE_BYTES + 1));
    assert!(encode_response(&path, &huge).is_err());
    assert!(!path.exists());
    for case in 0..7 {
        let mut value = response();
        let bench = value.bench.as_mut().unwrap();
        match case {
            0 => bench.ms_per_gemv = f64::NAN,
            1 => bench.ms_per_gemv = 0.0,
            2 => bench.gflops = -1.0,
            3 => bench.upload_gbps = f64::INFINITY,
            4 => bench.kind = CircuitKind::Cpu,
            5 => value.error = Some("fixture failure".into()),
            _ => value.bench = None,
        }
        encode_response(&path, &value).unwrap();
        assert!(decode_matching_response(&path, &value.request).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}

#[test]
fn owned_artifacts_cleanup_on_success_error_and_unwind_and_reject_unknown_entries() {
    for outcome in 0..3 {
        let owner = WorkerArtifacts::new().unwrap();
        let path = owner.response_path();
        let directory = path.parent().unwrap().to_path_buf();
        let result = std::panic::catch_unwind(move || -> Result<(), String> {
            std::fs::write(owner.response_path(), b"owned").unwrap();
            match outcome {
                0 => Ok(()),
                1 => Err("fixture error".into()),
                _ => panic!("fixture unwind"),
            }
        });
        match outcome {
            0 => assert!(result.unwrap().is_ok()),
            1 => assert!(result.unwrap().is_err()),
            _ => assert!(result.is_err()),
        }
        assert!(!directory.exists());
    }
    let owner = WorkerArtifacts::new().unwrap();
    let unexpected = owner.response_path().with_file_name("unexpected");
    std::fs::write(unexpected, b"x").unwrap();
    assert!(owner.check_budget().is_err());
}

#[test]
#[ignore = "self-executing bounded local child fixture"]
fn diagnostic_child_fixture() {
    let mode = std::env::var("ETALON_TEST_CHILD_MODE").unwrap();
    let path = std::env::var_os("ETALON_TEST_CHILD_LOCK").unwrap();
    let file = File::create(path).unwrap();
    file.lock().unwrap();
    let ready = std::env::var_os("ETALON_TEST_CHILD_READY").unwrap();
    std::fs::write(ready, b"ready").unwrap();
    match mode.as_str() {
        "success" => {
            print!("fixture stdout");
            eprint!("fixture stderr");
        }
        "error" => {
            eprint!("fixture exit failure");
            std::process::exit(9);
        }
        "stdout" => {
            std::io::stdout().write_all(&[b'x'; 8192]).unwrap();
        }
        "stderr" => {
            std::io::stderr().write_all(&[b'x'; 8192]).unwrap();
        }
        "wait" => std::thread::sleep(Duration::from_secs(30)),
        _ => panic!("unknown child mode"),
    }
}

fn child(mode: &str, directory: &std::path::Path) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "diagnostic_child_fixture",
            "--ignored",
            "--nocapture",
        ])
        .env("ETALON_TEST_CHILD_MODE", mode)
        .env("ETALON_TEST_CHILD_LOCK", directory.join("child.lock"))
        .env("ETALON_TEST_CHILD_READY", directory.join("ready"));
    command
}

fn limits() -> WorkerLimits {
    WorkerLimits {
        timeout: Duration::from_secs(5),
        stdout_bytes: 4096,
        stderr_bytes: 4096,
    }
}

fn assert_released(directory: &std::path::Path) {
    let path = directory.join("child.lock");
    assert!(path.exists(), "child fixture never started");
    let lock = File::options().read(true).write(true).open(path).unwrap();
    lock.try_lock()
        .expect("worker must release its owned OS file lock before returning");
}

#[test]
fn real_child_success_error_and_both_pipe_budgets_are_bounded_and_reaped() {
    for mode in ["success", "error", "stdout", "stderr"] {
        let directory = tempfile::Builder::new()
            .prefix("etalon-diagnostic-test-")
            .tempdir()
            .unwrap();
        let result = run_worker(
            &mut child(mode, directory.path()),
            &limits(),
            &|| false,
            &|| Ok(()),
        );
        match mode {
            "success" => {
                let output = result.unwrap();
                assert!(output.status.success());
                assert!(String::from_utf8_lossy(&output.stdout).contains("fixture stdout"));
                assert!(String::from_utf8_lossy(&output.stderr).contains("fixture stderr"));
            }
            "error" => assert!(result.err().unwrap().contains("fixture exit failure")),
            _ => assert!(result.err().unwrap().contains("output budget")),
        }
        assert_released(directory.path());
    }
}

#[test]
fn real_child_deadline_cancellation_artifact_error_and_unwind_release_ownership() {
    for mode in ["deadline", "cancel", "artifact", "unwind"] {
        let directory = tempfile::Builder::new()
            .prefix("etalon-diagnostic-test-")
            .tempdir()
            .unwrap();
        let ready = directory.path().join("ready");
        let mut limits = limits();
        if mode == "deadline" {
            limits.timeout = Duration::from_secs(2);
        }
        let started = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_worker(
                &mut child("wait", directory.path()),
                &limits,
                &|| mode == "cancel" && ready.exists(),
                &|| {
                    if ready.exists() {
                        if mode == "artifact" {
                            return Err("fixture artifact rejection".into());
                        }
                        if mode == "unwind" {
                            panic!("fixture checker unwind");
                        }
                    }
                    Ok(())
                },
            )
        }));
        if mode == "unwind" {
            assert!(result.is_err());
        } else {
            let error = result.unwrap().err().unwrap();
            assert!(
                error.contains(match mode {
                    "deadline" => "deadline",
                    "cancel" => "cancelled",
                    _ => "artifact rejection",
                }),
                "{error}"
            );
        }
        assert!(started.elapsed() < Duration::from_secs(7));
        assert_released(directory.path());
    }
}

#[test]
fn invalid_limits_prelaunch_cancel_and_artifact_rejection_never_launch_a_child() {
    let directory = tempfile::Builder::new()
        .prefix("etalon-diagnostic-test-")
        .tempdir()
        .unwrap();
    let mut invalid = limits();
    invalid.stdout_bytes = 0;
    let checks = Cell::new(0);
    let check = || {
        checks.set(checks.get() + 1);
        Ok(())
    };
    assert!(run_worker(
        &mut child("wait", directory.path()),
        &invalid,
        &|| false,
        &check
    )
    .is_err());
    assert!(run_worker(
        &mut child("wait", directory.path()),
        &limits(),
        &|| true,
        &check
    )
    .is_err());
    assert_eq!(checks.get(), 0);
    assert!(run_worker(
        &mut child("wait", directory.path()),
        &limits(),
        &|| false,
        &|| Err("preflight".into())
    )
    .is_err());
    assert!(!directory.path().join("ready").exists());
}
