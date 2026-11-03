# Testing in `etalon` and `naga`

There exist a large variety of tests within the `etalon` repository
to make sure we can easily test all the aspects of our libraries.
This document serves as a guide to each class of test, and what
they are used for.

## Requirements

The tests require that the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
is installed on the system and the `bin` folder of the SDK is in your `PATH`.
Without this some tests may fail to run, or report false negatives.

Additionally you require you run the tests with `cargo-nextest`.
This is what our xtask calls. You can install it with `cargo install cargo-nextest`.

## Run All Tests

To run all tests, run `cargo xtask test` from the root of the repository.

## Test Breakdown

This is a table of contents, in the form of the repository's directory structure.

- benches
  - [benches](#benchmark-tests)
- [QualiaDB/VibeScript integration](#qualiadbvibescript-integration)
- examples
  - [features](#example-tests)
- naga
  - tests
    - [example_wgsl](#naga-example-tests)
    - [snapshot](#naga-snapshot-tests)
    - [spirv-capabilities](#naga-spirv-capabilities-test)
    - [validation](#naga-validation)
    - [wgsl_errors](#naga-wgsl-error-tests)
- player
  - [tests](#player-tests)
- tests
  - [compile](#etalon-compile-tests)
  - [dependency](#etalon-dependency-tests)
  - [gpu](#etalon-gpu-tests)
  - [trace](#etalon-trace-tests)
  - [validation](#etalon-validation-tests)

And where applicable [unit-tests](#unit-tests)
are scatteredthroughout the codebase.

## Benchmark Tests

- Located in: `benches/benches`
- Run with `cargo nextest run --bench etalon-benchmark`
- `etalon` benchmarks for performance testing.

These are benchmarks that test the performance of `etalon` in various
scenarios. When run as part of the test suite, they run a single
iteration of each benchmark to ensure they continue to function.

These tests only run on your system's default GPU.

The benchmarks should be very careful to avoid doing any significant
work (including connecting to a GPU) outside of the various `benchmark`
`criterion` functions. If this is done, the benchmarks will take a long
time to list available tests, slowing down the test suite.

To run the benchmarks for benchmarking purposes, use `cargo bench`.

## Example Tests

- Located in: `examples/features`
- Run with `cargo xtask test --bin etalon-examples`
- Uses a custom `#[apply(gpu_test!)]` harness.
- `etalon` integration tests, with access to `etalon_test` helpers.

These tests validate that the examples are functioning correctly
and do not have any regressions. They use the same harness as the
[gpu tests](#etalon-gpu-tests), see that section for more information
on the harness.

These tests use `nv-flip`'s image comparison through the etalon
example framework to validate that the images outputted by the
examples are within tolerance of the expected output.

Examples written in `examples/standalone` do not have tests, as
they should be easy to copy into a standalone project.

## `naga` Example Tests

- Located in: `naga/tests/naga/example_wgsl`
- Run with `cargo nextest run --test naga example_wgsl`

This simple test ensures that all wgsl files in the `examples`
directory can be parsed by `naga`'s `wgsl` parser and validate correctly.

## `naga` Snapshot Tests

- Located in: `naga/tests/naga/snapshot`, `naga/tests/in`, and `naga/tests/out`
- Run with `cargo nextest run --test naga snapshots`
- Data driven snapshot tests for `naga`'s input/output.

These tests are snapshot tests for `naga`s parsers and code generators.
There are inputs in `wgsl`, `spirv`, and `glsl`. There are outputs for
`hlsl`, `spirv`, `wgsl`, `msl`, `glsl`, and naga's internal IR. The tests
can be configured by a sidecar toml file of the same name as the input file.

This is the goto tool for testing all kinds of codegen and parsing features.

To avoid clutter we generally use the following pattern:

- `wgsl` tests generate output to all backends.
- `spirv`, `glsl` tests generate `wgsl` output

This "butterfly" pattern ensures we don't need to test the
full matrix of possibilities to get full coverage.

While we do not run the results of the code generators, we do
test that the generated code is valid. This is done by running
`cargo xtask validate <backend>` in the `naga` directory and
will use the respective tool to validate the generated code.

## `naga` SPIR-V Capabilities Tests

- Located in: `naga/tests/naga/spirv_capabilities`
- Run with `cargo nextest run --test naga spirv_capabilities`
- Uses the standard `#[test]` harness.

These tests convert the given wgsl snippet to spirv and
then assert that the spirv has enabled the expected capabilities.

## `naga` Validation Tests

- Located in: `naga/tests/naga/validation`
- Run with `cargo nextest run --test naga validation`

These are hand rolled tests against the naga's validator.
If you don't need to test the validator with a custom module,
and can use the `wgsl` frontend, you should put the test in
the [wgsl errors](#naga-wgsl-error-tests) tests.

## `naga` WGSL Error Tests

- Located in: `naga/tests/naga/wgsl_errors`
- Run with `cargo nextest run --test naga wgsl_errors`

These are tests for the error messages that the `wgsl` frontend
produces. Additionally you can check that a given validation error
is produced by the validator from a given `wgsl` snippet.

## `player` Tests

- Located in: `player/tests`
- Run with `cargo nextest run --test player`
- Data driven tests using the `player`'s replay system.
- `etalon` integration tests.

These are soft-deprecated tests which are another way to write
API tests. These use captures of the api calls and replay them
to assert on the behavior. They are very difficult to write, and
the trace capturing system is currently broken, so these
tests exist, but you should not write new ones.

These tests only run on your system's default GPU.

## `etalon` Compile Tests

- Located in: `tests/tests/etalon-compile`
- Run with `cargo nextest run --test etalon-compile`
- `trybuild` tests of all rust files in `tests/tests/etalon-compile/fail` directory.

These use the `trybuild` crate to test a few scenarios where
the `etalon` crate is expected to fail to compile. This mainly
revolves around ensuring lifetimes are properly handled when
dropping passes, etc.

## `etalon` Dependency Tests

- Located in: `tests/tests/etalon-dependency`
- Run with `cargo nextest run --test etalon-dependency`
- Tests against `cargo tree`.

These tests ensure that the `etalon` crate has the correct dependency
tree on all platforms. It's super easy to subtly mess up the dependencies
which can cause issues or extra dependencies to be pulled in.

This provides a way to ensure that our `toml` files are correct.

## `etalon` GPU Tests

- Located in: `tests/tests/etalon-gpu`
- Run with `cargo xtask test --test etalon-gpu`
- Uses a custom `#[apply(gpu_test!)]` harness.
- `etalon` integration tests, with access to `etalon_test` helpers.

These tests use a custom harness to run each test on all GPUs
available on the system. They are general integration tests
that write code against the normal `etalon` API and assert on the behavior.

These tests are useful to check the runtime behavior of a program,
validate that there are no validation errors coming from the
`vulkan`/`dx12`/`metal` validation layers, and ensure behavior
is the same across GPUs. If the test does not need to run on a
real GPU, it should be in the [validation tests](#etalon-validation-tests) instead.

There is a special parameter system that deals with if a GPU
can support the given test, and dealing with expectation
management for tests that are expected to fail due to driver or etalon bugs.

Normal `#[test]`s will not be found in this test crate, as we use a custom harness.

See also the [example tests](#example-tests) for additional GPU tests.

## `etalon` Trace Tests

- Located in: `tests/tests/etalon_trace.rs`
- Run with `cargo nextest run --test etalon_trace`
- Use the standard `#[test]` harness.

These tests are focused on testing the tracing functionality in `etalon`. They
use the a special `noop` backend which does not connect to a real GPU.

## `etalon` Validation Tests

- Located in: `tests/tests/etalon-validation`
- Run with `cargo nextest run --test etalon-validation`
- Use the standard `#[test]` harness.
- `etalon` integration tests, with access to `etalon_test` helpers.

These tests are focused on testing the validation inside of `etalon-core`.
They are written against the `etalon` API, but are targeting a special `noop`
backend which does not connect to a real GPU.

This is significantly faster and simpler than running on real hardware,
and allows any validation logic to be checked, even if real hardware
does not support those features.

## Unit Tests

- Located throughout the codebase.
- Run with `cargo nextest test -p <package>`
- Standard `#[test]`s.

Throughout the codebase we have standard `#[test]`s that test individual
functions or small parts of the codebase. These don't run on the gpu.

## QualiaDB/VibeScript Integration

The consuming QualiaDB repository provides a VibeScript harness that exercises
its canonical host and real Etalon compute dispatch/readback against CPU references.
Integrated GPUs are supported; NVIDIA/CUDA is not required. CPU-only mode is
explicit and excludes GPU work.

See [conformance-coverage.md](conformance-coverage.md) for commands, the coverage
matrix and the conformance coverage lost when the embedded Deno CTS runner was
retired. This harness does not run the upstream WebGPU CTS. All Rust test suites
listed above remain in place.

## Memory Initialization Testing

### Selecting NVIDIA on native Windows

Vulkan is an existing native Etalon backend and does not require CUDA. Enable
the `vulkan` Cargo feature, construct an instance with `Backends::VULKAN`, and
select an adapter from that instance. `ETALON_BACKEND` affects instances created
with environment-aware descriptors; `ETALON_ADAPTER_NAME` is honored by
`etalon::util::initialize_adapter_from_env_or_default`, not by every direct
`Instance::request_adapter` call. Always inspect `Adapter::get_info()` when
qualifying a particular device.

The GPU test harness enumerates adapters and embeds their names in test names.
Use a nextest filter as well as the backend environment variable to target
NVIDIA on a machine with multiple GPUs:

```powershell
$env:ETALON_BACKEND = 'vulkan'
cargo xtask test -E 'binary(etalon-gpu) & test(NVIDIA) & test(prepared_compute)'
cargo xtask test -E 'binary(etalon-gpu) & test(NVIDIA) & (test(buffer) | test(compute))'
$env:ETALON_BACKEND = 'dx12'
cargo xtask test -E 'binary(etalon-gpu) & test(NVIDIA) & (test(buffer) | test(compute))'
Remove-Item Env:ETALON_BACKEND
```

Check that tests actually ran and inspect `.gpuconfig` for the selected adapter
names. Prepared native compute specifically requires Vulkan; ordinary compute
also supports other enabled native backends. The standalone ignored
qualification tests use `ETALON_ADAPTER_NAME`; see
[prepared target qualification](prepared-compute.md#qualification).

### Shared Forge routing and CUDA qualification

`etalon-wgsl-forge::emit_routed_shader` consumes one kernel specification and
schedule, an explicit routing policy, and capabilities for the selected device.
`Portable` selects WGSL/HLSL/MSL. `PreferCuda` selects CUDA-C when the driver,
compiler, device identity, architecture, and kernel emitter permit it; otherwise
it returns portable source with a fallback reason. `RequireCuda` errors rather
than falling back. These are capability and preference decisions, not benchmark
claims that CUDA is always fastest.

The native CUDA route is Forge specification → CUDA-C → NVCC PTX → owned CUDA
module → launch → readback. The portable Vulkan route is the same specification
→ WGSL → Naga/SPIR-V → Vulkan. `emit_shader_ptx` and `compile_cuda_to_ptx` require
an explicit compute capability; generic PTX emission does not guess one.

`etalon-cuda::CudaKernels` compiles shared Forge affine, FP32 matrix-vector and
IQ-power kernels for its selected `CudaContext`. Safe APIs check shapes and
use owned device buffers. `forge_capabilities` compares the caller's selected
PCI bus ID with the CUDA device, avoiding accidental routing to a different GPU.
The `cuda` features of `etalon-agent`, `etalon-tensor`, and `etalon-sense-gpu`
expose this runtime and domain APIs without requiring CUDA for portable defaults.
Loading arbitrary PTX and launching supplied kernels are explicitly unsafe:
the caller must satisfy the kernel ABI and memory-access contract. Core WGSL
validation remains in the portable path.

On Windows, NVCC requires an installed MSVC host compiler. Run from an appropriate
Visual Studio developer shell, or set `NVCC_CCBIN` to the supported compiler's
directory. The compiler helper uses `CUDA_PATH/bin/nvcc`, then `PATH`, and retains
failure diagnostics. No driver or toolkit is downloaded by the library.

CUDA Toolkit 13.4.3 installs under `v13.4`; its NVCC component reports 13.4.92.
Existing shells can retain an older `CUDA_PATH` after an upgrade. Open a new
developer shell or explicitly select the installed toolkit before qualification:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
& "$env:CUDA_PATH\bin\nvcc.exe" --version
nvidia-smi
```

The [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
identify R615 as the corresponding driver branch for CUDA 13.4. The locally
qualified combination uses driver 616.92; toolkit installation and driver
installation are separate. Nsight is optional profiling tooling and is not a
dependency of Etalon's compilation, execution or correctness tests.

After a driver upgrade, run GPU regressions through `cargo xtask test` so the
test harness regenerates `.gpuconfig`. Direct `cargo nextest run` with an older
inventory can reject the updated adapter before executing a workload.

```powershell
$env:ETALON_CUDA_DEVICE = '0'
$env:ETALON_ADAPTER_NAME = 'NVIDIA'
cargo test -p etalon-cuda -p etalon-wgsl-forge --all-features --lib -- --include-ignored --nocapture
cargo test -p etalon-tensor -p etalon-sense-gpu --all-features --lib cuda::tests -- --include-ignored --nocapture --test-threads=1
cargo run -p etalon-cuda --example routed_affine -- auto
cargo run -p etalon-cuda --example routed_affine -- cuda
cargo run -p etalon-cuda --example routed_affine -- vulkan
```

The example binds both routes to the selected Vulkan adapter's PCI identity,
prints the selected source and device, and checks readback against a CPU reference.
The runtime hardware tests also check buffer initialization, transfer bounds,
cross-context rejection, context restoration, and supplied PTX execution. Ignored
CUDA tests require real hardware/tooling and fail if those requirements are absent.

The legacy `CudaComputeLane` has only driver/host-pinning support and rejects
kernel execution. `CudaMegaPassExecutor` remains a scalar CPU transformer
reference; use `etalon_tensor::cuda::CudaTensorExecutor` for actual CUDA FP32
GEMV, GEMM and fused-FFN work. These are distinct APIs and execution receipts.

`CudaTensorExecutor::prepare_gemv`, `prepare_gemm` and `prepare_fused_ffn` retain
weights, dimensions, modules and input/output device buffers. Their returned
`CudaPreparedTensor::execute` accepts exact-length host slices and writes into
caller-owned output. Repeated calls upload only changing input; malformed shapes
are rejected and caller output is cleared on failure. Fused FFN computes
`w2 * GELU(w1 * x)` without biases or residual addition.

For supplied native code, `etalon_cuda::CudaProgram::compile` uses Forge's CUDA-C
compiler and the selected device architecture. `from_ptx` loads an existing
artifact without needing the toolkit. `prepare` declares argument types and
minimum buffer sizes and retains reusable launch storage. Loading supplied code
and launching it are explicitly unsafe: matching a declared ABI does not prove
the native kernel's accesses are in bounds. Programs and buffers retain their
context, and prepared kernels borrow the program to prevent early module unload.

Further implementation should extend this shared path across all three domains:

| Domain             | Implemented foundation                                                                                                                                          | Remaining work                                                                                                               |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Supplied kernels   | Explicit architecture compilation, owned buffers/modules, declared argument layouts, reusable dispatch storage, checked launch geometry and unsafe ABI boundary | Bounded compilation cache keyed by toolchain/device/source, reusable asynchronous dispatch and cancellation                  |
| Tensor inference   | Shared Forge FP32 GEMV/GEMM/fused-FFN, resident weights and buffers, repeated CPU differential checks                                                           | Quantized weights, attention/RMSNorm/RoPE, resident whole-model state, graph capture and whole-model numerical qualification |
| Sensing/IQ         | Shared Forge FP32 and signed-i16 IQ power, both byte orders, scale checks, reusable bounded device buffers and registered-provider budget admission             | Asynchronous streaming/overlap, live host-policy integration, and allocation qualification                                   |
| Route optimization | Explicit portable/prefer/require policy and physical-device matching                                                                                            | End-to-end timing including transfers, per-workload certification, persisted measured choices and fallback policy            |

The convenience CUDA operations allocate on each call. Prepared tensor operations
reuse device buffers and host argument storage. Both synchronize before returning;
neither is a CUDA graph implementation or a qualified whole-path zero-allocation
guarantee.

`etalon_sense_gpu::cuda::PreparedCudaIqPower` implements `RegisteredSignalProvider`
for signed-i16 inputs, with the portable provider's workspace/output shapes and
FP32-to-F64 result contract. Its two persistent CUDA buffers use `8 * max_pairs`
logical bytes. It reports `ColdBoundedBuffers` and rejects `HotZeroHeap`; this
budget excludes driver/module allocations. The caller's `DeviceId` is a routing
declaration, not hardware attestation. Native Vulkan `PreparedIqPower` now emits
WGSL from the same Forge `IqPowerI16` specification, during preparation only.

### CUDA profiling

The native tensor example checks every measured output against a CPU reference.
It prepares resident weights, warms up five times, then reports the median and
minimum duration of synchronous execution, including input upload and output
readback. Compilation, allocation, weight upload and CPU checking are outside
the measured interval. These are end-to-end call timings, not kernel-only timings.

```powershell
cargo build -p etalon-tensor --features cuda --example cuda_profile
target/debug/examples/cuda_profile.exe all 100
target/debug/examples/cuda_profile.exe ffn 100
```

Workloads are `gemv`, `gemm`, `ffn`, `ffn-tail`, and `ffn-small`, or `all`.
The optional second argument is a positive repetition count; the device is
selected through `ETALON_CUDA_DEVICE`, defaulting to ordinal zero. Use identical
build profiles, inputs, device and repetition counts when comparing changes.

On x64 Windows, select Nsight's x64 executable: Nsight Compute installs under
`target/windows-desktop-win7-x64/ncu.exe`, and Nsight Systems under
`target-windows-x64/nsys.exe`. The bundled `target-windows-armv8` executable
does not run on this platform. Set
`NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT=1` if Nsight Compute cannot
deploy sections into Documents; it then uses the installation's sections.
See the [Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

`ERR_NVGPUCTRPERM` is a separate driver permission error. Follow
[NVIDIA's performance-counter instructions](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
and verify access with an actual capture. A successful workload exit does not
mean counter collection succeeded. Nsight Systems can disable CPU sampling and
context-switch tracing with `--sample=none --cpuctxsw=none`; inspect the report
for actual CUDA kernel records before treating it as a successful GPU trace.

### CUDA FFN optimization qualification on 3 November 2026

On the RTX A2000 12GB with driver 616.92 and CUDA Toolkit 13.4.3, Forge's
CUDA fused-FFN emitter now shares hidden activations within each block. Each
thread computes one hidden row per tile; all threads participate in both
barriers, including lanes with no output. Tile advancement is bounded by the
remaining hidden rows. The existing entry-point ABI, safe tensor shape checks,
FP32 dot-product order and sequential output accumulation are retained.

The profiling example was built in the Rust dev profile. Three paired runs used
100 measured calls per workload after five warm-ups, with order reversed in the
middle pair. The table shows the median of the three run medians, in milliseconds,
including input upload, launch, synchronization and readback:

| Workload (input × hidden × output for FFN) | Before |  After | Observation                            |
| ------------------------------------------ | -----: | -----: | -------------------------------------- |
| FFN 256 × 512 × 512                        | 6.3480 | 0.6545 | 9.70× faster                           |
| FFN 127 × 267 × 263                        | 1.5224 | 0.4634 | 3.29× faster                           |
| FFN 11 × 13 × 263                          | 0.3691 | 0.4072 | No demonstrated end-to-end improvement |
| GEMV 512 × 256, unchanged control          | 1.4092 | 1.2508 | Timing variation                       |
| GEMM 64 × 128 × 64, unchanged control      | 0.4144 | 0.4446 | Timing variation                       |

All measured outputs matched CPU references. These are local microbenchmarks,
not whole-model inference results or evidence for automatic CUDA/Vulkan routing.
The raw paired logs are `target/nsight/timing-{1,2,3}-{baseline,optimized}.log`.

Nsight Systems 2026.3.2 successfully captured the actual emitted CUDA kernels
using a temporary harness that compiled PTX before profiler launch. Across 25
launches, median kernel duration changed from **8.101817 ms to 0.229891 ms** for
256 × 512 × 512, and **10.912 µs to 3.328 µs** for 11 × 13 × 263. This separate
harness used different deterministic inputs and a release-built host; its
kernel-only results must not be substituted for the end-to-end table. Reports
are `target/nsight/ffn-{baseline,optimized}.nsys-rep` and
`target/nsight/ffn-small-{baseline,optimized}.nsys-rep`, with corresponding
`*-stats.log` summaries. Each harness run checked its outputs against a CPU
reference. Nsight Compute 2026.3.1 still returned `ERR_NVGPUCTRPERM`, including
after the user enabled counter access and after a run outside the sandbox;
no occupancy, throughput-counter or other Nsight Compute metrics were collected.

Direct Nsight Systems launches that included runtime NVCC compilation failed:
one reported truncated-header compiler errors and another failed with
`cuModuleGetFunction` error 500. The same workloads ran successfully without
instrumentation. Loading the precompiled Forge PTX avoided those failures and
produced verified kernel traces. The initial profiling example also had a Rust
environment-parsing compilation error, corrected before measurements.

All **18 targeted CUDA tests passed**, including the new hidden-tile cases
(hidden sizes 255, 256, 257, 513 and 1025, partial outputs and repeated inputs).
Compute Sanitizer `memcheck`, `racecheck` and `synccheck` reported zero errors
or hazards for both 256 × 512 × 512 and 11 × 257 × 1, the latter requiring
inactive output lanes to supply hidden activations. Logs are
`target/nsight/cuda-tests.log`, `target/nsight/ffn-{memcheck,racecheck,synccheck}.log`
and `target/nsight/ffn-tail-{memcheck,racecheck,synccheck}.log`. All-target,
all-feature Clippy with warnings denied passed for the tensor and Forge crates.

The final full `cargo xtask test` passed **3,897 tests, zero failed, 46 skipped**
in 587.380 seconds with normal concurrency and all enumerated adapters. The new
ignored CUDA hardware test was included in the separate 18-test qualification.
The full suite used Vulkan validation layer 1.4.363.1 and CI's pinned 1.4.357.0
shader tools; generated shader content matched the committed snapshots.
Native compilation without the CUDA feature and WebAssembly compilation with
all features also passed for the profiling example. Full-run and lint logs are
`target/nsight/full-suite.log` and `target/nsight/clippy.log`. Harness pass counts
include unsupported cases and are not WebGPU CTS coverage.

### CUDA 13.4.3 upgrade qualification on 3 November 2026

The upgraded local configuration is CUDA Toolkit 13.4.3 (NVCC 13.4.92), NVIDIA
driver 616.92, RTX A2000 12GB at compute capability 8.6, MSVC 14.44, Rust 1.96.1,
and Vulkan validation layer 1.4.363.1. Toolkit selection was explicit through
`CUDA_PATH`; the existing process environment still referred to 13.3 after the
installer updated the machine environment.

All **17 targeted CUDA checks passed**, including owned buffers and context
restoration, supplied kernels, resident tensor operations, signed-IQ processing,
and valid/invalid NVCC input. No CUDA Driver API or generated CUDA-C changes were
needed for this toolkit version.

An initial Vulkan run passed 26 checks and failed 16 before workload execution
because `.gpuconfig` still described driver 610.47. Regenerating the inventory
through `cargo xtask test` produced **40 passes and two real failures**: sparse
multiview clears and draws modified inactive layers on NVIDIA driver 616.92.
The existing RTX A2000 workaround now also matches that exact driver version
(`2_585_198_592`), preserving the prior 610.47 match. Neighbouring versions,
other devices and other driver implementations remain excluded.

The corrected run passed **43 regression checks**, including both failing
multiview cases, descriptor/runtime-array bounds, planar textures and prepared
Vulkan allocation/IQ qualifications. A negative-test driver identifier unavailable
in the pinned Vulkan bindings initially prevented compilation; the corrected
fixture uses an available non-NVIDIA identifier. Logs are retained locally as
`target/cuda-13.4-qualification.log`, `target/cuda-13.4-vulkan-regressions.log`,
`target/cuda-13.4-vulkan-refreshed.log`, `target/cuda-13.4-vulkan-fixed.log`, and
`target/cuda-13.4-vulkan-fixed-recheck.log`. Harness pass counts include unsupported
cases and are not counts of successful hardware workloads or CTS coverage.

The final full `cargo xtask test` passed **3,897 tests, zero failed, 45 skipped**
in 616.127 seconds with normal concurrency and all enumerated adapters. HAL
Clippy with all targets/features and warnings denied, Rust formatting, spelling,
and documentation formatting also passed. Full-run and lint logs are
`target/cuda-13.4-full-suite.log` and `target/cuda-13.4-clippy.log`.
SPIRV-Cross from SDK 1.4.363.1 added a conservative-depth extension declaration
to one generated GLSL snapshot. Re-running the same SPIR-V through CI's pinned
1.4.357.0 tool reproduced the committed output; that baseline was retained.

### NVIDIA review on 3 November 2026

Local hardware was an NVIDIA RTX A2000 12GB, driver 610.47, CUDA compute capability
8.6, PCI `0000:01:00.0`, alongside Intel HD Graphics 530. Tests used Rust 1.96.1,
CUDA Toolkit 13.3, MSVC 14.44 for NVCC, and Vulkan validation layer 1.4.357.
The SDK was installed during the full run; subsequent serial checks explicitly
selected the installed 1.4.357 layer using `VK_LAYER_PATH`.

The working tree based on `44b3ff6c0` passed seven CUDA runtime/workload tests
(including live device tests), 22 Forge tests, and both explicit NVIDIA Vulkan
prepared allocation/IQ qualifications. The shared affine example verified 513
results through both CUDA and Vulkan. Clippy passed for all changed runtime,
domain, Forge and test packages; all-feature WASM checks passed for the native
CUDA consumers and Forge. The Aether link changes match upstream `8c064090e`.

The initial full `cargo xtask test` completed with **3,842 passed, 12 failed, 41 skipped**.
Harness pass counts include expected failures and unsupported-backend cases;
they are not a count of successful hardware workloads. Every full-run failure
is retained below, including the subsequent investigation. No baseline comparison
was used to dismiss a failure as unrelated.

| Adapter/backend | Test                                                                      | Full run and serial recheck                                                                                                                       |
| --------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| NVIDIA Vulkan   | `big_compute_buffers::tests::two_buffers`                                 | Out-of-memory/device loss reproduced serially; passes after budget-aware heap fallback                                                            |
| NVIDIA Vulkan   | `multiview::draw_multiview_preserves_inactive_layers`                     | Inactive layer 2 became zero instead of 159 with one sample; passes with the exact-driver explicit-clear workaround                               |
| NVIDIA Vulkan   | `queue_transfer::queue_write_texture_buffer_oob`                          | Full-run resource failure; passed serially                                                                                                        |
| NVIDIA Vulkan   | `query_set::drop_failed_timestamp_query_set`                              | Full-run out-of-memory; passed serially                                                                                                           |
| Intel Vulkan    | `big_compute_buffers::tests::two_buffers`                                 | Incorrect readback reproduced serially; passes after correcting inner runtime-array bounds in SPIR-V                                              |
| Intel Vulkan    | `multiview::draw_multiview_multisample`                                   | Readback mismatch; passes with the exact-driver per-layer resolve workaround                                                                      |
| Intel Vulkan    | `multiview::draw_multiview_multisample_layer_range_discard`               | Readback mismatch; passes with the exact-driver per-layer resolve workaround                                                                      |
| Intel Vulkan    | `multiview::draw_multiview_preserves_inactive_layers`                     | Readback mismatch; passes with the exact-driver per-layer resolve workaround                                                                      |
| Intel Vulkan    | `planar_texture::nv12_texture_rendering`                                  | Abort from a validation-callback panic on unsupported parent-image usage; now verifies a safe validation rejection, not successful NV12 rendering |
| Intel DX12      | `binding_array::storage_textures::binding_array_storage_textures`         | Dynamic loads return zero with both FXC and DXC; capability withdrawn only for the reproduced adapter/driver                                      |
| Intel DX12      | `binding_array::storage_textures::partial_binding_array_storage_textures` | Same dynamic-load failure; uniform/static-index coverage remains enabled                                                                          |
| Naga host test  | `wgsl_errors::recursion_depth_template`                                   | Stack-overflow abort reproduced serially; passes after iterative precedence parsing, including explicit 2 MiB stack regressions                   |

Earlier NVIDIA passthrough failures caused by a missing DXC executable passed
after installing the repository-pinned DXC. The auxiliary device-error test
passed after replacing the test harness's unconditional static-DXC selection
with `Auto`. Initial CUDA compiler setup and Windows verbatim-path failures
were corrected before the successful numerical checks. The big-buffer fixture
uses eight 128 MiB storage buffers plus eight readback buffers and GPU-assisted
validation; its failures must not be conflated with the bounded CUDA operations.

The follow-up native qualification ran **297 tests, all passing**, including live
CUDA supplied kernels, resident GEMV/GEMM/fused-FFN, signed-i16 IQ, both prepared
Vulkan qualifications, multiview clear/draw tests and WGSL parser regressions.
A newly added precedence fixture initially used invalid `const_assert`
parenthesization; correcting the fixture preceded this passing run.

The next investigation ran **57 tests: 55 passed and the two Intel DX12 dynamic
storage-texture cases failed**. Both large-buffer tests, the focused SPIR-V
out-of-bounds regression, and all planar cases passed. Intel big-buffer controls
also failed with GPU-assisted validation disabled and workgroup-uniform descriptor
indices. The source bug used the outer eight-descriptor count for an inner
runtime array, clamping accesses to element seven. A small numerical GPU
regression now covers this independently of the large-buffer example.

For Intel DX12, constant descriptor zero/fifteen and a static-index switch
produced correct results, while dynamic indexing failed even with a runtime-zero
index and one-thread workgroups. Both compiler choices reproduced the failure.
The `8086:1912` / `31.0.101.2111` workaround therefore withdraws
`STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING`; it does not claim those failed
workloads execute correctly. Uniform storage arrays remain available, and the
validation regression checks that requesting the unsupported feature fails.

The final boundary run passed **78 tests**, including external `spirv-val`
validation of eight bounded/unbounded descriptor-array and mixed bounds-policy
cases, planar resource validation, and Intel DX12 unsupported-feature rejection.
Full-workspace `cargo clippy --all-targets --all-features`, all-feature WASM checks
for CUDA consumers, and `cargo xtask sync-repo-files --check` also passed.

The final full `cargo xtask test` passed **3,897 tests, zero failed, 45 skipped**
in 642.864 seconds, using the normal concurrency and all enumerated adapters.
Both Vulkan large-buffer workloads and the NVIDIA resource-failure cases passed
in this run. Unsupported cases, including the two withdrawn Intel DX12 dynamic
storage-array workloads, remain harness passes rather than execution claims.
The separately enabled native qualification above covers ignored CUDA hardware
tests. Generated SPIR-V snapshots retain the new descriptor and runtime-array
bounds guards; line-ending-only snapshot rewrites were discarded.

The rebuilt `routed_affine` example verified 513 CPU-reference results for NVIDIA
automatic CUDA selection, required CUDA, explicit Vulkan, and Intel's portable
fallback on a CUDA device mismatch. Requiring CUDA for the selected Intel adapter
correctly rejected the mismatch. An optional all-feature example rebuild stalled
on the static DXC dependency's GitHub metadata request; the normal-feature example
build completed locally without that download.

Local logs are retained under `target/`: `nvidia-full-suite.log`,
`nvidia-serial-failure-recheck.log`, `nvidia-resource-failure-recheck.log`,
`nvidia-cuda-final-tests.log`, `nvidia-prepared-qualification.log`,
`nvidia-routing-auto.log`, `nvidia-routing-vulkan.log`, `nvidia-clippy.log`, and
`nvidia-wasm-check.log`. Follow-up logs include `swarm-native-qualification.log`,
`swarm-remaining-regressions.log`, `swarm-final-boundary-tests.log`,
`swarm-clippy-final.log`, `swarm-wasm-check.log`, and `swarm-full-suite.log`.
Final route receipts are `swarm-routing-auto.log`, `swarm-routing-cuda.log`,
`swarm-routing-vulkan.log`, `swarm-routing-intel-auto.log`, and
`swarm-routing-intel-required.log`.
These local artifacts are not committed or CTS evidence.

### Portable reservation accounting migration

`etalon-resources` supplies dependency-free, `no_std` classified byte accounting.
The caller supplies global/partition capacity and recorded-usage snapshots;
application policy and hardware completion remain outside this library. A guard
must live until the corresponding resource may safely release its reservation.
Separate total/class observations are not an atomic multi-counter snapshot.

On 3 November 2026, seven unit tests and nine independently authored integration
tests passed in both debug and optimised builds. The independent cases cover
concurrent fixed-budget admission, exact class totals, overflow, invalid classes,
shrinking budgets, zero-byte requests, unwind cleanup and measured zero heap
allocations across successful/refused admissions and release. The package passed
Clippy with warnings denied and `wasm32-unknown-unknown` compilation. These checks
qualify scalar accounting; they do not establish GPU execution or Qualia caller
adoption.

```text
cargo test --locked --offline -p etalon-resources
cargo test --locked --offline -p etalon-resources --release
cargo clippy --locked --offline -p etalon-resources --all-targets -- -D warnings
cargo check --locked --offline -p etalon-resources --target wasm32-unknown-unknown
```

### Fixed-capacity residency accounting

`etalon-resources::residency` holds caller-owned inline slot metadata and performs
no device operations. Six authored and seven independent tests cover all-state
preservation after refused resize, checked aggregate overflow, maximum and tied
LRU ticks, exact errors, zero slots/bytes and measured zero allocation with a
positive control. Same-size readmission only promotes pinning; resized readmission
uses the supplied pin value. Explicit release/cancellation can remove pinned
slots; pinning protects against automatic eviction.
On 3 November 2026, all 29 resources tests (16 reservation and 13 residency)
passed in debug and release; Clippy with warnings denied and WASM compilation
also passed.

These tests qualify metadata accounting. They do not prove physical GPU resource
retirement, synchronization or Qualia caller adoption. Run the same resources
package test, release, Clippy and WASM checks listed above.

### Borrowed P64 reader migration

The initial `etalon-model` library preserves P64 v4 layout constants and records,
and exposes a borrowed `P64View` with explicit Full, Metadata or Structure
integrity. Parsing and bounded views allocate no heap, including malformed-input
errors. Raw container bytes preserve unknown metadata and padding; legacy
hyperparameter read/write methods still normalize their reserved eight bytes.
CRC-32C is an unkeyed integrity check, not authentication.

On 3 November 2026, eight unit and seven independent tests passed in debug and
optimised builds. Independent tests include allocator-positive-control checks,
corruption and range refusal, all integrity modes, unknown tensor metadata,
raw names/tokenizer/manifold bytes, and bounded schedule access. Against frozen
actual Qualia reader/layout/CRC source, 47 hand-authored cases produced matching
acceptance for all three modes (141 decisions) and 63 complete tensor-entry
comparisons, with matching headers, hyperparameters and tokenizer bytes, in both
profiles. Unused semantic/GGUF boundaries in that isolated legacy fixture panic
if reached and are not qualified by it. Clippy and WASM compilation also passed.

```text
cargo test --locked --offline -p etalon-model
cargo test --locked --offline -p etalon-model --release
cargo clippy --locked --offline -p etalon-model --all-targets -- -D warnings
cargo check --locked --offline -p etalon-model --target wasm32-unknown-unknown
```

This packet does not migrate the compiler/transcoder or Qualia callers and does
not establish browser execution. It preserves the existing parser's scope:
schedule contents, alternate-blob validity, dtype/shape products and manifold
meaning are not certified by successful parsing. The schedule accessor adds
explicit bounds, alignment and metadata-containment checks. Environment-based
integrity selection and semantic/GGUF adapters remain Qualia responsibilities.

### Prepared-payload P64 writer migration

`etalon-model::p64::build` plans into caller-owned entry scratch and writes into
caller-owned output, with explicit byte budgets. It preserves three distinct
legacy layouts: GGUF layer packing and final 64-byte padding, raw-tensor page
alignment without final padding, and vision final page padding. The API accepts
already prepared borrowed payloads; it does not sort tensors, interpret semantic
metadata, convert weights or provide streaming transformation callbacks.

Five writer unit tests and seven independent tests cover full-file equality with
nine actual legacy outputs, checked arithmetic and 32-bit offset limits, capacity
and byte budgets, unchanged caller buffers on refusal, opaque metadata, and
zero-allocation planning/writing with an allocator positive control. The golden
profiles are F32 verbatim at page sizes 256, 512 and 4096 bytes; other numerical
conversion and FFN rewrite profiles are not qualified by these fixtures.
On 3 November 2026, all 27 model tests (15 existing reader and 12 new writer
tests) passed in debug and release, with Clippy warnings denied and WASM
compilation also passing.
The reproducible helper in `crates/etalon-model/tests/support/` checks immutable
Qualia source provenance and all nine SHA-256 hashes before retaining output.
It uses a separately supplied Qualia checkout; Etalon has no Qualia dependency.

Run the same package test, release, Clippy and WASM commands listed above. Actual
Qualia compiler/transcoder delegation and bounded streaming transforms have
separate adoption gates.

### Sized P64 planning and payload streaming

The sized planner shares the prepared-payload writer's layout implementation and
can plan large tensors without resident payload bytes. Its separate `FfnLegacy`
profile preserves the historical page-aligned FFN payloads and raw schedule
bytes. `stream_into` freezes requirements before emission, charges used entry,
requirement and checksum prefixes plus full supplied byte/f32 workspace capacities, checks exact lengths and
CRC replay across two payload passes, and latches ignored write refusals.
Arbitrary sink failures are explicitly incomplete; confirmed byte counts include
only successful calls, since a failing sink may have partially written its chunk.

On 3 November 2026, all 41 model tests passed in debug and release, including
11 independent streaming tests. Clippy with warnings denied and WASM compilation
passed. Coverage includes all nine earlier goldens, a 3,648-byte actual legacy
FFN ternary golden, nonzero raw schedules, one-GiB metadata-only planning,
zero-allocation measurements, insufficient budgets, short/oversized payloads,
ignored refusals, changed replay, source/sink failures, cancellation and unwind.
The regeneration helper checks the frozen Qualia source and golden SHA-256.
These fixtures do not establish Q4/AWQ numerical conversion parity.

Streaming standard GGUF schedule generation scans tensors per layer and emits
metadata twice, taking O(layers × tensors) work without extra schedule storage.
The existing borrowed writer retains its linear schedule path. Measure that
tradeoff before adopting streaming in a compiler; no throughput uplift is claimed.
Atomic filesystem publication and actual Qualia compiler/transcoder delegation
remain separately qualified adapters. Run the model package commands above.

### Detailed P64 planning diagnostics

`plan_into_detailed` and `plan_layout_into_detailed`, exported from
`etalon-model::p64::build`, return `PlanError` with the existing `BuildError`
kind and a `PlanErrorContext`. Context identifies the field, optional tensor
index and operation: validation, addition, multiplication, alignment or offset
conversion. Both new APIs use the same canonical planner. Existing `plan_into`
and `plan_layout_into` return the unchanged error kind and retain validation
order, checked arithmetic, exact bytes and untouched scratch on refusal.

On 3 November 2026, all 62 model tests passed in debug and release, including
four new unit tests and four independent diagnostic tests. Clippy with warnings
denied and WASM compilation passed. Coverage includes existing prepared and
streaming goldens, original error-category projection, distinct virtual blob
size/start/final-size failures, refusal transactionality and positive-controlled
zero allocation. Run the model package commands above.

Detailed context enables consumer-specific diagnostics; it does not establish
universal legacy compiler error precedence when several inputs are invalid.
Qualia compiler adoption and its error mapping remain separately qualified.

### Bounded neutral GGUF traversal

`etalon-model::gguf` provides complete metadata validation and a separate
successful-prefix KV cursor with sticky failure status. It retains raw names,
ordered/duplicate values, all tensor dimensions, dtype codes and offsets without
payload residency. Explicit version and alignment policies, cumulative work,
header-inclusive byte limits and a fixed nonrecursive stack bound traversal.
Tensor-name hashing retains all 64 FNV-1a bits.

On 3 November 2026, all 54 model tests passed in debug and release, including
six authored and seven independent GGUF cases; Clippy with warnings denied and
WASM compilation passed. Independent checks cover all 13 value types, raw-field
retention, metadata-only input, strict and prefix behavior, depth/work/byte limits,
truncation, extreme lengths and positive-controlled zero allocation. Against
19 actual legacy cases, 16 acceptance decisions agree and 11 raw descriptor
comparisons match. The strict parser intentionally rejects three malformed cases
accepted by the legacy path: unknown value type, unknown empty-array subtype and
incomplete string array.

These differences do not qualify replacing caller behavior. Tokenizer recovery,
rank projection, version defaults, hardcoded alignment, model interpretation and
filesystem ownership remain separate adapters. Run the model package commands
above; no Qualia dependency is introduced.

### Prepared allocation checks

The prepared native compute API has separate reuse/validation regressions and an
explicitly ignored allocator qualification binary. See
[Prepared native compute](prepared-compute.md#qualification) for its allocation
boundary, target evidence requirements, and qualification command.

Simple tests can fail to detect when necessary memory initialization is omitted because
allocations that happen to be satisfied by fresh kernel-provided pages are zero, even though
the actual allocator being invoked does not guarantee that.

To improve our coverage of memory initialization, we set `LVP_POISON_MEMORY=true` in Linux
(Vulkan) CI. This instructs llvmpipe to fill all newly initialized memory with a non-zero
value, so tests will reliably fail if the memory is not properly initialized.

### Numerical inference kernel

See [numerical inference kernel qualification](inference-kernel.md) for the
independent original/candidate checks, caller contracts and retained limitations.
The additive library passed 25 tests in debug and release, strict Clippy and WASM
compilation. Embedded CUDA source identity does not establish device execution.
