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
