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
