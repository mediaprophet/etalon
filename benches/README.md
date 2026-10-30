Collection of CPU benchmarks for `etalon`.

These benchmarks are designed as a first line of defence against performance regressions and generally approximate the performance for users.

## Usage

```sh
# Run all benchmarks
cargo bench -p etalon-benchmark
# Run a specific benchmarks that contains "filter" in its name
cargo bench -p etalon-benchmark -- "filter"
```

Use `ETALON_BACKEND` and `ETALON_ADAPTER_NAME` to adjust which device the benchmarks use. [More info on env vars](../README.md#environment-variables).

## Comparing Against a Baseline

To compare the current benchmarks against a baseline, you can use the `--save-baseline` and `--baseline` flags.

For example, to compare v28 against trunk, you could run the following:

```sh
git checkout v28
# Run the baseline benchmarks
cargo bench -p etalon-benchmark -- --save-baseline "v28"

git checkout trunk
# Run the current benchmarks
cargo bench -p etalon-benchmark -- --baseline "v28"
```

The current benchmarking framework was added before v28, so comparisons only work after it was added. Before that the same commands will work, but comparison will be done using `criterion`.

## Integration with Profilers

The benchmarks can be run with a profiler to get more detailed information about where time is being spent.
Integrations are available for `tracy` and `superluminal`.

#### Tracy

Tracy is available prebuilt for Windows on [github](https://github.com/wolfpld/tracy/releases/latest/).

```sh
# Once this is running, you can connect to it with the Tracy Profiler
cargo bench -p etalon-benchmark --features tracy,profiling/profile-with-tracy
```

Tracy is unavailable on OpenHarmony's public NDK. Its
[public pthread header](https://github.com/openharmony/interface_sdk_c/blob/OpenHarmony-6.0-Release/third_party/musl/ndk_musl_include/pthread.h)
does not expose pthread cancellation, and its
[errno header](https://github.com/openharmony/interface_sdk_c/blob/OpenHarmony-6.0-Release/third_party/musl/ndk_musl_include/errno.h)
omits the GNU program-name globals required by the bundled Tracy client.
Enabling this benchmark crate's `tracy` feature on OpenHarmony prints an
unsupported-profiling diagnostic and retains ordinary benchmark execution and
timing. The optional client is excluded for that target; its feature and behavior
on other native targets are unchanged. OpenHarmony CI continues checking every
benchmark with all crate features enabled. Do not enable the separate
`profiling/profile-with-tracy` backend there. Supporting Tracy on OpenHarmony
requires an upstream platform port using public NDK APIs and runtime validation.

#### Superluminal

Superluminal is a paid product for windows available [here](https://superluminal.eu/).

```sh
# This command will build the benchmarks, and display the path to the executable
cargo bench -p etalon-benchmark --features profiling/profile-with-superluminal -- -h

# Have Superluminal run the following command (replacing with the path to the executable)
<path_to_exe> --bench "filter"
```

#### `perf` and others

You can follow the same pattern as above to run the benchmarks with other profilers.
For example, the command line tool `perf` can be used to profile the benchmarks.

```sh
# This command will build the benchmarks, and display the path to the executable
cargo bench -p etalon-benchmark -- -h

# Run the benchmarks with perf
perf record <path_to_exe> --bench "filter"
```

## Benchmarks

#### `Renderpass Encoding`

This benchmark measures the performance of recording and submitting a render pass with a large
number of draw calls and resources, emulating an intense, more traditional graphics application.
By default it measures 10k draw calls, with 90k total resources.

Within this benchmark, both single threaded and multi-threaded recording are tested, as well as splitting
the render pass into multiple passes over multiple command buffers.
If available, it also tests a bindless approach, binding all textures at once instead of switching
the bind group for every draw call.

#### `Computepass Encoding`

This benchmark measures the performance of recording and submitting a compute pass with a large
number of dispatches and resources.
By default it measures 10k dispatch calls, with 60k total resources, emulating an unusually complex and sequential compute workload.

Within this benchmark, both single threaded and multi-threaded recording are tested, as well as splitting
the compute pass into multiple passes over multiple command buffers.
If available, it also tests a bindless approach, binding all resources at once instead of switching
the bind group for every draw call.
TODO(https://github.com/gfx-rs/wgpu/issues/5766): The bindless version uses only 1k dispatches with 6k resources since it would be too slow for a reasonable benchmarking time otherwise.

#### `Device::create_buffer`

This benchmark measures the performance of creating large buffers.

#### `Device::create_bind_group`

This benchmark measures the performance of creating large bind groups of 5 to 50,000 resources.

#### `naga::back`, `naga::compact`, `naga::front`, and `naga::valid`

These benchmark measures the performance of naga parsing, validating, and generating shaders.
