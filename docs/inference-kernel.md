# Numerical inference kernel

`etalon-inference-kernel` contains dependency-free numerical contracts extracted
from Qualia's existing shared kernel. It provides dense f32 matrix-vector apply,
GGML Q4_K reconstruction and activation lookup, caller-owned workspace views,
paged GQA scalar attention and the existing segmented/tiled CUDA source templates.
It does not select a device, execute CUDA, authorize inference or publish tokens.

The package preserves the original standard-library math requirement, empty
default feature set and source license declaration. It has no external dependencies
and compiles for native and wasm32-unknown-unknown targets. Q4_K payloads here are
144 bytes per 256 values; the 160-byte Q4_K_SOA and DirectML formats are distinct.

The numerical source and its tests are moved without algorithm changes. Narrow
Clippy style allowances on module declarations preserve that frozen implementation
and API. They do not suppress correctness lints. The original 20 tests remain.

## Existing caller contracts and limitations

Use validated descriptors and correctly sized caller buffers. Public structs can
be constructed directly, so an arbitrary OperatorView is not proof of validation.
The low-level Q4_K lookup assumes input/output capacities supplied by its wrapper;
its lookup preparation currently observes the entire supplied input slice. Avoid
extra input tails when calling that low-level API. These historical behaviors are
retained for compatibility, pending separately tested improvements.

Paged attention uses the caller's immutable arena and page table and writes output
one head at a time. MissingPage can leave partial output; it is not a transactional
output API. PagedKvConfig has public fields and some derived size helpers retain
historical saturation/unchecked arithmetic. Callers must provide a valid bounded
configuration; helpers do not constitute complete validation of arbitrary values.

Embedded CUDA text preserves the original source and launch contract. Compilation
of this Rust crate does not prove a CUDA launch, numerical GPU parity or throughput.
Real device integration and Qualia forwarding adoption have separate qualification
gates. Developer/governed session control is outside this numerical package.

## Checks

```sh
cargo test --locked -p etalon-inference-kernel
cargo test --locked -p etalon-inference-kernel --release
cargo clippy --locked -p etalon-inference-kernel --all-targets -- -D warnings
cargo check --locked -p etalon-inference-kernel --target wasm32-unknown-unknown
```

Independent migration fixtures compare the frozen old implementation with this
library on the same target, preserving operation order and error/output behavior.
Cross-target floating-point equivalence and runtime backend execution need their
own evidence. Temporary original-source references are qualification artifacts,
not a second maintained numerical implementation.

On 3 November 2026, all 25 tests (20 preserved unit tests and five independent
contract tests) passed in debug and optimized builds, 50 executions with no
failures or ignored tests. The same five independent fixtures passed against the
frozen original library. They cover exact dense/Q4_K/attention fingerprints,
workspace and output tails, refusal behavior, nonfinite inputs, CUDA source bytes
and positive-controlled zero allocation. Strict Clippy and WASM compilation pass.
The initial strict Clippy attempt reported only inherited style findings; the
scoped annotations described above were independently reviewed.

## Dense CPU transformer execution

`dense_cpu` executes embedding, RMS normalization, interleaved RoPE, grouped-query
causal attention, KV updates, gated FFN, residuals and output projection. A
`TensorKernels` adapter supplies row decoding and matrix-vector operations;
the library owns the forward computation. `DensePlan` borrows tensor bindings,
checks dimensions and computes workspace requirements. `ingest_token` executes
prefill without output projection; `run_token` also produces logits and selects
the lowest-index finite maximum.

The caller supplies all thirteen execution buffers and the position. Admission
charges their full supplied capacities against an explicit byte budget, then
passes only used slices to kernels. The budget excludes model weights, bindings
and adapter-owned storage. Allocation behavior and codec correctness of an
arbitrary adapter require their own qualification.

Missing-tensor checks, numerical operation order and partial-failure behavior
follow the original dense CPU path. A failed step can leave KV and scratch
modified. `reset_kv` clears only used KV storage; it does not restore a prior
step, reset a sampler or authorize publication. Tokenization, architecture
admission, environment policy and governance stay outside this module. Native
MoE and recurrent execution are separate paths and have no dense fallback here.

### Dense qualification

On 3 November 2026, all 36 package tests passed in debug and release (72
executions), including five new unit tests and six independent dense tests.
Strict Clippy, formatting and WASM compilation passed. The independent tests
compare 42 states containing 83 f32 values each against actual original Qualia
forward code at `07018bb8dbb123f105a91609532b9b564e54a2af`, with bit-identical
debug and optimized traces. The two-layer F32 fixture covers prefill, decode,
reset, logits, every workspace, partial KV failures, invalid shapes, nonfinite
values and output rows differing from configured vocabulary. Additional checks
cover all thirteen buffer shortages, spare capacity, callback failures and
positive-controlled zero allocation of the library plus the fixture adapter.

A separate local baseline exercised the original thirteen native codec
row-decode/GEMV routes, truncated rows and direct native Q8 processing, and
confirmed dtype 158 remains unsupported in that CPU path. Those codec results
are distinct from the full F32 engine traces: they do not qualify a new
quantized-model adapter or actual WASM SIMD execution. Pretrained full-prompt
execution, GPU execution, Qualia engine adoption and physical rollback remain
separate gates. No throughput improvement is claimed.

Run the package commands above and the retained helper checks:

```sh
python crates/etalon-inference-kernel/tests/support/regenerate_dense_cpu_golden.py verify
python -B crates/etalon-inference-kernel/tests/support/test_regenerate_dense_cpu_golden.py
```

The six helper tests cover provenance/output validation, exclusive promotion
and cleanup on failure or interruption. To regenerate the reference into a new
file, use:

```sh
python crates/etalon-inference-kernel/tests/support/regenerate_dense_cpu_golden.py generate --qualia-root <Qualia-Git-checkout> --dependencies <Qualia>/target/debug/deps --output <new-file> --optimized
```

The helper reads SHA-checked Git objects at the frozen revision, independently
of the checkout's current files. It requires cached Rust 1.96.1 libraries for
serde, serde_json, bytemuck, half and memmap2; it performs no Cargo build,
network access or model download. Only cold construction bypasses the original
1000-token vocabulary floor for the tiny fixture. Actual forward/math/codecs
remain original. Source extraction has a two-MiB budget; the 32-MiB compiled
scratch check limits retention after compilation, not compiler peak memory.
Temporary scratch is cleaned and an existing destination is never overwritten.
