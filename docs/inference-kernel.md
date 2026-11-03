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
