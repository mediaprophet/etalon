# Prepared native compute

`Adapter::prepare_compute` creates a private execution owner for a fixed compute
program. Preparation validates WGSL and the equivalent command stream through
Etalon Core, allocates buffers and completion state, and records reusable Vulkan
commands. Each execution uploads into persistent mapped storage, submits those
commands, waits on its private fence, and copies readback into caller storage.

This path is implemented for native Vulkan with the `wgsl` feature. Other
backends, including browser WebGPU, return `UnsupportedBackend`. It does not
replace the ordinary encoding, queue submission, or `map_async` APIs, which can
allocate. It does not provide WebGPU CTS coverage.

## API and resource ownership

The public entry points are:

```rust,ignore
let mut prepared = adapter.prepare_compute(&descriptor)?;
let sequence = prepared.execute(input, parameters, output, timeout)?;
prepared.retire()?;
```

`PreparedComputeDescriptor` supplies validated WGSL, its compute entry point,
fixed input/parameter/output byte capacities, fixed `[x, y, z]` workgroup counts,
and a logical buffer budget. Group zero has this fixed layout:

| Binding | Resource                  |
| ------- | ------------------------- |
| 0       | Read-only storage input   |
| 1       | Read-write storage output |
| 2       | Uniform parameters        |

Input can be any prefix of its prepared capacity; remaining input bytes are
zeroed on every execution. Parameters must have exactly the prepared length.
Readback can request any output prefix. The entire logical GPU output buffer is
cleared before each dispatch, including bytes the shader does not write.
Variable logical workloads must pass their length in parameters and guard work
inside the shader: the prepared dispatch dimensions do not change between runs.
The GPU regression source in
`tests/tests/etalon-gpu/prepared_compute.rs` provides a complete example.

Preparation requests a new private device and its matching queue from the
selected adapter. No ordinary device, queue, buffer, mapped view, or submission
index escapes the owner. It cannot be constructed from an existing
`DeviceQueuePair`. The private device is excluded from instance-wide polling,
so another thread calling `Instance::poll_all` cannot consume its completion or
retire its resources. Execution requires an exclusive mutable borrow and permits
one in-flight operation per owner.

Preparation requires default device limits and no optional device features.
WGSL runtime checks, normal Core validation, and shader workgroup-memory
initialization remain enabled. Instance `DEBUG`, `VALIDATION`,
`GPU_BASED_VALIDATION`, and `DEBUG_PRINTF` flags are rejected: those flags can
introduce backend diagnostic callbacks into the allocation boundary. This does
not disable Core's safe API or shader validation. An adapter's device-request
consumption rules still apply under strict WebGPU compliance.

## Allocation boundary

The implementation is designed to avoid Rust allocator calls in Etalon-owned
execution. This is a source property awaiting target qualification, not a
measurement certificate for every device or failure mode.

| Stage            | Storage and behavior                                                                                                    |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Preparation      | Cold allocation for validation, compilation, private device, buffers, mappings, command pool, command buffer, and fence |
| Upload           | Copies into an existing mapped upload allocation; zeroes unused input capacity                                          |
| Recording        | Performed once during preparation; no hot command encoder                                                               |
| Submission       | Reuses one command buffer and fence; submission descriptors use stack arrays                                            |
| Completion       | Direct private-fence wait; no boxed map or work-done callbacks, lifecycle vectors, or public polling                    |
| Readback         | Uses retained mapping metadata and copies into caller storage after successful completion                               |
| Execution errors | Fixed-size `PreparedComputeError`; no diagnostic string construction                                                    |
| Retirement       | Cold drain and destruction; may block or quarantine resources                                                           |

The prepared path bypasses ordinary `Queue::write_buffer`, command recording,
submission tracking, and map-callback collection only for its sealed resources
and queue. Ordinary Core creation and command validation establish the safe
boundary before the specialized HAL stream is recorded. This is not permission
to submit arbitrary HAL resources or reuse ordinary baked command buffers.

Native driver calls, OS services, externally injected layers and hooks may
allocate or block. Source review and the Rust allocator instrument do not
certify those components. Browser JavaScript promises and callback allocation
are outside this implementation's supported scope.

The logical buffer budget is checked before buffer creation. For input size
`I`, parameter size `P`, and output size `O`, the five buffers require:

```text
input + parameters + output + upload + readback
= I + P + O + (I + P) + O
= 2 * (I + P + O)
```

For IQ input/output of `4N` bytes each and 16 parameter bytes, this is `16N + 32`
bytes. The budget excludes allocator padding, device initialization resources,
shader/compiler data, command pools, descriptors, driver memory, and caller-side
sample/conversion buffers. It is not a total GPU or process memory cap.
Noncoherent mappings are accepted only when the cached atom-aligned flush and
invalidate ranges remain within their owned allocations; other configurations
fail during preparation.

## Completion, failures, and retirement

Successful `execute` returns a checked, nonwrapping sequence local to that owner.
It is a receipt, not a `SubmissionIndex` accepted by `Device::poll`. There are no
externally supplied completion tokens or late user callbacks to reuse.

Capacity and parameter-length errors occur before upload/submission, clear the
caller output, and leave the owner reusable. Backend failures, timeout, device
loss, and sequence exhaustion permanently poison the owner. Every execution
error clears caller output; a successful fence wait is required before reading
mapped results. Transfers and host visibility have explicit Vulkan barriers,
with noncoherent flush/invalidate operations where required.

The execution timeout bounds the fence wait, not host copies or driver calls.
A timeout does not cancel GPU work. The owner retains its buffers, mappings,
commands, and device; subsequent execution fails with `Poisoned`. Call `retire`
to make the owner terminal and establish completion before destruction. This
cold operation can block beyond an earlier execution timeout and can be retried
after an error.

Destruction drains uncertain work before freeing resources. If completion
cannot be established, it retains the entire private device and resource bundle
rather than freeing in-use memory. That exceptional quarantine retains memory;
it is not a bounded global recovery policy. A host requiring recovery accounting
should call `retire` explicitly and record its result. Vulkan device loss is
handled according to its resource-in-use retirement rules. See the
[Vulkan device-loss specification](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html)
and [host/device synchronization rules](https://docs.vulkan.org/spec/latest/chapters/synchronization.html).

Neither retirement, unmapping, nor destruction promises secure erasure or
hardware identity attestation. Host authorization/revocation checks require
their own immutable policy lease; device identifiers are not that authority.

## Qualification

Implementation and source review do not automatically admit a sensing provider
as `HotZeroHeap`. The consuming provider must retain its unqualified allocation
classification until its precise execution path and target have been measured.
The GPU numerical operation and any CPU-to-GPU conversions also need independent
correctness evidence.

Ordinary GPU regressions cover prepared reuse, input padding, output clearing,
capacity retry, descriptor/shader validation, and unsupported backends. Unit
tests cover descriptor budgets and nonwrapping poisoned state. The separate
`etalon-prepared` test binary contains an explicitly ignored allocator
qualification test: after cold preparation and warmup, it counts Rust allocator
activity across all threads during repeated execution and preflight failures.

After the build/test hold applicable to a work session is lifted, qualification
should include:

1. Relevant compile/lint/format checks and prepared GPU regressions.
2. The ignored allocator test run alone, with target, driver, features, source
   revision, and diagnostic configuration recorded.
3. Timeout/device-loss and uncertain-retirement fault injection, including
   zeroed caller output and prevention of resource reuse.
4. Noncoherent-memory targets and repeated execution with poisoned initial
   memory, plus independent CPU output oracles.
5. The full `cargo xtask test` required for changes affecting GPU behavior.

The explicit allocator command is:

```text
cargo test -p etalon-test --test etalon-prepared -- --ignored --test-threads=1
```

That command is a qualification instruction, not a statement that it has run or
passed. Build/test execution and target evidence belong to the coordinated
release qualification record. Software Vulkan results must be identified as
software execution, separately from physical CPU/iGPU/device qualification.
