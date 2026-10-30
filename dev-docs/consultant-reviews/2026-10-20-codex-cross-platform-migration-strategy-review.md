# Architectural Review: Cross-Platform Etalon Migration and Implementation Strategy

**Reviewer:** Codex / OpenAI  
**Date:** 2026-10-20  
**Review Target:** Etalon v0.0.1 and QualiaDB subsystem extraction  
**Status:** Proposed recommendations; preparation can proceed, implementation gates below remain open  
**Scope:** Dependencies, shader compilation, graphics, graph processing, tensor execution, memory, scheduling, browser workers, audio, vision, EMF, simulation, and extension boundaries.

**Research and literature:** W3C WebGPU/WGSL; Khronos Vulkan BDA, external memory, and cooperative matrices; Microsoft DirectX LinAlg, Work Graphs, and Enhanced Barriers; Apple Metal tensor operations; NVIDIA CUDA and CUTLASS; Triton; FlashAttention-3; PagedAttention; Onesweep. Links accompany the relevant conclusions.

## 1. Executive Verdict & State-of-the-Art Strategic Vision

**Modify the execution plan before broad migration. Proceed with dependency isolation, contract definition, and reproducible verification first.** The proposed separation of database semantics from reusable graphics and compute is sound. The present documents nevertheless mix implemented capabilities, unfinished adapters, speculative API designs, and performance claims without reproducible evidence.

Etalon should serve the complete workload portfolio. Its organizing abstraction should be a prepared resource-and-operation graph with backend-specific execution, not an inference engine or a CUDA-centric runtime. Rendering, graph queries, audio transforms, image processing, and tensor operations should share resource ownership, device discovery, compilation metadata, and completion tracking while retaining appropriate domain interfaces.

The highest-priority findings are:

1. **The proposed compatibility facade does not resolve the actual dependency version mismatch.** QualiaDB requests `wgpu = "30"` and `naga = "30"`; a package named `wgpu` at `0.0.1` cannot satisfy that requirement. Renaming a dependency locally does not redirect transitive users.
2. **A graph kernel has a correctness defect, not merely a performance opportunity.** The LSD radix scatter uses unordered atomic placement. Later passes can destroy ordering established by earlier digits. The caller checks the final result and can fall back, but migration must not certify this as a correct GPU sort.
3. **Native Metal inference is not implemented in the inspected lane.** Its availability predicate can nevertheless return true. This must be distinguished from Etalon's existing Metal rendering and cooperative-matrix support.
4. **Raw addressing, memory residency, and synchronization are separate capabilities.** BDA does not make file-backed host memory device-local or establish safe asynchronous ownership. ReBAR is not evidence that arbitrary host mappings can be imported.
5. **Compiler reflection does not establish shader memory safety.** Keep the existing safe validation boundary, and attach explicit obligations to native unsafe paths. Move reusable validation out of repeated execution where possible; do not replace it with a trust label.
6. **The latest backend landscape differs from the blueprint.** DirectX has a LinAlg matrix preview associated with SM 6.10; Metal 4 has tensor operations; WebGPU transient attachments are already present in the inspected editor's draft and Etalon types. Treat proposals, compiler support, driver support, and verified device execution as separate states.
7. **Migrate vertical slices with their consumer and oracle immediately.** Waiting until a final phase to reconnect QualiaDB postpones discovery of ownership, type identity, and domain-coupling problems.

No universal tile shape, zero-copy policy, or dispatch backend is mathematically optimal across this workload set. The defensible performance objective is a measured per-device Pareto frontier over correctness, latency, throughput, memory consumption, and responsiveness. Hardware-specific implementations belong behind shared operation contracts and explicit numerical policies.

## 2. Forensic Technical Analysis & Theoretical Foundations

### 2.1 Evidence baseline and limitations

Etalon HEAD inspected: `048e43a803984de9f6fdea152cba74c9fde663d3`.  
QualiaDB HEAD inspected: `fc870ae0061789c66a638087a65c0550a345c436`.

These are repository identifiers, not proof that every inspected working-tree file equals HEAD. QualiaDB contains extensive existing changes. Another consultant review appeared during this session. This submission does not modify those files.

The required planning sources were inspected: `initial-doc.md`, plans 01 and 02, audit 04, `TODO_INDEX.md`, the consultant mandate and template. Plan 03 was also inspected for extension-boundary implications. Source review targeted the seams and failure modes below; it is not an exhaustive certification of every shader or module.

### 2.2 Findings grounded in source

| Priority / finding                                              | Exact evidence inspected                                                                                                                                                                                                                                                     | Consequence and required action                                                                                                                                                                                                                                                                                      |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1 — dependency compatibility is unresolved                     | [Etalon workspace](C:/github/etalon/Cargo.toml:73), lines 73–105; [QualiaDB workspace](C:/github/qualiaDB/Cargo.toml:97), lines 97 and 127; [core-db manifest](C:/github/qualiaDB/crates/qualia-core-db/Cargo.toml:187), lines 187–190; plan 01 §3 phase 6                   | Etalon/Naga are `0.0.1`; consumer requests include version 30. `compat/wgpu` was absent during inspection. A facade needs a compatible version and verified API/features, or consumers must explicitly migrate. Naga also needs an intentional source/type-identity strategy.                                        |
| P1 — radix sort lacks the stability required between LSD passes | [scatter shader](C:/github/qualiaDB/crates/qualia-core-db/src/shaders/graph_radix_scatter.wgsl:27), lines 27–37; [host loop](C:/github/qualiaDB/crates/qualia-core-db/src/query/graph_accel/gpu.rs:204), lines 204–257, 310–337                                              | `atomicAdd` allocates positions in arrival order. A legal final-pass schedule can turn `[1,2]` into `[2,1]`. The host's ordering/permutation checks prevent accepting that result, but fallback can hide repeated GPU failure.                                                                                       |
| P1 — Metal capability reporting disagrees with execution        | [metal_lane.rs](C:/github/qualiaDB/crates/qualia-core-db/src/inference/metal_lane.rs:70), lines 70–118, 144–152; [compile_msl](C:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/execute/wgpu.rs:1109), lines 1109–1130                                                | The execution functions return unavailable; `metal_mega_pass_available` returns true for macOS plus `metal-native`. Require a successful preparation probe before reporting an executable native lane. WGSL-to-MSL remains a separate existing path.                                                                 |
| P1 — browser readback is written as synchronous polling         | [map_read](C:/github/qualiaDB/crates/qualia-core-db/src/query/graph_accel/gpu.rs:83), lines 83–116                                                                                                                                                                           | A loop of 10,000 polls does not yield to browser promise processing. Use an asynchronous completion interface and bounded work scheduling; migration should preserve CPU fallback explicitly.                                                                                                                        |
| P1 — safety-bypass work partially duplicates existing APIs      | [device.rs](C:/github/etalon/etalon/src/api/device.rs:180), lines 180–216; [features.rs](C:/github/etalon/etalon-types/src/features.rs:439), line 439                                                                                                                        | Unsafe trusted shader construction and unsafe passthrough already exist. `MAPPABLE_PRIMARY_BUFFERS` already exists. Audit these contracts before proposing parallel flags or weakening safe API validation.                                                                                                          |
| P2 — zero-allocation evidence is weaker than its label          | [uniform belt](C:/github/qualiaDB/crates/qualia-core-db/src/render/gpu/uniform_belt.rs:266), lines 266–313                                                                                                                                                                   | The test allows fewer than 50 allocations and skips when no GPU is available. It does not establish zero allocation. Define precisely whether a claim covers owned application code, Etalon submission, or opaque driver/browser internals.                                                                          |
| P2 — the generic graph path repeatedly crosses the CPU boundary | [graph acceleration](C:/github/qualiaDB/crates/qualia-core-db/src/query/graph_accel/gpu.rs:120), lines 120–257 and 341–435                                                                                                                                                   | Each radix digit downloads a histogram and uploads offsets; sieve creates resources and downloads results into new vectors. Prepared storage, device-side prefix sums, reusable bindings, and caller-owned outputs are concrete priorities before BDA.                                                               |
| P2 — matrix support already exceeds its documentation           | [feature documentation](C:/github/etalon/etalon-types/src/features.rs:1107), lines 1107–1125; [Vulkan query](C:/github/etalon/etalon-hal/src/vulkan/adapter.rs:3573), lines 3573–3674; [Metal query](C:/github/etalon/etalon-hal/src/metal/adapter.rs:1587), lines 1587–1620 | Documentation says only 8×8 f32, but Vulkan enumerates shapes/types and Metal lists f16 and mixed accumulation. Correct capability documentation and test each returned tuple. HLSL explicitly excludes cooperative matrices at `naga/src/back/hlsl/mod.rs:827`.                                                     |
| P2 — shader caches need distinct artifact and tuning identities | [DXC cache](C:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/emit/dxc_cache.rs:12), lines 12–58; [Forge version constants](C:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/mod.rs:84), lines 84–90                                                            | The inspected DXC cache hashes concatenated source and entry point. Include length-delimited fields, compiler build, arguments, target environment, numerical policy, and ABI version. Device/driver identity additionally belongs in executable/tuning caches. Existing schema constants must change with the fork. |
| P2 — renderer extraction has substantial domain coupling        | [PortalGpu imports](C:/github/qualiaDB/crates/qualia-core-db/src/render/gpu/mod.rs:66), lines 66–76 and 430–565; [frame renderer](C:/github/qualiaDB/crates/qualia-core-db/src/render/gpu/portal_gpu/render_frame.rs:8), lines 8–71                                          | Camera, telemetry, tensor exports, VRAM ledger, frame graph, and container opacity types cross the boundary. Extract POD scene/frame inputs and an injected resource budget first; moving the directory alone retains database coupling.                                                                             |
| P2 — adjacent domains are not interchangeable GPU libraries     | [vision facade](C:/github/qualiaDB/crates/qualia-vision/src/gpu/mod.rs:1), lines 1–3; [audio contract](C:/github/qualiaDB/crates/webizen-render/src/audio_contract.rs:1), lines 1–47                                                                                         | Vision's facade reexports core-db GPU code. Audio explicitly documents allocating spectral structs. Separate reusable operations from ontology, consent, media, and presentation contracts rather than moving whole application crates.                                                                              |
| P2 — DirectStorage claims require qualification                 | [DirectML bridge](C:/github/qualiaDB/crates/qualia-core-db/src/inference/directml_bridge.rs:813), lines 813–865; GEMM submission at lines 633–777                                                                                                                            | DirectStorage is marked as a stub; GEMM creates command resources and invokes submit-and-wait. Treat DirectML, DirectStorage, DX12, and NPU execution as distinct capabilities, each with its own validation.                                                                                                        |

The radix counterexample was executed in a small host-side schedule simulation during this review. It demonstrates an allowed ordering failure; it is not a measured GPU failure frequency. An unstable final sort contract permits arbitrary ordering only among equal complete keys. It does not permit unstable intermediate LSD digit passes.

### 2.3 ABI and ownership decisions

The existing [NQuin definition](C:/github/qualiaDB/crates/qualia-core-db/src/lib.rs:1112), lines 1112–1141, is already six `u64` fields with `repr(C, align(16))`: 48-byte size, 16-byte alignment, 48-byte array stride. Its shader representation in [graph_sieve_field.wgsl](C:/github/qualiaDB/crates/qualia-core-db/src/shaders/graph_sieve_field.wgsl:5), lines 5–12, is twelve `u32` fields. Preserve the wire meaning and little-endian low/high word order; a native Rust `u64` field does not require a browser WGSL `u64`.

Use compile-time size/alignment/offset assertions plus byte fixtures shared with shader tests. Do not increase the 48-byte record to 64 bytes merely for cache alignment: that increases storage/transfer traffic by one third. For repeated field-only scans, benchmark a derived structure-of-arrays index with its construction cost amortized over reuse.

P64 is not one uniform tensor weight format. [P64 layout](C:/github/qualiaDB/crates/qualia-core-db/src/q42/p64_weight/layout.rs:114), lines 114–178, defines 64-byte aligned headers/descriptors with relative offsets and separate tensor blobs. [P64GpuWords64](C:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/ir/core.rs:10), lines 10–45, is a different 64-byte, 16-byte aligned GPU word representation. Keep file parsing, checksums, semantic roles, and transaction lifecycle in QualiaDB; Etalon consumes validated tensor views plus dtype, dimensions, strides, scales, and ownership.

The browser representation is consistent with the current [WGSL scalar types](https://www.w3.org/TR/WGSL/#scalar-types). A bit-preserving paired-word ABI is distinct from native 64-bit arithmetic.

### 2.4 Target dependency direction and extraction units

| Layer                                                                   | Owns                                                                                                                       | Does not own                                                                              |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Existing `etalon`, core, HAL, types, Naga/bridge                        | Portable API, backend resources, validation, shader translation and native API extensions that are genuinely shared        | Database rules, model formats, application policy                                         |
| `etalon-agent`                                                          | Device/session capability records, allocation leases, resource budgets, prepared submissions, completion, optional interop | Universal assertions that every device has three independent engines or one pointer model |
| `etalon-forge`                                                          | Operation/kernel IR, compiler adapters, artifact manifests, specialization, oracle and tuning interfaces                   | A dependency back into QualiaDB; all application state                                    |
| `etalon-math`                                                           | Reusable scan/reduction/sort, FFT, convolution and geometry primitives when supported by migrated implementations          | A promise that all planned algorithms already exist                                       |
| `etalon-tensor`, `etalon-render`                                        | Tensor execution and generic render resources/pass scheduling respectively                                                 | Tokenization/storage semantics; ontology or UI policy                                     |
| Optional `etalon-nquin`, audio/EMF/vision modules and domain extensions | Specialized record operations and domain adapters over the generic substrate                                               | Dependencies from foundational crates upward into these modules                           |
| QualiaDB and application consumers                                      | Query planning, storage snapshots, N3Logic, rights decisions, scene meaning, media lifecycle                               | Duplicated driver/resource lifetime machinery                                             |

Use the names already selected in the plans. Reconcile `etalon-vision` versus `etalon-render`, and explicitly retain the audio/EMF/math scope from `initial-doc.md:569–613` that the later abbreviated inventories omit. Separate crates only where ownership, platform dependencies, or release boundaries justify them.

Forge's pure IR/manifest portion should be usable without creating a device or linking a native SDK. Execution interfaces can be injected or feature-gated. `etalon-tensor` must not become the home of resource services required by render or graph workloads. Optional native executors can live behind `etalon-agent` features without implementing graphics functionality they do not support.

## 3. Definitive Solutions to the Core Strategic Questions

### Q1. HAL integration versus dedicated CUDA/PTX execution

**Use a compute-specialized executor alongside the graphics HAL, coordinated by the common agent/session layer.** This is also the appropriate pattern for DirectML and future specialized accelerators. Keep Vulkan, DX12, and Metal graphics/compute on their existing HAL paths.

[The HAL Api trait](C:/github/etalon/etalon-hal/src/lib.rs:632), lines 632–687, requires surface, texture, and render-pipeline types as well as compute types. A CUDA backend that fills those obligations with unsupported placeholders makes unsupported operations representable and broadens every backend match. That costs maintenance without demonstrating lower driver latency.

Separate cold `prepare` from hot `submit`. Preparation resolves an operation graph, compiler outputs, resources, bindings, launch geometry, hazards, numerical policy, and scratch requirements. Submission accepts bounded parameter updates and returns a completion token. Backend selection occurs once per prepared plan or batch, rather than through virtual dispatch for every scalar operation.

CUDA can use streams/events and eligible captured graphs; Vulkan records commands with synchronization2/timeline semantics; DX12 uses command lists/fences; Metal uses its command/event mechanisms; browser execution uses WebGPU submissions. These are backend lowerings, not interchangeable native graph objects. CUDA's [asynchronous execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) supports overlap, but actual concurrency depends on hardware and dependencies.

Interop must match physical adapters by stable identity, negotiate external-memory/semaphore handle support, and track ownership transitions and completion. If unavailable, report and cost an explicit copy path. Independent API devices on the same GPU do not automatically share buffers.

Prefer per-queue completion counters. A shared monotonically numbered timeline can be signalled out of execution order by independent queues, incorrectly satisfying a wait for unrelated work. Cross-queue dependencies need the specific producer's completion, plus required visibility and ownership transfers.

Measure CPU enqueue cost, GPU idle gaps, submission count, graph preparation amortization, and end-to-end completion separately. Do not publish a dispatch-latency number without the device, driver, measurement method, and workload.

### Q2. BDA, host mmap, safe Rust, and browser portability

**Expose owned resource views in safe code; make raw addresses a backend-specific unsafe escape hatch.** An opaque handle with allocation identity, device identity, generation, byte range and access rights is a useful safe API. A bare `u64` is not a lifetime or bounds proof.

Proposed contract:

```text
OwnedAllocation -> checked BufferRegion(offset, length, access)
                -> PreparedPlan retaining allocation ownership
                -> Submission retaining plan/resources until completion
                -> retirement, optional scrub, reclamation
```

Checked construction rejects integer overflow and misalignment. Submission rejects foreign-device resources, incompatible access and stale generations. An asynchronous submission cannot merely borrow an mmap until the host call returns. Keep an owning mapping lease until GPU completion; cancellation stops future work but does not revoke already submitted accesses. Mutating file-backed storage also needs snapshot/COW or another explicit transaction protocol supplied by QualiaDB.

A bounded region alone cannot sandbox an arbitrary physical-pointer shader. Safe native entry points require compiler-enforced memory restrictions or trusted, audited fixed kernels with validated dimensions and resource access; otherwise construction/launch remains unsafe. Address tables, indirect pointer targets, aliasing and resource retirement all belong in that proof.

Vulkan [BDA](https://docs.vulkan.org/guide/latest/buffer_device_address.html) supplies shader addressability; [alignment operands](https://docs.vulkan.org/guide/latest/buffer_device_address_alignment.html) are promises that must hold for each access. DX12's GPU virtual address is used by facilities such as [root descriptors](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-descriptors-directly-in-the-root-signature); it does not establish a universal HLSL integer-to-pointer dereference model equivalent to Vulkan physical storage buffers.

Memory policy must consider:

| Placement                       | Appropriate candidate                                             | Obligations                                                                                                                   |
| ------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Device-local resident           | Repeated weights, graph indexes, render targets                   | Upload amortization, residency budget, eviction and completion                                                                |
| UMA/shared allocation           | CPU/GPU handoff on supported integrated systems                   | Ownership and visibility; shared memory does not remove synchronization                                                       |
| Imported/registered host memory | Eligible externally owned pages, often sparse or low-reuse access | Import/registration support, page and allocation alignment, stable lifetime, cache maintenance and measured interconnect cost |
| Staged upload/readback          | Portable/default unsupported-import path                          | Reusable bounded staging, batching and overlap                                                                                |

Vulkan [external host memory](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_external_memory_host.html) is an explicit import facility with eligibility constraints. CUDA distinguishes ordinary mmap memory from page-locked/registered memory in its [programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf). Neither BDA nor ReBAR makes host access equivalent to VRAM bandwidth.

A useful first-order comparison, omitting overlap/cache effects, is:

```text
host-direct cost ~ registration + reuse_count * accessed_bytes / host_link_bandwidth
device-resident cost ~ upload_bytes / host_link_bandwidth
                     + reuse_count * accessed_bytes / device_bandwidth
```

Fit these quantities to representative access patterns. Pointer chasing can reduce memory coalescing and increase latency; a resident indexed arena can outperform raw host pointers.

On WASM retain bounded bindings, segment-plus-offset addressing, and asynchronous uploads. CPU worker sharing is not GPU zero-copy. Preserve native-only capabilities outside the remote/browser serialization contract.

### Q3. Cooperative matrices and a common compiler contract

**Unify mathematical operations and numerical guarantees, then specialize kernel schedules.** Do not require every backend to consume identical low-level fragment types or a single WGSL extension dialect.

A matrix operation needs dimensions, layouts/strides, input and accumulator formats, epilogue, quantization scale layout, rounding/overflow policy, tolerance and determinism requirements. A selected kernel additionally records tile shape, subgroup/workgroup scope, shared-memory footprint, register estimates, tail handling, required features, and compiler/driver identity.

| Backend             | Evidence-backed route                                                                                                                                                                                   | Planning decision                                                                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Vulkan              | Existing Etalon enumeration of cooperative-matrix configurations; [KHR properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceCooperativeMatrixPropertiesKHR.html)         | Select an actually advertised tuple and supported compiler lowering. Feature presence alone is insufficient.                                        |
| Metal               | Existing Naga simdgroup lowering and Etalon tuple enumeration; newer [Metal 4 tensor operations](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4) | Preserve the working shader route; separately implement and certify native Forge execution. Evaluate Metal 4 on supported hardware/OS combinations. |
| DX12                | Current Naga HLSL path lacks cooperative matrices; [LinAlg matrix preview, April 2026](https://devblogs.microsoft.com/directx/d3d12-linalg-preview/) combines earlier vector/WaveMMA work under SM 6.10 | Add a capability-gated experimental compiler/executor adapter. Do not specify “SM 6.8+ WaveMatrix” as a universal shipped target.                   |
| Browser WebGPU      | Portable WGSL kernels; [subgroup-matrix proposal](https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroup-matrix.md)                                                                              | Keep portable implementations and feature-detect experimental matrix paths. A proposal is not browser availability.                                 |
| CUDA                | Existing native lane; tuned GEMV/GEMM and graph execution                                                                                                                                               | Select by architecture, shape and numerical policy. Hopper-specific TMA/WGMMA is not an RTX A2000 requirement.                                      |
| GLES/WebGL fallback | Restricted render/compute alternatives                                                                                                                                                                  | Maintain an explicit reduced capability contract; do not promise WebGPU compute parity.                                                             |

For GEMM, use blocked `C = alpha*A*B + beta*C`: device tiles, subgroup tiles and register accumulation, with coalesced input loads and fused epilogues when profitable. Jointly tune tile dimensions, pipeline stages, shared-memory layout and occupancy; increasing tile reuse can lose performance through register pressure. CUTLASS's [hierarchical GEMM formulation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) and Triton's [grouped ordering/autotuning](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) provide implementation references, not a mandate for a CUDA dependency.

Decode GEMV with one or few vectors often has low arithmetic intensity. Benchmark packed, coalesced weight reads and register/subgroup reductions against padded matrix-core GEMM. INT4/INT8 storage requires explicit signedness, scale/zero-point grouping and accumulator range; FP16/BF16 storage should normally accumulate reductions in FP32 unless the declared error budget permits otherwise. FP8 formats and block scales must be distinguished, not represented as one interchangeable `Float8`.

For attention, stream key/value tiles with online softmax statistics:

```text
m_new = max(m_old, max(scores_tile))
rescale = exp(m_old - m_new)
l_new = rescale*l_old + sum(exp(scores_tile - m_new))
o_new = rescale*o_old + exp(scores_tile - m_new) * V_tile
output = o_final / l_final
```

Define all-masked rows explicitly to avoid an undefined `-inf - -inf` path. Use FP32 statistics, a fixed reduction tree where repeatability is required, and mask sequence tails. RMSNorm uses FP32 sum-of-squares followed by reciprocal square root; SwiGLU can fuse its elementwise epilogue where this avoids intermediate traffic without unacceptable occupancy loss. [FlashAttention-3](https://arxiv.org/abs/2407.08608) specifically exploits Hopper asynchrony; adapt the IO-aware algorithm rather than assuming those instructions exist everywhere.

KV paging is a separate memory-management choice. [PagedAttention](https://arxiv.org/abs/2309.06180) informs bounded page pools and sharing; completion-aware reuse and explicit exhaustion behavior remain necessary. Select page size using measured fragmentation, table traffic, context lengths and workload concurrency.

The mandate's “Naga is ~19% slower than DXC” and the audit's “4 versus 83 tokens/s” are hypotheses until reproduced with matching kernels, precision, workload, hardware, clocks and timing boundaries. Compiler identity alone does not explain a throughput difference.

### Q4. Browser workers and shared rings

**A dedicated worker should own the decode device and its GPU resources.** When the main thread must execute rendering commands, give its renderer a separate device and exchange compact CPU-side results. There is no portable transfer of a live `GPUDevice` or `GPUBuffer` between those owners. The current [GPUDevice interface](https://gpuweb.github.io/gpuweb/#gpudevice) is worker-exposed but not transferable/serializable.

If inference and rendering need substantial GPU-resident data sharing, use one worker/device for both and transfer an `OffscreenCanvas`; the DOM canvas remains visible on the main page while its GPU context is worker-owned. That is a different ownership option, not main-thread WebGPU rendering. Negotiate it based on the application's presentation requirement.

SharedArrayBuffer is optional for a dedicated worker. Use transferable ArrayBuffer/message transport without shared memory, or a bounded shared ring when cross-origin isolation and shared WASM memory are available. Neither mechanism guarantees copy-free CPU-to-GPU upload. See [HTML worker messaging](https://html.spec.whatwg.org/multipage/workers.html#dom-worker-postmessage) and [WebGPU writeBuffer](https://gpuweb.github.io/gpuweb/#dom-gpuqueue-writebuffer).

Proposed SPSC protocol, separately instantiated in each direction:

1. Allocate a power-of-two slot count `N < 2^31`, fixed payload capacity, schema version and session generation. Separate producer and consumer counters to reduce false sharing.
2. Producer checks unsigned distance `(write-read) mod 2^32 < N`, fills its private slot, then publishes the new write counter with an atomic store. Consumer observes the published counter before reading and advances read only after consuming.
3. Rust uses release/acquire synchronization on shared atomics; JavaScript Atomics supplies sequentially consistent operations. Use one documented cross-language byte layout; ordinary JS reads of the counters are insufficient.
4. Never overwrite an unconsumed token/control message. Return backpressure or await space. A separate latest-frame channel may drop obsolete frames with an explicit acknowledgement/generation protocol.
5. Use event-loop-friendly notification or `Atomics.waitAsync` where supported. Do not block the worker responsible for `mapAsync` completions, or spin on the UI thread. Recheck the ring condition after arranging a wakeup to avoid lost notifications.
6. Carry cancellation, device-lost and restart generations. Publish failure and release host-side slots even when GPU work fails; retire GPU resources according to backend completion rules.

Keep weights and KV resident, batch bounded work, and overlap only independent uploads/readbacks. Autoregressive dependency prevents arbitrary overlap between successive tokens; GPU-side sampling or bounded multi-token submission requires its own cancellation/latency contract.

### Cross-domain algorithms and scheduling

Graph sorting needs stable digit placement: per-workgroup histograms, device-wide prefix sums, and input-order local ranks. A portable multi-dispatch prefix implementation avoids assuming cross-workgroup forward progress. Native backends can evaluate [Onesweep](https://arxiv.org/abs/2206.01784) once memory-order/progress requirements are satisfied. Do not translate a spin-waiting look-back algorithm into WGSL without that proof.

Rendering should first preserve material ordering, depth, culling, picking, temporal history, resize/device-loss behavior and resource budgets. Then optimize persistent binding tables, indirect drawing and temporary-target reuse. The current [transient attachment proposal is marked merged](https://github.com/gpuweb/gpuweb/blob/main/proposals/transient-attachments.md), and [Etalon already defines the usage](C:/github/etalon/etalon-types/src/texture.rs:191). Use it only for attachments whose contents are not needed afterward; temporal history is not transient.

Audio, vision and EMF should consume the same prepared-operation/resource services. Prioritize the existing [mel filter implementation](C:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/audio/mel.rs:32), [kinematics implementation](C:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/physics/kinematics.rs:29), and [EMF field upload/render boundary](C:/github/qualiaDB/crates/qualia-core-db/src/render/gpu/emf_pipeline.rs:237). For future FFT/convolution work, select Stockham/mixed-radix or partitioned overlap-save schedules by transform size and latency budget, with CPU oracles and fixed scratch. These are proposed additions, not audited existing capabilities.

Use measured deadline classes: presentation/audio work has bounded latency; tensor and batch graph work can fill remaining capacity. Multiple hardware queues do not create extra memory bandwidth. Evaluate total frame latency and audio underruns under compute load, rather than maximizing isolated tensor throughput. DX12 [Enhanced Barriers](https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html) distinguish access visibility from execution synchronization; [Work Graphs](https://microsoft.github.io/DirectX-Specs/d3d/WorkGraphs.html) are an optional device-driven lowering, not a replacement for those dependencies.

## 4. Risk Matrix, Edge Traps & Microarchitectural Failure Modes

| Risk                                                          | Evidence / likelihood                      | Impact                                                         | Required mitigation                                                                                                                                                                                          |
| ------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Duplicate wgpu/Naga identities                                | Observed manifest mismatch                 | Integration blocker                                            | Resolve package source/version and feature forwarding; compile representative downstream API crossings.                                                                                                      |
| Radix instability masked by fallback                          | Source-proven allowed failure              | Incorrect accelerator contract, wasted work                    | Stable per-digit algorithm; adversarial keys; report backend actually used.                                                                                                                                  |
| Unfinished lane advertised available                          | Observed Metal code                        | Misrouting/failure                                             | Capability state: detected → compilable → prepared → verified; downgrade on failure.                                                                                                                         |
| mmap released/truncated while GPU reads                       | Applies to proposed import design          | Memory fault or stale data                                     | Owning lease plus immutable snapshot; completion before unmap/reclamation.                                                                                                                                   |
| Physical addressing bypasses bounds guarantees                | Applies to proposed raw pointers           | Safe API unsoundness                                           | Restricted validated kernels or explicit unsafe contract; test offset overflow, aliases and hidden pointer targets.                                                                                          |
| CPU epoch mistaken for GPU completion                         | Applies to proposed SnatchLock replacement | Use-after-free                                                 | Reclamation requires CPU references and every relevant GPU queue completion. Keep existing locks until an alternative proof and tests exist.                                                                 |
| Shared timeline signals retire independent work early         | Multi-queue design risk                    | Races                                                          | Per-queue timelines or proven ordered aggregate completion.                                                                                                                                                  |
| Excessive fusion / graph duration                             | Workload dependent                         | Register spills, poor occupancy, watchdog/reset, UI starvation | Bounded batches, cancellation checkpoints, register/shared-memory and scheduling measurements.                                                                                                               |
| 256-byte alignment generalized to all buffers                 | Present audit framing                      | Wasted bandwidth or validation errors                          | Distinguish texture-copy row pitch, uniform/storage dynamic offsets, map/copy alignment, noncoherent atoms, BDA access alignment and host-import granularity.                                                |
| Browser mappedAtCreation treated as universally broken        | Unreproduced audit claim                   | Disabling a valid API broadly                                  | Preserve a narrow workaround pending a minimal browser/version repro; the [descriptor explicitly permits initialization mapping](https://gpuweb.github.io/gpuweb/#dom-gpubufferdescriptor-mappedatcreation). |
| Ring reuse or lost notifications                              | Worker design risk                         | Corruption or deadlock                                         | Sequence/generation tests, publication ordering, bounded backpressure, restart tests and asynchronous waits.                                                                                                 |
| Budget fragmentation                                          | Sustained mixed workloads                  | Allocation failures despite free bytes                         | Size classes/arenas, bounded staging and KV pools, retirement-aware reuse, measured peak and fragmentation.                                                                                                  |
| Thermal/power-state bias                                      | Benchmark risk                             | Misleading performance selection                               | Warmup and sustained runs; record clocks/power/temperature; label throttled results; avoid unsafe clock manipulation.                                                                                        |
| Shader signature mistaken for confinement                     | Blueprint policy design                    | False extension security guarantees                            | Authenticate provenance separately from runtime permissions. An AST cannot know every host copy, alias or external API access.                                                                               |
| “Zeroed on Drop” treated as an enclave                        | Blueprint design risk                      | Incomplete cleanup/isolation claims                            | Explicit GPU scrub followed by completion before reuse; define device-loss behavior. Ordinary allocations are not hardware confidential-computing enclaves.                                                  |
| New license metadata conflicts with existing package metadata | `Cargo.toml:77` versus plan 03             | Distribution ambiguity                                         | Preserve per-file provenance and produce an ownership/license manifest before release; this review makes no legal determination or license change.                                                           |

For single-row texture copies, do not require an explicit row pitch where the API permits omission. When a pitch is provided to the command encoder's buffer/texture copy, obey its alignment. A four-byte pixel and a 256-byte pitched staging row describe different quantities.

The rights architecture can support signed manifests and submission authorization without network/payment checks in per-dispatch hot paths. Bind authorization to code hash, resources, session and expiry, with explicit revocation behavior. This implements the application's declared policy boundary; it does not make a local process impossible to modify or provide legal classification.

## 5. Precision Execution Roadmap & Milestone Acceptance Criteria

### M0 — Correct the evidence and dependency baseline

Update plans 01/02/04 with implemented/partial/proposed classifications and exact source symbols. Refresh TODO paths against the current tree: the index contains old `wgpu-*` links and is a backlog inventory, not proof of completeness. Record the actual upstream base used for backports; preserve upstream identifiers in provenance and issue links rather than globally replacing them.

Select either explicit consumer aliases to Etalon or a version-compatible `wgpu` facade with complete feature/API verification. A facade alone cannot emulate incompatible API changes. Plan the Naga dependency separately. Cargo patches are version-sensitive and must be applied from the consuming workspace root; see the [Cargo dependency override rules](https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html).

**Acceptance:** a recorded dependency graph for representative QualiaDB, renderer and browser consumers; no unintended second GPU API type universe at public boundaries; no unsupported optional SDK required for the browser build. Replace the unmeasured “60% faster compilation” claim with measured cold/incremental build results.

Read-only diagnostic commands for that phase:

```text
cargo metadata --locked --format-version 1
cargo tree -d
cargo tree -e features -p etalon
```

Run metadata/tree in each relevant workspace and inspect actual package IDs. These commands are proposed gates, not results claimed by this review.

### M1 — Define shared contracts and collect representative workloads

Specify resource views/leases, byte layouts, device identity, capability states, graph hazards, bounded scratch, error/cancellation semantics and numerical modes. Collect workloads covering graph scan/sort/join, rendering with temporal/picking, mel/vision/field operations, and GEMV/GEMM/attention.

**Acceptance:** byte fixtures for NQuin/P64; all negative range/alignment/overflow cases fail before submission; exact backend and fallback reasons appear in results; no semantic/database dependency from foundational runtime crates. Record hardware, driver, OS, compiler builds, revisions and workload hashes.

### M2 — Extract Forge and one end-to-end portable slice

Extract pure IR/scheduling/manifest interfaces, WGSL emission/validation and an injected execution context. Move the equality sieve with its shader, CPU oracle and immediate QualiaDB adapter. Keep a resident input option and caller-buffered output. Separately correct radix stability before promoting sorting.

**Acceptance:** exact equality of result sets and sort order/permutation for empty, singleton, duplicate, reverse, high-bit and mixed-digit keys, plus workgroup-boundary sizes. No per-digit host histogram transfer in the prepared radix path. GPU-unavailable skip counts cannot satisfy a hardware acceptance gate.

Cache acceptance includes rejection on compiler options/ABI changes, bounded storage and lifecycle cleanup. Recompile failures must not leave a cache entry marked certified.

### M3 — Extract rendering, vision and signal-processing boundaries

Move generic resource pools, mesh/instance upload, culling, temporal producers/resolve, picking and EMF field views in small consumer-linked changes. Retain scene semantics and policy adapters in their owning application. Bring one mel and one vision operation through the same runtime to test that the abstraction is genuinely reusable.

**Acceptance:** golden-image tests with defined tolerances, exact picking IDs, transparent draw order, resize/device-loss recovery, bounded residency and concurrent compute/frame tests. Define a 60 Hz profile with total p99 frame time below 16.67 ms on a named reference scene/device; a 120 Hz profile needs its own 8.33 ms gate. These are proposed product budgets, not achieved measurements.

Audio acceptance records the configured callback quantum and sample rate; callback execution must fit `quantum/sample_rate`, with no blocking GPU wait in the callback and no underruns during the named stress workload.

### M4 — Add prepared tensor execution across capability tiers

Extract tensor views and scheduling independently of GGUF/P64 parser policy. Preserve the portable path; certify Vulkan and Metal matrix configurations; keep CUDA and DirectML optional; implement native Metal only with a truthful unavailable state until passing tests. Track DX12 LinAlg preview separately from stable support.

**Acceptance:** per-operation numerical budgets against CPU/reference implementations, with NaN/Inf, zero dimensions, tail shapes, masks and quantization extremes specified. Verify models across representative sequence lengths; token agreement alone is insufficient evidence for all floating-point paths.

Measure cold preparation, warm submission, transfer, kernel and end-to-end times. Use roofline bounds `P <= min(P_peak, bandwidth * arithmetic_intensity)` with actual bytes, including scale metadata and paging. A proposed optimization should show a reproducible benefit outside measurement noise on its named workload; reject regressions in the agreed latency/memory budget even if peak FLOPS rises.

### M5 — Introduce native memory and multi-queue optimizations selectively

Implement import/interop/BDA as separate capabilities after the ownership contract exists. Track producer/consumer access in the graph and lower precise barriers. Preserve a single-queue/staged mode on devices where overlap or direct host access loses performance.

For each shared native feature, the implementation plan must identify every affected boundary:

| Boundary                                            | Required work                                                                                                                     |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `etalon-types` and public `etalon` API              | Capability/usage descriptors, ownership-facing API, documented safety and unsupported behavior                                    |
| `naga-types`, Naga IR/frontends/validation/backends | Only the shader representations actually required; legality, uniformity, bounds and target lowering                               |
| `etalon-naga-bridge`                                | Feature-to-validator capability mapping consistent with device enablement                                                         |
| `etalon-core`                                       | Safe-boundary validation, initialization, lifetime/resource tracking, completion and error propagation                            |
| `etalon-hal` backends                               | Native discovery, allocation, dispatch and synchronization satisfying the HAL contracts                                           |
| `etalon-core-remote` and remote types               | Explicit exclusion or a separately reviewed protocol change; never serialize native pointers as browser/content-process authority |
| Tests and consumers                                 | Positive/negative shader and API cases, backend execution, browser rejection/fallback, consumer integration                       |

Changes specific to Deno belong in its bindings; shared behavior belongs in the supplying shared crate. The inspected workspace names that member `deno_etalon`, so reconcile stale `deno_webgpu` references when preparing concrete commands.

**Acceptance:** destruction-while-in-flight, cross-device rejection, export/import lifetime, failed submission, device loss, queue overlap and aliasing tests; validation-layer checks where supported. Show measured benefit against the same prepared workload with reusable bindings before attributing gains to BDA.

Zero-allocation gates require exactly zero allocations in the specifically declared warmed Etalon-owned execution region. Bound cold allocation and exclude only clearly documented external driver/browser allocations; do not silently replace zero with a threshold.

### M6 — Browser workers, integration and release readiness

Implement both transferable-message and isolated shared-ring transports. Exercise worker restart, cancellation, backpressure, background-tab behavior, browser device loss and main-thread responsiveness. Use OffscreenCanvas only for the chosen ownership mode.

**Acceptance:** no synchronous polling path for browser GPU completion; no lost/duplicated control messages across wraparound/restart tests; bounded outstanding GPU work; UI and rendering meet the selected workload budget. Validate the final consumer graph and remove old implementations only after equivalent acceptance passes.

### Existing repository checks to apply during implementation

Use the relevant platform feature subsets; `--all-features` alone is not a browser or cross-platform test.

```text
cargo check -p etalon -p etalon-core -p etalon-hal -p etalon-types
cargo check -p etalon --target wasm32-unknown-unknown --no-default-features --features std,webgpu,wgsl
cargo nextest run -p naga
cargo nextest run -p etalon-core
cargo xtask test -E 'binary(etalon-gpu) & test(cooperative_matrix)'
```

Confirm test discovery and required features before treating the last filter as coverage. Add migrated crate tests only once those crates exist. Run Metal checks on an Apple runner and Windows/Linux backend checks on matching runners. After GPU behavior changes, run the full `cargo xtask test` as required by repository instructions, plus relevant formatting/clippy checks. Report every failure and hardware skip. CTS runs and tests at `trunk` require the separate authorization specified in `AGENTS.md`.

Every milestone ends with a consumer-linked, independently revertible change and its evidence. Do not combine renaming, ABI changes, numerical changes and allocator replacement in one migration step.

## 6. Traceable Sign-Off & Attestation

**Reviewer/model:** Codex, OpenAI; runtime identifies the agent as GPT-6 based. No more specific model identifier is attested.  
**Confidence:** 8/10 for the source-grounded architecture findings; hardware performance and complete cross-platform readiness remain unverified.

**Performed:** planning/source inspection; manifest comparison; read-only Git identification; primary-source research; an executable host simulation demonstrating the LSD atomic-scatter counterexample.

**Not performed:** implementation changes, Cargo compilation/test suites, GPU benchmarks, CTS, baseline tests at trunk, or exhaustive shader certification. No performance number in this review is presented as measured on the user's hardware.

**Research limitations:** the repository specification-download script was attempted, but this environment does not expose `sh`/`jq` through the inspected process PATH. A direct download attempt also failed. WebGPU was instead inspected through its authoritative raw specification and WGSL through the W3C publication; the large rendered WebGPU page exceeded the browsing tool's size limit. Vendor previews were treated as previews, not assumed deployed features.

**Assumptions to carry into implementation:** safe Etalon API guarantees remain mandatory; native and browser capabilities differ; hardware/compiler availability must be probed; local working-tree changes are part of the reviewed snapshot; application semantics and rights decisions retain explicit owners.

This document is a proposed technical review, not a cryptographic attestation, implementation sign-off, or benchmark certification.
