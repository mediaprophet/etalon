# Sensing GPU Completion and Polling Proposal

Historical proposal: the private native Vulkan prepared-execution implementation
is now described in [Prepared native compute](../prepared-compute.md). The
ordinary callback/encoding APIs discussed below still allocate; target allocation
qualification and consuming-provider admission remain separate requirements.

Status: proposal only. No Etalon API or backend changes are included here. The proposal targets bounded caller-owned readback completion for sensing providers and separates application-level improvements from work required inside Etalon.

## Current API boundary

Etalon's public `BufferSlice::map_async` accepts a `'static` callback, marks its `MapContext` as mapped when the request is issued, and boxes the callback before dispatch (`etalon/src/api/buffer.rs:634-656`). That `mapped` bit is not a completion signal. A mapped view can be acquired only after the callback reports success; every view must be dropped before `unmap` or `destroy` (`etalon/src/api/buffer.rs:331-337, 678-700`). While mapped, the buffer is unavailable to GPU commands.

The native backend boxes a second closure around the application callback (`etalon/src/backend/etalon_core.rs:1839-1854`). WebGPU maps through a JavaScript promise and registers success and rejection closures held through `Rc<RefCell<...>>` (`etalon/src/backend/webgpu.rs:1086-1122, 3150-3164`). These backend callback lifecycles are distinct from an application-created `mpsc` channel.

`Device::poll(PollType::Wait { submission_index, timeout })` can block natively until the submission and its callbacks finish. `PollType::Poll` is nonblocking. Etalon documents that WebGPU `Poll` is a no-op and `Wait` always returns `CantWait`; callbacks run from the browser event loop (`etalon-types/src/lib.rs:284-334`, `etalon/src/api/device.rs:96-110`). `Queue::submit` returns `Some(SubmissionIndex)` for waitable native work and `None` on WebGPU (`etalon/src/api/queue.rs:279-300`). `Queue::on_submitted_work_done` accepts a boxed `'static` callback and is ordered after map callbacks registered for the same submission (`etalon/src/api/queue.rs:300-329`, `etalon/src/api/command_buffer_actions.rs:113-143`).

Do not use `map_buffer_on_submit` as a zero-allocation shortcut. Etalon stores deferred mappings and submitted callbacks in `Vec`s and boxes each callback (`etalon/src/api/command_buffer_actions.rs:11-29, 103-143`). Its normal command recording and submission path also lacks a caller-owned recording arena. A completion redesign alone cannot establish a zero-heap GPU hot path.

Etalon exposes no public, reliable comparison between a `Device` and a `Queue`. The safe construction path returns them together from device request, while `Device::poll` warns that other threads may submit to a shared queue during a poll. A sensing owner must preserve the pair's construction relationship and coordinate submissions itself; a caller-provided numeric device ID does not prove queue ownership.

## Application-level bounded completion

The first stage removes avoidable channel creation from the sensing provider while keeping today's Etalon API:

1. Allocate a fixed number of completion slots when the provider is constructed. Each slot owns a mutex-protected state, a checked `u64` generation, and a terminal status. Allocate each slot's `Arc` once; cloning it into the required Etalon callback does not allocate a new completion cell.
2. Permit at most one outstanding map per readback buffer. Start a map by incrementing the slot generation and clearing its prior result. A callback captures the slot and generation and does only one bounded action: publish `Mapped` or `MapFailed` if the generation still matches. It must not poll, map a view, unmap, run user code, or take a lock held by the polling caller.
3. Do not hold the slot lock during `Device::poll`. After a native bounded `Wait`, inspect the slot. Read only after `Mapped`; otherwise invalidate the generation, unmap, poison that provider instance, and discard it. `Poll` may be used by an externally driven event loop, but a single poll result is not proof that a requested map callback ran.
4. After copying the readback bytes into caller storage, drop every `BufferView`, then unmap. Never submit the staging buffer while mapped. Keep readback and staging allocations inside a caller-selected capacity and byte budget.
5. Register device-loss handling once at provider setup. Device loss invalidates all outstanding generations and makes the provider terminal. The lost-device callback must only publish a small state change; formatting or recording its `String` reason is cold-path work.

The generation is never allowed to wrap. If increment overflows, retire that slot. Cancellation marks the generation invalid before calling `Buffer::unmap`; this ensures any callback that arrives after cancellation cannot complete the slot. Etalon documents that unmap terminates the effect of prior map requests, but does not promise callback destruction or cancellation of already submitted GPU work. Therefore a timeout cancels readback use, not GPU execution. Do not recycle a timed-out provider or its buffer until the API's map lifecycle is known to be settled; otherwise discard the provider and preserve the buffer's ownership until drop/device teardown.

This stage removes the provider's per-call `mpsc` machinery, but it does not meet strict `HotZeroHeap`: Etalon boxes the public callback and its native backend adds another box. The command encoder, finished command buffer, queue submission, backend synchronization and possible error formatting also remain outside this slot's allocation control.

## Etalon API direction

An eventual caller-owned completion API should avoid accepting arbitrary borrowed callback state with a `'static` bound. It needs an explicit, capacity-bounded registration established before hot execution, for example a device-owned completion table with caller-provided slot storage and tokens of `(slot_index, generation)`. Submission and map requests would write only to a previously registered slot; polling would expose `Pending`, `Mapped`, `MapFailed`, `Cancelled`, and `DeviceLost` states. Tokens are invalid after cancellation, device loss, generation overflow, or buffer destruction.

The implementation must decide where completion storage lives. A table owned by the Etalon `Device` can centralize native poll and device-loss handling, but requires a construction-time capacity and synchronization contract. A table supplied by the caller avoids hidden ownership but must remain pinned/alive until every request and callback is terminal. In either design, the native `etalon-core` mapping API currently consumes a callback object (`etalon/src/backend/etalon_core.rs:1839-1854`); removing per-map boxes may require an upstream core/backend change, not only a wrapper change.

WebGPU cannot implement synchronous completion through `Device::poll`. Its provider needs a separate `async`/future or event-loop callback surface, with cancellation represented as generation invalidation and `unmap`. JavaScript promise and closure registration currently allocates (`etalon/src/backend/webgpu.rs:1086-1122, 3150-3164`); a WebGPU implementation must not claim strict zero-heap unless those browser/runtime allocations are measured and the target contract explicitly permits them. It may still offer bounded caller-owned sample buffers and bounded outstanding operation count.

Provide a `DeviceQueuePair` owner created only from Etalon's paired device request result, or an equivalent construction token, to prevent accidental cross-device submission. This wrapper proves only that the caller kept the returned pair together; it does not attest the physical adapter or qualify its behavior. Coordinate access to the pair if other threads can submit, so a completion token refers to the exact intended submission.

## Lifecycle and safety states

| State                   | Allowed action                                        | Terminal/transition rule                                       |
| ----------------------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| `Idle`                  | Begin one map with a new generation                   | Capacity and byte budget checked before submission             |
| `Submitted`             | Poll or await; no readback view                       | A timeout or loss invalidates the generation                   |
| `Mapping`               | Callback may publish status only                      | No buffer reuse until callback/poll state is resolved          |
| `Mapped`                | Acquire one bounded view and copy to caller buffer    | Drop view before unmap                                         |
| `Cancelled`             | No further read; unmap and discard/retire as required | Late callback is ignored by generation                         |
| `Failed` / `DeviceLost` | Return failure; no sample output is trusted           | Provider is poisoned; create new resources through host policy |

Do not retain sample bytes in internal completion slots. Slots carry only request generation and status. The GPU input/output buffers and map buffer still retain sample-derived data until overwritten or destroyed; neither `unmap` nor dropping the provider is a secure-erasure guarantee.

## Verification and rollout

1. **Provider-only channel removal:** test all slot transitions, stale callbacks, callback failure, timeout, generation overflow, output unchanged on failure, view drop before unmap, and poison behavior with a fake completion source. Source tests do not prove backend allocation behavior.
2. **Etalon native completion API:** add bounded slot registration and polling tests at the safe API boundary, including submit-index ownership, wrong-device/index failure, repeated mapping, cancellation races, device loss, and `BufferView` lifetime checks. Keep core/backend validation authoritative.
3. **Allocator evidence:** measure a warmed operation using the repository's real allocation instrumentation around command recording, submission, polling, callback execution, mapping, and copy. Report native callback-thread allocations and cold setup separately. If any backend allocation remains, the manifest must remain `ColdBounded`; do not claim `HotZeroHeap` from absence of `Vec` in the sensing crate.
4. **WebGPU async path:** test callback and promise completion on the browser event loop, cancellation/unmap with late resolution, device loss, bounded in-flight operations, and output lifetime. Measure JS/WASM-side allocations with target instrumentation; native tests cannot establish this.
5. **Qualification:** test only on declared target/driver/device combinations after API correctness. Record device, backend, driver, limits, timeout, precision, and failure results. Successful software tests or an `Etalon` device ID do not constitute hardware qualification or attestation.

The existing native sensing provider can use Stage 1 as an allocation reduction and correctness improvement. Strict zero-heap GPU sensing remains blocked on Etalon command/submission and map callback/backend APIs, and a synchronous WebGPU path is unavailable by design.
