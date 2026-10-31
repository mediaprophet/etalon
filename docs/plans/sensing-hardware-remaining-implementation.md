# Remaining sensing hardware implementation boundary

Source assessment only. No runtime, device, build, test, dependency resolution
or network operation was performed.

## Implemented GPU construction link

`etalon-sense-gpu::DeviceQueuePair::request` retains the device and queue from
one safe Etalon request and runs host policy callbacks before and after it.
The fields are private and the pair is not constructible from unrelated handles.
Its provider factory borrows this exact pair. The older direct constructor is
still available and requires the caller to supply a correctly paired device
and queue, as documented by Etalon's own API.

The owner registers one device-loss callback with a fixed atomic state.
Providers created from the owner reject reported loss before dispatch and
after readback; live dispatch reports them faulted. This does not attest a
physical device, bound driver memory, guarantee cancellation or secure erasure,
or make policy revocation atomic with asynchronous device setup. The path stays
cold and unqualified. Device-request feature/limit and single-request panic
preconditions remain those of the existing safe Etalon API.

## External SDK and shared API dependencies

- General DSP kernels still require a selected device and locally supplied
  authoritative vendor SDK/toolchain/runtime. The inspected local paths did
  not establish such a Hexagon or TI installation. Cached ORT QNN bindings
  supply HTP tensor inference configuration, not a general DSP SDK.
- Native SDR acquisition still requires a locally available supported driver
  interface. Prior source surveys did not establish RTL-SDR, SoapySDR, UHD,
  Airspy or HackRF headers/bindings/runtime. The existing host reader boundary
  does not implement those drivers.
- GPU hot zero-heap execution needs coordinated safe Etalon callback,
  command-recording and submission APIs. Keeping a device/queue pair together
  and using a fixed completion state does not remove those allocations.
- OpenVINO NPU and QNN HTP drafts in the independent vendor workspace need
  compatible local ORT/provider libraries, drivers and supported targets.
  Their internal allocation and network behavior require host isolation.

No new vendor ABI was guessed and no binary was loaded to infer availability.
Actual driver/device behavior, failure cleanup, numeric precision, allocations,
policy revocation and device-loss handling need deferred qualification.
