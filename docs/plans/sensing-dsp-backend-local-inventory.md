# DSP sensing backend: local inventory and implementation boundary

The sensing code provides CPU signal-processing primitives and a live-provider
admission contract. Neither is a DSP hardware backend. The accelerator planner
and a host's classification of a device also do not establish executable DSP
support.

The local Windows system directory contains `OpenCL.dll`, an OpenCL loader.
The presence of a loader does not establish that an installable client driver,
an OpenCL accelerator device, or a DSP device is available. An OpenCL
accelerator device is not necessarily a DSP. No device enumeration, library
loading, hardware access, build, or test was performed during this inventory.

No Qualcomm Hexagon/QNN or Texas Instruments DSP development SDK was found in
the inspected standard installation paths, and no OpenCL Rust binding was
found in the local Cargo registry. A bounded filename survey did not find
OpenCL headers in the installed Windows SDK include trees or the portion of
the local repositories inspected. These are observations about the searched
locations, not proof that no SDK exists elsewhere on the machine.

To add an actual backend, identify a target DSP device and its supported,
locally supplied vendor SDK, runtime, redistribution terms and authoritative
ABI. Then implement device discovery and lifecycle, operation preparation,
bounded dispatch and completion, device-loss/error handling, and explicit
output provenance. Admit a device only when an executable provider for the
requested operation exists. Verify numerical behavior against the CPU oracle,
allocation and buffer bounds, resource cleanup, and hardware execution once
the user authorizes testing. Where a generic OpenCL implementation is useful,
label it as an OpenCL accelerator provider; do not infer DSP hardware from the
OpenCL accelerator class.

## Candidate platforms from the local survey

- Qualcomm QNN HTP: the locally cached `ort 2.0.0-rc.12` bindings include QNN provider configuration (`C:/Users/tim_h/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ort-2.0.0-rc.12/src/ep/qnn.rs`). The isolated `etalon-sense-qnn` source requests an explicit HTP backend and disables ORT CPU graph fallback. It requires a separately supplied compatible ORT/QNN runtime, target driver and supported ARM64 device; the bindings alone are not that runtime. HTP model inference is not a general-purpose DSP kernel implementation. Provider-internal execution, memory and connectivity remain host-controlled and unqualified.
- Qualcomm Hexagon and Texas Instruments DSP devices are vendor-specific candidates where an appropriate local development SDK, driver and compatible hardware exist. Their availability and supported operation interfaces have not been established by this survey; no implementation is claimed.
- OpenCL could provide a generic accelerator interface if a compatible local SDK, ICD and device exist. Neither the loader nor the accelerator classification proves DSP support.

The QNN implementation's facade feature is optional, but the Etalon workspace includes `crates/*` as default members, so optional execution does not exclude that crate from default workspace dependency resolution or builds. Dependency integration and qualification remain pending.

## Local follow-up survey

The cached ONNX Runtime Rust QNN provider source names `QnnHtp.dll` and `libQnnHtp.so` as accelerated backend libraries. This supports QNN HTP model execution as a candidate, not arbitrary DSP kernels. The isolated `etalon-sense-qnn` draft is the existing implementation boundary for that candidate; neither file presence nor execution was established.

For a general Qualcomm DSP-kernel target, the Qualcomm Hexagon SDK and its target-compatible compiler, RPC/driver interface and device runtime are the most directly relevant candidate to investigate once made available locally. For a TI target, the appropriate TI DSP SDK must match the specific SoC and toolchain; neither target has been selected or qualified. Version, licensing, host/target support and runtime ABI cannot be confirmed from the inspected files. The local QNN Rust API alone cannot fill those gaps.

A follow-up filename survey of `C:/Program Files`, `C:/Program Files (x86)`, Windows SDK include trees, the local Cargo registry and local GitHub trees found no Qualcomm QNN/Hexagon DSP headers, compiler or `QnnHtp.dll`, and no TI C7x header in the searched paths. The Qualcomm directory inspected contains Bluetooth/WLAN components; it is not evidence of a DSP SDK. Searches are limited to those paths and do not prove absence elsewhere. No devices, dynamic libraries, builds, tests, or network were accessed.
