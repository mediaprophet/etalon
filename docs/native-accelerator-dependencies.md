# Native accelerator dependency provenance

The modernization review pins the dependencies below separately from device
discovery and successful workload execution. A Cargo license expression describes
the Rust package; it does not inventory every native library that package loads
or statically links. Binary distributors must retain the applicable upstream
licenses, copyright notices and third-party notices for the exact artifacts shipped.

## Static DXC

The optional `static-dxc` feature uses `mach-dxcompiler-rs`
`0.1.7+2026.10.26-141dd7f.1`, pinned as `=0.1.7`. Its Rust wrapper is MIT licensed.
The published build script checks MSVC 19.41 or newer, verifies that its native
release is immutable, and verifies the downloaded archive's SHA-256 before linking.
The Windows 2022 hosted image reviewed on 2026-10-30 contains Visual Studio 17.14;
Windows linking remains an acceptance gate, not an inference from that inventory.

The native release is
[`2026.10.26+141dd7f.1`](https://github.com/DouglasDwyer/mach-dxcompiler/releases/tag/2026.10.26%2B141dd7f.1).
Its API reported `immutable: true` during review. The published crate's
[`targets.rs`](https://docs.rs/crate/mach-dxcompiler-rs/0.1.7+2026.10.26-141dd7f.1/source/targets.rs)
contains these expected Windows archive hashes; they match the release metadata.
They are expected artifact hashes, not a claim that every archive has been built
or executed locally.

| Native archive                                                     | SHA-256                                                            |
| ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `x86_64-windows-msvc_ReleaseFast_lib.tar.gz` (static CRT)          | `908305e46a7f100b240f29235b0844006ced9c0cf22c95d019e3d59c14346725` |
| `x86_64-windows-msvc_ReleaseFast_Dynamic_lib.tar.gz` (dynamic CRT) | `abe009a4abb88dab9a1659ca12515e35baa34478a2ed42103595ebf65f8f6b25` |

Native DXC contains additional licensed code. Microsoft's
[`LICENSE.TXT`](https://github.com/microsoft/DirectXShaderCompiler/blob/main/LICENSE.TXT)
includes legacy University of Illinois/NCSA LLVM terms, Microsoft MIT terms and
component notices. Do not describe the linked compiler solely as the wrapper's
MIT package or assume modern LLVM's license describes this DXC fork. Preserve
the notices corresponding to the selected native release in binary packages;
this dependency review does not replace an artifact-level notice inventory.

The retained shader passthrough tests also require the standalone Microsoft DXC
executable. CI uses release
[`v1.9.2602.24`](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.9.2602.24),
asset `dxc_2026_05_27.zip`, whose published and locally verified SHA-256 is
`cf658aacf070d3045e31b8f1f8a696c2945f37c1095019481ef7c513368db3b4`.
The install action exposes its `dxc.exe` on PATH and retains the matching
`dxcompiler.dll` and `dxil.dll`. This executable is separate from the statically
linked compiler and its version must not be inferred from the Rust wrapper.

## ONNX Runtime and OpenVINO

QualiaDB's optional vision integrations use the Rust wrapper `ort =2.0.0-rc.15`
(MIT OR Apache-2.0) and its matching native API. This version's upstream
[release notes](https://github.com/pykeio/ort/releases/tag/v2.0.0-rc.15)
identify memory-safety fixes. Sessions now receive an explicit environment;
the session retains an environment handle after the constructor returns. Qualia's
default vision loaders share one environment owner across both crates because
ORT permits one environment per process. Additive explicit-environment loader
variants support applications that already own that environment; a conflicting
default initialization returns an error instead of creating a second owner.

Native ONNX Runtime 1.31 uses its own
[MIT license](https://github.com/microsoft/onnxruntime/blob/v1.31.0/LICENSE) and
[third-party notice inventory](https://github.com/microsoft/onnxruntime/blob/v1.31.0/ThirdPartyNotices.txt).
QualiaDB's vision feature may obtain native binaries through ORT's configured
download mechanism. A release bundling those binaries must include their
matching notices, rather than relying only on Rust metadata.

The separate accelerator harness enables ORT's `std`, `load-dynamic` and `api-31`
features with default features disabled. It neither downloads nor bundles a
runtime, provider, driver or model weights. Operators explicitly select installed
libraries with `ORT_DYLIB_PATH` and `QUALIA_OPENVINO_EP_PATH`.

The harness uses the current
[OpenVINO plugin API](https://github.com/onnxruntime/onnxruntime-ep-openvino/blob/main/docs/QuickStart.md),
not the deprecated built-in EP. The plugin is
[MIT licensed](https://github.com/onnxruntime/onnxruntime-ep-openvino/blob/main/LICENSE);
the OpenVINO toolkit has a separate
[Apache-2.0 license](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE)
and additional component notices. Their transitive libraries and device drivers
must be reviewed separately if a future installer redistributes them. Explicit
local loading does not certify those installations' provenance or compatibility.

The NPU arithmetic model is generated locally from the documented
[ONNX protobuf format](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto):
IR 8, opset 13, an Add node and two fixed-shape inputs. It contains no downloaded
model or pretrained weights. Runtime NPU device selection, disabled CPU fallback,
correct output and attributed execution profiling are all required before this
workload can be reported as executed on the NPU. Device discovery or library
initialization alone is insufficient.
