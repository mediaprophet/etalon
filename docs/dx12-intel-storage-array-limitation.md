# DX12 non-uniform storage texture arrays on one Intel driver

Etalon does not advertise `STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING` for
vendor `0x8086`, device `0x5917` (the tested Intel UHD Graphics 620), driver
`31.0.101.2141`. Uniform storage texture arrays, partially bound arrays,
sampled texture arrays and storage buffer indexing remain available according
to their existing capability checks. The restriction does not apply to Vulkan,
other devices or other driver versions.

This is a capability correction for a reproduced execution failure, not an
expected-failure annotation or a claim that Intel has published a matching bug.
Safe WGSL validation rejects non-uniform storage texture indexing when the
device has not enabled its capability. Affected non-uniform GPU tests report
unsupported; they do not establish executed coverage. Dedicated uniform-array
and capability-rejection tests still run on the affected adapter.

The failing oracle uploads sixteen distinct one-pixel RGBA8 inputs, selects an
input per invocation and writes its color into a 4-by-4 output. On the exact
adapter/driver above:

| Probe                                                         | Result                    |
| ------------------------------------------------------------- | ------------------------- |
| CPU comparison of every uploaded input                        | Exact match               |
| Constant write through storage texture descriptor array       | Exact match               |
| Uniform read through the same descriptor array                | Exact match               |
| Varying read marked `NonUniformResourceIndex`                 | All sixteen pixels zero   |
| Raw HLSL passthrough, bypassing Naga                          | Same varying-read failure |
| Shader Model 6.0 and 6.5                                      | Same failure              |
| `unorm float4` and `float4` resources                         | Same failure              |
| Official DXC-generated DXIL with optimizations disabled       | Same failure              |
| Constant-index switch selecting the same per-invocation input | Exact match               |

The failure reproduces with both the system D3D12 runtime and Agility SDK 619,
with official dynamic DXC and with debug/GBV disabled. DXIL disassembly retains
the non-uniform resource-handle flag. Upload synchronization therefore does not
explain the varying-load failure. WARP passes the expanded storage-array CPU
oracles, including changed input bytes uploaded immediately before dispatch;
the original Vulkan non-uniform storage-array tests also execute successfully.

Microsoft's [HLSL resource binding contract](https://learn.microsoft.com/en-us/windows/win32/direct3d12/resource-binding-in-hlsl)
requires `NonUniformResourceIndex` for indices varying within a wave.
The generated shader supplies it. Microsoft's [hardware tier limits](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-support)
allow full-heap UAV tables on Tier 3. These contracts support identifying this
as an advertised operation that fails on the observed configuration; they are
not an official diagnosis of Intel's driver implementation.

The standalone switch experiment is not a general runtime fallback. A correct
fallback would need bounded shader generation covering all storage image
dimensions, formats, access paths and layout sizes. That implementation is not
provided by this change. No OS driver is installed or updated automatically.
