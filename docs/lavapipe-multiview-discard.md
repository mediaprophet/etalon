# Lavapipe multiview discard and sparse resolve workarounds

Linux GPU CI on Mesa 26.1.3 crashed in the sparse-view, base-layer, multisample
resolve regression with `LVP_POISON_MEMORY=true`. The driver resolves the color
attachment, then poisons discarded attachments in `handle_end_rendering`.
Its inner `u_foreach_bit(i, viewmask)` shadows the outer color attachment index
`i`, so view 3 selects `color_att[3]` instead of attachment 0. This can dereference
an absent attachment or corrupt another attachment's stored contents.

The defect is present in the official [Mesa 26.1.3 source](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.1.3/src/gallium/frontends/lavapipe/lvp_execute.c#L1977),
Mesa 26.1.5 and upstream main inspected on 2026-10-30. The
[loop macro](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.1.3/src/util/bitscan.h#L109)
declares a new index variable. Updating the available CI binary from 26.1.3 to
26.1.5 would retain the defect.

Etalon promotes discarded color attachments to Vulkan `STORE` for any multiview
render pass on `VK_DRIVER_ID_MESA_LLVMPIPE`. This also covers a single selected
view with multiple color attachments, where the incorrect attachment index can
destroy a different attachment's required contents. Other drivers, ordinary
render passes, load operations and requested stores retain their existing mapping.
Core initialization tracking still treats the public attachment as discarded.

The [Vulkan store-operation specification](https://docs.vulkan.org/refpages/latest/refpages/source/VkAttachmentStoreOp.html)
allows discarded contents to be undefined; preserving physical contents does not
promise them to the caller. The workaround retains rendering, resolving and all
layer readback assertions in the regression. It does not disable memory poisoning
for other tests or change the separate Intel resolve workaround.

The next Linux CI run confirmed the crash was fixed: 1624 of 1625 tests passed.
The same regression then exposed an inactive-layer readback error, expected zero
but observed 128. Mesa's [`resolve_color`](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.1.3/src/gallium/frontends/lavapipe/lvp_execute.c#L1697)
resolves every layer through the highest view bit as a contiguous range,
including inactive views. Etalon therefore uses its existing prepared single-layer
resolve passes for sparse masks on Lavapipe. Contiguous masks retain the native
resolve path. The independently fingerprinted Intel resolve workaround is unchanged.

Core initialization tracking also incorrectly marked the entire attachment view
initialized after a sparse clear or resolve, and recorded only its first layer
as discarded. It now tracks implicitly initialized and discarded layers selected
by the view mask, relative to the view's base layer. Inactive initialized user
data is preserved; inactive uninitialized layers still require zero initialization
before subsequent reads. Depth/stencil aspect fixups use the same selected ranges.
Conservative load initialization and partial 3D slice handling are unchanged.

Pure Rust coverage checks mask guards and real initialization-tracker state for
single-bit, sparse and full masks at nonzero view bases. The existing GPU oracle
is retained, its resolved Lavapipe expected-failure exception is removed, and an
additional case pre-fills all layers before sparse ordinary and multisample passes
to verify that inactive user data survives. Linux CI must verify these subsequent
source changes; the 1624/1625 result predates the sparse resolve and core fixes.

Local verification after those fixes passed 135 selected core, HAL, API validation
and multiview cases. The new preservation case executed on Intel Vulkan; canonical
DX12 cases executed on Intel and WARP. Unsupported configurations retain their
explicit status. This host does not provide Lavapipe; Linux CI remains the driver
verification gate.

The final [Linux CI run](https://github.com/mediaprophet/etalon/actions/runs/41030580991) passed all 1,637 tests, including actual poisoned Lavapipe execution of the sparse ordinary, discard/resolve and prefilled inactive-layer cases. Windows, macOS and WebAssembly test jobs also passed for the same implementation revision.
