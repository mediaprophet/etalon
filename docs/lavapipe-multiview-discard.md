# Lavapipe multiview discard workaround

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

Pure Rust guard coverage checks both single-bit and sparse masks, every load
operation, ordinary passes and unguarded drivers. Linux CI must verify the actual
driver regression after this source change; the original run passed 1618 of 1619
tests and crashed in this case.
