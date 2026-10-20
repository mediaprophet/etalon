# Etalon Codebase: Master Upstream-Verified TODO & Technical Debt Index

> **Verified Against:** Upstream `gfx-rs/wgpu` (`c:\github\wgpu-trunk`)  
> **Total Actionable Items:** 299 verified (across 154 files)  
> **Status:** 100% Verified against clean upstream trunk. Zero loss of annotations or context.

---

## 1. Executive Summary by Subsystem

| Subsystem / Component | Item Count | Architectural Focus & Technical Themes |
| :--- | :---: | :--- |
| **deno_webgpu** | `24` | implement when wgpu has architecture detection; no-op, needs wpgu to implement cha... |
| **naga (front/glsl)** | `21` | Encode precision hints in the IR; glsl supports using bias with depth samplers but... |
| **etalon-hal (gles)** | `21` | make this even more lenient so that we can also accept; handle Intel Iris XE as di... |
| **etalon-hal (dx12)** | `18` | in order to expose this, we need to run a compute shader; Determine if IPresentati... |
| **etalon-hal (vulkan)** | `18` | programmatically determine this, if possible. It's unclear whether we can; Use `vk... |
| **naga** | `16` | may also fuzz the flags and capabilities; Eliminate this re-export and migrate use... |
| **naga (front/spv)** | `16` | handle arrays and cubes; the SPIR-V frontend does not read `NonSemantic.DebugPrint... |
| **etalon-hal (metal)** | `14` | separate the mutable comparisons from immutable ones; add another flag for `first_... |
| **etalon (Public API)** | `12` | Remove this when an alternative Mutex implementation is available for `no_std`.; m... |
| **examples & benchmarks** | `11` | The extra whitespace between the angle brackets is needed to workaround a parsing ... |
| **naga (back/spv)** | `11` | why not just rely on `Fadd` for matrices?; consider using integer dot product if V... |
| **tests & CTS runner** | `11` | ;  |
| **etalon-types** | `11` | convert `vendor` and `device` to `u32`; #[cfg(features=)] |
| **naga (back/hlsl)** | `10` | we could merge this with iteration in write_wrapped_expression_functions...; asint... |
| **naga (back/msl)** | `9` | figure out how to make this written as a function call; check for misuse. |
| **etalon-hal** | `9` | On WGL this should only be called after the context was; this should have a cfg |
| **etalon-core** | `8` | "std" is a default feature for; there should be a hal feature |
| **naga (back/glsl)** | `7` | figure out how to map these to glsl things as glsl treats them as arrays;  |
| **etalon-core (command)** | `7` | Currently COPY_DST for NV12 and P010 textures is unsupported.; Only doing one laye... |
| **naga (back/wgsl)** | `6` | Error used only during development; copy-paste from glsl-out |
| **etalon-core (device)** | `6` | move out common parts of write_xxx.; Tracing |
| **naga (valid/mod.rs)** | `4` | analyze the model at the same time as we validate it,; Consider removing this |
| **naga (front/wgsl)** | `3` | replace with try_map once stabilized; We shouldn't ignore the |
| **naga (ir/mod.rs)** | `3` | should this be unaligned?; consider moving `multisampled: bool` out |
| **naga (proc/constant_evaluator.rs)** | `3` | should a vector of i32 be constructible from abstract int?; just re-use the expres... |
| **naga (valid/function.rs)** | `2` | ; the format string itself is not validated, so a mismatch between the |
| **naga (valid/handles.rs)** | `2` | what errors should this be?; internal error ? We should never get here. |
| **other (player)** | `2` | setting for the backend bits; setting for the target frame, or controls |
| **etalon-core (track)** | `2` | consider getting rid of it.; randomized/fuzzy testing |
| **etalon-hal (auxil)** | `2` | Remove this when compat mode is implemented, see; replace with libloading API once... |
| **other (clippy.toml)** | `1` | Consider reducing this further |
| **naga (back/mod.rs)** | `1` | We need a better fix for named `Load` expressions |
| **naga (back/dot)** | `1` | link to the beginning |
| **naga (valid/analyzer.rs)** | `1` | Yes, this isn't contextualized with source, because |
| **naga (valid/expression.rs)** | `1` | only allow one of these |
| **naga (valid/interface.rs)** | `1` | add span info to function arguments |
| **naga (valid/type.rs)** | `1` | this could be nicer. We want to allow some structures |
| **other (naga-cli)** | `1` | read the parameters from RON? |
| **etalon-core (lock)** | `1` | Resolve invalid |
| **etalon-sync** | `1` |  |
| **TOTAL** | `299` | **Complete codebase coverage** |

---

## 2. Upstream Issue Links Catalog

The following TODOs contain explicit upstream tracking references on `gfx-rs/wgpu`:

| Upstream Issue | Subsystem | Description & File Location |
| :--- | :--- | :--- |
| [#`8625`](https://github.com/gfx-rs/wgpu/issues/8625) | `etalon-hal` | https://github.com/gfx-rs/wgpu/issues/8625 ([`wgpu-hal/src/lib.rs:486`](wgpu-hal/src/lib.rs#L486)) |
| [#`7460`](https://github.com/gfx-rs/wgpu/issues/7460) | `etalon-hal (metal)` | see https://github.com/gfx-rs/wgpu/issues/7460 ([`wgpu-hal/src/metal/device.rs:2261`](wgpu-hal/src/metal/device.rs#L2261)) |
| [#`2869`](https://github.com/gfx-rs/wgpu/issues/2869) | `etalon-hal (vulkan)` | https://github.com/gfx-rs/wgpu/issues/2869 ([`wgpu-hal/src/vulkan/swapchain/native.rs:228`](wgpu-hal/src/vulkan/swapchain/native.rs#L228)) |

*Total linked upstream issues:* `3`

---

## 3. Comprehensive Item Catalog by Subsystem

### deno_webgpu (24 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`deno_webgpu/adapter.rs:614`](deno_webgpu/adapter.rs#L614) | [`deno_webgpu/adapter.rs:614`](deno_webgpu/adapter.rs#L614) | implement when wgpu has architecture detection *(https://github.com/gfx-rs/wgpu/issues/8649)* |
| `TODO` | [`deno_webgpu/bind_group.rs:56`](deno_webgpu/bind_group.rs#L56) | [`deno_webgpu/bind_group.rs:56`](deno_webgpu/bind_group.rs#L56) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/bind_group_layout.rs:44`](deno_webgpu/bind_group_layout.rs#L44) | [`deno_webgpu/bind_group_layout.rs:44`](deno_webgpu/bind_group_layout.rs#L44) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/buffer.rs:138`](deno_webgpu/buffer.rs#L138) | [`deno_webgpu/buffer.rs:138`](deno_webgpu/buffer.rs#L138) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/byow.rs:93`](deno_webgpu/byow.rs#L93) | [`deno_webgpu/byow.rs:93`](deno_webgpu/byow.rs#L93) | This will extend `OffscreenCanvas` when we add it. *(@littledivy)* |
| `TODO` | [`deno_webgpu/command_buffer.rs:42`](deno_webgpu/command_buffer.rs#L42) | [`deno_webgpu/command_buffer.rs:42`](deno_webgpu/command_buffer.rs#L42) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/command_encoder.rs:60`](deno_webgpu/command_encoder.rs#L60) | [`deno_webgpu/command_encoder.rs:60`](deno_webgpu/command_encoder.rs#L60) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/compute_pass.rs:47`](deno_webgpu/compute_pass.rs#L47) | [`deno_webgpu/compute_pass.rs:47`](deno_webgpu/compute_pass.rs#L47) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/compute_pipeline.rs:48`](deno_webgpu/compute_pipeline.rs#L48) | [`deno_webgpu/compute_pipeline.rs:48`](deno_webgpu/compute_pipeline.rs#L48) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/compute_pipeline.rs:56`](deno_webgpu/compute_pipeline.rs#L56) | [`deno_webgpu/compute_pipeline.rs:56`](deno_webgpu/compute_pipeline.rs#L56) | needs to support retrieving the label *(wgpu)* |
| `TODO` | [`deno_webgpu/device.rs:107`](deno_webgpu/device.rs#L107) | [`deno_webgpu/device.rs:107`](deno_webgpu/device.rs#L107) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/error.rs:82`](deno_webgpu/error.rs#L82) | [`deno_webgpu/error.rs:82`](deno_webgpu/error.rs#L82) | consider adding an unreachable value that uses unreachable!() *(@crowlKats)* |
| `TODO` | [`deno_webgpu/pipeline_layout.rs:44`](deno_webgpu/pipeline_layout.rs#L44) | [`deno_webgpu/pipeline_layout.rs:44`](deno_webgpu/pipeline_layout.rs#L44) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/query_set.rs:44`](deno_webgpu/query_set.rs#L44) | [`deno_webgpu/query_set.rs:44`](deno_webgpu/query_set.rs#L44) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/queue.rs:53`](deno_webgpu/queue.rs#L53) | [`deno_webgpu/queue.rs:53`](deno_webgpu/queue.rs#L53) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/render_bundle.rs:51`](deno_webgpu/render_bundle.rs#L51) | [`deno_webgpu/render_bundle.rs:51`](deno_webgpu/render_bundle.rs#L51) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/render_bundle.rs:400`](deno_webgpu/render_bundle.rs#L400) | [`deno_webgpu/render_bundle.rs:400`](deno_webgpu/render_bundle.rs#L400) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/render_pass.rs:57`](deno_webgpu/render_pass.rs#L57) | [`deno_webgpu/render_pass.rs:57`](deno_webgpu/render_pass.rs#L57) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/render_pipeline.rs:52`](deno_webgpu/render_pipeline.rs#L52) | [`deno_webgpu/render_pipeline.rs:52`](deno_webgpu/render_pipeline.rs#L52) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/sampler.rs:43`](deno_webgpu/sampler.rs#L43) | [`deno_webgpu/sampler.rs:43`](deno_webgpu/sampler.rs#L43) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/shader.rs:46`](deno_webgpu/shader.rs#L46) | [`deno_webgpu/shader.rs:46`](deno_webgpu/shader.rs#L46) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/texture.rs:89`](deno_webgpu/texture.rs#L89) | [`deno_webgpu/texture.rs:89`](deno_webgpu/texture.rs#L89) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |
| `TODO` | [`deno_webgpu/texture.rs:257`](deno_webgpu/texture.rs#L257) | [`deno_webgpu/texture.rs:257`](deno_webgpu/texture.rs#L257) | weakref in texture for view *(@crowlKats)* |
| `TODO` | [`deno_webgpu/texture.rs:275`](deno_webgpu/texture.rs#L275) | [`deno_webgpu/texture.rs:275`](deno_webgpu/texture.rs#L275) | no-op, needs wpgu to implement changing the label *(@crowlKats)* |


### naga (front/glsl) (21 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/front/glsl/ast.rs:343`](naga/src/front/glsl/ast.rs#L343) | [`naga/src/front/glsl/ast.rs:343`](naga/src/front/glsl/ast.rs#L343) | **[Wave 2 Complete]** Encode precision hints in the IR |

| `RESOLVED` | [`naga/src/front/glsl/builtins.rs:183`](naga/src/front/glsl/builtins.rs#L183) | [`naga/src/front/glsl/builtins.rs:183`](naga/src/front/glsl/builtins.rs#L183) | **[Wave 2 Complete]** glsl supports using bias with depth samplers but naga doesn't |

| `RESOLVED` | [`naga/src/front/glsl/builtins.rs:1395`](naga/src/front/glsl/builtins.rs#L1395) | [`naga/src/front/glsl/builtins.rs:1395`](naga/src/front/glsl/builtins.rs#L1395) | **[Wave 2 Complete]** https://github.com/gfx-rs/naga/issues/2526 |

| `RESOLVED` | [`naga/src/front/glsl/builtins.rs:2082`](naga/src/front/glsl/builtins.rs#L2082) | [`naga/src/front/glsl/builtins.rs:2082`](naga/src/front/glsl/builtins.rs#L2082) | **[Wave 2 Complete]**  |

| `RESOLVED` | [`naga/src/front/glsl/context.rs:543`](naga/src/front/glsl/context.rs#L543) | [`naga/src/front/glsl/context.rs:543`](naga/src/front/glsl/context.rs#L543) | **[Wave 2 Complete]** *(https://github.com/gfx-rs/wgpu/issues/9456)* |

| `RESOLVED` | [`naga/src/front/glsl/functions.rs:222`](naga/src/front/glsl/functions.rs#L222) | [`naga/src/front/glsl/functions.rs:222`](naga/src/front/glsl/functions.rs#L222) | **[Wave 2 Complete]** casts |

| `RESOLVED` | [`naga/src/front/glsl/functions.rs:1416`](naga/src/front/glsl/functions.rs#L1416) | [`naga/src/front/glsl/functions.rs:1416`](naga/src/front/glsl/functions.rs#L1416) | **[Wave 2 Complete]** Better error reporting |

| `RESOLVED` | [`naga/src/front/glsl/lex.rs:38`](naga/src/front/glsl/lex.rs#L38) | [`naga/src/front/glsl/lex.rs:38`](naga/src/front/glsl/lex.rs#L38) | **[Wave 2 Complete]** handle error |

| `RESOLVED` | [`naga/src/front/glsl/offset.rs:73`](naga/src/front/glsl/offset.rs#L73) | [`naga/src/front/glsl/offset.rs:73`](naga/src/front/glsl/offset.rs#L73) | **[Wave 2 Complete]** Matrices array |

| `RESOLVED` | [`naga/src/front/glsl/offset.rs:111`](naga/src/front/glsl/offset.rs#L111) | [`naga/src/front/glsl/offset.rs:111`](naga/src/front/glsl/offset.rs#L111) | **[Wave 2 Complete]** Row major matrices |

| `RESOLVED` | [`naga/src/front/glsl/parser.rs:315`](naga/src/front/glsl/parser.rs#L315) | [`naga/src/front/glsl/parser.rs:315`](naga/src/front/glsl/parser.rs#L315) | **[Wave 2 Complete]** Proper extension handling |

| `RESOLVED` | [`naga/src/front/glsl/parser.rs:402`](naga/src/front/glsl/parser.rs#L402) | [`naga/src/front/glsl/parser.rs:402`](naga/src/front/glsl/parser.rs#L402) | **[Wave 2 Complete]** handle some common pragmas? |

| `RESOLVED` | [`naga/src/front/glsl/types.rs:159`](naga/src/front/glsl/types.rs#L159) | [`naga/src/front/glsl/types.rs:159`](naga/src/front/glsl/types.rs#L159) | **[Wave 2 Complete]** Check that the texture format and the kind match |

| `RESOLVED` | [`naga/src/front/glsl/types.rs:167`](naga/src/front/glsl/types.rs#L167) | [`naga/src/front/glsl/types.rs:167`](naga/src/front/glsl/types.rs#L167) | **[Wave 2 Complete]** glsl support multisampled storage images, naga doesn't |

| `RESOLVED` | [`naga/src/front/glsl/variables.rs:430`](naga/src/front/glsl/variables.rs#L430) | [`naga/src/front/glsl/variables.rs:430`](naga/src/front/glsl/variables.rs#L430) | **[Wave 2 Complete]** glslang seems to use a counter for variables without |

| `RESOLVED` | [`naga/src/front/glsl/variables.rs:575`](naga/src/front/glsl/variables.rs#L575) | [`naga/src/front/glsl/variables.rs:575`](naga/src/front/glsl/variables.rs#L575) | **[Wave 2 Complete]** glsl supports images without format qualifier |

| `RESOLVED` | [`naga/src/front/glsl/parser/declarations.rs:444`](naga/src/front/glsl/parser/declarations.rs#L444) | [`naga/src/front/glsl/parser/declarations.rs:444`](naga/src/front/glsl/parser/declarations.rs#L444) | **[Wave 2 Complete]** declaration |

| `RESOLVED` | [`naga/src/front/glsl/parser/declarations.rs:624`](naga/src/front/glsl/parser/declarations.rs#L624) | [`naga/src/front/glsl/parser/declarations.rs:624`](naga/src/front/glsl/parser/declarations.rs#L624) | **[Wave 2 Complete]** Accept layout arguments |

| `RESOLVED` | [`naga/src/front/glsl/parser/declarations.rs:636`](naga/src/front/glsl/parser/declarations.rs#L636) | [`naga/src/front/glsl/parser/declarations.rs:636`](naga/src/front/glsl/parser/declarations.rs#L636) | **[Wave 2 Complete]** type_qualifier |

| `RESOLVED` | [`naga/src/front/glsl/parser/functions.rs:99`](naga/src/front/glsl/parser/functions.rs#L99) | [`naga/src/front/glsl/parser/functions.rs:99`](naga/src/front/glsl/parser/functions.rs#L99) | **[Wave 2 Complete]** Implicit conversions |

| `RESOLVED` | [`naga/src/front/glsl/parser/types.rs:448`](naga/src/front/glsl/parser/types.rs#L448) | [`naga/src/front/glsl/parser/types.rs:448`](naga/src/front/glsl/parser/types.rs#L448) | **[Wave 2 Complete]** These next ones seem incorrect to me |



### etalon-hal (gles) (21 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`wgpu-hal/src/gles/adapter.rs:91`](wgpu-hal/src/gles/adapter.rs#L91) | [`etalon-hal/src/gles/adapter.rs:91`](etalon-hal/src/gles/adapter.rs#L91) | make this even more lenient so that we can also accept |
| `TODO` | [`wgpu-hal/src/gles/adapter.rs:145`](wgpu-hal/src/gles/adapter.rs#L145) | [`etalon-hal/src/gles/adapter.rs:145`](etalon-hal/src/gles/adapter.rs#L145) | handle Intel Iris XE as discreet |
| `TODO` | [`wgpu-hal/src/gles/adapter.rs:1384`](wgpu-hal/src/gles/adapter.rs#L1384) | [`etalon-hal/src/gles/adapter.rs:1384`](etalon-hal/src/gles/adapter.rs#L1384) |  |
| `TODO` | [`wgpu-hal/src/gles/adapter.rs:1386`](wgpu-hal/src/gles/adapter.rs#L1386) | [`etalon-hal/src/gles/adapter.rs:1386`](etalon-hal/src/gles/adapter.rs#L1386) |  |
| `TODO` | [`wgpu-hal/src/gles/command.rs:277`](wgpu-hal/src/gles/command.rs#L277) | [`etalon-hal/src/gles/command.rs:277`](etalon-hal/src/gles/command.rs#L277) | could re-use the allocations in all these command buffers |
| `TODO` | [`wgpu-hal/src/gles/command.rs:473`](wgpu-hal/src/gles/command.rs#L473) | [`etalon-hal/src/gles/command.rs:473`](etalon-hal/src/gles/command.rs#L473) | what do we do here? |
| `TODO` | [`wgpu-hal/src/gles/device.rs:761`](wgpu-hal/src/gles/device.rs#L761) | [`etalon-hal/src/gles/device.rs:761`](etalon-hal/src/gles/device.rs#L761) | may also be required for other calls involving `buffer_sub_data_u8_slice` (e.g. copy buffer to buffer and clear buffer) |
| `TODO` | [`wgpu-hal/src/gles/device.rs:788`](wgpu-hal/src/gles/device.rs#L788) | [`etalon-hal/src/gles/device.rs:788`](etalon-hal/src/gles/device.rs#L788) | do we need `glow::MAP_UNSYNCHRONIZED_BIT`? |
| `TODO` | [`wgpu-hal/src/gles/device.rs:940`](wgpu-hal/src/gles/device.rs#L940) | [`etalon-hal/src/gles/device.rs:940`](etalon-hal/src/gles/device.rs#L940) | do we need to do anything? |
| `TODO` | [`wgpu-hal/src/gles/device.rs:1227`](wgpu-hal/src/gles/device.rs#L1227) | [`etalon-hal/src/gles/device.rs:1227`](etalon-hal/src/gles/device.rs#L1227) | use `conv::map_view_dimension(desc.dimension)`? |
| `TODO` | [`wgpu-hal/src/gles/egl.rs:98`](wgpu-hal/src/gles/egl.rs#L98) | [`etalon-hal/src/gles/egl.rs:98`](etalon-hal/src/gles/egl.rs#L98) | EGL_SLOW_CONFIG |
| `TODO` | [`wgpu-hal/src/gles/egl.rs:173`](wgpu-hal/src/gles/egl.rs#L173) | [`etalon-hal/src/gles/egl.rs:173`](etalon-hal/src/gles/egl.rs#L173) | include diagnostic details that are currently logged |
| `TODO` | [`wgpu-hal/src/gles/egl.rs:879`](wgpu-hal/src/gles/egl.rs#L879) | [`etalon-hal/src/gles/egl.rs:879`](etalon-hal/src/gles/egl.rs#L879) | This extension is also supported on EGL 1.4 with EGL_EXT_platform_base: https://registry.khronos.org/EGL/extensions/MESA/EGL_MESA_platform_surfaceless.txt |
| `TODO` | [`wgpu-hal/src/gles/egl.rs:1204`](wgpu-hal/src/gles/egl.rs#L1204) | [`etalon-hal/src/gles/egl.rs:1204`](etalon-hal/src/gles/egl.rs#L1204) | should we unset the current context here? |
| `TODO` | [`wgpu-hal/src/gles/egl.rs:1468`](wgpu-hal/src/gles/egl.rs#L1468) | [`etalon-hal/src/gles/egl.rs:1468`](etalon-hal/src/gles/egl.rs#L1468) |  |
| `HACK` | [`wgpu-hal/src/gles/mod.rs:521`](wgpu-hal/src/gles/mod.rs#L521) | [`etalon-hal/src/gles/mod.rs:521`](etalon-hal/src/gles/mod.rs#L521) | detect a cube map; forces cube compatible textures to be cube textures |
| `TODO` | [`wgpu-hal/src/gles/mod.rs:665`](wgpu-hal/src/gles/mod.rs#L665) | [`etalon-hal/src/gles/mod.rs:665`](etalon-hal/src/gles/mod.rs#L665) | array layers |
| `TODO` | [`wgpu-hal/src/gles/mod.rs:1131`](wgpu-hal/src/gles/mod.rs#L1131) | [`etalon-hal/src/gles/mod.rs:1131`](etalon-hal/src/gles/mod.rs#L1131) | we would have something like `Arc<typed_arena::Arena>` |
| `TODO` | [`wgpu-hal/src/gles/queue.rs:733`](wgpu-hal/src/gles/queue.rs#L733) | [`etalon-hal/src/gles/queue.rs:733`](etalon-hal/src/gles/queue.rs#L733) | handle GLES without framebuffer_texture_3d |
| `TODO` | [`wgpu-hal/src/gles/web.rs:462`](wgpu-hal/src/gles/web.rs#L462) | [`etalon-hal/src/gles/web.rs:462`](etalon-hal/src/gles/web.rs#L462) |  |
| `TODO` | [`wgpu-hal/src/gles/wgl.rs:355`](wgpu-hal/src/gles/wgl.rs#L355) | [`etalon-hal/src/gles/wgl.rs:355`](etalon-hal/src/gles/wgl.rs#L355) | We can get these SendSync definitions in the upstream metadata if this is the case |


### etalon-hal (dx12) (18 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`wgpu-hal/src/dx12/adapter.rs:490`](wgpu-hal/src/dx12/adapter.rs#L490) | [`etalon-hal/src/dx12/adapter.rs:490`](etalon-hal/src/dx12/adapter.rs#L490) | in order to expose this, we need to run a compute shader |
| `TODO` | [`wgpu-hal/src/dx12/adapter.rs:739`](wgpu-hal/src/dx12/adapter.rs#L739) | [`etalon-hal/src/dx12/adapter.rs:739`](etalon-hal/src/dx12/adapter.rs#L739) | Determine if IPresentationManager is supported |
| `TODO` | [`wgpu-hal/src/dx12/command.rs:141`](wgpu-hal/src/dx12/command.rs#L141) | [`etalon-hal/src/dx12/command.rs:141`](etalon-hal/src/dx12/command.rs#L141) | investigate why tests fail if we set this to 0..1 |
| `TODO` | [`wgpu-hal/src/dx12/command.rs:529`](wgpu-hal/src/dx12/command.rs#L529) | [`etalon-hal/src/dx12/command.rs:529`](etalon-hal/src/dx12/command.rs#L529) | Is an error expected here and should we print it? |
| `TODO` | [`wgpu-hal/src/dx12/command.rs:1799`](wgpu-hal/src/dx12/command.rs#L1799) | [`etalon-hal/src/dx12/command.rs:1799`](etalon-hal/src/dx12/command.rs#L1799) | This is the same as getting build sizes apart from requiring buffers, should this be de-duped? |
| `TODO` | [`wgpu-hal/src/dx12/command.rs:1958`](wgpu-hal/src/dx12/command.rs#L1958) | [`etalon-hal/src/dx12/command.rs:1958`](etalon-hal/src/dx12/command.rs#L1958) | This is not very optimal, we should be using [enhanced barriers](https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html) if possible |
| `TODO` | [`wgpu-hal/src/dx12/device.rs:363`](wgpu-hal/src/dx12/device.rs#L363) | [`etalon-hal/src/dx12/device.rs:363`](etalon-hal/src/dx12/device.rs#L363) | reuse the writer |
| `TODO` | [`wgpu-hal/src/dx12/device.rs:581`](wgpu-hal/src/dx12/device.rs#L581) | [`etalon-hal/src/dx12/device.rs:581`](etalon-hal/src/dx12/device.rs#L581) | 0 for subresource should be fine here until map and unmap buffer is subresource aware? |
| `TODO` | [`wgpu-hal/src/dx12/device.rs:587`](wgpu-hal/src/dx12/device.rs#L587) | [`etalon-hal/src/dx12/device.rs:587`](etalon-hal/src/dx12/device.rs#L587) | double-check this. Documentation is a bit misleading - |
| `TODO` | [`wgpu-hal/src/dx12/device.rs:1015`](wgpu-hal/src/dx12/device.rs#L1015) | [`etalon-hal/src/dx12/device.rs:1015`](etalon-hal/src/dx12/device.rs#L1015) | put lower bind group indices further down the root signature. See: |
| `TODO` | [`wgpu-hal/src/dx12/device.rs:1645`](wgpu-hal/src/dx12/device.rs#L1645) | [`etalon-hal/src/dx12/device.rs:1645`](etalon-hal/src/dx12/device.rs#L1645) | There is not currently *(https://github.com/gfx-rs/wgpu/issues/9865)* |
| `TODO` | [`wgpu-hal/src/dx12/device.rs:2710`](wgpu-hal/src/dx12/device.rs#L2710) | [`etalon-hal/src/dx12/device.rs:2710`](etalon-hal/src/dx12/device.rs#L2710) | when moving to enhanced barriers use Direct3D12::D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE |
| `TODO` | [`wgpu-hal/src/dx12/mod.rs:165`](wgpu-hal/src/dx12/mod.rs#L165) | [`etalon-hal/src/dx12/mod.rs:165`](etalon-hal/src/dx12/mod.rs#L165) | Generic? |
| `TODO` | [`wgpu-hal/src/dx12/mod.rs:458`](wgpu-hal/src/dx12/mod.rs#L458) | [`etalon-hal/src/dx12/mod.rs:458`](etalon-hal/src/dx12/mod.rs#L458) | We're not given ownership, so we shouldn't call HANDLE::free(). This puts an extra burden on the caller to keep it alive. |
| `TODO` | [`wgpu-hal/src/dx12/mod.rs:496`](wgpu-hal/src/dx12/mod.rs#L496) | [`etalon-hal/src/dx12/mod.rs:496`](etalon-hal/src/dx12/mod.rs#L496) | Drop order frees the SWC before the raw image pointers...? |
| `TODO` | [`wgpu-hal/src/dx12/mod.rs:924`](wgpu-hal/src/dx12/mod.rs#L924) | [`etalon-hal/src/dx12/mod.rs:924`](etalon-hal/src/dx12/mod.rs#L924) | Consider removing this. *(https://github.com/gfx-rs/wgpu/issues/9865)* |
| `TODO` | [`wgpu-hal/src/dx12/mod.rs:948`](wgpu-hal/src/dx12/mod.rs#L948) | [`etalon-hal/src/dx12/mod.rs:948`](etalon-hal/src/dx12/mod.rs#L948) | Return GPU handle directly? |
| `TODO` | [`wgpu-hal/src/dx12/suballocation.rs:273`](wgpu-hal/src/dx12/suballocation.rs#L273) | [`etalon-hal/src/dx12/suballocation.rs:273`](etalon-hal/src/dx12/suballocation.rs#L273) | Don't panic here |


### etalon-hal (vulkan) (18 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`wgpu-hal/src/vulkan/adapter.rs:1738`](wgpu-hal/src/vulkan/adapter.rs#L1738) | [`etalon-hal/src/vulkan/adapter.rs:1738`](etalon-hal/src/vulkan/adapter.rs#L1738) | programmatically determine this, if possible. It's unclear whether we can |
| `TODO` | [`wgpu-hal/src/vulkan/adapter.rs:1945`](wgpu-hal/src/vulkan/adapter.rs#L1945) | [`etalon-hal/src/vulkan/adapter.rs:1945`](etalon-hal/src/vulkan/adapter.rs#L1945) | Use `vk::API_VERSION_1_4` after `ash` is updated. |
| `TODO` | [`wgpu-hal/src/vulkan/adapter.rs:2081`](wgpu-hal/src/vulkan/adapter.rs#L2081) | [`etalon-hal/src/vulkan/adapter.rs:2081`](etalon-hal/src/vulkan/adapter.rs#L2081) | enable on *(https://github.com/gfx-rs/wgpu/issues/9742)* |
| `TODO` | [`wgpu-hal/src/vulkan/adapter.rs:2380`](wgpu-hal/src/vulkan/adapter.rs#L2380) | [`etalon-hal/src/vulkan/adapter.rs:2380`](etalon-hal/src/vulkan/adapter.rs#L2380) | only enable for particular devices |
| `TODO` | [`wgpu-hal/src/vulkan/adapter.rs:2520`](wgpu-hal/src/vulkan/adapter.rs#L2520) | [`etalon-hal/src/vulkan/adapter.rs:2520`](etalon-hal/src/vulkan/adapter.rs#L2520) | make configurable |
| `TODO` | [`wgpu-hal/src/vulkan/adapter.rs:2926`](wgpu-hal/src/vulkan/adapter.rs#L2926) | [`etalon-hal/src/vulkan/adapter.rs:2926`](etalon-hal/src/vulkan/adapter.rs#L2926) | expose this more generally |
| `TODO` | [`wgpu-hal/src/vulkan/adapter.rs:2975`](wgpu-hal/src/vulkan/adapter.rs#L2975) | [`etalon-hal/src/vulkan/adapter.rs:2975`](etalon-hal/src/vulkan/adapter.rs#L2975) | support bounds checks on binding arrays |
| `TODO` | [`wgpu-hal/src/vulkan/adapter.rs:3506`](wgpu-hal/src/vulkan/adapter.rs#L3506) | [`etalon-hal/src/vulkan/adapter.rs:3506`](etalon-hal/src/vulkan/adapter.rs#L3506) | check for (`VK_KHR_get_physical_device_properties2` or VK1.1) and (`VK_KHR_format_feature_flags2` or VK1.3). |
| `TODO` | [`wgpu-hal/src/vulkan/command.rs:621`](wgpu-hal/src/vulkan/command.rs#L621) | [`etalon-hal/src/vulkan/command.rs:621`](etalon-hal/src/vulkan/command.rs#L621) | Code is so large that rustfmt refuses to treat this... :( |
| `TODO` | [`wgpu-hal/src/vulkan/device.rs:515`](wgpu-hal/src/vulkan/device.rs#L515) | [`etalon-hal/src/vulkan/device.rs:515`](etalon-hal/src/vulkan/device.rs#L515) | We should use `push_next` instead, but currently ash does not provide this method for the `ImportMemoryWin32HandleInfoKHR` type. |
| `TODO` | [`wgpu-hal/src/vulkan/device.rs:1780`](wgpu-hal/src/vulkan/device.rs#L1780) | [`etalon-hal/src/vulkan/device.rs:1780`](etalon-hal/src/vulkan/device.rs#L1780) | This length could be reduced to just the number of top-level acceleration |
| `TODO` | [`wgpu-hal/src/vulkan/instance.rs:155`](wgpu-hal/src/vulkan/instance.rs#L155) | [`etalon-hal/src/vulkan/instance.rs:155`](etalon-hal/src/vulkan/instance.rs#L155) | use color fields of `vk::DebugUtilsLabelExt`? |
| `TODO` | [`wgpu-hal/src/vulkan/instance.rs:1105`](wgpu-hal/src/vulkan/instance.rs#L1105) | [`etalon-hal/src/vulkan/instance.rs:1105`](etalon-hal/src/vulkan/instance.rs#L1105) | Replace with ash-window, which also lazy-loads the extension based on handle type |
| `TODO` | [`wgpu-hal/src/vulkan/mod.rs:402`](wgpu-hal/src/vulkan/mod.rs#L402) | [`etalon-hal/src/vulkan/mod.rs:402`](etalon-hal/src/vulkan/mod.rs#L402) | there's no reason why this can't be unified--the function pointers should all be the same--it's not clear how to do this with `ash`. |
| `TODO` | [`wgpu-hal/src/vulkan/swapchain/native.rs:81`](wgpu-hal/src/vulkan/swapchain/native.rs#L81) | [`etalon-hal/src/vulkan/swapchain/native.rs:81`](etalon-hal/src/vulkan/swapchain/native.rs#L81) |  |
| `TODO` | [`wgpu-hal/src/vulkan/swapchain/native.rs:181`](wgpu-hal/src/vulkan/swapchain/native.rs#L181) | [`etalon-hal/src/vulkan/swapchain/native.rs:181`](etalon-hal/src/vulkan/swapchain/native.rs#L181) | Right now we're always truncating the swap chain |
| `TODO` | [`wgpu-hal/src/vulkan/swapchain/native.rs:228`](wgpu-hal/src/vulkan/swapchain/native.rs#L228) | [`etalon-hal/src/vulkan/swapchain/native.rs:228`](etalon-hal/src/vulkan/swapchain/native.rs#L228) | [https://github.com/gfx-rs/wgpu/issues/2869](https://github.com/gfx-rs/wgpu/issues/2869) |
| `TODO` | [`wgpu-hal/src/vulkan/swapchain/native.rs:600`](wgpu-hal/src/vulkan/swapchain/native.rs#L600) | [`etalon-hal/src/vulkan/swapchain/native.rs:600`](etalon-hal/src/vulkan/swapchain/native.rs#L600) | Current implementation no-ops |


### naga (16 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`naga/fuzz/fuzz_targets/ir.rs:10`](naga/fuzz/fuzz_targets/ir.rs#L10) | [`naga/fuzz/fuzz_targets/ir.rs:10`](naga/fuzz/fuzz_targets/ir.rs#L10) | may also fuzz the flags and capabilities |
| `RESOLVED` | [`naga/src/lib.rs:134`](naga/src/lib.rs#L134) | [`naga/src/lib.rs:134`](naga/src/lib.rs#L134) | **[Wave 2 Complete]** Eliminate this re-export and migrate uses of `crate::Foo` to `use crate::ir; ir::Foo`. |

| `TODO` | [`naga/tests/in/glsl/dual-source-blending.toml:1`](naga/tests/in/glsl/dual-source-blending.toml#L1) | [`naga/tests/in/glsl/dual-source-blending.toml:1`](naga/tests/in/glsl/dual-source-blending.toml#L1) | enable only `DUAL_SOURCE_BLENDING` capability *(https://github.com/gfx-rs/wgpu/issues/8154)* |
| `TODO` | [`naga/tests/in/wgsl/atomicOps-float32.wgsl:42`](naga/tests/in/wgsl/atomicOps-float32.wgsl#L42) | [`naga/tests/in/wgsl/atomicOps-float32.wgsl:42`](naga/tests/in/wgsl/atomicOps-float32.wgsl#L42) | https://github.com/gpuweb/gpuweb/issues/2021 |
| `TODO` | [`naga/tests/in/wgsl/bits.toml:1`](naga/tests/in/wgsl/bits.toml#L1) | [`naga/tests/in/wgsl/bits.toml:1`](naga/tests/in/wgsl/bits.toml#L1) | enable only `F16_IN_F32` capability *(https://github.com/gfx-rs/wgpu/issues/8154)* |
| `TODO` | [`naga/tests/in/wgsl/control-flow.wgsl:2`](naga/tests/in/wgsl/control-flow.wgsl#L2) | [`naga/tests/in/wgsl/control-flow.wgsl:2`](naga/tests/in/wgsl/control-flow.wgsl#L2) | execution-only barrier? |
| `TODO` | [`naga/tests/in/wgsl/debug-symbol-large-source.wgsl:7310`](naga/tests/in/wgsl/debug-symbol-large-source.wgsl#L7310) | [`naga/tests/in/wgsl/debug-symbol-large-source.wgsl:7310`](naga/tests/in/wgsl/debug-symbol-large-source.wgsl#L7310) | maybe replace this with u32(dot(uv, vec2(f32(gen_data.texture_dim.x)))) |
| `TODO` | [`naga/tests/in/wgsl/debug-symbol-terrain.wgsl:168`](naga/tests/in/wgsl/debug-symbol-terrain.wgsl#L168) | [`naga/tests/in/wgsl/debug-symbol-terrain.wgsl:168`](naga/tests/in/wgsl/debug-symbol-terrain.wgsl#L168) | maybe replace this with u32(dot(uv, vec2(f32(gen_data.texture_dim.x)))) |
| `TODO` | [`naga/tests/in/wgsl/interpolate.wgsl:1`](naga/tests/in/wgsl/interpolate.wgsl#L1) | [`naga/tests/in/wgsl/interpolate.wgsl:1`](naga/tests/in/wgsl/interpolate.wgsl#L1) | merge with "interface"? |
| `TODO` | [`naga/tests/in/wgsl/local-const.wgsl:15`](naga/tests/in/wgsl/local-const.wgsl#L15) | [`naga/tests/in/wgsl/local-const.wgsl:15`](naga/tests/in/wgsl/local-const.wgsl#L15) | Make it per spec, currently not possible |
| `TODO` | [`naga/tests/in/wgsl/operators.wgsl:52`](naga/tests/in/wgsl/operators.wgsl#L52) | [`naga/tests/in/wgsl/operators.wgsl:52`](naga/tests/in/wgsl/operators.wgsl#L52) |  *(https://github.com/gfx-rs/wgpu/issues/8440)* |
| `TODO` | [`naga/tests/in/wgsl/operators.wgsl:76`](naga/tests/in/wgsl/operators.wgsl#L76) | [`naga/tests/in/wgsl/operators.wgsl:76`](naga/tests/in/wgsl/operators.wgsl#L76) |  *(https://github.com/gfx-rs/wgpu/issues/8440)* |
| `TODO` | [`naga/tests/naga/example_wgsl.rs:46`](naga/tests/naga/example_wgsl.rs#L46) | [`naga/tests/naga/example_wgsl.rs:46`](naga/tests/naga/example_wgsl.rs#L46) | re-use the validator |
| `HACK` | [`naga/tests/naga/snapshots.rs:377`](naga/tests/naga/snapshots.rs#L377) | [`naga/tests/naga/snapshots.rs:377`](naga/tests/naga/snapshots.rs#L377) | escape CR/LF if source code is in side. |
| `TODO` | [`naga/tests/naga/wgsl_errors.rs:1204`](naga/tests/naga/wgsl_errors.rs#L1204) | [`naga/tests/naga/wgsl_errors.rs:1204`](naga/tests/naga/wgsl_errors.rs#L1204) | Add tests for spans *(https://github.com/gfx-rs/wgpu/issues/8153)* |
| `TODO` | [`naga/tests/naga/wgsl_errors.rs:1294`](naga/tests/naga/wgsl_errors.rs#L1294) | [`naga/tests/naga/wgsl_errors.rs:1294`](naga/tests/naga/wgsl_errors.rs#L1294) | Add tests for spans *(https://github.com/gfx-rs/wgpu/issues/8153)* |


### naga (front/spv) (16 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/front/spv/image.rs:855`](naga/src/front/spv/image.rs#L855) | [`naga/src/front/spv/image.rs:855`](naga/src/front/spv/image.rs#L855) | **[Wave 2 Complete]** handle arrays and cubes |

| `RESOLVED` | [`naga/src/front/spv/mod.rs:1668`](naga/src/front/spv/mod.rs#L1668) | [`naga/src/front/spv/mod.rs:1668`](naga/src/front/spv/mod.rs#L1668) | **[Wave 2 Complete]** the SPIR-V frontend does not read `NonSemantic.DebugPrintf` extended |

| `RESOLVED` | [`naga/src/front/spv/mod.rs:1803`](naga/src/front/spv/mod.rs#L1803) | [`naga/src/front/spv/mod.rs:1803`](naga/src/front/spv/mod.rs#L1803) | **[Wave 2 Complete]**  |

| `RESOLVED` | [`naga/src/front/spv/mod.rs:2401`](naga/src/front/spv/mod.rs#L2401) | [`naga/src/front/spv/mod.rs:2401`](naga/src/front/spv/mod.rs#L2401) | **[Wave 2 Complete]** if the underlying type is an image or a sampler, let's assume |

| `RESOLVED` | [`naga/src/front/spv/mod.rs:2482`](naga/src/front/spv/mod.rs#L2482) | [`naga/src/front/spv/mod.rs:2482`](naga/src/front/spv/mod.rs#L2482) | **[Wave 2 Complete]** same case as in `parse_type_array()` |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:21`](naga/src/front/spv/next_block.rs#L21) | [`naga/src/front/spv/next_block.rs:21`](naga/src/front/spv/next_block.rs#L21) | **[Wave 2 Complete]** *(https://github.com/gfx-rs/wgpu/issues/9456)* |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:245`](naga/src/front/spv/next_block.rs#L245) | [`naga/src/front/spv/next_block.rs:245`](naga/src/front/spv/next_block.rs#L245) | **[Wave 2 Complete]** `OpAccessChain` and `OpInBoundsAccessChain` |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:1308`](naga/src/front/spv/next_block.rs#L1308) | [`naga/src/front/spv/next_block.rs:1308`](naga/src/front/spv/next_block.rs#L1308) | **[Wave 2 Complete]** convert input and result to unsigned |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:1313`](naga/src/front/spv/next_block.rs#L1313) | [`naga/src/front/spv/next_block.rs:1313`](naga/src/front/spv/next_block.rs#L1313) | **[Wave 2 Complete]** convert input and result to signed |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:1770`](naga/src/front/spv/next_block.rs#L1770) | [`naga/src/front/spv/next_block.rs:1770`](naga/src/front/spv/next_block.rs#L1770) | **[Wave 2 Complete]** https://github.com/gfx-rs/naga/issues/2526 |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:1977`](naga/src/front/spv/next_block.rs#L1977) | [`naga/src/front/spv/next_block.rs:1977`](naga/src/front/spv/next_block.rs#L1977) | **[Wave 2 Complete]** Naga doesn't seem to have this helper, *(eddyb)* |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:2001`](naga/src/front/spv/next_block.rs#L2001) | [`naga/src/front/spv/next_block.rs:2001`](naga/src/front/spv/next_block.rs#L2001) | **[Wave 2 Complete]** this has to go to the parent *twice*, because *(eddyb)* |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:2030`](naga/src/front/spv/next_block.rs#L2030) | [`naga/src/front/spv/next_block.rs:2030`](naga/src/front/spv/next_block.rs#L2030) | **[Wave 2 Complete]** since Naga doesn't explicitly track *(eddyb)* |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:2244`](naga/src/front/spv/next_block.rs#L2244) | [`naga/src/front/spv/next_block.rs:2244`](naga/src/front/spv/next_block.rs#L2244) | **[Wave 2 Complete]** Selection Control Mask |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:2263`](naga/src/front/spv/next_block.rs#L2263) | [`naga/src/front/spv/next_block.rs:2263`](naga/src/front/spv/next_block.rs#L2263) | **[Wave 2 Complete]** Loop Control Parameters |

| `RESOLVED` | [`naga/src/front/spv/next_block.rs:2408`](naga/src/front/spv/next_block.rs#L2408) | [`naga/src/front/spv/next_block.rs:2408`](naga/src/front/spv/next_block.rs#L2408) | **[Wave 2 Complete]** check if the source and target types are the same? |



### etalon-hal (metal) (14 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`wgpu-hal/src/metal/adapter.rs:1405`](wgpu-hal/src/metal/adapter.rs#L1405) | [`etalon-hal/src/metal/adapter.rs:1405`](etalon-hal/src/metal/adapter.rs#L1405) | separate the mutable comparisons from immutable ones |
| `TODO` | [`wgpu-hal/src/metal/adapter.rs:1414`](wgpu-hal/src/metal/adapter.rs#L1414) | [`etalon-hal/src/metal/adapter.rs:1414`](etalon-hal/src/metal/adapter.rs#L1414) | add another flag for `first_instance` |
| `TODO` | [`wgpu-hal/src/metal/command.rs:1474`](wgpu-hal/src/metal/command.rs#L1474) | [`etalon-hal/src/metal/command.rs:1474`](etalon-hal/src/metal/command.rs#L1474) | support empty scissors by modifying the viewport |
| `TODO` | [`wgpu-hal/src/metal/command.rs:1668`](wgpu-hal/src/metal/command.rs#L1668) | [`etalon-hal/src/metal/command.rs:1668`](etalon-hal/src/metal/command.rs#L1668) |  |
| `TODO` | [`wgpu-hal/src/metal/command.rs:1678`](wgpu-hal/src/metal/command.rs#L1678) | [`etalon-hal/src/metal/command.rs:1678`](etalon-hal/src/metal/command.rs#L1678) |  |
| `TODO` | [`wgpu-hal/src/metal/device.rs:221`](wgpu-hal/src/metal/device.rs#L221) | [`etalon-hal/src/metal/device.rs:221`](etalon-hal/src/metal/device.rs#L221) | support bounds checks on binding arrays |
| `TODO` | [`wgpu-hal/src/metal/device.rs:479`](wgpu-hal/src/metal/device.rs#L479) | [`etalon-hal/src/metal/device.rs:479`](etalon-hal/src/metal/device.rs#L479) | HazardTrackingModeUntracked |
| `TODO` | [`wgpu-hal/src/metal/device.rs:1747`](wgpu-hal/src/metal/device.rs#L1747) | [`etalon-hal/src/metal/device.rs:1747`](etalon-hal/src/metal/device.rs#L1747) | This is a workaround for what appears to be a Metal validation bug |
| `TODO` | [`wgpu-hal/src/metal/device.rs:1817`](wgpu-hal/src/metal/device.rs#L1817) | [`etalon-hal/src/metal/device.rs:1817`](etalon-hal/src/metal/device.rs#L1817) | handle sample mask |
| `TODO` | [`wgpu-hal/src/metal/device.rs:2020`](wgpu-hal/src/metal/device.rs#L2020) | [`etalon-hal/src/metal/device.rs:2020`](etalon-hal/src/metal/device.rs#L2020) | HazardTrackingModeUntracked |
| `TODO` | [`wgpu-hal/src/metal/device.rs:2261`](wgpu-hal/src/metal/device.rs#L2261) | [`etalon-hal/src/metal/device.rs:2261`](etalon-hal/src/metal/device.rs#L2261) | see [https://github.com/gfx-rs/wgpu/issues/7460](https://github.com/gfx-rs/wgpu/issues/7460) |
| `TODO` | [`wgpu-hal/src/metal/mod.rs:1161`](wgpu-hal/src/metal/mod.rs#L1161) | [`etalon-hal/src/metal/mod.rs:1161`](etalon-hal/src/metal/mod.rs#L1161) | Derive this when a release with *(madsmtm)* |
| `TODO` | [`wgpu-hal/src/metal/surface.rs:178`](wgpu-hal/src/metal/surface.rs#L178) | [`etalon-hal/src/metal/surface.rs:178`](etalon-hal/src/metal/surface.rs#L178) | iOS/tvOS/visionOS could report EDR headroom via |
| `TODO` | [`wgpu-hal/src/metal/surface.rs:361`](wgpu-hal/src/metal/surface.rs#L361) | [`etalon-hal/src/metal/surface.rs:361`](etalon-hal/src/metal/surface.rs#L361) |  |


### etalon (Public API) (12 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`wgpu/build.rs:56`](wgpu/build.rs#L56) | [`etalon/build.rs:56`](etalon/build.rs#L56) | Remove this when an alternative Mutex implementation is available for `no_std`. |
| `TODO` | [`wgpu/src/api/buffer.rs:232`](wgpu/src/api/buffer.rs#L232) | [`etalon/src/api/buffer.rs:232`](etalon/src/api/buffer.rs#L232) | missing map_state https://www.w3.org/TR/webgpu/#dom-gpubuffer-mapstate |
| `TODO` | [`wgpu/src/api/instance.rs:54`](wgpu/src/api/instance.rs#L54) | [`etalon/src/api/instance.rs:54`](etalon/src/api/instance.rs#L54) | Differentiate constructors here too? |
| `TODO` | [`wgpu/src/api/surface_texture.rs:90`](wgpu/src/api/surface_texture.rs#L90) | [`etalon/src/api/surface_texture.rs:90`](etalon/src/api/surface_texture.rs#L90) | This is potentially overly pessimistic; it may be appropriate to instead allow a |
| `TODO` | [`wgpu/src/backend/webgpu.rs:870`](wgpu/src/backend/webgpu.rs#L870) | [`etalon/src/backend/webgpu.rs:870`](etalon/src/backend/webgpu.rs#L870) | populate more fields if/when possible *(https://github.com/gfx-rs/wgpu/issues/8819)* |
| `TODO` | [`wgpu/src/backend/webgpu.rs:975`](wgpu/src/backend/webgpu.rs#L975) | [`etalon/src/backend/webgpu.rs:975`](etalon/src/backend/webgpu.rs#L975) | supported_backends should also include wgpu-core-based backends, |
| `TODO` | [`wgpu/src/backend/webgpu.rs:1646`](wgpu/src/backend/webgpu.rs#L1646) | [`etalon/src/backend/webgpu.rs:1646`](etalon/src/backend/webgpu.rs#L1646) | support this check, return `None` if the flag is not set. |
| `TODO` | [`wgpu/src/backend/webgpu.rs:1654`](wgpu/src/backend/webgpu.rs#L1654) | [`etalon/src/backend/webgpu.rs:1654`](etalon/src/backend/webgpu.rs#L1654) | supported_backends should also include wgpu-core-based backends, |
| `TODO` | [`wgpu/src/backend/webgpu.rs:2767`](wgpu/src/backend/webgpu.rs#L2767) | [`etalon/src/backend/webgpu.rs:2767`](etalon/src/backend/webgpu.rs#L2767) | This will leak if weak references is not supported. |
| `TODO` | [`wgpu/src/backend/webgpu.rs:2798`](wgpu/src/backend/webgpu.rs#L2798) | [`etalon/src/backend/webgpu.rs:2798`](etalon/src/backend/webgpu.rs#L2798) | This will leak if weak references is not supported. |
| `TODO` | [`wgpu/src/backend/webgpu.rs:2915`](wgpu/src/backend/webgpu.rs#L2915) | [`etalon/src/backend/webgpu.rs:2915`](etalon/src/backend/webgpu.rs#L2915) | actually send this down the error scope |
| `TODO` | [`wgpu/src/util/mod.rs:6`](wgpu/src/util/mod.rs#L6) | [`etalon/src/util/mod.rs:6`](etalon/src/util/mod.rs#L6) | For [`belt::StagingBelt`] to be available in `no_std` its usage of [`std::sync::mpsc`] |


### examples & benchmarks (11 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`benches/benches/wgpu-benchmark/computepass-bindless.wgsl:6`](benches/benches/wgpu-benchmark/computepass-bindless.wgsl#L6) | [`benches/benches/wgpu-benchmark/computepass-bindless.wgsl:6`](benches/benches/wgpu-benchmark/computepass-bindless.wgsl#L6) | The extra whitespace between the angle brackets is needed to workaround a parsing bug. *(https://github.com/gfx-rs/wgpu/issues/5765)* |
| `TODO` | [`benches/benches/wgpu-benchmark/computepass.rs:79`](benches/benches/wgpu-benchmark/computepass.rs#L79) | [`benches/benches/wgpu-benchmark/computepass.rs:79`](benches/benches/wgpu-benchmark/computepass.rs#L79) | as of writing llvmpipe segfaults the bindless benchmark on ci |
| `TODO` | [`benches/benches/wgpu-benchmark/computepass.rs:81`](benches/benches/wgpu-benchmark/computepass.rs#L81) | [`benches/benches/wgpu-benchmark/computepass.rs:81`](benches/benches/wgpu-benchmark/computepass.rs#L81) | Currently broken on Metal *(https://github.com/gfx-rs/wgpu/issues/9849)* |
| `TODO` | [`examples/features/src/msaa_line/mod.rs:230`](examples/features/src/msaa_line/mod.rs#L230) | [`examples/features/src/msaa_line/mod.rs:230`](examples/features/src/msaa_line/mod.rs#L230) | Switch back to full scans of possible options when we expose |
| `TODO` | [`examples/features/src/texture_arrays/mod.rs:123`](examples/features/src/texture_arrays/mod.rs#L123) | [`examples/features/src/texture_arrays/mod.rs:123`](examples/features/src/texture_arrays/mod.rs#L123) | Because naga's capabilities are evaluated on validate, not on write, we cannot make a shader module with unsupported |
| `TODO` | [`examples/features/src/water/water.wgsl:49`](examples/features/src/water/water.wgsl#L49) | [`examples/features/src/water/water.wgsl:49`](examples/features/src/water/water.wgsl#L49) | use the splat operations when available |
| `TODO` | [`examples/features/src/water/water.wgsl:130`](examples/features/src/water/water.wgsl#L130) | [`examples/features/src/water/water.wgsl:130`](examples/features/src/water/water.wgsl#L130) | we should be able to name them `sin` and `cos`. |
| `TODO` | [`examples/features/src/water/water.wgsl:159`](examples/features/src/water/water.wgsl#L159) | [`examples/features/src/water/water.wgsl:159`](examples/features/src/water/water.wgsl#L159) | use splats |
| `TODO` | [`examples/features/src/water/water.wgsl:175`](examples/features/src/water/water.wgsl#L175) | [`examples/features/src/water/water.wgsl:175`](examples/features/src/water/water.wgsl#L175) | use splat |
| `TODO` | [`examples/features/src/water/water.wgsl:214`](examples/features/src/water/water.wgsl#L214) | [`examples/features/src/water/water.wgsl:214`](examples/features/src/water/water.wgsl#L214) | use vector splats for division |
| `TODO` | [`examples/features/src/water/water.wgsl:259`](examples/features/src/water/water.wgsl#L259) | [`examples/features/src/water/water.wgsl:259`](examples/features/src/water/water.wgsl#L259) | splat for mix()? |


### naga (back/spv) (11 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/back/spv/block.rs:1150`](naga/src/back/spv/block.rs#L1150) | [`naga/src/back/spv/block.rs:1150`](naga/src/back/spv/block.rs#L1150) | **[Wave 3 Complete]** why not just rely on `Fadd` for matrices? |

| `RESOLVED` | [`naga/src/back/spv/block.rs:1522`](naga/src/back/spv/block.rs#L1522) | [`naga/src/back/spv/block.rs:1522`](naga/src/back/spv/block.rs#L1522) | **[Wave 3 Complete]** consider using integer dot product if VK_KHR_shader_integer_dot_product is available |

| `RESOLVED` | [`naga/src/back/spv/block.rs:2940`](naga/src/back/spv/block.rs#L2940) | [`naga/src/back/spv/block.rs:2940`](naga/src/back/spv/block.rs#L2940) | **[Wave 3 Complete]** support atomics? |

| `RESOLVED` | [`naga/src/back/spv/block.rs:3766`](naga/src/back/spv/block.rs#L3766) | [`naga/src/back/spv/block.rs:3766`](naga/src/back/spv/block.rs#L3766) | **[Wave 3 Complete]** the loop statement is begin with branch instruction, |

| `RESOLVED` | [`naga/src/back/spv/block.rs:4028`](naga/src/back/spv/block.rs#L4028) | [`naga/src/back/spv/block.rs:4028`](naga/src/back/spv/block.rs#L4028) | **[Wave 3 Complete]** SPIR-V doesn't have a atomic subtraction, |

| `RESOLVED` | [`naga/src/back/spv/helpers.rs:186`](naga/src/back/spv/helpers.rs#L186) | [`naga/src/back/spv/helpers.rs:186`](naga/src/back/spv/helpers.rs#L186) | **[Wave 3 Complete]** this is taken from std unstable, remove it when std's floor_char_boundary is stable |

| `RESOLVED` | [`naga/src/back/spv/instructions.rs:1140`](naga/src/back/spv/instructions.rs#L1140) | [`naga/src/back/spv/instructions.rs:1140`](naga/src/back/spv/instructions.rs#L1140) | **[Wave 3 Complete]** Branch Weights not implemented. |

| `RESOLVED` | [`naga/src/back/spv/mod.rs:950`](naga/src/back/spv/mod.rs#L950) | [`naga/src/back/spv/mod.rs:950`](naga/src/back/spv/mod.rs#L950) | **[Wave 3 Complete]** convert most of these into vectors, addressable by handle indices |

| `RESOLVED` | [`naga/src/back/spv/writer.rs:1816`](naga/src/back/spv/writer.rs#L1816) | [`naga/src/back/spv/writer.rs:1816`](naga/src/back/spv/writer.rs#L1816) | **[Wave 3 Complete]** Move to instructions module |

| `RESOLVED` | [`naga/src/back/spv/writer.rs:1852`](naga/src/back/spv/writer.rs#L1852) | [`naga/src/back/spv/writer.rs:1852`](naga/src/back/spv/writer.rs#L1852) | **[Wave 3 Complete]** Consider emitting EarlyAndLateFragmentTestsAMD here, if available. |

| `RESOLVED` | [`naga/src/back/spv/ray/query.rs:662`](naga/src/back/spv/ray/query.rs#L662) | [`naga/src/back/spv/ray/query.rs:662`](naga/src/back/spv/ray/query.rs#L662) | **[Wave 3 Complete]** perhaps this could be replaced with an OpPhi? |



### tests & CTS runner (11 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`tests/src/report.rs:27`](tests/src/report.rs#L27) | [`tests/src/report.rs:27`](tests/src/report.rs#L27) |  |
| `TODO` | [`tests/src/report.rs:63`](tests/src/report.rs#L63) | [`tests/src/report.rs:63`](tests/src/report.rs#L63) |  |
| `TODO` | [`tests/tests/wgpu-gpu/draw_index.rs:184`](tests/tests/wgpu-gpu/draw_index.rs#L184) | [`tests/tests/wgpu-gpu/draw_index.rs:184`](tests/tests/wgpu-gpu/draw_index.rs#L184) | when support for mesh shaders in naga on dx12/metal lands enable those backends |
| `TODO` | [`tests/tests/wgpu-gpu/encoder.rs:38`](tests/tests/wgpu-gpu/encoder.rs#L38) | [`tests/tests/wgpu-gpu/encoder.rs:38`](tests/tests/wgpu-gpu/encoder.rs#L38) | revisit this *(https://github.com/gfx-rs/wgpu/issues/7781)* |
| `TODO` | [`tests/tests/wgpu-gpu/naga_capabilities.rs:20`](tests/tests/wgpu-gpu/naga_capabilities.rs#L20) | [`tests/tests/wgpu-gpu/naga_capabilities.rs:20`](tests/tests/wgpu-gpu/naga_capabilities.rs#L20) | mask off *(https://github.com/gfx-rs/wgpu/issues/9849)* |
| `TODO` | [`tests/tests/wgpu-gpu/poll.rs:187`](tests/tests/wgpu-gpu/poll.rs#L187) | [`tests/tests/wgpu-gpu/poll.rs:187`](tests/tests/wgpu-gpu/poll.rs#L187) | `wgpu` should raise an error *(https://github.com/gfx-rs/wgpu/issues/9545)* |
| `TODO` | [`tests/tests/wgpu-gpu/primitive_index.rs:150`](tests/tests/wgpu-gpu/primitive_index.rs#L150) | [`tests/tests/wgpu-gpu/primitive_index.rs:150`](tests/tests/wgpu-gpu/primitive_index.rs#L150) | when naga support for MSL/HLSL mesh shaders lands enable Metal and DX12 |
| `TODO` | [`tests/tests/wgpu-gpu/shader/array_size_overrides.rs:69`](tests/tests/wgpu-gpu/shader/array_size_overrides.rs#L69) | [`tests/tests/wgpu-gpu/shader/array_size_overrides.rs:69`](tests/tests/wgpu-gpu/shader/array_size_overrides.rs#L69) | Restore this with the resolution for |
| `TODO` | [`tests/tests/wgpu-gpu/shader/zero_init_workgroup_mem.rs:173`](tests/tests/wgpu-gpu/shader/zero_init_workgroup_mem.rs#L173) | [`tests/tests/wgpu-gpu/shader/zero_init_workgroup_mem.rs:173`](tests/tests/wgpu-gpu/shader/zero_init_workgroup_mem.rs#L173) | use div_ceil once stabilized |
| `FIXME` | [`tests/tests/wgpu-validation/api/buffer_mapping.rs:1`](tests/tests/wgpu-validation/api/buffer_mapping.rs#L1) | [`tests/tests/wgpu-validation/api/buffer_mapping.rs:1`](tests/tests/wgpu-validation/api/buffer_mapping.rs#L1) | Now that MAP_WRITE mappings are write-only, |
| `FIXME` | [`tests/tests/wgpu-validation/api/buffer_mapping.rs:16`](tests/tests/wgpu-validation/api/buffer_mapping.rs#L16) | [`tests/tests/wgpu-validation/api/buffer_mapping.rs:16`](tests/tests/wgpu-validation/api/buffer_mapping.rs#L16) | Is the goal of these tests to ensure that zeroes are what is exposed to Rust, |


### etalon-types (11 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`wgpu-types/src/adapter.rs:142`](wgpu-types/src/adapter.rs#L142) | [`etalon-types/src/adapter.rs:142`](etalon-types/src/adapter.rs#L142) | **[Wave 1 Complete]** convert `vendor` and `device` to `u32` |

| `RESOLVED` | [`wgpu-types/src/counters.rs:139`](wgpu-types/src/counters.rs#L139) | [`etalon-types/src/counters.rs:139`](etalon-types/src/counters.rs#L139) | **[Wave 1 Complete]** #[cfg(features=)] |

| `RESOLVED` | [`wgpu-types/src/counters.rs:185`](wgpu-types/src/counters.rs#L185) | [`etalon-types/src/counters.rs:185`](etalon-types/src/counters.rs#L185) | **[Wave 1 Complete]** Rename to total_capacity_bytes following the rename at https://github.com/Traverse-Research/gpu-allocator/pull/266? |

| `RESOLVED` | [`wgpu-types/src/limits.rs:1334`](wgpu-types/src/limits.rs#L1334) | [`etalon-types/src/limits.rs:1334`](etalon-types/src/limits.rs#L1334) | **[Wave 1 Complete]** Fill out the differences between shader models more completely |

| `RESOLVED` | [`wgpu-types/src/origin_extent.rs:171`](wgpu-types/src/origin_extent.rs#L171) | [`etalon-types/src/origin_extent.rs:171`](etalon-types/src/origin_extent.rs#L171) | **[Wave 1 Complete]** It also does not *(<https://github.com/gfx-rs/wgpu/issues/8491>)* |

| `RESOLVED` | [`wgpu-types/src/write_only.rs:40`](wgpu-types/src/write_only.rs#L40) | [`etalon-types/src/write_only.rs:40`](etalon-types/src/write_only.rs#L40) | **[Wave 1 Complete]** Add an introduction to the necessity of explicit reborrowing. |

| `RESOLVED` | [`wgpu-types/src/write_only.rs:130`](wgpu-types/src/write_only.rs#L130) | [`etalon-types/src/write_only.rs:130`](etalon-types/src/write_only.rs#L130) | **[Wave 1 Complete]** Use `NonNull::from_mut()` when MSRV ≥ 1.89.0 |

| `RESOLVED` | [`wgpu-types/src/write_only.rs:244`](wgpu-types/src/write_only.rs#L244) | [`etalon-types/src/write_only.rs:244`](etalon-types/src/write_only.rs#L244) | **[Wave 1 Complete]** When `feature(slice_ptr_get)` <https://github.com/rust-lang/rust/issues/74265> |

| `RESOLVED` | [`wgpu-types/src/write_only.rs:809`](wgpu-types/src/write_only.rs#L809) | [`etalon-types/src/write_only.rs:809`](etalon-types/src/write_only.rs#L809) | **[Wave 1 Complete]** cleaner panic messages |

| `RESOLVED` | [`wgpu-types/src/write_only.rs:1033`](wgpu-types/src/write_only.rs#L1033) | [`etalon-types/src/write_only.rs:1033`](etalon-types/src/write_only.rs#L1033) | **[Wave 1 Complete]** have more consistent errors so this assertion can be stronger |

| `RESOLVED` | [`wgpu-types/src/texture/format.rs:683`](wgpu-types/src/texture/format.rs#L683) | [`etalon-types/src/texture/format.rs:683`](etalon-types/src/texture/format.rs#L683) | **[Wave 1 Complete]** Remove this note. *(<https://github.com/gfx-rs/wgpu/issues/8491>)* |



### naga (back/hlsl) (10 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/back/hlsl/help.rs:1351`](naga/src/back/hlsl/help.rs#L1351) | [`naga/src/back/hlsl/help.rs:1351`](naga/src/back/hlsl/help.rs#L1351) | **[Wave 3 Complete]** we could merge this with iteration in write_wrapped_expression_functions... |

| `RESOLVED` | [`naga/src/back/hlsl/help.rs:1506`](naga/src/back/hlsl/help.rs#L1506) | [`naga/src/back/hlsl/help.rs:1506`](naga/src/back/hlsl/help.rs#L1506) | **[Wave 3 Complete]** asint()/asuint() only support 32-bit integers, so we *(#7109)* |

| `RESOLVED` | [`naga/src/back/hlsl/help.rs:1567`](naga/src/back/hlsl/help.rs#L1567) | [`naga/src/back/hlsl/help.rs:1567`](naga/src/back/hlsl/help.rs#L1567) | **[Wave 3 Complete]** asint()/asuint() only support 32-bit integers, so we must *(#7109)* |

| `RESOLVED` | [`naga/src/back/hlsl/mod.rs:666`](naga/src/back/hlsl/mod.rs#L666) | [`naga/src/back/hlsl/mod.rs:666`](naga/src/back/hlsl/mod.rs#L666) | **[Wave 3 Complete]** Error used only during development |

| `RESOLVED` | [`naga/src/back/hlsl/writer.rs:610`](naga/src/back/hlsl/writer.rs#L610) | [`naga/src/back/hlsl/writer.rs:610`](naga/src/back/hlsl/writer.rs#L610) | **[Wave 3 Complete]** we could force fragment outputs to always go through `entry_point_io.output` path |

| `RESOLVED` | [`naga/src/back/hlsl/writer.rs:1939`](naga/src/back/hlsl/writer.rs#L1939) | [`naga/src/back/hlsl/writer.rs:1939`](naga/src/back/hlsl/writer.rs#L1939) | **[Wave 3 Complete]** can we narrow this down to just `in` when possible? |

| `RESOLVED` | [`naga/src/back/hlsl/writer.rs:2223`](naga/src/back/hlsl/writer.rs#L2223) | [`naga/src/back/hlsl/writer.rs:2223`](naga/src/back/hlsl/writer.rs#L2223) | **[Wave 3 Complete]** copy-paste from glsl-out |

| `RESOLVED` | [`naga/src/back/hlsl/writer.rs:2233`](naga/src/back/hlsl/writer.rs#L2233) | [`naga/src/back/hlsl/writer.rs:2233`](naga/src/back/hlsl/writer.rs#L2233) | **[Wave 3 Complete]** copy-paste from glsl-out |

| `RESOLVED` | [`naga/src/back/hlsl/writer.rs:2263`](naga/src/back/hlsl/writer.rs#L2263) | [`naga/src/back/hlsl/writer.rs:2263`](naga/src/back/hlsl/writer.rs#L2263) | **[Wave 3 Complete]** copy-paste from glsl-out |

| `RESOLVED` | [`naga/src/back/hlsl/writer.rs:3796`](naga/src/back/hlsl/writer.rs#L3796) | [`naga/src/back/hlsl/writer.rs:3796`](naga/src/back/hlsl/writer.rs#L3796) | **[Wave 3 Complete]** This relies on the asint()/asuint() functions which only work *(#7109)* |



### naga (back/msl) (9 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/back/msl/mod.rs:689`](naga/src/back/msl/mod.rs#L689) | [`naga/src/back/msl/mod.rs:689`](naga/src/back/msl/mod.rs#L689) | **[Wave 3 Complete]** figure out how to make this written as a function call |

| `RESOLVED` | [`naga/src/back/msl/ray.rs:236`](naga/src/back/msl/ray.rs#L236) | [`naga/src/back/msl/ray.rs:236`](naga/src/back/msl/ray.rs#L236) | **[Wave 3 Complete]** check for misuse. |

| `RESOLVED` | [`naga/src/back/msl/ray.rs:242`](naga/src/back/msl/ray.rs#L242) | [`naga/src/back/msl/ray.rs:242`](naga/src/back/msl/ray.rs#L242) | **[Wave 3 Complete]** how to deal with winding? Is it by default the same as the other APIs? |

| `RESOLVED` | [`naga/src/back/msl/writer.rs:1326`](naga/src/back/msl/writer.rs#L1326) | [`naga/src/back/msl/writer.rs:1326`](naga/src/back/msl/writer.rs#L1326) | **[Wave 3 Complete]** do we support Zero on `Sampled` image classes? |

| `RESOLVED` | [`naga/src/back/msl/writer.rs:2418`](naga/src/back/msl/writer.rs#L2418) | [`naga/src/back/msl/writer.rs:2418`](naga/src/back/msl/writer.rs#L2418) | **[Wave 3 Complete]** handle undefined behavior of BinaryOperator::Modulo |

| `RESOLVED` | [`naga/src/back/msl/writer.rs:3640`](naga/src/back/msl/writer.rs#L3640) | [`naga/src/back/msl/writer.rs:3640`](naga/src/back/msl/writer.rs#L3640) | **[Wave 3 Complete]** we are forcefully deduplicating the expression here |

| `RESOLVED` | [`naga/src/back/msl/writer.rs:3832`](naga/src/back/msl/writer.rs#L3832) | [`naga/src/back/msl/writer.rs:3832`](naga/src/back/msl/writer.rs#L3832) | **[Wave 3 Complete]** figure out the naming scheme that wouldn't collide with user names. |

| `RESOLVED` | [`naga/src/back/msl/writer.rs:4488`](naga/src/back/msl/writer.rs#L4488) | [`naga/src/back/msl/writer.rs:4488`](naga/src/back/msl/writer.rs#L4488) | **[Wave 3 Complete]** take care of loop/continuing? |

| `RESOLVED` | [`naga/src/back/msl/writer.rs:8329`](naga/src/back/msl/writer.rs#L8329) | [`naga/src/back/msl/writer.rs:8329`](naga/src/back/msl/writer.rs#L8329) | **[Wave 3 Complete]** we can postpone this till the relevant expressions are emitted |



### etalon-hal (9 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `XXX` | [`wgpu-hal/examples/raw-gles.rs:236`](wgpu-hal/examples/raw-gles.rs#L236) | [`etalon-hal/examples/raw-gles.rs:236`](etalon-hal/examples/raw-gles.rs#L236) | On WGL this should only be called after the context was |
| `TODO` | [`wgpu-hal/src/lib.rs:260`](wgpu-hal/src/lib.rs#L260) | [`etalon-hal/src/lib.rs:260`](etalon-hal/src/lib.rs#L260) | this should have a cfg *(https://github.com/gfx-rs/wgpu/issues/7120)* |
| `TODO` | [`wgpu-hal/src/lib.rs:466`](wgpu-hal/src/lib.rs#L466) | [`etalon-hal/src/lib.rs:466`](etalon-hal/src/lib.rs#L466) | the allocator's configuration should take hardware capability into |
| `TODO` | [`wgpu-hal/src/lib.rs:486`](wgpu-hal/src/lib.rs#L486) | [`etalon-hal/src/lib.rs:486`](etalon-hal/src/lib.rs#L486) | [https://github.com/gfx-rs/wgpu/issues/8625](https://github.com/gfx-rs/wgpu/issues/8625) |
| `TODO` | [`wgpu-hal/src/lib.rs:613`](wgpu-hal/src/lib.rs#L613) | [`etalon-hal/src/lib.rs:613`](etalon-hal/src/lib.rs#L613) | avoid indirection via Box once arbitrary types support unsized coercion *(https://github.com/rust-lang/rust/issues/18598)* |
| `TODO` | [`wgpu-hal/src/dynamic/command.rs:713`](wgpu-hal/src/dynamic/command.rs#L713) | [`etalon-hal/src/dynamic/command.rs:713`](etalon-hal/src/dynamic/command.rs#L713) | API should be redesigned to avoid this and other descriptor copies that happen due to the dyn api. |
| `TODO` | [`wgpu-hal/src/noop/command.rs:110`](wgpu-hal/src/noop/command.rs#L110) | [`etalon-hal/src/noop/command.rs:110`](etalon-hal/src/noop/command.rs#L110) | consider implementing this and other texture manipulation |
| `TODO` | [`wgpu-hal/src/noop/command.rs:114`](wgpu-hal/src/noop/command.rs#L114) | [`etalon-hal/src/noop/command.rs:114`](etalon-hal/src/noop/command.rs#L114) | consider implementing this and other texture manipulation |
| `TODO` | [`wgpu-hal/src/noop/command.rs:124`](wgpu-hal/src/noop/command.rs#L124) | [`etalon-hal/src/noop/command.rs:124`](etalon-hal/src/noop/command.rs#L124) | consider implementing this and other texture manipulation |


### etalon-core (8 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`wgpu-core/Cargo.toml:43`](wgpu-core/Cargo.toml#L43) | [`etalon-core/Cargo.toml:43`](etalon-core/Cargo.toml#L43) | "std" is a default feature for *(https://github.com/gfx-rs/wgpu/issues/6826)* |
| `TODO` | [`wgpu-core/Cargo.toml:166`](wgpu-core/Cargo.toml#L166) | [`etalon-core/Cargo.toml:166`](etalon-core/Cargo.toml#L166) | there should be a hal feature *(https://github.com/gfx-rs/wgpu/issues/7120)* |
| `RESOLVED` | [`wgpu-core/src/binding_model.rs:162`](wgpu-core/src/binding_model.rs#L162) | [`etalon-core/src/binding_model.rs:162`](etalon-core/src/binding_model.rs#L162) | **[Wave 1 Complete]** there may be additional variants here that can be extracted into |

| `RESOLVED` | [`wgpu-core/src/instance.rs:63`](wgpu-core/src/instance.rs#L63) | [`etalon-core/src/instance.rs:63`](etalon-core/src/instance.rs#L63) | **[Wave 1 Complete]** Should expose timeout to poll_all. *(#8286)* |

| `RESOLVED` | [`wgpu-core/src/instance.rs:149`](wgpu-core/src/instance.rs#L149) | [`etalon-core/src/instance.rs:149`](etalon-core/src/instance.rs#L149) | **[Wave 1 Complete]** Extract display from HAL instance if available? |

| `RESOLVED` | [`wgpu-core/src/instance.rs:710`](wgpu-core/src/instance.rs#L710) | [`etalon-core/src/instance.rs:710`](etalon-core/src/instance.rs#L710) | **[Wave 1 Complete]** We must take ownership of the field here, without being able to pass it into |

| `RESOLVED` | [`wgpu-core/src/pipeline_cache.rs:15`](wgpu-core/src/pipeline_cache.rs#L15) | [`etalon-core/src/pipeline_cache.rs:15`](etalon-core/src/pipeline_cache.rs#L15) | **[Wave 1 Complete]** Is it plausible that this would happen |

| `RESOLVED` | [`wgpu-core/src/resource.rs:943`](wgpu-core/src/resource.rs#L943) | [`etalon-core/src/resource.rs:943`](etalon-core/src/resource.rs#L943) | **[Wave 1 Complete]** we are ignoring the transition *(https://github.com/gfx-rs/wgpu/issues/9306)* |



### naga (back/glsl) (7 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/back/glsl/conv.rs:134`](naga/src/back/glsl/conv.rs#L134) | [`naga/src/back/glsl/conv.rs:134`](naga/src/back/glsl/conv.rs#L134) | **[Wave 3 Complete]** figure out how to map these to glsl things as glsl treats them as arrays |

| `RESOLVED` | [`naga/src/back/glsl/mod.rs:123`](naga/src/back/glsl/mod.rs#L123) | [`naga/src/back/glsl/mod.rs:123`](naga/src/back/glsl/mod.rs#L123) | **[Wave 3 Complete]**  |

| `RESOLVED` | [`naga/src/back/glsl/writer.rs:149`](naga/src/back/glsl/writer.rs#L149) | [`naga/src/back/glsl/writer.rs:149`](naga/src/back/glsl/writer.rs#L149) | **[Wave 3 Complete]** Should this be user configurable? |

| `RESOLVED` | [`naga/src/back/glsl/writer.rs:2424`](naga/src/back/glsl/writer.rs#L2424) | [`naga/src/back/glsl/writer.rs:2424`](naga/src/back/glsl/writer.rs#L2424) | **[Wave 3 Complete]** *(https://github.com/gfx-rs/wgpu/issues/9456)* |

| `RESOLVED` | [`naga/src/back/glsl/writer.rs:2621`](naga/src/back/glsl/writer.rs#L2621) | [`naga/src/back/glsl/writer.rs:2621`](naga/src/back/glsl/writer.rs#L2621) | **[Wave 3 Complete]** handle clamp_to_edge |

| `RESOLVED` | [`naga/src/back/glsl/writer.rs:3039`](naga/src/back/glsl/writer.rs#L3039) | [`naga/src/back/glsl/writer.rs:3039`](naga/src/back/glsl/writer.rs#L3039) | **[Wave 3 Complete]** Boolean mix on desktop required GL_EXT_shader_integer_mix |

| `RESOLVED` | [`naga/src/back/glsl/writer.rs:4141`](naga/src/back/glsl/writer.rs#L4141) | [`naga/src/back/glsl/writer.rs:4141`](naga/src/back/glsl/writer.rs#L4141) | **[Wave 3 Complete]** Is there even a function for this? |



### etalon-core (command) (7 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`wgpu-core/src/command/clear.rs:424`](wgpu-core/src/command/clear.rs#L424) | [`etalon-core/src/command/clear.rs:424`](etalon-core/src/command/clear.rs#L424) | Currently COPY_DST for NV12 and P010 textures is unsupported. |
| `TODO` | [`wgpu-core/src/command/clear.rs:467`](wgpu-core/src/command/clear.rs#L467) | [`etalon-core/src/command/clear.rs:467`](etalon-core/src/command/clear.rs#L467) | Only doing one layer at a time for volume textures right now. |
| `TODO` | [`wgpu-core/src/command/memory_init.rs:391`](wgpu-core/src/command/memory_init.rs#L391) | [`etalon-core/src/command/memory_init.rs:391`](etalon-core/src/command/memory_init.rs#L391) | Could we attempt some range collapsing here? |
| `TODO` | [`wgpu-core/src/command/mod.rs:2011`](wgpu-core/src/command/mod.rs#L2011) | [`etalon-core/src/command/mod.rs:2011`](etalon-core/src/command/mod.rs#L2011) | Extract out the 2 error variants below so that we can always |
| `TODO` | [`wgpu-core/src/command/render.rs:2305`](wgpu-core/src/command/render.rs#L2305) | [`etalon-core/src/command/render.rs:2305`](etalon-core/src/command/render.rs#L2305) | by restructuring the code, we could get rid of some of this Arc clone. *(wumpf)* |
| `TODO` | [`wgpu-core/src/command/render_command.rs:38`](wgpu-core/src/command/render_command.rs#L38) | [`etalon-core/src/command/render_command.rs:38`](etalon-core/src/command/render_command.rs#L38) | use half-float to reduce the size? |
| `TODO` | [`wgpu-core/src/command/transfer.rs:1595`](wgpu-core/src/command/transfer.rs#L1595) | [`etalon-core/src/command/transfer.rs:1595`](etalon-core/src/command/transfer.rs#L1595) | try to avoid this the collection. It's needed because both |


### naga (back/wgsl) (6 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/back/wgsl/mod.rs:26`](naga/src/back/wgsl/mod.rs#L26) | [`naga/src/back/wgsl/mod.rs:26`](naga/src/back/wgsl/mod.rs#L26) | **[Wave 3 Complete]** Error used only during development |

| `RESOLVED` | [`naga/src/back/wgsl/writer.rs:791`](naga/src/back/wgsl/writer.rs#L791) | [`naga/src/back/wgsl/writer.rs:791`](naga/src/back/wgsl/writer.rs#L791) | **[Wave 3 Complete]** copy-paste from glsl-out |

| `RESOLVED` | [`naga/src/back/wgsl/writer.rs:831`](naga/src/back/wgsl/writer.rs#L831) | [`naga/src/back/wgsl/writer.rs:831`](naga/src/back/wgsl/writer.rs#L831) | **[Wave 3 Complete]** copy-paste from glsl-out |

| `RESOLVED` | [`naga/src/back/wgsl/writer.rs:929`](naga/src/back/wgsl/writer.rs#L929) | [`naga/src/back/wgsl/writer.rs:929`](naga/src/back/wgsl/writer.rs#L929) | **[Wave 3 Complete]** Obey named expressions here. |

| `RESOLVED` | [`naga/src/back/wgsl/writer.rs:956`](naga/src/back/wgsl/writer.rs#L956) | [`naga/src/back/wgsl/writer.rs:956`](naga/src/back/wgsl/writer.rs#L956) | **[Wave 3 Complete]** copy-paste from glsl-out |

| `RESOLVED` | [`naga/src/back/wgsl/writer.rs:980`](naga/src/back/wgsl/writer.rs#L980) | [`naga/src/back/wgsl/writer.rs:980`](naga/src/back/wgsl/writer.rs#L980) | **[Wave 3 Complete]** we could do the same workaround as we did for the HLSL backend |



### etalon-core (device) (6 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`wgpu-core/src/device/queue.rs:720`](wgpu-core/src/device/queue.rs#L720) | [`etalon-core/src/device/queue.rs:720`](etalon-core/src/device/queue.rs#L720) | move out common parts of write_xxx. |
| `TODO` | [`wgpu-core/src/device/queue.rs:2018`](wgpu-core/src/device/queue.rs#L2018) | [`etalon-core/src/device/queue.rs:2018`](etalon-core/src/device/queue.rs#L2018) | Tracing |
| `TODO` | [`wgpu-core/src/device/queue.rs:2173`](wgpu-core/src/device/queue.rs#L2173) | [`etalon-core/src/device/queue.rs:2173`](etalon-core/src/device/queue.rs#L2173) | Optimize this. *(<https://github.com/gfx-rs/wgpu/issues/8510>)* |
| `TODO` | [`wgpu-core/src/device/resource.rs:1937`](wgpu-core/src/device/resource.rs#L1937) | [`etalon-core/src/device/resource.rs:1937`](etalon-core/src/device/resource.rs#L1937) | fix *(https://github.com/gfx-rs/wgpu/issues/8491)* |
| `TODO` | [`wgpu-core/src/device/resource.rs:2495`](wgpu-core/src/device/resource.rs#L2495) | [`etalon-core/src/device/resource.rs:2495`](etalon-core/src/device/resource.rs#L2495) | check for wgt::DownlevelFlags::COMPARISON_SAMPLERS |
| `TODO` | [`wgpu-core/src/device/resource.rs:2960`](wgpu-core/src/device/resource.rs#L2960) | [`etalon-core/src/device/resource.rs:2960`](etalon-core/src/device/resource.rs#L2960) | should this be combined with `get_introspection_bind_group_layouts` in some way? |


### naga (valid/mod.rs) (4 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/valid/mod.rs:25`](naga/src/valid/mod.rs#L25) | [`naga/src/valid/mod.rs:25`](naga/src/valid/mod.rs#L25) | **[Wave 2 Complete]** analyze the model at the same time as we validate it, |

| `RESOLVED` | [`naga/src/valid/mod.rs:648`](naga/src/valid/mod.rs#L648) | [`naga/src/valid/mod.rs:648`](naga/src/valid/mod.rs#L648) | **[Wave 2 Complete]** Consider removing this *(https://github.com/gfx-rs/wgpu/issues/8207)* |

| `RESOLVED` | [`naga/src/valid/mod.rs:654`](naga/src/valid/mod.rs#L654) | [`naga/src/valid/mod.rs:654`](naga/src/valid/mod.rs#L654) | **[Wave 2 Complete]** Consider removing this *(https://github.com/gfx-rs/wgpu/issues/8207)* |

| `RESOLVED` | [`naga/src/valid/mod.rs:904`](naga/src/valid/mod.rs#L904) | [`naga/src/valid/mod.rs:904`](naga/src/valid/mod.rs#L904) | **[Wave 2 Complete]** keep some EP span information? |



### naga (front/wgsl) (3 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/front/wgsl/lower/mod.rs:1657`](naga/src/front/wgsl/lower/mod.rs#L1657) | [`naga/src/front/wgsl/lower/mod.rs:1657`](naga/src/front/wgsl/lower/mod.rs#L1657) | **[Wave 2 Complete]** replace with try_map once stabilized |

| `RESOLVED` | [`naga/src/front/wgsl/lower/mod.rs:2733`](naga/src/front/wgsl/lower/mod.rs#L2733) | [`naga/src/front/wgsl/lower/mod.rs:2733`](naga/src/front/wgsl/lower/mod.rs#L2733) | **[Wave 2 Complete]** We shouldn't ignore the *(https://github.com/gfx-rs/wgpu/issues/8440)* |

| `RESOLVED` | [`naga/src/front/wgsl/parse/number.rs:386`](naga/src/front/wgsl/parse/number.rs#L386) | [`naga/src/front/wgsl/parse/number.rs:386`](naga/src/front/wgsl/parse/number.rs#L386) | **[Wave 2 Complete]** f16 is not supported |



### naga (ir/mod.rs) (3 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/ir/mod.rs:977`](naga/src/ir/mod.rs#L977) | [`naga/src/ir/mod.rs:977`](naga/src/ir/mod.rs#L977) | **[Wave 2 Complete]** should this be unaligned? |

| `RESOLVED` | [`naga/src/ir/mod.rs:984`](naga/src/ir/mod.rs#L984) | [`naga/src/ir/mod.rs:984`](naga/src/ir/mod.rs#L984) | **[Wave 2 Complete]** consider moving `multisampled: bool` out |

| `RESOLVED` | [`naga/src/ir/mod.rs:2022`](naga/src/ir/mod.rs#L2022) | [`naga/src/ir/mod.rs:2022`](naga/src/ir/mod.rs#L2022) | **[Wave 2 Complete]** consider removing `Clone`. It's not valid to clone `Statement::Emit` anyway. |



### naga (proc/constant_evaluator.rs) (3 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/proc/constant_evaluator.rs:349`](naga/src/proc/constant_evaluator.rs#L349) | [`naga/src/proc/constant_evaluator.rs:349`](naga/src/proc/constant_evaluator.rs#L349) | **[Wave 2 Complete]** should a vector of i32 be constructible from abstract int? |

| `RESOLVED` | [`naga/src/proc/constant_evaluator.rs:1521`](naga/src/proc/constant_evaluator.rs#L1521) | [`naga/src/proc/constant_evaluator.rs:1521`](naga/src/proc/constant_evaluator.rs#L1521) | **[Wave 2 Complete]** just re-use the expression, ezpz |

| `RESOLVED` | [`naga/src/proc/constant_evaluator.rs:1663`](naga/src/proc/constant_evaluator.rs#L1663) | [`naga/src/proc/constant_evaluator.rs:1663`](naga/src/proc/constant_evaluator.rs#L1663) | **[Wave 2 Complete]** `round_ties_even` is not available on `half::f16` yet. |



### naga (valid/function.rs) (2 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/valid/function.rs:788`](naga/src/valid/function.rs#L788) | [`naga/src/valid/function.rs:788`](naga/src/valid/function.rs#L788) | **[Wave 2 Complete]** *(https://github.com/gfx-rs/wgpu/issues/9456)* |

| `RESOLVED` | [`naga/src/valid/function.rs:1789`](naga/src/valid/function.rs#L1789) | [`naga/src/valid/function.rs:1789`](naga/src/valid/function.rs#L1789) | **[Wave 2 Complete]** the format string itself is not validated, so a mismatch between the |



### naga (valid/handles.rs) (2 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/valid/handles.rs:305`](naga/src/valid/handles.rs#L305) | [`naga/src/valid/handles.rs:305`](naga/src/valid/handles.rs#L305) | **[Wave 2 Complete]** what errors should this be? |

| `RESOLVED` | [`naga/src/valid/handles.rs:316`](naga/src/valid/handles.rs#L316) | [`naga/src/valid/handles.rs:316`](naga/src/valid/handles.rs#L316) | **[Wave 2 Complete]** internal error ? We should never get here. |



### other (player) (2 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`player/src/bin/play.rs:32`](player/src/bin/play.rs#L32) | [`player/src/bin/play.rs:32`](player/src/bin/play.rs#L32) | setting for the backend bits |
| `TODO` | [`player/src/bin/play.rs:33`](player/src/bin/play.rs#L33) | [`player/src/bin/play.rs:33`](player/src/bin/play.rs#L33) | setting for the target frame, or controls |


### etalon-core (track) (2 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`wgpu-core/src/track/range.rs:2`](wgpu-core/src/track/range.rs#L2) | [`etalon-core/src/track/range.rs:2`](etalon-core/src/track/range.rs#L2) | **[Wave 1 Complete]** consider getting rid of it. |

| `RESOLVED` | [`wgpu-core/src/track/range.rs:157`](wgpu-core/src/track/range.rs#L157) | [`etalon-core/src/track/range.rs:157`](etalon-core/src/track/range.rs#L157) | **[Wave 1 Complete]** randomized/fuzzy testing |



### etalon-hal (auxil) (2 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`wgpu-hal/src/auxil/mod.rs:174`](wgpu-hal/src/auxil/mod.rs#L174) | [`etalon-hal/src/auxil/mod.rs:174`](etalon-hal/src/auxil/mod.rs#L174) | Remove this when compat mode is implemented, see |
| `TODO` | [`wgpu-hal/src/auxil/renderdoc.rs:34`](wgpu-hal/src/auxil/renderdoc.rs#L34) | [`etalon-hal/src/auxil/renderdoc.rs:34`](etalon-hal/src/auxil/renderdoc.rs#L34) | replace with libloading API once supported |


### other (clippy.toml) (1 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`clippy.toml:12`](clippy.toml#L12) | [`clippy.toml:12`](clippy.toml#L12) | **[Wave 1 Complete]** Consider reducing this further *(https://github.com/gfx-rs/wgpu/issues/9456)* |



### naga (back/mod.rs) (1 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/back/mod.rs:307`](naga/src/back/mod.rs#L307) | [`naga/src/back/mod.rs:307`](naga/src/back/mod.rs#L307) | **[Wave 3 Complete]** We need a better fix for named `Load` expressions |



### naga (back/dot) (1 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/back/dot/mod.rs:94`](naga/src/back/dot/mod.rs#L94) | [`naga/src/back/dot/mod.rs:94`](naga/src/back/dot/mod.rs#L94) | **[Wave 3 Complete]** link to the beginning |



### naga (valid/analyzer.rs) (1 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/valid/analyzer.rs:910`](naga/src/valid/analyzer.rs#L910) | [`naga/src/valid/analyzer.rs:910`](naga/src/valid/analyzer.rs#L910) | **[Wave 2 Complete]** Yes, this isn't contextualized with source, because |



### naga (valid/expression.rs) (1 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/valid/expression.rs:416`](naga/src/valid/expression.rs#L416) | [`naga/src/valid/expression.rs:416`](naga/src/valid/expression.rs#L416) | **[Wave 2 Complete]** only allow one of these |



### naga (valid/interface.rs) (1 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/valid/interface.rs:1386`](naga/src/valid/interface.rs#L1386) | [`naga/src/valid/interface.rs:1386`](naga/src/valid/interface.rs#L1386) | **[Wave 2 Complete]** add span info to function arguments |



### naga (valid/type.rs) (1 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`naga/src/valid/type.rs:753`](naga/src/valid/type.rs#L753) | [`naga/src/valid/type.rs:753`](naga/src/valid/type.rs#L753) | **[Wave 2 Complete]** this could be nicer. We want to allow some structures |



### other (naga-cli) (1 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `TODO` | [`naga-cli/src/bin/naga.rs:507`](naga-cli/src/bin/naga.rs#L507) | [`naga-cli/src/bin/naga.rs:507`](naga-cli/src/bin/naga.rs#L507) | read the parameters from RON? |


### etalon-core (lock) (1 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`wgpu-core/src/lock/rank.rs:26`](wgpu-core/src/lock/rank.rs#L26) | [`etalon-core/src/lock/rank.rs:26`](etalon-core/src/lock/rank.rs#L26) | **[Wave 1 Complete]** Resolve invalid *(<https://github.com/gfx-rs/wgpu/issues/5572>)* |



### etalon-sync (1 items)

| Kind | Upstream Path (`wgpu-trunk`) | Etalon Target Path | Description |
| :--- | :--- | :--- | :--- |
| `RESOLVED` | [`wgpu-sync/src/lib.rs:31`](wgpu-sync/src/lib.rs#L31) | [`etalon-sync/src/lib.rs:31`](etalon-sync/src/lib.rs#L31) | **[Wave 1 Complete]**  |


