# Change Log

<!--
Please add your PR to the changelog! Choose from a top level and bottom
level category, then write your changes like follows:

- Describe your change in a user friendly format. By @yourslug in [#99999](https://github.com/mediaprophet/etalon/pull/99999)

You can add additional user facing information if it's a major breaking change. You can use the following to help:

```diff
- Old code
+ New code
```

Top level categories:

- Major changes
- Added/New Features
- Changes
- Bug Fixes (that don't change API)
- Performance
- Documentation
- Dependency Updates
- deno_webgpu
- Examples
- Testing/Internal

Bottom level categories:

- General
- naga
- Validation
- DX12
- Vulkan
- Metal
- GLES / OpenGL
- WebGPU
- Emscripten
- Hal
-->

This changelog covers Etalon only. Etalon was forked from [wgpu](https://github.com/gfx-rs/wgpu) 30. For the history of changes made before the fork, see the [upstream wgpu changelog](https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md).

## Unreleased

### Bug Fixes

#### General

- `etalon-render`: EMF pipeline entry points now match the shader's `vertex_main` and `fragment_main`. (4e77ca3f7)

### Documentation

- README: upstream attribution to [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu), corrected the fork's version wording, a git-dependency quickstart matching the current API, and the homepage.
- `upstream-licenses/`: restored the upstream MIT copyright notice (`The gfx-rs developers`) and clarified which material each license covers.
- Workspace `homepage` is now <https://mediaprophet.github.io/etalon/>.
- This changelog now records Etalon changes only. Upstream history is linked above rather than copied.

## v0.0.1 (2026-10-21)

First Etalon release, tagged `v0.0.1`. Forked from wgpu 30.

### Major changes

- Renamed the `wgpu` crates, modules and documentation to `etalon` (`etalon`, `etalon-core`, `etalon-hal`, `etalon-types`, `etalon-core-remote`, `etalon-core-remote-types`, `etalon-sync`, `etalon-naga-bridge`, `etalon-info`). `naga` and `naga-types` keep their names. (5e1908e38 and earlier)
- Set every workspace package to version `0.0.1` with `publish = false`. Etalon is consumed through git, not crates.io. (f20f5abaf, ed0b671d8)
- Licensing: Etalon modifications and additions are CC BY-NC-ND 4.0 (`LICENSE`, `LICENSE-QUALIA`). Upstream wgpu license texts are kept in `upstream-licenses/`. (bcfb954fb, 29df9cf5e)

### Added/New Features

- Extension crates under `crates/`: `etalon-agent`, `etalon-nquin`, `etalon-render`, `etalon-tensor`, `etalon-wasm` and `etalon-wgsl-forge`. (e1193443a)
- Implementation work across the foundation/types/sync, naga IR/frontends/validation, naga backends, `etalon-core`, `etalon-hal`, and surface/deno/examples layers. (9ebe40ef3, 2aa8295a5, baf80e4c5, 38c7237d2, e6ac82c3d, 38723252e)

### Bug Fixes

#### WebGPU

- `Device::pop_error_scope` no longer panics on `wasm32` when the error scope caught no error and the browser resolves it to `null` or `undefined`. Unexpected JavaScript errors now surface as `Error::Internal` instead of a panic. (bcfb954fb)
- `request_device` omits `requiredLimits` keys that the adapter does not expose. This avoids `OperationError` when browsers remove limits such as `maxInterStageShaderComponents`. (bcfb954fb)

#### Vulkan

- `BFLOAT16_KHR` is matched by its raw discriminant (`1000141000`). (dfbaad4c3)

#### Hal

- The `static-dxc` feature is stubbed to remove the dependency on the unmaintained `mach-dxcompiler-rs`. (7957b224a)

### Dependency Updates

- Workspace dependencies harmonized with consuming projects. (0c9e6c466)
