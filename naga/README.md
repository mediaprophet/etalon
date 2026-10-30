# Naga

The shader translation library for the needs of [etalon](https://github.com/mediaprophet/etalon). Derived from upstream [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga), part of [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu).

## Supported end-points

| Front-end       | Status             | Feature | Notes                               |
| --------------- | ------------------ | ------- | ----------------------------------- |
| SPIR-V (binary) | :white_check_mark: | spv-in  |                                     |
| WGSL            | :white_check_mark: | wgsl-in | Fully validated                     |
| GLSL            | :ok:               | glsl-in | GLSL 440+ and Vulkan semantics only |

| Back-end       | Status             | Feature  | Notes                           |
| -------------- | ------------------ | -------- | ------------------------------- |
| SPIR-V         | :white_check_mark: | spv-out  |                                 |
| WGSL           | :ok:               | wgsl-out |                                 |
| Metal          | :white_check_mark: | msl-out  |                                 |
| HLSL           | :white_check_mark: | hlsl-out | Shader Model 5.0+ (DirectX 11+) |
| GLSL           | :ok:               | glsl-out | GLSL 330+ and GLSL ES 300+      |
| AIR            |                    |          |                                 |
| DXIL/DXIR      |                    |          |                                 |
| DXBC           |                    |          |                                 |
| DOT (GraphViz) | :ok:               | dot-out  | Not a shading language          |

:white_check_mark: = Primary support — :ok: = Secondary support — :construction: = Unsupported, but support in progress

## Conversion tool

Naga can be used as a CLI, which allows testing the conversion of different code paths.

First, install `naga-cli` from crates.io or directly from GitHub.

```bash
cargo install naga-cli --git https://github.com/mediaprophet/etalon.git
```

Then, you can run `naga` command.

```bash
naga my_shader.wgsl # validate only
naga my_shader.spv my_shader.txt # dump the IR module into a file
naga my_shader.spv my_shader.metal --flow-dir flow-dir # convert the SPV to Metal, also dump the SPIR-V flow graph to `flow-dir`
naga my_shader.wgsl my_shader.vert --profile es310 # convert the WGSL to GLSL vertex stage under ES 3.20 profile
```

As naga includes a default binary target, you can also use `cargo run` without installation. This is useful when you develop naga itself or investigate the behavior of naga at a specific commit (e.g. [etalon](https://github.com/mediaprophet/etalon) might pin a different version of naga than the `HEAD` of this repository).

```bash
cargo run my_shader.wgsl
```

## Development workflow

The main instrument aiding the development is the good old `cargo test --all-features --workspace`,
which will run the unit tests and also update all the snapshots. You'll see these
changes in git before committing the code.

If working on a particular front-end or back-end, it may be convenient to
enable the relevant features in `Cargo.toml`, e.g.

```toml
default = ["spv-out"] #TEMP!
```

This allows IDE basic checks to report errors there unless your IDE is sufficiently configurable already.

Finally, when changes to the snapshots are made, we should verify that the produced shaders
are indeed valid for the target platforms they are compiled for:

```bash
cargo xtask validate spv # for Vulkan shaders, requires SPIRV-Tools installed
cargo xtask validate msl # for Metal shaders, requires XCode command-line tools installed
cargo xtask validate glsl # for OpenGL shaders, requires GLSLang installed
cargo xtask validate dot # for dot files, requires GraphViz installed
cargo xtask validate wgsl # for WGSL shaders
cargo xtask validate hlsl dxc # for HLSL shaders via DXC
cargo xtask validate hlsl fxc # for HLSL shaders via FXC
```
