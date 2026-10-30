# deno_webgpu

This op crate implements the WebGPU API as defined in
https://gpuweb.github.io/gpuweb/ in Deno. The implementation targets the spec
draft as of March 31, 2024. The spec is still very much in flux. This extension
tries to stay up to date with the spec, but is constrained by the features
implemented in our GPU backend library [etalon](https://github.com/mediaprophet/etalon).

The spec is still very bare bones, and is still missing many details. As the
spec becomes more concrete, we will implement to follow the spec more closely.

In addition, setting the `DENO_WEBGPU_TRACE` environmental variable will output
a
[etalon trace](https://mediaprophet.github.io/etalon/doc/etalon/documentation/debugging/debugging_applications/index.html#tracing-infrastructure)
to the specified directory.

This op crate is tested primarily by running the
[WebGPU conformance test suite](https://github.com/gpuweb/cts) using `etalon`'s
[`cts_runner`](../cts_runner/README.md).
`cts_runner` also has a few
[directed tests](../cts_runner/tests)
to fill in missing coverage.

GPU availability in GitHub CI is limited, so some configurations rely on
software like DX WARP & Vulkan lavapipe.

## Links

Specification: https://gpuweb.github.io/gpuweb/

Design documents: https://github.com/gpuweb/gpuweb/tree/main/design

Conformance tests suite: https://github.com/gpuweb/cts

WebGPU examples for Deno: https://github.com/crowlKats/webgpu-examples
