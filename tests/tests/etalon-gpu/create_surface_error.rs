//! Test that `create_surface_*()` accurately reports those errors we can provoke.
#![cfg(wasm_test)]

use etalon_test::GpuTestInitializer;
use etalon_test::{apply, gpu_test, GpuTestConfiguration};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(CANVAS_GET_CONTEXT_RETURNED_NULL);
}

/// This test applies to those cfgs that can create a surface from a canvas, which
/// include WebGL and WebGPU, but *not* Emscripten GLES.
#[apply(gpu_test!)]
static CANVAS_GET_CONTEXT_RETURNED_NULL: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(etalon_test::TestParameters::default().enable_noop())
    .run_async(|_ctx| async move {
        #[cfg(target_arch = "wasm32")]
        {
            // Not using the normal testing infrastructure because that goes straight to creating the canvas for us.
            let instance = etalon_test::initialize_instance(
                etalon::Backends::all(),
                &etalon_test::TestParameters::default(),
            );
            // Create canvas
            let canvas = etalon_test::initialize_html_canvas();

            // Using a context id that is not "webgl2" or "webgpu" will render the canvas unusable by etalon.
            canvas.get_context("2d").unwrap();

            #[allow(
                clippy::redundant_clone,
                reason = "false positive — can't and shouldn't move out."
            )]
            let error = instance
                .create_surface(etalon::SurfaceTarget::Canvas(canvas.clone()))
                .unwrap_err();

            assert!(
                error
                    .to_string()
                    .contains("canvas.getContext() returned null"),
                "{error}"
            );
        }
    });
