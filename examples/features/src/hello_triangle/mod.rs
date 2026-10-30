use std::{borrow::Cow, future::Future, sync::Arc};
use etalon::CurrentSurfaceTexture;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    window::Window,
};

/// Runs a future to completion. On native this blocks synchronously via pollster.
/// On wasm this spawns a local task so control returns to the browser immediately.
#[cfg(not(target_arch = "wasm32"))]
fn spawn(f: impl Future<Output = ()> + 'static) {
    pollster::block_on(f);
}

/// Runs a future to completion. On native this blocks synchronously via pollster.
/// On wasm this spawns a local task so control returns to the browser immediately.
#[cfg(target_arch = "wasm32")]
fn spawn(f: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(f);
}

struct EtalonState {
    instance: etalon::Instance,
    window: Arc<Window>,
    device: etalon::Device,
    queue: etalon::Queue,
    surface: etalon::Surface<'static>,
    config: etalon::SurfaceConfiguration,
    render_pipeline: etalon::RenderPipeline,
}

enum TriangleAction {
    Initialized(EtalonState),
}

#[expect(clippy::large_enum_variant)]
enum AppState {
    Uninitialized,
    Loading,
    Running(EtalonState),
}

struct App {
    proxy: EventLoopProxy<TriangleAction>,
    window: Option<Arc<Window>>,
    state: AppState,
}

impl App {
    fn new(event_loop: &EventLoop<TriangleAction>) -> Self {
        Self {
            proxy: event_loop.create_proxy(),
            window: None,
            state: AppState::Uninitialized,
        }
    }

    fn resize_surface(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        let AppState::Running(etalon_state) = &mut self.state else {
            return;
        };

        // Reconfigure the surface with the new size
        etalon_state.config.width = new_size.width.max(1);
        etalon_state.config.height = new_size.height.max(1);
        etalon_state
            .surface
            .configure(&etalon_state.device, &etalon_state.config);

        // On macos the window needs to be redrawn manually after resizing
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl ApplicationHandler<TriangleAction> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !matches!(self.state, AppState::Uninitialized) {
            return;
        }
        self.state = AppState::Loading;

        #[cfg_attr(
            not(target_arch = "wasm32"),
            expect(unused_mut, reason = "wasm32 re-assigns to specify canvas")
        )]
        let mut attributes = Window::default_attributes();

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("canvas")
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            attributes = attributes.with_canvas(Some(canvas));
        }

        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("Failed to create window"),
        );
        self.window = Some(window.clone());

        let display_handle = event_loop.owned_display_handle();
        let proxy = self.proxy.clone();

        spawn(async move {
            let mut size = window.inner_size();
            size.width = size.width.max(1);
            size.height = size.height.max(1);

            let instance =
                etalon::Instance::new(etalon::InstanceDescriptor::new_with_display_handle_from_env(
                    Box::new(display_handle),
                ));

            let surface = instance.create_surface(window.clone()).unwrap();
            let adapter = instance
                .request_adapter(&etalon::RequestAdapterOptions {
                    power_preference: etalon::PowerPreference::default(),
                    // Request an adapter which can render to our surface
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .expect("Failed to find an appropriate adapter");

            // Create the logical device and command queue
            let (device, queue) = adapter
                .request_device(&etalon::DeviceDescriptor {
                    label: None,
                    required_features: etalon::Features::empty(),
                    // Make sure we use the texture resolution limits from the adapter,
                    // so we can support images the size of the swapchain.
                    required_limits: etalon::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    default_queue: etalon::QueueDescriptor { label: None },
                    experimental_features: etalon::ExperimentalFeatures::disabled(),
                    memory_hints: etalon::MemoryHints::MemoryUsage,
                    trace: etalon::Trace::Off,
                })
                .await
                .expect("Failed to create device");

            // Load the shaders from disk
            let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
                label: None,
                source: etalon::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            });

            let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                immediate_size: 0,
            });

            let swapchain_capabilities = surface.get_capabilities(&adapter);
            let swapchain_format = swapchain_capabilities.formats[0];

            let render_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: etalon::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(etalon::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(swapchain_format.into())],
                }),
                primitive: etalon::PrimitiveState::default(),
                depth_stencil: None,
                multisample: etalon::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            let config = surface
                .get_default_config(&adapter, size.width, size.height)
                .unwrap();

            surface.configure(&device, &config);

            let _ = proxy.send_event(TriangleAction::Initialized(EtalonState {
                instance,
                window,
                device,
                queue,
                surface,
                config,
                render_pipeline,
            }));
        });
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: TriangleAction) {
        match event {
            TriangleAction::Initialized(etalon_state) => {
                self.state = AppState::Running(etalon_state);

                // winit might have updated the window size while we were
                // creating the surface asynchronously, so resize the surface.
                if let Some(window) = &self.window {
                    self.resize_surface(window.inner_size());
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let AppState::Running(etalon_state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::Resized(new_size) => {
                self.resize_surface(new_size);
            }
            WindowEvent::RedrawRequested => {
                let frame = match etalon_state.surface.get_current_texture() {
                    CurrentSurfaceTexture::Success(frame) => frame,
                    CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => {
                        // Try again later
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                    CurrentSurfaceTexture::Suboptimal(texture) => {
                        drop(texture);

                        etalon_state
                            .surface
                            .configure(&etalon_state.device, &etalon_state.config);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                    CurrentSurfaceTexture::Outdated => {
                        etalon_state
                            .surface
                            .configure(&etalon_state.device, &etalon_state.config);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                    CurrentSurfaceTexture::Validation => {
                        unreachable!("No error scope registered, so validation errors will panic")
                    }
                    CurrentSurfaceTexture::Lost => {
                        etalon_state.surface = etalon_state
                            .instance
                            .create_surface(etalon_state.window.clone())
                            .unwrap();
                        etalon_state
                            .surface
                            .configure(&etalon_state.device, &etalon_state.config);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&etalon::TextureViewDescriptor::default());
                let mut encoder = etalon_state
                    .device
                    .create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
                {
                    let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(etalon::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: etalon::Operations {
                                load: etalon::LoadOp::Clear(etalon::Color::GREEN),
                                store: etalon::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    rpass.set_pipeline(&etalon_state.render_pipeline);
                    rpass.draw(0..3, 0..1);
                }

                etalon_state.queue.submit(Some(encoder.finish()));
                if let Some(window) = &self.window {
                    window.pre_present_notify();
                }
                etalon_state.queue.present(frame);
            }
            WindowEvent::Occluded(false) => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }
}

pub fn main() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init().expect("could not initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::with_user_event().build().unwrap();

    #[cfg_attr(target_arch = "wasm32", expect(unused_mut))]
    let mut app = App::new(&event_loop);

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use winit::platform::web::EventLoopExtWebSys;
            event_loop.spawn_app(app);
        } else {
            event_loop.run_app(&mut app).unwrap();
        }
    }
}
