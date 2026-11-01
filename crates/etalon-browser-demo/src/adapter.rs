use crate::browser::{bounded, Gpu, Health};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use wasm_bindgen::JsValue;

pub(crate) async fn request_gpu(
    health: &Arc<Mutex<Health>>,
    profile: &str,
) -> Result<(Gpu, Value), Value> {
    let levels: &[etalon::FeatureLevel] = match profile {
        "core" => &[etalon::FeatureLevel::Core],
        "compatibility" => &[etalon::FeatureLevel::Compatibility],
        "auto" => &[
            etalon::FeatureLevel::Core,
            etalon::FeatureLevel::Compatibility,
        ],
        _ => {
            return Err(
                json!({"status":"fail", "code":"invalid-profile", "message":"Unknown WebGPU feature level"}),
            )
        }
    };
    let navigator = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("navigator"))
        .unwrap_or(JsValue::UNDEFINED);
    let exposed_gpu =
        js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu")).unwrap_or(JsValue::UNDEFINED);
    if exposed_gpu.is_null() || exposed_gpu.is_undefined() {
        return Err(
            json!({"status":"unsupported", "code":"adapter-unavailable", "message":"navigator.gpu is unavailable in this browser context"}),
        );
    }
    let instance = etalon::Instance::new(etalon::InstanceDescriptor {
        backends: etalon::Backends::BROWSER_WEBGPU,
        ..etalon::InstanceDescriptor::new_without_display_handle()
    });
    let mut attempts = Vec::with_capacity(levels.len());
    let mut selected = None;
    for level in levels {
        let name = match level {
            etalon::FeatureLevel::Core => "core",
            etalon::FeatureLevel::Compatibility => "compatibility",
        };
        match instance
            .request_webgpu_adapter(&etalon::RequestAdapterOptions::default(), *level)
            .await
        {
            Ok(adapter) => {
                attempts.push(json!({"feature_level":name,"status":"available"}));
                selected = Some((adapter, name));
                break;
            }
            Err(error) => {
                let (status, code) = match &error {
                    etalon::RequestAdapterError::NotFound { .. } => {
                        ("unsupported", "adapter-unavailable")
                    }
                    etalon::RequestAdapterError::BrowserRequestRejected { .. } => {
                        ("fail", "adapter-request-rejected")
                    }
                    _ => ("fail", "adapter-request-failed"),
                };
                attempts.push(json!({"feature_level":name,"status":status,"code":code,"message":bounded(&error.to_string())}));
                if status == "fail" || attempts.len() == levels.len() {
                    return Err(
                        json!({"status":status,"code":code,"stage":"adapter-request","message":bounded(&error.to_string()),"adapter_attempts":attempts}),
                    );
                }
            }
        }
    }
    let (adapter, selected_level) = selected.ok_or_else(
        || json!({"status":"fail","code":"adapter-selection","message":"No adapter selected"}),
    )?;
    let adapter_info = adapter.get_info();
    let required = crate::requirements::required_limits(&adapter.limits());
    if !required.check_limits(&adapter.limits()) {
        return Err(
            json!({"status":"unsupported","code":"insufficient-adapter-limits","message":"Adapter cannot satisfy the declared domain-kernel suite limits","required_limits":limits_json(&required),"adapter_limits":limits_json(&adapter.limits()),"adapter_attempts":attempts}),
        );
    }
    let (device, queue) = adapter.request_device(&etalon::DeviceDescriptor {
        label: Some("Etalon browser library qualification"),
        required_features: etalon::Features::empty(),
        required_limits: required.clone(),
        ..Default::default()
    }).await.map_err(|error| json!({"status":"fail","code":"device-request-failed","message":bounded(&error.to_string()),"adapter_attempts":attempts}))?;
    let lost_health = health.clone();
    device.set_device_lost_callback(move |reason, message| {
        if let Ok(mut state) = lost_health.lock() {
            state.lost = Some(bounded(&format!("{reason:?}: {message}")));
        }
    });
    let error_health = health.clone();
    device.on_uncaptured_error(Arc::new(move |error: etalon::Error| {
        if let Ok(mut state) = error_health.lock() {
            state.uncaptured = Some(bounded(&error.to_string()));
        }
    }));
    let info = json!({
        "gpu_status":"available", "backend":format!("{:?}",adapter_info.backend),
        "adapter":bounded(&adapter_info.name),"vendor":adapter_info.vendor,"device":adapter_info.device,
        "device_type":format!("{:?}",adapter_info.device_type),"driver":bounded(&adapter_info.driver),"driver_info":bounded(&adapter_info.driver_info),
        "selected_feature_level":selected_level,"adapter_attempts":attempts,
        "available_features":bounded(&format!("{:?}",adapter.features())),"enabled_features":format!("{:?}",device.features()),
        "required_limits":limits_json(&required),"adapter_limits":limits_json(&adapter.limits()),"device_limits":limits_json(&device.limits())
    });
    Ok((
        Gpu {
            instance,
            adapter,
            device,
            queue,
        },
        info,
    ))
}

fn limits_json(l: &etalon::Limits) -> Value {
    json!({"max_buffer_size":l.max_buffer_size,"max_storage_buffer_binding_size":l.max_storage_buffer_binding_size,
        "max_compute_workgroup_size_x":l.max_compute_workgroup_size_x,"max_compute_invocations_per_workgroup":l.max_compute_invocations_per_workgroup,
        "max_compute_workgroup_storage_size":l.max_compute_workgroup_storage_size,"max_compute_workgroups_per_dimension":l.max_compute_workgroups_per_dimension,
        "max_texture_dimension_2d":l.max_texture_dimension_2d,"max_bind_groups":l.max_bind_groups,"max_bind_groups_plus_vertex_buffers":l.max_bind_groups_plus_vertex_buffers,
        "max_bindings_per_bind_group":l.max_bindings_per_bind_group,"max_storage_buffers_per_shader_stage":l.max_storage_buffers_per_shader_stage,
        "max_uniform_buffers_per_shader_stage":l.max_uniform_buffers_per_shader_stage,"max_uniform_buffer_binding_size":l.max_uniform_buffer_binding_size,
        "max_color_attachment_bytes_per_sample":l.max_color_attachment_bytes_per_sample,"min_uniform_buffer_offset_alignment":l.min_uniform_buffer_offset_alignment,
        "min_storage_buffer_offset_alignment":l.min_storage_buffer_offset_alignment})
}
