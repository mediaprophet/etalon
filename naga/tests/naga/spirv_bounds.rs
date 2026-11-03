#![cfg(all(feature = "wgsl-in", spv_out))]

use std::{
    io::Write as _,
    process::{Command, Stdio},
};

use naga::{
    back::spv,
    proc::{BoundsCheckPolicies, BoundsCheckPolicy},
};

#[test]
#[cfg_attr(miri, ignore)]
fn binding_array_runtime_bounds_validate_with_mixed_policies() {
    for binding_size in [", 8", ""] {
        let source = format!(
            r"
            enable etalon_binding_array;
            struct Data {{ inner: array<u32> }}
            @group(0) @binding(0)
            var<storage, read_write> buffers: binding_array<Data{binding_size}>;
            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) id: vec3u) {{
                let descriptor = id.x + 8u;
                let element = id.y + 16u;
                let length = arrayLength(&buffers[descriptor].inner);
                buffers[descriptor].inner[element] = buffers[descriptor].inner[element] + length;
            }}
        "
        );
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let policies = [
            BoundsCheckPolicy::Restrict,
            BoundsCheckPolicy::ReadZeroSkipWrite,
        ];
        for buffer in policies {
            for binding_array in policies {
                let mut options = spv::Options {
                    fake_missing_bindings: false,
                    bounds_check_policies: BoundsCheckPolicies {
                        buffer,
                        binding_array,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                options.binding_map.insert(
                    naga::ResourceBinding {
                        group: 0,
                        binding: 0,
                    },
                    spv::BindingInfo {
                        descriptor_set: 0,
                        binding: 0,
                        binding_array_size: Some(8),
                        ..Default::default()
                    },
                );
                let words = spv::write_vec(&module, &info, &options, None).unwrap();
                let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                let mut child = Command::new("spirv-val")
                    .args(["--target-env", "vulkan1.1"])
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .expect("spirv-val requires the Vulkan SDK bin directory on PATH");
                child.stdin.take().unwrap().write_all(&bytes).unwrap();
                let output = child.wait_with_output().unwrap();
                assert!(output.status.success(),
                    "binding size {binding_size:?}, buffer {buffer:?}, binding_array {binding_array:?}:\n{}",
                    String::from_utf8_lossy(&output.stderr),
                );
            }
        }
    }
}
