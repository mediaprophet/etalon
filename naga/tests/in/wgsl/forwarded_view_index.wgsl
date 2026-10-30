struct VertexInput {
    @builtin(vertex_index) vertex: u32,
    @builtin(view_index) view: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) unused: f32,
}

struct FragmentInput {
    @builtin(position) position: vec4f,
    @builtin(view_index) view: u32,
}

@vertex
fn scalar_output() -> @builtin(position) @invariant vec4f {
    return vec4f(0.0, 0.0, 0.0, 1.0);
}

@vertex
fn scalar_existing_view(@builtin(view_index) view: u32) -> @builtin(position) vec4f {
    if view == 0u {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    return vec4f(1.0, 0.0, 0.0, 1.0);
}

@vertex
fn struct_existing_view(input: VertexInput) -> VertexOutput {
    return VertexOutput(vec4f(f32(input.view), f32(input.vertex), 0.0, 1.0), 3.0);
}

@fragment
fn fragment_view(@builtin(view_index) view: u32) -> @location(0) vec4f {
    return vec4f(f32(view));
}

@fragment
fn fragment_struct(input: FragmentInput) -> @location(0) vec4f {
    return vec4f(f32(input.view)) + input.position;
}

@fragment
fn fragment_without_view() -> @location(0) vec4f {
    return vec4f(1.0);
}
