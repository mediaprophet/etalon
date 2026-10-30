struct VertexOutput {
    @location(5) color: vec4f,
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) index: u32,
}

struct FragmentInput {
    @location(0) @interpolate(flat) index: u32,
    @builtin(view_index) view: u32,
    @location(5) color: vec4f,
    @builtin(position) position: vec4f,
    @builtin(front_facing) front: bool,
}

@vertex
fn vertex_main(@builtin(vertex_index) vertex: u32) -> VertexOutput {
    return VertexOutput(vec4f(0.5), vec4f(0.0, 0.0, 0.0, 1.0), vertex);
}

@fragment
fn fragment_main(input: FragmentInput) -> @location(0) vec4f {
    return input.color * f32(input.index + input.view) + input.position * f32(input.front);
}
