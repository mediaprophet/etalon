const triangles = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(view_index) view_index: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) offset: u32,
    @location(5) colour: vec4f,
}

struct FragmentInput {
    @location(5) colour: vec4f,
    @builtin(front_facing) front: bool,
    @location(0) @interpolate(flat) offset: u32,
    @builtin(view_index) view_index: u32,
    @builtin(position) position: vec4f,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4f(triangles[input.vertex_index], 0.0, 1.0);
    output.offset = 7u + input.view_index;
    output.colour = vec4f(0.125);
    return output;
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4f {
    return input.colour + vec4f(f32(input.view_index + input.offset - 7u) * 0.125);
}
