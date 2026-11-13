// Depth changes are in shadow texels, independent of world/camera scale.
const SHADOW_RECEIVER_DEPTH_EPSILON: f32 = 0.00000095367431640625;

fn shadow_receiver_coordinates(clip: vec4<f32>) -> vec3<f32> {
    let ndc = clip.xyz / max(abs(clip.w), 1e-8);
    return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

fn shadow_receiver_gradient(dx: vec3<f32>, dy: vec3<f32>, texel: f32, bias_limit: f32) -> vec3<f32> {
    let scale = max(max(abs(dx.x), abs(dx.y)), max(abs(dy.x), abs(dy.y)));
    if (scale <= 1e-12) { return vec3<f32>(0.0); }
    let x = dx / scale;
    let y = dy / scale;
    let determinant = x.x * y.y - x.y * y.x;
    if (abs(determinant) <= 1e-5) { return vec3<f32>(0.0); }
    let step = vec2<f32>(y.y * x.z - x.y * y.z, x.x * y.z - y.x * x.z) * (texel / determinant);
    if (any(step != step)) { return vec3<f32>(0.0); }
    let limit = min(1.0, max(abs(bias_limit), 4.0 * texel));
    let bounded = clamp(step, vec2<f32>(-limit), vec2<f32>(limit));
    return vec3<f32>(bounded, select(0.0, 1.0, all(abs(step) <= vec2<f32>(limit))));
}

fn shadow_receiver_depth_bias(plane: vec3<f32>, fraction: vec2<f32>, normal_light: f32, bias_limit: f32) -> f32 {
    if (plane.z < 0.5) {
        return max(SHADOW_RECEIVER_DEPTH_EPSILON, clamp(abs(bias_limit), 0.0, 1.0) * (1.0 - clamp(normal_light, 0.0, 1.0)));
    }
    // Exact nonzero bilinear support depends on the fractional texel phase.
    let lower = -fraction;
    let upper = select(vec2<f32>(0.0), vec2<f32>(1.0) - fraction, fraction > vec2<f32>(0.0));
    let nearest = min(plane.xy * lower, plane.xy * upper);
    return SHADOW_RECEIVER_DEPTH_EPSILON - nearest.x - nearest.y;
}

fn shadow_receiver_compare_depth(depth: f32, plane: vec3<f32>, tap: vec2<f32>, fraction: vec2<f32>, normal_light: f32, bias_limit: f32) -> f32 {
    let correction = select(0.0, dot(plane.xy, tap), plane.z > 0.5);
    return depth + correction - shadow_receiver_depth_bias(plane, fraction, normal_light, bias_limit);
}
