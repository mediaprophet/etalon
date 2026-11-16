// EMF 5D volumetric visualizer — renders 2D slices of the 4D EMF field grid
// (x×y×z×t) with 10D manifold tags and Stokes polarization mapped to color.
//
// Fragment shader samples a storage buffer containing the flat field grid
// (indexed [t][z][y][x]) and maps:
//   amplitude  → brightness (HDR value)
//   phase      → hue (via Full-Spectrum σ → CIE XYZ / False-Color → linear sRGB)
//   manifold.scale + attention_depth → saturation gain
//   manifold.manifold_curvature + Stokes (Q,U,V) → rim glow / polarization highlight
//
// @group(0) camera + slice params · @group(1) field grid storage

struct EmfSliceUniform {
    // Grid dimensions.
    nx: u32,
    ny: u32,
    nz: u32,
    nt: u32,
    // Current slice indices.
    slice_z: u32,
    slice_t: u32,
    // Grid bounds: [x_min, x_max, y_min, y_max, z_min, z_max].
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    z_min: f32,
    z_max: f32,
    // Display controls.
    amplitude_scale: f32,
    phase_offset: f32,
    manifold_gain: f32,
    projection_mode: u32,
};

// Per-cell field data: amplitude, phase, frequency + 10D manifold coords + 3 Stokes polarization params.
// Matches the host-side EmfFieldCell struct (64 bytes = 3 f32 + 10 f32 + 3 f32).
struct EmfFieldCell {
    amplitude: f32,
    phase: f32,
    frequency: f32,
    // 10D manifold coordinate.
    scale: f32,
    attention_depth: f32,
    epistemic_weight: f32,
    topological_spin: f32,
    temporal_decay: f32,
    entropy_bias: f32,
    spatial_phase: f32,
    recurrence_frequency: f32,
    density_threshold: f32,
    manifold_curvature: f32,
    // Stokes polarization parameters (Q, U, V).
    stokes_q: f32,
    stokes_u: f32,
    stokes_v: f32,
};

@group(0) @binding(0) var<uniform> params: EmfSliceUniform;
@group(1) @binding(0) var<storage, read> field: array<EmfFieldCell>;

const TWO_PI: f32 = 6.283185307;

// σ → CIE 1931 XYZ → linear sRGB (shared spectral mapping).
fn sigma_to_cie_xyz(sigma: f32) -> vec3<f32> {
    let s = fract(sigma);
    let lambda = 380.0 + (s * 370.0);
    let x1 = 1.056 * exp(-0.5 * pow((lambda - 599.8) / 43.2, 2.0));
    let x2 = 0.362 * exp(-0.5 * pow((lambda - 442.0) / 32.0, 2.0));
    let x3 = -0.065 * exp(-0.5 * pow((lambda - 501.1) / 20.4, 2.0));
    let X = x1 + x2 + x3;
    let y1 = 0.821 * exp(-0.5 * pow((lambda - 568.8) / 46.9, 2.0));
    let y2 = 0.286 * exp(-0.5 * pow((lambda - 530.9) / 16.3, 2.0));
    let Y = y1 + y2;
    let z1 = 1.217 * exp(-0.5 * pow((lambda - 437.0) / 11.8, 2.0));
    let z2 = 0.681 * exp(-0.5 * pow((lambda - 459.0) / 26.0, 2.0));
    let Z = z1 + z2;
    return vec3<f32>(X, Y, Z);
}

fn xyz_to_linear_srgb(xyz: vec3<f32>) -> vec3<f32> {
    let R = 3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z;
    let G = -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z;
    let B = 0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z;
    return max(vec3<f32>(R, G, B), vec3<f32>(0.0));
}

// Full-Spectrum False-Color Lens across 6 electromagnetic bands.
fn sigma_to_linear_rgb(sigma: f32) -> vec3<f32> {
    let band = i32(floor(sigma));
    let s = fract(sigma);
    if (band == 0) {
        // Band 0 (Visible 380-750 nm): CIE 1931 XYZ analytic Gaussian fit -> linear sRGB
        return xyz_to_linear_srgb(sigma_to_cie_xyz(s));
    } else if (band == 1) {
        // Band 1 (Radio/Microwave): deep crimson-to-amber false-color ramp
        return vec3<f32>(
            mix(0.35, 1.0, s),
            mix(0.02, 0.55, s * s),
            mix(0.05, 0.08, s)
        );
    } else if (band == 2) {
        // Band 2 (Infrared / Thermal): blackbody ironbow ramp (indigo -> magenta -> orange -> white-hot)
        let r = clamp(s * 1.6, 0.0, 1.0);
        let g = clamp((s - 0.35) * 1.55, 0.0, 1.0);
        let b = clamp(sin(s * 3.14159265) * 0.75 + max(0.0, (s - 0.75) * 4.0), 0.0, 1.0);
        return vec3<f32>(r, g, b);
    } else if (band == 3) {
        // Band 3 (Ultraviolet): violet-to-electric-cyan fluorescence ramp
        return vec3<f32>(
            mix(0.42, 0.05, s),
            mix(0.05, 0.92, s),
            mix(0.75, 1.0, s)
        );
    } else if (band == 4) {
        // Band 4 (X-Ray / CT attenuation): high-contrast radiographic cyan-white ramp
        let c = pow(s, 0.85);
        return vec3<f32>(c * 0.75, c * 0.95, clamp(c * 1.15, 0.0, 1.0));
    } else {
        // Band >= 5 (Gamma / Cherenkov): Cherenkov blue-violet-white radiation ramp
        return vec3<f32>(
            mix(0.15, 0.95, s * s),
            mix(0.35, 0.98, s),
            1.0
        );
    }
}

// Map phase (radians) to σ in [0,1) for spectral coloring.
fn phase_to_sigma(phase: f32) -> f32 {
    return fract(phase / TWO_PI);
}

// Sample the field grid at integer (ix, iy) for the current (slice_z, slice_t).
fn sample_field(ix: u32, iy: u32) -> EmfFieldCell {
    let idx = ((params.slice_t * params.nz + params.slice_z) * params.ny + iy) * params.nx + ix;
    return field[idx];
}

// Bilinear sample at normalized UV in [0,1].
fn sample_field_bilinear(uv: vec2<f32>) -> EmfFieldCell {
    let fx = clamp(uv.x, 0.0, 0.999) * f32(params.nx);
    let fy = clamp(uv.y, 0.0, 0.999) * f32(params.ny);
    let ix0 = u32(floor(fx));
    let iy0 = u32(floor(fy));
    let ix1 = min(ix0 + 1u, params.nx - 1u);
    let iy1 = min(iy0 + 1u, params.ny - 1u);
    let tx = fx - f32(ix0);
    let ty = fy - f32(iy0);

    let c00 = sample_field(ix0, iy0);
    let c10 = sample_field(ix1, iy0);
    let c01 = sample_field(ix0, iy1);
    let c11 = sample_field(ix1, iy1);

    var result: EmfFieldCell;
    result.amplitude = mix(mix(c00.amplitude, c10.amplitude, tx), mix(c01.amplitude, c11.amplitude, tx), ty);
    result.phase = mix(mix(c00.phase, c10.phase, tx), mix(c01.phase, c11.phase, tx), ty);
    result.frequency = mix(mix(c00.frequency, c10.frequency, tx), mix(c01.frequency, c11.frequency, tx), ty);
    result.scale = mix(mix(c00.scale, c10.scale, tx), mix(c01.scale, c11.scale, tx), ty);
    result.attention_depth = mix(mix(c00.attention_depth, c10.attention_depth, tx), mix(c01.attention_depth, c11.attention_depth, tx), ty);
    result.epistemic_weight = mix(mix(c00.epistemic_weight, c10.epistemic_weight, tx), mix(c01.epistemic_weight, c11.epistemic_weight, tx), ty);
    result.topological_spin = mix(mix(c00.topological_spin, c10.topological_spin, tx), mix(c01.topological_spin, c11.topological_spin, tx), ty);
    result.temporal_decay = mix(mix(c00.temporal_decay, c10.temporal_decay, tx), mix(c01.temporal_decay, c11.temporal_decay, tx), ty);
    result.entropy_bias = mix(mix(c00.entropy_bias, c10.entropy_bias, tx), mix(c01.entropy_bias, c11.entropy_bias, tx), ty);
    result.spatial_phase = mix(mix(c00.spatial_phase, c10.spatial_phase, tx), mix(c01.spatial_phase, c11.spatial_phase, tx), ty);
    result.recurrence_frequency = mix(mix(c00.recurrence_frequency, c10.recurrence_frequency, tx), mix(c01.recurrence_frequency, c11.recurrence_frequency, tx), ty);
    result.density_threshold = mix(mix(c00.density_threshold, c10.density_threshold, tx), mix(c01.density_threshold, c11.density_threshold, tx), ty);
    result.manifold_curvature = mix(mix(c00.manifold_curvature, c10.manifold_curvature, tx), mix(c01.manifold_curvature, c11.manifold_curvature, tx), ty);
    result.stokes_q = mix(mix(c00.stokes_q, c10.stokes_q, tx), mix(c01.stokes_q, c11.stokes_q, tx), ty);
    result.stokes_u = mix(mix(c00.stokes_u, c10.stokes_u, tx), mix(c01.stokes_u, c11.stokes_u, tx), ty);
    result.stokes_v = mix(mix(c00.stokes_v, c10.stokes_v, tx), mix(c01.stokes_v, c11.stokes_v, tx), ty);
    return result;
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    // Full-screen triangle strip (2 triangles covering [-1,1]²).
    let positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    let uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );
    var out: VertexOutput;
    let p = positions[vid];
    out.clip_position = vec4<f32>(p, 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if params.projection_mode == 1u {
        // Precomputed |E| display derivative, nearest cell; no interpolation over missing data.
        let ix = min(u32(clamp(input.uv.x, 0.0, 1.0) * f32(params.nx)), params.nx - 1u);
        let iy = min(u32(clamp(input.uv.y, 0.0, 1.0) * f32(params.ny)), params.ny - 1u);
        let display = sample_field(ix, iy);
        let brightness = clamp(display.amplitude * params.amplitude_scale, 0.0, 1.0);
        return vec4<f32>(vec3<f32>(brightness), display.epistemic_weight);
    }
    let cell = sample_field_bilinear(input.uv);
    if cell.epistemic_weight <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Amplitude → brightness (HDR, scaled).
    let amp = abs(cell.amplitude) * params.amplitude_scale;
    let brightness = clamp(amp, 0.0, 4.0);

    // Select full-spectrum recurrence_frequency (σ across [0.0, 6.0)) when > 0.0,
    // falling back to phase_to_sigma(cell.phase + params.phase_offset) when <= 0.0.
    let sigma = select(
        phase_to_sigma(cell.phase + params.phase_offset),
        cell.recurrence_frequency,
        cell.recurrence_frequency > 0.0
    );
    let base_rgb = sigma_to_linear_rgb(sigma);

    // Manifold fields modulate saturation and tint.
    let manifold_gain = params.manifold_gain;
    let saturation = clamp(cell.scale * manifold_gain, 0.0, 1.5);
    let luminance = dot(base_rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let saturated_rgb = mix(vec3<f32>(luminance), base_rgb, saturation);

    // Degree of polarization Π = clamp(sqrt(Q² + U² + V²) / (abs(cell.amplitude) + 1e-5), 0.0, 1.0).
    let pol_mag = sqrt(
        cell.stokes_q * cell.stokes_q +
        cell.stokes_u * cell.stokes_u +
        cell.stokes_v * cell.stokes_v
    );
    let degree_of_polarization = clamp(pol_mag / (abs(cell.amplitude) + 1e-5), 0.0, 1.0);
    let pol_highlight = 1.0 + 0.35 * degree_of_polarization;

    // Curvature → rim glow / contour enhancement modulated by polarization.
    let curvature_glow = clamp(abs(cell.manifold_curvature) * manifold_gain, 0.0, 1.0);
    let glow_tint = sigma_to_linear_rgb(phase_to_sigma(cell.spatial_phase));
    let rgb = saturated_rgb * brightness * pol_highlight + glow_tint * curvature_glow * (0.3 + 0.2 * degree_of_polarization);

    // Alpha: amplitude-based opacity, fades to transparent at zero amplitude.
    let alpha = clamp(brightness * 0.5 + 0.1, 0.0, 1.0);

    return vec4<f32>(rgb, alpha);
}
