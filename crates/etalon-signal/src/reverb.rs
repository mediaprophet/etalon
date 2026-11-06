//! Zero-heap Schroeder/Freeverb comb & allpass steps, 4×4 Hadamard FDN mixing,
//! and stereo mux/demux/Mid-Side width primitives.

/// Execute one step of a Schroeder allpass diffuser over caller-supplied delay line `buf`:
/// `y[n] = -x[n] + buf[idx]`, `buf'[idx] = x[n] + feedback * buf[idx]`.
#[inline]
pub fn allpass_step(buf: &mut [f32], idx: &mut usize, feedback: f32, x: f32) -> f32 {
    if buf.is_empty() {
        return -x;
    }
    let i = (*idx).min(buf.len() - 1);
    let bufout = buf[i];
    let output = -x + bufout;
    buf[i] = x + bufout * feedback;
    *idx = if i + 1 >= buf.len() { 0 } else { i + 1 };
    output
}

/// Execute one step of a Freeverb feedback comb filter with one-pole damping low-pass:
/// `y[n] = buf[idx]`, `store' = y[n]*(1 - damp) + store*damp`, `buf'[idx] = x[n] + store'*feedback`.
#[inline]
pub fn comb_step(
    buf: &mut [f32],
    idx: &mut usize,
    feedback: f32,
    damp: f32,
    filter_store: &mut f32,
    x: f32,
) -> f32 {
    if buf.is_empty() {
        return x;
    }
    let i = (*idx).min(buf.len() - 1);
    let output = buf[i];
    *filter_store = output * (1.0 - damp) + (*filter_store) * damp;
    buf[i] = x + (*filter_store) * feedback;
    *idx = if i + 1 >= buf.len() { 0 } else { i + 1 };
    output
}

/// Orthonormal $4 \times 4$ Hadamard feedback matrix $\frac{1}{2} H_4$ for 4-channel
/// Feedback Delay Networks (FDN). Preserves Euclidean energy: $\|H_4 x / 2\|_2 = \|x\|_2$.
#[inline]
pub const fn fdn4_hadamard_mix(x: [f32; 4]) -> [f32; 4] {
    [
        0.5 * (x[0] + x[1] + x[2] + x[3]),
        0.5 * (x[0] - x[1] + x[2] - x[3]),
        0.5 * (x[0] + x[1] - x[2] - x[3]),
        0.5 * (x[0] - x[1] - x[2] + x[3]),
    ]
}

/// Deinterleave stereo frames `[L0, R0, L1, R1, ...]` into planar `left` and `right`.
/// Returns the number of stereo frames written.
pub fn stereo_demux_into(interleaved: &[f32], left: &mut [f32], right: &mut [f32]) -> usize {
    let frames = (interleaved.len() / 2).min(left.len()).min(right.len());
    for i in 0..frames {
        left[i] = interleaved[i * 2];
        right[i] = interleaved[i * 2 + 1];
    }
    frames
}

/// Interleave planar `left` and `right` channels into `interleaved` (`[L0, R0, L1, R1, ...]`).
/// Returns the number of stereo frames written.
pub fn stereo_mux_into(left: &[f32], right: &[f32], interleaved: &mut [f32]) -> usize {
    let frames = left.len().min(right.len()).min(interleaved.len() / 2);
    for i in 0..frames {
        interleaved[i * 2] = left[i];
        interleaved[i * 2 + 1] = right[i];
    }
    frames
}

/// Apply Mid/Side stereo width adjustment in place to planar `left` and `right` buffers:
/// $M = (L + R)/2$, $S = (L - R) \cdot \text{width} / 2$, $L' = M + S$, $R' = M - S$.
pub fn stereo_width_in_place(left: &mut [f32], right: &mut [f32], width: f32) {
    let frames = left.len().min(right.len());
    for i in 0..frames {
        let l = left[i];
        let r = right[i];
        let mid = 0.5 * (l + r);
        let side = 0.5 * (l - r) * width;
        left[i] = mid + side;
        right[i] = mid - side;
    }
}

/// Apply Mid/Side stereo width adjustment from interleaved `in_stereo` into `out_stereo`.
/// Returns the number of stereo frames processed.
pub fn stereo_width_interleaved_into(
    in_stereo: &[f32],
    out_stereo: &mut [f32],
    width: f32,
) -> usize {
    let frames = in_stereo.len().min(out_stereo.len()) / 2;
    for i in 0..frames {
        let l = in_stereo[i * 2];
        let r = in_stereo[i * 2 + 1];
        let mid = 0.5 * (l + r);
        let side = 0.5 * (l - r) * width;
        out_stereo[i * 2] = mid + side;
        out_stereo[i * 2 + 1] = mid - side;
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdn4_hadamard_is_orthonormal_involution() {
        let v = [1.0f32, -2.0, 0.5, 3.0];
        let y = fdn4_hadamard_mix(v);
        let back = fdn4_hadamard_mix(y);
        for i in 0..4 {
            assert!((back[i] - v[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn stereo_mux_demux_and_width() {
        let mut l = [1.0f32, 0.5];
        let mut r = [-1.0f32, 0.5];
        let mut inter = [0.0f32; 4];
        assert_eq!(stereo_mux_into(&l, &r, &mut inter), 2);
        let mut l2 = [0.0f32; 2];
        let mut r2 = [0.0f32; 2];
        assert_eq!(stereo_demux_into(&inter, &mut l2, &mut r2), 2);
        assert_eq!(l2, l);
        assert_eq!(r2, r);

        stereo_width_in_place(&mut l, &mut r, 0.0);
        assert!((l[0] - r[0]).abs() < 1e-6);
    }
}
