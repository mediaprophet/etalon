//! Online-softmax FlashAttention with guarded all-masked row handling.

#[inline(always)]
pub fn online_softmax_rescale(m_old: f32, m_new: f32) -> (f32, f32) {
    const MASK_FLOOR: f32 = -3.0e38;
    if m_new <= MASK_FLOOR {
        (0.0, 0.0)
    } else if m_old <= MASK_FLOOR {
        (0.0, 1.0)
    } else {
        let safe_diff = m_old - m_new;
        let rescale = safe_diff.exp();
        (rescale, 1.0)
    }
}
