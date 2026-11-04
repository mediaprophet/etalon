use super::Error;
use etalon_model::inference::qwen_roles::Qwen35Geometry;

/// Exact logical extents; the workspace includes candidate state and history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceLayout {
    pub floats: usize,
    pub bytes: usize,
    pub history: usize,
    pub recurrent: usize,
    pub channels: usize,
    pub value: usize,
    pub expanded_key: usize,
    pub projection: usize,
    pub conv_weights: usize,
}
impl WorkspaceLayout {
    pub fn checked(g: Qwen35Geometry) -> Result<Self, Error> {
        fn mul(a: usize, b: usize) -> Result<usize, Error> {
            a.checked_mul(b).ok_or(Error::Overflow)
        }
        fn add(a: usize, b: usize) -> Result<usize, Error> {
            a.checked_add(b).ok_or(Error::Overflow)
        }
        if g.hidden == 0
            || g.key_heads == 0
            || g.value_heads == 0
            || g.key_dim == 0
            || g.value_dim == 0
            || g.kernel == 0
            || !g.value_heads.is_multiple_of(g.key_heads)
        {
            return Err(Error::Shape);
        }
        let key = mul(g.key_heads, g.key_dim)?;
        let value = mul(g.value_heads, g.value_dim)?;
        let channels = add(mul(2, key)?, value)?;
        let history = mul(channels, g.kernel - 1)?;
        let recurrent = mul(mul(g.value_heads, g.key_dim)?, g.value_dim)?;
        if (
            g.qk_width,
            g.value_width,
            g.channels,
            g.history_floats,
            g.state_floats,
        ) != (key, value, channels, history, recurrent)
        {
            return Err(Error::Shape);
        }
        let expanded_key = mul(g.value_heads, g.key_dim)?;
        let projection = add(channels, value)?;
        let conv_weights = mul(channels, g.kernel)?;
        let mut floats = 0;
        for count in [
            history,
            recurrent,
            conv_weights,
            g.value_heads,
            g.value_heads,
            g.value_dim,
            projection,
            mul(2, g.value_heads)?,
            channels,
            value,
            g.value_heads,
            g.value_heads,
            channels,
            expanded_key,
            expanded_key,
            g.value_heads,
            g.value_heads,
            value,
            value,
            g.hidden,
        ] {
            floats = add(floats, count)?;
        }
        Ok(Self {
            floats,
            bytes: mul(floats, core::mem::size_of::<f32>())?,
            history,
            recurrent,
            channels,
            value,
            expanded_key,
            projection,
            conv_weights,
        })
    }
}

pub(super) fn take<'a>(rest: &mut &'a mut [f32], count: usize) -> &'a mut [f32] {
    let (head, tail) = core::mem::take(rest).split_at_mut(count);
    *rest = tail;
    head
}
