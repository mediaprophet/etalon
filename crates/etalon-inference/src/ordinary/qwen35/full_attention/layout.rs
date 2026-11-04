use super::super::Error;
use etalon_model::inference::qwen_attention::Qwen35AttentionGeometry;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceLayout {
    pub floats: usize,
    pub bytes: usize,
    pub query: usize,
    pub kv: usize,
    pub capacity: usize,
}
impl WorkspaceLayout {
    pub fn checked(g: Qwen35AttentionGeometry, capacity: usize) -> Result<Self, Error> {
        g.validate().map_err(Error::Roles)?;
        if capacity == 0 {
            return Err(Error::Shape);
        }
        let query = g
            .query_heads
            .checked_mul(g.head_dim)
            .ok_or(Error::Overflow)?;
        let kv = g.kv_heads.checked_mul(g.head_dim).ok_or(Error::Overflow)?;
        capacity
            .checked_mul(kv)
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::Overflow)?;
        let mut floats = 0usize;
        // q_gate, q, k, v, norm weights, attention, output, one head's scores.
        for n in [
            query, query, query, kv, kv, g.head_dim, g.head_dim, query, g.hidden, capacity,
        ] {
            floats = floats.checked_add(n).ok_or(Error::Overflow)?;
        }
        Ok(Self {
            floats,
            bytes: floats.checked_mul(4).ok_or(Error::Overflow)?,
            query,
            kv,
            capacity,
        })
    }
}
