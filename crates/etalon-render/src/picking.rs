//! R32Uint pick encoding with full-width caller-owned ID snapshots.

pub const MESH_PICK_FLAG: u32 = 1 << 31;
pub const MESH_PICK_SLOT_MASK: u32 = !MESH_PICK_FLAG;

pub fn encode_mesh_slot(slot: u32) -> Option<u32> {
    (slot < MESH_PICK_FLAG).then_some(MESH_PICK_FLAG | slot)
}

pub fn decode_mesh_slot(encoded: u32) -> Option<usize> {
    (encoded & MESH_PICK_FLAG != 0).then_some((encoded & MESH_PICK_SLOT_MASK) as usize)
}

pub fn snapshot_ids<T>(
    source: &[T],
    out: &mut [u64],
    identity: impl Fn(&T) -> u64,
) -> Option<usize> {
    if source.len() > out.len() || source.len() >= MESH_PICK_FLAG as usize {
        return None;
    }
    for (record, slot) in source.iter().zip(out.iter_mut()) {
        *slot = identity(record);
    }
    Some(source.len())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickResult {
    NoHit,
    Tensor(u32),
    Mesh(u64),
    InvalidSlot,
}

pub fn resolve_pick(raw: u32, snapshot: &[u64]) -> PickResult {
    if raw == 0 {
        return PickResult::NoHit;
    }
    if let Some(slot) = decode_mesh_slot(raw) {
        snapshot
            .get(slot)
            .copied()
            .map(PickResult::Mesh)
            .unwrap_or(PickResult::InvalidSlot)
    } else {
        PickResult::Tensor(raw - 1)
    }
}
