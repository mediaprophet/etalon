use super::*;
pub(super) fn capacity<T>(n: usize) -> Result<Vec<T>, Qwen4ExpDecodeError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(n)
        .map_err(|_| Qwen4ExpDecodeError::AllocationFailed)?;
    Ok(output)
}
pub(super) fn zeros<T: Clone + Default>(n: usize) -> Result<Vec<T>, Qwen4ExpDecodeError> {
    let mut output = capacity(n)?;
    output.resize(n, T::default());
    Ok(output)
}
pub(super) fn admit(required: usize, limit: u64) -> Result<(), Qwen4ExpDecodeError> {
    let required = u64::try_from(required).map_err(|_| Qwen4ExpDecodeError::SizeOverflow)?;
    if required > limit {
        return Err(Qwen4ExpDecodeError::BudgetExceeded { required, limit });
    }
    Ok(())
}
pub(super) fn add_bytes(
    total: &mut usize,
    count: usize,
    size: usize,
) -> Result<(), Qwen4ExpDecodeError> {
    *total = count
        .checked_mul(size)
        .and_then(|n| total.checked_add(n))
        .ok_or(Qwen4ExpDecodeError::SizeOverflow)?;
    Ok(())
}
