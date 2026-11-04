//! Public CPU host for synthetic f32 row-major projection fixtures.
use super::{finite, Error, Host};
use etalon_model::inference::{qwen_roles::Projection, TensorInfo};

/// Tensor byte offsets address `values` (four bytes per element). This host
/// rejects all scaled/quantized roles; it makes no local checkpoint claim.
pub struct F32FixtureHost<'a> {
    pub values: &'a [f32],
}
impl F32FixtureHost<'_> {
    fn data(&self, info: &TensorInfo, count: usize) -> Result<&[f32], Error> {
        if info.ggml_type != 0 || !info.byte_offset.is_multiple_of(4) {
            return Err(Error::Provider);
        }
        let start = usize::try_from(info.byte_offset / 4).map_err(|_| Error::Overflow)?;
        let end = start.checked_add(count).ok_or(Error::Overflow)?;
        let data = self.values.get(start..end).ok_or(Error::Provider)?;
        finite(data)?;
        Ok(data)
    }
}
impl Host for F32FixtureHost<'_> {
    fn admits(&self, p: &Projection) -> bool {
        p.weight.ggml_type == 0 && p.weight.n_dims == 2 && p.row_scale.is_none()
    }
    fn gemm(&self, p: &Projection, input: &[f32], out: &mut [f32]) -> Result<(), Error> {
        if !self.admits(p) {
            return Err(Error::UnsupportedProjection);
        }
        let (cols, rows) = (p.cols(), p.rows());
        if cols == 0 || rows == 0 || input.len() != cols || out.len() != rows {
            return Err(Error::Shape);
        }
        let weights = self.data(&p.weight, cols.checked_mul(rows).ok_or(Error::Overflow)?)?;
        finite(input)?;
        for (row, dst) in weights.chunks_exact(cols).zip(out.iter_mut()) {
            *dst = row.iter().zip(input).map(|(w, x)| w * x).sum();
        }
        finite(out)
    }
    fn read(&self, info: &TensorInfo, out: &mut [f32]) -> Result<(), Error> {
        let dims = usize::try_from(info.n_dims).map_err(|_| Error::Overflow)?;
        if dims == 0 || dims > 4 {
            return Err(Error::Shape);
        }
        let count = info.dims[..dims].iter().try_fold(1usize, |n, &d| {
            n.checked_mul(usize::try_from(d).map_err(|_| Error::Overflow)?)
                .ok_or(Error::Overflow)
        })?;
        if count != out.len() {
            return Err(Error::Shape);
        }
        out.copy_from_slice(self.data(info, count)?);
        Ok(())
    }
}
