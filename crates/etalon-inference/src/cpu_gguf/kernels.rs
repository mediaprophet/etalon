use super::{GgufCpuError as Error, PackedTensor};
use etalon_inference_kernel::dense_cpu::{TensorBinding, TensorKernels};

pub const MAX_INPUT_WIDTH: usize = 10_240;

#[derive(Debug, Default)]
pub struct GgmlCpuKernels;

impl GgmlCpuKernels {
    fn checked(t: TensorBinding<PackedTensor<'_>>) -> Result<PackedTensor<'_>, Error> {
        if t.input_width != t.handle.width || t.output_rows != t.handle.rows {
            Err(Error::InvalidDimensions)
        } else {
            Ok(t.handle)
        }
    }
    fn row(t: PackedTensor<'_>, row: usize, out: &mut [f32]) -> Result<usize, Error> {
        if row >= t.rows || out.len() < t.width {
            return Err(Error::Capacity);
        }
        let start = row
            .checked_mul(t.row_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        let raw = t
            .bytes
            .get(start..start + t.row_bytes)
            .ok_or(Error::Payload)?;
        etalon_model::ggml::dequantize_row_into(raw, t.dtype, t.width, out)
            .map_err(|_| Error::Kernel)
    }
}
impl TensorKernels<PackedTensor<'_>> for GgmlCpuKernels {
    type Error = Error;
    fn decode_row_into(
        &mut self,
        tensor: TensorBinding<PackedTensor<'_>>,
        row: usize,
        out: &mut [f32],
    ) -> Result<usize, Error> {
        Self::row(Self::checked(tensor)?, row, out)
    }
    fn gemv_into(
        &mut self,
        tensor: TensorBinding<PackedTensor<'_>>,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), Error> {
        let t = Self::checked(tensor)?;
        if input.len() < t.width || out.len() < t.rows || t.width > MAX_INPUT_WIDTH {
            return Err(Error::Capacity);
        }
        let mut row = [0.0f32; MAX_INPUT_WIDTH];
        for (i, dst) in out[..t.rows].iter_mut().enumerate() {
            if Self::row(t, i, &mut row[..t.width])? != t.width {
                return Err(Error::Kernel);
            }
            *dst = row[..t.width]
                .iter()
                .zip(&input[..t.width])
                .map(|(w, x)| w * x)
                .sum();
        }
        Ok(())
    }
}
