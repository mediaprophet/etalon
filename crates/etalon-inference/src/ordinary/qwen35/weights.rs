//! Borrowed CPU weights. No tensor ownership, I/O or activation quantization.
use super::{finite, Error, Host};
use etalon_model::inference::{qwen_roles::Projection, TensorInfo, WeightEncoding};

/// Resolve the exact payload of `tensor`, including row-range descriptors. The
/// returned borrow keeps the source owner alive throughout each provider call.
/// Sources must return stable, immutable bytes and perform no hot-path allocation.
pub trait TensorBytes {
    fn tensor_bytes(&self, tensor: &TensorInfo) -> Result<&[u8], Error>;
}

/// A contiguous borrowed data block; descriptor offsets are relative to `bytes`.
pub struct SliceBytes<'a> {
    pub bytes: &'a [u8],
}
impl TensorBytes for SliceBytes<'_> {
    fn tensor_bytes(&self, tensor: &TensorInfo) -> Result<&[u8], Error> {
        let (_, len) = layout(tensor)?;
        let start = usize::try_from(tensor.byte_offset).map_err(|_| Error::Overflow)?;
        let end = start.checked_add(len).ok_or(Error::Overflow)?;
        self.bytes.get(start..end).ok_or(Error::Provider)
    }
}

/// Row-major F32/F16/BF16 and FP8 E4M3FN CPU matrix/vector access.
/// FP8 projections require a positive finite scale per row, interpreted as
/// `sum_col((decode(weight[row,col]) * scale[row]) * input[col])`.
/// Other encodings reject scales. This does not quantize activations.
pub struct CpuByteHost<'a, S: TensorBytes + ?Sized> {
    pub source: &'a S,
}

fn width(encoding: WeightEncoding) -> Result<usize, Error> {
    match encoding {
        WeightEncoding::IeeeF32 => Ok(4),
        WeightEncoding::IeeeF16 | WeightEncoding::BrainF16 => Ok(2),
        WeightEncoding::Float8E4M3 => Ok(1),
        _ => Err(Error::UnsupportedProjection),
    }
}
fn layout(info: &TensorInfo) -> Result<(usize, usize), Error> {
    let dims = usize::try_from(info.n_dims).map_err(|_| Error::Overflow)?;
    if !(1..=4).contains(&dims) {
        return Err(Error::Shape);
    }
    let count = info.dims[..dims].iter().try_fold(1usize, |count, &dim| {
        let dim = usize::try_from(dim).map_err(|_| Error::Overflow)?;
        if dim == 0 {
            return Err(Error::Shape);
        }
        count.checked_mul(dim).ok_or(Error::Overflow)
    })?;
    let len = count
        .checked_mul(width(info.encoding())?)
        .ok_or(Error::Overflow)?;
    info.byte_offset
        .checked_add(u64::try_from(len).map_err(|_| Error::Overflow)?)
        .ok_or(Error::Overflow)?;
    Ok((count, len))
}

struct View<'a> {
    bytes: &'a [u8],
    encoding: WeightEncoding,
    count: usize,
}
impl View<'_> {
    fn scalar(&self, index: usize) -> Result<f32, Error> {
        if index >= self.count {
            return Err(Error::Shape);
        }
        let value = match self.encoding {
            WeightEncoding::IeeeF32 => {
                let i = index * 4;
                f32::from_le_bytes(
                    self.bytes[i..i + 4]
                        .try_into()
                        .map_err(|_| Error::Provider)?,
                )
            }
            WeightEncoding::IeeeF16 => {
                let i = index * 2;
                half::f16::from_le_bytes(
                    self.bytes[i..i + 2]
                        .try_into()
                        .map_err(|_| Error::Provider)?,
                )
                .to_f32()
            }
            WeightEncoding::BrainF16 => {
                let i = index * 2;
                let bits = u16::from_le_bytes(
                    self.bytes[i..i + 2]
                        .try_into()
                        .map_err(|_| Error::Provider)?,
                );
                f32::from_bits(u32::from(bits) << 16)
            }
            WeightEncoding::Float8E4M3 => {
                let code = self.bytes[index];
                let magnitude = code & 0x7f;
                if magnitude == 0x7f {
                    return Err(Error::Provider);
                }
                let exponent = magnitude >> 3;
                let mantissa = magnitude & 7;
                let absolute = if exponent == 0 {
                    f32::from(mantissa) * (1.0 / 512.0)
                } else {
                    // E4M3FN bias 7, including finite exponent-15 encodings.
                    f32::from_bits((u32::from(exponent) + 120) << 23 | u32::from(mantissa) << 20)
                };
                if code & 0x80 != 0 {
                    -absolute
                } else {
                    absolute
                }
            }
            _ => return Err(Error::UnsupportedProjection),
        };
        finite(&[value])?;
        Ok(value)
    }
}
impl<S: TensorBytes + ?Sized> CpuByteHost<'_, S> {
    fn view(&self, info: &TensorInfo) -> Result<View<'_>, Error> {
        let (count, len) = layout(info)?;
        let bytes = self.source.tensor_bytes(info)?;
        if bytes.len() != len {
            return Err(Error::Provider);
        }
        Ok(View {
            bytes,
            encoding: info.encoding(),
            count,
        })
    }
    fn projection(
        &self,
        p: &Projection,
    ) -> Result<(View<'_>, Option<View<'_>>, usize, usize), Error> {
        if p.weight.n_dims != 2 {
            return Err(Error::Shape);
        }
        let cols = usize::try_from(p.weight.dims[0]).map_err(|_| Error::Overflow)?;
        let rows = usize::try_from(p.weight.dims[1]).map_err(|_| Error::Overflow)?;
        let weights = self.view(&p.weight)?;
        let scales = match (weights.encoding, &p.row_scale) {
            (WeightEncoding::Float8E4M3, Some(info)) => {
                if info.n_dims != 1
                    || info.dims[0] != p.weight.dims[1]
                    || !matches!(
                        info.encoding(),
                        WeightEncoding::IeeeF32
                            | WeightEncoding::IeeeF16
                            | WeightEncoding::BrainF16
                    )
                {
                    return Err(Error::UnsupportedProjection);
                }
                Some(self.view(info)?)
            }
            (WeightEncoding::Float8E4M3, None) | (_, Some(_)) => {
                return Err(Error::UnsupportedProjection)
            }
            (_, None) => None,
        };
        // Admission validates all data, including scales, before any execution.
        for i in 0..weights.count {
            weights.scalar(i)?;
        }
        if let Some(scales) = &scales {
            for i in 0..rows {
                if scales.scalar(i)? <= 0.0 {
                    return Err(Error::Provider);
                }
            }
        }
        Ok((weights, scales, cols, rows))
    }
}
impl<S: TensorBytes + ?Sized> Host for CpuByteHost<'_, S> {
    fn admits(&self, p: &Projection) -> bool {
        self.projection(p).is_ok()
    }
    fn gemm(&self, p: &Projection, input: &[f32], out: &mut [f32]) -> Result<(), Error> {
        let (weights, scales, cols, rows) = self.projection(p)?;
        if input.len() != cols || out.len() != rows {
            return Err(Error::Shape);
        }
        finite(input)?;
        for (row, dst) in out.iter_mut().enumerate() {
            let scale = match &scales {
                Some(s) => s.scalar(row)?,
                None => 1.0,
            };
            let mut sum = 0.0;
            for (col, &x) in input.iter().enumerate() {
                let w = weights.scalar(row * cols + col)? * scale;
                finite(&[w])?;
                sum += w * x;
                finite(&[sum])?;
            }
            *dst = sum;
        }
        Ok(())
    }
    fn read(&self, tensor: &TensorInfo, out: &mut [f32]) -> Result<(), Error> {
        let view = self.view(tensor)?;
        if view.count != out.len() {
            return Err(Error::Shape);
        }
        // Preflight so an invalid scalar never leaves a partial read result.
        for i in 0..view.count {
            view.scalar(i)?;
        }
        if view.encoding == WeightEncoding::Float8E4M3 {
            for (i, dst) in out.iter_mut().enumerate() {
                *dst = view.scalar(i)?;
            }
        } else {
            let written = etalon_model::ggml::dequantize_row_into(
                view.bytes,
                tensor.ggml_type,
                view.count,
                out,
            )
            .map_err(|_| Error::Provider)?;
            if written != view.count {
                return Err(Error::Provider);
            }
        }
        finite(out)
    }
}
