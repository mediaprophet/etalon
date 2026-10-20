//! Validated tensor views and quantization formats.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantScheme {
    Fp32,
    Fp16,
    Bf16,
    Q4KSoa,
    Q8_0,
    Q8_1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorView {
    pub scheme: QuantScheme,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub byte_offset: usize,
    pub total_bytes: usize,
}

impl TensorView {
    pub fn new(scheme: QuantScheme, shape: Vec<usize>, byte_offset: usize, total_bytes: usize) -> Self {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1usize;
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride = stride.saturating_mul(dim);
        }
        strides.reverse();
        Self {
            scheme,
            shape,
            strides,
            byte_offset,
            total_bytes,
        }
    }
}
