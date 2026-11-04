#![cfg(feature = "inference-model")]
use etalon_inference::ordinary::qwen35::{weights::*, Error, Host};
use etalon_model::{
    ggml::*,
    inference::{qwen_roles::Projection, TensorInfo},
};

fn info(kind: u32, cols: u64, rows: u64, offset: u64) -> TensorInfo {
    TensorInfo {
        dims: [cols, rows, 0, 0],
        n_dims: if rows == 0 { 1 } else { 2 },
        ggml_type: kind,
        byte_offset: offset,
    }
}
#[test]
fn little_endian_scalars_and_row_major_gemm() {
    for (kind, bytes) in [
        (
            GGML_TYPE_F32,
            [1f32, -2., 3., 4.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            GGML_TYPE_F16,
            [0x3c00u16, 0xc000, 0x4200, 0x4400]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
        (
            GGML_TYPE_BF16,
            [0x3f80u16, 0xc000, 0x4040, 0x4080]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
    ] {
        let source = SliceBytes { bytes: &bytes };
        let host = CpuByteHost { source: &source };
        let p = Projection {
            weight: info(kind, 2, 2, 0),
            row_scale: None,
        };
        assert!(host.admits(&p));
        let mut out = [0.; 2];
        host.gemm(&p, &[2., 3.], &mut out).unwrap();
        assert_eq!(out, [-4., 18.]);
        let mut read = [0.; 4];
        host.read(&info(kind, 4, 0, 0), &mut read).unwrap();
        assert_eq!(read, [1., -2., 3., 4.]);
    }
}
#[test]
fn fp8_fn_boundaries_and_explicit_row_scale() {
    let mut bytes = vec![0x38, 0x40, 0xb8, 0x48]; // [1,2],[-1,4]
    bytes.extend_from_slice(&2f32.to_le_bytes());
    bytes.extend_from_slice(&0.5f32.to_le_bytes());
    let source = SliceBytes { bytes: &bytes };
    let host = CpuByteHost { source: &source };
    let p = Projection {
        weight: info(GGML_TYPE_FP8_E4M3, 2, 2, 0),
        row_scale: Some(info(GGML_TYPE_F32, 2, 0, 4)),
    };
    let mut out = [0.; 2];
    host.gemm(&p, &[3., 5.], &mut out).unwrap();
    assert_eq!(out, [26., 8.5]);
    assert!(!host.admits(&Projection {
        row_scale: None,
        ..p
    }));
    let codes = [0, 0x80, 1, 7, 8, 0x78, 0x7e, 0xfe];
    let source = SliceBytes { bytes: &codes };
    let host = CpuByteHost { source: &source };
    let mut values = [0.; 8];
    host.read(&info(GGML_TYPE_FP8_E4M3, 8, 0, 0), &mut values)
        .unwrap();
    assert_eq!(
        values,
        [0., -0., 1. / 512., 7. / 512., 1. / 64., 256., 448., -448.]
    );
    assert_eq!(values[1].to_bits(), (-0f32).to_bits());
}
#[test]
fn rejects_nonfinite_truncated_overflow_and_bad_scales() {
    for (kind, bytes) in [
        (GGML_TYPE_F32, f32::INFINITY.to_le_bytes().to_vec()),
        (GGML_TYPE_F16, 0x7c00u16.to_le_bytes().to_vec()),
        (GGML_TYPE_BF16, 0x7fc0u16.to_le_bytes().to_vec()),
        (GGML_TYPE_FP8_E4M3, vec![0x7f]),
        (GGML_TYPE_FP8_E4M3, vec![0xff]),
    ] {
        let source = SliceBytes { bytes: &bytes };
        let host = CpuByteHost { source: &source };
        let mut out = [123.];
        assert!(host.read(&info(kind, 1, 0, 0), &mut out).is_err());
        assert_eq!(out, [123.]);
    }
    for scale in [0., -1., f32::NAN, f32::INFINITY] {
        let mut bytes = vec![0x38];
        bytes.extend_from_slice(&scale.to_le_bytes());
        let source = SliceBytes { bytes: &bytes };
        let host = CpuByteHost { source: &source };
        assert!(!host.admits(&Projection {
            weight: info(GGML_TYPE_FP8_E4M3, 1, 1, 0),
            row_scale: Some(info(GGML_TYPE_F32, 1, 0, 1))
        }));
    }
    let source = SliceBytes { bytes: &[0; 3] };
    let host = CpuByteHost { source: &source };
    assert!(host.read(&info(GGML_TYPE_F32, 1, 0, 0), &mut [0.]).is_err());
    assert_eq!(
        host.read(&info(GGML_TYPE_F32, u64::MAX, 2, 0), &mut [0.]),
        Err(Error::Overflow)
    );
    assert!(host
        .read(&info(GGML_TYPE_Q4_K, 256, 0, 0), &mut [0.; 256])
        .is_err());
}

struct Excess;
impl TensorBytes for Excess {
    fn tensor_bytes(&self, _: &TensorInfo) -> Result<&[u8], Error> {
        Ok(&[0; 8])
    }
}
#[test]
fn callback_must_return_exact_payload() {
    let host = CpuByteHost { source: &Excess };
    assert!(host.read(&info(GGML_TYPE_F32, 1, 0, 0), &mut [0.]).is_err());
}
