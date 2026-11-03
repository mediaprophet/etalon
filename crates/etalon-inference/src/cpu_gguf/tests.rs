use super::*;
use etalon_inference_kernel::dense_cpu::TensorKernels;
use etalon_model::gguf::{ParseLimits, VersionPolicy};

fn string(out: &mut Vec<u8>, s: &[u8]) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s);
}
fn fixture() -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&11u64.to_le_bytes());
    out.extend_from_slice(&5u64.to_le_bytes());
    string(&mut out, b"general.architecture");
    out.extend_from_slice(&8u32.to_le_bytes());
    string(&mut out, b"llama");
    for (key, v) in [
        (b"llama.block_count".as_slice(), 1u32),
        (b"llama.embedding_length", 2),
        (b"llama.attention.head_count", 1),
        (b"llama.feed_forward_length", 2),
    ] {
        string(&mut out, key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    let names = [
        "token_embd.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ];
    let mut offset = 0u64;
    for name in names {
        let norm = name.contains("norm");
        string(&mut out, name.as_bytes());
        out.extend_from_slice(&(if norm { 1u32 } else { 2 }).to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes());
        if !norm {
            out.extend_from_slice(&2u64.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += if norm { 8 } else { 16 };
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    for i in 0..offset / 4 {
        out.extend_from_slice(&((i % 7) as f32 * 0.25 - 0.5).to_le_bytes());
    }
    out
}
fn limits() -> ParseLimits {
    ParseLimits {
        max_metadata_bytes: 8192,
        max_nesting_depth: 2,
        max_work: 1000,
        versions: VersionPolicy::V2V3,
    }
}
fn bindings() -> BindingLimits {
    BindingLimits {
        byte_budget: 8192,
        max_tensor_visits: 1000,
    }
}

#[test]
fn real_payload_output_fallback_and_native_gemv() {
    let bytes = fixture();
    let config = inspect_dense_gguf(&bytes, limits(), 4).unwrap();
    assert_eq!(config.numerical_config().rope_base, 100_000.0);
    let mut storage = [empty_layer(); 2];
    let model = prepare_dense_gguf_into(&config, &mut storage, bindings()).unwrap();
    assert_eq!(model.layers.len(), 1);
    let mut kernels = GgmlCpuKernels;
    let mut row = [0.0; 3];
    assert_eq!(
        kernels
            .decode_row_into(model.embedding, 1, &mut row)
            .unwrap(),
        2
    );
    assert_eq!(row, [0.0, 0.25, 0.0]);
    let mut result = [99.0; 3];
    kernels
        .gemv_into(model.output.unwrap(), &[2.0, 3.0], &mut result)
        .unwrap();
    assert_eq!(result, [-1.75, 0.75, 99.0]);
    let mut forged = model.embedding;
    forged.input_width = 1;
    assert_eq!(
        kernels.decode_row_into(forged, 0, &mut row),
        Err(GgufCpuError::InvalidDimensions)
    );
    assert_eq!(
        kernels.decode_row_into(model.embedding, 2, &mut row),
        Err(GgufCpuError::Capacity)
    );
    assert!(storage[1].q.is_none());
}
#[test]
fn refusals_preserve_storage() {
    let bytes = fixture();
    let config = inspect_dense_gguf(&bytes, limits(), 4).unwrap();
    assert_eq!(
        prepare_dense_gguf_into(&config, &mut [], bindings()).unwrap_err(),
        GgufCpuError::Capacity
    );
    let mut storage = [empty_layer(); 1];
    assert_eq!(
        prepare_dense_gguf_into(
            &config,
            &mut storage,
            BindingLimits {
                byte_budget: 0,
                ..bindings()
            }
        )
        .unwrap_err(),
        GgufCpuError::Budget
    );
    assert_eq!(
        prepare_dense_gguf_into(
            &config,
            &mut storage,
            BindingLimits {
                max_tensor_visits: 0,
                ..bindings()
            }
        )
        .unwrap_err(),
        GgufCpuError::Budget
    );
    let truncated = &bytes[..bytes.len() - 1];
    let config = inspect_dense_gguf(truncated, limits(), 4).unwrap();
    assert_eq!(
        prepare_dense_gguf_into(&config, &mut storage, bindings()).unwrap_err(),
        GgufCpuError::Payload
    );
    assert!(storage[0].q.is_none());
    assert_eq!(
        inspect_dense_gguf(&bytes, limits(), 0).unwrap_err(),
        GgufCpuError::InvalidDimensions
    );
}
