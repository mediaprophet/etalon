/// ggml types supported by coop fused FFN (incl. Q4_K_SOA for 3B layouts). Not F16.
pub(super) fn fused_ffn_quant_supported(ggml_type: u32) -> bool {
    use etalon_model::ggml::{
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K_SOA, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K,
        GGML_TYPE_Q8_0,
    };
    matches!(
        ggml_type,
        GGML_TYPE_Q4_0
            | GGML_TYPE_Q5_0
            | GGML_TYPE_Q8_0
            | GGML_TYPE_Q4_K
            | GGML_TYPE_Q4_K_SOA
            | GGML_TYPE_Q6_K
    )
}
