#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_wave_emits_wave_intrinsics() {
        let kernel = crate::numerical::BuiltinKernel::Gemv.spec();
        let schedule = Schedule {
            workgroup_size: 64,
            ..Default::default()
        };
        let mut source = String::new();
        emit_gemv_wave_hlsl(&mut source, &kernel, schedule).unwrap();
        assert!(source.contains("WaveActiveSum"));
        assert!(source.contains("WaveGetLaneCount"));
        assert!(source.contains("WaveGetLaneIndex"));
    }

    #[test]
    fn fused_ffn_wave_emits_wave_intrinsics() {
        let kernel = crate::numerical::BuiltinKernel::FusedFfn.spec();
        let schedule = Schedule {
            workgroup_size: 64,
            ..Default::default()
        };
        let mut source = String::new();
        emit_fused_ffn_wave_hlsl(&mut source, &kernel, schedule).unwrap();
        assert!(source.contains("WaveActiveSum"));
        assert!(source.contains("WaveGetLaneCount"));
    }

    #[test]
    fn topk_wave_emits_groupshared() {
        let kernel = crate::numerical::BuiltinKernel::TopK.spec();
        let schedule = Schedule {
            workgroup_size: 256,
            ..Default::default()
        };
        let mut source = String::new();
        emit_topk_wave_hlsl(&mut source, &kernel, schedule).unwrap();
        assert!(source.contains("groupshared"));
        assert!(source.contains("GroupMemoryBarrierWithGroupSync"));
    }
}
