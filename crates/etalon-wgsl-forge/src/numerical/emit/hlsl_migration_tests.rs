#[cfg(test)]
mod wavematrix_tests {
    use super::*;

    #[test]
    fn emit_gemv_wavematrix_emits_intrinsics() {
        let kernel = KernelSpec {
            id: "gemv".to_string(),
            semantic_version: 1,
            entry_point: "gemv_wm".to_string(),
            description: "WaveMatrix GEMV".to_string(),
            buffers: Vec::new(),
            ops: Vec::new(),
            shared_memory: Vec::new(),
        };
        let schedule = Schedule {
            workgroup_size: 32,
            ..Default::default()
        };
        let mut source = String::new();
        emit_gemv_wavematrix_hlsl(&mut source, &kernel, schedule).expect("wavematrix emit");
        assert!(source.contains("WaveMatrixA"), "should use WaveMatrixA");
        assert!(source.contains("WaveMatrixB"), "should use WaveMatrixB");
        assert!(source.contains("WaveMatrixC"), "should use WaveMatrixC");
        assert!(
            source.contains("WaveMatrixMultiply"),
            "should call WaveMatrixMultiply"
        );
        assert!(source.contains("gemv_wm"), "should contain entry point");
    }
}
