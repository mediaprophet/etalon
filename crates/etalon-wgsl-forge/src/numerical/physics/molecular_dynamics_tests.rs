use super::*;
    use crate::numerical::validate::validate_wgsl;

    /// The MD kernel must naga-validate and expose the `md_step` entry point.
    #[test]
    fn md_wgsl_validates() {
        let report = validate_wgsl(MD_STEP_WGSL).expect("MD WGSL must naga-validate");
        assert!(
            report.entry_points.iter().any(|e| e == MD_STEP_ENTRY),
            "validated module must expose {MD_STEP_ENTRY}; got {:?}",
            report.entry_points
        );
    }

    /// Hand-checked single molecule, no PBC wrap (box large). x0=0, v0=1, f=2, m=2,
    /// dt=0.5 → a=1: x = 0 + 1·0.5 + ½·1·0.25 = 0.625; v = 1 + 1·0.5 = 1.5. The mass
    /// slot and the un-accelerated y/z axes stay put.
    #[test]
    fn md_oracle_hand_checked() {
        // [px,py,pz, vx,vy,vz, fx,fy,fz, mass]
        let mut s = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0];
        md_step_cpu(&mut s, [100.0, 100.0, 100.0], 0.5);
        assert!((s[0] - 0.625).abs() < 1e-6, "px = {}", s[0]);
        assert!((s[3] - 1.5).abs() < 1e-6, "vx = {}", s[3]);
        assert_eq!(s[1], 0.0);
        assert_eq!(s[9], 2.0); // mass untouched
    }

    /// PBC: a molecule at x=9.9 moving +x past a box of 10 wraps back near 0.
    #[test]
    fn md_oracle_pbc_wraps() {
        let mut s = vec![9.9, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        md_step_cpu(&mut s, [10.0, 0.0, 0.0], 0.5); // x → 9.9+0.5 = 10.4 → wrap → 0.4
        assert!((s[0] - 0.4).abs() < 1e-5, "wrapped px = {}", s[0]);
    }

    /// GPU certify: the kernel on a real adapter must match the CPU oracle within f32
    /// tolerance over a deterministic multi-molecule scene. Run by the orchestrator.
    #[test]
    #[serial_test::serial(gpu)]
    fn md_gpu_matches_oracle() {
        if !crate::numerical::test_gpu_available() {
            return;
        }
        let count = 200usize;
        let mut state = Vec::with_capacity(count * MD_STRIDE);
        for i in 0..count {
            let f = i as f32;
            state.extend_from_slice(&[
                (f * 0.13) % 10.0,
                (f * 0.27) % 10.0,
                (f * 0.07) % 10.0,
                (f * 0.01) - 1.0,
                0.5 - (f * 0.013),
                (f * 0.005),
                (f * 0.02) - 1.0,
                (f * 0.011) - 0.5,
                0.3,
                1.0 + (f % 5.0) * 0.5,
            ]);
        }
        let dt = 0.01f32;
        let bx = [10.0f32, 10.0, 10.0];
        let mut expected = state.clone();
        md_step_cpu(&mut expected, bx, dt);
        let gpu = md_step_gpu(&state, bx, dt).expect("md_step_gpu");
        assert_eq!(gpu.len(), expected.len());
        for (g, e) in gpu.iter().zip(expected.iter()) {
            assert!((g - e).abs() <= 1e-3, "GPU/CPU mismatch: {g} vs {e}");
        }
    }
