use super::*;
    use crate::numerical::validate::validate_wgsl;

    /// The N-body kernel must naga-validate and expose the `nbody_step` entry point.
    #[test]
    fn kinematics_wgsl_validates() {
        let report = validate_wgsl(KIN_STEP_WGSL).expect("kinematics WGSL must naga-validate");
        assert!(
            report.entry_points.iter().any(|e| e == KIN_STEP_ENTRY),
            "validated module must expose {KIN_STEP_ENTRY}; got {:?}",
            report.entry_points
        );
    }

    /// Two equal positive charges on the x-axis must repel: with `coupling = +1`, the
    /// left particle ends up moving in `−x` and the right in `+x`, so they separate.
    #[test]
    fn nbody_oracle_like_charges_repel() {
        // p0 at x=0, p1 at x=1; both q=+1, m=1, at rest.
        let state = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, // particle 0
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, // particle 1
        ];
        let out = nbody_step_cpu(&state, 0.1, 1e-4, 1.0);
        // Particle 0 (left) pushed −x → new x < 0; particle 1 (right) pushed +x → x > 1.
        assert!(out[0] < 0.0, "left particle should move −x, got {}", out[0]);
        assert!(
            out[8] > 1.0,
            "right particle should move +x, got {}",
            out[8]
        );
        // x-velocities are equal and opposite (symmetric two-body).
        assert!((out[3] + out[11]).abs() < 1e-5, "momentum not conserved");
        // Mass/charge slots carried through unchanged.
        assert_eq!(out[6], 1.0);
        assert_eq!(out[7], 1.0);
    }

    /// Opposite charges attract: `q0=+1, q1=−1, coupling=+1` → they move together.
    #[test]
    fn nbody_oracle_opposite_charges_attract() {
        let state = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, //
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, //
        ];
        let out = nbody_step_cpu(&state, 0.1, 1e-4, 1.0);
        assert!(out[0] > 0.0, "left particle should move +x, got {}", out[0]);
        assert!(
            out[8] < 1.0,
            "right particle should move −x, got {}",
            out[8]
        );
    }

    /// GPU certify: the kernel on a real adapter must match the CPU oracle within f32
    /// tolerance over a deterministic multi-particle scene. Run by the orchestrator.
    #[test]
    #[serial_test::serial(gpu)]
    fn nbody_gpu_matches_oracle() {
        if !crate::numerical::test_gpu_available() {
            return;
        }
        let count = 128usize;
        let mut state = Vec::with_capacity(count * KIN_STRIDE);
        for i in 0..count {
            let f = i as f32;
            state.extend_from_slice(&[
                (f * 0.21) - 12.0,
                (f * 0.13) - 8.0,
                (f * 0.07) - 4.0,
                0.0,
                0.0,
                0.0,
                1.0 + (f % 3.0),
                if i % 2 == 0 { 1.0 } else { -1.0 },
            ]);
        }
        let (dt, soft, coupling) = (0.005f32, 1e-2, 1.0);
        let expected = nbody_step_cpu(&state, dt, soft, coupling);
        let gpu = nbody_step_gpu(&state, dt, soft, coupling).expect("nbody_step_gpu");
        assert_eq!(gpu.len(), expected.len());
        for (g, e) in gpu.iter().zip(expected.iter()) {
            let tol = 1e-3 * e.abs().max(1.0);
            assert!((g - e).abs() <= tol, "GPU/CPU mismatch: {g} vs {e}");
        }
    }
