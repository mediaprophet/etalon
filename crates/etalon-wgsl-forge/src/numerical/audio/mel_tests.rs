use super::*;
use crate::numerical::validate::validate_wgsl;
    /// The mel-apply kernel must naga-validate under naga 30 and expose the `mel_apply`
    /// entry point. Runs without a GPU — the always-on proof the shader is valid WGSL.
    #[test]
    fn mel_apply_wgsl_validates() {
        let report = validate_wgsl(MEL_APPLY_WGSL).expect("mel-apply WGSL must naga-validate");
        assert!(
            report.entry_points.iter().any(|e| e == MEL_APPLY_ENTRY),
            "validated module must expose {MEL_APPLY_ENTRY}; got {:?}",
            report.entry_points
        );
    }

    /// Hand-computed small case: n_frames=2, n_bins=4, n_mel=2 with a known filterbank.
    /// Two triangular-ish bands, each covering two of the four bins.
    #[test]
    fn mel_apply_cpu_matches_reference() {
        // spectrum: 2 frames × 4 bins.
        //   frame 0: [1, 2, 3, 4]
        //   frame 1: [5, 6, 7, 8]
        let spectrum = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // mel_fb: 2 mel bands × 4 bins.
        //   band 0: [1, 1, 0, 0]  (low bins)
        //   band 1: [0, 0, 1, 1]  (high bins)
        let mel_fb = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        let out = mel_apply_cpu(&spectrum, &mel_fb, 2, 4, 2);

        // frame 0: band0 = 1+2 = 3 ; band1 = 3+4 = 7
        // frame 1: band0 = 5+6 = 11; band1 = 7+8 = 15
        assert_eq!(out, vec![3.0, 7.0, 11.0, 15.0]);
    }

