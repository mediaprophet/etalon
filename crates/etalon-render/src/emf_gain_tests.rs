use super::*;
fn sample(amplitude: f32, valid: bool, gain: f32) -> [f32; 4] {
    let p = EmfSliceUniform {
        nx: 1,
        ny: 1,
        nz: 1,
        nt: 1,
        projection_mode: 1,
        amplitude_scale: gain,
        ..Default::default()
    };
    let c = EmfFieldCell {
        amplitude,
        epistemic_weight: if valid { 1. } else { 0. },
        ..Default::default()
    };
    evaluate_emf_fragment_cpu(&p, &[c], [0.5, 0.5])
}
#[test]
fn magnitude_display_gain_changes_brightness() {
    assert_eq!(sample(0.2, true, 1.), [0.2, 0.2, 0.2, 1.]);
    assert_eq!(sample(0.2, true, 4.), [0.8, 0.8, 0.8, 1.]);
}
#[test]
fn magnitude_gain_saturates_without_changing_validity() {
    assert_eq!(sample(0.8, true, 4.), [1.; 4]);
    assert_eq!(sample(0.2, false, 4.)[3], 0.);
    assert_eq!(sample(0., true, 4.), [0., 0., 0., 1.]);
}
#[test]
fn unit_gain_preserves_existing_projection_and_input() {
    let p = EmfSliceUniform {
        nx: 1,
        ny: 1,
        nz: 1,
        nt: 1,
        projection_mode: 1,
        amplitude_scale: 1.,
        ..Default::default()
    };
    let input = [EmfFieldCell {
        amplitude: 1. - (-1.0f32).exp(),
        epistemic_weight: 1.,
        ..Default::default()
    }];
    let original = input;
    let rgba = evaluate_emf_fragment_cpu(&p, &input, [0.5, 0.5]);
    assert_eq!(rgba[0], input[0].amplitude);
    assert_eq!(input, original);
}
#[test]
fn subunit_gain_dims_existing_projection() {
    assert_eq!(sample(0.8, true, 0.25), [0.2, 0.2, 0.2, 1.]);
}
