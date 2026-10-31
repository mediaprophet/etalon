use super::OnePoleLowPass;
use crate::SignalError;

#[test]
fn filters_into_caller_buffer_and_preserves_extra_capacity() {
    let mut filter = OnePoleLowPass::new(0.5, 0.0).unwrap();
    let mut output = [99.0; 4];
    assert_eq!(filter.process_into(&[2.0, 2.0, 2.0], &mut output), Ok(3));
    assert_eq!(&output[..3], &[1.0, 1.5, 1.75]);
    assert_eq!(output[3], 99.0);
    assert_eq!(filter.state(), 1.75);
}

#[test]
fn validates_parameters_and_rejects_invalid_samples_transactionally() {
    assert!(OnePoleLowPass::new(f32::NAN, 0.0).is_err());
    assert!(OnePoleLowPass::new(1.1, 0.0).is_err());
    assert!(OnePoleLowPass::new(0.5, f32::INFINITY).is_err());

    let mut filter = OnePoleLowPass::new(0.5, 3.0).unwrap();
    let mut output = [17.0; 2];
    assert_eq!(
        filter.process_into(&[1.0, f32::NAN], &mut output),
        Err(SignalError::NonFiniteInput)
    );
    assert_eq!(output, [17.0; 2]);
    assert_eq!(filter.state(), 3.0);
}

#[test]
fn rejects_short_output_buffer_without_changing_state() {
    let mut filter = OnePoleLowPass::new(0.25, -1.0).unwrap();
    let mut output = [8.0; 1];
    assert_eq!(
        filter.process_into(&[0.0, 1.0], &mut output),
        Err(SignalError::LengthMismatch)
    );
    assert_eq!(output, [8.0]);
    assert_eq!(filter.state(), -1.0);
}
