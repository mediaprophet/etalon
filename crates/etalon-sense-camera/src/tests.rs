use std::{
    cell::Cell,
    rc::Rc,
    sync::atomic::{AtomicU32, Ordering},
};

use etalon_sense_core::{extended_capability::ExtendedCapability, quantity::SampleEncoding};

use crate::{
    CameraModality, FrameDeclaration, FrameError, FrameFormat, FrameIngestAdapter,
    FramePixelFormat, FramePlane, FramePolicyGate, FramePolicyOperation, FramePolicyScope,
    FrameProvider, FrameProviderError, FrameTimestamp, NegotiatedFrame, PolicyDecision,
    ProviderFrame, RadiometricStatus,
};

fn declaration(modality: CameraModality, radiometric: RadiometricStatus) -> FrameDeclaration {
    let cap =
        ExtendedCapability::new(90, 91, None, SampleEncoding::U8, [2, 2, 0, 0], None).unwrap();
    FrameDeclaration::new(1, modality, cap, 2, 2, radiometric).unwrap()
}

fn format() -> FrameFormat {
    FrameFormat::new(
        7,
        2,
        2,
        4,
        SampleEncoding::U8,
        [
            FramePlane::new(0, 2, 2, 2),
            FramePlane::default(),
            FramePlane::default(),
            FramePlane::default(),
        ],
        1,
    )
    .unwrap()
}

struct FakeProvider {
    starts: Rc<Cell<u32>>,
    reads: Rc<Cell<u32>>,
    cancels: Rc<Cell<u32>>,
    cancel_failures: Rc<Cell<u32>>,
    source_id: u64,
    modality: CameraModality,
    radiometric: RadiometricStatus,
    panic_start: bool,
    panic_cancel: bool,
}

impl FakeProvider {
    fn new(modality: CameraModality, radiometric: RadiometricStatus) -> Self {
        Self {
            starts: Rc::new(Cell::new(0)),
            reads: Rc::new(Cell::new(0)),
            cancels: Rc::new(Cell::new(0)),
            cancel_failures: Rc::new(Cell::new(0)),
            source_id: 1,
            modality,
            radiometric,
            panic_start: false,
            panic_cancel: false,
        }
    }
}

impl FrameProvider for FakeProvider {
    fn start(&mut self) -> Result<NegotiatedFrame, FrameProviderError> {
        self.starts.set(self.starts.get() + 1);
        assert!(
            !self.panic_start,
            "simulated provider panic after start attempt"
        );
        Ok(NegotiatedFrame {
            source_id: self.source_id,
            modality: self.modality,
            format: format(),
            radiometric: self.radiometric,
        })
    }

    fn next_frame(&mut self, output: &mut [u8]) -> Result<ProviderFrame, FrameProviderError> {
        self.reads.set(self.reads.get() + 1);
        output[..4].copy_from_slice(&[10, 20, 30, 40]);
        let sequence = u64::from(self.reads.get() - 1);
        Ok(ProviderFrame {
            bytes_written: 4,
            metadata: crate::FrameMetadata {
                format_id: 7,
                sequence,
                gap_before: if sequence == 0 { None } else { Some(0) },
                timestamp: FrameTimestamp::Raw {
                    ticks: sequence as i64,
                    clock_domain_id: 3,
                    unit_id: 4,
                },
            },
        })
    }

    fn cancel(&mut self) -> Result<(), FrameProviderError> {
        self.cancels.set(self.cancels.get() + 1);
        assert!(!self.panic_cancel, "simulated provider cancellation panic");
        if self.cancel_failures.get() > 0 {
            self.cancel_failures.set(self.cancel_failures.get() - 1);
            Err(FrameProviderError::CancelFailed)
        } else {
            Ok(())
        }
    }
}

struct Gate {
    decision: PolicyDecision,
    calls: AtomicU32,
    invoke: bool,
}
impl FramePolicyGate for Gate {
    fn with_decision(
        &self,
        _scope: FramePolicyScope,
        _operation: FramePolicyOperation,
        perform: &mut dyn FnMut(),
    ) -> PolicyDecision {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.invoke && self.decision == PolicyDecision::Permit {
            perform();
        }
        self.decision
    }
}

#[test]
fn negotiated_thermal_frame_keeps_calibration_unknown_and_gap_metadata() {
    let provider = FakeProvider::new(CameraModality::Thermal, RadiometricStatus::Unknown);
    let mut adapter = FrameIngestAdapter::new_host_managed(
        provider,
        declaration(CameraModality::Thermal, RadiometricStatus::Unknown),
    );
    adapter.start(55).unwrap();
    let mut bytes = [0u8; 4];
    let first = adapter.capture(&mut bytes).unwrap();
    assert_eq!(bytes, [10, 20, 30, 40]);
    assert_eq!(first.gap_before, None);
    assert_eq!(
        adapter.negotiated().unwrap().radiometric,
        RadiometricStatus::Unknown
    );
    let second = adapter.capture(&mut bytes).unwrap();
    assert_eq!(second.gap_before, Some(0));
    adapter.cancel().unwrap();
}

#[test]
fn undersized_caller_buffer_does_not_read_or_start_an_image() {
    let provider = FakeProvider::new(CameraModality::Visible, RadiometricStatus::NotApplicable);
    let reads = Rc::clone(&provider.reads);
    let mut adapter = FrameIngestAdapter::new_host_managed(
        provider,
        declaration(CameraModality::Visible, RadiometricStatus::NotApplicable),
    );
    adapter.start(2).unwrap();
    let mut bytes = [0xAA; 3];
    assert_eq!(
        adapter.capture(&mut bytes),
        Err(FrameError::BufferTooSmall { required: 4 })
    );
    assert_eq!(bytes, [0xAA; 3]);
    assert_eq!(reads.get(), 0);
}

#[test]
fn unknown_policy_does_not_call_provider_and_deny_clears_on_bad_gate_outcome() {
    let provider = FakeProvider::new(CameraModality::NearInfrared, RadiometricStatus::Unknown);
    let starts = Rc::clone(&provider.starts);
    let gate = Gate {
        decision: PolicyDecision::Unknown,
        calls: AtomicU32::new(0),
        invoke: false,
    };
    let mut adapter = FrameIngestAdapter::with_gate(
        provider,
        declaration(CameraModality::NearInfrared, RadiometricStatus::Unknown),
        &gate,
    );
    assert_eq!(adapter.start(3), Err(FrameError::PolicyUnknown));
    assert_eq!(starts.get(), 0);
    assert_eq!(gate.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn failed_cancel_is_reported_and_retryable() {
    let provider = FakeProvider::new(CameraModality::Visible, RadiometricStatus::NotApplicable);
    provider.cancel_failures.set(1);
    let mut adapter = FrameIngestAdapter::new_host_managed(
        provider,
        declaration(CameraModality::Visible, RadiometricStatus::NotApplicable),
    );
    adapter.start(4).unwrap();
    assert_eq!(adapter.cancel(), Err(FrameError::CancelUnconfirmed));
    assert_eq!(adapter.cancel(), Ok(()));
}

#[test]
fn panicking_start_provider_is_cancelled_after_callback_attempt() {
    let mut provider = FakeProvider::new(CameraModality::Visible, RadiometricStatus::NotApplicable);
    provider.panic_start = true;
    let cancels = Rc::clone(&provider.cancels);
    let gate = Gate {
        decision: PolicyDecision::Permit,
        calls: AtomicU32::new(0),
        invoke: true,
    };
    let mut adapter = FrameIngestAdapter::with_gate(
        provider,
        declaration(CameraModality::Visible, RadiometricStatus::NotApplicable),
        &gate,
    );
    assert_eq!(adapter.start(8), Err(FrameError::PolicyUnknown));
    assert_eq!(cancels.get(), 1);
}

#[test]
fn panicking_cancel_is_contained_and_remains_retryable() {
    let mut provider = FakeProvider::new(CameraModality::Visible, RadiometricStatus::NotApplicable);
    provider.panic_start = true;
    provider.panic_cancel = true;
    let cancels = Rc::clone(&provider.cancels);
    let mut adapter = FrameIngestAdapter::new_host_managed(
        provider,
        declaration(CameraModality::Visible, RadiometricStatus::NotApplicable),
    );
    assert_eq!(adapter.start(8), Err(FrameError::CancelUnconfirmed));
    assert_eq!(adapter.cancel(), Err(FrameError::CancelUnconfirmed));
    assert_eq!(cancels.get(), 2);
}

#[test]
fn explicit_bgra_capability_accounts_for_channels_and_row_padding() {
    let cap =
        ExtendedCapability::new(90, 92, None, SampleEncoding::U8, [2, 2, 4, 0], None).unwrap();
    let declaration = FrameDeclaration::new_with_pixel_format(
        1,
        CameraModality::Visible,
        cap,
        FramePixelFormat::Bgra8,
        2,
        2,
        20,
        RadiometricStatus::NotApplicable,
    )
    .unwrap();
    let frame = FrameFormat::new_with_pixel_format(
        8,
        2,
        2,
        20,
        FramePixelFormat::Bgra8,
        [
            FramePlane::new(0, 10, 8, 2),
            FramePlane::default(),
            FramePlane::default(),
            FramePlane::default(),
        ],
        1,
    )
    .unwrap();
    assert_eq!(declaration.pixel_format(), FramePixelFormat::Bgra8);
    assert_eq!(declaration.max_frame_bytes(), 20);
    assert_eq!(frame.encoding(), SampleEncoding::U8);
    assert_eq!(frame.pixel_format(), FramePixelFormat::Bgra8);
    assert_eq!(
        crate::format::validate_negotiated_format(
            declaration,
            frame,
            RadiometricStatus::NotApplicable
        ),
        Ok(()),
    );
}

#[test]
fn gray16_unsigned_has_two_byte_pixels_and_checked_padded_rows() {
    let cap =
        ExtendedCapability::new(90, 93, None, SampleEncoding::U16, [2, 2, 0, 0], None).unwrap();
    let declaration = FrameDeclaration::new_with_pixel_format(
        1,
        CameraModality::Thermal,
        cap,
        FramePixelFormat::Gray16,
        2,
        2,
        12,
        RadiometricStatus::Unknown,
    )
    .unwrap();
    let frame = FrameFormat::new_with_pixel_format(
        9,
        2,
        2,
        12,
        FramePixelFormat::Gray16,
        [
            FramePlane::new(0, 6, 4, 2),
            FramePlane::default(),
            FramePlane::default(),
            FramePlane::default(),
        ],
        1,
    )
    .unwrap();
    assert_eq!(declaration.capability().encoding(), SampleEncoding::U16);
    assert_eq!(frame.pixel_format(), FramePixelFormat::Gray16);
    assert_eq!(frame.frame_bytes(), 12);
    assert_eq!(declaration.radiometric(), RadiometricStatus::Unknown);
    assert_eq!(
        crate::format::validate_negotiated_format(declaration, frame, RadiometricStatus::Unknown),
        Ok(()),
    );
}

#[test]
fn explicit_interleaved_pixel_formats_reject_short_rows_or_small_frame_bounds() {
    assert_eq!(
        FrameFormat::new_with_pixel_format(
            10,
            2,
            2,
            16,
            FramePixelFormat::Bgra8,
            [
                FramePlane::new(0, 8, 7, 2),
                FramePlane::default(),
                FramePlane::default(),
                FramePlane::default()
            ],
            1,
        ),
        Err(crate::FrameFormatError::InvalidPlane),
    );
    let cap =
        ExtendedCapability::new(90, 94, None, SampleEncoding::U8, [2, 2, 4, 0], None).unwrap();
    assert_eq!(
        FrameDeclaration::new_with_pixel_format(
            1,
            CameraModality::Visible,
            cap,
            FramePixelFormat::Rgba8,
            2,
            2,
            15,
            RadiometricStatus::NotApplicable,
        ),
        Err(crate::FrameFormatError::InvalidExtent),
    );
}
