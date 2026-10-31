//! Native RTL-SDR receipts and borrowed payloads in the common sensing envelope.

use crate::{
    conversion::{
        center_double_u8_iq_into, ConversionError, ConvertedIq, DigitalIqConversion,
        MAX_CONVERSION_PAIRS,
    },
    ReadReport,
};
use etalon_sense_core::{
    buffer::SampleLayout,
    evidence::EvidenceOrigin,
    observation::{
        HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind,
        QuantityUnitIds,
    },
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor, SourceKind},
    quantity::SampleEncoding,
    ContractError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtlSdrObservationMapping {
    /// Host registry declarations; no ADC-code to physical-unit map is inferred.
    pub quantity_unit: QuantityUnitIds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationError {
    InvalidReceipt,
    LengthMismatch,
    Contract(ContractError),
    Conversion(ConversionError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationReadError {
    Native(crate::Error),
    Observation(ObservationError),
}

/// The receipt preserves device/configuration declarations. Its sequence counts
/// accepted driver calls, not samples or uninterrupted USB acquisition. Timing,
/// overruns and sample continuity remain unknown in the common envelope.
pub struct RtlSdrObservation<'raw> {
    envelope: ObservationEnvelope,
    report: ReadReport,
    raw: &'raw [u8],
    layout: SampleLayout,
}

impl<'raw> RtlSdrObservation<'raw> {
    pub const fn envelope(&self) -> &ObservationEnvelope {
        &self.envelope
    }
    pub const fn report(&self) -> &ReadReport {
        &self.report
    }
    pub const fn bytes(&self) -> &'raw [u8] {
        self.raw
    }
    /// Pair-major U8 [N, 2], with I then Q. This does not require a new enum ABI.
    pub const fn layout(&self) -> &SampleLayout {
        &self.layout
    }

    /// Explicit caller-authorized derivation. The output semantic mapping is
    /// separate from the source mapping because amplitude is doubled. This
    /// function makes no fresh processing permission or calibration assertion.
    pub fn center_double_into<'frame, 'out>(
        &'frame self,
        mapping: RtlSdrObservationMapping,
        output: &'out mut [u8],
    ) -> Result<ConvertedRtlSdrObservation<'frame, 'raw, 'out>, ObservationError> {
        let envelope = match make_envelope(
            self.report.target_id(),
            ObservationKind::HostDerived,
            mapping,
        ) {
            Ok(envelope) => envelope,
            Err(error) => {
                output.fill(0);
                return Err(error);
            }
        };
        let converted =
            center_double_u8_iq_into(self.raw, output).map_err(ObservationError::Conversion)?;
        Ok(ConvertedRtlSdrObservation {
            original: self,
            envelope,
            converted,
        })
    }
}

/// Retains the complete original capture and explicit reversible transform.
/// Its LE ComplexI16 layout/bytes can be passed to existing live processors.
pub struct ConvertedRtlSdrObservation<'frame, 'raw, 'out> {
    original: &'frame RtlSdrObservation<'raw>,
    envelope: ObservationEnvelope,
    converted: ConvertedIq<'raw, 'out>,
}

impl<'frame, 'raw, 'out> ConvertedRtlSdrObservation<'frame, 'raw, 'out> {
    pub const fn original(&self) -> &'frame RtlSdrObservation<'raw> {
        self.original
    }
    pub const fn envelope(&self) -> &ObservationEnvelope {
        &self.envelope
    }
    pub const fn bytes(&self) -> &'out [u8] {
        self.converted.bytes()
    }
    pub const fn layout(&self) -> &SampleLayout {
        self.converted.layout()
    }
    pub const fn conversion(&self) -> DigitalIqConversion {
        self.converted.conversion()
    }
}

/// Prevalidate metadata/layout, perform one explicitly gated native read, then
/// borrow that exact result with its receipt. A read has no deadline/cancel ABI;
/// this deliberately does not implement the bounded-duration `SdrReader` trait.
/// Every returned error clears the entire caller-provided output buffer.
pub fn read_observation_into<'raw>(
    device: &mut crate::RtlSdrDevice<'_, '_>,
    gate: &impl crate::RtlSdrGate,
    mapping: RtlSdrObservationMapping,
    output: &'raw mut [u8],
) -> Result<RtlSdrObservation<'raw>, ObservationReadError> {
    let count = output.len() / 2;
    let preparation = (|| {
        if output.is_empty()
            || output.len() % 2 != 0
            || count > device.max_pairs()
            || count > MAX_CONVERSION_PAIRS
            || device.generation() == 0
        {
            return Err(ObservationError::LengthMismatch);
        }
        let envelope = make_envelope(device.target_id(), ObservationKind::LiveFrame, mapping)?;
        let layout =
            SampleLayout::new(2, [count as u32, 2, 0, 0], [2, 1, 0, 0], SampleEncoding::U8)
                .map_err(ObservationError::Contract)?;
        Ok((envelope, layout))
    })();
    let (envelope, layout) = match preparation {
        Ok(prepared) => prepared,
        Err(error) => {
            output.fill(0);
            return Err(ObservationReadError::Observation(error));
        }
    };
    let target = device.target_id();
    let generation = device.generation();
    let index = device.device_index();
    let report = match device.read_into(gate, output) {
        Ok(report) => report,
        Err(error) => {
            output.fill(0);
            return Err(ObservationReadError::Native(error));
        }
    };
    if report.target_id() != target
        || report.generation() != generation
        || report.device_index() != index
        || report.pair_count() != count
    {
        output.fill(0);
        return Err(ObservationReadError::Observation(
            ObservationError::InvalidReceipt,
        ));
    }
    Ok(RtlSdrObservation {
        envelope,
        report,
        raw: output,
        layout,
    })
}

/// Structural pairing only: size equality does not authenticate that a receipt
/// describes these particular bytes. Prefer the native read composition to
/// separate receipt/payload adaptation; retain original raw bytes for custody.
pub fn adapt_read_report<'raw>(
    report: ReadReport,
    raw: &'raw [u8],
    mapping: RtlSdrObservationMapping,
    output: &mut Option<RtlSdrObservation<'raw>>,
) -> Result<(), ObservationError> {
    *output = None;
    let pairs = report.pair_count();
    if report.target_id() == 0
        || report.generation() == 0
        || pairs == 0
        || pairs > MAX_CONVERSION_PAIRS
    {
        return Err(ObservationError::InvalidReceipt);
    }
    if pairs.checked_mul(2) != Some(raw.len()) {
        return Err(ObservationError::LengthMismatch);
    }
    let envelope = make_envelope(report.target_id(), ObservationKind::LiveFrame, mapping)?;
    let layout = SampleLayout::new(2, [pairs as u32, 2, 0, 0], [2, 1, 0, 0], SampleEncoding::U8)
        .map_err(ObservationError::Contract)?;
    *output = Some(RtlSdrObservation {
        envelope,
        report,
        raw,
        layout,
    });
    Ok(())
}

fn make_envelope(
    target_id: u64,
    kind: ObservationKind,
    mapping: RtlSdrObservationMapping,
) -> Result<ObservationEnvelope, ObservationError> {
    let source =
        SourceDescriptor::new(target_id, SourceKind::Live).map_err(ObservationError::Contract)?;
    let calibration =
        CalibrationRecord::new(CalibrationState::Missing).map_err(ObservationError::Contract)?;
    ObservationEnvelope::new(
        kind,
        source,
        None,
        None,
        None,
        None,
        mapping.quantity_unit,
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        calibration,
        EvidenceOrigin::Unknown,
        None,
        HostPolicyDecision::NotEvaluated,
    )
    .map_err(ObservationError::Contract)
}
