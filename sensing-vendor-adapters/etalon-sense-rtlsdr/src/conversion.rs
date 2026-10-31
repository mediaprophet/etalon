//! Explicit digital-code conversion, separate from RF calibration or capture.

use etalon_sense_core::{buffer::SampleLayout, quantity::SampleEncoding, ContractError};

pub const MAX_CONVERSION_PAIRS: usize = 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConversionError { InvalidLength, Capacity, Contract(ContractError) }

/// The exact operation is `i16 = 2 * raw_u8 - 255`, written little-endian.
/// Its inverse is `(i16 + 255) / 2` on these odd integer outputs. It centers
/// digital codes at 127.5 and doubles amplitude; it does not establish volts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigitalIqConversion { Center127Point5DoubleToLeI16 }

/// Both original ADC bytes and the derived bytes remain borrowed from callers.
/// Keep the conversion declaration when using the derived bytes with an
/// existing ComplexI16 power processor: its unscaled power is in doubled-code².
pub struct ConvertedIq<'raw, 'out> {
    original: &'raw [u8],
    bytes: &'out [u8],
    layout: SampleLayout,
}

impl<'raw, 'out> ConvertedIq<'raw, 'out> {
    pub const fn original(&self) -> &'raw [u8] { self.original }
    pub const fn bytes(&self) -> &'out [u8] { self.bytes }
    pub const fn layout(&self) -> &SampleLayout { &self.layout }
    pub const fn conversion(&self) -> DigitalIqConversion {
        DigitalIqConversion::Center127Point5DoubleToLeI16
    }
    pub fn sample_count(&self) -> usize { self.original.len() / 2 }
}

/// Caller-selected arithmetic conversion with no acquisition, policy decision
/// or metadata upgrade. Every error clears the entire supplied output slice.
pub fn center_double_u8_iq_into<'raw, 'out>(original: &'raw [u8], output: &'out mut [u8])
    -> Result<ConvertedIq<'raw, 'out>, ConversionError>
{
    let layout = match conversion_layout(original.len(), output.len()) {
        Ok(layout) => layout,
        Err(error) => { output.fill(0); return Err(error); }
    };
    for (sample, destination) in original.iter().zip(output.chunks_exact_mut(2)) {
        let centered = i16::from(*sample) * 2 - 255;
        destination.copy_from_slice(&centered.to_le_bytes());
    }
    Ok(ConvertedIq { original, bytes: output, layout })
}

fn conversion_layout(input_bytes: usize, output_bytes: usize) -> Result<SampleLayout, ConversionError> {
    if input_bytes == 0 || input_bytes % 2 != 0 || input_bytes / 2 > MAX_CONVERSION_PAIRS {
        return Err(ConversionError::InvalidLength);
    }
    if input_bytes.checked_mul(2) != Some(output_bytes) { return Err(ConversionError::Capacity); }
    SampleLayout::new(1, [(input_bytes / 2) as u32, 0, 0, 0], [4, 0, 0, 0],
        SampleEncoding::ComplexI16).map_err(ConversionError::Contract)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_adc_codes_round_trip_and_preserve_original() {
        let mut raw = [0; 256];
        for (index, byte) in raw.iter_mut().enumerate() { *byte = index as u8; }
        let mut output = [0; 512];
        let converted = center_double_u8_iq_into(&raw, &mut output).unwrap();
        assert_eq!(converted.original(), raw);
        for (original, encoded) in raw.iter().zip(converted.bytes().chunks_exact(2)) {
            let value = i16::from_le_bytes([encoded[0], encoded[1]]);
            assert_eq!((value + 255) / 2, i16::from(*original));
            assert_eq!(value % 2 == 0, false);
        }
        assert_eq!(converted.bytes()[..2], (-255i16).to_le_bytes());
        assert_eq!(converted.bytes()[510..], 255i16.to_le_bytes());
        assert_eq!(converted.layout().encoding(), &SampleEncoding::ComplexI16);
    }

    #[test]
    fn rejected_lengths_clear_entire_destination() {
        for raw in [&[][..], &[1][..], &[1, 2, 3][..]] {
            let mut output = [0xaa; 9];
            assert!(center_double_u8_iq_into(raw, &mut output).is_err());
            assert_eq!(output, [0; 9]);
        }
        let mut output = [0xaa; 5];
        assert!(matches!(center_double_u8_iq_into(&[1, 2], &mut output), Err(ConversionError::Capacity)));
        assert_eq!(output, [0; 5]);
    }
}
