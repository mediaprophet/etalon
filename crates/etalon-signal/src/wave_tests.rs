use super::*;

const MONO: [u8; 48] = [
    b'R',b'I',b'F',b'F',40,0,0,0,b'W',b'A',b'V',b'E',
    b'f',b'm',b't',b' ',16,0,0,0,1,0,1,0,0x40,0x1f,0,0,
    0x80,0x3e,0,0,2,0,16,0,b'd',b'a',b't',b'a',4,0,0,0,1,0,0xff,0xff,
];

#[test]
fn parses_pcm16_without_copying_sample_payload() {
    let wave = Pcm16Wave::parse(&MONO).unwrap();
    assert_eq!(wave.channels(), 1);
    assert_eq!(wave.sample_rate_hz(), 8000);
    assert_eq!(wave.frames(), 2);
    assert_eq!(wave.samples(), &[1, 0, 0xff, 0xff]);
}

#[test]
fn rejects_truncation_and_invalid_block_alignment() {
    assert_eq!(Pcm16Wave::parse(&MONO[..47]), Err(SignalError::LengthMismatch));
    let mut invalid = MONO;
    invalid[32] = 1;
    assert_eq!(Pcm16Wave::parse(&invalid), Err(SignalError::InvalidParameter));
}

fn with_fmt_extension(extension: &[u8]) -> std::vec::Vec<u8> {
    let mut bytes = MONO.to_vec();
    bytes.splice(36..36, extension.iter().copied());
    let fmt_size = 16 + extension.len() as u32;
    bytes[16..20].copy_from_slice(&fmt_size.to_le_bytes());
    let riff_size = (bytes.len() - 8) as u32;
    bytes[4..8].copy_from_slice(&riff_size.to_le_bytes());
    bytes
}

#[test]
fn accepts_only_defined_pcm_fmt_lengths() {
    let extended = with_fmt_extension(&[0, 0]);
    assert_eq!(Pcm16Wave::parse(&extended).unwrap().frames(), 2);

    let partial_extension = with_fmt_extension(&[0]);
    assert_eq!(Pcm16Wave::parse(&partial_extension), Err(SignalError::InvalidParameter));

    let nonempty_extension = with_fmt_extension(&[1, 0]);
    assert_eq!(Pcm16Wave::parse(&nonempty_extension), Err(SignalError::InvalidParameter));
}

#[test]
fn skips_odd_unknown_chunk_only_with_its_pad_byte() {
    let mut padded = MONO.to_vec();
    padded.splice(36..36, [b'J', b'U', b'N', b'K', 1, 0, 0, 0, 7, 0]);
    let riff_size = (padded.len() - 8) as u32;
    padded[4..8].copy_from_slice(&riff_size.to_le_bytes());
    assert_eq!(Pcm16Wave::parse(&padded).unwrap().samples(), &[1, 0, 0xff, 0xff]);

    let mut unpadded = MONO.to_vec();
    unpadded.splice(36..36, [b'J', b'U', b'N', b'K', 1, 0, 0, 0, 7]);
    let riff_size = (unpadded.len() - 8) as u32;
    unpadded[4..8].copy_from_slice(&riff_size.to_le_bytes());
    assert!(Pcm16Wave::parse(&unpadded).is_err());
}
