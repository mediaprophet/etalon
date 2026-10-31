use super::*;

fn wave_format(tag: u16, channels: u16, rate: u32, bits: u16, align: u16) -> WAVEFORMATEX {
    WAVEFORMATEX {
        wFormatTag: tag,
        nChannels: channels,
        nSamplesPerSec: rate,
        nAvgBytesPerSec: rate * u32::from(align),
        nBlockAlign: align,
        wBitsPerSample: bits,
        cbSize: 0,
    }
}

#[test]
fn accepts_consistent_pcm16_and_float32_mix_formats() {
    let pcm = wave_format(WAVE_FORMAT_PCM_TAG, 2, 48_000, 16, 4);
    let pcm = unsafe { WasapiFormat::parse(&pcm) }.unwrap();
    assert_eq!(pcm.encoding(), SampleEncoding::I16);
    assert_eq!(pcm.channels(), 2);

    let float = wave_format(WAVE_FORMAT_IEEE_FLOAT, 1, 44_100, 32, 4);
    let float = unsafe { WasapiFormat::parse(&float) }.unwrap();
    assert_eq!(float.encoding(), SampleEncoding::F32);
    assert_eq!(float.sample_rate_hz(), 44_100);
}

#[test]
fn rejects_unsupported_or_inconsistent_mix_formats() {
    let pcm24 = wave_format(WAVE_FORMAT_PCM_TAG, 2, 48_000, 24, 6);
    assert!(unsafe { WasapiFormat::parse(&pcm24) }.is_err());

    let bad_align = wave_format(WAVE_FORMAT_PCM_TAG, 2, 48_000, 16, 2);
    assert!(unsafe { WasapiFormat::parse(&bad_align) }.is_err());
}
