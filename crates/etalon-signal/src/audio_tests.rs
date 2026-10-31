use super::*;

#[test]
fn pcm_channel_power_uses_correct_interleaved_channel() {
    let stereo = [
        b'R', b'I', b'F', b'F', 44, 0, 0, 0, b'W', b'A', b'V', b'E', b'f', b'm', b't', b' ', 16, 0,
        0, 0, 1, 0, 2, 0, 0x40, 0x1f, 0, 0, 0x00, 0x7d, 0, 0, 4, 0, 16, 0, b'd', b'a', b't', b'a',
        8, 0, 0, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x40,
    ];
    let wave = Pcm16Wave::parse(&stereo).unwrap();
    assert_eq!(pcm16_channel_power(wave, 0).unwrap().mean_square, 0.5);
    assert_eq!(pcm16_channel_power(wave, 1).unwrap().mean_square, 0.125);
    assert_eq!(
        pcm16_channel_power(wave, 2),
        Err(SignalError::InvalidParameter)
    );
}
