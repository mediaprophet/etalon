pub const MAX_VALUE: usize = 512;
pub const MAX_EVENTS: usize = 32;

pub fn uuid(value: &str) -> Result<String, String> {
    if matches!(value.len(), 4 | 8) && value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Ok(format!(
            "{:0>8}-0000-1000-8000-00805f9b34fb",
            value.to_ascii_lowercase()
        ));
    }
    if value.len() == 36
        && value.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
    {
        return Ok(value.to_ascii_lowercase());
    }
    Err("Use a 4/8-digit hexadecimal or canonical 128-bit Bluetooth UUID".into())
}
pub fn decode(value: &str, out: &mut [u8; MAX_VALUE]) -> Result<usize, String> {
    if value.is_empty()
        || value.len() > MAX_VALUE * 2
        || !value.len().is_multiple_of(2)
        || !value.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("Write requires 1–512 bytes as hexadecimal pairs without separators".into());
    }
    for (i, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |b: u8| {
            if b.is_ascii_digit() {
                b - b'0'
            } else {
                b.to_ascii_lowercase() - b'a' + 10
            }
        };
        out[i] = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Ok(value.len() / 2)
}
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|b| {
            [
                DIGITS[(b >> 4) as usize] as char,
                DIGITS[(b & 15) as usize] as char,
            ]
        })
        .collect()
}
#[derive(Clone, Copy)]
pub struct Packet {
    pub bytes: [u8; MAX_VALUE],
    pub len: usize,
    pub timestamp_ms: f64,
}
impl Default for Packet {
    fn default() -> Self {
        Self {
            bytes: [0; MAX_VALUE],
            len: 0,
            timestamp_ms: 0.0,
        }
    }
}
pub struct Events {
    pub items: [Packet; MAX_EVENTS],
    pub len: usize,
    pub dropped: u32,
    pub errors: u32,
}
impl Default for Events {
    fn default() -> Self {
        Self {
            items: [Packet::default(); MAX_EVENTS],
            len: 0,
            dropped: 0,
            errors: 0,
        }
    }
}
impl Events {
    pub fn push(&mut self, packet: Packet) {
        if self.len == MAX_EVENTS {
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        self.items[self.len] = packet;
        self.len += 1;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uuid_and_write_validation() {
        assert_eq!(
            uuid("180F").unwrap(),
            "0000180f-0000-1000-8000-00805f9b34fb"
        );
        assert!(uuid("heart_rate").is_err());
        assert!(uuid("0000180f-0000-1000-8000-00805f9b34fg").is_err());
        let mut b = [0; MAX_VALUE];
        assert_eq!(decode("00ffA5", &mut b).unwrap(), 3);
        assert_eq!(&b[..3], &[0, 255, 165]);
        assert_eq!(hex(&b[..3]), "00ffa5");
        for bad in ["", "0", "gg", "00 ff"] {
            assert!(decode(bad, &mut b).is_err());
        }
        assert!(decode(&"ff".repeat(513), &mut b).is_err());
    }
    #[test]
    fn queue_bound_reports_loss() {
        let mut q = Events::default();
        for _ in 0..40 {
            q.push(Packet::default());
        }
        assert_eq!(q.len, 32);
        assert_eq!(q.dropped, 8);
        assert_eq!(q.items[0].bytes, [0; MAX_VALUE]);
        assert_eq!(q.items[0].len, 0);
        assert_eq!(q.items[0].timestamp_ms, 0.0);
        assert_eq!(q.errors, 0);
    }
}
