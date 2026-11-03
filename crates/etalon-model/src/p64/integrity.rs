/// Integrity is always selected explicitly at the parse boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegrityMode {
    Full,
    Metadata,
    Structure,
}

const fn table() -> [u32; 256] {
    let mut values = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = (crc >> 1) ^ (0x82F6_3B78 & 0u32.wrapping_sub(crc & 1));
            bit += 1;
        }
        values[i] = crc;
        i += 1;
    }
    values
}

const TABLE: [u32; 256] = table();

/// CRC-32C (Castagnoli, reflected), with the standard initial and final XOR.
pub fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc = TABLE[((crc ^ u32::from(byte)) & 255) as usize] ^ (crc >> 8);
    }
    !crc
}
