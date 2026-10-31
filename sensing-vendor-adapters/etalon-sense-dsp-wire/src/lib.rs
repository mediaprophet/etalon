//! Original byte-buffer RPC ABI; not QAIC-generated sequence marshalling.
#![no_std]
#![forbid(unsafe_code)]

pub const SERVICE_ID: u64 = 0x4554_4951_5057_3031;
pub const SCHEMA_VERSION: u16 = 1;
pub const HEADER_BYTES: usize = 64;
pub const MAX_PAIRS: u32 = 1_048_576;
pub const IQ_POWER_OPERATION_ID: u64 = 0x4951_504f_5745_5231;
pub const IQ_POWER_REVISION: u32 = 1;
pub const METHOD_INDEX: u8 = 2;
pub const METHOD_SCALARS: u32 = 0x0201_0100;
pub const OPEN_SCALARS: u32 = 0x0002_0001;
pub const CLOSE_SCALARS: u32 = 0x0100_0010;
pub const MAX_URI_BYTES: usize = 512;
pub const SKELETON_EXPORT: &str = "etalon_iq_power_skel_handle_invoke";
pub const SCHEMA_DOCUMENT: &str = include_str!("../service-schema.json");
// SHA-256 of the exact UTF-8 service-schema.json bytes, including final newline.
pub const SCHEMA_SHA256: [u8; 32] = [0xbe, 0xe2, 0x3c, 0x17, 0x49, 0x9a, 0xb0, 0xdb,
    0x8a, 0x73, 0x09, 0x37, 0x2a, 0xc8, 0xb2, 0xd0, 0x98, 0x55, 0xdc, 0xed,
    0xb8, 0x25, 0x8b, 0xda, 0x1d, 0xbd, 0x3a, 0x04, 0x77, 0x1d, 0xa0, 0x29];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireError { InvalidIdentity, InvalidCount, InvalidLength, InvalidHeader, IdentityMismatch, InvalidUri }

/// Parse the original forward-loader contract. Input excludes the C NUL.
/// Domain/session resolution and image authorization remain host obligations.
pub fn generation_from_uri(uri: &[u8]) -> Result<u32, WireError> {
    if uri.len() > MAX_URI_BYTES || !uri.starts_with(b"file:///")
        || uri.iter().any(|b| !(0x21..=0x7e).contains(b) || *b == b'#' || *b == b'%')
    { return Err(WireError::InvalidUri); }
    let question = uri.iter().position(|b| *b == b'?').ok_or(WireError::InvalidUri)?;
    if question <= 8 || uri[question+1..].contains(&b'?') { return Err(WireError::InvalidUri); }
    let mut parts = uri[question+1..].split(|b| *b == b'&');
    if parts.next() != Some(SKELETON_EXPORT.as_bytes()) { return Err(WireError::InvalidUri); }
    let mut generation = None;
    let mut module_version = false;
    for (index, part) in parts.enumerate() {
        let mut pair = part.splitn(2, |b| *b == b'=');
        let key = pair.next().ok_or(WireError::InvalidUri)?;
        let value = pair.next().ok_or(WireError::InvalidUri)?;
        if key.is_empty() || value.is_empty() || value.contains(&b'=') { return Err(WireError::InvalidUri); }
        // Some public loader versions compare reserved keys by incoming length.
        // Reject abbreviated aliases and every repeated key to remove ambiguity.
        if (key != b"_modver" && b"_modver".starts_with(key))
            || uri[question+1..].split(|b| *b == b'&').skip(1).take(index)
                .any(|prior| prior.split(|b| *b == b'=').next() == Some(key))
        { return Err(WireError::InvalidUri); }
        if key == b"_modver" {
            if module_version || value != b"1.0" { return Err(WireError::InvalidUri); }
            module_version = true;
        } else if key == b"etalon_generation" {
            if generation.is_some() || value.len() > 10 || value[0] == b'0' { return Err(WireError::InvalidUri); }
            let mut number = 0u32;
            for digit in value {
                if !digit.is_ascii_digit() { return Err(WireError::InvalidUri); }
                number = number.checked_mul(10).and_then(|n| n.checked_add(u32::from(*digit - b'0')))
                    .ok_or(WireError::InvalidUri)?;
            }
            generation = Some(number);
        }
    }
    if !module_version { return Err(WireError::InvalidUri); }
    generation.ok_or(WireError::InvalidUri)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestIdentity { provider_id: u64, generation: u32, request_id: u64 }

impl RequestIdentity {
    pub const fn new(provider_id: u64, generation: u32, request_id: u64) -> Result<Self, WireError> {
        if provider_id == 0 || generation == 0 || request_id == 0 {
            return Err(WireError::InvalidIdentity);
        }
        Ok(Self { provider_id, generation, request_id })
    }
    pub const fn provider_id(self) -> u64 { self.provider_id }
    pub const fn generation(self) -> u32 { self.generation }
    pub const fn request_id(self) -> u64 { self.request_id }
}

pub const fn request_bytes(pairs: u32) -> Result<usize, WireError> { message_bytes(pairs, 4) }
pub const fn response_bytes(pairs: u32) -> Result<usize, WireError> { message_bytes(pairs, 8) }

const fn message_bytes(pairs: u32, stride: usize) -> Result<usize, WireError> {
    if pairs == 0 || pairs > MAX_PAIRS { return Err(WireError::InvalidCount); }
    match (pairs as usize).checked_mul(stride) {
        Some(payload) => match payload.checked_add(HEADER_BYTES) {
            Some(total) => Ok(total), None => Err(WireError::InvalidLength),
        },
        None => Err(WireError::InvalidLength),
    }
}

pub fn encode_request(identity: RequestIdentity, pairs: u32, iq: &[u8], out: &mut [u8])
    -> Result<(), WireError>
{
    let bytes = request_bytes(pairs)?;
    if out.len() != bytes || iq.len() != bytes - HEADER_BYTES { return Err(WireError::InvalidLength); }
    write_header(identity, pairs, 1, &mut out[..HEADER_BYTES]);
    out[HEADER_BYTES..].copy_from_slice(iq);
    Ok(())
}

pub fn decode_request(bytes: &[u8]) -> Result<(RequestIdentity, u32, &[u8]), WireError> {
    let (identity, pairs) = read_header(bytes, 1)?;
    if bytes.len() != request_bytes(pairs)? { return Err(WireError::InvalidLength); }
    Ok((identity, pairs, &bytes[HEADER_BYTES..]))
}

/// Validate all echoed fields before exposing any result payload to the caller.
pub fn decode_response(identity: RequestIdentity, pairs: u32, bytes: &[u8]) -> Result<&[u8], WireError> {
    let (actual_identity, actual_pairs) = read_header(bytes, 2)?;
    if actual_identity != identity || actual_pairs != pairs { return Err(WireError::IdentityMismatch); }
    if bytes.len() != response_bytes(pairs)? { return Err(WireError::InvalidLength); }
    let payload = &bytes[HEADER_BYTES..];
    // Revision 1 returns exact integer-valued f64 powers in [0, 2^31].
    for value in payload.chunks_exact(8) {
        let power = f64::from_le_bytes(value.try_into().map_err(|_| WireError::InvalidLength)?);
        if !power.is_finite() || !(0.0..=2_147_483_648.0).contains(&power)
            || (power as u64) as f64 != power || power.to_bits() == (-0.0f64).to_bits()
        { return Err(WireError::InvalidHeader); }
    }
    Ok(payload)
}

/// Only publish a response header after the native kernel has succeeded.
pub fn finish_response(identity: RequestIdentity, pairs: u32, out: &mut [u8]) -> Result<(), WireError> {
    if out.len() != response_bytes(pairs)? { return Err(WireError::InvalidLength); }
    write_header(identity, pairs, 2, &mut out[..HEADER_BYTES]);
    Ok(())
}

fn write_header(identity: RequestIdentity, pairs: u32, kind: u32, out: &mut [u8]) {
    out.fill(0);
    out[..8].copy_from_slice(b"ETIQPW01");
    out[8..10].copy_from_slice(&SCHEMA_VERSION.to_le_bytes());
    out[10..12].copy_from_slice(&(HEADER_BYTES as u16).to_le_bytes());
    out[12..16].copy_from_slice(&IQ_POWER_REVISION.to_le_bytes());
    out[16..24].copy_from_slice(&IQ_POWER_OPERATION_ID.to_le_bytes());
    out[24..32].copy_from_slice(&identity.provider_id.to_le_bytes());
    out[32..36].copy_from_slice(&identity.generation.to_le_bytes());
    out[36..40].copy_from_slice(&pairs.to_le_bytes());
    out[40..48].copy_from_slice(&identity.request_id.to_le_bytes());
    out[48..52].copy_from_slice(&(pairs * 4).to_le_bytes());
    out[52..56].copy_from_slice(&(pairs * 8).to_le_bytes());
    out[56..60].copy_from_slice(&kind.to_le_bytes());
}

fn read_header(bytes: &[u8], kind: u32) -> Result<(RequestIdentity, u32), WireError> {
    if bytes.len() < HEADER_BYTES || &bytes[..8] != b"ETIQPW01"
        || bytes[8..10] != SCHEMA_VERSION.to_le_bytes()
        || bytes[10..12] != (HEADER_BYTES as u16).to_le_bytes()
        || read_u32(bytes, 12) != IQ_POWER_REVISION
        || read_u64(bytes, 16) != IQ_POWER_OPERATION_ID
        || read_u32(bytes, 56) != kind || read_u32(bytes, 60) != 0
    { return Err(WireError::InvalidHeader); }
    let pairs = read_u32(bytes, 36);
    request_bytes(pairs)?;
    if read_u32(bytes, 48) != pairs * 4 || read_u32(bytes, 52) != pairs * 8 {
        return Err(WireError::InvalidLength);
    }
    let identity = RequestIdentity::new(read_u64(bytes, 24), read_u32(bytes, 32), read_u64(bytes, 40))?;
    Ok((identity, pairs))
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at+1], bytes[at+2], bytes[at+3]])
}
fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([bytes[at], bytes[at+1], bytes[at+2], bytes[at+3],
        bytes[at+4], bytes[at+5], bytes[at+6], bytes[at+7]])
}

#[cfg(test)]
mod tests;
