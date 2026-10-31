#![deny(unsafe_code)]
//! Windows PC/SC reader enumeration and opt-in UID request support.
//!
//! The UID command is a reader-specific extension, not a universal NFC
//! command; a reader may reject it. PC/SC readers can also be contact readers.
//! Reported UIDs do not prove identity, tag type, ownership, or proximity.
//! Operations are synchronous and Windows or the reader may block.

#[cfg(windows)]
#[allow(unsafe_code)]
mod windows;

#[cfg(windows)]
pub use windows::{list_readers, read_uid, PcscProvider};

use etalon_sense_nfc::MAX_NFC_UID_BYTES;

pub const MAX_READER_NAME_UNITS: usize = 256;
pub const MAX_READER_LIST_UNITS: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcscOperation {
    ListReaders,
    ReadUid,
}

/// The caller owns policy and must approve every OS operation.
pub trait PcscGate {
    /// `reader` is the exact, unterminated UTF-16 target for UID requests and
    /// is absent for enumeration. The host can bind source IDs to readers.
    fn approve(&mut self, operation: PcscOperation, source_id: u64, reader: Option<&[u16]>)
        -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcscError {
    PolicyDenied,
    InvalidSource,
    InvalidReader,
    InvalidReaderList,
    OutputTooSmall { required: usize },
    System(i32),
    UnsupportedProtocol(u32),
    InvalidResponse,
    ReaderStatus { sw1: u8, sw2: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcscUid {
    bytes: [u8; MAX_NFC_UID_BYTES],
    len: u8,
}

impl PcscUid {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}

fn uid_from_response(response: &[u8]) -> Result<PcscUid, PcscError> {
    if response.len() < 2 {
        return Err(PcscError::InvalidResponse);
    }
    let data_len = response.len() - 2;
    let sw1 = response[data_len];
    let sw2 = response[data_len + 1];
    if sw1 != 0x90 || sw2 != 0x00 {
        return Err(PcscError::ReaderStatus { sw1, sw2 });
    }
    if data_len == 0 || data_len > MAX_NFC_UID_BYTES {
        return Err(PcscError::InvalidResponse);
    }
    let mut bytes = [0; MAX_NFC_UID_BYTES];
    bytes[..data_len].copy_from_slice(&response[..data_len]);
    Ok(PcscUid {
        bytes,
        len: data_len as u8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uid_response_preserves_zero_and_status() {
        assert_eq!(
            uid_from_response(&[0, 1, 0x90, 0]).unwrap().bytes(),
            &[0, 1]
        );
        assert_eq!(
            uid_from_response(&[0x6a, 0x81]),
            Err(PcscError::ReaderStatus {
                sw1: 0x6a,
                sw2: 0x81
            })
        );
        assert_eq!(
            uid_from_response(&[0x90, 0]),
            Err(PcscError::InvalidResponse)
        );
    }
}
