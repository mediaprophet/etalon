use core::{ffi::c_void, ptr};

use crate::{uid_from_response, PcscError, PcscGate, PcscOperation, PcscUid, MAX_READER_LIST_UNITS, MAX_READER_NAME_UNITS};
use etalon_sense_nfc::{HostNfcMetadata, NfcProvider, NfcProviderError, NfcProtocol, NfcProximity, NfcTagKind, MAX_NFC_UID_BYTES};

type ContextHandle = usize;
type CardHandle = usize;
const SCOPE_USER: u32 = 0;
const SHARE_SHARED: u32 = 2;
const LEAVE_CARD: u32 = 0;
const PROTOCOL_T0: u32 = 1;
const PROTOCOL_T1: u32 = 2;

#[repr(C)]
struct IoRequest { protocol: u32, pci_length: u32 }

#[link(name = "winscard")]
unsafe extern "system" {
    fn SCardEstablishContext(scope: u32, reserved1: *const c_void, reserved2: *const c_void, context: *mut ContextHandle) -> i32;
    fn SCardReleaseContext(context: ContextHandle) -> i32;
    fn SCardListReadersW(context: ContextHandle, groups: *const u16, readers: *mut u16, length: *mut u32) -> i32;
    fn SCardConnectW(context: ContextHandle, reader: *const u16, share: u32, protocols: u32, card: *mut CardHandle, active: *mut u32) -> i32;
    fn SCardDisconnect(card: CardHandle, disposition: u32) -> i32;
    fn SCardTransmit(card: CardHandle, send_pci: *const IoRequest, send: *const u8, send_len: u32, recv_pci: *mut IoRequest, recv: *mut u8, recv_len: *mut u32) -> i32;
}

struct Context(ContextHandle);
impl Context {
    fn open() -> Result<Self, PcscError> {
        let mut context = 0;
        let status = unsafe { SCardEstablishContext(SCOPE_USER, ptr::null(), ptr::null(), &mut context) };
        if status != 0 { return Err(PcscError::System(status)); }
        Ok(Self(context))
    }
}
impl Drop for Context {
    fn drop(&mut self) { unsafe { SCardReleaseContext(self.0); } }
}

struct Card(CardHandle);
impl Drop for Card {
    fn drop(&mut self) { unsafe { SCardDisconnect(self.0, LEAVE_CARD); } }
}

/// Copies the native UTF-16 MULTI_SZ reader list, including its two trailing
/// NUL units. Caller output remains unchanged on error. Reader names may be
/// identifying; the host must decide how to handle them.
pub fn list_readers<G: PcscGate>(gate: &mut G, source_id: u64, out: &mut [u16]) -> Result<usize, PcscError> {
    if source_id == 0 { return Err(PcscError::InvalidSource); }
    if !gate.approve(PcscOperation::ListReaders, source_id, None) { return Err(PcscError::PolicyDenied); }
    let context = Context::open()?;
    let mut readers = [0u16; MAX_READER_LIST_UNITS];
    let mut length = readers.len() as u32;
    let status = unsafe { SCardListReadersW(context.0, ptr::null(), readers.as_mut_ptr(), &mut length) };
    if status != 0 { return Err(PcscError::System(status)); }
    let length = length as usize;
    if length < 2 || length > readers.len() || readers[length - 2..length] != [0, 0]
        || readers[..length - 2].windows(2).any(|pair| pair == [0, 0])
    { return Err(PcscError::InvalidReaderList); }
    if out.len() < length { return Err(PcscError::OutputTooSmall { required: length }); }
    out[..length].copy_from_slice(&readers[..length]);
    Ok(length)
}

/// Query a single reader using the common PC/SC Get Data UID extension.
/// The reader name is UTF-16 without a terminator. A successful response is
/// still only an unverified reader observation.
pub fn read_uid<G: PcscGate>(gate: &mut G, source_id: u64, reader: &[u16]) -> Result<PcscUid, PcscError> {
    if source_id == 0 { return Err(PcscError::InvalidSource); }
    if reader.is_empty() || reader.len() > MAX_READER_NAME_UNITS || reader.contains(&0) {
        return Err(PcscError::InvalidReader);
    }
    if !gate.approve(PcscOperation::ReadUid, source_id, Some(reader)) { return Err(PcscError::PolicyDenied); }
    let context = Context::open()?;
    let mut name = [0u16; MAX_READER_NAME_UNITS + 1];
    name[..reader.len()].copy_from_slice(reader);
    let mut card = 0;
    let mut protocol = 0;
    let status = unsafe { SCardConnectW(context.0, name.as_ptr(), SHARE_SHARED, PROTOCOL_T0 | PROTOCOL_T1, &mut card, &mut protocol) };
    if status != 0 { return Err(PcscError::System(status)); }
    let card = Card(card);
    if protocol != PROTOCOL_T0 && protocol != PROTOCOL_T1 { return Err(PcscError::UnsupportedProtocol(protocol)); }
    let pci = IoRequest { protocol, pci_length: core::mem::size_of::<IoRequest>() as u32 };
    let command = [0xff, 0xca, 0x00, 0x00, 0x00];
    let mut response = [0u8; etalon_sense_nfc::MAX_NFC_UID_BYTES + 2];
    let mut length = response.len() as u32;
    let status = unsafe { SCardTransmit(card.0, &pci, command.as_ptr(), command.len() as u32, ptr::null_mut(), response.as_mut_ptr(), &mut length) };
    if status != 0 { return Err(PcscError::System(status)); }
    if length as usize > response.len() { return Err(PcscError::InvalidResponse); }
    uid_from_response(&response[..length as usize])
}

/// Adapter into the existing one-observation NFC host boundary. The stream
/// identifiers and sequence originate from the host; timestamps, tag kind and
/// protocol remain unknown. All PC/SC requests are individually authorized.
/// Each call is a new poll, not proof of a new arrival or continuous presence.
pub struct PcscProvider<'a, G> {
    gate: &'a mut G,
    reader: &'a [u16],
    source_id: u64,
    stream_id: u64,
    generation: u32,
    sequence: u64,
    last_error: Option<PcscError>,
}

impl<'a, G: PcscGate> PcscProvider<'a, G> {
    pub fn new(gate: &'a mut G, reader: &'a [u16], source_id: u64, stream_id: u64, generation: u32, sequence: u64) -> Result<Self, PcscError> {
        if source_id == 0 || stream_id == 0 || generation == 0 { return Err(PcscError::InvalidSource); }
        if reader.is_empty() || reader.len() > MAX_READER_NAME_UNITS || reader.contains(&0) { return Err(PcscError::InvalidReader); }
        Ok(Self { gate, reader, source_id, stream_id, generation, sequence, last_error: None })
    }

    pub fn last_error(&self) -> Option<PcscError> { self.last_error }
}

impl<G: PcscGate> NfcProvider for PcscProvider<'_, G> {
    fn read_next(&mut self, uid_output: &mut [u8; MAX_NFC_UID_BYTES]) -> Result<HostNfcMetadata, NfcProviderError> {
        let next = self.sequence.checked_add(1).ok_or(NfcProviderError::EndOfStream)?;
        let uid = match read_uid(self.gate, self.source_id, self.reader) {
            Ok(value) => { self.last_error = None; value }
            Err(error) => { self.last_error = Some(error); return Err(NfcProviderError::Unavailable); }
        };
        uid_output.fill(0);
        uid_output[..uid.bytes().len()].copy_from_slice(uid.bytes());
        let metadata = HostNfcMetadata {
            source_id: self.source_id, stream_id: self.stream_id,
            generation: self.generation, sequence: self.sequence,
            gap_before: None, uid_len: uid.bytes().len() as u8,
            protocol: NfcProtocol::Unknown, tag_kind: NfcTagKind::Unknown,
            proximity: NfcProximity::Unknown, timestamp: None,
        };
        self.sequence = next;
        Ok(metadata)
    }
}
