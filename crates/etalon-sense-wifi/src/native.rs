use core::{
    mem::{offset_of, size_of},
    ptr,
};

use windows::{
    core::GUID,
    Win32::{
        Foundation::HANDLE,
        NetworkManagement::IpHelper::{ConvertInterfaceGuidToLuid, GetIfEntry2, MIB_IF_ROW2},
        NetworkManagement::Ndis::NET_LUID_LH,
        NetworkManagement::WiFi::{
            WlanCloseHandle, WlanFreeMemory, WlanGetNetworkBssList, WlanOpenHandle, DOT11_BSS_TYPE,
            WLAN_API_VERSION_2_0, WLAN_BSS_ENTRY, WLAN_BSS_LIST,
        },
    },
};

use crate::TimestampStatus;
use etalon_sense_core::radio::{
    RadioAddress48, RadioIdentifier, RadioIdentifierObservation, RadioObservationTime,
};

/// Maximum accepted BSS rows copied into caller-owned storage per call.
pub const MAX_BSS_OBSERVATIONS: usize = 256;
/// Reject unexpectedly large API result blocks before traversing them. This
/// bounds our parsing work; the WLAN service has already allocated the block.
pub const MAX_BSS_LIST_BYTES: usize = 16 * 1024 * 1024;

/// One cached Windows BSS record. RSSI is in dBm; link quality is a separate
/// 0–100 style driver metric. No calibration, distance, CSI, or identity
/// semantics are inferred.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BssObservation {
    /// Raw BSSID bytes reported by WLAN API, not a person or verified device identity.
    pub bssid: [u8; 6],
    /// Windows `WLAN_BSS_ENTRY.lRssi`, in dBm as reported by the driver.
    pub rssi_dbm: i32,
    /// Windows `uLinkQuality`; this is not dBm and is not converted here.
    pub link_quality: u32,
    /// Center frequency in kHz reported by the WLAN API; not calibrated here.
    pub center_frequency_khz: u32,
}

impl BssObservation {
    /// Preserve the reported BSSID in its own namespace. It is distinct from
    /// the host's `SourceDescriptor` and this adapter does not query the
    /// interface MAC address.
    pub const fn radio_identifier(self) -> RadioIdentifier {
        RadioIdentifier::Bssid(RadioAddress48::from_octets(self.bssid))
    }
}

/// Summary for a cache read. The API does not provide a clock-domain mapping
/// here, so the timestamp and cache age are explicitly unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BssReadSummary {
    pub written: usize,
    pub timestamp: TimestampStatus,
    pub cache_age_known: bool,
}

/// Host policy boundary. The implementation must freshly check current
/// authorization for the requested interface on every invocation; a prior
/// one-shot decision must not be treated as reusable. This adapter calls it
/// immediately before each WLAN acquisition API operation. It may permit
/// environmental or public-science contexts; no human-subject or guardian
/// model is imposed. Handle close and allocation free are cleanup operations.
pub trait PolicyGate {
    fn approve_cached_bss_read(&mut self, interface: &GUID) -> Result<(), WifiError>;
}

/// Host policy boundary for querying the address Windows currently reports
/// for an interface. This query is separate from cached BSS reads and must be
/// freshly approved before each IP Helper call.
pub trait InterfaceMacPolicyGate {
    fn approve_interface_mac_read(&mut self, interface: &GUID) -> Result<(), WifiError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WifiError {
    PolicyDenied,
    WlanStatus(u32),
    InvalidHandle,
    NullResult,
    InvalidList,
    TooManyResults,
    OutputTooSmall { required: usize },
    InterfaceQueryStatus(u32),
    InterfaceMismatch,
    InvalidPhysicalAddressLength(u32),
}

/// Query the current OS-reported physical address for a caller-selected WLAN
/// interface. This uses IP Helper's typed interface row, does not scan or
/// activate a radio, and returns the current `PhysicalAddress`, never the
/// `PermanentPhysicalAddress`. The value may be randomized or OS-overridden;
/// it is not an authenticated or permanent hardware identity and is distinct
/// from any observed BSSID. The address observation has unknown time/cache age.
///
/// The host gate is rechecked immediately before each OS query. A gate must
/// perform its own current policy/revocation check; this adapter does not
/// establish trust or identity.
pub fn read_current_interface_mac<G: InterfaceMacPolicyGate>(
    gate: &mut G,
    interface: &GUID,
) -> Result<RadioIdentifierObservation, WifiError> {
    gate.approve_interface_mac_read(interface)?;
    let mut luid = NET_LUID_LH::default();
    let status = unsafe { ConvertInterfaceGuidToLuid(interface, &mut luid) };
    if status.0 != 0 {
        return Err(WifiError::InterfaceQueryStatus(status.0));
    }

    gate.approve_interface_mac_read(interface)?;
    let mut row = MIB_IF_ROW2 {
        InterfaceLuid: luid,
        ..Default::default()
    };
    let status = unsafe { GetIfEntry2(&mut row) };
    if status.0 != 0 {
        return Err(WifiError::InterfaceQueryStatus(status.0));
    }
    checked_current_interface_mac(
        *interface,
        row.InterfaceGuid,
        row.PhysicalAddressLength,
        row.PhysicalAddress,
    )
}

fn checked_current_interface_mac(
    requested_interface: GUID,
    returned_interface: GUID,
    address_length: u32,
    address: [u8; 32],
) -> Result<RadioIdentifierObservation, WifiError> {
    if requested_interface != returned_interface {
        return Err(WifiError::InterfaceMismatch);
    }
    if address_length != 6 {
        return Err(WifiError::InvalidPhysicalAddressLength(address_length));
    }
    let mut octets = [0u8; 6];
    octets.copy_from_slice(&address[..6]);
    Ok(RadioIdentifierObservation {
        identifier: RadioIdentifier::Mac(RadioAddress48::from_octets(octets)),
        observed_at: RadioObservationTime::Unknown,
    })
}

struct WlanClient(HANDLE);

impl Drop for WlanClient {
    fn drop(&mut self) {
        if !self.0.is_invalid() {
            // Cleanup is required even after a policy change; this only closes
            // the local WLAN client handle and does not initiate a capture.
            unsafe {
                let _ = WlanCloseHandle(self.0, None);
            }
        }
    }
}

struct WlanAllocation(*mut WLAN_BSS_LIST);

impl Drop for WlanAllocation {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // WLAN API allocations must be released with WlanFreeMemory.
            unsafe {
                WlanFreeMemory(self.0.cast());
            }
        }
    }
}

/// Read the operating system's current cached BSS list for one caller-selected
/// interface. This function never calls `WlanScan`; cache freshness is unknown.
/// It requires a fresh host-policy approval before opening WLAN and again
/// immediately before fetching BSS data. Provide caller-owned output sized for
/// the expected maximum. On error, output is left unchanged.
pub fn read_cached_bss<G: PolicyGate>(
    gate: &mut G,
    interface: &GUID,
    output: &mut [BssObservation],
) -> Result<BssReadSummary, WifiError> {
    gate.approve_cached_bss_read(interface)?;

    let mut negotiated_version = 0u32;
    let mut raw_handle = HANDLE(ptr::null_mut());
    let status = unsafe {
        WlanOpenHandle(
            WLAN_API_VERSION_2_0,
            None,
            &mut negotiated_version,
            &mut raw_handle,
        )
    };
    if status != 0 {
        if !raw_handle.is_invalid() {
            unsafe {
                let _ = WlanCloseHandle(raw_handle, None);
            }
        }
        return Err(WifiError::WlanStatus(status));
    }
    if raw_handle.is_invalid() {
        return Err(WifiError::InvalidHandle);
    }
    let client = WlanClient(raw_handle);

    gate.approve_cached_bss_read(interface)?;
    let mut raw_list: *mut WLAN_BSS_LIST = ptr::null_mut();
    // DOT11_BSS_TYPE_ANY is the WLAN ABI value 3. No scan is triggered by this
    // retrieval call; it returns records currently held in the OS cache.
    const DOT11_BSS_TYPE_ANY_VALUE: i32 = 3;
    let status = unsafe {
        WlanGetNetworkBssList(
            client.0,
            interface,
            None,
            DOT11_BSS_TYPE(DOT11_BSS_TYPE_ANY_VALUE),
            false,
            None,
            &mut raw_list,
        )
    };
    if status != 0 {
        if !raw_list.is_null() {
            unsafe {
                WlanFreeMemory(raw_list.cast());
            }
        }
        return Err(WifiError::WlanStatus(status));
    }
    if raw_list.is_null() {
        return Err(WifiError::NullResult);
    }
    let allocation = WlanAllocation(raw_list);
    let copied = unsafe { copy_validated_list(allocation.0, output)? };
    drop(allocation);
    drop(client);

    Ok(BssReadSummary {
        written: copied,
        timestamp: TimestampStatus::Unknown,
        cache_age_known: false,
    })
}

/// Validate the OS-described flexible array and every record's optional IE
/// span before reading or changing output. The allocation originates from the
/// WLAN API and its documented `dwTotalSize` is the only available extent; this
/// cannot independently prove the actual allocation size if the OS violates
/// its contract. `MAX_BSS_LIST_BYTES` bounds traversal after reading the
/// header.
unsafe fn copy_validated_list(
    raw: *const WLAN_BSS_LIST,
    output: &mut [BssObservation],
) -> Result<usize, WifiError> {
    if raw.is_null() {
        return Err(WifiError::NullResult);
    }

    // Read the two-field header without imposing a Rust alignment assumption
    // on the flexible-array allocation returned by the system API.
    let raw_bytes = raw.cast::<u8>();
    let total_u32 = unsafe { raw_bytes.cast::<u32>().read_unaligned() };
    let count_u32 = unsafe {
        raw_bytes
            .add(size_of::<u32>())
            .cast::<u32>()
            .read_unaligned()
    };
    let total = usize::try_from(total_u32).map_err(|_| WifiError::InvalidList)?;
    let count = usize::try_from(count_u32).map_err(|_| WifiError::InvalidList)?;
    let entries_offset = offset_of!(WLAN_BSS_LIST, wlanBssEntries);
    let entry_size = size_of::<WLAN_BSS_ENTRY>();
    let entries_end = validate_table(total, count, output.len(), entries_offset, entry_size)?;

    for index in 0..count {
        let entry_offset = entries_offset
            .checked_add(
                index
                    .checked_mul(entry_size)
                    .ok_or(WifiError::InvalidList)?,
            )
            .ok_or(WifiError::InvalidList)?;
        let entry = unsafe {
            raw_bytes
                .add(entry_offset)
                .cast::<WLAN_BSS_ENTRY>()
                .read_unaligned()
        };
        // The Windows SDK defines ulIeOffset relative to the beginning of
        // this WLAN_BSS_ENTRY, not relative to WLAN_BSS_LIST/allocation.
        let ie_relative = usize::try_from(entry.ulIeOffset).map_err(|_| WifiError::InvalidList)?;
        let ie_size = usize::try_from(entry.ulIeSize).map_err(|_| WifiError::InvalidList)?;
        let ie_start = entry_relative_ie_start(entry_offset, ie_relative)?;
        validate_ie_span(total, entries_end, ie_start, ie_size)?;
    }

    for (index, slot) in output.iter_mut().enumerate().take(count) {
        let entry_offset = entries_offset + index * entry_size;
        let entry = unsafe {
            raw_bytes
                .add(entry_offset)
                .cast::<WLAN_BSS_ENTRY>()
                .read_unaligned()
        };
        *slot = BssObservation {
            bssid: entry.dot11Bssid,
            rssi_dbm: entry.lRssi,
            link_quality: entry.uLinkQuality,
            center_frequency_khz: entry.ulChCenterFrequency,
        };
    }
    Ok(count)
}

fn validate_table(
    total_bytes: usize,
    item_count: usize,
    output_capacity: usize,
    items_offset: usize,
    item_size: usize,
) -> Result<usize, WifiError> {
    if item_size == 0 || total_bytes < items_offset || total_bytes > MAX_BSS_LIST_BYTES {
        return Err(WifiError::InvalidList);
    }
    let available = (total_bytes - items_offset) / item_size;
    if item_count > available {
        return Err(WifiError::InvalidList);
    }
    if item_count > MAX_BSS_OBSERVATIONS {
        return Err(WifiError::TooManyResults);
    }
    if output_capacity < item_count {
        return Err(WifiError::OutputTooSmall {
            required: item_count,
        });
    }
    items_offset
        .checked_add(
            item_count
                .checked_mul(item_size)
                .ok_or(WifiError::InvalidList)?,
        )
        .ok_or(WifiError::InvalidList)
}

fn validate_ie_span(
    total_bytes: usize,
    items_end: usize,
    ie_offset: usize,
    ie_size: usize,
) -> Result<(), WifiError> {
    if ie_size == 0 {
        return Ok(());
    }
    let end = ie_offset
        .checked_add(ie_size)
        .ok_or(WifiError::InvalidList)?;
    if ie_offset < items_end || end > total_bytes {
        return Err(WifiError::InvalidList);
    }
    Ok(())
}

fn entry_relative_ie_start(entry_offset: usize, ie_offset: usize) -> Result<usize, WifiError> {
    entry_offset
        .checked_add(ie_offset)
        .ok_or(WifiError::InvalidList)
}

#[cfg(test)]
#[path = "wifi_tests.rs"]
mod tests;
