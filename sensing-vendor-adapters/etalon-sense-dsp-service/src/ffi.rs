use crate::{native_context::NativeContext, HexagonIqPowerService};
use etalon_sense_dsp_wire::{decode_request, generation_from_uri, request_bytes, response_bytes,
    HEADER_BYTES, MAX_PAIRS, MAX_URI_BYTES};

static CONTEXT: NativeContext = NativeContext::new();

/// # Safety
/// Supply a live readable URI allocation including its final NUL, and a live
/// writable disjoint eight-byte handle allocation. Neither may be unmapped or
/// accessed concurrently. Image provider/capacity must be immutable. The image
/// must use panic=abort. Numeric checks do not establish allocation provenance.
#[no_mangle]
pub unsafe extern "C" fn etalon_iq_power_open(provider_id: u64, max_pairs: u32,
    uri: *const u8, uri_len: usize, handle: *mut u64) -> i32 {
    if !cfg!(target_arch = "hexagon") { return -2; }
    if provider_id == 0 || max_pairs == 0 || max_pairs > MAX_PAIRS || uri_len < 2
        || uri_len > MAX_URI_BYTES + 1 || !disjoint(uri, uri_len, handle.cast(), 8)
    { return -1; }
    let uri = unsafe { core::slice::from_raw_parts(uri, uri_len) };
    if uri[uri_len - 1] != 0 { return -1; }
    let generation = match generation_from_uri(&uri[..uri_len - 1]) {
        Ok(generation) => generation, Err(_) => return -1,
    };
    match CONTEXT.open(generation) {
        Ok(token) => { unsafe { handle.write_unaligned(token) }; 0 },
        Err(_) => -3,
    }
}

#[no_mangle]
pub extern "C" fn etalon_iq_power_close(handle: u64) -> i32 {
    match CONTEXT.close(handle) { Ok(()) => 0, Err(_) => -3 }
}

/// # Safety
/// Both slices must be live, mapped, disjoint allocations of these lengths,
/// readable/writable respectively, and exclusively borrowed until return.
/// Supply the image's immutable provider/capacity. The final image must abort
/// on panic; allocation provenance and cache/mapping lifetime are SDK duties.
#[no_mangle]
pub unsafe extern "C" fn etalon_iq_power_process(provider_id: u64, max_pairs: u32,
    handle: u64, input: *const u8, input_len: usize, output: *mut u8, output_len: usize,
) -> i32 {
    if max_pairs == 0 || max_pairs > MAX_PAIRS || input_len < HEADER_BYTES
        || output_len < HEADER_BYTES || !disjoint(input, input_len, output, output_len)
    { return -1; }
    let max_input = match request_bytes(max_pairs) { Ok(n) => n, Err(_) => return -1 };
    let max_output = match response_bytes(max_pairs) { Ok(n) => n, Err(_) => return -1 };
    if input_len > max_input || output_len > max_output { return -1; }
    match CONTEXT.with_active(handle, |active| {
        let generation = active.generation();
        let mut service = match HexagonIqPowerService::new(provider_id, generation, max_pairs) {
            Ok(service) => service, Err(_) => return -2,
        };
        // The unsafe caller supplies valid allocations; the native lease spans dispatch.
        let input = unsafe { core::slice::from_raw_parts(input, input_len) };
        let (identity, pairs, _) = match decode_request(input) { Ok(request) => request, Err(_) => return -1 };
        if identity.provider_id() != provider_id || identity.generation() != generation
            || pairs > max_pairs || response_bytes(pairs) != Ok(output_len)
        { return -1; }
        // Consume after full validation; a subsequent kernel failure cannot reuse it.
        if active.consume_request(identity.request_id()).is_err() { return -3; }
        let output = unsafe { core::slice::from_raw_parts_mut(output, output_len) };
        match service.process(input, output) { Ok(()) => 0, Err(_) => -3 }
    }) { Ok(status) => status, Err(_) => -3 }
}

fn disjoint(input: *const u8, input_len: usize, output: *mut u8, output_len: usize) -> bool {
    if input.is_null() || output.is_null() || input_len > isize::MAX as usize
        || output_len > isize::MAX as usize { return false; }
    let Some(input_end) = (input as usize).checked_add(input_len) else { return false; };
    let Some(output_end) = (output as usize).checked_add(output_len) else { return false; };
    (input as usize) >= output_end || (output as usize) >= input_end
}
