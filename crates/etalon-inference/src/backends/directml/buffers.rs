use super::*;
// ─── D3D12 helpers ───────────────────────────────────────────────────────────

pub(super) unsafe fn commit_buffer(
    d3d: &ID3D12Device,
    size: u64,
    heap_type: D3D12_HEAP_TYPE,
    initial_state: D3D12_RESOURCE_STATES,
    flags: D3D12_RESOURCE_FLAGS,
) -> windows::core::Result<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size.max(4),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: heap_type,
        ..Default::default()
    };
    let mut out: Option<ID3D12Resource> = None;
    d3d.CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initial_state,
        None,
        &mut out,
    )?;
    Ok(out.unwrap())
}

pub(super) unsafe fn upload_buffer(
    d3d: &ID3D12Device,
    data: &[u8],
) -> windows::core::Result<ID3D12Resource> {
    let buf = commit_buffer(
        d3d,
        data.len() as u64,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_RESOURCE_FLAG_NONE,
    )?;
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    buf.Map(0, None, Some(&mut ptr))?;
    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
    buf.Unmap(0, None);
    Ok(buf)
}

pub(super) unsafe fn gpu_buffer(
    d3d: &ID3D12Device,
    size: u64,
) -> windows::core::Result<ID3D12Resource> {
    commit_buffer(
        d3d,
        size,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    )
}

pub(super) unsafe fn readback_buffer(
    d3d: &ID3D12Device,
    size: u64,
) -> windows::core::Result<ID3D12Resource> {
    commit_buffer(
        d3d,
        size,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE,
    )
}

/// Spin-wait until the D3D12 fence reaches `value` (avoids OS event dependency).
pub(super) unsafe fn fence_wait(
    d3d: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    value: u64,
) -> windows::core::Result<()> {
    let fence: ID3D12Fence = d3d.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
    queue.Signal(&fence, value)?;
    while fence.GetCompletedValue() < value {
        std::hint::spin_loop();
    }
    Ok(())
}

/// Execute and flush a single command list, then spin-wait for completion.
pub(super) unsafe fn submit_and_wait(
    d3d: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    alloc: &ID3D12CommandAllocator,
    cmd: &ID3D12GraphicsCommandList,
    value: u64,
) -> windows::core::Result<()> {
    cmd.Close()?;
    queue.ExecuteCommandLists(&[Some(cmd.cast()?)]);
    fence_wait(d3d, queue, value)?;
    alloc.Reset()?;
    Ok(())
}
