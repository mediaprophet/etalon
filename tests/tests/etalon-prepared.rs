#![cfg(not(target_family = "wasm"))]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Duration,
};

struct CountingAllocator;
static MEASURE: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if MEASURE.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if MEASURE.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if MEASURE.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if MEASURE.load(Ordering::Relaxed) {
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(pointer, layout) }
    }
}

struct Measurement;

impl Measurement {
    fn begin() -> Self {
        ALLOCATIONS.store(0, Ordering::SeqCst);
        DEALLOCATIONS.store(0, Ordering::SeqCst);
        MEASURE.store(true, Ordering::SeqCst);
        Self
    }
}

impl Drop for Measurement {
    fn drop(&mut self) {
        MEASURE.store(false, Ordering::SeqCst);
    }
}

#[test]
#[ignore = "explicit Vulkan target qualification; run alone with --ignored --test-threads=1"]
fn prepared_compute_allocator_qualification() {
    let instance = etalon::Instance::new(etalon::InstanceDescriptor {
        backends: etalon::Backends::VULKAN,
        flags: etalon::InstanceFlags::empty(),
        ..etalon::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&etalon::RequestAdapterOptions::default())).unwrap();
    let desc = etalon::PreparedComputeDescriptor {
        shader: r"
            @group(0) @binding(0) var<storage, read> input: array<u32>;
            @group(0) @binding(1) var<storage, read_write> output: array<u32>;
            @group(0) @binding(2) var<uniform> parameters: vec4<u32>;
            @compute @workgroup_size(1)
            fn main() { output[0] = input[0] + parameters.x; }
        ",
        entry_point: "main",
        input_size: 64,
        parameter_size: 16,
        output_size: 64,
        workgroups: [1; 3],
        buffer_budget: 288,
    };
    let mut lane = adapter.prepare_compute(&desc).unwrap();
    let mut output = [0; 64];
    let parameters = [0; 16];
    lane.execute(&[0; 64], &parameters, &mut output, Duration::from_secs(10)).unwrap();
    let mut result = Ok(0);
    let measurement = Measurement::begin();
    for _ in 0..128 {
        result = lane.execute(&[0; 64], &parameters, &mut output, Duration::from_secs(10));
        if result.is_err() { break; }
    }
    drop(measurement);
    assert_eq!(result, Ok(129));
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
    assert_eq!(DEALLOCATIONS.load(Ordering::SeqCst), 0);

    let measurement = Measurement::begin();
    let capacity = lane.execute(&[0; 65], &parameters, &mut output, Duration::ZERO);
    let parameters_error = lane.execute(&[], &parameters[..12], &mut output, Duration::ZERO);
    drop(measurement);
    assert_eq!(capacity, Err(etalon::PreparedComputeError::Capacity));
    assert_eq!(parameters_error, Err(etalon::PreparedComputeError::InvalidDescriptor));
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
    assert_eq!(DEALLOCATIONS.load(Ordering::SeqCst), 0);
}
