use core::{future::Future, task::Poll};
use etalon_nquin::{GpuQueryError, GpuQuinSieve, GpuRadixSorter, NQuinWire, QuinField};
use serde_json::{json, Value};

use crate::browser::failure;

fn keys(count: usize) -> Vec<u64> {
    const PATTERNS: [u64; 8] = [
        0,
        u64::MAX,
        1 << 63,
        1 << 32,
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0x00ff_00ff_00ff_00ff,
        0xff00_ff00_ff00_ff00,
    ];
    (0..count)
        .rev()
        .map(|i| {
            PATTERNS[(i / 3) % PATTERNS.len()]
                ^ ((i / (PATTERNS.len() * 3)) as u64).wrapping_mul(0x0102_0304_0506_0708)
        })
        .collect()
}

async fn check_sort(
    sorter: &GpuRadixSorter,
    device: &etalon::Device,
    queue: &etalon::Queue,
    count: usize,
) -> Result<Value, Value> {
    let keys = keys(count);
    let mut oracle_keys = keys.clone();
    let mut expected: Vec<u32> = (0..count as u32).collect();
    etalon_nquin::radix_sort_u64_cpu(&mut oracle_keys, &mut expected);
    let mut independent: Vec<u32> = (0..count as u32).collect();
    independent.sort_by_key(|&row| keys[row as usize]);
    if expected != independent {
        return Err(failure(
            "radix-cpu-oracle",
            "CPU radix and independent stable sort disagree",
        ));
    }
    let mut output = vec![u32::MAX; count + 1];
    let written = sorter
        .sort_indices_async(device, queue, &keys, &mut output)
        .await
        .map_err(|error| failure("gpu-radix-readback", &format!("length {count}: {error}")))?;
    if written != count || output[..written] != expected || output[count] != u32::MAX {
        return Err(failure(
            "gpu-radix-oracle",
            &format!("Stable GPU order or output bounds differ at length {count}"),
        ));
    }
    Ok(json!({"elements": count, "passes": 8, "mismatches": 0,
        "first_indices": &expected[..count.min(8)],
        "first_keys_u64": oracle_keys.iter().take(4).map(u64::to_string).collect::<Vec<_>>()}))
}

pub(crate) async fn radix(device: &etalon::Device, queue: &etalon::Queue) -> Result<Value, Value> {
    let sorter = GpuRadixSorter::new(device);
    let mut empty_output = [0xaabb_ccdd];
    if sorter
        .sort_indices_async(device, queue, &[], &mut empty_output)
        .await
        != Ok(0)
        || empty_output != [0xaabb_ccdd]
    {
        return Err(failure(
            "gpu-radix-empty",
            "Empty sort changed its caller output",
        ));
    }
    let short = sorter
        .sort_indices_async(device, queue, &[2, 1], &mut [0])
        .await;
    if short
        != Err(GpuQueryError::OutputTooSmall {
            required: 2,
            provided: 1,
        })
    {
        return Err(failure(
            "gpu-radix-capacity",
            "Short output did not return the typed capacity error",
        ));
    }
    let mut checks = Vec::new();
    for count in [1usize, 63, 64, 65, 255, 256, 257, 1025] {
        checks.push(check_sort(&sorter, device, queue, count).await?);
    }

    let cancel_keys = keys(65);
    let mut cancel_output = [0; 65];
    let mut pending =
        Box::pin(sorter.sort_indices_async(device, queue, &cancel_keys, &mut cancel_output));
    let first_poll = core::future::poll_fn(|cx| Poll::Ready(pending.as_mut().poll(cx))).await;
    if !first_poll.is_pending() {
        return Err(failure(
            "gpu-radix-cancel",
            "Browser readback completed before cancellation could be exercised",
        ));
    }
    drop(pending);
    let after_cancel = check_sort(&sorter, device, queue, 65).await?;
    Ok(
        json!({"execution": "Etalon GPU radix shaders with async WASM readback",
        "workgroup_size": 64, "digit_bins": 256, "cases": checks,
        "cpu_oracles": ["eight-pass CPU radix", "independent stable sort"],
        "empty_checked": true, "typed_short_output_checked": true,
        "pending_map_cancelled": true, "after_cancellation": after_cancel, "mismatches": 0}),
    )
}

fn quins(count: usize) -> Vec<NQuinWire> {
    (0..count)
        .map(|i| {
            let mut quin = NQuinWire::new(
                i as u64,
                if i % 3 == 0 { u64::MAX } else { 7 },
                (1 << 63) | (i % 5) as u64,
                0x0123_4567_89ab_cdef,
                0,
                0,
            );
            quin.parity = quin.compute_parity();
            quin
        })
        .collect()
}

async fn check_sieve(
    sieve: &GpuQuinSieve,
    device: &etalon::Device,
    queue: &etalon::Queue,
    quins: &[NQuinWire],
    field: QuinField,
    needle: u64,
) -> Result<Value, Value> {
    let expected: Vec<u32> = quins
        .iter()
        .enumerate()
        .filter(|(_, quin)| field.extract(quin) == needle)
        .map(|(i, _)| i as u32)
        .collect();
    let mut cpu_quins = vec![NQuinWire::new(0, 0, 0, 0, 0, 0); quins.len()];
    let cpu_count = etalon_nquin::sieve_eq_cpu(quins, field, needle, &mut cpu_quins);
    if cpu_count != expected.len()
        || expected
            .iter()
            .enumerate()
            .any(|(i, &row)| cpu_quins[i] != quins[row as usize])
    {
        return Err(failure(
            "sieve-cpu-oracle",
            "CPU wire sieve and row-index oracle disagree",
        ));
    }
    let mut output = vec![u32::MAX; quins.len() + 1];
    let written = sieve
        .sieve_eq_indices_async(device, queue, quins, field, needle, &mut output)
        .await
        .map_err(|error| failure("gpu-sieve-readback", &error.to_string()))?;
    if written != expected.len()
        || output[..written] != expected
        || output[written..].iter().any(|&v| v != u32::MAX)
    {
        return Err(failure(
            "gpu-sieve-oracle",
            "GPU matching rows or output bounds differ from CPU oracle",
        ));
    }
    Ok(
        json!({"elements": quins.len(), "field": field.as_u32(), "needle_u64": needle.to_string(),
        "matches": written, "first_indices": expected.iter().take(12).collect::<Vec<_>>(), "mismatches": 0}),
    )
}

pub(crate) async fn sieve(device: &etalon::Device, queue: &etalon::Queue) -> Result<Value, Value> {
    let sieve = GpuQuinSieve::new(device);
    let mut empty_output = [0xaabb_ccdd];
    if sieve
        .sieve_eq_indices_async(device, queue, &[], QuinField::Subject, 0, &mut empty_output)
        .await
        != Ok(0)
        || empty_output != [0xaabb_ccdd]
    {
        return Err(failure(
            "gpu-sieve-empty",
            "Empty sieve changed its caller output",
        ));
    }
    let mut checks = Vec::new();
    for count in [1usize, 31, 32, 33, 65, 257] {
        let quins = quins(count);
        for (field, needle) in [
            (QuinField::Predicate, u64::MAX),
            (QuinField::Subject, u64::MAX),
            (QuinField::Object, (1 << 63) | 3),
            (QuinField::Context, 0x0123_4567_89ab_cdef),
        ] {
            checks.push(check_sieve(&sieve, device, queue, &quins, field, needle).await?);
        }
    }
    let quins = quins(33);
    let mut short_output = [0xaabb_ccdd; 2];
    let short = sieve
        .sieve_eq_indices_async(
            device,
            queue,
            &quins,
            QuinField::Context,
            0x0123_4567_89ab_cdef,
            &mut short_output,
        )
        .await;
    if short
        != Err(GpuQueryError::OutputTooSmall {
            required: 33,
            provided: 2,
        })
        || short_output != [0xaabb_ccdd; 2]
    {
        return Err(failure(
            "gpu-sieve-capacity",
            "Short output did not fail with an unchanged caller buffer",
        ));
    }
    let no_matches = sieve
        .sieve_eq_indices_async(device, queue, &quins, QuinField::Subject, u64::MAX, &mut [])
        .await;
    if no_matches != Ok(0) {
        return Err(failure(
            "gpu-sieve-zero-capacity",
            "No matches should fit an empty caller buffer",
        ));
    }
    checks.push(
        check_sieve(
            &sieve,
            device,
            queue,
            &quins,
            QuinField::Context,
            0x0123_4567_89ab_cdef,
        )
        .await?,
    );
    Ok(
        json!({"execution": "Etalon warp-transposed GPU sieve with async WASM readback", "workgroup_size": 32,
        "wire_bytes": 48, "cases": checks, "all_fields_checked": true, "empty_checked": true,
        "all_and_no_matches_checked": true, "typed_short_output_checked": true,
        "zero_capacity_no_matches_checked": true, "mismatches": 0}),
    )
}
