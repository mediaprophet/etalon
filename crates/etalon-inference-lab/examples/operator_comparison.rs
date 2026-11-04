//! Cold Q4_K comparison against caller-selected packed tensor bytes.
use etalon_inference_lab::operator_compare::compare_single_tensor;
use etalon_inference_kernel::operators::{q4k_lookup_workspace_floats, Q4K_SUPERBLOCK_BYTES, Q4K_SUPERBLOCK_ELEMS};
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: operator_comparison PACKED_Q4K_FILE IN_FEATURES OUT_FEATURES".into());
    }
    let input_dim = args[1].parse::<usize>()?;
    let output_dim = args[2].parse::<usize>()?;
    let block_elems = Q4K_SUPERBLOCK_ELEMS as usize;
    if input_dim == 0 || output_dim == 0 || input_dim % block_elems != 0 {
        return Err("dimensions must be positive and IN_FEATURES divisible by 256".into());
    }
    let expected = input_dim.checked_div(block_elems)
        .and_then(|n| n.checked_mul(output_dim))
        .and_then(|n| n.checked_mul(Q4K_SUPERBLOCK_BYTES))
        .ok_or("tensor size overflow")?;
    let budget = 256 * 1024 * 1024;
    let ws_len = q4k_lookup_workspace_floats(input_dim).map_err(|_| "workspace size overflow")?;
    let total_bytes = input_dim.checked_mul(2)
        .and_then(|n| n.checked_add(output_dim.checked_mul(2)?))
        .and_then(|n| n.checked_add(ws_len))
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|n| n.checked_add(expected))
        .ok_or("comparison memory size overflow")?;
    if total_bytes > budget {
        return Err("comparison exceeds the 256 MiB example memory budget".into());
    }
    let mut file = std::fs::File::open(&args[0])?;
    if file.metadata()?.len() != expected as u64 {
        return Err("file length does not match the declared packed Q4_K matrix".into());
    }
    let mut weights = vec![0; expected];
    file.read_exact(&mut weights)?;
    let mut extra = [0; 1];
    if file.read(&mut extra)? != 0 {
        return Err("file length changed during read".into());
    }
    // Deterministic input is a comparison stimulus, not model inference output.
    let input: Vec<f32> = (0..input_dim).map(|i| ((i % 128) as f32 * 0.02).sin()).collect();
    let receipt = compare_single_tensor(&weights, input_dim, output_dim, &input);
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    if !receipt.fidelity_satisfied {
        return Err("operator comparison exceeded the numerical fidelity tolerance".into());
    }
    Ok(())
}