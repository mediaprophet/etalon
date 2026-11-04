
use super::*;

#[test]
fn precomputed_chain_requires_every_expected_level_and_byte_count() {
    let levels = [vec![0u8; 4 * 2 * 4], vec![0u8; 2 * 1 * 4], vec![0u8; 4]];
    let borrowed = levels.iter().map(Vec::as_slice).collect::<Vec<_>>();
    assert_eq!(validate_precomputed_chain(4, 2, 8, &borrowed), Ok((3, 44)));
    assert_eq!(
        validate_precomputed_chain(4, 2, 8, &borrowed[..2]),
        Err(TextureUploadError::InvalidMipLevelCount)
    );
    let invalid = [borrowed[0], &borrowed[1][..7], borrowed[2]];
    assert_eq!(
        validate_precomputed_chain(4, 2, 8, &invalid),
        Err(TextureUploadError::InvalidMipLevelLength { level: 1 })
    );
}

#[test]
fn precomputed_window_accounts_only_selected_suffix() {
    let levels = [vec![0u8; 32], vec![0u8; 8], vec![0u8; 4]];
    let borrowed = levels.iter().map(Vec::as_slice).collect::<Vec<_>>();
    assert_eq!(validate_precomputed_chain(4, 2, 8, &borrowed), Ok((3, 44)));
    assert_eq!(
        resident_mip_window(4, 2, &borrowed, 1),
        Ok(ResidentMipWindow {
            first_mip: 1,
            width: 2,
            height: 1,
            mip_count: 2,
            resident_bytes: 12,
        })
    );
    assert_eq!(
        resident_mip_window(4, 2, &borrowed, 2),
        Ok(ResidentMipWindow {
            first_mip: 2,
            width: 1,
            height: 1,
            mip_count: 1,
            resident_bytes: 4,
        })
    );
    assert_eq!(
        resident_mip_window(4, 2, &borrowed, 3),
        Err(TextureUploadError::InvalidMipLevelCount)
    );
}

#[test]
fn coarse_base_dimensions_and_generated_chain_budget_start_at_selected_mip() {
    // Repeated floor-halving gives source mips 17x9, 8x4, 4x2, 2x1, 1x1.
    let (width, height) = coarse_base_dimensions(17, 9, 2).unwrap();
    assert_eq!((width, height), (4, 2));
    let (base_bytes, _) = validate_upload(width, height, 64, 4 * 2 * 4).unwrap();
    let (mip_count, resident_bytes) = mip_chain_layout(width, height).unwrap();
    assert_eq!(base_bytes, 32);
    assert_eq!(mip_count, 3);
    assert_eq!(resident_bytes, 44);

    assert_eq!(
        coarse_base_dimensions(17, 9, 5),
        Err(TextureUploadError::InvalidMipLevelCount)
    );
    assert_eq!(
        validate_upload(width, height, 64, 31),
        Err(TextureUploadError::InvalidRgba8Length)
    );
}

#[test]
fn precomputed_alpha_diagnostics_report_quantized_level_residuals() {
    let base = [
        255u8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
    ];
    let mip1 = [255u8, 255, 255, 255];
    let mip2 = [255u8, 255, 255, 255];
    let report = alpha_diagnostics(&[&base, &mip1, &mip2], 0.5);
    assert_eq!(report.level_count, 3);
    assert_eq!(report.base_coverage, 0.75);
    assert_eq!(report.signed_residuals[1], 0.25);
    assert_eq!(report.signed_residuals[2], 0.25);
}
