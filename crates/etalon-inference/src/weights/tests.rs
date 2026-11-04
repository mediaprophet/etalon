use super::*;

#[test]
fn reads_only_the_requested_flattened_row() {
    let info = TensorInfo {
        dims: [2, 2, 2, 1],
        n_dims: 3,
        ggml_type: etalon_model::ggml::GGML_TYPE_F32,
        byte_offset: 16,
    };
    let mut source = vec![0u8; 16];
    for value in 0..8u32 {
        source.extend_from_slice(&(value as f32).to_le_bytes());
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trunk.gguf");
    std::fs::write(&path, source).unwrap();
    let mut reader = TrunkNvmeReader::open(&path, 0).unwrap();
    let mut raw = [0u8; 8];
    let mut output = [0f32; 2];
    assert_eq!(
        reader
            .read_row_into(&info, 3, &mut raw, &mut output)
            .unwrap(),
        2
    );
    assert_eq!(output, [6.0, 7.0]);
}

#[test]
fn streams_only_the_selected_expert_plane() {
    let info = TensorInfo {
        dims: [2, 2, 2, 1],
        n_dims: 3,
        ggml_type: etalon_model::ggml::GGML_TYPE_F32,
        byte_offset: 0,
    };
    // expert 0: [[1, 2], [3, 4]], expert 1: [[5, 6], [7, 8]]
    let mut source = Vec::new();
    for value in 1..=8u32 {
        source.extend_from_slice(&(value as f32).to_le_bytes());
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("experts.gguf");
    std::fs::write(&path, source).unwrap();
    let mut reader = TrunkNvmeReader::open(&path, 0).unwrap();
    let mut raw = [0u8; 8];
    let mut row = [0f32; 2];
    let mut output = [0f32; 2];
    assert_eq!(
        reader
            .gemv_expert_plane_into(&info, 1, &[1.0, 1.0], &mut output, &mut raw, &mut row,)
            .unwrap(),
        2
    );
    assert_eq!(output, [11.0, 15.0]);
}

#[test]
fn argmax_streams_vocabulary_without_logits_buffer() {
    let info = TensorInfo {
        dims: [2, 3, 1, 1],
        n_dims: 2,
        ggml_type: etalon_model::ggml::GGML_TYPE_F32,
        byte_offset: 0,
    };
    // Dot with [1, 1] yields 3, 9, and 9. The earlier token must win.
    let mut source = Vec::new();
    for value in [1.0f32, 2.0, 4.0, 5.0, 7.0, 2.0] {
        source.extend_from_slice(&value.to_le_bytes());
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("logits.gguf");
    std::fs::write(&path, source).unwrap();
    let mut reader = TrunkNvmeReader::open(&path, 0).unwrap();
    let mut raw = [0u8; 8];
    let mut row = [0f32; 2];
    assert_eq!(
        reader
            .argmax_rows(&info, &[1.0, 1.0], &mut raw, &mut row)
            .unwrap(),
        StreamedArgmax {
            token_id: 1,
            logit: 9.0
        }
    );
}

#[test]
fn mapped_parallel_paths_match_streamed_results() {
    // Enough rows to span several parallel chunks, with a deliberate
    // cross-chunk tie (rows 1 and 4097 both score 9) so the merge must
    // keep the ascending-row winner the sequential scan would pick.
    let rows = 5000usize;
    let info = TensorInfo {
        dims: [2, rows as u64, 1, 1],
        n_dims: 2,
        ggml_type: etalon_model::ggml::GGML_TYPE_F32,
        byte_offset: 0,
    };
    let mut source = Vec::with_capacity(rows * 8);
    for row in 0..rows {
        let (a, b) = if row == 1 || row == 4097 {
            (4.0f32, 5.0f32)
        } else {
            ((row % 7) as f32, (row % 3) as f32)
        };
        source.extend_from_slice(&a.to_le_bytes());
        source.extend_from_slice(&b.to_le_bytes());
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("mapped.gguf");
    std::fs::write(&path, source).unwrap();

    let input = [1.0f32, 1.0];
    let mut streamed = TrunkNvmeReader::open(&path, 0).unwrap();
    let mut mapped = TrunkNvmeReader::open_mapped(&path, 0).unwrap();
    assert!(mapped.is_mapped());
    let mut raw = [0u8; 8];
    let mut row = [0f32; 2];

    let mut gemv_streamed = [0f32; 5000];
    let mut gemv_mapped = [0f32; 5000];
    streamed
        .gemv_rows_into(
            &info,
            0,
            rows,
            &input,
            &mut gemv_streamed,
            &mut raw,
            &mut row,
        )
        .unwrap();
    mapped
        .gemv_rows_into(&info, 0, rows, &input, &mut gemv_mapped, &mut raw, &mut row)
        .unwrap();
    assert_eq!(gemv_streamed, gemv_mapped);

    let mut top_streamed = [StreamedArgmax {
        token_id: 0,
        logit: 0.0,
    }; 4];
    let mut top_mapped = [StreamedArgmax {
        token_id: 0,
        logit: 0.0,
    }; 4];
    streamed
        .topk_rows(&info, &input, &mut raw, &mut row, &mut top_streamed)
        .unwrap();
    mapped
        .topk_rows(&info, &input, &mut raw, &mut row, &mut top_mapped)
        .unwrap();
    assert_eq!(top_streamed, top_mapped);
    assert_eq!(top_mapped[0].token_id, 1);
    assert_eq!(top_mapped[1].token_id, 4097);
}
