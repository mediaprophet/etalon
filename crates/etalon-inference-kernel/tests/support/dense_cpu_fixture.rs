// Synthetic immutable input shared with the separately compiled original engine.
#[derive(Clone)]
pub struct Tensor {
    pub name: String,
    pub width: usize,
    pub rows: usize,
    pub values: Vec<f32>,
}
pub const LENGTHS: [usize; 13] = [32, 4, 4, 4, 4, 2, 2, 4, 6, 6, 6, 4, 5];
pub fn tensors(case: usize) -> Vec<Tensor> {
    let mut tensors = Vec::new();
    let mut add = |name: String, width: usize, rows: usize, norm: bool| {
        let seed = tensors.len() + 1;
        let values = (0..width * rows)
            .map(|i| {
                if norm {
                    0.75 + (i % 4) as f32 * 0.125
                } else {
                    (((i * 7 + seed * 3) % 19) as f32 - 9.0) * 0.0625
                }
            })
            .collect();
        tensors.push(Tensor {
            name,
            width,
            rows,
            values,
        });
    };
    add("token_embd.weight".into(), 4, 5, false);
    for layer in 0..2 {
        for (role, width, rows, norm) in [
            ("attn_norm", 4, 1, true),
            ("attn_q", 4, 4, false),
            ("attn_k", 4, 2, false),
            ("attn_v", 4, 2, false),
            ("attn_output", 4, 4, false),
            ("ffn_norm", 4, 1, true),
            ("ffn_gate", 4, 6, false),
            ("ffn_up", 4, 6, false),
            ("ffn_down", 6, 4, false),
        ] {
            add(format!("blk.{layer}.{role}.weight"), width, rows, norm);
        }
    }
    add("output_norm.weight".into(), 4, 1, true);
    add(
        "output.weight".into(),
        4,
        if case == 4 { 3 } else { 5 },
        false,
    );
    if case == 1 {
        tensors.retain(|t| !t.name.contains("norm"));
    }
    if case == 2 {
        tensors.retain(|t| t.name != "blk.1.ffn_down.weight");
    }
    if case == 3 {
        let t = tensors
            .iter_mut()
            .find(|t| t.name == "blk.1.attn_q.weight")
            .unwrap();
        t.rows = 3;
        t.values.truncate(12);
    }
    if case == 5 {
        tensors
            .iter_mut()
            .find(|t| t.name == "blk.1.attn_q.weight")
            .unwrap()
            .values
            .fill(f32::NAN);
    }
    if case == 6 {
        tensors.last_mut().unwrap().values.fill(f32::NAN);
    }
    tensors
}
