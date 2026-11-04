use super::{ResourceBudget, ResourceError, ResourcePlan};

pub(super) fn uniform(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    buffer(
        device,
        label,
        size,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    )
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);
const WEIGHT: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);
const UNIFORM: wgpu::BufferUsages = wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST);
const READBACK: wgpu::BufferUsages =
    wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);

fn admit(
    device: &wgpu::Device,
    budget: ResourceBudget,
    sizes: &[(u64, wgpu::BufferUsages)],
    policy: impl FnOnce(ResourcePlan) -> Result<(), ResourceError>,
) -> Result<(), ResourceError> {
    let mut bytes = 0u64;
    for &(size, usage) in sizes {
        bytes = bytes.checked_add(size).ok_or(ResourceError::SizeOverflow)?;
        budget.check_buffer(device, size, usage)?;
    }
    let plan = ResourcePlan {
        buffer_bytes: bytes,
        ..ResourcePlan::default()
    };
    budget.admit(plan)?;
    policy(plan)
}

#[derive(Clone, Copy, Debug)]
pub struct KvGeometry {
    pub cache_bytes: u64,
    pub attention_params_bytes: u64,
    pub mask_bytes: u64,
}

pub struct KvBuffers {
    pub cache: wgpu::Buffer,
    pub attention_params: wgpu::Buffer,
    pub mask: wgpu::Buffer,
}

impl KvBuffers {
    pub fn build(
        device: &wgpu::Device,
        geometry: KvGeometry,
        budget: ResourceBudget,
        policy: impl FnOnce(ResourcePlan) -> Result<(), ResourceError>,
    ) -> Result<Self, ResourceError> {
        let cache = geometry.cache_bytes.max(4);
        let params = geometry.attention_params_bytes.max(4);
        let mask = geometry.mask_bytes.max(4);
        admit(
            device,
            budget,
            &[(cache, STORAGE), (params, UNIFORM), (mask, WEIGHT)],
            policy,
        )?;
        Ok(Self {
            cache: buffer(device, "StaticKvCacheArena", cache, STORAGE),
            attention_params: uniform(device, "AttentionParams", params),
            mask: buffer(device, "AttentionKvMaskBatch", mask, WEIGHT),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GemmGeometry {
    pub weight_bytes: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub gemm_params_bytes: u64,
    pub elem_params_bytes: u64,
    pub norm_bytes: u64,
    pub staging_bytes: u64,
    #[cfg(target_arch = "wasm32")]
    pub prefill_work_bytes: u64,
}

pub struct GemmBuffers {
    pub input: wgpu::Buffer,
    pub weight: wgpu::Buffer,
    pub output: wgpu::Buffer,
    pub params: wgpu::Buffer,
    pub staging: wgpu::Buffer,
    pub aux: wgpu::Buffer,
    pub ffn: wgpu::Buffer,
    pub elem_params: wgpu::Buffer,
    pub norm: wgpu::Buffer,
    #[cfg(target_arch = "wasm32")]
    pub browser: BrowserGemmBuffers,
}

#[cfg(target_arch = "wasm32")]
pub struct BrowserGemmBuffers {
    pub role_weights: [wgpu::Buffer; 7],
    pub weight_b: wgpu::Buffer,
    pub prefill_scratch: wgpu::Buffer,
    pub work_a: wgpu::Buffer,
    pub work_b: wgpu::Buffer,
    pub q_proj: wgpu::Buffer,
    pub k_proj: wgpu::Buffer,
    pub v_proj: wgpu::Buffer,
}

impl GemmBuffers {
    pub fn build(
        device: &wgpu::Device,
        g: GemmGeometry,
        budget: ResourceBudget,
        policy: impl FnOnce(ResourcePlan) -> Result<(), ResourceError>,
    ) -> Result<Self, ResourceError> {
        let (w, i, o) = (
            g.weight_bytes.max(4),
            g.input_bytes.max(4),
            g.output_bytes.max(4),
        );
        let (p, e, n, s) = (
            g.gemm_params_bytes.max(4),
            g.elem_params_bytes.max(4),
            g.norm_bytes.max(4),
            g.staging_bytes.max(4),
        );
        let base = [
            (i, STORAGE),
            (w, WEIGHT),
            (o, STORAGE),
            (p, UNIFORM),
            (s, READBACK),
            (o, STORAGE),
            (o, STORAGE),
            (e, UNIFORM),
            (n, WEIGHT),
        ];
        #[cfg(not(target_arch = "wasm32"))]
        admit(device, budget, &base, policy)?;
        #[cfg(target_arch = "wasm32")]
        {
            let mut all = [(4, WEIGHT); 23];
            all[..9].copy_from_slice(&base);
            all[17] = (i, STORAGE);
            for slot in &mut all[18..] {
                *slot = (g.prefill_work_bytes.max(4), STORAGE);
            }
            admit(device, budget, &all, policy)?;
        }
        Ok(Self {
            input: buffer(device, "LayerGemmInput", i, STORAGE),
            weight: buffer(device, "LayerGemmWeight", w, WEIGHT),
            output: buffer(device, "LayerGemmOutput", o, STORAGE),
            params: uniform(device, "LayerGemmParams", p),
            staging: buffer(device, "LayerGemmStaging", s, READBACK),
            aux: buffer(device, "LayerGemmAux", o, STORAGE),
            ffn: buffer(device, "LayerGemmFfnUp", o, STORAGE),
            elem_params: uniform(device, "ElemParams", e),
            norm: buffer(device, "NormWeights", n, WEIGHT),
            #[cfg(target_arch = "wasm32")]
            browser: BrowserGemmBuffers {
                role_weights: [
                    "MC8WeightAttnK",
                    "MC8WeightAttnV",
                    "MC8WeightAttnQ",
                    "MC8WeightOProj",
                    "MC8WeightGate",
                    "MC8WeightUp",
                    "MC8WeightDown",
                ]
                .map(|label| buffer(device, label, 4, WEIGHT)),
                weight_b: buffer(device, "LayerGemmWeightB", 4, WEIGHT),
                prefill_scratch: buffer(device, "PrefillBatchScratch", i, STORAGE),
                work_a: buffer(
                    device,
                    "PrefillBatchWorkA",
                    g.prefill_work_bytes.max(4),
                    STORAGE,
                ),
                work_b: buffer(
                    device,
                    "PrefillBatchWorkB",
                    g.prefill_work_bytes.max(4),
                    STORAGE,
                ),
                q_proj: buffer(device, "MC8QProj", g.prefill_work_bytes.max(4), STORAGE),
                k_proj: buffer(device, "MC8KProj", g.prefill_work_bytes.max(4), STORAGE),
                v_proj: buffer(device, "MC8VProj", g.prefill_work_bytes.max(4), STORAGE),
            },
        })
    }
}
