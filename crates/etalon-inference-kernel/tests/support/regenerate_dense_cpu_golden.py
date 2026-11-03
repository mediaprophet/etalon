"""Reproduce full dense CPU traces using immutable original Qualia source.

python regenerate_dense_cpu_golden.py verify
python regenerate_dense_cpu_golden.py generate --qualia-root PATH --dependencies PATH/target/debug/deps --output NEW_FILE

Reads git objects at the frozen revision, verifies every source hash, and uses
cached Rust 1.96.1 libraries. No network, Cargo build, GPU or downloaded model.
Original forward/math/index/codecs are unchanged. Only cold construction bypasses
the unrelated 1000-token admission floor for this tiny synthetic model; observations
read all 13 buffers. Unused cold constructors are omitted, not replaced by stubs.
TemporaryDirectory cleans scratch on success/error/unwind; 32 MiB is a post-compile
retention check, not a bound on compiler peak output. Promotion is exclusive and
removes a newly created partial output if writing fails. Existing files survive.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

REVISION = '07018bb8dbb123f105a91609532b9b564e54a2af'
SOURCE_HASHES = {'crates/qualia-core-db/src/gguf_bridge/cpu_ops.rs': 'c0e2dd70865805176ab9954d780e43e091972fc650741ee0092fe9ccc0f1456d',
 'crates/qualia-core-db/src/gguf_bridge/mod.rs': '8be9f57d5a97548e40a87120f75edfd09e1f8a91ef7f8221e9849bd82a04aebe',
 'crates/qualia-core-db/src/gguf_bridge/wasm_cpu/forward.rs': '5e2d885b2920557987a4078e48f5b84d06201a7e1b6d404c48bff001d8dc5622',
 'crates/qualia-core-db/src/gguf_bridge/wasm_cpu/model.rs': '24bfeee492ea1daec760ff3fa0a6ce9d3f357fc236ec88ae0e901fd5f8fceaa4',
 'crates/qualia-core-db/src/inference/ggml_quants.rs': '6b7421fc4db3cbffe5d99ec80cdaf9f5fef0be75ff43ea13eb5340554873c716',
 'crates/qualia-core-db/src/inference/gguf_sharder/hyperparams.rs': 'd220bb6ae04ef6fc63083f3d9d65ceae917e39e14f8105997b60f422449c353a',
 'crates/qualia-core-db/src/inference/gguf_sharder/mod.rs': 'bc448c25040f24224ae7eb1ce96a70c871afce5a06e23e549f560525603cab48',
 'crates/qualia-core-db/src/inference/gguf_sharder/sharder.rs': '103f71a45a33e6b0754a21f7bb14186f1df5109e2ffcec2eb87b6d8d37822a0d',
 'crates/qualia-core-db/src/inference/gguf_sharder/tensor_index.rs': 'ee6dae6d57365f9efe2ec3cca654094a1573b0579cbb3433741ffc706f9e8d40',
 'crates/qualia-core-db/src/inference/gguf_sharder/tests.rs': '3fb494caf3c4c58971cc66486a0f027a56cf479fdfd2e562dc5799803c3461d2',
 'crates/qualia-core-db/src/inference/gguf_sharder/tokenizer.rs': '9230d9a50a72e17163ccaf24ccd28055e8e8881b61c61d3a3c84d00cb2eeeb41',
 'crates/qualia-core-db/src/inference/gguf_sharder/tokenizer/decode.rs': '9de13066c995be20c7ddc23cee858435baaeb486c8d54fc46bdeba7a21c1071d',
 'crates/qualia-core-db/src/inference/gguf_sharder/tokenizer/hf_json.rs': '385f402707beed1530470f33b38e00eb3c6a20beee1c0a67f3be203a7426a84d',
 'crates/qualia-core-db/src/inference/gguf_sharder/tokenizer/pretokenizer/mod.rs': '5aa79dbd048b08ef441ccf340e5656b915d26413f3ab72b7741c76dac020b2d5',
 'crates/qualia-core-db/src/inference/gguf_sharder/tokenizer/pretokenizer/scalar.rs': '39de6b59830f970b93304a36711581839142acff76a06365be649096df3714cf',
 'crates/qualia-core-db/src/inference/gguf_sharder/tokenizer/pretokenizer/simd.rs': 'c15df57b4e75f347003c64fc70e480e541ef2859f64da2ee2f81a1c4df3a68b0',
 'crates/qualia-core-db/src/inference/gguf_sharder/tokenizer/pretokenizer/tests.rs': 'bffc1628568515c2a332c4d503f7cad59b64f8cfa836007e1b8699c791f3b0fb',
 'crates/qualia-core-db/src/inference/gguf_sharder/types.rs': '5fa74d3907293e1685f87260a2ef4f702d669bb3f27280ed70f7e6324c958ad0',
 'crates/qualia-core-db/src/inference/moe/nvfp4.rs': 'a2eacb23be4440c40f2c8f264d39c3ef71db40cf65403b4de14ae800c52848ae',
 'crates/qualia-core-db/src/inference/tensor_roles.rs': '6f33e4c974a63dbc4a77ba676cff3d053b00ed7e2eef41b72f3a2445c9ec054d',
 'crates/qualia-core-db/src/inference/ternary.rs': '0954baf8316740aaae5461ac6c6038a5308d29061962a47e65a7536f136d6a9c',
 'crates/qualia-core-db/src/lib.rs': '15f1fe80ee182678e7af3febee47405970e38eadca2c62e996159ffb4f627337',
 'crates/qualia-core-db/src/q42/p64_weight/layout.rs': 'd36a7189095da8c37a1defac449f0a28132de89d437e170826af3e58025828d7',
 'crates/qualia-core-db/src/shaders/ternary_gemm.wgsl': '650209f662f49ee014b068e6c8f714ef5aa385a3ce3840355fd51de4a63fd7e1',
 'crates/qualia-core-db/src/shaders/ternary_gemm_2bit.wgsl': 'b6eb7142eb6badabf469673eff2391c2ab62cbbddf47ca44bb645fda90c40d02'}
GOLDEN_HASH = '9df017b733061c9ef188ac026ba9d30f635c64d9c33c3c7eead361af22abb4dc'
TEMPLATE = r'''
#![allow(dead_code,unused_imports)]
#[path="@ROOT@/crates/qualia-core-db/src/inference/ternary.rs"] pub mod ternary;
#[path="@ROOT@/crates/qualia-core-db/src/q42/p64_weight/layout.rs"] pub mod p64_weight;
#[path="@ROOT@/crates/qualia-core-db/src/inference/gguf_sharder/mod.rs"] pub mod gguf_sharder;
#[path="@ROOT@/crates/qualia-core-db/src/inference/ggml_quants.rs"] pub mod ggml_quants;
#[path="@ROOT@/crates/qualia-core-db/src/inference/tensor_roles.rs"] pub mod tensor_roles;
pub mod inference {pub use crate::tensor_roles;pub mod moe {pub mod nvfp4 {include!("actual_fp8.rs");}}}
pub struct NQuin{pub subject:u64,pub predicate:u64,pub object:u64,pub context:u64,pub metadata:u64,pub parity:u64}
pub struct QualiaSuperBlock;
pub const MODALITY_FLAG_VISION_TENSOR:u8=0;
pub const MODALITY_FLAG_LLM_TENSOR:u8=0;
pub const MODALITY_FLAG_DENSE_PHYSICS:u8=0;
include!("q_hash.rs");
#[path="@FIXTURE@"] mod fixture;
mod gguf_bridge {
include!("actual_helpers.rs");
pub mod wasm_cpu {
const CPU_WASM_DEFAULT_CONTEXT:usize=512;
const CPU_WASM_MAX_CONTEXT:usize=4096;
#[path="@SCRATCH@/model_decl.rs"] mod model;
#[path="@ROOT@/crates/qualia-core-db/src/gguf_bridge/wasm_cpu/forward.rs"] mod forward;
pub use model::*;
pub fn make(bytes:Vec<u8>)->CpuWasmEngine {
let index=crate::gguf_sharder::GgufTensorIndex::from_gguf(&bytes);
assert_eq!(index.hyperparams.effective_n_kv_head(),1);
CpuWasmEngine {model:bytes.into(),index,tokenizer:crate::gguf_sharder::GgufTokenizer::default(),
max_context:4,n_embd:4,n_ffn:6,n_layer:2,n_head:2,n_kv_head:1,head_dim:2,kv_plane_elems:16,
kv:vec![0.;32],hidden:vec![0.;4],normed:vec![0.;4],norm_weight:vec![0.;4],q:vec![0.;4],k:vec![0.;2],v:vec![0.;2],attention:vec![0.;4],projection:vec![0.;6],gate:vec![0.;6],up:vec![0.;6],scores:vec![0.;4],logits:vec![0.;5]}
}
pub fn snapshot(e:&CpuWasmEngine)->Vec<u32> {
[&e.kv,&e.hidden,&e.normed,&e.norm_weight,&e.q,&e.k,&e.v,&e.attention,&e.projection,&e.gate,&e.up,&e.scores,&e.logits].iter().flat_map(|b|b.iter().map(|v|v.to_bits())).collect()
}
}
}
fn str64(b:&mut Vec<u8>,s:&str){b.extend_from_slice(&(s.len() as u64).to_le_bytes());b.extend_from_slice(s.as_bytes());}
fn make_model(case:usize)->Vec<u8>{
 let ts=fixture::tensors(case);let mut b=b"GGUF".to_vec();b.extend_from_slice(&3u32.to_le_bytes());b.extend_from_slice(&(ts.len()as u64).to_le_bytes());b.extend_from_slice(&4u64.to_le_bytes());
 for(key,v)in[("llama.block_count",2u32),("llama.embedding_length",4),("llama.attention.head_count",2),("llama.attention.head_count_kv",1)]{str64(&mut b,key);b.extend_from_slice(&4u32.to_le_bytes());b.extend_from_slice(&v.to_le_bytes());}
 let mut offset=0u64;
 for t in &ts {str64(&mut b,&t.name);b.extend_from_slice(&2u32.to_le_bytes());b.extend_from_slice(&(t.width as u64).to_le_bytes());b.extend_from_slice(&(t.rows as u64).to_le_bytes());b.extend_from_slice(&0u32.to_le_bytes());b.extend_from_slice(&offset.to_le_bytes());offset+=(t.values.len()*4)as u64;}
 b.resize((b.len()+31)&!31,0);
 for t in &ts {for v in &t.values {b.extend_from_slice(&v.to_le_bytes());}}
 b
}
fn main(){
 println!("// Generated from frozen actual original CpuWasmEngine; see dense_cpu_contract.rs provenance.");
 println!("#[rustfmt::skip]");
 println!("pub const GOLDEN: &[(usize,usize,i64,u32,&[u32])] = &[");
 for case in 0..7 {let mut e=gguf_bridge::wasm_cpu::make(make_model(case));
 for stage in 0..6 {
 let result=match stage {0=>e.ingest_token(1,0).map(|_|None),1=>e.run_token(3,1).map(Some),2=>e.run_token(2,2).map(Some),3=>{e.reset();Ok(None)},4=>e.run_token(0,0).map(Some),_=>e.run_token(1,4).map(Some)};
 let(token,logit)=match result{Ok(Some(s))=>(s.token_id as i64,s.max_logit.to_bits()),Ok(None)=>(-1,0),Err(ref err)=>{eprintln!("case={case} stage={stage} error={err:?}");(-2,0)}};
 println!("({case},{stage},{token},{logit},&{:?}),",gguf_bridge::wasm_cpu::snapshot(&e));
 }
 }
 println!("];");
}
'''


def function(text, signature):
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def verify_bytes(data):
    if len(data) > 65536:
        raise ValueError("trace exceeds 64 KiB")
    if hashlib.sha256(data).hexdigest() != GOLDEN_HASH:
        raise ValueError("trace SHA256 differs from frozen original")


def promote(data, destination, opener=open):
    verify_bytes(data)
    owned = False
    try:
        with opener(destination, "xb") as output:
            owned = True
            if output.write(data) != len(data):
                raise OSError("short golden write")
    except BaseException:
        if owned:
            Path(destination).unlink(missing_ok=True)
        raise


def generate(args):
    root = args.qualia_root.resolve(strict=True)
    dependencies = args.dependencies.resolve(strict=True)
    destination = args.output.absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dense-original-", dir=destination.parent) as directory:
        scratch = Path(directory)
        source = scratch / "original"
        total = 0
        for path, expected in SOURCE_HASHES.items():
            data = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "show", f"{REVISION}:{path}"])
            total += len(data)
            if total > 2 * 1024 * 1024:
                raise ValueError("source budget exceeded")
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"source SHA256 differs: {path}")
            target = source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        def read(path):
            return (source / "crates/qualia-core-db/src" / path).read_text(encoding="utf8")
        helpers = "use crate::gguf_sharder::GgufTensorInfo;\nconst MAX_STACK_GEMM_IN:usize=10240;\npub(crate) const RMS_NORM_EPS:f32=1e-5;\nfn wlog(_: &str){panic!(\"out-of-scope oversized stack GEMM diagnostic\");}\n"
        for path, signature in [("gguf_bridge/cpu_ops.rs", "pub(crate) fn rms_norm_inplace"), ("gguf_bridge/cpu_ops.rs", "pub(crate) fn dequant_norm_row_into"), ("gguf_bridge/mod.rs", "fn rope_inplace"), ("gguf_bridge/mod.rs", "pub(crate) fn stack_gemm_quant")]:
            helpers += function(read(path), signature) + "\n"
        (scratch / "actual_helpers.rs").write_text(helpers, encoding="utf8")
        (scratch / "actual_fp8.rs").write_text(function(read("inference/moe/nvfp4.rs"), "pub fn fp8_e4m3_to_f32"), encoding="utf8")
        (scratch / "q_hash.rs").write_text(function(read("lib.rs"), "pub const fn q_hash"), encoding="utf8")
        model = read("gguf_bridge/wasm_cpu/model.rs")
        declarations = model[:model.index("impl CpuWasmEngine {")] + "impl CpuWasmEngine {" + function(model, "pub fn reset(&mut self)") + "}"
        (scratch / "model_decl.rs").write_text(declarations, encoding="utf8")
        text = TEMPLATE.replace("@ROOT@", source.as_posix()).replace("@SCRATCH@", scratch.as_posix()).replace("@FIXTURE@", Path(__file__).with_name("dense_cpu_fixture.rs").resolve().as_posix())
        (scratch / "generate.rs").write_text(text, encoding="utf8")
        executable = scratch / "generate.exe"
        command = ["rustup", "run", "1.96.1", "rustc", "--edition=2021", "-D", "warnings", str(scratch / "generate.rs"), "-L", f"dependency={dependencies}"]
        for name in ["serde", "serde_json", "bytemuck", "half", "memmap2"]:
            matches = list(dependencies.glob(f"lib{name}-*.rlib"))
            if not matches:
                raise ValueError(f"missing cached {name}")
            selected = max(matches, key=lambda path: path.stat().st_mtime_ns)
            command.extend(["--extern", f"{name}={selected}"])
        if args.optimized:
            command.extend(["-C", "opt-level=3"])
        subprocess.run([*command, "-o", str(executable)], check=True, timeout=180)
        if sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file()) > 32 * 1024 * 1024:
            raise ValueError("compiled scratch exceeds post-compilation retention check")
        result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)
        promote(result.stdout.encode("utf8"), destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify")
    check.add_argument("file", type=Path, nargs="?", default=Path(__file__).with_name("dense_cpu_golden.rs"))
    build = commands.add_parser("generate")
    build.add_argument("--qualia-root", type=Path, required=True)
    build.add_argument("--dependencies", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--optimized", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        verify_bytes(args.file.read_bytes())
        print(f"verified {GOLDEN_HASH}")
    else:
        generate(args)

if __name__ == "__main__":
    main()
