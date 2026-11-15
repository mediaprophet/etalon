use crate::camera::Aabb;
use crate::mesh::TriangleMesh;
use crate::occupancy::OccupancyVolume;

const SDF_BRICK_MAGIC: &[u8; 8] = b"Q42SDF16";
const OCC1_MAGIC: &[u8; 4] = b"OCC1";

const GLTF_MAGIC: u32 = 0x4654_6C67;
const GLTF_VERSION: u32 = 2;
const GLTF_CHUNK_JSON: u32 = 0x4E4F_534A;
const GLTF_CHUNK_BIN: u32 = 0x004E_4942;

pub fn mesh_to_obj(mesh: &TriangleMesh, object_name: &str) -> String {
    let mut out = String::with_capacity(mesh.positions.len() * 64 + mesh.indices.len() * 24);
    out.push_str("# etalon-reconstruct Wavefront OBJ\n");
    out.push_str(&format!("o {}\n", sanitize(object_name)));
    for (i, p) in mesh.positions.iter().enumerate() {
        if let Some(c) = mesh.colors.get(i) {
            out.push_str(&format!(
                "v {:.6} {:.6} {:.6} {:.4} {:.4} {:.4}\n",
                p[0],
                p[1],
                p[2],
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0
            ));
        } else {
            out.push_str(&format!("v {:.6} {:.6} {:.6}\n", p[0], p[1], p[2]));
        }
    }
    for n in &mesh.normals {
        out.push_str(&format!("vn {:.6} {:.6} {:.6}\n", n[0], n[1], n[2]));
    }
    for tri in mesh.indices.chunks_exact(3) {
        let a = tri[0] + 1;
        let b = tri[1] + 1;
        let c = tri[2] + 1;
        out.push_str(&format!("f {a}//{a} {b}//{b} {c}//{c}\n"));
    }
    out
}

pub fn mesh_to_glb(mesh: &TriangleMesh, scene_id: &str) -> Vec<u8> {
    let name = sanitize(scene_id);
    let (pos_min, pos_max) = mesh.bounds();
    let mut norm_min = [f32::INFINITY; 3];
    let mut norm_max = [f32::NEG_INFINITY; 3];
    for n in &mesh.normals {
        for i in 0..3 {
            norm_min[i] = norm_min[i].min(n[i]);
            norm_max[i] = norm_max[i].max(n[i]);
        }
    }
    let v_cnt = mesh.positions.len();
    let use_u32 = v_cnt > 65535;
    let pos_byte_len = v_cnt * 12;
    let norm_byte_len = v_cnt * 12;
    let color_byte_len = v_cnt * 16;
    let idx_count = mesh.indices.len();
    let idx_stride = if use_u32 { 4 } else { 2 };
    let idx_byte_len = idx_count * idx_stride;

    let mut bin =
        Vec::with_capacity(pos_byte_len + norm_byte_len + color_byte_len + idx_byte_len + 4);
    for p in &mesh.positions {
        for &c in p {
            bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    for n in &mesh.normals {
        for &c in n {
            bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    for col in &mesh.colors {
        for k in 0..4 {
            let f = col[k] as f32 / 255.0;
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    if use_u32 {
        for &idx in &mesh.indices {
            bin.extend_from_slice(&idx.to_le_bytes());
        }
    } else {
        for &idx in &mesh.indices {
            bin.extend_from_slice(&(idx as u16).to_le_bytes());
        }
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }

    let color_off = pos_byte_len + norm_byte_len;
    let idx_off = color_off + color_byte_len;
    let idx_type = if use_u32 { 5125 } else { 5123 };
    let max_idx = v_cnt.saturating_sub(1);
    let json = format!(
        concat!(
            r#"{{"asset":{{"version":"2.0","generator":"etalon-reconstruct"}},"#,
            r#""scene":0,"scenes":[{{"name":"{id}","nodes":[0]}}],"#,
            r#""nodes":[{{"name":"{id}_node","mesh":0}}],"#,
            r#""meshes":[{{"name":"{id}","primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"COLOR_0":2}},"indices":3,"mode":4}}]}}],"#,
            r#""accessors":["#,
            r#"{{"bufferView":0,"componentType":5126,"count":{v},"type":"VEC3","min":[{p0:.6},{p1:.6},{p2:.6}],"max":[{P0:.6},{P1:.6},{P2:.6}]}},"#,
            r#"{{"bufferView":1,"componentType":5126,"count":{v},"type":"VEC3","min":[{n0:.6},{n1:.6},{n2:.6}],"max":[{N0:.6},{N1:.6},{N2:.6}]}},"#,
            r#"{{"bufferView":2,"componentType":5126,"count":{v},"type":"VEC4"}},"#,
            r#"{{"bufferView":3,"componentType":{it},"count":{ic},"type":"SCALAR","max":[{mi}]}}"#,
            r#"],"#,
            r#""bufferViews":["#,
            r#"{{"buffer":0,"byteOffset":0,"byteLength":{pl},"target":34962}},"#,
            r#"{{"buffer":0,"byteOffset":{no},"byteLength":{nl},"target":34962}},"#,
            r#"{{"buffer":0,"byteOffset":{co},"byteLength":{cl},"target":34962}},"#,
            r#"{{"buffer":0,"byteOffset":{io},"byteLength":{il},"target":34963}}"#,
            r#"],"#,
            r#""buffers":[{{"byteLength":{bl}}}]}}"#
        ),
        id = name,
        v = v_cnt,
        p0 = pos_min[0],
        p1 = pos_min[1],
        p2 = pos_min[2],
        P0 = pos_max[0],
        P1 = pos_max[1],
        P2 = pos_max[2],
        n0 = norm_min[0],
        n1 = norm_min[1],
        n2 = norm_min[2],
        N0 = norm_max[0],
        N1 = norm_max[1],
        N2 = norm_max[2],
        it = idx_type,
        ic = idx_count,
        mi = max_idx,
        pl = pos_byte_len,
        no = pos_byte_len,
        nl = norm_byte_len,
        co = color_off,
        cl = color_byte_len,
        io = idx_off,
        il = idx_byte_len,
        bl = bin.len(),
    );
    let mut json_chunk = json.into_bytes();
    while json_chunk.len() % 4 != 0 {
        json_chunk.push(b' ');
    }
    let total = 12 + 8 + json_chunk.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(total);
    glb.extend_from_slice(&GLTF_MAGIC.to_le_bytes());
    glb.extend_from_slice(&GLTF_VERSION.to_le_bytes());
    glb.extend_from_slice(&(total as u32).to_le_bytes());
    glb.extend_from_slice(&(json_chunk.len() as u32).to_le_bytes());
    glb.extend_from_slice(&GLTF_CHUNK_JSON.to_le_bytes());
    glb.extend_from_slice(&json_chunk);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&GLTF_CHUNK_BIN.to_le_bytes());
    glb.extend_from_slice(&bin);
    glb
}

/// Official Hypermedia `Q42SDF16` brick (same magic as `hypermedia.manifold.mesh_to_sdf_brick`).
pub fn volume_to_q42sdf16(volume: &OccupancyVolume) -> Vec<u8> {
    let n = volume.resolution as usize;
    let mut bytes = Vec::with_capacity(36 + n * n * n * 4);
    bytes.extend_from_slice(SDF_BRICK_MAGIC);
    bytes.extend_from_slice(&(volume.resolution).to_le_bytes());
    let origin = volume.bounds.min;
    let size = volume.bounds.size();
    let denom = (volume.resolution.saturating_sub(1).max(1)) as f32;
    let voxel = [size[0] / denom, size[1] / denom, size[2] / denom];
    for v in origin {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in voxel {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for &d in &volume.sdf {
        bytes.extend_from_slice(&d.to_le_bytes());
    }
    bytes
}

/// Occupancy sidecar: tissue labels for QualiaDB m64 / VOX0 sealing. Not a format fork of `.10d`.
pub fn volume_to_occ1(volume: &OccupancyVolume) -> Vec<u8> {
    let n = volume.resolution as usize;
    let mut bytes = Vec::with_capacity(36 + n * n * n);
    bytes.extend_from_slice(OCC1_MAGIC);
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&volume.resolution.to_le_bytes());
    for v in volume.bounds.min {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in volume.bounds.max {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&volume.material);
    bytes
}

pub fn occ1_from_bytes(bytes: &[u8]) -> Result<OccupancyVolume, String> {
    if bytes.len() < 36 || &bytes[0..4] != OCC1_MAGIC {
        return Err("OCC1 magic missing".into());
    }
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    if version != 1 {
        return Err(format!("OCC1 version {version} unsupported"));
    }
    let resolution = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if !(8..=256).contains(&resolution) {
        return Err("OCC1 resolution out of range".into());
    }
    let mut min = [0.0f32; 3];
    let mut max = [0.0f32; 3];
    for i in 0..3 {
        min[i] = f32::from_le_bytes(bytes[12 + i * 4..16 + i * 4].try_into().unwrap());
        max[i] = f32::from_le_bytes(bytes[24 + i * 4..28 + i * 4].try_into().unwrap());
    }
    let n = resolution as usize;
    let expected = 36 + n * n * n;
    if bytes.len() != expected {
        return Err(format!(
            "OCC1 length {} does not match resolution {} (need {expected})",
            bytes.len(),
            resolution
        ));
    }
    let material = bytes[36..].to_vec();
    let mut occ = vec![0u8; material.len()];
    for (i, &m) in material.iter().enumerate() {
        occ[i] = u8::from(m != 0);
    }
    let sdf = occupancy_sdf_from_labels(&occ, resolution);
    Ok(OccupancyVolume {
        resolution,
        bounds: Aabb::new(min, max),
        sdf,
        material,
    })
}

fn occupancy_sdf_from_labels(occ: &[u8], n: u32) -> Vec<f32> {
    let n_us = n as usize;
    let mut sdf = vec![0.0f32; occ.len()];
    let inf = 1.0e6f32;
    let mut dist = vec![inf; occ.len()];
    for i in 0..occ.len() {
        dist[i] = if occ[i] != 0 { 0.0 } else { inf };
    }
    chamfer3_export(&mut dist, n_us);
    for i in 0..occ.len() {
        if occ[i] == 0 {
            sdf[i] = dist[i].max(0.5);
        }
    }
    for i in 0..occ.len() {
        dist[i] = if occ[i] == 0 { 0.0 } else { inf };
    }
    chamfer3_export(&mut dist, n_us);
    for i in 0..occ.len() {
        if occ[i] != 0 {
            sdf[i] = -dist[i].max(0.5);
        }
    }
    sdf
}

fn chamfer3_export(d: &mut [f32], n: usize) {
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let i = x + n * (y + n * z);
                let mut best = d[i];
                if x > 0 {
                    best = best.min(d[i - 1] + 1.0);
                }
                if y > 0 {
                    best = best.min(d[i - n] + 1.0);
                }
                if z > 0 {
                    best = best.min(d[i - n * n] + 1.0);
                }
                d[i] = best;
            }
        }
    }
    for z in (0..n).rev() {
        for y in (0..n).rev() {
            for x in (0..n).rev() {
                let i = x + n * (y + n * z);
                let mut best = d[i];
                if x + 1 < n {
                    best = best.min(d[i + 1] + 1.0);
                }
                if y + 1 < n {
                    best = best.min(d[i + n] + 1.0);
                }
                if z + 1 < n {
                    best = best.min(d[i + n * n] + 1.0);
                }
                d[i] = best;
            }
        }
    }
}

fn sanitize(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "mesh".into()
    } else {
        out
    }
}
