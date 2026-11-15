use crate::camera::Aabb;
use crate::error::ReconstructError;
use crate::mesh::TriangleMesh;

const NULL_VERTEX: u32 = u32::MAX;

const CUBE_CORNERS: [[u32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

const CUBE_CORNER_VECTORS: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];

const CUBE_EDGES: [[u32; 2]; 12] = [
    [0b000, 0b001],
    [0b000, 0b010],
    [0b000, 0b100],
    [0b001, 0b011],
    [0b001, 0b101],
    [0b010, 0b011],
    [0b010, 0b110],
    [0b011, 0b111],
    [0b100, 0b101],
    [0b100, 0b110],
    [0b101, 0b111],
    [0b110, 0b111],
];

pub fn surface_nets(sdf: &[f32], n: u32, bounds: Aabb) -> Result<TriangleMesh, ReconstructError> {
    let n_us = n as usize;
    if sdf.len() != n_us * n_us * n_us {
        return Err(ReconstructError::InvalidResolution);
    }
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut stride_to_index = vec![NULL_VERTEX; sdf.len()];
    let mut surface_points: Vec<[u32; 3]> = Vec::new();

    for z in 0..n.saturating_sub(1) {
        for y in 0..n.saturating_sub(1) {
            for x in 0..n.saturating_sub(1) {
                let stride = linearize(x, y, z, n);
                if let Some((centroid, grad)) = estimate_cube(sdf, n, x, y, z) {
                    stride_to_index[stride] = positions.len() as u32;
                    positions.push(centroid);
                    normals.push(grad);
                    surface_points.push([x, y, z]);
                }
            }
        }
    }

    let mut indices = Vec::new();
    let xs = 1usize;
    let ys = n as usize;
    let zs = (n as usize) * (n as usize);
    for &[x, y, z] in &surface_points {
        let p = linearize(x, y, z, n);
        if y != 0 && z != 0 && x + 1 < n {
            maybe_make_quad(
                sdf,
                &stride_to_index,
                &positions,
                p,
                p + xs,
                ys,
                zs,
                &mut indices,
            );
        }
        if x != 0 && z != 0 && y + 1 < n {
            maybe_make_quad(
                sdf,
                &stride_to_index,
                &positions,
                p,
                p + ys,
                zs,
                xs,
                &mut indices,
            );
        }
        if x != 0 && y != 0 && z + 1 < n {
            maybe_make_quad(
                sdf,
                &stride_to_index,
                &positions,
                p,
                p + zs,
                xs,
                ys,
                &mut indices,
            );
        }
    }

    if positions.is_empty() || indices.is_empty() {
        return Err(ReconstructError::EmptyMesh);
    }

    let size = bounds.size();
    let inv = 1.0 / n.saturating_sub(1).max(1) as f32;
    for p in &mut positions {
        p[0] = bounds.min[0] + p[0] * inv * size[0];
        p[1] = bounds.min[1] + p[1] * inv * size[1];
        p[2] = bounds.min[2] + p[2] * inv * size[2];
    }
    for nrm in &mut normals {
        *nrm = crate::camera::normalize(*nrm);
    }

    let colors = vec![[180u8, 70, 70, 255]; positions.len()];
    Ok(TriangleMesh {
        positions,
        normals,
        colors,
        indices,
    })
}

fn linearize(x: u32, y: u32, z: u32, n: u32) -> usize {
    (x + n * (y + n * z)) as usize
}

fn estimate_cube(sdf: &[f32], n: u32, x: u32, y: u32, z: u32) -> Option<([f32; 3], [f32; 3])> {
    let mut dists = [0.0f32; 8];
    let mut negative = 0u32;
    for (i, corner) in CUBE_CORNERS.iter().enumerate() {
        let d = sdf[linearize(x + corner[0], y + corner[1], z + corner[2], n)];
        dists[i] = d;
        if d < 0.0 {
            negative += 1;
        }
    }
    if negative == 0 || negative == 8 {
        return None;
    }
    let c = centroid_of_edge_intersections(&dists);
    let g = sdf_gradient(&dists, c);
    Some(([x as f32 + c[0], y as f32 + c[1], z as f32 + c[2]], g))
}

fn centroid_of_edge_intersections(dists: &[f32; 8]) -> [f32; 3] {
    let mut count = 0.0f32;
    let mut sum = [0.0f32; 3];
    for &[c1, c2] in &CUBE_EDGES {
        let d1 = dists[c1 as usize];
        let d2 = dists[c2 as usize];
        if (d1 < 0.0) != (d2 < 0.0) {
            count += 1.0;
            let p = estimate_edge(c1, c2, d1, d2);
            sum[0] += p[0];
            sum[1] += p[1];
            sum[2] += p[2];
        }
    }
    if count < 1.0 {
        [0.5, 0.5, 0.5]
    } else {
        [sum[0] / count, sum[1] / count, sum[2] / count]
    }
}

fn estimate_edge(c1: u32, c2: u32, v1: f32, v2: f32) -> [f32; 3] {
    let denom = v1 - v2;
    let t = if denom.abs() < 1.0e-8 {
        0.5
    } else {
        v1 / denom
    };
    let a = CUBE_CORNER_VECTORS[c1 as usize];
    let b = CUBE_CORNER_VECTORS[c2 as usize];
    [
        (1.0 - t) * a[0] + t * b[0],
        (1.0 - t) * a[1] + t * b[1],
        (1.0 - t) * a[2] + t * b[2],
    ]
}

fn sdf_gradient(dists: &[f32; 8], s: [f32; 3]) -> [f32; 3] {
    let p00 = [dists[0b001], dists[0b010], dists[0b100]];
    let n00 = [dists[0b000], dists[0b000], dists[0b000]];
    let p10 = [dists[0b101], dists[0b011], dists[0b110]];
    let n10 = [dists[0b100], dists[0b001], dists[0b010]];
    let p01 = [dists[0b011], dists[0b110], dists[0b101]];
    let n01 = [dists[0b010], dists[0b100], dists[0b001]];
    let p11 = [dists[0b111], dists[0b111], dists[0b111]];
    let n11 = [dists[0b110], dists[0b101], dists[0b011]];
    let d00 = sub3(p00, n00);
    let d10 = sub3(p10, n10);
    let d01 = sub3(p01, n01);
    let d11 = sub3(p11, n11);
    let neg = [1.0 - s[0], 1.0 - s[1], 1.0 - s[2]];
    let yzx_n = [neg[1], neg[2], neg[0]];
    let zxy_n = [neg[2], neg[0], neg[1]];
    let yzx_s = [s[1], s[2], s[0]];
    let zxy_s = [s[2], s[0], s[1]];
    add3(
        add3(mul3(mul3(yzx_n, zxy_n), d00), mul3(mul3(yzx_n, zxy_s), d10)),
        add3(mul3(mul3(yzx_s, zxy_n), d01), mul3(mul3(yzx_s, zxy_s), d11)),
    )
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn maybe_make_quad(
    sdf: &[f32],
    stride_to_index: &[u32],
    positions: &[[f32; 3]],
    p1: usize,
    p2: usize,
    axis_b: usize,
    axis_c: usize,
    indices: &mut Vec<u32>,
) {
    if p1 < axis_b || p1 < axis_c || p1 < axis_b + axis_c {
        return;
    }
    let d1 = sdf[p1];
    let d2 = sdf[p2];
    let negative_face = match (d1 < 0.0, d2 < 0.0) {
        (true, false) => false,
        (false, true) => true,
        _ => return,
    };
    let v1 = stride_to_index[p1];
    let v2 = stride_to_index[p1 - axis_b];
    let v3 = stride_to_index[p1 - axis_c];
    let v4 = stride_to_index[p1 - axis_b - axis_c];
    if v1 == NULL_VERTEX || v2 == NULL_VERTEX || v3 == NULL_VERTEX || v4 == NULL_VERTEX {
        return;
    }
    let pos1 = positions[v1 as usize];
    let pos2 = positions[v2 as usize];
    let pos3 = positions[v3 as usize];
    let pos4 = positions[v4 as usize];
    let d14 = dist2(pos1, pos4);
    let d23 = dist2(pos2, pos3);
    let quad = if d14 < d23 {
        if negative_face {
            [v1, v4, v2, v1, v3, v4]
        } else {
            [v1, v2, v4, v1, v4, v3]
        }
    } else if negative_face {
        [v2, v3, v4, v2, v1, v3]
    } else {
        [v2, v4, v3, v2, v3, v1]
    };
    indices.extend_from_slice(&quad);
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}
