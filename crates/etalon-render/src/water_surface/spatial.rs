//! Cold caller-buffered native BVH construction and allocation-free nearest
//! shoreline queries. Exact segment distances remain the acceptance criterion.

use bytemuck::{Pod, Zeroable};

use super::{ShoreError, ShoreReceipt, ShoreTopology};

const MIXED: u32 = u32::MAX;
const INVALID_INDEX: u32 = u32::MAX;
const LEAF_SIZE: usize = 4;

/// Node size in bytes.
pub const BVH_NODE_SIZE: usize = 40;

/// Maximum BVH depth (bounded by 2^21 primitives -> 21 levels).
pub const MAX_BVH_DEPTH: usize = 32;

/// 2D point in double precision for shoreline segment distance queries.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// 3D point in double precision for AABB and BVH queries.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Squared distance from point `p` to segment `ab` in 2D.
#[inline]
pub fn point_segment_distance_sq_2d(p: Point2, a: Point2, b: Point2) -> f64 {
    let abx = b.x - a.x;
    let aby = b.y - a.y;
    let apx = p.x - a.x;
    let apy = p.y - a.y;
    let ab_sq = abx * abx + aby * aby;
    if ab_sq == 0.0 {
        return apx * apx + apy * apy;
    }
    let t = ((apx * abx + apy * aby) / ab_sq).clamp(0.0, 1.0);
    let cx = a.x + t * abx;
    let cy = a.y + t * aby;
    let dx = p.x - cx;
    let dy = p.y - cy;
    dx * dx + dy * dy
}

/// Axis-aligned bounding box in 3D.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Point3,
    pub max: Point3,
}

impl Aabb {
    #[inline]
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    #[inline]
    pub fn contains_point(&self, p: Point3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Squared distance from point `p` to this AABB (0 if inside).
    #[inline]
    pub fn distance_sq_to_point(&self, p: Point3) -> f64 {
        let dx = (self.min.x - p.x).max(0.0).max(p.x - self.max.x);
        let dy = (self.min.y - p.y).max(0.0).max(p.y - self.max.y);
        let dz = (self.min.z - p.z).max(0.0).max(p.z - self.max.z);
        dx * dx + dy * dy + dz * dz
    }

    #[inline]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb::new(
            Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        )
    }
}

/// Errors raised by spatial ordering functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialOrderError {
    CodeBufferTooSmall { required: usize },
    IndexBufferTooSmall { required: usize },
    NonFiniteCoordinate { index: usize },
}

/// Errors raised by the BVH builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvhError {
    NodeBufferTooSmall { required: usize },
    IndexBufferTooSmall { required: usize },
    CodeBufferTooSmall { required: usize },
    SortBufferTooSmall { required: usize },
    InvalidAabb { index: usize },
}

impl From<SpatialOrderError> for BvhError {
    fn from(err: SpatialOrderError) -> Self {
        match err {
            SpatialOrderError::CodeBufferTooSmall { required } => {
                BvhError::CodeBufferTooSmall { required }
            }
            SpatialOrderError::IndexBufferTooSmall { required } => {
                BvhError::SortBufferTooSmall { required }
            }
            SpatialOrderError::NonFiniteCoordinate { index } => BvhError::InvalidAabb { index },
        }
    }
}

/// BVH node: 40 bytes, `repr(C)`, naturally aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct BvhNode {
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
    pub left_or_first: u32,
    pub right_or_count: u32,
    pub parent: u32,
    pub node_type: u32,
}

impl Default for BvhNode {
    fn default() -> Self {
        Self {
            bbox_min: [0.0; 3],
            bbox_max: [0.0; 3],
            left_or_first: 0,
            right_or_count: 0,
            parent: INVALID_INDEX,
            node_type: 0,
        }
    }
}

#[inline]
fn spread2_16(x: u16) -> u64 {
    let mut x = x as u64;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x
}

#[inline]
fn morton_encode_3d(x: u16, y: u16, z: u16) -> u64 {
    spread2_16(x) | (spread2_16(y) << 1) | (spread2_16(z) << 2)
}

#[inline]
fn quantize_axis(v: f64, min: f64, extent: f64) -> u16 {
    if extent <= 0.0 {
        return 0;
    }
    let n = ((v - min) / extent).clamp(0.0, 1.0);
    (n * 65535.0 + 0.5) as u16
}

#[inline]
fn bbox_3d(points: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in points {
        for a in 0..3 {
            if p[a] < min[a] {
                min[a] = p[a];
            }
            if p[a] > max[a] {
                max[a] = p[a];
            }
        }
    }
    if points.is_empty() {
        min = [0.0; 3];
        max = [0.0; 3];
    }
    (min, max)
}

/// Compute Morton codes for a 3D point set and sort indices by code.
pub fn sort_by_morton_3d(
    points: &[[f64; 3]],
    codes: &mut [u64],
    indices: &mut [u32],
) -> Result<usize, SpatialOrderError> {
    let n = points.len();
    if codes.len() < n {
        return Err(SpatialOrderError::CodeBufferTooSmall { required: n });
    }
    if indices.len() < n {
        return Err(SpatialOrderError::IndexBufferTooSmall { required: n });
    }

    let (min, max) = bbox_3d(points);
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];

    for (i, p) in points.iter().enumerate() {
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            return Err(SpatialOrderError::NonFiniteCoordinate { index: i });
        }
        let qx = quantize_axis(p[0], min[0], extent[0]);
        let qy = quantize_axis(p[1], min[1], extent[1]);
        let qz = quantize_axis(p[2], min[2], extent[2]);
        codes[i] = morton_encode_3d(qx, qy, qz);
        indices[i] = i as u32;
    }

    indices[..n].sort_unstable_by_key(|&idx| codes[idx as usize]);
    Ok(n)
}

fn build_recursive(
    primitives: &[Aabb],
    prim_indices: &mut [u32],
    start: usize,
    count: usize,
    nodes: &mut [BvhNode],
    next_node: &mut usize,
    parent: u32,
) -> usize {
    let node_idx = *next_node;
    *next_node += 1;

    let mut bbox = Aabb::new(
        primitives[prim_indices[start] as usize].min,
        primitives[prim_indices[start] as usize].max,
    );
    for i in 1..count {
        let p_idx = prim_indices[start + i] as usize;
        bbox = bbox.union(&primitives[p_idx]);
    }

    if count <= LEAF_SIZE {
        nodes[node_idx] = BvhNode {
            bbox_min: [bbox.min.x as f32, bbox.min.y as f32, bbox.min.z as f32],
            bbox_max: [bbox.max.x as f32, bbox.max.y as f32, bbox.max.z as f32],
            left_or_first: start as u32,
            right_or_count: count as u32,
            parent,
            node_type: 1,
        };
        return 1;
    }

    let mid = start + count / 2;

    let left_child = *next_node;
    let left_size = build_recursive(
        primitives,
        prim_indices,
        start,
        count / 2,
        nodes,
        next_node,
        node_idx as u32,
    );

    let right_child = *next_node;
    let right_size = build_recursive(
        primitives,
        prim_indices,
        mid,
        count - count / 2,
        nodes,
        next_node,
        node_idx as u32,
    );

    nodes[node_idx] = BvhNode {
        bbox_min: [bbox.min.x as f32, bbox.min.y as f32, bbox.min.z as f32],
        bbox_max: [bbox.max.x as f32, bbox.max.y as f32, bbox.max.z as f32],
        left_or_first: left_child as u32,
        right_or_count: right_child as u32,
        parent,
        node_type: 0,
    };

    1 + left_size + right_size
}

/// Build a BVH using the deterministic Morton-sorted recursive builder.
pub fn build_bvh_recursive(
    primitives: &[Aabb],
    nodes: &mut [BvhNode],
    prim_indices: &mut [u32],
    morton_codes: &mut [u64],
    sort_indices: &mut [u32],
) -> Result<(usize, usize), BvhError> {
    let n = primitives.len();
    if n == 0 {
        return Ok((0, 0));
    }

    for (i, aabb) in primitives.iter().enumerate() {
        if aabb.min.x > aabb.max.x || aabb.min.y > aabb.max.y || aabb.min.z > aabb.max.z {
            return Err(BvhError::InvalidAabb { index: i });
        }
    }

    if morton_codes.len() < n {
        return Err(BvhError::CodeBufferTooSmall { required: n });
    }
    if sort_indices.len() < n {
        return Err(BvhError::SortBufferTooSmall { required: n });
    }
    if prim_indices.len() < n {
        return Err(BvhError::IndexBufferTooSmall { required: n });
    }

    let max_nodes = 2 * n - 1;
    if nodes.len() < max_nodes {
        return Err(BvhError::NodeBufferTooSmall {
            required: max_nodes,
        });
    }

    let centroids: Vec<[f64; 3]> = primitives
        .iter()
        .map(|aabb| {
            [
                0.5 * (aabb.min.x + aabb.max.x),
                0.5 * (aabb.min.y + aabb.max.y),
                0.5 * (aabb.min.z + aabb.max.z),
            ]
        })
        .collect();

    sort_by_morton_3d(&centroids, morton_codes, sort_indices)?;
    prim_indices[..n].copy_from_slice(&sort_indices[..n]);

    let mut next_node = 0usize;
    let node_count = build_recursive(
        primitives,
        prim_indices,
        0,
        n,
        nodes,
        &mut next_node,
        INVALID_INDEX,
    );

    Ok((node_count, 0))
}

pub struct ShoreSpatialScratch<'a> {
    pub bounds: &'a mut [Aabb],
    pub nodes: &'a mut [BvhNode],
    pub primitive_order: &'a mut [u32],
    pub morton: &'a mut [u64],
    pub sort_order: &'a mut [u32],
    pub node_components: &'a mut [u32],
}

/// Includes caller arrays and the native BVH builder's bounded centroid scratch.
pub fn spatial_working_bytes(edges: usize) -> Option<usize> {
    let nodes = edges.checked_mul(2)?.checked_sub(1)?;
    edges
        .checked_mul(std::mem::size_of::<Aabb>() + 4 + 8 + 4 + 24)?
        .checked_add(nodes.checked_mul(std::mem::size_of::<BvhNode>() + 4)?)
}

fn bbox(node: &BvhNode) -> Aabb {
    Aabb::new(
        Point3::new(
            node.bbox_min[0] as f64,
            node.bbox_min[1] as f64,
            node.bbox_min[2] as f64,
        ),
        Point3::new(
            node.bbox_max[0] as f64,
            node.bbox_max[1] as f64,
            node.bbox_max[2] as f64,
        ),
    )
}

fn nearest(
    positions: &[[f32; 3]],
    topology: &ShoreTopology<'_>,
    nodes: &[BvhNode],
    order: &[u32],
    components: &[u32],
    point: Point2,
    component: u32,
) -> Result<(f64, u64), ShoreError> {
    let query = Point3::new(point.x, 0.0, point.y);
    let mut best = f64::INFINITY;
    let mut evaluated = 0;
    let mut stack = [0u32; MAX_BVH_DEPTH * 2];
    let mut pending = 1;
    while pending > 0 {
        pending -= 1;
        let index = stack[pending] as usize;
        let node = &nodes[index];
        if components[index] != MIXED && components[index] != component {
            continue;
        }
        if bbox(node).distance_sq_to_point(query) > best {
            continue;
        }
        if node.node_type == 1 {
            for primitive in &order
                [node.left_or_first as usize..(node.left_or_first + node.right_or_count) as usize]
            {
                let edge = topology.edges[*primitive as usize];
                let a = (edge >> 32) as usize;
                let b = edge as u32 as usize;
                if topology.component(a) != component {
                    continue;
                }
                let p = positions[a];
                let q = positions[b];
                evaluated += 1;
                best = best.min(point_segment_distance_sq_2d(
                    point,
                    Point2::new(p[0] as f64, p[2] as f64),
                    Point2::new(q[0] as f64, q[2] as f64),
                ));
            }
        } else {
            if pending + 2 > stack.len() {
                return Err(ShoreError::SpatialIndex);
            }
            let a = node.left_or_first;
            let b = node.right_or_count;
            let ad = bbox(&nodes[a as usize]).distance_sq_to_point(query);
            let bd = bbox(&nodes[b as usize]).distance_sq_to_point(query);
            let (near, far) = if ad <= bd { (a, b) } else { (b, a) };
            stack[pending] = far;
            stack[pending + 1] = near;
            pending += 2;
        }
    }
    if !best.is_finite() {
        return Err(ShoreError::SpatialIndex);
    }
    Ok((best.sqrt(), evaluated))
}

/// Accelerated shore-distance field into `out`.
pub fn shore_distance_accelerated_into(
    positions: &[[f32; 3]],
    topology: &ShoreTopology<'_>,
    distances: &mut [f32],
    scales: &mut [f32],
    scratch: ShoreSpatialScratch<'_>,
) -> Result<ShoreReceipt, ShoreError> {
    let n = topology.edges.len();
    let node_capacity = n
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .ok_or(ShoreError::InvalidInput)?;
    if positions.len() != topology.vertices()
        || positions
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 1e8)
    {
        return Err(ShoreError::InvalidInput);
    }
    if distances.len() < positions.len() || scales.len() < positions.len() {
        return Err(ShoreError::OutputTooSmall);
    }
    if scratch.bounds.len() < n
        || scratch.nodes.len() < node_capacity
        || scratch.primitive_order.len() < n
        || scratch.morton.len() < n
        || scratch.sort_order.len() < n
        || scratch.node_components.len() < node_capacity
    {
        return Err(ShoreError::ScratchTooSmall);
    }
    for (index, edge) in topology.edges.iter().enumerate() {
        let a = positions[(*edge >> 32) as usize];
        let b = positions[*edge as u32 as usize];
        scratch.bounds[index] = Aabb::new(
            Point3::new(a[0].min(b[0]) as f64, 0.0, a[2].min(b[2]) as f64),
            Point3::new(a[0].max(b[0]) as f64, 0.0, a[2].max(b[2]) as f64),
        );
    }
    let (node_count, _) = build_bvh_recursive(
        &scratch.bounds[..n],
        scratch.nodes,
        scratch.primitive_order,
        scratch.morton,
        scratch.sort_order,
    )
    .map_err(|_| ShoreError::SpatialIndex)?;
    for index in (0..node_count).rev() {
        let node = &scratch.nodes[index];
        scratch.node_components[index] = if node.node_type == 1 {
            let mut component = MIXED;
            for (i, primitive) in scratch.primitive_order
                [node.left_or_first as usize..(node.left_or_first + node.right_or_count) as usize]
                .iter()
                .enumerate()
            {
                let candidate =
                    topology.component((topology.edges[*primitive as usize] >> 32) as usize);
                if i == 0 {
                    component = candidate;
                } else if component != candidate {
                    component = MIXED;
                    break;
                }
            }
            component
        } else {
            let a = scratch.node_components[node.left_or_first as usize];
            let b = scratch.node_components[node.right_or_count as usize];
            if a == b {
                a
            } else {
                MIXED
            }
        };
    }
    scales[..positions.len()].fill(0.0);
    let mut max_distance = 0.0_f32;
    let mut segment_evaluations = 0;
    for index in 0..positions.len() {
        if !topology.referenced(index) {
            distances[index] = 0.0;
            continue;
        }
        let p = positions[index];
        let component = topology.component(index);
        let (distance, evaluations) = nearest(
            positions,
            topology,
            &scratch.nodes[..node_count],
            scratch.primitive_order,
            &scratch.node_components[..node_count],
            Point2::new(p[0] as f64, p[2] as f64),
            component,
        )?;
        let distance = distance as f32;
        distances[index] = distance;
        segment_evaluations += evaluations;
        scales[component as usize] = scales[component as usize].max(distance);
        max_distance = max_distance.max(distance);
    }
    for index in 0..positions.len() {
        scales[index] = scales[topology.component(index) as usize];
    }
    Ok(ShoreReceipt {
        boundary_edges: n,
        max_distance,
        segment_evaluations,
    })
}
