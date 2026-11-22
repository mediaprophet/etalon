struct Params {
    vertex_count: u32,
    face_count: u32,
    edge_count: u32,
    face_capacity: u32,
    edge_capacity: u32,
    row_capacity: u32,
    level: u32,
    margin_m: f32,
}

struct VertexMotion {
    before: vec4<f32>,
    after: vec4<f32>,
}

struct Bounds {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

@group(0) @binding(0) var<uniform> cfg: Params;
@group(0) @binding(1) var<storage, read> positions: array<VertexMotion>;
@group(0) @binding(2) var<storage, read> topology: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> leaf_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> nodes: array<Bounds>;
@group(0) @binding(5) var<storage, read_write> rows: array<u32>;

fn empty_bounds() -> Bounds {
    return Bounds(vec4<f32>(1e30), vec4<f32>(-1e30));
}

fn vertex_bounds(index: u32) -> Bounds {
    let p = positions[index];
    return Bounds(min(p.before, p.after), max(p.before, p.after));
}

fn merge_bounds(a: Bounds, b: Bounds) -> Bounds {
    return Bounds(min(a.lo, b.lo), max(a.hi, b.hi));
}

fn expand_bounds(motion: Bounds) -> Bounds {
    // Outward padding also covers rounding in large-coordinate f32 addition.
    let magnitude = max(vec3<f32>(1.0), max(abs(motion.lo.xyz), abs(motion.hi.xyz)));
    let inflation = vec3<f32>(cfg.margin_m) + magnitude * 2.38418579e-7;
    return Bounds(vec4<f32>(motion.lo.xyz - inflation, 0.0), vec4<f32>(motion.hi.xyz + inflation, 0.0));
}

fn overlaps(a: Bounds, b: Bounds) -> bool {
    return all(a.lo.xyz <= b.hi.xyz) && all(b.lo.xyz <= a.hi.xyz);
}

@compute @workgroup_size(64)
fn refit_leaves(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let lane = invocation.x;
    if (lane < cfg.face_capacity) {
        var bounds = empty_bounds();
        if (lane < cfg.face_count) {
            let face = topology[leaf_ids[lane]];
            bounds = merge_bounds(vertex_bounds(face.x), vertex_bounds(face.y));
            bounds = merge_bounds(bounds, vertex_bounds(face.z));
        }
        nodes[cfg.face_capacity + lane] = bounds;
    } else if (lane < cfg.face_capacity + cfg.edge_capacity) {
        let slot = lane - cfg.face_capacity;
        var bounds = empty_bounds();
        if (slot < cfg.edge_count) {
            let edge = topology[cfg.face_count + leaf_ids[cfg.face_count + slot]];
            bounds = merge_bounds(vertex_bounds(edge.x), vertex_bounds(edge.y));
        }
        nodes[2u * cfg.face_capacity + cfg.edge_capacity + slot] = bounds;
    }
}

@compute @workgroup_size(64)
fn refit_level(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let lane = invocation.x;
    let face_nodes = cfg.face_capacity >> cfg.level;
    let edge_nodes = cfg.edge_capacity >> cfg.level;
    if (lane < face_nodes) {
        let index = face_nodes + lane;
        let left = 2u * index;
        nodes[index] = merge_bounds(nodes[left], nodes[left + 1u]);
    } else if (lane < face_nodes + edge_nodes) {
        let offset = 2u * cfg.face_capacity;
        let index = edge_nodes + (lane - face_nodes);
        let left = offset + 2u * index;
        nodes[offset + index] = merge_bounds(nodes[left], nodes[left + 1u]);
    }
}

fn after_subtree(index: u32) -> u32 {
    var cursor = index;
    while ((cursor & 1u) == 1u) {
        cursor = cursor >> 1u;
    }
    return select(cursor + 1u, 0u, cursor == 0u);
}

@compute @workgroup_size(64)
fn collect_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let query = invocation.x;
    if (query >= cfg.vertex_count + cfg.edge_count) {
        return;
    }

    let row = query * (cfg.row_capacity + 1u);
    var source = query;
    var subject = vec2<u32>(0u);
    var query_bounds: Bounds;
    var capacity = cfg.face_capacity;
    var primitive_count = cfg.face_count;
    var tree_offset = 0u;
    var source_offset = 0u;
    let edge_query = query >= cfg.vertex_count;
    if edge_query {
        source = query - cfg.vertex_count;
        capacity = cfg.edge_capacity;
        primitive_count = cfg.edge_count;
        tree_offset = 2u * cfg.face_capacity;
        source_offset = cfg.face_count;
        subject = topology[source_offset + source].xy;
        query_bounds = merge_bounds(vertex_bounds(subject.x), vertex_bounds(subject.y));
    } else {
        query_bounds = vertex_bounds(source);
    }
    query_bounds = expand_bounds(query_bounds);

    var accepted = 0u;
    var index = 1u;
    while (index != 0u) {
        if overlaps(query_bounds, nodes[tree_offset + index]) {
            if (index < capacity) {
                index = index * 2u;
                continue;
            }
            let slot = index - capacity;
            if (slot < primitive_count) {
                let target = leaf_ids[source_offset + slot];
                let incident = topology[source_offset + target];
                var valid = source != incident.x && source != incident.y && source != incident.z;
                if edge_query {
                    valid = target > source && all(subject != vec2<u32>(incident.x)) && all(subject != vec2<u32>(incident.y));
                }
                if valid {
                    if (accepted == cfg.row_capacity) {
                        rows[row] = cfg.row_capacity + 1u;
                        return;
                    }
                    accepted += 1u;
                    rows[row + accepted] = target;
                }
            }
        }
        index = after_subtree(index);
    }
    rows[row] = accepted;
}
