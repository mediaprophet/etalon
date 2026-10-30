@group(0)
@binding(0)
var<storage, read_write> storage_buffer: array<vec2<u32>>;

var<workgroup> workgroup_buffer: u32;

fn add_result_to_mask(mask: ptr<function, vec2<u32>>, index: u32, value: bool) {
   (*mask)[index / 32u] |= u32(value) << (index % 32u);
}

@compute
@workgroup_size(128)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    var passed = vec2<u32>(0u);
    var expected: u32;

    add_result_to_mask(&passed, 0u, num_subgroups == 128u / subgroup_size);
    add_result_to_mask(&passed, 1u, subgroup_id == global_id.x / subgroup_size);
    add_result_to_mask(&passed, 2u, subgroup_invocation_id == global_id.x % subgroup_size);

    var expected_ballot = vec4<u32>(0u);
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected_ballot[i / 32u] |= ((global_id.x - subgroup_invocation_id + i) & 1u) << (i % 32u);
    }
    add_result_to_mask(&passed, 3u, dot(vec4<u32>(1u), vec4<u32>(subgroupBallot((subgroup_invocation_id & 1u) == 1u) == expected_ballot)) == 4u);

    add_result_to_mask(&passed, 4u, subgroupAll(true));
    add_result_to_mask(&passed, 5u, !subgroupAll(subgroup_invocation_id != 0u));

    add_result_to_mask(&passed, 6u, subgroupAny(subgroup_invocation_id == 0u));
    add_result_to_mask(&passed, 7u, !subgroupAny(false));

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 8u, subgroupAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 9u, subgroupMul(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected = max(expected, global_id.x - subgroup_invocation_id + i + 1u);
    }
    add_result_to_mask(&passed, 10u, subgroupMax(global_id.x + 1u) == expected);

    expected = 0xFFFFFFFFu;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected = min(expected, global_id.x - subgroup_invocation_id + i + 1u);
    }
    add_result_to_mask(&passed, 11u, subgroupMin(global_id.x + 1u) == expected);

    expected = 0xFFFFFFFFu;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected &= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 12u, subgroupAnd(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected |= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 13u, subgroupOr(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected ^= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 14u, subgroupXor(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_invocation_id; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 15u, subgroupExclusiveAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i < subgroup_invocation_id; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 16u, subgroupExclusiveMul(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i <= subgroup_invocation_id; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 17u, subgroupInclusiveAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i <= subgroup_invocation_id; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 18u, subgroupInclusiveMul(global_id.x + 1u) == expected);

    add_result_to_mask(&passed, 19u, subgroupBroadcastFirst(u32(subgroup_invocation_id != 0u)) == 0u);
    add_result_to_mask(&passed, 20u, subgroupBroadcastFirst(u32(subgroup_invocation_id == 0u)) == 1u);
    add_result_to_mask(&passed, 21u, subgroupBroadcast(subgroup_invocation_id, 1u) == 1u);
    add_result_to_mask(&passed, 22u, subgroupShuffle(subgroup_invocation_id, subgroup_invocation_id) == subgroup_invocation_id);
    add_result_to_mask(&passed, 23u, subgroupShuffle(subgroup_invocation_id, subgroup_size - 1u - subgroup_invocation_id) == subgroup_size - 1u - subgroup_invocation_id);
    add_result_to_mask(&passed, 24u, subgroupShuffleDown(subgroup_invocation_id, 1u) == subgroup_invocation_id + 1u || subgroup_invocation_id == subgroup_size - 1u);
    add_result_to_mask(&passed, 25u, subgroupShuffleUp(subgroup_invocation_id, 1u) == subgroup_invocation_id - 1u || subgroup_invocation_id == 0u);
    add_result_to_mask(&passed, 26u, subgroupShuffleXor(subgroup_invocation_id, subgroup_size - 1u) == (subgroup_invocation_id ^ (subgroup_size - 1u)));

    let even_count = subgroupAdd(u32(subgroup_invocation_id % 2u == 0u));
    let odd_count = subgroupAdd(u32(subgroup_invocation_id % 2u == 1u));
    add_result_to_mask(&passed, 27u, even_count == subgroup_size / 2u && odd_count == subgroup_size / 2u);

    let payload = vec4<u32>(subgroup_invocation_id, subgroup_invocation_id * 17u + 3u,
        subgroup_invocation_id ^ 0xa5a5a5a5u, subgroup_invocation_id * subgroup_invocation_id);
    let source = 2u;
    let expected_source = vec4<u32>(source, source * 17u + 3u, source ^ 0xa5a5a5a5u, source * source);
    let first_payload = subgroupBroadcastFirst(payload);
    let middle_payload = subgroupBroadcast(payload, 2u);
    add_result_to_mask(&passed, 28u, all(first_payload == vec4<u32>(0u, 3u, 0xa5a5a5a5u, 0u)) && all(middle_payload == expected_source));

    var passed_29 = true;
    for (var cutoff = 0u; cutoff <= subgroup_size; cutoff += 1u) {
        let count = subgroupAdd(u32(subgroup_invocation_id < cutoff));
        passed_29 &= count == cutoff;
    }
    add_result_to_mask(&passed, 29u, passed_29);

    if global_id.x == 0u {
        workgroup_buffer = subgroup_size;
    }
    workgroupBarrier();
    add_result_to_mask(&passed, 30u, workgroup_buffer == subgroup_size);

    add_result_to_mask(&passed, 31u, quadBroadcast(quadBroadcast(subgroup_invocation_id, 0u) ^ subgroup_invocation_id, 0u) == 0u);
    add_result_to_mask(&passed, 32u, quadBroadcast(quadBroadcast(subgroup_invocation_id, 1u) ^ quadSwapX(subgroup_invocation_id), 0u) == 0u);
    add_result_to_mask(&passed, 33u, quadBroadcast(quadBroadcast(subgroup_invocation_id, 2u) ^ quadSwapY(subgroup_invocation_id), 0u) == 0u);
    add_result_to_mask(&passed, 34u, quadBroadcast(quadBroadcast(subgroup_invocation_id, 3u) ^ quadSwapDiagonal(subgroup_invocation_id), 0u) == 0u);
    add_result_to_mask(&passed, 35u, quadSwapX(quadSwapY(subgroup_invocation_id)) == quadSwapDiagonal(subgroup_invocation_id));

    let scalar_payload = global_id.x * 1664525u + 1013904223u;
    add_result_to_mask(&passed, 37u, subgroupShuffle(scalar_payload, subgroup_invocation_id) == scalar_payload);
    add_result_to_mask(&passed, 38u, all(subgroupShuffle(payload, subgroup_invocation_id) == payload));
    let other_lane = subgroup_size - 1u - subgroup_invocation_id;
    let other_payload = vec4<u32>(other_lane, other_lane * 17u + 3u, other_lane ^ 0xa5a5a5a5u, other_lane * other_lane);
    add_result_to_mask(&passed, 39u, all(subgroupShuffle(payload, other_lane) == other_payload));

    // Keep this test last, verify we are still convergent after running other tests
    add_result_to_mask(&passed, 36u, subgroupAdd(1u) == subgroup_size);

    // Increment TEST_COUNT in subgroup_operations/mod.rs if adding more tests

    storage_buffer[global_id.x] = passed;
}
