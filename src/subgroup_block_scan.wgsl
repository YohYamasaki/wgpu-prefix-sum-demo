const WG_SIZE: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sum: array<u32>;

// For subgroup sum + offsets
var<workgroup> local_data: array<u32, 128u>;

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

@compute @workgroup_size(WG_SIZE)
fn block_scan_write_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32, // maybe 32 or 64, depends on the GPU
    @builtin(subgroup_invocation_id) sg_lane: u32, // 0..sg_size, most probably 0..32
    @builtin(subgroup_id) sg_id: u32, // 0..workgroup_size/subgroup_size, most probably 0..4
) {
    let n = arrayLength(&global_data);

    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let global_idx = wg_linear * WG_SIZE + lid.x;
    let in_range = global_idx < n;
    var v = 0u;
    if (in_range) {
        v = global_data[global_idx];
    }

    // exclusive scan result in the same subgroup until this element
    let sg_prefix = subgroupExclusiveAdd(v);
    // calculate the sum of all elements in the subgroup.
    // The same result will be returned for the same subgroup, no matter which lane we are in.
    let sg_sum = subgroupAdd(v);
    // Store the sum from each subgroup into workgroup shared
    if (sg_lane == 0u) {
        local_data[sg_id] = sg_sum;
    }
    workgroupBarrier();

    // Build offsets to collect the each subgroup's scan result
    let num_sg = (WG_SIZE + sg_size - 1u) / sg_size;
    if (lid.x == 0u) {
        // run exclusive scan on the subgroup sum results array
        var sg_sum_total = 0u;
        for (var i = 0u; i < num_sg; i = i + 1u) {
            let tmp = local_data[i];
            local_data[i] = sg_sum_total;
            sg_sum_total = sg_sum_total + tmp;
        }
        // store the block sum for the next block scan
        let n_blocks = arrayLength(&block_sum);
        if (wg_linear < n_blocks) {
            block_sum[wg_linear] = sg_sum_total;
        }
    }
    workgroupBarrier();

    // Add carry from each subgroups to the subgroup prefix
    if (in_range) {
        global_data[global_idx] = local_data[sg_id] + sg_prefix;
    }
}

@compute @workgroup_size(WG_SIZE)
fn block_scan_no_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,

    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_id) sg_id: u32,
) {
    let n = arrayLength(&global_data);

    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let global_idx = wg_linear * WG_SIZE + lid.x;
    let in_range = global_idx < n;
    var v = 0u;
    if (in_range) {
        v = global_data[global_idx];
    }

    let sg_prefix = subgroupExclusiveAdd(v);
    let sg_sum    = subgroupAdd(v);

    if (sg_lane == 0u) {
        local_data[sg_id] = sg_sum;
    }
    workgroupBarrier();

    let num_sg = (WG_SIZE + sg_size - 1u) / sg_size;
    if (lid.x == 0u) {
        var run = 0u;
        for (var i = 0u; i < num_sg; i = i + 1u) {
            let tmp = local_data[i];
            local_data[i] = run;
            run = run + tmp;
        }
    }
    workgroupBarrier();

    let carry = local_data[sg_id];
    if (in_range) {
        global_data[global_idx] = carry + sg_prefix;
    }
}