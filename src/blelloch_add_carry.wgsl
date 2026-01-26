const WG_SIZE: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sum: array<u32>;

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

@compute @workgroup_size(WG_SIZE)
fn add_carry(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let n_data = arrayLength(&global_data);
    let n_blocks = arrayLength(&block_sum);

    // Linear workgroup index is same as the index of the block sum
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    if (wg_linear >= n_blocks) {
        return;
    }

    let global_idx = wg_linear * WG_SIZE + lid.x;
    if (global_idx >= n_data) {
        return;
    }

    let carry = block_sum[wg_linear];
    global_data[global_idx] += carry;
}