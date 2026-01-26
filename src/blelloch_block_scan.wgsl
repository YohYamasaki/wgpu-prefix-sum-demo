const WG_SIZE: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sum: array<u32>;

var<workgroup> local_data: array<u32, 64u>;

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

/**
 * Get local and global index.
 */
fn get_indices(lid: vec3<u32>, wid: vec3<u32>, num_wg: vec3<u32>) -> array<u32, 2> {
    let local_idx = lid.x;
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let block_base = wg_linear * WG_SIZE;
    let global_idx = block_base + local_idx;
    return array<u32, 2>(local_idx, global_idx);
}

/**
 * Load data from the storage to the workgroup variable.
 */
fn copy_global_data_to_local(n: u32, local_idx: u32, global_idx: u32) {
    var global_val = 0u;
    if (global_idx < n) {
        global_val = global_data[global_idx];
    }
    local_data[local_idx] = global_val;
    workgroupBarrier();
}

/**
 * Execute up-sweep step of the Blelloch scan. Returns sum of the local block.
 */
fn up_sweep(local_idx: u32) {
    var step = 2u;
    while (step <= WG_SIZE) {
        let num_targets = WG_SIZE / step;
        if (local_idx < num_targets) {
            // Map each participating thread t to the rightmost element of its span.
            // This avoids an expensive modulo/division check +
            // makes active lanes contiguous (t < num_targets), which (probably) reduces
            // intra-warp branch divergence compared to a strided predicate.
            let target_idx = (local_idx + 1u) * step - 1u;
            // target_idx - (step >> 1u) -> index of the sum target (step/2 back)
            local_data[target_idx] += local_data[target_idx - (step >> 1u)];
        }
        workgroupBarrier();
        step = step << 1u;
    }
}

/**
 * Execute down-sweep step of the Blelloch scan.
 */
fn down_sweep(local_idx: u32) {
    var step = WG_SIZE;
    while (step >= 2u) {
     let num_targets = WG_SIZE / step;
     if (local_idx < num_targets) {
         let target_idx = (local_idx + 1u) * step - 1u;
         let prev_idx = target_idx - (step >> 1u);
         let prev_val = local_data[prev_idx];
         local_data[prev_idx] = local_data[target_idx];
         local_data[target_idx] += prev_val;
     }
     workgroupBarrier();
     step = step >> 1u;
    }
}

@compute @workgroup_size(WG_SIZE)
fn block_scan_write_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let n = arrayLength(&global_data);
    let indices = get_indices(lid, wid, num_wg);
    let local_idx = indices[0];
    let global_idx = indices[1];
    copy_global_data_to_local(n, local_idx, global_idx);

    up_sweep(local_idx);

    // write out the block sum here before overwriting with 0
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let n_blocks = arrayLength(&block_sum);
    if (local_idx == 0u) {
        if (wg_linear < n_blocks) {
            block_sum[wg_linear] = local_data[WG_SIZE - 1u];
        }
        local_data[WG_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    down_sweep(local_idx);

    // write out the local scan result to the global storage
    if (global_idx < n) {
        global_data[global_idx] = local_data[local_idx];
    }
}

@compute @workgroup_size(WG_SIZE)
fn block_scan_no_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
     let n = arrayLength(&global_data);
     let indices = get_indices(lid, wid, num_wg);
     let local_idx = indices[0];
     let global_idx = indices[1];
     copy_global_data_to_local(n, local_idx, global_idx);

     up_sweep(local_idx);

     if (local_idx == 0u) {
         local_data[WG_SIZE - 1u] = 0u;
     }
     workgroupBarrier();

     down_sweep(local_idx);

     // write out the local scan result to the global storage
     if (global_idx < n) {
         global_data[global_idx] = local_data[local_idx];
     }
}