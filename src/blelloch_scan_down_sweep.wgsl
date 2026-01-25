struct Uniform {
  step: u32, // this has to be a power of 2
};

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> uni: Uniform;

@compute
@workgroup_size(64)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let total = arrayLength(&data);

    let width = nwg.x * 64u;
    let plane = width * nwg.y;
    let i = gid.x + gid.y * width + gid.z * plane;

    if (i >= total) {
        return;
    }

    // Check whether the current index is positioned on a power of 2. Same as (i + 1) % uni.step != 0
    if (((i + 1u) & (uni.step - 1u)) != 0) {
        return;
    }

    let prev_i = i - (uni.step >> 1);
    let left = data[i];
    data[i] += data[prev_i];
    data[prev_i] = left;
}