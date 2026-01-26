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
    let n = arrayLength(&data);
        let step = uni.step;
        let half = step >> 1u;

        let width = nwg.x * 64u;
        let plane = width * nwg.y;
        let t = gid.x + gid.y * width + gid.z * plane;

        let active_idx = n / step;
        if (t >= active_idx) { return; }

        // We need (step - 1u) to target the last element of the current block
        let i = (step - 1u) + t * step;
        let prev = i - half;

        let left = data[i];
        data[i] = data[i] + data[prev];
        data[prev] = left;
}