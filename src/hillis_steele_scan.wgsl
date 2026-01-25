struct Uniforms {
  step: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> uni: Uniforms;

@compute
@workgroup_size(64)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let total = arrayLength(&src);

    let width = nwg.x * 64u;
    let i = gid.x + gid.y * width;

    if (i >= total) {
        return;
    }

    if (i < uni.step) {
        dst[i] = src[i];
    } else {
        dst[i] = src[i] + src[i - uni.step];
    }
}