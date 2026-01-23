struct Uniforms {
  step: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> uni: Uniforms;

@compute
@workgroup_size(64)
fn main(
     @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
) {
    let i = global_invocation_id.x;
    let total = arrayLength(&src);
    // We need to stop the operation since the workgroup size may not be exact multiple of the array size.
    if (i >= total) {
        return;
    }

    if (i < uni.step) {
        dst[i] = src[i];
    } else {
        dst[i] = src[i] + src[i - uni.step];
    }
}