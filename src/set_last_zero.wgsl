@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main() {
  let n = arrayLength(&data);
  if (n > 0u) {
    data[n - 1u] = 0u;
  }
}