extern crate core;

use wgpu_prefix_sum_demo::block_blelloch_scan::BlockBlellochGpuContext;
use wgpu_prefix_sum_demo::cpu_prefix_scan::cpu_prefix_sum;

fn main() -> anyhow::Result<()> {
    let n = 10_000_000u32.next_power_of_two() as usize;
    let data = vec![1u32; n];
    let cpu_res = cpu_prefix_sum(&data);

    let gpu_ctx = pollster::block_on(BlockBlellochGpuContext::new(n))?;
    println!("n: {}", n);
    gpu_ctx.upload_data(&data);
    gpu_ctx.run_prefix_sum();
    let gpu_res = gpu_ctx.read_computed_data()?;

    assert_eq!(cpu_res[n - 1], gpu_res[n - 1] + data[n - 1]);
    Ok(())
}
