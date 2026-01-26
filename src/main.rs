extern crate core;

use wgpu_prefix_sum_demo::global_blelloch_scan::GlobalBlellochGpuContext;

fn main() -> anyhow::Result<()> {
    let n = 100_000_000u32.next_power_of_two() as usize;
    let data = vec![1u32; n];
    // let gpu_ctx = pollster::block_on(HillisSteeleGpuContext::new(n))?;
    // let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    // let n = data.len().next_power_of_two();
    let gpu_ctx = pollster::block_on(GlobalBlellochGpuContext::new(n))?;
    println!("n: {}", n);
    gpu_ctx.upload_data(&data);
    gpu_ctx.run_prefix_sum();
    let res = gpu_ctx.read_computed_data()?;
    println!("Last element: {}", res[n - 1]);
    Ok(())
}
