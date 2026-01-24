extern crate core;

use wgpu_prefix_sum_demo::GpuContext;

fn main() -> anyhow::Result<()> {
    let n = 1_000_000_000;
    let data = vec![1u32; n];
    let gpu_ctx = pollster::block_on(GpuContext::new(n))?;
    gpu_ctx.upload_data(&data);
    gpu_ctx.run_prefix_scan();
    let res = gpu_ctx.read_computed_data()?;
    println!("Last element: {}", res[res.len() - 1]);
    Ok(())
}
