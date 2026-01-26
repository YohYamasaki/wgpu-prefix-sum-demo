use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;
use wgpu_prefix_sum_demo::cpu_prefix_scan::cpu_prefix_sum;
use wgpu_prefix_sum_demo::global_blelloch_scan::GlobalBlellochGpuContext;
use wgpu_prefix_sum_demo::hillis_steele_scan::HillisSteeleGpuContext;

fn criterion_benchmark(c: &mut Criterion) {
    let n = 10_000_000u32.next_power_of_two() as usize;
    let data = vec![1u32; n];
    let hillis_steele_gpu_ctx = pollster::block_on(HillisSteeleGpuContext::new(n)).unwrap();
    let global_blelloch_gpu_ctx = pollster::block_on(GlobalBlellochGpuContext::new(n)).unwrap();
    hillis_steele_gpu_ctx.upload_data(&data);
    global_blelloch_gpu_ctx.upload_data(&data);

    c.bench_function("CPU prefix sum", |b| {
        b.iter(|| {
            let v = cpu_prefix_sum(black_box(&data));
            black_box(v);
        })
    });

    c.bench_function("GPU prefix sum (Hillis-Steele)", |b| {
        b.iter_batched(
            || {
                hillis_steele_gpu_ctx.run_prefix_scan();
            },
            |_| {
                hillis_steele_gpu_ctx.wait_idle().unwrap();
            },
            BatchSize::PerIteration,
        )
    });

    c.bench_function("GPU prefix sum (Global Blelloch)", |b| {
        b.iter_batched(
            || {
                let mut encoder = global_blelloch_gpu_ctx.get_command_encoder();
                global_blelloch_gpu_ctx.encode_up_sweep(&mut encoder);
                global_blelloch_gpu_ctx.encode_set_last_zero(&mut encoder);
                global_blelloch_gpu_ctx.encode_down_sweep(&mut encoder);
                global_blelloch_gpu_ctx.submit(encoder);
            },
            |_| {
                global_blelloch_gpu_ctx.wait_idle().unwrap();
            },
            BatchSize::PerIteration,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
