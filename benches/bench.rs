use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use wgpu_prefix_sum_demo::cpu_prefix_scan::cpu_prefix_sum;
use wgpu_prefix_sum_demo::hillis_steele_scan::HillisSteeleGpuContext;

fn criterion_benchmark(c: &mut Criterion) {
    let n = 1_000_000;
    let data = vec![1u32; n];
    let gpu_ctx = pollster::block_on(HillisSteeleGpuContext::new(n)).unwrap();
    gpu_ctx.upload_data(&data);

    c.bench_function("CPU prefix sum", |b| {
        b.iter(|| {
            let v = cpu_prefix_sum(black_box(&data));
            black_box(v);
        })
    });

    c.bench_function("GPU prefix sum", |b| {
        b.iter_batched(
            || {
                gpu_ctx.run_prefix_scan();
            },
            |_| {
                gpu_ctx.wait_idle().unwrap();
            },
            BatchSize::PerIteration,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
