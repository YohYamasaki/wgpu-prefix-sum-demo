use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use wgpu_prefix_sum_demo::{GpuContext, cpu_prefix_sum};

fn criterion_benchmark(c: &mut Criterion) {
    let n = 1_000_000;
    let data = vec![1u32; n];
    let gpu_ctx = pollster::block_on(GpuContext::new(n)).unwrap();
    gpu_ctx.upload_data(&data);

    c.bench_function("CPU prefix sum", |b| {
        b.iter(|| {
            let v = cpu_prefix_sum(black_box(&data));
            black_box(v);
        })
    });

    c.bench_function("GPU prefix sum", |b| {
        b.iter(|| {
            gpu_ctx.run_prefix_scan();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
