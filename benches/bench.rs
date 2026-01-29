use wgpu_prefix_sum_demo::block_blelloch_scan::BlockBlellochGpuContext;
use wgpu_prefix_sum_demo::cpu_prefix_scan::cpu_prefix_sum;
use wgpu_prefix_sum_demo::global_blelloch_scan::GlobalBlellochGpuContext;
use wgpu_prefix_sum_demo::hillis_steele_scan::HillisSteeleGpuContext;
use wgpu_prefix_sum_demo::subgroup_scan::SubgroupScanGpuContext;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn bench_prefix_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("Prefix scan comparison");

    let sizes: Vec<usize> = (1..=29).map(|p| 1usize << p).collect();

    for &n in &sizes {
        let data = vec![1u32; n];

        let hillis = pollster::block_on(HillisSteeleGpuContext::new(n)).unwrap();
        hillis.upload_data(&data);

        let global = pollster::block_on(GlobalBlellochGpuContext::new(n)).unwrap();
        global.upload_data(&data);
        
        let blocked = pollster::block_on(BlockBlellochGpuContext::new(n)).unwrap();
        blocked.upload_data(&data);
        
        let subgroup = pollster::block_on(SubgroupScanGpuContext::new(n)).unwrap();
        subgroup.upload_data(&data);

        group.bench_with_input(BenchmarkId::new("CPU Sequential", n), &n, |b, &_n| {
            b.iter(|| {
                let v = cpu_prefix_sum(&data);
                std::hint::black_box(v);
            });
        });

        group.bench_with_input(BenchmarkId::new("GPU Hillis-Steele", n), &n, |b, &_n| {
            b.iter_batched(
                || {
                    hillis.run_prefix_scan();
                },
                |_| {
                    hillis.wait_idle().unwrap();
                },
                BatchSize::PerIteration,
            )
        });

        group.bench_with_input(BenchmarkId::new("GPU Global Blelloch", n), &n, |b, &_n| {
            b.iter_batched(
                || {
                    let mut enc = global.get_command_encoder();
                    global.encode_up_sweep(&mut enc);
                    global.encode_set_last_zero(&mut enc);
                    global.encode_down_sweep(&mut enc);
                    global.submit(enc);
                },
                |_| {
                    global.wait_idle().unwrap();
                },
                BatchSize::PerIteration,
            )
        });
    
        group.bench_with_input(BenchmarkId::new("GPU Blocked Blelloch", n), &n, |b, &_n| {
            b.iter_batched(
                || {
                    let mut enc = blocked.get_command_encoder();
                    blocked.encode_scan(&mut enc);
                    blocked.submit(enc);
                },
                |_| {
                    blocked.wait_idle().unwrap();
                },
                BatchSize::PerIteration,
            )
        });
        
        group.bench_with_input(BenchmarkId::new("GPU Subgroup", n), &n, |b, &_n| {
            b.iter_batched(
                || {
                    let mut enc = subgroup.get_command_encoder();
                    subgroup.encode_scan(&mut enc);
                    subgroup.submit(enc);
                },
                |_| {
                    subgroup.wait_idle().unwrap();
                },
                BatchSize::PerIteration,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_prefix_scan);
criterion_main!(benches);
