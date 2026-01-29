# wgpu-prefix-sum-demo

A demo of GPU prefix-sum (scan) implementations in Rust using wgpu, with multiple algorithms and a small benchmark harness for comparisons.

These implementations are for demonstration purposes only and do not support arrays of arbitrary length, overflow handling or any other edge cases. That means these implementations are not suitable for practical use, although they might be a good starting point for your implementation.

## Benchmark results

Run on Mac mini M4 Pro 24GB with 16 core GPU.

![Parallel prefix sum comparison](/benches/parallel-prefix-sum-comparison.png)

## What this project includes

- CPU baseline: sequential inclusive prefix sum (`src/cpu_prefix_scan.rs`).
- GPU Hillis-Steele scan (inclusive) with double buffers (`src/hillis_steele_scan.rs`).
- GPU Blelloch scan (exclusive) in two forms:
  - On global memory (`src/global_blelloch_scan.rs`).
  - Blocked scan using shared memory (`src/block_blelloch_scan.rs`).
- GPU subgroup scan (exclusive) using subgroup operations (`src/subgroup_scan.rs`).

WGSL shaders for each GPU path in `src/*.wgsl`.

## Requirements

- Rust toolchain with 2024 edition support.
- A GPU/driver that supports `wgpu` compute and the `SUBGROUP` feature (required by the subgroup scan).

## Run benchmarks

Criterion benchmarks compare all implementations across powers of two from 2^1 to 2^29.

```bash
cargo bench
```

The benchmark entry point is `benches/bench.rs`.

## Notes

- The Hillis-Steele implementation produces an inclusive scan.
- The Blelloch and subgroup implementations produce exclusive scans.
- GPU implementations assume the input length is a power of two.
