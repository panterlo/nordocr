//! End-to-end pipeline benchmarks using criterion.
//!
//! Run with: cargo bench --bench pipeline_bench
//!
//! Requires:
//! - CUDA-capable GPU
//! - TensorRT engine files in models/
//! - Test images in bench/datasets/

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_preprocess(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocess");

    for &(width, height) in &[(2480, 3508), (1240, 1754), (620, 877)] {
        group.bench_with_input(
            BenchmarkId::new("denoise+deskew+binarize", format!("{width}x{height}")),
            &(width, height),
            |b, &(w, h)| {
                // In production:
                //   let ctx = GpuContext::default_device().unwrap();
                //   let pipeline = PreprocessPipeline::new(&ctx, Default::default()).unwrap();
                //   let input = ctx.memory_pool.alloc::<u8>((w * h) as usize).unwrap();
                //   b.iter(|| pipeline.execute(&ctx, &input, w, h).unwrap());

                b.iter(|| {
                    // Placeholder until GPU is available.
                    std::hint::black_box((w, h));
                });
            },
        );
    }
    group.finish();
}

fn bench_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection");

    for &batch_size in &[1, 2, 4] {
        group.bench_with_input(
            BenchmarkId::new("dbnet_infer", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    std::hint::black_box(bs);
                });
            },
        );
    }
    group.finish();
}

fn bench_recognition(c: &mut Criterion) {
    let mut group = c.benchmark_group("recognition");

    for &batch_size in &[16, 32, 64, 128] {
        group.bench_with_input(
            BenchmarkId::new("parseq_infer", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    std::hint::black_box(bs);
                });
            },
        );
    }
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e");
    group.sample_size(10); // fewer iterations for slow benchmarks

    group.bench_function("single_page_a4_300dpi", |b| {
        b.iter(|| {
            std::hint::black_box(());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_preprocess,
    bench_detection,
    bench_recognition,
    bench_end_to_end
);
criterion_main!(benches);
