//! Criterion performance benchmarks targeting < 1.5 ms per 720p/1080p frame.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use camera_guard_rs::{compute_optical_metrics, evaluate_rgb_frame, GuardConfig};

fn make_synthetic_rgb(height: usize, width: usize) -> Vec<u8> {
    let mut buf = vec![0u8; height * width * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            buf[idx] = ((x * 7 + y * 3) & 0xFF) as u8;
            buf[idx + 1] = ((x * 5 + y * 11) & 0xFF) as u8;
            buf[idx + 2] = ((x * 13 + y * 2) & 0xFF) as u8;
        }
    }
    buf
}

fn bench_frames(c: &mut Criterion) {
    let cfg = GuardConfig::default();
    let rgb_720p = make_synthetic_rgb(720, 1280);
    let rgb_1080p = make_synthetic_rgb(1080, 1920);

    let mut group = c.benchmark_group("camera_guard_rs");
    group.throughput(Throughput::Bytes(rgb_720p.len() as u64));
    group.bench_function("metrics_720p_1280x720", |b| {
        b.iter(|| compute_optical_metrics(black_box(&rgb_720p), 720, 1280).unwrap())
    });
    group.bench_function("evaluate_frame_720p_1280x720", |b| {
        b.iter(|| evaluate_rgb_frame(black_box(&rgb_720p), 720, 1280, Some(0.4075), true, &cfg).unwrap())
    });

    group.throughput(Throughput::Bytes(rgb_1080p.len() as u64));
    group.bench_function("metrics_1080p_1920x1080", |b| {
        b.iter(|| compute_optical_metrics(black_box(&rgb_1080p), 1080, 1920).unwrap())
    });
    group.bench_function("evaluate_frame_1080p_1920x1080", |b| {
        b.iter(|| evaluate_rgb_frame(black_box(&rgb_1080p), 1080, 1920, Some(0.4075), true, &cfg).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_frames);
criterion_main!(benches);
