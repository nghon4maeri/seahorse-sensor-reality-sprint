# `camera_guard_rs` — High-Performance Rust ADAS Camera Safety Guard

Real-time, memory-safe, zero-allocation optical quality & hybrid health monitor for ADAS camera pipelines, exposing both a standalone Rust library/CLI and zero-copy Python bindings via **PyO3 / Maturin**.

## Architecture

- `src/metrics.rs`: Single-pass, Rayon-parallel, auto-vectorized RGB24 → grayscale (`CV_DESCALE` fixed-point), 2D discrete Laplacian variance (`BORDER_REFLECT_101`, `ksize=1`, `ddof=0`), saturation clipping `S_pct`, 256-bin Shannon entropy `H_bit`, mean luma, `P95` luma, and dark pixel ratio.
- `src/ambient.rs`: `AmbientLuminanceEstimator` computing `daylight_factor` and log-linear dynamic `B_baseline(D)` across Day (`3544.14`) and Night (`86.30`).
- `src/guard.rs`: Hybrid Health Engine (`H_hybrid`), Discordance Detector ($\Delta_{\text{discordance}} > \tau$), and smooth Hermite-curve dynamic sensor fusion weight $w_{\text{cam}} \in [0.0, 1.0]$.
- `src/lib.rs`: PyO3 bindings exposing `evaluate_frame(image_np, pred_health=None, adaptive_day_night=True)` and `compute_metrics(image_np)` over zero-copy NumPy buffers (`allow_threads` releases the Python GIL).
- `src/bin/guard_cli.rs`: Standalone CLI binary for profiling and JSON diagnostics.

## Build, Test & Benchmark

### 1. Run Rust Unit & Integration Tests
Validates exact numerical equivalence against OpenCV (`B = 3544.1384 ± 0.005`, `S_pct = 1.6558% ± 0.001`, `H_bit = 7.5152 ± 0.001`), strict monotonicity on `S01` and `S06`, and Day/Night separation:
```powershell
cargo test --manifest-path crates/camera_guard_rs/Cargo.toml --release
```

### 2. Run Criterion Latency Benchmarks (720p & 1080p)
```powershell
cargo bench --manifest-path crates/camera_guard_rs/Cargo.toml
```

### 3. Run Standalone CLI Profiler
```powershell
cargo run --manifest-path crates/camera_guard_rs/Cargo.toml --release --bin guard_cli -- outputs/stage3_small/S01_motion_blur_s2_f54.png --pred-health 0.407456 --iters 200
```

### 4. Build Python Bindings (`camera_guard_rs`)
```powershell
cd crates/camera_guard_rs
python -m maturin develop --release
```
Then in Python:
```python
import cv2, camera_guard_rs
rgb = cv2.cvtColor(cv2.imread("outputs/stage3_small/S01_motion_blur_s2_f54.png"), cv2.COLOR_BGR2RGB)
res = camera_guard_rs.evaluate_frame(rgb, pred_health=0.407456, adaptive_day_night=True)
print(res)
```
