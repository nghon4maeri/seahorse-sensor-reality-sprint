//! Standalone CLI binary for profiling and evaluating RGB images in Rust.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use camera_guard_rs::{evaluate_rgb_frame, GuardConfig};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: guard_cli <image.png> [--pred-health <float>] [--no-adaptive] [--iters <int>]");
        return ExitCode::from(2);
    }

    let img_path = PathBuf::from(&args[1]);
    let mut pred_health: Option<f64> = None;
    let mut adaptive = true;
    let mut iters: usize = 1;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--pred-health" => {
                i += 1;
                pred_health = args.get(i).and_then(|s| s.parse::<f64>().ok());
            }
            "--no-adaptive" => {
                adaptive = false;
            }
            "--iters" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse::<usize>().ok()) {
                    iters = v.max(1);
                }
            }
            other => {
                eprintln!("Unknown argument: {other}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let img = match image::open(&img_path) {
        Ok(im) => im.to_rgb8(),
        Err(err) => {
            eprintln!("Failed to open {}: {err}", img_path.display());
            return ExitCode::from(1);
        }
    };
    let (width, height) = (img.width() as usize, img.height() as usize);
    let raw = img.as_raw();
    let cfg = GuardConfig::default();

    // Warm up once when profiling multiple iterations
    if iters > 1 {
        let _ = evaluate_rgb_frame(raw, height, width, pred_health, adaptive, &cfg);
    }

    let t0 = Instant::now();
    let mut last_res = None;
    for _ in 0..iters {
        match evaluate_rgb_frame(raw, height, width, pred_health, adaptive, &cfg) {
            Ok(res) => last_res = Some(res),
            Err(err) => {
                eprintln!("Evaluation error: {err}");
                return ExitCode::from(1);
            }
        }
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

    if let Some(res) = last_res {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "image": img_path.display().to_string(),
                "dimensions": [height, width],
                "iterations": iters,
                "mean_latency_ms": elapsed_ms,
                "result": res,
            }))
            .unwrap()
        );
    }
    ExitCode::SUCCESS
}
