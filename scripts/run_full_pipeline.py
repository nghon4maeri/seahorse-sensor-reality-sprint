"""Unified end-to-end ADAS Camera Degradation Guard & Downstream Benchmark Pipeline.

Orchestrates all 5 improvement components:
  1. Downstream YOLOv8 Object Detection Benchmark (count, confidence, mAP@0.50 proxy)
  2. Hybrid Health Score (H_hybrid) with Discordance Detection
  3. Day/Night Adaptive Thresholding (AmbientLuminanceEstimator)
  4. Spatial Uncertainty Map (pred_pix) extraction & BBox overlap visualization
  5. CameraDegradationGuard module telemetry & Dynamic Sensor Fusion Weighting
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.day_night_adaptive import run_day_night_comparison
from scripts.downstream_yolov8_benchmark import run_downstream_benchmark
from scripts.spatial_uncertainty_viz import run_spatial_uncertainty_analysis


def build_executive_dashboard(
    downstream_summary: List[Dict[str, Any]],
    day_night_csv: Path,
    output_path: Path,
) -> None:
    """Create a 4-panel executive dashboard linking Neural, Hybrid, Downstream, and Day/Night."""
    labels = ["Clean", "Blur s1\n(11 px)", "Blur s2\n(13 px)", "Blur s5\n(33 px)"]
    x = np.arange(len(downstream_summary))

    neural_f54 = [float(r["f54_pred_health"]) for r in downstream_summary]
    hybrid_f54 = [float(r["f54_hybrid_health"]) for r in downstream_summary]
    hybrid_mean8 = [float(r["mean8_hybrid_health"]) for r in downstream_summary]
    map50_8f = [float(r["mean_map50_proxy_8f"]) for r in downstream_summary]
    retention_8f = [max(0.0, 1.0 - float(r["object_drop_pct_8f"]) / 100.0) for r in downstream_summary]
    conf_8f = [float(r["mean_confidence_8f"]) for r in downstream_summary]
    cam_weights = [float(r["mean8_camera_weight"]) for r in downstream_summary]
    discordances = [max(0.0, n - (float(r["f54_B"]) / 3544.1384)) for r, n in zip(downstream_summary, neural_f54)]

    with day_night_csv.open("r", encoding="utf-8", newline="") as f:
        dn_rows = {r["sample_id"]: r for r in csv.DictReader(f)}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), constrained_layout=True)

    # Panel 1: Neural failure vs Hybrid Health
    ax = axes[0, 0]
    ax.plot(x, neural_f54, "o--", color="#d62728", linewidth=2.0, label="Neural pred_health (fails at s2)")
    ax.plot(x, hybrid_f54, "^-", color="#1f77b4", linewidth=2.4, label="Hybrid Health (frame 54)")
    ax.plot(x, hybrid_mean8, "s:", color="#2ca02c", linewidth=1.8, label="Hybrid Health (mean 8 frames)")
    ax.axhspan(0.75, 1.05, color="#2ca02c", alpha=0.07)
    ax.axhspan(0.35, 0.75, color="#ff7f0e", alpha=0.07)
    ax.axhspan(-0.05, 0.35, color="#d62728", alpha=0.07)
    ax.annotate(
        f"Failure case s2:\nNeural={neural_f54[2]:.3f} > Clean!",
        xy=(2, neural_f54[2]),
        xytext=(1.1, 0.62),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4),
        fontsize=8.5,
        color="#a91d1d",
        fontweight="bold",
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Health Score [0-1]")
    ax.set_title("1. Neural Failure vs Monotonic Hybrid Health", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    # Panel 2: Downstream YOLOv8 Perception Degradation
    ax = axes[0, 1]
    ax.plot(x, map50_8f, "D-", color="#9467bd", linewidth=2.2, label="YOLOv8 mAP@0.50 proxy (8f)")
    ax.plot(x, retention_8f, "o-", color="#17becf", linewidth=2.0, label="Object Retention Rate (8f)")
    ax.plot(x, conf_8f, "s--", color="#ff7f0e", linewidth=1.8, label="Mean Detection Confidence (8f)")
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Downstream Score [0-1]")
    ax.set_title("2. Downstream YOLOv8 Impact (8-Frame Mean)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    # Panel 3: Dynamic Gating Weight & Discordance
    ax = axes[1, 0]
    width = 0.36
    ax.bar(x - width / 2, cam_weights, width, color="#1f77b4", label="Dynamic Camera Weight [0-1]")
    ax.bar(x + width / 2, discordances, width, color="#e74c3c", alpha=0.85, label="Discordance Delta (pred - B_norm)")
    ax.axhline(0.05, color="#c0392b", linestyle=":", linewidth=1.2, label="Discordance threshold tau=0.05")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Weight / Discordance")
    ax.set_title("3. Sensor Fusion Dynamic Gating & Discordance", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, axis="y")

    # Panel 4: Day vs Night Adaptive Thresholding
    ax = axes[1, 1]
    dn_ids = ["S01_clean", "S01_underexposure_s3", "S06_clean", "S06_motion_blur_s2", "S06_underexposure_s3"]
    dn_labels = ["S01\nClean\n(Day)", "S01\nUnderexp\ns3 (Day)", "S06\nClean\n(Night)", "S06\nBlur s2\n(Night)", "S06\nUnderexp\ns3 (Night)"]
    fixed_vals = [float(dn_rows[k]["fixed_health"]) for k in dn_ids]
    adapt_vals = [float(dn_rows[k]["adaptive_health"]) for k in dn_ids]
    dx = np.arange(len(dn_ids))
    ax.bar(dx - width / 2, fixed_vals, width, color="#7f7f7f", label="Fixed Day Threshold")
    ax.bar(dx + width / 2, adapt_vals, width, color="#2ecc71", label="Day/Night Adaptive Guard")
    ax.axhline(0.75, color="#27ae60", linestyle="--", linewidth=1.0, label="HEALTHY (>=0.75)")
    ax.axhline(0.35, color="#e67e22", linestyle="--", linewidth=1.0, label="CRITICAL (<0.35)")
    ax.set_xticks(dx, dn_labels, fontsize=8.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Health Score [0-1]")
    ax.set_title("4. Day/Night Adaptive Calibration (Avoids Night Clean False Alarm)", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Seahorse ADAS CameraDegradationGuard: End-to-End Hybrid & Downstream Validation", fontsize=13.5, fontweight="bold")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["S01_clean", "S01_motion_blur_s1", "S01_motion_blur_s2", "S01_motion_blur_s5"],
    )
    parser.add_argument("--weights", type=Path, default=ROOT / ".lab_cache/weights/yolov8n.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pipeline_v2")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--skip-spatial", action="store_true", help="Skip pred_pix extraction if checkpoint unavailable")
    args = parser.parse_args()

    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    ckpt_path = ROOT / "drive-c-dataset/checkpoints/epoch_021_best.pth"
    source_ok = (ROOT / ".lab_cache/drive-c-source/scripts/add_gshi_pred.py").exists()
    ckpt_ok = ckpt_path.exists() and ckpt_path.stat().st_size == 94255859
    use_neural = source_ok and ckpt_ok and (not args.skip_spatial)

    print(f"[1/4] Running Downstream YOLOv8 Benchmark (use_neural_guard={use_neural})...", flush=True)
    downstream_res = run_downstream_benchmark(
        sample_ids=args.samples,
        output_dir=args.output_dir,
        weights_path=args.weights,
        conf_thresh=args.conf,
        use_neural_guard=use_neural,
    )

    print("[2/4] Running Day/Night Adaptive Thresholding Benchmark...", flush=True)
    day_night_res = run_day_night_comparison(
        input_csv=ROOT / "outputs/phn_24_results.csv",
        output_dir=args.output_dir,
    )

    spatial_res: Dict[str, Any]
    if use_neural:
        print("[3/4] Running Spatial Uncertainty Map (pred_pix) & BBox Overlap...", flush=True)
        spatial_res = run_spatial_uncertainty_analysis(
            sample_ids=args.samples,
            output_dir=args.output_dir,
            weights_path=args.weights,
            frame_idx=54,
            conf_thresh=args.conf,
        )
    else:
        print("[3/4] Skipping pred_pix spatial extraction (checkpoint not ready or --skip-spatial).", flush=True)
        spatial_res = {"skipped": True, "reason": "checkpoint_missing_or_skipped"}

    print("[4/4] Building Executive 4-Panel Dashboard...", flush=True)
    dashboard_path = args.output_dir / "hybrid_vs_downstream_dashboard.png"
    build_executive_dashboard(
        downstream_summary=downstream_res["summary"],
        day_night_csv=args.output_dir / "day_night_adaptive_comparison.csv",
        output_path=dashboard_path,
    )

    manifest = {
        "generated_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "live_neural_monitor_used": use_neural,
        "wall_seconds": round(time.perf_counter() - t_start, 3),
        "downstream_benchmark": downstream_res,
        "day_night_adaptive": day_night_res,
        "spatial_uncertainty": spatial_res,
        "dashboard_path": str(dashboard_path.relative_to(ROOT)),
    }
    manifest_path = args.output_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DONE in {manifest['wall_seconds']:.2f}s -> {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
