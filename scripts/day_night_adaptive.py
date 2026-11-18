"""Day/Night Adaptive Thresholding benchmark for CameraDegradationGuard (Item 3).

Compares fixed-baseline Hybrid Health against AmbientLuminanceEstimator adaptive
baselines across DRIVE-C Day (S01) and Night (S06/S07/S08) scenarios.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.camera_guard import CameraDegradationGuard


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def check_monotonicity(rows: List[Dict[str, Any]], key: str) -> bool:
    vals = [float(r[key]) for r in rows]
    return all(a >= b for a, b in zip(vals, vals[1:]))


def run_day_night_comparison(
    input_csv: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = load_csv(input_csv)
    guard = CameraDegradationGuard(use_neural=False)

    comparison_rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        B = float(r["B"])
        S_pct = float(r["S_pct"])
        H_bit = float(r["H_bit"])
        pred_f54 = float(r["gshi_pred_f54"])

        fixed_res = guard.score_from_metrics(
            B=B, S_pct=S_pct, H_bit=H_bit, pred_health=pred_f54, adaptive_day_night=False
        )
        adapt_res = guard.score_from_metrics(
            B=B, S_pct=S_pct, H_bit=H_bit, pred_health=pred_f54, adaptive_day_night=True
        )

        comparison_rows.append(
            {
                "sample_id": r["sample_id"],
                "time_of_day": r["time_of_day"],
                "corruption": r["corruption"],
                "severity_value": float(r["severity_value"]),
                "B": B,
                "S_pct": S_pct,
                "H_bit": H_bit,
                "pred_health_f54": pred_f54,
                "fixed_health": round(fixed_res.health_score, 6),
                "fixed_state": fixed_res.system_state,
                "fixed_weight": round(fixed_res.camera_weight, 4),
                "adaptive_regime": adapt_res.ambient_regime,
                "daylight_factor": round(adapt_res.daylight_factor, 4),
                "B_baseline_dynamic": round(adapt_res.B_baseline_dynamic, 2),
                "B_norm_adaptive": round(adapt_res.B_norm, 4),
                "delta_discordance": round(adapt_res.delta_discordance, 4),
                "discordance_triggered": adapt_res.discordance_triggered,
                "adaptive_health": round(adapt_res.health_score, 6),
                "adaptive_state": adapt_res.system_state,
                "adaptive_weight": round(adapt_res.camera_weight, 4),
                "adaptive_early_warning": adapt_res.early_warning_flag,
            }
        )

    csv_path = output_dir / "day_night_adaptive_comparison.csv"
    write_csv(csv_path, comparison_rows)

    # Check monotonicity across all 4 degradation series (clean + s1..s5)
    series_defs = [
        ("S01_motion_blur", "S01_clean", "S01_motion_blur_s"),
        ("S01_underexposure", "S01_clean", "S01_underexposure_s"),
        ("S06_motion_blur", "S06_clean", "S06_motion_blur_s"),
        ("S06_underexposure", "S06_clean", "S06_underexposure_s"),
    ]
    mono_summary: Dict[str, Dict[str, bool]] = {}
    for name, clean_id, prefix in series_defs:
        subset = [x for x in comparison_rows if x["sample_id"] == clean_id or x["sample_id"].startswith(prefix)]
        mono_summary[name] = {
            "pred_health_monotonic": check_monotonicity(subset, "pred_health_f54"),
            "fixed_health_monotonic": check_monotonicity(subset, "fixed_health"),
            "adaptive_health_monotonic": check_monotonicity(subset, "adaptive_health"),
        }

    # Plot comparison across the 4 series + clean night clips
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    titles = [
        "Day S01: Motion Blur (s0-s5)",
        "Day S01: Underexposure (s0-s5)",
        "Night S06: Motion Blur (s0-s5)",
        "Night S06: Underexposure (s0-s5)",
    ]
    for ax, (name, clean_id, prefix), title in zip(axes.flat, series_defs, titles):
        subset = [x for x in comparison_rows if x["sample_id"] == clean_id or x["sample_id"].startswith(prefix)]
        x_idx = list(range(len(subset)))
        labels = ["Clean"] + [f"s{i}" for i in range(1, len(subset))]
        ax.plot(x_idx, [r["pred_health_f54"] for r in subset], "o--", color="#d62728", label="Neural pred_health (raw)")
        ax.plot(x_idx, [r["fixed_health"] for r in subset], "s:", color="#7f7f7f", label="Hybrid (Fixed Day Threshold)")
        ax.plot(x_idx, [r["adaptive_health"] for r in subset], "^-", color="#1f77b4", linewidth=2.2, label="Hybrid (Day/Night Adaptive)")
        ax.axhline(0.75, color="#2ca02c", linestyle="--", alpha=0.5, linewidth=1.0, label="HEALTHY (>=0.75)")
        ax.axhline(0.35, color="#ff7f0e", linestyle="--", alpha=0.5, linewidth=1.0, label="CRITICAL (<0.35)")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(x_idx, labels)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Health Score [0-1]")
        ax.grid(alpha=0.25)
        if name == "S01_motion_blur":
            ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Day/Night Adaptive Hybrid Health vs Raw PerceptionHealthNet & Fixed Baseline",
        fontsize=13,
        fontweight="bold",
    )
    plot_path = output_dir / "day_night_adaptive_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary = {
        "rows_evaluated": len(comparison_rows),
        "monotonicity_checks": mono_summary,
        "S06_clean_fixed_state": next(x["fixed_state"] for x in comparison_rows if x["sample_id"] == "S06_clean"),
        "S06_clean_adaptive_state": next(x["adaptive_state"] for x in comparison_rows if x["sample_id"] == "S06_clean"),
        "S06_clean_adaptive_health": next(x["adaptive_health"] for x in comparison_rows if x["sample_id"] == "S06_clean"),
        "S01_underexposure_s5_adaptive_state": next(
            x["adaptive_state"] for x in comparison_rows if x["sample_id"] == "S01_underexposure_s5"
        ),
        "csv_path": str(csv_path.relative_to(ROOT)),
        "plot_path": str(plot_path.relative_to(ROOT)),
    }
    (output_dir / "day_night_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "outputs/phn_24_results.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pipeline_v2")
    args = parser.parse_args()
    summary = run_day_night_comparison(args.input_csv, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
