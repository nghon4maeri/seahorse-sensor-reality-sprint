"""Spatial Uncertainty Map (pred_pix) extraction and bbox overlap analysis (Item 4).

Extracts the pixel-level uncertainty head pred_pix from PerceptionHealthNet,
overlays it on RGB frames alongside Clean YOLOv8 reference bounding boxes, and
quantifies uncertainty inside vs outside detected objects.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.camera_guard import CameraDegradationGuard
from scripts.downstream_yolov8_benchmark import read_frames_at_indices, resolve_sample_videos


def render_spatial_overlay(
    rgb: np.ndarray,
    pred_pix: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    names: Dict[int, str],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend pred_pix heatmap onto RGB and draw reference Clean YOLOv8 boxes."""
    h, w, _ = rgb.shape
    pix_resized = cv2.resize(pred_pix, (w, h), interpolation=cv2.INTER_LINEAR)
    pix_norm = np.clip((pix_resized - pix_resized.min()) / max(pix_resized.max() - pix_resized.min(), 1e-6), 0.0, 1.0)
    heat_bgr = cv2.applyColorMap((pix_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(rgb, 1.0 - alpha, heat_rgb, alpha, 0.0)
    for box, cls_id in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            blended,
            names.get(int(cls_id), str(cls_id)),
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return blended


def run_spatial_uncertainty_analysis(
    sample_ids: Sequence[str],
    output_dir: Path,
    weights_path: Path,
    frame_idx: int = 54,
    conf_thresh: float = 0.25,
) -> Dict[str, Any]:
    import torch
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    guard = CameraDegradationGuard(use_neural=True, device=device)
    yolo = YOLO(str(weights_path))
    videos = resolve_sample_videos(sample_ids)

    frames_f54: Dict[str, np.ndarray] = {}
    for sid in sample_ids:
        frames_f54[sid] = read_frames_at_indices(videos[sid], [frame_idx])[0]

    # Reference Clean YOLOv8 bounding boxes
    clean_rgb = frames_f54[sample_ids[0]]
    clean_pred = yolo.predict(cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR), device=device, conf=conf_thresh, verbose=False)[0]
    gt_boxes = clean_pred.boxes.xyxy.detach().cpu().numpy() if len(clean_pred.boxes) else np.zeros((0, 4), dtype=np.float32)
    gt_classes = clean_pred.boxes.cls.detach().cpu().numpy().astype(int) if len(clean_pred.boxes) else np.zeros((0,), dtype=int)

    h, w, _ = clean_rgb.shape
    bbox_mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in gt_boxes:
        bx1, by1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        bx2, by2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        bbox_mask[by1:by2, bx1:bx2] = True

    stats_rows: List[Dict[str, Any]] = []
    overlays: List[tuple[str, np.ndarray, Dict[str, Any]]] = []

    frame_list = [frames_f54[sid] for sid in sample_ids]
    guard_results, spatial_maps = guard.evaluate_frames_batch(frame_list, return_spatial_map=True)
    assert spatial_maps is not None

    for sid, g_res, pred_pix in zip(sample_ids, guard_results, spatial_maps):
        rgb = frames_f54[sid]
        pix_full = cv2.resize(pred_pix, (w, h), interpolation=cv2.INTER_LINEAR)
        inside_mean = float(pix_full[bbox_mask].mean()) if bbox_mask.any() else float(pix_full.mean())
        outside_mean = float(pix_full[~bbox_mask].mean()) if (~bbox_mask).any() else float(pix_full.mean())
        ratio = inside_mean / max(outside_mean, 1e-6)

        row = {
            "sample_id": sid,
            "frame_idx": frame_idx,
            "pred_health": round(float(g_res.pred_health or 0.0), 6),
            "hybrid_health": round(g_res.health_score, 6),
            "top1_issue": g_res.top1_issue,
            "top1_prob": round(float(g_res.top1_prob or 0.0), 4),
            "pred_pix_mean": round(float(pix_full.mean()), 6),
            "pred_pix_std": round(float(pix_full.std()), 6),
            "pred_pix_p95": round(float(np.percentile(pix_full, 95)), 6),
            "pred_pix_max": round(float(pix_full.max()), 6),
            "inside_gt_bbox_mean": round(inside_mean, 6),
            "outside_gt_bbox_mean": round(outside_mean, 6),
            "bbox_vs_bg_ratio": round(ratio, 4),
        }
        stats_rows.append(row)
        overlay = render_spatial_overlay(rgb, pred_pix, gt_boxes, gt_classes, yolo.names)
        overlays.append((sid, overlay, row))

    csv_path = output_dir / "spatial_uncertainty_stats.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for ax, (sid, overlay, row) in zip(axes.flat, overlays):
        ax.imshow(overlay)
        ax.set_title(
            f"{sid} | pred_pix mean={row['pred_pix_mean']:.4f}, max={row['pred_pix_max']:.4f}\n"
            f"bbox/bg ratio={row['bbox_vs_bg_ratio']:.2f}x | top1={row['top1_issue']} ({row['top1_prob']:.2f})",
            fontsize=9.5,
        )
        ax.axis("off")
    fig.suptitle(
        "Spatial Uncertainty Map (pred_pix) Overlaid with Clean YOLOv8 Reference Boxes (Frame 54)",
        fontsize=12,
        fontweight="bold",
    )
    grid_path = output_dir / "spatial_uncertainty_grid.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)

    return {
        "frame_idx": frame_idx,
        "gt_box_count": int(len(gt_boxes)),
        "stats": stats_rows,
        "csv_path": str(csv_path.relative_to(ROOT)),
        "grid_path": str(grid_path.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["S01_clean", "S01_motion_blur_s1", "S01_motion_blur_s2", "S01_motion_blur_s5"],
    )
    parser.add_argument("--weights", type=Path, default=ROOT / ".lab_cache/weights/yolov8n.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pipeline_v2")
    parser.add_argument("--frame-idx", type=int, default=54)
    args = parser.parse_args()

    result = run_spatial_uncertainty_analysis(
        sample_ids=args.samples,
        output_dir=args.output_dir,
        weights_path=args.weights,
        frame_idx=args.frame_idx,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
