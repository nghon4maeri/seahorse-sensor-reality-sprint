"""Downstream YOLOv8 Object Detection Benchmark across DRIVE-C corruptions (Item 1).

Quantifies how motion blur degrades YOLOv8 detection count, confidence,
Recall@0.50, and IoU-matched mAP@0.50 proxy (against Clean pseudo-GT) across
the 8 sampled frames of each clip, exposing PerceptionHealthNet's s2 failure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.camera_guard import CameraDegradationGuard, compute_image_metrics

ADAS_COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}

DEFAULT_FRAME_INDICES = [0, 18, 36, 54, 73, 91, 109, 127]


def resolve_sample_videos(sample_ids: Sequence[str]) -> Dict[str, Path]:
    """Map sample_id -> cached MP4 path using fetch_sha256.txt."""
    manifest_lines = (ROOT / "fetch_sha256.txt").read_text(encoding="utf-8").splitlines()
    entries = [line.split(maxsplit=1)[1] for line in manifest_lines if line.strip()]
    resolved: Dict[str, Path] = {}
    for sid in sample_ids:
        if sid.endswith("_clean"):
            scenario = sid.split("_", 1)[0]
            hits = [e for e in entries if f"/clean_clips/{scenario}_" in e and e.endswith("clean.mp4")]
        else:
            hits = [e for e in entries if e.endswith(f"/{sid}.mp4")]
        if len(hits) != 1:
            raise FileNotFoundError(f"Cannot uniquely resolve {sid} in fetch_sha256.txt (hits={hits})")
        path = ROOT / ".lab_cache/data" / hits[0]
        if not path.exists():
            raise FileNotFoundError(f"Cached video missing: {path}")
        resolved[sid] = path
    return resolved


def read_frames_at_indices(video_path: Path, indices: Sequence[int]) -> List[np.ndarray]:
    """Read RGB frames at exact zero-based indices from an MP4 video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between Nx4 and Mx4 xyxy boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    xa1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    ya1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    xa2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    ya2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0.0, xa2 - xa1) * np.maximum(0.0, ya2 - ya1)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_ap50_and_recall(
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    det_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_thresh: float = 0.50,
) -> Tuple[float, float, int]:
    """Match predictions against Clean pseudo-GT boxes of same class at IoU >= 0.50."""
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return (1.0 if len(det_boxes) == 0 else 0.0), (1.0 if len(det_boxes) == 0 else 0.0), 0
    if len(det_boxes) == 0:
        return 0.0, 0.0, 0

    order = np.argsort(-det_scores)
    det_boxes = det_boxes[order]
    det_classes = det_classes[order]

    iou_mat = compute_iou_matrix(det_boxes, gt_boxes)
    matched_gt = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(len(det_boxes), dtype=np.float32)
    fp = np.zeros(len(det_boxes), dtype=np.float32)

    for i in range(len(det_boxes)):
        cls_i = det_classes[i]
        candidates = np.where((gt_classes == cls_i) & (~matched_gt))[0]
        if len(candidates) == 0:
            fp[i] = 1.0
            continue
        best_j = candidates[int(np.argmax(iou_mat[i, candidates]))]
        if iou_mat[i, best_j] >= iou_thresh:
            tp[i] = 1.0
            matched_gt[best_j] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / float(n_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # All-point interpolated AP
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap50 = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    recall50 = float(cum_tp[-1] / float(n_gt))
    return ap50, recall50, int(cum_tp[-1])


def draw_detections(
    rgb: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    names: Dict[int, str],
) -> np.ndarray:
    """Render crisp YOLOv8 bounding boxes + class confidence tags onto an RGB copy."""
    canvas = rgb.copy()
    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = (46, 204, 113) if score >= 0.45 else (243, 156, 18)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(int(cls_id), str(cls_id))} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(0, y1 - th - 6)
        cv2.rectangle(canvas, (x1, y_top), (x1 + tw + 6, y_top + th + 6), color, -1)
        cv2.putText(canvas, label, (x1 + 3, y_top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_downstream_benchmark(
    sample_ids: Sequence[str],
    output_dir: Path,
    weights_path: Path,
    conf_thresh: float = 0.25,
    use_neural_guard: bool = False,
) -> Dict[str, Any]:
    import torch
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(weights_path))
    guard = CameraDegradationGuard(use_neural=use_neural_guard)

    # Load historical/reference per-frame pred_health from stage3_small if neural guard is off
    ref_health_map: Dict[Tuple[str, int], float] = {}
    ref_per_frame = ROOT / "outputs/stage3_small/per_frame.csv"
    if ref_per_frame.exists():
        with ref_per_frame.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                ref_health_map[(r["sample_id"], int(r["frame_idx"]))] = float(r["gshi_pred"])

    videos = resolve_sample_videos(sample_ids)
    frames_by_sample: Dict[str, List[np.ndarray]] = {}
    detections_by_sample: Dict[str, List[Dict[str, Any]]] = {}

    for sid in sample_ids:
        frames = read_frames_at_indices(videos[sid], DEFAULT_FRAME_INDICES)
        frames_by_sample[sid] = frames
        bgr_frames = [cv2.cvtColor(fr, cv2.COLOR_RGB2BGR) for fr in frames]
        preds = yolo.predict(bgr_frames, device=device, conf=conf_thresh, verbose=False)
        parsed: List[Dict[str, Any]] = []
        for pred in preds:
            boxes = pred.boxes.xyxy.detach().cpu().numpy() if len(pred.boxes) else np.zeros((0, 4), dtype=np.float32)
            scores = pred.boxes.conf.detach().cpu().numpy() if len(pred.boxes) else np.zeros((0,), dtype=np.float32)
            classes = pred.boxes.cls.detach().cpu().numpy().astype(int) if len(pred.boxes) else np.zeros((0,), dtype=int)
            parsed.append({"boxes": boxes, "scores": scores, "classes": classes})
        detections_by_sample[sid] = parsed

    clean_sid = sample_ids[0]
    clean_dets = detections_by_sample[clean_sid]

    per_frame_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    annotated_f54: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []

    for sid in sample_ids:
        frames = frames_by_sample[sid]
        dets = detections_by_sample[sid]
        neural_out = guard.infer_neural_batch(frames) if use_neural_guard else None

        clip_rows: List[Dict[str, Any]] = []
        for idx_pos, frame_idx in enumerate(DEFAULT_FRAME_INDICES):
            rgb = frames[idx_pos]
            det = dets[idx_pos]
            gt = clean_dets[idx_pos]

            metrics = compute_image_metrics(rgb)
            pred_h = (
                float(neural_out["pred_health"][idx_pos])
                if neural_out is not None
                else ref_health_map.get((sid, frame_idx))
            )
            g_res = guard.score_from_metrics(
                B=metrics["B"],
                S_pct=metrics["S_pct"],
                H_bit=metrics["H_bit"],
                pred_health=pred_h,
                mean_luma=metrics["mean_luma"],
                p95_luma=metrics["p95_luma"],
                dark_ratio_pct=metrics["dark_ratio_pct"],
            )

            ap50, recall50, tp_count = compute_ap50_and_recall(
                det["boxes"], det["scores"], det["classes"], gt["boxes"], gt["classes"]
            )
            adas_mask = np.isin(det["classes"], list(ADAS_COCO_CLASSES.keys())) if len(det["classes"]) else np.zeros(0, dtype=bool)
            adas_count = int(adas_mask.sum())
            clean_count = len(gt["boxes"])
            det_count = len(det["boxes"])
            drop_pct = 0.0 if clean_count == 0 else float((clean_count - det_count) / clean_count * 100.0)
            mean_conf = float(det["scores"].mean()) if det_count > 0 else 0.0
            max_conf = float(det["scores"].max()) if det_count > 0 else 0.0

            row = {
                "sample_id": sid,
                "frame_idx": frame_idx,
                "detection_count": det_count,
                "adas_detection_count": adas_count,
                "clean_gt_count": clean_count,
                "tp_matched_count": tp_count,
                "object_drop_pct": round(drop_pct, 2),
                "mean_confidence": round(mean_conf, 4),
                "max_confidence": round(max_conf, 4),
                "map50_proxy": round(ap50, 4),
                "recall_at_50": round(recall50, 4),
                "B": round(metrics["B"], 4),
                "pred_health": None if pred_h is None else round(pred_h, 6),
                "hybrid_health": round(g_res.health_score, 6),
                "system_state": g_res.system_state,
                "camera_weight": round(g_res.camera_weight, 4),
                "early_warning_flag": g_res.early_warning_flag,
            }
            per_frame_rows.append(row)
            clip_rows.append(row)

            if frame_idx == 54:
                vis = draw_detections(rgb, det["boxes"], det["scores"], det["classes"], yolo.names)
                annotated_f54.append((sid, vis, row))

        clean_summary_conf = float(np.mean([r["mean_confidence"] for r in per_frame_rows[: len(DEFAULT_FRAME_INDICES)]]))
        mean_det = float(np.mean([r["detection_count"] for r in clip_rows]))
        clean_mean_det = float(np.mean([r["clean_gt_count"] for r in clip_rows]))
        mean_conf_clip = float(np.mean([r["mean_confidence"] for r in clip_rows]))
        f54_row = next(r for r in clip_rows if r["frame_idx"] == 54)
        summary_rows.append(
            {
                "sample_id": sid,
                "frames_evaluated": len(clip_rows),
                "mean_detection_count": round(mean_det, 3),
                "total_detections_8f": int(sum(r["detection_count"] for r in clip_rows)),
                "clean_total_detections_8f": int(sum(r["clean_gt_count"] for r in clip_rows)),
                "object_drop_pct_8f": round(
                    0.0 if clean_mean_det == 0 else (clean_mean_det - mean_det) / clean_mean_det * 100.0, 2
                ),
                "mean_confidence_8f": round(mean_conf_clip, 4),
                "confidence_drop_pct_8f": round(
                    0.0 if clean_summary_conf == 0 else (clean_summary_conf - mean_conf_clip) / clean_summary_conf * 100.0,
                    2,
                ),
                "mean_map50_proxy_8f": round(float(np.mean([r["map50_proxy"] for r in clip_rows])), 4),
                "mean_recall50_8f": round(float(np.mean([r["recall_at_50"] for r in clip_rows])), 4),
                "f54_detection_count": f54_row["detection_count"],
                "f54_mean_confidence": f54_row["mean_confidence"],
                "f54_map50_proxy": f54_row["map50_proxy"],
                "f54_B": f54_row["B"],
                "f54_pred_health": f54_row["pred_health"],
                "mean8_pred_health": round(float(np.mean([r["pred_health"] for r in clip_rows if r["pred_health"] is not None])), 6),
                "f54_hybrid_health": f54_row["hybrid_health"],
                "mean8_hybrid_health": round(float(np.mean([r["hybrid_health"] for r in clip_rows])), 6),
                "mean8_camera_weight": round(float(np.mean([r["camera_weight"] for r in clip_rows])), 4),
                "early_warning_triggered": any(r["early_warning_flag"] for r in clip_rows),
            }
        )

    per_frame_csv = output_dir / "downstream_yolov8_per_frame.csv"
    summary_csv = output_dir / "downstream_yolov8_summary.csv"
    write_csv(per_frame_csv, per_frame_rows)
    write_csv(summary_csv, summary_rows)

    # Render 2x2 YOLOv8 detection comparison on frame 54
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for ax, (sid, vis, row) in zip(axes.flat, annotated_f54):
        ax.imshow(vis)
        ax.set_title(
            f"{sid} | det={row['detection_count']} (mAP50_proxy={row['map50_proxy']:.2f}, "
            f"conf={row['mean_confidence']:.2f})\n"
            f"Neural={row['pred_health']:.4f} vs Hybrid={row['hybrid_health']:.4f} ({row['system_state']})",
            fontsize=9.5,
        )
        ax.axis("off")
    fig.suptitle("YOLOv8n Downstream Detection vs Neural Health & Hybrid Guard (S01 Frame 54)", fontsize=12, fontweight="bold")
    grid_path = output_dir / "yolov8_detection_comparison_f54.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)

    return {
        "device": device,
        "weights": str(weights_path.relative_to(ROOT)),
        "conf_thresh": conf_thresh,
        "summary": summary_rows,
        "per_frame_csv": str(per_frame_csv.relative_to(ROOT)),
        "summary_csv": str(summary_csv.relative_to(ROOT)),
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
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--use-neural-guard", action="store_true")
    args = parser.parse_args()

    result = run_downstream_benchmark(
        sample_ids=args.samples,
        output_dir=args.output_dir,
        weights_path=args.weights,
        conf_thresh=args.conf,
        use_neural_guard=args.use_neural_guard,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
