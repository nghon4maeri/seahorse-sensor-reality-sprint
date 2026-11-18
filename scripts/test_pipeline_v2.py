"""Unit & integration tests for CameraDegradationGuard and downstream metrics."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.camera_guard import (
    CameraDegradationGuard,
    HybridHealthConfig,
    compute_dynamic_sensor_weight,
    compute_image_metrics,
)
from scripts.downstream_yolov8_benchmark import compute_ap50_and_recall


class TestCameraGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = CameraDegradationGuard(use_neural=False)
        self.cfg = HybridHealthConfig()

    def test_s01_motion_blur_monotonic_and_fixes_s2(self) -> None:
        # Observed values from outputs/stage3_small/benchmark_summary.csv
        samples = [
            ("clean", 3544.1384, 1.6558, 7.5152, 0.148770),
            ("s1", 292.6951, 1.0954, 7.4132, 0.275987),
            ("s2", 199.4923, 0.9299, 7.4035, 0.407456),
            ("s5", 14.2645, 0.4564, 7.3848, 0.076573),
        ]
        results = [
            self.guard.score_from_metrics(B=b, S_pct=s, H_bit=h, pred_health=ph)
            for _, b, s, h, ph in samples
        ]
        scores = [r.health_score for r in results]
        self.assertEqual(results[0].system_state, "HEALTHY")
        self.assertTrue(all(a > b for a, b in zip(scores, scores[1:])), f"Not strictly decreasing: {scores}")
        self.assertTrue(results[2].discordance_triggered)
        self.assertTrue(results[2].early_warning_flag)
        self.assertFalse(results[0].early_warning_flag)

    def test_day_night_adaptive_separates_night_clean_from_day_underexposure(self) -> None:
        night_clean = self.guard.score_from_metrics(
            B=86.3, S_pct=49.7, H_bit=4.55, pred_health=0.2794, adaptive_day_night=True
        )
        night_fixed = self.guard.score_from_metrics(
            B=86.3, S_pct=49.7, H_bit=4.55, pred_health=0.2794, adaptive_day_night=False
        )
        day_underexp_s5 = self.guard.score_from_metrics(
            B=345.1, S_pct=1.8, H_bit=5.98, pred_health=0.0112, adaptive_day_night=True
        )
        self.assertEqual(night_fixed.system_state, "CRITICAL")
        self.assertEqual(night_clean.ambient_regime, "night")
        self.assertEqual(night_clean.system_state, "HEALTHY")
        self.assertEqual(day_underexp_s5.ambient_regime, "day")
        self.assertEqual(day_underexp_s5.system_state, "CRITICAL")

    def test_dynamic_sensor_weight_continuity_and_bounds(self) -> None:
        grid = np.linspace(0.0, 1.0, 201)
        weights = [compute_dynamic_sensor_weight(float(h), self.cfg) for h in grid]
        self.assertEqual(weights[0], 0.0)
        self.assertEqual(weights[-1], 1.0)
        self.assertTrue(all(0.0 <= w <= 1.0 for w in weights))
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(weights, weights[1:])))

    def test_image_metrics_match_stage3_clean(self) -> None:
        img_path = ROOT / "outputs/stage3_small/S01_clean_f54.png"
        bgr = cv2.imread(str(img_path))
        self.assertIsNotNone(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = compute_image_metrics(rgb)
        self.assertAlmostEqual(m["B"], 3544.1383649406853, places=3)
        self.assertAlmostEqual(m["S_pct"], 1.6558159722222223, places=4)
        self.assertAlmostEqual(m["H_bit"], 7.515182938623269, places=4)

    def test_ap50_and_recall_computation(self) -> None:
        gt_boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32)
        gt_classes = np.array([2, 0], dtype=int)
        # Perfect match
        ap, rec, tp = compute_ap50_and_recall(gt_boxes, np.array([0.9, 0.8]), gt_classes, gt_boxes, gt_classes)
        self.assertAlmostEqual(ap, 1.0)
        self.assertAlmostEqual(rec, 1.0)
        self.assertEqual(tp, 2)
        # Missing one object
        ap2, rec2, tp2 = compute_ap50_and_recall(gt_boxes[:1], np.array([0.9]), gt_classes[:1], gt_boxes, gt_classes)
        self.assertAlmostEqual(rec2, 0.5)
        self.assertEqual(tp2, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
