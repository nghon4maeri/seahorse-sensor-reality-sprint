"""CameraDegradationGuard: Hybrid ADAS camera health monitor with day/night adaptation.

Combines handcrafted optical/exposure statistics (Laplacian blur variance B,
saturation clipping S_pct, Shannon entropy H_bit), ambient luminance estimation,
and PerceptionHealthNet neural outputs (pred_health + discordance detection)
into a monotonic Hybrid Health Score, dynamic sensor-fusion weight, and
early-warning flag.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import camera_guard_rs as _rust_guard  # type: ignore
except ImportError:
    _rust_guard = None

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HybridHealthConfig:
    """Configuration for Hybrid Health Score and ADAS Sensor Safety Guard."""

    # Linear blend weights (B_norm vs neural pred_health)
    w_blur: float = 0.85
    w_neural: float = 0.15

    # Discordance detector: triggers when pred_health - B_norm > tau
    discordance_tau: float = 0.05
    discordance_gamma: float = 1.50

    # Saturation & entropy quality penalties
    alpha_saturation: float = 0.65
    alpha_entropy: float = 0.50

    # Day / Night baseline profiles (derived from DRIVE-C S01 day & S06 night clean)
    day_B_baseline: float = 3544.1384
    night_B_baseline: float = 86.30
    day_S_baseline_pct: float = 5.0
    night_S_baseline_pct: float = 52.0
    day_H_baseline_bit: float = 7.40
    night_H_baseline_bit: float = 4.50

    # System state thresholds (Item 5)
    healthy_threshold: float = 0.75
    degraded_threshold: float = 0.35

    # Minimum fusion weight at the DEGRADED/CRITICAL boundary
    degraded_min_weight: float = 0.25


@dataclass
class AmbientEstimate:
    """Estimated ambient lighting regime and dynamic reference baselines."""

    regime: str  # "day", "twilight", or "night"
    daylight_factor: float  # [0.0 = night, 1.0 = day]
    B_baseline: float
    S_baseline_pct: float
    H_baseline_bit: float
    mean_luma: float
    p95_luma: float
    dark_ratio_pct: float


@dataclass
class GuardResult:
    """Output packet produced by CameraDegradationGuard for a single frame."""

    health_score: float
    system_state: str  # "HEALTHY", "DEGRADED", "CRITICAL"
    camera_weight: float  # Smooth [0.0, 1.0] weight for Sensor Fusion / Kalman filter
    early_warning_flag: bool

    # Optical & information metrics
    B: float
    B_norm: float
    S_pct: float
    H_bit: float

    # Ambient adaptation
    ambient_regime: str
    daylight_factor: float
    B_baseline_dynamic: float
    mean_luma: float
    dark_ratio_pct: float

    # Neural & discordance telemetry
    pred_health: Optional[float]
    delta_discordance: float
    discordance_triggered: bool
    discordance_factor: float
    quality_penalty: float
    top1_issue: Optional[str] = None
    top1_prob: Optional[float] = None
    pred_pix_mean: Optional[float] = None
    pred_pix_p95: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_image_metrics(rgb: np.ndarray) -> Dict[str, float]:
    """Compute handcrafted metrics matching configs/benchmark.json + luminance stats.

    Uses the zero-copy Rust extension (`camera_guard_rs`) when built via maturin,
    falling back seamlessly to OpenCV/NumPy otherwise.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {rgb.shape}")
    if _rust_guard is not None and rgb.dtype == np.uint8:
        return dict(_rust_guard.compute_metrics(rgb))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var(ddof=0))
    s_pct = float(np.mean((gray <= 5) | (gray >= 250)) * 100.0)
    hist = np.bincount(gray.ravel(), minlength=256) / gray.size
    hist = hist[hist > 0]
    h_bit = float(-np.sum(hist * np.log2(hist)))
    mean_luma = float(np.mean(gray))
    p95_luma = float(np.percentile(gray, 95))
    dark_ratio_pct = float(np.mean(gray <= 15) * 100.0)
    return {
        "B": lap_var,
        "S_pct": s_pct,
        "H_bit": h_bit,
        "mean_luma": mean_luma,
        "p95_luma": p95_luma,
        "dark_ratio_pct": dark_ratio_pct,
    }


class AmbientLuminanceEstimator:
    """Distinguish daytime/twilight/nighttime scenes from intensity and entropy.

    Uses both dark-pixel fraction (night scenes typically have >=25% near-black
    sky/road pixels) and Shannon entropy / mean luma so severe daytime underexposure
    is not misclassified as a clean night scene.
    """

    def __init__(self, cfg: HybridHealthConfig):
        self.cfg = cfg

    def estimate(
        self,
        H_bit: float,
        mean_luma: Optional[float] = None,
        p95_luma: Optional[float] = None,
        dark_ratio_pct: Optional[float] = None,
        S_pct: Optional[float] = None,
    ) -> AmbientEstimate:
        # Fallback proxy when operating solely on summary CSV columns (B, S_pct, H_bit)
        if dark_ratio_pct is None:
            dark_ratio_pct = float(S_pct) if S_pct is not None else 0.0
        if mean_luma is None:
            mean_luma = 125.0 if H_bit >= 5.5 else 35.0
        if p95_luma is None:
            p95_luma = 240.0 if H_bit >= 5.5 else 150.0

        # Night scenes exhibit high dark-pixel ratio (~45-87%) AND lower entropy (<=5.2 bits).
        # Severe daytime underexposure lowers luma/entropy (~5.98 bits) while keeping dark_ratio low (<15%).
        dark_day_score = float(np.clip(1.0 - (dark_ratio_pct - 12.0) / 28.0, 0.0, 1.0))
        entropy_day_score = float(np.clip((H_bit - 4.70) / 1.15, 0.0, 1.0))
        luma_day_score = float(np.clip((mean_luma - 28.0) / 55.0, 0.0, 1.0))

        # Weighted geometric/arithmetic blend: dark_ratio & entropy prevent daytime underexposure confusion
        daylight_factor = float(
            np.clip(0.45 * dark_day_score + 0.40 * entropy_day_score + 0.15 * luma_day_score, 0.0, 1.0)
        )

        # Log-linear interpolation of Laplacian variance baseline across illuminance orders
        log_day = np.log(max(self.cfg.day_B_baseline, 1.0))
        log_night = np.log(max(self.cfg.night_B_baseline, 1.0))
        b_baseline = float(np.exp(daylight_factor * log_day + (1.0 - daylight_factor) * log_night))

        s_baseline = float(
            daylight_factor * self.cfg.day_S_baseline_pct
            + (1.0 - daylight_factor) * self.cfg.night_S_baseline_pct
        )
        h_baseline = float(
            daylight_factor * self.cfg.day_H_baseline_bit
            + (1.0 - daylight_factor) * self.cfg.night_H_baseline_bit
        )

        if daylight_factor >= 0.70:
            regime = "day"
        elif daylight_factor <= 0.30:
            regime = "night"
        else:
            regime = "twilight"

        return AmbientEstimate(
            regime=regime,
            daylight_factor=daylight_factor,
            B_baseline=b_baseline,
            S_baseline_pct=s_baseline,
            H_baseline_bit=h_baseline,
            mean_luma=float(mean_luma),
            p95_luma=float(p95_luma),
            dark_ratio_pct=float(dark_ratio_pct),
        )


def compute_dynamic_sensor_weight(health_score: float, cfg: HybridHealthConfig) -> float:
    """Map Hybrid Health Score -> smooth [0.0, 1.0] Sensor Fusion weight (no hard drop)."""
    h = float(np.clip(health_score, 0.0, 1.0))
    if h >= cfg.healthy_threshold:
        return 1.0
    if h >= cfg.degraded_threshold:
        t = (h - cfg.degraded_threshold) / (cfg.healthy_threshold - cfg.degraded_threshold)
        smooth_t = t * t * (3.0 - 2.0 * t)
        return float(cfg.degraded_min_weight + (1.0 - cfg.degraded_min_weight) * smooth_t)
    t = h / cfg.degraded_threshold
    return float(cfg.degraded_min_weight * (t ** 1.5))


def classify_system_state(health_score: float, cfg: HybridHealthConfig) -> str:
    """Classify frame health into HEALTHY / DEGRADED / CRITICAL."""
    if health_score >= cfg.healthy_threshold:
        return "HEALTHY"
    if health_score >= cfg.degraded_threshold:
        return "DEGRADED"
    return "CRITICAL"


class CameraDegradationGuard:
    """ADAS camera reliability monitor combining physical proxies and PerceptionHealthNet."""

    def __init__(
        self,
        cfg: Optional[HybridHealthConfig] = None,
        use_neural: bool = True,
        source_dir: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
        input_hw: Tuple[int, int] = (384, 1280),
    ):
        self.cfg = cfg or HybridHealthConfig()
        self.ambient_estimator = AmbientLuminanceEstimator(self.cfg)
        self.source_dir = source_dir or (ROOT / ".lab_cache/drive-c-source")
        self.checkpoint_path = checkpoint_path or (ROOT / "drive-c-dataset/checkpoints/epoch_021_best.pth")
        self.input_hw = input_hw

        self.model = None
        self.issues: List[str] = []
        self.device = None
        self._gshi_module = None

        if use_neural:
            self._load_neural_monitor(device=device)

    def _load_neural_monitor(self, device: Optional[str] = None) -> None:
        import torch

        if not (self.source_dir / "scripts/add_gshi_pred.py").exists():
            raise FileNotFoundError(
                f"Frozen DRIVE-C source not found at {self.source_dir}. Run scripts/setup_demo.py first."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}. Run scripts/setup_demo.py first."
            )

        scripts_str = str(self.source_dir / "scripts")
        if scripts_str not in sys.path:
            sys.path.insert(0, scripts_str)
        import add_gshi_pred as G  # type: ignore

        self._gshi_module = G
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        h, w = self.input_hw
        self.model = G.load_model(self.checkpoint_path, self.device, h, w, pretrained_backbone=False)
        taxonomy = G.load_taxonomy(str(self.source_dir / "configs/taxonomy/camera_issues.yaml"))
        self.issues = list(taxonomy.issues)

    def score_from_metrics(
        self,
        B: float,
        S_pct: float,
        H_bit: float,
        pred_health: Optional[float] = None,
        mean_luma: Optional[float] = None,
        p95_luma: Optional[float] = None,
        dark_ratio_pct: Optional[float] = None,
        top1_issue: Optional[str] = None,
        top1_prob: Optional[float] = None,
        pred_pix_mean: Optional[float] = None,
        pred_pix_p95: Optional[float] = None,
        adaptive_day_night: bool = True,
    ) -> GuardResult:
        """Compute Hybrid Health Score and Guard outputs from optical & neural metrics."""
        if adaptive_day_night:
            ambient = self.ambient_estimator.estimate(
                H_bit=H_bit,
                mean_luma=mean_luma,
                p95_luma=p95_luma,
                dark_ratio_pct=dark_ratio_pct,
                S_pct=S_pct,
            )
        else:
            ambient = AmbientEstimate(
                regime="day_fixed",
                daylight_factor=1.0,
                B_baseline=self.cfg.day_B_baseline,
                S_baseline_pct=self.cfg.day_S_baseline_pct,
                H_baseline_bit=self.cfg.day_H_baseline_bit,
                mean_luma=float(mean_luma if mean_luma is not None else 127.0),
                p95_luma=float(p95_luma if p95_luma is not None else 245.0),
                dark_ratio_pct=float(dark_ratio_pct if dark_ratio_pct is not None else 2.0),
            )

        b_norm = float(np.clip(B / max(ambient.B_baseline, 1e-6), 0.0, 1.0))

        if pred_health is None:
            neural_term = b_norm
            delta_disc = 0.0
            disc_triggered = False
            disc_factor = 1.0
            base_score = b_norm
        else:
            neural_val = float(np.clip(pred_health, 0.0, 1.0))
            delta_disc = max(0.0, neural_val - b_norm)
            disc_triggered = delta_disc > self.cfg.discordance_tau
            disc_factor = (
                float(np.clip(1.0 - self.cfg.discordance_gamma * delta_disc, 0.05, 1.0))
                if disc_triggered
                else 1.0
            )
            base_score = self.cfg.w_blur * b_norm + self.cfg.w_neural * neural_val

        # Adaptive saturation & entropy penalty
        s_excess = max(0.0, S_pct - ambient.S_baseline_pct) / max(100.0 - ambient.S_baseline_pct, 1.0)
        h_norm = float(np.clip(H_bit / max(ambient.H_baseline_bit, 1e-6), 0.0, 1.0))
        sat_factor = float(np.clip(1.0 - self.cfg.alpha_saturation * s_excess, 0.0, 1.0))
        ent_factor = float(np.clip(1.0 - self.cfg.alpha_entropy * (1.0 - h_norm), 0.0, 1.0))
        quality_penalty = sat_factor * ent_factor

        health_score = float(np.clip(base_score * disc_factor * quality_penalty, 0.0, 1.0))
        system_state = classify_system_state(health_score, self.cfg)
        camera_weight = compute_dynamic_sensor_weight(health_score, self.cfg)
        early_warning_flag = (system_state != "HEALTHY") or disc_triggered

        return GuardResult(
            health_score=health_score,
            system_state=system_state,
            camera_weight=camera_weight,
            early_warning_flag=early_warning_flag,
            B=float(B),
            B_norm=b_norm,
            S_pct=float(S_pct),
            H_bit=float(H_bit),
            ambient_regime=ambient.regime,
            daylight_factor=ambient.daylight_factor,
            B_baseline_dynamic=ambient.B_baseline,
            mean_luma=ambient.mean_luma,
            dark_ratio_pct=ambient.dark_ratio_pct,
            pred_health=None if pred_health is None else float(pred_health),
            delta_discordance=float(delta_disc),
            discordance_triggered=disc_triggered,
            discordance_factor=float(disc_factor),
            quality_penalty=float(quality_penalty),
            top1_issue=top1_issue,
            top1_prob=top1_prob,
            pred_pix_mean=pred_pix_mean,
            pred_pix_p95=pred_pix_p95,
        )

    def infer_neural_batch(
        self,
        frames_rgb: Sequence[np.ndarray],
        return_spatial_map: bool = False,
    ) -> Dict[str, Any]:
        """Run PerceptionHealthNet on a batch of RGB frames, optionally keeping pred_pix."""
        if self.model is None or self._gshi_module is None:
            raise RuntimeError("Neural monitor is disabled; instantiate with use_neural=True.")
        import torch

        h, w = self.input_hw
        x_np = np.stack([self._gshi_module.preprocess_frame(fr, h, w, "resize") for fr in frames_rgb], axis=0)
        x = torch.from_numpy(x_np).to(self.device)
        with torch.inference_mode():
            out = self.model(x)

        probs = torch.sigmoid(out["logits_pres"]).detach().cpu().numpy()
        pred_sev = out["pred_sev"].detach().cpu().numpy()
        pred_health = out["pred_health"].reshape(-1).detach().cpu().numpy()
        pred_pix = out["pred_pix"][:, 0].detach().cpu().numpy() if "pred_pix" in out else None

        top1_idx = np.argmax(probs, axis=1)
        result: Dict[str, Any] = {
            "pred_health": pred_health,
            "probs": probs,
            "pred_sev": pred_sev,
            "top1_issue": [self.issues[int(i)] for i in top1_idx],
            "top1_prob": [float(probs[r, int(i)]) for r, i in enumerate(top1_idx)],
        }
        if pred_pix is not None:
            result["pred_pix_mean"] = pred_pix.mean(axis=(1, 2))
            result["pred_pix_p95"] = np.percentile(pred_pix, 95, axis=(1, 2))
            if return_spatial_map:
                result["pred_pix"] = pred_pix
        return result

    def evaluate_frames_batch(
        self,
        frames_rgb: Sequence[np.ndarray],
        adaptive_day_night: bool = True,
        return_spatial_map: bool = False,
    ) -> Tuple[List[GuardResult], Optional[np.ndarray]]:
        """Evaluate a batch of RGB frames through both handcrafted metrics and neural monitor."""
        neural = self.infer_neural_batch(frames_rgb, return_spatial_map=return_spatial_map) if self.model else None
        results: List[GuardResult] = []
        for idx, rgb in enumerate(frames_rgb):
            metrics = compute_image_metrics(rgb)
            kwargs: Dict[str, Any] = {}
            if neural is not None:
                kwargs["pred_health"] = float(neural["pred_health"][idx])
                kwargs["top1_issue"] = str(neural["top1_issue"][idx])
                kwargs["top1_prob"] = float(neural["top1_prob"][idx])
                if "pred_pix_mean" in neural:
                    kwargs["pred_pix_mean"] = float(neural["pred_pix_mean"][idx])
                    kwargs["pred_pix_p95"] = float(neural["pred_pix_p95"][idx])
            results.append(
                self.score_from_metrics(
                    B=metrics["B"],
                    S_pct=metrics["S_pct"],
                    H_bit=metrics["H_bit"],
                    mean_luma=metrics["mean_luma"],
                    p95_luma=metrics["p95_luma"],
                    dark_ratio_pct=metrics["dark_ratio_pct"],
                    adaptive_day_night=adaptive_day_night,
                    **kwargs,
                )
            )
        spatial_maps = neural.get("pred_pix") if (neural is not None and return_spatial_map) else None
        return results, spatial_maps

    def evaluate_frame(
        self,
        rgb: np.ndarray,
        adaptive_day_night: bool = True,
        return_spatial_map: bool = False,
    ) -> Tuple[GuardResult, Optional[np.ndarray]]:
        """Evaluate a single RGB frame."""
        results, maps = self.evaluate_frames_batch(
            [rgb], adaptive_day_night=adaptive_day_night, return_spatial_map=return_spatial_map
        )
        return results[0], (maps[0] if maps is not None else None)

    def evaluate_video(
        self,
        video_path: Path,
        num_frames: int = 8,
        adaptive_day_night: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate sampled frames from a video stream and return per-frame + summary telemetry."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.round(np.linspace(0, max(total_frames - 1, 0), num=min(num_frames, total_frames))).astype(int).tolist()
        frames: List[np.ndarray] = []
        valid_indices: List[int] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                valid_indices.append(idx)
        cap.release()
        if not frames:
            raise RuntimeError(f"No readable frames in video: {video_path}")

        per_frame_results, _ = self.evaluate_frames_batch(frames, adaptive_day_night=adaptive_day_night)
        mean_health = float(np.mean([r.health_score for r in per_frame_results]))
        mean_weight = float(np.mean([r.camera_weight for r in per_frame_results]))
        return {
            "video": str(video_path),
            "frame_indices": valid_indices,
            "mean_health_score": mean_health,
            "mean_system_state": classify_system_state(mean_health, self.cfg),
            "mean_camera_weight": mean_weight,
            "any_early_warning": any(r.early_warning_flag for r in per_frame_results),
            "per_frame": [r.to_dict() for r in per_frame_results],
        }


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Run CameraDegradationGuard on an RGB image or MP4 video.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to an image file")
    group.add_argument("--video", type=Path, help="Path to a video stream / MP4")
    parser.add_argument("--no-neural", action="store_true", help="Run optical-only mode without PerceptionHealthNet")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable Day/Night adaptive baseline")
    parser.add_argument("--num-frames", type=int, default=8, help="Sampled frames when evaluating a video")
    args = parser.parse_args()

    guard = CameraDegradationGuard(use_neural=not args.no_neural)
    if args.image is not None:
        bgr = cv2.imread(str(args.image))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res, _ = guard.evaluate_frame(rgb, adaptive_day_night=not args.no_adaptive)
        print(json.dumps(res.to_dict(), indent=2))
    else:
        res = guard.evaluate_video(args.video, num_frames=args.num_frames, adaptive_day_night=not args.no_adaptive)
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
