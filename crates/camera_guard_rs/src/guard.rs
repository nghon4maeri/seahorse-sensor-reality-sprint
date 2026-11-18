//! Hybrid Health Engine, Discordance Detector, and Dynamic Sensor Fusion Weighting.

use crate::ambient::estimate_ambient;
use crate::metrics::compute_optical_metrics;
use crate::types::{AmbientEstimate, AmbientRegime, GuardConfig, GuardResult, OpticalMetrics, SystemState};

/// Map Hybrid Health Score `[0.0, 1.0]` to smooth sensor fusion weight `[0.0, 1.0]`.
#[inline]
pub fn compute_dynamic_sensor_weight(health_score: f64, cfg: &GuardConfig) -> f64 {
    let h = health_score.clamp(0.0, 1.0);
    if h >= cfg.healthy_threshold {
        return 1.0;
    }
    if h >= cfg.degraded_threshold {
        let t = (h - cfg.degraded_threshold) / (cfg.healthy_threshold - cfg.degraded_threshold);
        let smooth_t = t * t * (3.0 - 2.0 * t);
        return cfg.degraded_min_weight + (1.0 - cfg.degraded_min_weight) * smooth_t;
    }
    let t = h / cfg.degraded_threshold;
    cfg.degraded_min_weight * t.powf(1.5)
}

/// Classify frame health into HEALTHY / DEGRADED / CRITICAL.
#[inline]
pub fn classify_system_state(health_score: f64, cfg: &GuardConfig) -> SystemState {
    if health_score >= cfg.healthy_threshold {
        SystemState::Healthy
    } else if health_score >= cfg.degraded_threshold {
        SystemState::Degraded
    } else {
        SystemState::Critical
    }
}

/// Evaluate Hybrid Health Score from precomputed optical metrics and optional neural health.
pub fn score_from_metrics(
    metrics: &OpticalMetrics,
    pred_health: Option<f64>,
    adaptive_day_night: bool,
    cfg: &GuardConfig,
) -> GuardResult {
    let ambient = if adaptive_day_night {
        estimate_ambient(metrics, cfg)
    } else {
        AmbientEstimate {
            regime: AmbientRegime::DayFixed,
            daylight_factor: 1.0,
            b_baseline: cfg.day_b_baseline,
            s_baseline_pct: cfg.day_s_baseline_pct,
            h_baseline_bit: cfg.day_h_baseline_bit,
            mean_luma: metrics.mean_luma,
            p95_luma: metrics.p95_luma,
            dark_ratio_pct: metrics.dark_ratio_pct,
        }
    };

    let b_norm = (metrics.b / ambient.b_baseline.max(1e-6)).clamp(0.0, 1.0);

    let (base_score, delta_discordance, discordance_triggered, discordance_factor, pred_clamped) =
        match pred_health {
            None => (b_norm, 0.0, false, 1.0, None),
            Some(ph) => {
                let neural_val = ph.clamp(0.0, 1.0);
                let delta = (neural_val - b_norm).max(0.0);
                let triggered = delta > cfg.discordance_tau;
                let disc_factor = if triggered {
                    (1.0 - cfg.discordance_gamma * delta).clamp(0.05, 1.0)
                } else {
                    1.0
                };
                let base = cfg.w_blur * b_norm + cfg.w_neural * neural_val;
                (base, delta, triggered, disc_factor, Some(neural_val))
            }
        };

    let s_excess =
        (metrics.s_pct - ambient.s_baseline_pct).max(0.0) / (100.0 - ambient.s_baseline_pct).max(1.0);
    let h_norm = (metrics.h_bit / ambient.h_baseline_bit.max(1e-6)).clamp(0.0, 1.0);
    let sat_factor = (1.0 - cfg.alpha_saturation * s_excess).clamp(0.0, 1.0);
    let ent_factor = (1.0 - cfg.alpha_entropy * (1.0 - h_norm)).clamp(0.0, 1.0);
    let quality_penalty = sat_factor * ent_factor;

    let health_score = (base_score * discordance_factor * quality_penalty).clamp(0.0, 1.0);
    let system_state = classify_system_state(health_score, cfg);
    let camera_weight = compute_dynamic_sensor_weight(health_score, cfg);
    let early_warning_flag = (system_state != SystemState::Healthy) || discordance_triggered;

    GuardResult {
        health_score,
        system_state,
        camera_weight,
        early_warning_flag,
        b: metrics.b,
        b_norm,
        s_pct: metrics.s_pct,
        h_bit: metrics.h_bit,
        ambient_regime: ambient.regime,
        daylight_factor: ambient.daylight_factor,
        b_baseline_dynamic: ambient.b_baseline,
        mean_luma: ambient.mean_luma,
        p95_luma: ambient.p95_luma,
        dark_ratio_pct: ambient.dark_ratio_pct,
        pred_health: pred_clamped,
        delta_discordance,
        discordance_triggered,
        discordance_factor,
        quality_penalty,
    }
}

/// Evaluate a contiguous RGB24 frame buffer end-to-end.
pub fn evaluate_rgb_frame(
    rgb: &[u8],
    height: usize,
    width: usize,
    pred_health: Option<f64>,
    adaptive_day_night: bool,
    cfg: &GuardConfig,
) -> Result<GuardResult, String> {
    let metrics = compute_optical_metrics(rgb, height, width)?;
    Ok(score_from_metrics(&metrics, pred_health, adaptive_day_night, cfg))
}
