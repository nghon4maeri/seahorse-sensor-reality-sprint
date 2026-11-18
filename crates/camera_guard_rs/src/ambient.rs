//! Day/Night ambient luminance estimation and dynamic baseline interpolation.

use crate::types::{AmbientEstimate, AmbientRegime, GuardConfig, OpticalMetrics};

/// Estimate ambient lighting regime and dynamic reference baselines.
#[inline]
pub fn estimate_ambient(metrics: &OpticalMetrics, cfg: &GuardConfig) -> AmbientEstimate {
    let dark_day_score = (1.0 - (metrics.dark_ratio_pct - 12.0) / 28.0).clamp(0.0, 1.0);
    let entropy_day_score = ((metrics.h_bit - 4.70) / 1.15).clamp(0.0, 1.0);
    let luma_day_score = ((metrics.mean_luma - 28.0) / 55.0).clamp(0.0, 1.0);

    let daylight_factor =
        (0.45 * dark_day_score + 0.40 * entropy_day_score + 0.15 * luma_day_score).clamp(0.0, 1.0);

    let log_day = cfg.day_b_baseline.max(1.0).ln();
    let log_night = cfg.night_b_baseline.max(1.0).ln();
    let b_baseline = (daylight_factor * log_day + (1.0 - daylight_factor) * log_night).exp();

    let s_baseline_pct =
        daylight_factor * cfg.day_s_baseline_pct + (1.0 - daylight_factor) * cfg.night_s_baseline_pct;
    let h_baseline_bit =
        daylight_factor * cfg.day_h_baseline_bit + (1.0 - daylight_factor) * cfg.night_h_baseline_bit;

    let regime = if daylight_factor >= 0.70 {
        AmbientRegime::Day
    } else if daylight_factor <= 0.30 {
        AmbientRegime::Night
    } else {
        AmbientRegime::Twilight
    };

    AmbientEstimate {
        regime,
        daylight_factor,
        b_baseline,
        s_baseline_pct,
        h_baseline_bit,
        mean_luma: metrics.mean_luma,
        p95_luma: metrics.p95_luma,
        dark_ratio_pct: metrics.dark_ratio_pct,
    }
}
