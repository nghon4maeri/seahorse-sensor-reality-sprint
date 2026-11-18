//! Core data structures for `camera_guard_rs`.

use serde::{Deserialize, Serialize};

/// System reliability classification matching ADAS Sensor Safety Guard thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SystemState {
    Healthy,
    Degraded,
    Critical,
}

impl SystemState {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            SystemState::Healthy => "HEALTHY",
            SystemState::Degraded => "DEGRADED",
            SystemState::Critical => "CRITICAL",
        }
    }
}

/// Ambient lighting regime inferred by `AmbientLuminanceEstimator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AmbientRegime {
    Day,
    Twilight,
    Night,
    DayFixed,
}

impl AmbientRegime {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            AmbientRegime::Day => "day",
            AmbientRegime::Twilight => "twilight",
            AmbientRegime::Night => "night",
            AmbientRegime::DayFixed => "day_fixed",
        }
    }
}

/// Handcrafted optical, exposure, entropy, and luminance statistics for a single frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OpticalMetrics {
    /// Variance of 2D discrete Laplacian (`cv2.Laplacian(gray, CV_64F, ksize=1).var(ddof=0)`).
    pub b: f64,
    /// Percentage of extreme dark (`<= 5`) or saturated bright (`>= 250`) pixels.
    pub s_pct: f64,
    /// Shannon entropy of the 256-bin grayscale histogram in bits.
    pub h_bit: f64,
    /// Arithmetic mean grayscale luminance `[0, 255]`.
    pub mean_luma: f64,
    /// 95th percentile grayscale luminance `[0, 255]`.
    pub p95_luma: f64,
    /// Percentage of very dark pixels (`gray <= 15`).
    pub dark_ratio_pct: f64,
}

/// Configuration parameters for the Hybrid Health Engine and ADAS Guard.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GuardConfig {
    pub w_blur: f64,
    pub w_neural: f64,
    pub discordance_tau: f64,
    pub discordance_gamma: f64,
    pub alpha_saturation: f64,
    pub alpha_entropy: f64,
    pub day_b_baseline: f64,
    pub night_b_baseline: f64,
    pub day_s_baseline_pct: f64,
    pub night_s_baseline_pct: f64,
    pub day_h_baseline_bit: f64,
    pub night_h_baseline_bit: f64,
    pub healthy_threshold: f64,
    pub degraded_threshold: f64,
    pub degraded_min_weight: f64,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            w_blur: 0.85,
            w_neural: 0.15,
            discordance_tau: 0.05,
            discordance_gamma: 1.50,
            alpha_saturation: 0.65,
            alpha_entropy: 0.50,
            day_b_baseline: 3544.1384,
            night_b_baseline: 86.30,
            day_s_baseline_pct: 5.0,
            night_s_baseline_pct: 52.0,
            day_h_baseline_bit: 7.40,
            night_h_baseline_bit: 4.50,
            healthy_threshold: 0.75,
            degraded_threshold: 0.35,
            degraded_min_weight: 0.25,
        }
    }
}

/// Estimated ambient lighting condition and dynamic baseline targets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AmbientEstimate {
    pub regime: AmbientRegime,
    pub daylight_factor: f64,
    pub b_baseline: f64,
    pub s_baseline_pct: f64,
    pub h_baseline_bit: f64,
    pub mean_luma: f64,
    pub p95_luma: f64,
    pub dark_ratio_pct: f64,
}

/// Complete safety guard telemetry packet returned per frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GuardResult {
    pub health_score: f64,
    pub system_state: SystemState,
    pub camera_weight: f64,
    pub early_warning_flag: bool,

    pub b: f64,
    pub b_norm: f64,
    pub s_pct: f64,
    pub h_bit: f64,

    pub ambient_regime: AmbientRegime,
    pub daylight_factor: f64,
    pub b_baseline_dynamic: f64,
    pub mean_luma: f64,
    pub p95_luma: f64,
    pub dark_ratio_pct: f64,

    pub pred_health: Option<f64>,
    pub delta_discordance: f64,
    pub discordance_triggered: bool,
    pub discordance_factor: f64,
    pub quality_penalty: f64,
}
