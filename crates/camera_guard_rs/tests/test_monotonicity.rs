//! Validate monotonic degradation on S01 & S06 and Day/Night separation.

use std::path::PathBuf;

use camera_guard_rs::{
    compute_dynamic_sensor_weight, evaluate_rgb_frame, score_from_metrics, AmbientRegime,
    GuardConfig, OpticalMetrics, SystemState,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn s01_motion_blur_is_strictly_monotonic_and_flags_s2() {
    let cfg = GuardConfig::default();
    let cases = [
        ("S01_clean_f54.png", 0.148770, SystemState::Healthy, false),
        ("S01_motion_blur_s1_f54.png", 0.275987, SystemState::Critical, true),
        ("S01_motion_blur_s2_f54.png", 0.407456, SystemState::Critical, true),
        ("S01_motion_blur_s5_f54.png", 0.076573, SystemState::Critical, true),
    ];

    let mut scores = Vec::new();
    for (name, pred_h, expected_state, expected_disc) in cases {
        let img = image::open(repo_root().join("outputs/stage3_small").join(name))
            .unwrap()
            .to_rgb8();
        let res = evaluate_rgb_frame(
            img.as_raw(),
            img.height() as usize,
            img.width() as usize,
            Some(pred_h),
            true,
            &cfg,
        )
        .unwrap();
        assert_eq!(res.system_state, expected_state, "{name}");
        assert_eq!(res.discordance_triggered, expected_disc, "{name}");
        scores.push(res.health_score);
    }

    assert!(
        scores.windows(2).all(|w| w[0] > w[1]),
        "Expected strictly decreasing scores Clean > s1 > s2 > s5, got {scores:?}"
    );
}

#[test]
fn s06_night_clean_stays_healthy_and_night_series_is_monotonic() {
    let cfg = GuardConfig::default();

    // S06 night clean metrics (from DRIVE-C S06_clean frame 54)
    let s06_clean = OpticalMetrics {
        b: 86.3,
        s_pct: 49.7,
        h_bit: 4.55,
        mean_luma: 35.0,
        p95_luma: 150.0,
        dark_ratio_pct: 49.7,
    };

    let fixed = score_from_metrics(&s06_clean, Some(0.2794), false, &cfg);
    assert_eq!(fixed.system_state, SystemState::Critical);

    let adaptive = score_from_metrics(&s06_clean, Some(0.2794), true, &cfg);
    assert_eq!(adaptive.ambient_regime, AmbientRegime::Night);
    assert_eq!(adaptive.system_state, SystemState::Healthy);
    assert!(!adaptive.early_warning_flag);

    // S06 motion blur s0..s5
    let s06_blur_series = [
        (86.3, 49.7, 4.55, 0.2794),
        (18.5, 49.4, 4.58, 0.1672),
        (6.3, 49.3, 4.59, 0.0923),
        (5.8, 49.0, 4.59, 0.0766),
        (5.4, 49.0, 4.59, 0.0784),
        (2.9, 48.6, 4.61, 0.0629),
    ];
    let scores: Vec<f64> = s06_blur_series
        .iter()
        .map(|&(b, s, h, ph)| {
            let m = OpticalMetrics {
                b,
                s_pct: s,
                h_bit: h,
                mean_luma: 35.0,
                p95_luma: 150.0,
                dark_ratio_pct: s,
            };
            score_from_metrics(&m, Some(ph), true, &cfg).health_score
        })
        .collect();
    assert!(
        scores.windows(2).all(|w| w[0] > w[1]),
        "S06 night blur not strictly decreasing: {scores:?}"
    );
}

#[test]
fn dynamic_sensor_weight_is_continuous_and_bounded() {
    let cfg = GuardConfig::default();
    let mut prev = -1.0_f64;
    for i in 0..=200 {
        let h = (i as f64) / 200.0;
        let w = compute_dynamic_sensor_weight(h, &cfg);
        assert!((0.0..=1.0).contains(&w));
        assert!(w >= prev - 1e-12);
        prev = w;
    }
}
