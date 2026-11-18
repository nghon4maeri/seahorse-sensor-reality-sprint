//! Verify exact numerical equivalence between Rust and Python/OpenCV on S01 PNGs.

use std::path::PathBuf;

use camera_guard_rs::compute_optical_metrics;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn matches_stage3_s01_clean_f54_metrics() {
    let img_path = repo_root().join("outputs/stage3_small/S01_clean_f54.png");
    let img = image::open(&img_path)
        .expect("Failed to open S01_clean_f54.png")
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    assert_eq!((h, w), (720, 1280));

    let m = compute_optical_metrics(img.as_raw(), h, w).expect("Metric computation failed");
    assert!(
        (m.b - 3544.1383649406853).abs() <= 0.005,
        "Laplacian variance mismatch: got {}",
        m.b
    );
    assert!(
        (m.s_pct - 1.6558159722222223).abs() <= 0.001,
        "S_pct mismatch: got {}",
        m.s_pct
    );
    assert!(
        (m.h_bit - 7.515182938623269).abs() <= 0.001,
        "H_bit mismatch: got {}",
        m.h_bit
    );
}

#[test]
fn matches_stage3_s01_blur_s1_s2_s5_metrics() {
    let cases = [
        ("S01_motion_blur_s1_f54.png", 292.6951, 1.0954, 7.4132),
        ("S01_motion_blur_s2_f54.png", 199.4923, 0.9299, 7.4035),
        ("S01_motion_blur_s5_f54.png", 14.2645, 0.4564, 7.3848),
    ];
    for (filename, exp_b, exp_s, exp_h) in cases {
        let img_path = repo_root().join("outputs/stage3_small").join(filename);
        let img = image::open(&img_path).expect("Missing stage3 PNG").to_rgb8();
        let m = compute_optical_metrics(img.as_raw(), img.height() as usize, img.width() as usize)
            .unwrap();
        assert!((m.b - exp_b).abs() <= 0.005, "{filename} B={}", m.b);
        assert!((m.s_pct - exp_s).abs() <= 0.001, "{filename} S_pct={}", m.s_pct);
        assert!((m.h_bit - exp_h).abs() <= 0.001, "{filename} H_bit={}", m.h_bit);
    }
}
