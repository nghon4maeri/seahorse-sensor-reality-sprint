//! Zero-allocation parallel handcrafted optical metrics matching OpenCV.

use rayon::prelude::*;

use crate::types::OpticalMetrics;

const CHUNK_ROWS: usize = 32;

/// OpenCV fixed-point RGB -> Grayscale weights (`4899 + 9617 + 1868 == 16384`).
#[inline(always)]
fn rgb_to_gray_u8(r: u8, g: u8, b: u8) -> u8 {
    (((r as u32) * 4899 + (g as u32) * 9617 + (b as u32) * 1868 + 8192) >> 14) as u8
}

#[inline(always)]
fn reflect_101(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    if idx < 0 {
        (-idx) as usize
    } else if idx >= len as isize {
        (2 * len as isize - 2 - idx) as usize
    } else {
        idx as usize
    }
}

#[inline]
fn fill_gray_row(rgb: &[u8], width: usize, y: usize, out: &mut [u8]) {
    let row_bytes = width * 3;
    let src = &rgb[y * row_bytes..(y + 1) * row_bytes];
    for (dst, px) in out.iter_mut().zip(src.chunks_exact(3)) {
        *dst = rgb_to_gray_u8(px[0], px[1], px[2]);
    }
}

#[derive(Clone)]
struct StripAccumulator {
    hist: [u64; 256],
    sum_l: i64,
    sum_l2: i64,
}

impl Default for StripAccumulator {
    fn default() -> Self {
        Self {
            hist: [0u64; 256],
            sum_l: 0,
            sum_l2: 0,
        }
    }
}

struct WorkerScratch {
    row_prev: Vec<u8>,
    row_curr: Vec<u8>,
    row_next: Vec<u8>,
}

impl WorkerScratch {
    fn new(width: usize) -> Self {
        Self {
            row_prev: vec![0u8; width],
            row_curr: vec![0u8; width],
            row_next: vec![0u8; width],
        }
    }
}

#[inline]
fn process_row_laplacian_and_hist(
    prev: &[u8],
    curr: &[u8],
    next: &[u8],
    acc: &mut StripAccumulator,
) {
    let width = curr.len();
    for &g in curr {
        acc.hist[g as usize] += 1;
    }

    if width == 0 {
        return;
    }
    if width == 1 {
        let l = (prev[0] as i32) + (next[0] as i32) + 2 * (curr[0] as i32) - 4 * (curr[0] as i32);
        acc.sum_l += l as i64;
        acc.sum_l2 += (l as i64) * (l as i64);
        return;
    }

    // Left boundary (BORDER_REFLECT_101: left neighbor is x = 1)
    let l0 = (prev[0] as i32) + (next[0] as i32) + 2 * (curr[1] as i32) - 4 * (curr[0] as i32);
    let mut local_sum = l0 as i64;
    let mut local_sum2 = (l0 as i64) * (l0 as i64);

    // Interior columns: auto-vectorizable contiguous slice scan
    for x in 1..(width - 1) {
        let l = (prev[x] as i32)
            + (next[x] as i32)
            + (curr[x - 1] as i32)
            + (curr[x + 1] as i32)
            - 4 * (curr[x] as i32);
        let l64 = l as i64;
        local_sum += l64;
        local_sum2 += l64 * l64;
    }

    // Right boundary (BORDER_REFLECT_101: right neighbor is x = width - 2)
    let rx = width - 1;
    let lr =
        (prev[rx] as i32) + (next[rx] as i32) + 2 * (curr[rx - 1] as i32) - 4 * (curr[rx] as i32);
    local_sum += lr as i64;
    local_sum2 += (lr as i64) * (lr as i64);

    acc.sum_l += local_sum;
    acc.sum_l2 += local_sum2;
}

/// Compute 95th percentile (linear interpolation matching `numpy.percentile`) from 256-bin histogram.
fn percentile_from_hist(hist: &[u64; 256], total: u64, pct: f64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let pos = (pct / 100.0) * ((total - 1) as f64);
    let k0 = pos.floor() as u64;
    let k1 = pos.ceil() as u64;
    let frac = pos - (k0 as f64);

    let mut v0 = 0usize;
    let mut v1 = 0usize;
    let mut cum = 0u64;
    let mut found0 = false;
    for (bin, &count) in hist.iter().enumerate() {
        if count == 0 {
            continue;
        }
        cum += count;
        if !found0 && cum > k0 {
            v0 = bin;
            found0 = true;
        }
        if cum > k1 {
            v1 = bin;
            break;
        }
    }
    (v0 as f64) * (1.0 - frac) + (v1 as f64) * frac
}

/// Compute all optical metrics in a single parallel pass over a contiguous RGB24 buffer.
pub fn compute_optical_metrics(rgb: &[u8], height: usize, width: usize) -> Result<OpticalMetrics, String> {
    if height == 0 || width == 0 {
        return Err(format!("Image dimensions must be non-zero, got {height}x{width}"));
    }
    let expected_len = height
        .checked_mul(width)
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| "Image byte size overflow".to_string())?;
    if rgb.len() != expected_len {
        return Err(format!(
            "RGB buffer length mismatch: expected {expected_len} bytes for {height}x{width}x3, got {}",
            rgb.len()
        ));
    }

    let num_chunks = (height + CHUNK_ROWS - 1) / CHUNK_ROWS;
    let combined = (0..num_chunks)
        .into_par_iter()
        .map_init(
            || WorkerScratch::new(width),
            |scratch, chunk_idx| {
                let y_start = chunk_idx * CHUNK_ROWS;
                let y_end = (y_start + CHUNK_ROWS).min(height);
                let mut acc = StripAccumulator::default();

                fill_gray_row(
                    rgb,
                    width,
                    reflect_101(y_start as isize - 1, height),
                    &mut scratch.row_prev,
                );
                fill_gray_row(rgb, width, y_start, &mut scratch.row_curr);

                for y in y_start..y_end {
                    let y_next = reflect_101(y as isize + 1, height);
                    fill_gray_row(rgb, width, y_next, &mut scratch.row_next);
                    process_row_laplacian_and_hist(
                        &scratch.row_prev,
                        &scratch.row_curr,
                        &scratch.row_next,
                        &mut acc,
                    );
                    std::mem::swap(&mut scratch.row_prev, &mut scratch.row_curr);
                    std::mem::swap(&mut scratch.row_curr, &mut scratch.row_next);
                }
                acc
            },
        )
        .reduce(StripAccumulator::default, |mut a, b| {
            for i in 0..256 {
                a.hist[i] += b.hist[i];
            }
            a.sum_l += b.sum_l;
            a.sum_l2 += b.sum_l2;
            a
        });

    let total_px = (height * width) as u64;
    let total_f = total_px as f64;

    let mean_l = (combined.sum_l as f64) / total_f;
    let lap_var = ((combined.sum_l2 as f64) / total_f - mean_l * mean_l).max(0.0);

    let mut sat_count: u64 = 0;
    let mut dark15_count: u64 = 0;
    let mut luma_sum: u64 = 0;
    let mut h_bit: f64 = 0.0;

    for (val, &cnt) in combined.hist.iter().enumerate() {
        if cnt == 0 {
            continue;
        }
        if val <= 5 || val >= 250 {
            sat_count += cnt;
        }
        if val <= 15 {
            dark15_count += cnt;
        }
        luma_sum += (val as u64) * cnt;
        let p = (cnt as f64) / total_f;
        h_bit -= p * p.log2();
    }

    let s_pct = (sat_count as f64) / total_f * 100.0;
    let dark_ratio_pct = (dark15_count as f64) / total_f * 100.0;
    let mean_luma = (luma_sum as f64) / total_f;
    let p95_luma = percentile_from_hist(&combined.hist, total_px, 95.0);

    Ok(OpticalMetrics {
        b: lap_var,
        s_pct,
        h_bit,
        mean_luma,
        p95_luma,
        dark_ratio_pct,
    })
}
