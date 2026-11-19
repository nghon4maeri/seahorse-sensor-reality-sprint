/**
 * Main controller for Seahorse ADAS Guard Visualization UI (Phase 1 -> 3).
 */

import { renderHealthGauge, renderLineChart } from "./charts.js";
import { initImageComparator } from "./comparator.js";
import { loadAllArtifacts } from "./data-loader.js";
import { DEFAULT_GUARD_CONFIG, evaluateSeriesMonotonicity, scoreFromMetrics } from "./guard-engine.js";

const SAMPLE_META = [
  { id: "S01_clean", short: "Clean", label: "S01 Clean (0 px)", kernel: "0 px (Baseline)", img: "/outputs/stage3_small/S01_clean_f54.png" },
  { id: "S01_motion_blur_s1", short: "Blur s1", label: "Blur s1 (11 px)", kernel: "11 px · sev 0.08", img: "/outputs/stage3_small/S01_motion_blur_s1_f54.png" },
  { id: "S01_motion_blur_s2", short: "Blur s2", label: "Blur s2 (13 px · Failure)", kernel: "13 px · sev 0.18", img: "/outputs/stage3_small/S01_motion_blur_s2_f54.png" },
  { id: "S01_motion_blur_s5", short: "Blur s5", label: "Blur s5 (33 px)", kernel: "33 px · sev 0.75", img: "/outputs/stage3_small/S01_motion_blur_s5_f54.png" },
];

const FRAME_INDICES = [0, 18, 36, 54, 73, 91, 109, 127];

const state = {
  data: null,
  selectedSampleId: "S01_motion_blur_s2",
  viewMode: "split", // "split" | "yolo" | "spatial"
  selectedFrameIdx: 54,
  whatIfConfig: { ...DEFAULT_GUARD_CONFIG },
  adaptiveDayNight: true,
};

function badgeClassForState(sysState) {
  if (sysState === "HEALTHY") return "badge-healthy";
  if (sysState === "DEGRADED") return "badge-degraded";
  return "badge-critical";
}

function colorForState(sysState) {
  if (sysState === "HEALTHY") return "#22C55E";
  if (sysState === "DEGRADED") return "#F59E0B";
  return "#EF4444";
}

function formatIsoShort(isoStr) {
  if (!isoStr) return "Offline Verified";
  return isoStr.replace("T", " ").slice(0, 19);
}

function initHeaderAndKpis(data) {
  const { manifest, downstreamSummary } = data;
  const s2 = downstreamSummary.find((r) => r.sample_id === "S01_motion_blur_s2");
  const clean = downstreamSummary.find((r) => r.sample_id === "S01_clean");

  document.getElementById("provenance-text").textContent =
    `OFFLINE VERIFIED · ${manifest.gpu_name ?? "CPU"} · ${formatIsoShort(manifest.generated_at)}`;

  const bDropPct = ((1 - s2.f54_B / clean.f54_B) * 100).toFixed(2);
  document.getElementById("kpi-s2-bdrop").textContent = `-${bDropPct}%`;
  document.getElementById("kpi-s2-detail").textContent =
    `Neural: ${clean.f54_pred_health.toFixed(4)} -> ${s2.f54_pred_health.toFixed(4)} (tăng sai) | Hybrid: ${s2.f54_hybrid_health.toFixed(4)}`;

  document.getElementById("kpi-yolo-drop").textContent = `-${s2.object_drop_pct_8f.toFixed(1)}%`;
  document.getElementById("kpi-yolo-detail").textContent =
    `mAP@0.50 proxy tụt 1.000 -> ${s2.mean_map50_proxy_8f.toFixed(3)} · conf -${s2.confidence_drop_pct_8f.toFixed(1)}% (8 frame)`;

  document.getElementById("kpi-daynight-val").textContent = "4 / 4 PASS";
  document.getElementById("kpi-daynight-detail").textContent =
    `S06 Night Clean: Fixed CRITICAL -> Adaptive HEALTHY (${data.manifest.day_night_adaptive.S06_clean_adaptive_health.toFixed(4)})`;
}

function updateInspector() {
  const { stage3Summary, downstreamSummary, spatialStats } = state.data;
  const meta = SAMPLE_META.find((m) => m.id === state.selectedSampleId) ?? SAMPLE_META[2];
  const st3 = stage3Summary.find((r) => r.sample_id === meta.id);
  const ds = downstreamSummary.find((r) => r.sample_id === meta.id);
  const sp = spatialStats.find((r) => r.sample_id === meta.id);

  // Sync sample buttons
  document.querySelectorAll("[data-sample-id]").forEach((btn) => {
    btn.setAttribute("aria-pressed", String(btn.dataset.sampleId === meta.id));
  });

  // Sync view mode visibility
  document.querySelectorAll("[data-view-mode]").forEach((btn) => {
    btn.setAttribute("aria-pressed", String(btn.dataset.viewMode === state.viewMode));
  });
  document.getElementById("view-split-wrap").hidden = state.viewMode !== "split";
  document.getElementById("view-yolo-wrap").hidden = state.viewMode !== "yolo";
  document.getElementById("view-spatial-wrap").hidden = state.viewMode !== "spatial";

  // Update split comparator right image & label
  const rightImg = document.getElementById("comparator-right-img");
  rightImg.src = meta.img;
  rightImg.alt = `${meta.label} camera frame 54`;
  document.getElementById("comparator-right-tag").textContent = meta.label;

  // Evaluate exact Guard metrics on frame 54
  const res = scoreFromMetrics(
    {
      B: st3.B,
      S_pct: st3.S_pct,
      H_bit: st3.H_bit,
      predHealth: ds.f54_pred_health,
      adaptiveDayNight: true,
    },
    DEFAULT_GUARD_CONFIG
  );

  // System state badge
  const stateBadge = document.getElementById("hud-state-badge");
  stateBadge.className = `badge ${badgeClassForState(res.systemState)}`;
  stateBadge.textContent = res.systemState;

  // Gauges
  renderHealthGauge(document.getElementById("gauge-hybrid"), {
    value: res.healthScore,
    label: "Hybrid Health (H_hybrid)",
    color: colorForState(res.systemState),
    sublabel: res.systemState,
  });

  const neuralColor = meta.id === "S01_motion_blur_s2" || meta.id === "S01_motion_blur_s1" ? "#EF4444" : "#38BDF8";
  renderHealthGauge(document.getElementById("gauge-neural"), {
    value: ds.f54_pred_health,
    label: "Raw Neural (pred_health)",
    color: neuralColor,
    sublabel: sp ? `top1: ${sp.top1_issue}` : "direct head",
  });

  // Discordance banner
  const alertEl = document.getElementById("hud-discordance-banner");
  if (res.discordanceTriggered) {
    alertEl.className = "alert-banner is-danger";
    alertEl.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true" style="flex-shrink:0;margin-top:2px">
        <path d="M12 9v4m0 4h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/>
      </svg>
      <div>
        <strong>DISCORDANCE TRIGGERED (Δ = ${res.deltaDiscordance.toFixed(4)} &gt; τ = 0.05)</strong><br>
        Mạng nơ-ron chấm <code>${ds.f54_pred_health.toFixed(4)}</code> dù độ nét <code>B_norm = ${res.bNorm.toFixed(4)}</code>.
        Kích hoạt hệ số phạt <code>γ = ${res.discordanceFactor.toFixed(3)}</code> và bật <code>early_warning_flag</code>.
      </div>
    `;
  } else {
    alertEl.className = "alert-banner is-ok";
    alertEl.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true" style="flex-shrink:0;margin-top:2px">
        <path d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/>
      </svg>
      <div>
        <strong>OPTICAL &amp; GUARD ALIGNED (Δ = ${res.deltaDiscordance.toFixed(4)})</strong><br>
        Camera ở trạng thái <strong>${res.systemState}</strong>, giữ toàn phần trọng số Sensor Fusion cho camera RGB.
      </div>
    `;
  }

  // Dynamic Sensor Fusion Weight bars
  const camPct = Math.round(res.cameraWeight * 1000) / 10;
  const lidarPct = Math.round((1 - res.cameraWeight) * 1000) / 10;
  document.getElementById("fusion-cam-pct").textContent = `${camPct.toFixed(1)}% (w = ${res.cameraWeight.toFixed(4)})`;
  document.getElementById("fusion-lidar-pct").textContent = `${lidarPct.toFixed(1)}% fallback`;
  const camBar = document.getElementById("fusion-cam-bar");
  camBar.style.width = `${Math.max(1.5, camPct)}%`;
  camBar.style.background = colorForState(res.systemState);
  document.getElementById("fusion-lidar-bar").style.width = `${Math.max(1.5, lidarPct)}%`;

  // Optical & Downstream KV readouts
  document.getElementById("kv-b").textContent = `${st3.B.toFixed(2)} (${st3.B_percent_change_vs_clean.toFixed(1)}%)`;
  document.getElementById("kv-sh").textContent = `${st3.S_pct.toFixed(2)}% / ${st3.H_bit.toFixed(3)} bit`;
  document.getElementById("kv-yolo-cnt").textContent = `${ds.f54_detection_count} obj (8f: ${ds.total_detections_8f}/52)`;
  document.getElementById("kv-yolo-map").textContent = `mAP50: ${ds.f54_map50_proxy.toFixed(3)} · conf ${ds.f54_mean_confidence.toFixed(3)}`;

  renderFrameTimeline();
}

function renderMainCharts() {
  const { downstreamSummary } = state.data;
  const tooltip = document.getElementById("chart-tooltip");
  const labels = ["Clean", "Blur s1 (11px)", "Blur s2 (13px)", "Blur s5 (33px)"];

  // Chart 1: Neural Failure vs Hybrid Monotonic Health
  renderLineChart(document.getElementById("chart-health"), tooltip, {
    labels,
    highlightXIndex: 2,
    highlightLabel: "FAILURE CASE s2",
    ariaLabel: "Comparison of raw neural pred_health against monotonic Hybrid Health across S01 blur levels",
    bands: [
      { from: 0.75, to: 1.0, color: "rgba(34, 197, 94, 0.06)" },
      { from: 0.35, to: 0.75, color: "rgba(245, 158, 11, 0.05)" },
      { from: 0.0, to: 0.35, color: "rgba(239, 68, 68, 0.06)" },
    ],
    thresholds: [
      { value: 0.75, color: "#22C55E", label: "HEALTHY >= 0.75" },
      { value: 0.35, color: "#F59E0B", label: "CRITICAL < 0.35" },
    ],
    series: [
      {
        name: "Neural pred_health (f54, lỗi)",
        values: downstreamSummary.map((r) => r.f54_pred_health),
        color: "#EF4444",
        dash: "6 4",
        shape: "circle",
        tooltipExtra: (idx) =>
          idx === 2 ? `<div style="color:#FCA5A5">Tăng ngược +0.259 dù B giảm 94.4%</div>` : "",
      },
      {
        name: "Hybrid Health (f54)",
        values: downstreamSummary.map((r) => r.f54_hybrid_health),
        color: "#22C55E",
        shape: "triangle",
        strokeWidth: 2.6,
      },
      {
        name: "Hybrid Health (mean 8f)",
        values: downstreamSummary.map((r) => r.mean8_hybrid_health),
        color: "#38BDF8",
        dash: "2 3",
        shape: "square",
      },
    ],
  });

  // Chart 2: Downstream YOLOv8 Impact
  renderLineChart(document.getElementById("chart-downstream"), tooltip, {
    labels,
    highlightXIndex: 2,
    highlightLabel: "YOLOv8 COLLAPSE",
    ariaLabel: "Downstream YOLOv8 mAP50 proxy, object retention, and mean confidence across S01 blur levels",
    series: [
      {
        name: "mAP@0.50 Proxy (8f)",
        values: downstreamSummary.map((r) => r.mean_map50_proxy_8f),
        color: "#A78BFA",
        shape: "diamond",
        strokeWidth: 2.6,
      },
      {
        name: "Object Retention (8f)",
        values: downstreamSummary.map((r) => Math.max(0, 1 - r.object_drop_pct_8f / 100)),
        color: "#38BDF8",
        shape: "circle",
      },
      {
        name: "Mean Confidence (8f)",
        values: downstreamSummary.map((r) => r.mean_confidence_8f),
        color: "#F59E0B",
        dash: "5 4",
        shape: "triangle",
      },
    ],
  });
}

function renderFrameTimeline() {
  const { downstreamPerFrame } = state.data;
  const tooltip = document.getElementById("chart-tooltip");
  const clipRows = downstreamPerFrame.filter((r) => r.sample_id === state.selectedSampleId);
  const meta = SAMPLE_META.find((m) => m.id === state.selectedSampleId);

  document.getElementById("timeline-clip-label").textContent = meta?.label ?? state.selectedSampleId;

  // Frame pills
  const pillWrap = document.getElementById("frame-pills");
  pillWrap.innerHTML = "";
  FRAME_INDICES.forEach((fIdx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn";
    btn.textContent = `f${fIdx}`;
    btn.setAttribute("aria-pressed", String(fIdx === state.selectedFrameIdx));
    btn.addEventListener("click", () => {
      state.selectedFrameIdx = fIdx;
      renderFrameTimeline();
    });
    pillWrap.appendChild(btn);
  });

  renderLineChart(document.getElementById("chart-timeline"), tooltip, {
    labels: clipRows.map((r) => `f${r.frame_idx}`),
    highlightXIndex: FRAME_INDICES.indexOf(state.selectedFrameIdx),
    highlightLabel: `f${state.selectedFrameIdx}`,
    height: 220,
    ariaLabel: `Per-frame telemetry timeline for ${state.selectedSampleId}`,
    series: [
      {
        name: "Hybrid Health",
        values: clipRows.map((r) => r.hybrid_health),
        color: "#22C55E",
        shape: "triangle",
        strokeWidth: 2.5,
      },
      {
        name: "Neural pred_health",
        values: clipRows.map((r) => r.pred_health),
        color: "#EF4444",
        dash: "6 4",
        shape: "circle",
      },
      {
        name: "YOLOv8 mAP50 Proxy",
        values: clipRows.map((r) => r.map50_proxy),
        color: "#A78BFA",
        shape: "diamond",
      },
    ],
  });

  // Render per-frame accessible table
  const tbody = document.getElementById("timeline-table-body");
  tbody.innerHTML = clipRows
    .map(
      (r) => `
      <tr class="${r.frame_idx === state.selectedFrameIdx ? "is-highlight" : ""}">
        <td class="mono">f${r.frame_idx}</td>
        <td class="mono">${r.B.toFixed(1)}</td>
        <td class="mono">${r.pred_health.toFixed(4)}</td>
        <td class="mono">${r.hybrid_health.toFixed(4)}</td>
        <td><span class="badge ${badgeClassForState(r.system_state)}">${r.system_state}</span></td>
        <td class="mono">${r.camera_weight.toFixed(4)}</td>
        <td class="mono">${r.detection_count} / ${r.clean_gt_count}</td>
        <td class="mono">${r.map50_proxy.toFixed(3)}</td>
      </tr>
    `
    )
    .join("");
}

function renderWhatIfLab() {
  const { dayNightRows } = state.data;
  const tooltip = document.getElementById("chart-tooltip");
  const { evaluated, series } = evaluateSeriesMonotonicity(dayNightRows, state.whatIfConfig, state.adaptiveDayNight);

  // Monotonicity badges for all 4 series
  const monoWrap = document.getElementById("mono-checks");
  monoWrap.innerHTML = series
    .map(
      (s) => `
      <div class="mono-chip">
        <span>${s.label}</span>
        <span class="badge ${s.monotonic ? "badge-healthy" : "badge-critical"}">
          ${s.monotonic ? "MONOTONIC" : "VIOLATED"}
        </span>
      </div>
    `
    )
    .join("");

  // Highlight S06_clean status
  const s06 = evaluated.find((r) => r.sample_id === "S06_clean");
  const s06El = document.getElementById("s06-live-status");
  s06El.innerHTML = `
    <strong>S06 Night Clean:</strong>
    <span class="badge ${badgeClassForState(s06.sim.systemState)}">${s06.sim.systemState}</span>
    <span class="mono">H = ${s06.sim.healthScore.toFixed(4)} · B_base = ${s06.sim.bBaselineDynamic.toFixed(1)} · w = ${s06.sim.cameraWeight.toFixed(3)}</span>
  `;

  // Render 4-series comparison chart (s0..s5)
  const palette = ["#22C55E", "#38BDF8", "#F59E0B", "#A78BFA"];
  const shapes = ["triangle", "circle", "diamond", "square"];
  const dashes = ["none", "5 4", "none", "3 3"];
  renderLineChart(document.getElementById("chart-whatif"), tooltip, {
    labels: ["Clean", "s1", "s2", "s3", "s4", "s5"],
    height: 260,
    ariaLabel: "Simulated Hybrid Health across Day S01 and Night S06 blur and underexposure series",
    thresholds: [
      { value: state.whatIfConfig.healthyThreshold, color: "#22C55E", label: "HEALTHY 0.75" },
      { value: state.whatIfConfig.degradedThreshold, color: "#F59E0B", label: "CRITICAL 0.35" },
    ],
    series: series.map((s, idx) => ({
      name: s.label,
      values: s.rows.map((r) => r.sim.healthScore),
      color: palette[idx],
      shape: shapes[idx],
      dash: dashes[idx],
      strokeWidth: 2.3,
    })),
  });

  // Render full 24-clip simulated table
  const tbody = document.getElementById("whatif-table-body");
  tbody.innerHTML = evaluated
    .map(
      (r) => `
      <tr class="${r.sample_id === "S01_motion_blur_s2" || r.sample_id === "S06_clean" ? "is-highlight" : ""}">
        <td class="mono">${r.sample_id}</td>
        <td>${r.time_of_day}</td>
        <td class="mono">${r.B.toFixed(1)}</td>
        <td class="mono">${r.sim.bBaselineDynamic.toFixed(1)}</td>
        <td class="mono">${r.pred_health_f54.toFixed(4)}</td>
        <td class="mono">${r.sim.deltaDiscordance.toFixed(3)}</td>
        <td class="mono"><strong>${r.sim.healthScore.toFixed(4)}</strong></td>
        <td><span class="badge ${badgeClassForState(r.sim.systemState)}">${r.sim.systemState}</span></td>
        <td class="mono">${r.sim.cameraWeight.toFixed(3)}</td>
      </tr>
    `
    )
    .join("");
}

function renderSpatialSection() {
  const { spatialStats } = state.data;
  const tbody = document.getElementById("spatial-table-body");
  tbody.innerHTML = spatialStats
    .map(
      (r) => `
      <tr class="${r.sample_id === "S01_motion_blur_s2" ? "is-highlight" : ""}">
        <td class="mono">${r.sample_id}</td>
        <td><span class="badge badge-critical">${r.top1_issue} (${(r.top1_prob * 100).toFixed(1)}%)</span></td>
        <td class="mono">${r.pred_pix_mean.toFixed(4)}</td>
        <td class="mono">${r.inside_gt_bbox_mean.toFixed(4)}</td>
        <td class="mono">${r.outside_gt_bbox_mean.toFixed(4)}</td>
        <td class="mono">${r.bbox_vs_bg_ratio.toFixed(3)}x</td>
      </tr>
    `
    )
    .join("");
}

function bindEvents() {
  document.querySelectorAll("[data-sample-id]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.selectedSampleId = btn.dataset.sampleId;
      updateInspector();
    });
  });

  document.querySelectorAll("[data-view-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.viewMode = btn.dataset.viewMode;
      updateInspector();
    });
  });

  const sliderWBlur = document.getElementById("slider-wblur");
  const sliderTau = document.getElementById("slider-tau");
  const sliderGamma = document.getElementById("slider-gamma");
  const adaptiveSwitch = document.getElementById("toggle-adaptive");
  const resetBtn = document.getElementById("btn-reset-whatif");

  function syncSlidersFromState() {
    sliderWBlur.value = String(state.whatIfConfig.wBlur);
    sliderTau.value = String(state.whatIfConfig.discordanceTau);
    sliderGamma.value = String(state.whatIfConfig.discordanceGamma);
    adaptiveSwitch.setAttribute("aria-checked", String(state.adaptiveDayNight));
    document.getElementById("val-wblur").textContent =
      `${state.whatIfConfig.wBlur.toFixed(2)} / ${state.whatIfConfig.wNeural.toFixed(2)}`;
    document.getElementById("val-tau").textContent = state.whatIfConfig.discordanceTau.toFixed(2);
    document.getElementById("val-gamma").textContent = state.whatIfConfig.discordanceGamma.toFixed(2);
  }

  sliderWBlur.addEventListener("input", () => {
    const wb = Number(sliderWBlur.value);
    state.whatIfConfig.wBlur = wb;
    state.whatIfConfig.wNeural = Math.round((1 - wb) * 100) / 100;
    syncSlidersFromState();
    renderWhatIfLab();
  });

  sliderTau.addEventListener("input", () => {
    state.whatIfConfig.discordanceTau = Number(sliderTau.value);
    syncSlidersFromState();
    renderWhatIfLab();
  });

  sliderGamma.addEventListener("input", () => {
    state.whatIfConfig.discordanceGamma = Number(sliderGamma.value);
    syncSlidersFromState();
    renderWhatIfLab();
  });

  adaptiveSwitch.addEventListener("click", () => {
    state.adaptiveDayNight = !state.adaptiveDayNight;
    syncSlidersFromState();
    renderWhatIfLab();
  });

  resetBtn.addEventListener("click", () => {
    state.whatIfConfig = { ...DEFAULT_GUARD_CONFIG };
    state.adaptiveDayNight = true;
    syncSlidersFromState();
    renderWhatIfLab();
  });

  syncSlidersFromState();
}

async function boot() {
  initImageComparator({
    stageEl: document.getElementById("comparator-stage"),
    overlayEl: document.getElementById("comparator-overlay"),
    dividerEl: document.getElementById("comparator-divider"),
    handleEl: document.getElementById("comparator-handle"),
    readoutsEl: document.getElementById("comparator-readout"),
    presetButtons: Array.from(document.querySelectorAll("[data-split-pct]")),
  });

  try {
    state.data = await loadAllArtifacts();
    initHeaderAndKpis(state.data);
    bindEvents();
    updateInspector();
    renderMainCharts();
    renderWhatIfLab();
    renderSpatialSection();
  } catch (err) {
    const container = document.getElementById("main-content");
    container.insertAdjacentHTML(
      "afterbegin",
      `<div class="alert-banner is-danger" role="alert"><strong>Không tải được dữ liệu artifacts:</strong> ${err.message}. Hãy chạy <code>python scripts/serve_ui.py</code> từ thư mục gốc repository.</div>`
    );
  }
}

boot();
