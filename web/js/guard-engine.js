/**
 * Exact JavaScript port of crates/camera_guard_rs/src/{ambient.rs, guard.rs}
 * and scripts/camera_guard.py for the client-side Interactive What-If Simulator.
 */

export const DEFAULT_GUARD_CONFIG = Object.freeze({
  wBlur: 0.85,
  wNeural: 0.15,
  discordanceTau: 0.05,
  discordanceGamma: 1.5,
  alphaSaturation: 0.65,
  alphaEntropy: 0.5,
  dayBBaseline: 3544.1384,
  nightBBaseline: 86.3,
  daySBaselinePct: 5.0,
  nightSBaselinePct: 52.0,
  dayHBaselineBit: 7.4,
  nightHBaselineBit: 4.5,
  healthyThreshold: 0.75,
  degradedThreshold: 0.35,
  degradedMinWeight: 0.25,
});

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

export function estimateAmbient({ H_bit, meanLuma = null, darkRatioPct = null, S_pct = 0 }, cfg = DEFAULT_GUARD_CONFIG) {
  const darkPct = darkRatioPct !== null ? darkRatioPct : S_pct;
  const luma = meanLuma !== null ? meanLuma : H_bit >= 5.5 ? 125.0 : 35.0;

  const darkDayScore = clamp(1.0 - (darkPct - 12.0) / 28.0, 0.0, 1.0);
  const entropyDayScore = clamp((H_bit - 4.7) / 1.15, 0.0, 1.0);
  const lumaDayScore = clamp((luma - 28.0) / 55.0, 0.0, 1.0);

  const daylightFactor = clamp(
    0.45 * darkDayScore + 0.4 * entropyDayScore + 0.15 * lumaDayScore,
    0.0,
    1.0
  );

  const logDay = Math.log(Math.max(cfg.dayBBaseline, 1.0));
  const logNight = Math.log(Math.max(cfg.nightBBaseline, 1.0));
  const bBaseline = Math.exp(daylightFactor * logDay + (1.0 - daylightFactor) * logNight);
  const sBaselinePct = daylightFactor * cfg.daySBaselinePct + (1.0 - daylightFactor) * cfg.nightSBaselinePct;
  const hBaselineBit = daylightFactor * cfg.dayHBaselineBit + (1.0 - daylightFactor) * cfg.nightHBaselineBit;

  const regime = daylightFactor >= 0.7 ? "day" : daylightFactor <= 0.3 ? "night" : "twilight";
  return { regime, daylightFactor, bBaseline, sBaselinePct, hBaselineBit };
}

export function computeDynamicSensorWeight(healthScore, cfg = DEFAULT_GUARD_CONFIG) {
  const h = clamp(healthScore, 0.0, 1.0);
  if (h >= cfg.healthyThreshold) return 1.0;
  if (h >= cfg.degradedThreshold) {
    const t = (h - cfg.degradedThreshold) / (cfg.healthyThreshold - cfg.degradedThreshold);
    const smoothT = t * t * (3.0 - 2.0 * t);
    return cfg.degradedMinWeight + (1.0 - cfg.degradedMinWeight) * smoothT;
  }
  const t = h / cfg.degradedThreshold;
  return cfg.degradedMinWeight * Math.pow(t, 1.5);
}

export function classifySystemState(healthScore, cfg = DEFAULT_GUARD_CONFIG) {
  if (healthScore >= cfg.healthyThreshold) return "HEALTHY";
  if (healthScore >= cfg.degradedThreshold) return "DEGRADED";
  return "CRITICAL";
}

export function scoreFromMetrics(
  { B, S_pct, H_bit, predHealth = null, meanLuma = null, darkRatioPct = null, adaptiveDayNight = true },
  cfg = DEFAULT_GUARD_CONFIG
) {
  const ambient = adaptiveDayNight
    ? estimateAmbient({ H_bit, meanLuma, darkRatioPct, S_pct }, cfg)
    : {
        regime: "day_fixed",
        daylightFactor: 1.0,
        bBaseline: cfg.dayBBaseline,
        sBaselinePct: cfg.daySBaselinePct,
        hBaselineBit: cfg.dayHBaselineBit,
      };

  const bNorm = clamp(B / Math.max(ambient.bBaseline, 1e-6), 0.0, 1.0);

  let baseScore = bNorm;
  let deltaDiscordance = 0.0;
  let discordanceTriggered = false;
  let discordanceFactor = 1.0;

  if (predHealth !== null && predHealth !== undefined) {
    const neuralVal = clamp(predHealth, 0.0, 1.0);
    deltaDiscordance = Math.max(0.0, neuralVal - bNorm);
    discordanceTriggered = deltaDiscordance > cfg.discordanceTau;
    discordanceFactor = discordanceTriggered
      ? clamp(1.0 - cfg.discordanceGamma * deltaDiscordance, 0.05, 1.0)
      : 1.0;
    baseScore = cfg.wBlur * bNorm + cfg.wNeural * neuralVal;
  }

  const sExcess = Math.max(0.0, S_pct - ambient.sBaselinePct) / Math.max(100.0 - ambient.sBaselinePct, 1.0);
  const hNorm = clamp(H_bit / Math.max(ambient.hBaselineBit, 1e-6), 0.0, 1.0);
  const satFactor = clamp(1.0 - cfg.alphaSaturation * sExcess, 0.0, 1.0);
  const entFactor = clamp(1.0 - cfg.alphaEntropy * (1.0 - hNorm), 0.0, 1.0);
  const qualityPenalty = satFactor * entFactor;

  const healthScore = clamp(baseScore * discordanceFactor * qualityPenalty, 0.0, 1.0);
  const systemState = classifySystemState(healthScore, cfg);
  const cameraWeight = computeDynamicSensorWeight(healthScore, cfg);
  const earlyWarningFlag = systemState !== "HEALTHY" || discordanceTriggered;

  return {
    healthScore,
    systemState,
    cameraWeight,
    earlyWarningFlag,
    bNorm,
    ambientRegime: ambient.regime,
    daylightFactor: ambient.daylightFactor,
    bBaselineDynamic: ambient.bBaseline,
    deltaDiscordance,
    discordanceTriggered,
    discordanceFactor,
    qualityPenalty,
  };
}

export function evaluateSeriesMonotonicity(rows, cfg, adaptiveDayNight) {
  const evaluated = rows.map((r) => {
    const res = scoreFromMetrics(
      {
        B: r.B,
        S_pct: r.S_pct,
        H_bit: r.H_bit,
        predHealth: r.pred_health_f54,
        adaptiveDayNight,
      },
      cfg
    );
    return { ...r, sim: res };
  });

  const seriesDefs = [
    { key: "S01_motion_blur", label: "S01 Day · Blur", cleanId: "S01_clean", prefix: "S01_motion_blur_s" },
    { key: "S01_underexposure", label: "S01 Day · Underexp", cleanId: "S01_clean", prefix: "S01_underexposure_s" },
    { key: "S06_motion_blur", label: "S06 Night · Blur", cleanId: "S06_clean", prefix: "S06_motion_blur_s" },
    { key: "S06_underexposure", label: "S06 Night · Underexp", cleanId: "S06_clean", prefix: "S06_underexposure_s" },
  ];

  const series = seriesDefs.map((def) => {
    const subset = evaluated.filter((x) => x.sample_id === def.cleanId || x.sample_id.startsWith(def.prefix));
    const vals = subset.map((x) => x.sim.healthScore);
    const monotonic = vals.every((v, idx) => idx === 0 || vals[idx - 1] >= v);
    return { ...def, rows: subset, monotonic };
  });

  return { evaluated, series };
}
