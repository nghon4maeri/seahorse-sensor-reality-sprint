/**
 * Data loader for Seahorse ADAS Guard visualization UI.
 * Fetches verified CSV and JSON artifacts from /outputs/pipeline_v2 and /outputs/stage3_small.
 */

export function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = splitCsvLine(line);
    const obj = {};
    headers.forEach((h, idx) => {
      const raw = cells[idx] ?? "";
      obj[h] = coerceValue(raw);
    });
    return obj;
  });
}

function splitCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      out.push(cur.trim());
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur.trim());
  return out;
}

function coerceValue(raw) {
  if (raw === "" || raw === "null" || raw === "None") return null;
  if (raw === "True" || raw === "true") return true;
  if (raw === "False" || raw === "false") return false;
  if (/^-?\d+(\.\d+)?(e[+-]?\d+)?$/i.test(raw)) return Number(raw);
  return raw;
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
  return res.text();
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
  return res.json();
}

export async function loadAllArtifacts() {
  const [
    manifest,
    stage3SummaryText,
    downstreamSummaryText,
    downstreamPerFrameText,
    dayNightCsvText,
    spatialCsvText,
  ] = await Promise.all([
    fetchJson("/outputs/pipeline_v2/pipeline_manifest.json"),
    fetchText("/outputs/stage3_small/benchmark_summary.csv"),
    fetchText("/outputs/pipeline_v2/downstream_yolov8_summary.csv"),
    fetchText("/outputs/pipeline_v2/downstream_yolov8_per_frame.csv"),
    fetchText("/outputs/pipeline_v2/day_night_adaptive_comparison.csv"),
    fetchText("/outputs/pipeline_v2/spatial_uncertainty_stats.csv"),
  ]);

  return {
    manifest,
    stage3Summary: parseCsv(stage3SummaryText),
    downstreamSummary: parseCsv(downstreamSummaryText),
    downstreamPerFrame: parseCsv(downstreamPerFrameText),
    dayNightRows: parseCsv(dayNightCsvText),
    spatialStats: parseCsv(spatialCsvText),
  };
}
