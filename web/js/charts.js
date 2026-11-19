/**
 * Accessible SVG Interactive Charts & Semi-Circular ADAS Health Gauge.
 * Follows ui-ux-pro-max-skill chart contracts: distinct line styles, distinct
 * marker shapes, anomaly highlight band, text annotations, and hover/focus tooltips.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, text = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    node.setAttribute(k, String(v));
  }
  if (text !== null) node.textContent = text;
  return node;
}

function drawMarker(shape, cx, cy, r, fill, stroke = "#0F172A") {
  if (shape === "triangle") {
    const pts = `${cx},${cy - r - 1} ${cx - r - 1},${cy + r} ${cx + r + 1},${cy + r}`;
    return el("polygon", { points: pts, fill, stroke, "stroke-width": 1.4 });
  }
  if (shape === "diamond") {
    const pts = `${cx},${cy - r - 1} ${cx + r + 1},${cy} ${cx},${cy + r + 1} ${cx - r - 1},${cy}`;
    return el("polygon", { points: pts, fill, stroke, "stroke-width": 1.4 });
  }
  if (shape === "square") {
    return el("rect", {
      x: cx - r,
      y: cy - r,
      width: r * 2,
      height: r * 2,
      rx: 1.5,
      fill,
      stroke,
      "stroke-width": 1.4,
    });
  }
  return el("circle", { cx, cy, r, fill, stroke, "stroke-width": 1.4 });
}

export function renderLineChart(containerEl, tooltipEl, config) {
  const {
    labels,
    series,
    yMin = 0,
    yMax = 1,
    yTicks = [0, 0.25, 0.5, 0.75, 1.0],
    yFormat = (v) => v.toFixed(2),
    bands = [],
    thresholds = [],
    highlightXIndex = null,
    highlightLabel = null,
    width = 560,
    height = 250,
    ariaLabel = "Telemetry line chart",
  } = config;

  const pad = { top: 20, right: 20, bottom: 38, left: 46 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const xPos = (idx) =>
    labels.length <= 1 ? pad.left + plotW / 2 : pad.left + (idx / (labels.length - 1)) * plotW;
  const yPos = (val) => {
    const clamped = Math.min(yMax, Math.max(yMin, val));
    return pad.top + (1 - (clamped - yMin) / (yMax - yMin || 1)) * plotH;
  };

  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "chart-svg",
    role: "img",
    "aria-label": ariaLabel,
  });

  // Status background bands
  bands.forEach((b) => {
    const yTop = yPos(b.to);
    const yBot = yPos(b.from);
    svg.appendChild(
      el("rect", {
        x: pad.left,
        y: yTop,
        width: plotW,
        height: Math.max(0, yBot - yTop),
        fill: b.color,
      })
    );
  });

  // Anomaly vertical highlight band (e.g. s2 Failure Case)
  if (highlightXIndex !== null && labels.length > 1) {
    const cx = xPos(highlightXIndex);
    const bw = Math.min(82, plotW / labels.length);
    svg.appendChild(
      el("rect", {
        x: cx - bw / 2,
        y: pad.top,
        width: bw,
        height: plotH,
        fill: "rgba(239, 68, 68, 0.12)",
        stroke: "rgba(239, 68, 68, 0.45)",
        "stroke-dasharray": "4 3",
        rx: 4,
      })
    );
    if (highlightLabel) {
      svg.appendChild(
        el(
          "text",
          {
            x: cx,
            y: pad.top + 13,
            "text-anchor": "middle",
            fill: "#FCA5A5",
            "font-family": "var(--font-heading)",
            "font-size": 10,
            "font-weight": 700,
          },
          highlightLabel
        )
      );
    }
  }

  // Horizontal grid + Y labels
  yTicks.forEach((tick) => {
    const y = yPos(tick);
    svg.appendChild(
      el("line", {
        x1: pad.left,
        y1: y,
        x2: pad.left + plotW,
        y2: y,
        stroke: "rgba(148, 163, 184, 0.16)",
        "stroke-width": 1,
      })
    );
    svg.appendChild(
      el(
        "text",
        {
          x: pad.left - 8,
          y: y + 3.5,
          "text-anchor": "end",
          fill: "#94A3B8",
          "font-family": "var(--font-heading)",
          "font-size": 10,
        },
        yFormat(tick)
      )
    );
  });

  // Threshold dashed lines
  thresholds.forEach((th) => {
    const y = yPos(th.value);
    svg.appendChild(
      el("line", {
        x1: pad.left,
        y1: y,
        x2: pad.left + plotW,
        y2: y,
        stroke: th.color,
        "stroke-width": 1.2,
        "stroke-dasharray": "5 4",
        opacity: 0.75,
      })
    );
    svg.appendChild(
      el(
        "text",
        {
          x: pad.left + plotW - 4,
          y: y - 4,
          "text-anchor": "end",
          fill: th.color,
          "font-family": "var(--font-heading)",
          "font-size": 9.5,
          "font-weight": 600,
        },
        th.label
      )
    );
  });

  // X axis labels
  labels.forEach((lbl, idx) => {
    const x = xPos(idx);
    svg.appendChild(
      el(
        "text",
        {
          x,
          y: height - 12,
          "text-anchor": "middle",
          fill: idx === highlightXIndex ? "#FCA5A5" : "#CBD5E1",
          "font-family": "var(--font-heading)",
          "font-size": 10.5,
          "font-weight": idx === highlightXIndex ? 700 : 500,
        },
        lbl
      )
    );
  });

  // Series lines & focusable points
  series.forEach((s) => {
    const d = s.values
      .map((val, idx) => `${idx === 0 ? "M" : "L"} ${xPos(idx).toFixed(2)} ${yPos(val).toFixed(2)}`)
      .join(" ");
    svg.appendChild(
      el("path", {
        d,
        fill: "none",
        stroke: s.color,
        "stroke-width": s.strokeWidth ?? 2.3,
        "stroke-dasharray": s.dash ?? "none",
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
      })
    );

    s.values.forEach((val, idx) => {
      const cx = xPos(idx);
      const cy = yPos(val);
      const g = el("g", {
        class: "chart-point",
        tabindex: "0",
        role: "button",
        "aria-label": `${s.name} at ${labels[idx]}: ${yFormat(val)}`,
      });
      g.appendChild(el("circle", { cx, cy, r: 12, fill: "transparent" }));
      g.appendChild(drawMarker(s.shape ?? "circle", cx, cy, 4.8, s.color));

      const showTip = (evt) => {
        const extra = s.tooltipExtra ? s.tooltipExtra(idx, val) : "";
        tooltipEl.innerHTML = `<div style="color:${s.color};font-weight:700">${s.name}</div><div>${labels[idx]}: <strong>${yFormat(val)}</strong></div>${extra}`;
        tooltipEl.classList.add("is-visible");
        const rect = g.getBoundingClientRect();
        const tx = evt?.clientX ?? rect.left + rect.width / 2;
        const ty = evt?.clientY ?? rect.top;
        tooltipEl.style.left = `${Math.min(window.innerWidth - 270, Math.max(12, tx + 14))}px`;
        tooltipEl.style.top = `${Math.max(12, ty - 54)}px`;
      };
      const hideTip = () => tooltipEl.classList.remove("is-visible");

      g.addEventListener("mouseenter", showTip);
      g.addEventListener("mousemove", showTip);
      g.addEventListener("mouseleave", hideTip);
      g.addEventListener("focus", showTip);
      g.addEventListener("blur", hideTip);
      svg.appendChild(g);
    });
  });

  containerEl.innerHTML = "";
  containerEl.appendChild(svg);
}

/**
 * Render an accessible semi-circular health gauge SVG into target container.
 */
export function renderHealthGauge(containerEl, { value, label, color, sublabel }) {
  const clamped = Math.min(1, Math.max(0, value ?? 0));
  const r = 54;
  const cx = 72;
  const cy = 68;
  const arcLen = Math.PI * r;
  const dashOffset = arcLen * (1 - clamped);

  containerEl.innerHTML = `
    <svg class="gauge-svg" viewBox="0 0 144 88" role="img" aria-label="${label}: ${(clamped * 100).toFixed(1)}%">
      <path d="M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}"
            fill="none" stroke="#272F42" stroke-width="11" stroke-linecap="round" />
      <path d="M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}"
            fill="none" stroke="${color}" stroke-width="11" stroke-linecap="round"
            stroke-dasharray="${arcLen.toFixed(2)}"
            stroke-dashoffset="${dashOffset.toFixed(2)}" />
      <text x="${cx}" y="${cy - 8}" text-anchor="middle" fill="#F8FAFC"
            font-family="var(--font-heading)" font-size="20" font-weight="700">${clamped.toFixed(4)}</text>
      <text x="${cx}" y="${cy + 10}" text-anchor="middle" fill="#94A3B8"
            font-family="var(--font-body)" font-size="10.5">${sublabel}</text>
    </svg>
    <div class="gauge-caption">${label}</div>
  `;
}
