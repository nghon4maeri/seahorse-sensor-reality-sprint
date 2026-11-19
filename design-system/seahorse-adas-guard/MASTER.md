# Design System: Seahorse ADAS Guard

Generated via `ui-ux-pro-max-skill` (`--design-system --density 8 --tokens css`) and domain searches (`chart`, `ux`).

## Pattern & Visual Direction
- **Pattern:** Real-Time / Operations Telemetry Console (Density 8/10 — Dense Dashboard)
- **Style:** Dark Glassmorphism (`backdrop-filter: blur(12px)`, crisp 1px borders, layered surfaces)
- **Telemetry Integrity:** Never fake "LIVE" badges on offline benchmark outputs — always display verified provenance (`OFFLINE BENCHMARK · RTX 4060 / RUST`), exact timestamp from `pipeline_manifest.json`, and latency benchmarks.

## Color Tokens (Dark Theme, >= 4.5:1 Contrast)
- `--color-background`: `#0F172A`
- `--color-card`: `#1B2336`
- `--color-muted`: `#272F42`
- `--color-primary`: `#1E293B`
- `--color-secondary`: `#334155`
- `--color-border`: `#475569`
- `--color-foreground`: `#F8FAFC`
- `--color-muted-foreground`: `#94A3B8`
- `--color-accent` (`HEALTHY`): `#22C55E`
- `--color-warning` (`DEGRADED`): `#F59E0B`
- `--color-destructive` (`CRITICAL` / Anomaly): `#EF4444`
- `--color-info` (Downstream / Optical): `#38BDF8`
- `--color-ring`: `#FFFFFF`

## Typography
- **Heading & Tabular Numerals:** `Fira Code`, monospace (`font-variant-numeric: tabular-nums`)
- **Body & UI Labels:** `Fira Sans`, system-ui, sans-serif

## Accessibility & UX Contracts
1. **Zero Emojis as Icons:** Inline SVG vector icons with `1.75px` stroke width (`aria-hidden="true"` when decorative).
2. **Chart Accessibility:** Combine distinct line styles (solid vs dashed), distinct point shapes (circle, triangle, diamond), anomaly highlight band, text annotations, and accessible data tables — never rely on color alone.
3. **Dragging Movements (WCAG 2.2 AA):** Before/After image comparator provides pointer drag, keyboard `ArrowLeft`/`ArrowRight`/`Home`/`End`, and one-click preset buttons (`0% / 50% / 100%`).
4. **Motion & Focus:** Visible `:focus-visible` outline (`2px solid #FFFFFF`), smooth `150–250ms` transitions, full `@media (prefers-reduced-motion: reduce)` support.
