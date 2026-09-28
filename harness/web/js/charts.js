// Small SVG charts: one axis, hairline grid, thin marks, hover tooltips (textContent only).
import { el, fmt } from "./ui.js";

const NS = "http://www.w3.org/2000/svg";

export function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) if (value != null) node.setAttribute(key, value);
  for (const child of children.flat(Infinity)) if (child) node.append(child);
  return node;
}

const tooltipNode = () => document.getElementById("tooltip");

export function showTooltip(event, rows) {
  const tip = tooltipNode();
  tip.replaceChildren(...rows.map(([value, key]) => el("div", {},
    el("span", { class: "tv", text: value }), key ? el("span", { class: "tk", text: `  ${key}` }) : null)));
  tip.hidden = false;
  const x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(event.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
export const hideTooltip = () => { tooltipNode().hidden = true; };

export function niceTicks(lo, hi, count = 5) {
  const span = hi - lo || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

// Horizontal dot plot: one row per category, optional interval, optional reference line.
export function dotPlot(rows, { valueLabel, reference, referenceLabel, width = 820, rowHeight = 18,
                                color = "var(--series-1)", diverging = false, tooltip } = {}) {
  const left = 128, right = 24, top = 22, bottom = 30;
  const values = rows.flatMap((r) => [r.value, ...(r.interval || [])]).filter((v) => v != null);
  if (reference != null) values.push(reference);
  let lo = Math.min(0, ...values), hi = Math.max(...values, 0.001);
  if (diverging) { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
  const ticks = niceTicks(lo, hi);
  lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const height = top + rows.length * rowHeight + bottom;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (width - left - right);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, style: "max-width: 100%; height: auto",
                            role: "img", "aria-label": valueLabel });
  for (const t of ticks) {
    root.append(svg("line", { class: "grid", x1: x(t), x2: x(t), y1: top - 4, y2: height - bottom }),
      svg("text", { x: x(t), y: height - bottom + 16, "text-anchor": "middle" }, document.createTextNode(fmt(t, Math.abs(t) < 10 ? 1 : 0))));
  }
  root.append(svg("text", { x: width - right, y: height - 4, "text-anchor": "end" }, document.createTextNode(valueLabel)));
  if (reference != null) {
    root.append(svg("line", { class: "ref", x1: x(reference), x2: x(reference), y1: top - 10, y2: height - bottom }),
      svg("text", { x: x(reference) + 4, y: top - 10 }, document.createTextNode(referenceLabel || "")));
  }
  rows.forEach((row, i) => {
    const cy = top + i * rowHeight + rowHeight / 2;
    root.append(svg("text", { x: left - 8, y: cy + 4, "text-anchor": "end" }, document.createTextNode(row.label)));
    if (row.value == null) return;
    const fill = diverging ? (row.value < 0 ? "var(--div-neg)" : row.value > 0 ? "var(--div-pos)" : "var(--muted)") : color;
    if (row.interval) {
      root.append(svg("line", { x1: x(row.interval[0]), x2: x(row.interval[1]), y1: cy, y2: cy,
                                stroke: "var(--axis)", "stroke-width": 2, "stroke-linecap": "round" }));
    }
    const dot = svg("circle", { cx: x(row.value), cy, r: 4.5, fill, stroke: "var(--surface)", "stroke-width": 2 });
    const hit = svg("rect", { class: "hit", x: left, y: cy - rowHeight / 2, width: width - left - right, height: rowHeight });
    hit.addEventListener("pointermove", (e) => { dot.setAttribute("r", 6); showTooltip(e, tooltip(row)); });
    hit.addEventListener("pointerleave", () => { dot.setAttribute("r", 4.5); hideTooltip(); });
    root.append(dot, hit);
  });
  return root;
}

// Empirical CDFs from quantile curves (values at evenly spaced probabilities); one axis.
// series: [{label, quantiles, color}]. Budget lines are drawn only when inside the range.
export function cdfPlot(series, { valueLabel, budgets = [], width = 820, height = 240 } = {}) {
  const left = 44, right = 20, top = 12, bottom = 34;
  const hi0 = Math.max(...series.flatMap((s) => s.quantiles), 0.001);
  const ticks = niceTicks(0, hi0);
  const hi = Math.max(hi0, ticks[ticks.length - 1]);
  const x = (v) => left + (v / hi) * (width - left - right);
  const y = (p) => top + (1 - p) * (height - top - bottom);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, style: "max-width: 100%; height: auto",
                            role: "img", "aria-label": valueLabel });
  for (const t of ticks) {
    root.append(svg("line", { class: "grid", x1: x(t), x2: x(t), y1: top, y2: height - bottom }),
      svg("text", { x: x(t), y: height - bottom + 15, "text-anchor": "middle" }, document.createTextNode(fmt(t, t < 10 ? 1 : 0))));
  }
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    root.append(svg("line", { class: "grid", x1: left, x2: width - right, y1: y(p), y2: y(p) }),
      svg("text", { x: left - 6, y: y(p) + 4, "text-anchor": "end" }, document.createTextNode(fmt(p, 2))));
  }
  root.append(svg("text", { x: width - right, y: height - 4, "text-anchor": "end" }, document.createTextNode(valueLabel)));
  for (const b of budgets.filter((b) => b <= hi)) {
    root.append(svg("line", { class: "budget", x1: x(b), x2: x(b), y1: top, y2: height - bottom }),
      svg("text", { x: x(b) - 4, y: top + 10, "text-anchor": "end" }, document.createTextNode(`${b} ms`)));
  }
  for (const s of series) {
    const n = s.quantiles.length;
    const points = s.quantiles.map((v, i) => `${x(v)},${y(i / (n - 1))}`).join(" ");
    root.append(svg("polyline", { points, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }));
    if (series.length > 1) {
      const p50 = s.quantiles[Math.floor(n / 2)];
      root.append(svg("text", { x: x(p50) + 6, y: y(0.5) + 4 }, document.createTextNode(s.label)));
    }
  }
  const cross = svg("line", { class: "ref", y1: top, y2: height - bottom, visibility: "hidden" });
  const hit = svg("rect", { class: "hit", x: left, y: top, width: width - left - right, height: height - top - bottom });
  hit.addEventListener("pointermove", (e) => {
    const box = root.getBoundingClientRect();
    const value = ((e.clientX - box.left) * (width / box.width) - left) / (width - left - right) * hi;
    cross.setAttribute("x1", x(value)); cross.setAttribute("x2", x(value)); cross.setAttribute("visibility", "visible");
    showTooltip(e, [[`≤ ${fmt(value, 2)} ms`, ""], ...series.map((s) =>
      [`${fmt(100 * s.quantiles.filter((q) => q <= value).length / s.quantiles.length, 1)}%`, s.label])]);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTooltip(); });
  root.append(cross, hit);
  return root;
}

export function lineLegend(series) {
  return el("div", { class: "legend" }, series.map((s) =>
    el("span", {}, el("span", { class: "line", style: `background: ${s.color}` }), s.label)));
}

// Tiny inline interval (leaderboard): whisker + dot on a shared scale.
export function intervalGlyph(value, interval, lo, hi, width = 120) {
  const x = (v) => 4 + ((v - lo) / (hi - lo || 1)) * (width - 8);
  const root = svg("svg", { width, height: 14, viewBox: `0 0 ${width} 14`, "aria-hidden": "true" });
  if (interval) root.append(svg("line", { x1: x(interval[0]), x2: x(interval[1]), y1: 7, y2: 7, stroke: "var(--axis)",
                                          "stroke-width": 2, "stroke-linecap": "round" }));
  if (value != null) root.append(svg("circle", { cx: x(value), cy: 7, r: 4, fill: "var(--series-1)",
                                                 stroke: "var(--surface)", "stroke-width": 2 }));
  return root;
}
