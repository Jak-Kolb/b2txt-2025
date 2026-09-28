// Live player: stream a trial through a loaded pipeline at 1x and watch it decode.
// Neural panels show released features: threshold-crossing counts reconstructed from the
// z-scored release (binned per 20 ms; no spike times) and z-scored spike-band power.
import { diffLine, el, fill, fmt, flagBadges, getJSON } from "../ui.js";
import { hideTooltip, showTooltip, svg } from "../charts.js";

const WINDOW = 200;            // bins visible (4 s)
const GAP = 2;                 // separator rows between arrays
const N_E = 256;
const ARRAY_NAMES = ["ventral 6v", "area 4", "55b", "dorsal 6v"];
const ROW_OF = (e) => e + GAP * Math.floor(e / 64);
const ROWS = N_E + GAP * 3;
const SAT = 255;
const MAX_PINS = 4;

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function rgb(hex) {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

function palettes() {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const surface = rgb(css("--surface"));
  const countTop = rgb(dark ? "#b7d3f6" : "#0d366b");    // one-hue sequential ramp (blue)
  const steps = [0, 0.32, 0.47, 0.6, 0.72, 0.84, 1];
  const tx = Array.from({ length: 256 }, (_, c) => (c === SAT ? rgb("#fab219") : mix(surface, countTop, steps[Math.min(c, 6)])));
  const mid = rgb(dark ? "#383835" : "#f0efec"), neg = rgb(css("--div-neg")), pos = rgb(css("--div-pos"));
  const sbp = Array.from({ length: 256 }, (_, i) => {
    const z = (((i + 128) % 256) - 128) / (127 / 4);    // int8 byte -> z
    const t = Math.min(Math.abs(z) / 3, 1);
    return z < 0 ? mix(mid, neg, t) : mix(mid, pos, t);
  });
  return { tx, sbp, surface, grid: rgb(css("--axis")), muted: rgb("#898781"), series: [1, 2, 3, 4].map((i) => css(`--series-${i}`)) };
}

function bytes(b64) {
  const raw = atob(b64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

export async function liveView(main) {
  const registry = await getJSON("/api/registry");
  let colors = palettes();
  const s = { ws: null, closed: false, engine: null, trial: null, pins: [], blind: false, trials: [], dirty: false };

  // ---- controls ---------------------------------------------------------------------------
  const status = el("span", { class: "sub", text: "Connecting…" });
  const errorBox = el("span", { class: "error" });
  const presetSelect = el("select", {}, Object.keys(registry.pipelines).map((name) =>
    el("option", { value: name, text: `${name}: ${registry.pipelines[name].description || ""}`.slice(0, 90) })));
  presetSelect.value = registry.pipelines.la0_gen4g ? "la0_gen4g" : presetSelect.value;
  const custom = el("input", { type: "checkbox" });
  const acousticSelect = el("select", {}, Object.keys(registry.acoustic).map((n) => el("option", { value: n, text: n })));
  const lmSelect = el("select", {}, Object.keys(registry.lm).map((n) => el("option", { value: n, text: n })));
  const decodeKeys = ["acoustic_scale", "blank_penalty", "beam", "lattice_beam", "max_active", "min_active", "length_penalty", "blank_skip_thresh"];
  const decodeInputs = Object.fromEntries(decodeKeys.map((k) => [k, el("input", { type: "number", step: "any", style: "width: 80px" })]));
  const overrideSmoothing = el("input", { type: "checkbox" });
  const lookaheadInput = el("input", { type: "number", min: 0, max: 10, step: 1, value: 0, style: "width: 60px" });
  const customBox = el("div", { class: "card", hidden: true },
    el("div", { class: "row" }, el("label", {}, "Acoustic ", acousticSelect), el("label", {}, "LM ", lmSelect)),
    el("div", { class: "row" }, decodeKeys.map((k) => el("label", { class: "sub" }, `${k} `, decodeInputs[k]))),
    el("div", { class: "row" }, el("label", {}, overrideSmoothing, " override smoothing lookahead (bins) "), lookaheadInput,
      el("span", { class: "badge warn", text: "train/inference mismatch", title: "Inference smoothing will differ from the model's training smoothing" })));
  function fillDecode() {
    const preset = registry.pipelines[presetSelect.value];
    const lmName = custom.checked ? lmSelect.value : preset.lm;
    const values = { ...registry.lm[lmName].default_decode, ...(custom.checked ? {} : preset.decode || {}) };
    for (const k of decodeKeys) decodeInputs[k].value = values[k];
  }
  custom.addEventListener("change", () => {
    customBox.hidden = !custom.checked;
    const preset = registry.pipelines[presetSelect.value];
    acousticSelect.value = preset.acoustic; lmSelect.value = preset.lm;
    fillDecode();
  });
  lmSelect.addEventListener("change", fillDecode);
  presetSelect.addEventListener("change", fillDecode);
  fillDecode();

  function pipelineRequest() {
    if (!custom.checked) return { preset: presetSelect.value };
    const defaults = registry.lm[lmSelect.value].default_decode;
    const decode = {};
    for (const k of decodeKeys) if (Number(decodeInputs[k].value) !== Number(defaults[k])) decode[k] = Number(decodeInputs[k].value);
    const request = { acoustic: acousticSelect.value, lm: lmSelect.value };
    if (Object.keys(decode).length) request.decode = decode;
    if (overrideSmoothing.checked) request.preprocess = { smooth_lookahead: Number(lookaheadInput.value) };
    return request;
  }

  const splitSelect = el("select", {}, [["val", "validation (val-dev / former val-test)"], ["train", "train (in-sample)"],
    ["test", "test (unlabeled)"]].map(([v, t]) => el("option", { value: v, text: t })));
  const sessionSelect = el("select", {});
  const trialSelect = el("select", {});
  const speedSelect = el("select", {}, [["1", "1× (timing valid)"], ["2", "2× (display only)"], ["4", "4× (display only)"],
    ["max", "max (display only)"]].map(([v, t]) => el("option", { value: v, text: t })));
  const blind = el("input", { type: "checkbox", onchange: (e) => { s.blind = e.target.checked; renderSentence(); } });

  async function loadSessions() {
    const sessions = await getJSON(`/api/data/sessions?split=${splitSelect.value}`);
    sessionSelect.replaceChildren(...sessions.map((x) => el("option", { value: x.session,
      text: `${x.session} · ${x.partition} · ${x.n_trials} trials` })));
    await loadTrials();
  }
  async function loadTrials() {
    s.trials = await getJSON(`/api/data/trials?split=${splitSelect.value}&session=${sessionSelect.value}`);
    trialSelect.replaceChildren(...s.trials.map((t) => el("option", { value: t.trial_key,
      text: `${t.trial_key} — ${fmt(t.n_bins * 0.02, 1)} s · ${t.corpus || "source unknown"} (block ${t.block_num})` })));
  }
  splitSelect.addEventListener("change", loadSessions);
  sessionSelect.addEventListener("change", loadTrials);

  const send = (command) => {
    errorBox.textContent = "";
    if (s.ws && s.ws.readyState === 1) s.ws.send(JSON.stringify(command));
    else errorBox.textContent = "Not connected to the server";
  };
  const speed = () => (speedSelect.value === "max" ? "max" : Number(speedSelect.value));
  const item = (key) => ({ split: splitSelect.value, session: sessionSelect.value, trial_key: key });
  const loadButton = el("button", { class: "primary", onclick: () => send({ op: "load", pipeline: pipelineRequest() }) }, "Load pipeline");
  const unloadButton = el("button", { onclick: () => send({ op: "unload" }) }, "Unload");
  const playButton = el("button", { class: "primary", onclick: () => send({ op: "play", items: [item(trialSelect.value)], speed: speed() }) }, "Play trial");
  const sessionButton = el("button", { onclick: () => {
    const start = s.trials.findIndex((t) => t.trial_key === trialSelect.value);
    send({ op: "play", items: s.trials.slice(start).map((t) => item(t.trial_key)), speed: speed(), gap_s: 1.5 });
  } }, "Play session from here");
  const randomButton = el("button", { onclick: async () => {
    const options = [...sessionSelect.options];
    sessionSelect.value = options[Math.floor(Math.random() * options.length)].value;
    await loadTrials();
    trialSelect.value = s.trials[Math.floor(Math.random() * s.trials.length)].trial_key;
    send({ op: "play", items: [item(trialSelect.value)], speed: speed() });
  } }, "Random trial");
  const stopButton = el("button", { onclick: () => send({ op: "stop" }) }, "Stop");

  // ---- neural panels ----------------------------------------------------------------------
  const txCanvas = el("canvas", { width: WINDOW, height: ROWS, class: "neural" });
  const sbpCanvas = el("canvas", { width: WINDOW, height: ROWS, class: "neural" });
  const arrayLabels = () => el("div", { class: "array-labels" }, ARRAY_NAMES.map((name, a) =>
    el("div", { style: `top: ${(100 * (ROW_OF(a * 64) + 32)) / ROWS}%`, text: name })));
  const activity = svg("svg", { viewBox: `0 0 ${WINDOW * 4} 90`, width: "100%", height: 90, preserveAspectRatio: "none" });
  const activityLegend = el("div", { class: "legend" }, ARRAY_NAMES.map((name, a) =>
    el("span", {}, el("span", { class: "line", style: `background: var(--series-${a + 1})` }), name)));
  const timeAxis = el("div", { class: "time-axis" }, el("span", { text: "−4 s" }), el("span", { text: "−2 s" }), el("span", { text: "now" }));
  const txLegend = el("div", { class: "legend" }, [0, 1, 2, 3, 4, 5, 6].map((c) =>
    el("span", {}, el("span", { class: "key", style: `background: rgb(${colors.tx[c].join(",")})` }), c === 6 ? "6+" : String(c))),
    el("span", {}, el("span", { class: "key", style: "background: #fab219" }), "⚠ saturated (clip rail)"),
    el("span", { class: "sub", text: "crossings per 20 ms bin, reconstructed from released z-scores; no spike times" }));
  const sbpLegend = el("div", { class: "legend" }, [-3, -1.5, 0, 1.5, 3].map((z) => {
    const byte = (Math.round(z * 127 / 4) + 256) % 256;
    return el("span", {}, el("span", { class: "key", style: `background: rgb(${colors.sbp[byte].join(",")})` }), `${z > 0 ? "+" : ""}${z}σ`);
  }), el("span", { class: "sub", text: "block z-scored spike-band power" }));

  function electrodeAt(canvas, event) {
    const box = canvas.getBoundingClientRect();
    const row = Math.floor(((event.clientY - box.top) / box.height) * ROWS);
    const bin = Math.floor(((event.clientX - box.left) / box.width) * WINDOW);
    for (let e = 0; e < N_E; e++) if (ROW_OF(e) === row) return { e, bin };
    return null;
  }
  for (const [canvas, kind] of [[txCanvas, "tx"], [sbpCanvas, "sbp"]]) {
    canvas.addEventListener("pointermove", (event) => {
      const hit = electrodeAt(canvas, event);
      const t = s.trial;
      if (!hit || !t) return hideTooltip();
      const bin = t.lastBin - (WINDOW - 1 - hit.bin);
      const rows = [[`electrode ${hit.e}`, `${ARRAY_NAMES[Math.floor(hit.e / 64)]} · click to pin`]];
      if (bin >= 0 && bin <= t.lastBin) {
        const c = t.tx[bin * N_E + hit.e], z = t.sbp[bin * N_E + hit.e] / (127 / 4);
        rows.push([c === SAT ? "saturated" : String(c), "TX count"], [fmt(z, 2), "SBP z"]);
      }
      showTooltip(event, rows);
    });
    canvas.addEventListener("pointerleave", hideTooltip);
    canvas.addEventListener("click", (event) => { const hit = electrodeAt(canvas, event); if (hit) pin(hit.e); });
  }

  // ---- electrode inspector ----------------------------------------------------------------
  const electrodeSelect = el("select", {}, Array.from({ length: N_E }, (_, e) =>
    el("option", { value: e, text: `${e} · ${ARRAY_NAMES[Math.floor(e / 64)]}` })));
  const pinsBox = el("div", { class: "pins" });
  function pin(e) {
    if (s.pins.includes(e)) return;
    s.pins = [...s.pins, e].slice(-MAX_PINS);
    renderPins();
  }
  function renderPins() {
    pinsBox.replaceChildren(...(s.pins.length ? s.pins.map((e) => {
      const txC = el("canvas", { width: WINDOW * 3, height: 60, class: "trace" });
      const sbpC = el("canvas", { width: WINDOW * 3, height: 60, class: "trace" });
      const stats = el("span", { class: "sub" });
      const node = el("div", { class: "pin card", "data-e": e },
        el("div", { class: "row" }, el("strong", { text: `Electrode ${e}` }),
          el("span", { class: "sub", text: `${ARRAY_NAMES[Math.floor(e / 64)]} · TX feature ${e} · SBP feature ${256 + e}` }), stats,
          el("button", { onclick: () => { s.pins = s.pins.filter((x) => x !== e); renderPins(); } }, "Unpin")),
        el("div", { class: "sub", text: "Threshold crossings (count per bin)" }), txC,
        el("div", { class: "sub", text: "Spike-band power (z)" }), sbpC);
      node.refs = { txC, sbpC, stats };
      return node;
    }) : [el("div", { class: "sub", text: "Click a row in the raster or heatmap, or pick an electrode, to watch it stream." })]));
    s.dirty = true;
  }

  // ---- sentence, phonemes, lag -----------------------------------------------------------
  const sentence = el("div", { class: "card sentence" });
  const phonemes = el("div", { class: "card" });
  const lagBox = el("div", { class: "card" });
  const trialHeader = el("div", { class: "row" });

  function renderSentence() {
    const t = s.trial;
    if (!t) { fill(sentence, el("div", { class: "sub", text: "Load a pipeline and play a trial." })); return; }
    const showRef = t.reference && !s.blind;
    const refLine = t.reference ? (showRef ? el("div", { class: "reference", text: t.reference }) : el("div", { class: "sub", text: "(reference hidden: blind mode)" }))
      : el("div", { class: "sub", text: "(no reference: unlabeled trial)" });
    let live;
    if (!t.words.length) live = el("div", { class: "words muted", text: t.final ? "(empty output)" : "…" });
    else if (showRef && !t.final) live = diffLine(t.reference, t.words.join(" "), { prefix: true });
    else if (showRef && t.final) live = diffLine(t.reference, t.words.join(" "));
    else live = el("div", { class: "words" }, t.words.map((w, i) => [el("span", { class: `w${t.prevWords[i] === w ? "" : " new"}`, text: w }), " "]));
    fill(sentence, 
      el("div", { class: "sub", text: "Sentence cue shown to the participant (copy task)" }), refLine,
      el("div", { class: "sub", text: t.final ? "Final decoded text" : "Decoding (partial output)" }), live,
      el("div", { class: "row sub" },
        el("span", { text: `revisions so far: ${t.revisions}` }),
        t.wer ? el("strong", { text: `trial WER ${fmt(t.wer.percent, 1)}% (${t.wer.edits}/${t.wer.ref_words})` }) : null,
        t.endpointToFinal != null ? el("span", { text: `trial end → final text ${fmt(t.endpointToFinal, 1)} ms` }) : null));
  }

  function renderPhonemes() {
    const t = s.trial;
    if (!t) { fill(phonemes, ); return; }
    const top = t.top || [];
    fill(phonemes, el("div", { class: "sub", text: "Greedy phoneme stream (CTC-collapsed; | = word boundary)" }),
      el("div", { class: "phones mono" }, t.phones.slice(-48).map((p) => el("span", { class: p === "|" ? "phone sep" : "phone", text: p }))),
      el("div", { class: "sub", text: "Latest frame: top-3 class probabilities" }),
      el("div", {}, top.map(([name, p]) => el("div", { class: "prob" }, el("span", { class: "mono", text: name.padEnd(5) }),
        el("span", { class: "bar", style: `width: ${Math.round(p * 160)}px` }), el("span", { class: "sub", text: fmt(p, 2) })))));
  }

  function renderLag() {
    const t = s.trial;
    if (!t) { fill(lagBox, ); return; }
    const lags = t.lags, n = lags.length;
    const sorted = [...lags].sort((a, b) => a - b);
    const q = (p) => (n ? sorted[Math.min(n - 1, Math.floor(p * n))] : null);
    const w = 320, h = 70, hi = Math.max(20, ...lags.slice(-120));
    const recent = lags.slice(-120);
    const points = recent.map((v, i) => `${(i / 119) * w},${h - (v / hi) * (h - 6)}`).join(" ");
    fill(lagBox, 
      el("div", { class: "sub", text: "Frame processing lag (frame window available → decoded output); not word latency" }),
      svg("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", height: h, preserveAspectRatio: "none" },
        svg("line", { class: "grid", x1: 0, x2: w, y1: h - 0.5, y2: h - 0.5 }),
        svg("polyline", { points, fill: "none", stroke: "var(--series-1)", "stroke-width": 2 })),
      el("div", { class: "row sub" }, el("span", { text: `latest ${fmt(lags[n - 1], 1)} ms` }),
        el("span", { text: `p50 ${fmt(q(0.5), 1)} · p95 ${fmt(q(0.95), 1)} · max ${fmt(sorted[n - 1], 1)} ms` }),
        el("span", { text: `axis 0–${fmt(hi, 0)} ms` }),
        t.timingValid ? el("span", { class: "badge", text: "timing valid (1×, uncontended)" })
          : el("span", { class: "badge warn", text: t.speed !== 1 ? `display only (${t.speed}×)` : "contended: benchmark running" })),
      t.summary ? el("div", { class: "sub", text: `trial summary: first output ${fmt(t.summary.first_output_elapsed_ms / 1000, 2)} s after start · `
        + `${t.summary.revision_edits} revision edits · measured lag p95 ${fmt(t.summary.paced.frame_window_to_output_ms.p95, 1)} ms`
        + (t.droppedBins ? ` · ${t.droppedBins} display bins dropped` : "") }) : null);
  }

  // ---- drawing -----------------------------------------------------------------------------
  const txCtx = txCanvas.getContext("2d"), sbpCtx = sbpCanvas.getContext("2d");
  const txImage = txCtx.createImageData(WINDOW, ROWS), sbpImage = sbpCtx.createImageData(WINDOW, ROWS);
  function paint(image, ctx, values, lut, fallback) {
    const t = s.trial, data = image.data;
    for (let i = 0; i < data.length; i += 4) { data[i] = colors.surface[0]; data[i + 1] = colors.surface[1]; data[i + 2] = colors.surface[2]; data[i + 3] = 255; }
    for (let a = 1; a < 4; a++) {
      for (let g = 0; g < GAP; g++) {
        const row = ROW_OF(a * 64) - GAP + g;
        for (let x = 0; x < WINDOW; x++) { const o = (row * WINDOW + x) * 4; data[o] = colors.grid[0]; data[o + 1] = colors.grid[1]; data[o + 2] = colors.grid[2]; }
      }
    }
    if (t) {
      for (let x = 0; x < WINDOW; x++) {
        const bin = t.lastBin - (WINDOW - 1 - x);
        if (bin < 0 || !t.received[bin]) continue;
        for (let e = 0; e < N_E; e++) {
          const v = values[bin * N_E + e];
          const c = fallback && fallback.has(e) ? colors.muted : lut[v & 255];
          const o = (ROW_OF(e) * WINDOW + x) * 4;
          data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2];
        }
      }
    }
    ctx.putImageData(image, 0, 0);
  }
  function drawActivity() {
    const t = s.trial;
    activity.replaceChildren(svg("line", { class: "grid", x1: 0, x2: WINDOW * 4, y1: 89.5, y2: 89.5 }));
    if (!t) return;
    const means = ARRAY_NAMES.map(() => []);
    let hi = 0.5;
    for (let x = 0; x < WINDOW; x++) {
      const bin = t.lastBin - (WINDOW - 1 - x);
      for (let a = 0; a < 4; a++) {
        let sum = 0, n = 0;
        if (bin >= 0 && t.received[bin]) {
          for (let e = a * 64; e < a * 64 + 64; e++) { const v = t.tx[bin * N_E + e]; if (v !== SAT && !t.fallback.has(e)) { sum += v; n++; } }
        }
        const m = n ? sum / n : null;
        means[a].push(m);
        if (m != null) hi = Math.max(hi, m);
      }
    }
    means.forEach((series, a) => {
      const pts = series.map((m, x) => (m == null ? null : `${x * 4},${88 - (m / hi) * 80}`)).filter(Boolean).join(" ");
      if (pts) activity.append(svg("polyline", { points: pts, fill: "none", stroke: `var(--series-${a + 1})`, "stroke-width": 2,
                                                 "vector-effect": "non-scaling-stroke" }));
    });
  }
  function drawPins() {
    const t = s.trial;
    for (const node of pinsBox.children) {
      if (!node.refs) continue;
      const e = Number(node.dataset.e);
      const { txC, sbpC, stats } = node.refs;
      const cw = txC.width, ch = txC.height, bw = cw / WINDOW;
      const tctx = txC.getContext("2d"), sctx = sbpC.getContext("2d");
      tctx.clearRect(0, 0, cw, ch); sctx.clearRect(0, 0, cw, ch);
      if (!t) continue;
      let total = 0, n = 0;
      tctx.fillStyle = colors.series[0];
      sctx.strokeStyle = colors.series[1]; sctx.lineWidth = 2; sctx.beginPath();
      sctx.fillStyle = css("--grid"); sctx.fillRect(0, ch / 2, cw, 1);
      let started = false;
      for (let x = 0; x < WINDOW; x++) {
        const bin = t.lastBin - (WINDOW - 1 - x);
        if (bin < 0 || !t.received[bin]) continue;
        const c = t.tx[bin * N_E + e];
        if (c === SAT) { tctx.fillStyle = "#fab219"; tctx.fillRect(x * bw, 0, Math.max(1, bw - 1), ch); tctx.fillStyle = colors.series[0]; }
        else if (c > 0) { const hgt = Math.min(1, c / 6) * (ch - 2); tctx.fillRect(x * bw, ch - hgt, Math.max(1, bw - 1), hgt); total += c; n++; }
        else n++;
        const z = t.sbp[bin * N_E + e] / (127 / 4);
        const y = ch / 2 - (Math.max(-4, Math.min(4, z)) / 4) * (ch / 2 - 2);
        if (!started) { sctx.moveTo(x * bw, y); started = true; } else sctx.lineTo(x * bw, y);
      }
      sctx.stroke();
      const current = t.lastBin >= 0 ? t.tx[t.lastBin * N_E + e] : 0;
      stats.textContent = `now ${current === SAT ? "saturated" : `${current / 0.02} Hz`} · window mean ${fmt(n ? total / n / 0.02 : 0, 0)} Hz (TX 0–6+ axis; SBP ±4σ)`;
    }
  }
  let raf = null;
  function frame() {
    if (s.dirty) {
      s.dirty = false;
      const t = s.trial;
      paint(txImage, txCtx, t ? t.tx : null, colors.tx, t ? t.fallback : null);
      paint(sbpImage, sbpCtx, t ? t.sbp : null, colors.sbp, null);
      drawActivity(); drawPins();
    }
    raf = requestAnimationFrame(frame);
  }

  // ---- events ------------------------------------------------------------------------------
  function onEvent(event) {
    const t = s.trial;
    if (event.type === "state") {
      s.engine = event;
      const p = event.pipeline;
      status.textContent = `Engine ${event.state}${p ? ` · ${p.name} (${p.acoustic} + ${p.lm}, ac ${p.decode.acoustic_scale} / bp ${p.decode.blank_penalty})` : ""}`
        + `${event.lm_expected_rss_gb ? ` · LM ~${event.lm_expected_rss_gb} GB RAM` : ""}${event.message ? ` · ${event.message}` : ""}`;
      playButton.disabled = sessionButton.disabled = randomButton.disabled = event.state !== "ready";
    } else if (event.type === "error") {
      errorBox.textContent = event.message;
    } else if (event.type === "trial_start") {
      s.trial = { playId: event.play_id, nBins: event.n_bins, tx: new Uint8Array(event.n_bins * N_E), sbp: new Int8Array(event.n_bins * N_E),
        received: new Uint8Array(event.n_bins), lastBin: -1, words: [], prevWords: [], revisions: 0, phones: [], lastArg: null,
        lags: [], reference: event.reference, final: false, fallback: new Set(event.tx_fallback), speed: event.speed,
        timingValid: event.timing_valid, summary: null, wer: null, endpointToFinal: null, top: [] };
      fill(trialHeader, el("strong", { text: `${event.item.session} · ${event.item.trial_key}` }),
        el("span", { class: "badge", text: event.partition }),
        el("span", { class: "badge", title: "Sentence source corpus of this recording block", text: `source: ${event.corpus || "unknown"}` }),
        flagBadges(event.flags),
        el("span", { class: "sub", text: `${fmt(event.n_bins * 0.02, 1)} s · ${event.n_frames} frames · trace ${event.trace}` }),
        event.tx_fallback.length ? el("span", { class: "badge warn", text: `${event.tx_fallback.length} electrodes not count-reconstructable` }) : null);
      renderSentence(); renderPhonemes(); renderLag(); s.dirty = true;
    } else if (!t || event.play_id !== t.playId) {
      return;
    } else if (event.type === "bin") {
      t.tx.set(bytes(event.tx), event.bin * N_E);
      t.sbp.set(new Int8Array(bytes(event.sbp).buffer), event.bin * N_E);
      t.received[event.bin] = 1;
      t.lastBin = Math.max(t.lastBin, event.bin);
      s.dirty = true;
    } else if (event.type === "frame") {
      const prev = t.words;
      if (!prev.every((w, i) => event.words[i] === w)) t.revisions++;
      t.prevWords = prev; t.words = event.words;
      if (event.phone !== t.lastArg && event.phone) t.phones.push(event.phone);
      t.lastArg = event.phone;
      t.top = event.top;
      t.lags.push(event.lag_ms);
      renderSentence(); renderPhonemes(); renderLag();
    } else if (event.type === "final") {
      t.prevWords = t.words; t.words = event.words; t.final = true; t.endpointToFinal = event.endpoint_to_final_ms;
      renderSentence();
    } else if (event.type === "trial_end") {
      t.summary = event.summary; t.wer = event.wer || null; t.droppedBins = event.dropped_bins;
      if (event.status === "stopped") errorBox.textContent = "Stopped (trace kept, marked incomplete)";
      renderSentence(); renderLag();
    }
  }
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/live`);
    s.ws = ws;
    ws.onmessage = (m) => onEvent(JSON.parse(m.data));
    ws.onclose = () => { if (!s.closed) { status.textContent = "Disconnected; retrying…"; setTimeout(connect, 2000); } };
  }

  // ---- layout ------------------------------------------------------------------------------
  main.replaceChildren(
    el("h1", { text: "Live" }),
    el("div", { class: "sub", text: "Replays released features at their 20 ms cadence through the loaded pipeline (the same code path as paced benchmarks). "
      + "Every played trial is recorded under results/harness/live/. Trial ends are the dataset's (oracle) endpoints." }),
    el("div", { class: "filters" }, el("label", {}, "Pipeline ", presetSelect), el("label", {}, custom, " custom"), loadButton, unloadButton, status),
    customBox,
    el("div", { class: "filters" }, splitSelect, sessionSelect, trialSelect, speedSelect, playButton, sessionButton, randomButton, stopButton,
      el("label", {}, blind, " blind (hide reference)"), errorBox),
    trialHeader,
    el("div", { class: "live-grid" },
      el("div", { class: "live-neural" },
        el("div", { class: "card" }, el("div", { class: "panel-title", text: "Threshold crossings" }), txLegend,
          el("div", { class: "canvas-wrap" }, arrayLabels(), txCanvas), timeAxis.cloneNode(true)),
        el("div", { class: "card" }, el("div", { class: "panel-title", text: "Spike-band power" }), sbpLegend,
          el("div", { class: "canvas-wrap" }, arrayLabels(), sbpCanvas), timeAxis.cloneNode(true)),
        el("div", { class: "card" }, el("div", { class: "panel-title", text: "Array activity (mean crossings per electrode per bin)" }),
          activityLegend, activity, timeAxis)),
      el("div", { class: "live-side" }, sentence, phonemes, lagBox)),
    el("h2", { text: "Electrode inspector" }),
    el("div", { class: "filters" }, electrodeSelect, el("button", { onclick: () => pin(Number(electrodeSelect.value)) }, "Pin electrode"),
      el("span", { class: "sub", text: `up to ${MAX_PINS} at once` })),
    pinsBox);
  playButton.disabled = sessionButton.disabled = randomButton.disabled = true;
  renderSentence(); renderPins();
  await loadSessions();
  connect();
  const scheme = matchMedia("(prefers-color-scheme: dark)");
  const onScheme = () => { colors = palettes(); s.dirty = true; };
  scheme.addEventListener("change", onScheme);
  s.dirty = true;
  frame();
  return () => { s.closed = true; if (s.ws) s.ws.close(); cancelAnimationFrame(raf); scheme.removeEventListener("change", onScheme); hideTooltip(); };
}
