// Replay one trial's recorded partial outputs (optionally two runs side by side).
// Accuracy traces are unpaced cached-logit runs: they replay on the NOMINAL input schedule
// (frame f at (window + f*stride + lookahead) * 20 ms), which excludes compute time.
import { diffLegend, diffLine, el, fmt, getJSON } from "../ui.js";
import { CATEGORY_TEXT } from "./run.js";

function phonemeLine(entries) {
  // Greedy phonemes aligned to the reference: substitutions and insertions washed, deletions struck.
  return el("div", { class: "ph" }, entries.map((g) => [
    g.op === "del" ? el("span", { class: "p del", title: "missing from greedy output", text: g.ref })
      : el("span", { class: `p ${g.op === "ok" ? "" : g.op}`, title: g.op === "sub" ? `reference: ${g.ref}` : g.op === "ins" ? "extra phoneme" : null,
                     text: g.hyp }), " "]));
}

function wordColumns(detail) {
  if (!detail) return null;
  if (detail.status !== "ok") {
    return el("div", { class: "sub", text: "Reference words and reference phoneme groups differ in count; showing the raw sequences only. "
      + `Reference phonemes: ${detail.ref_phones.join(" ")} · greedy: ${detail.greedy.map((g) => g.p).join(" ")}` });
  }
  const cards = [];
  const inserted = (after) => detail.insertions.filter((x) => x.after === after).map((x) =>
    el("div", { class: "wordcol" }, el("span", { class: "cat cat-lm_inserted", text: "LM inserted" }),
      el("div", { class: "k", text: "reference" }), el("div", { class: "rw muted", text: "—" }),
      el("div", { class: "k", text: "LM word" }), el("div", { class: "lmw", text: x.word }),
      el("div", { class: "ph muted", text: (x.lm_phones || []).join(" ") })));
  cards.push(...inserted(-1));
  detail.words.forEach((w, k) => {
    cards.push(el("div", { class: "wordcol", title: CATEGORY_TEXT[w.category][1] },
      el("span", { class: `cat cat-${w.category}`, text: CATEGORY_TEXT[w.category][0] }),
      w.t_ms != null ? el("span", { class: "sub", text: `  ≈${fmt(w.t_ms / 1000, 1)} s` }) : null,
      el("div", { class: "k", text: "reference" }), el("div", { class: "rw", text: w.ref }),
      el("div", { class: "ph", text: w.ref_phones.join(" ") }),
      el("div", { class: "k", text: `greedy phonemes${w.phone_edits ? ` · ${w.phone_edits} edit${w.phone_edits > 1 ? "s" : ""}` : ""}` }),
      phonemeLine(w.greedy),
      el("div", { class: "k", text: "LM word" }),
      el("div", { class: `lmw${w.lm_op === "ok" ? "" : " w sub"}`, text: w.lm_word || "(deleted)" }),
      el("div", { class: "ph muted", text: w.lm_word ? (w.lm_phones || ["(not in lexicon)"]).join(" ") : "" })));
    cards.push(...inserted(k));
  });
  return el("div", {},
    el("div", { class: "legend" }, Object.entries(CATEGORY_TEXT).map(([key, [name]]) => el("span", { class: `cat cat-${key}`, text: name })),
      el("span", {}, el("span", { class: "p", style: "background: var(--wash-sub)", text: "AA" }), " substituted phoneme"),
      el("span", {}, el("span", { class: "p", style: "background: var(--wash-ins)", text: "AA" }), " extra phoneme"),
      el("span", {}, el("span", { class: "p del", text: "AA" }), " missing phoneme")),
    el("div", { class: "wordcols" }, cards),
    el("div", { class: "sub", text: "Greedy phonemes: per-frame argmax, repeats and blanks collapsed (a proxy for the acoustic evidence; "
      + "the LM search uses full posteriors). Times are the nominal input clock of each word's first greedy phoneme. "
      + "LM pronunciations come from the decoding lexicon." }));
}

async function load(runId, index) {
  const [{ manifest }, rows, events, phonemes] = await Promise.all([
    getJSON(`/api/runs/${encodeURIComponent(runId)}`),
    getJSON(`/api/runs/${encodeURIComponent(runId)}/trials`),
    getJSON(`/api/runs/${encodeURIComponent(runId)}/trials/${index}/events?trace=accuracy`),
    getJSON(`/api/runs/${encodeURIComponent(runId)}/trials/${index}/phonemes`).catch(() => null)]);
  const row = rows[index];
  const model = manifest.pipeline.model;
  const lookahead = manifest.pipeline.preprocess.effective.smooth_lookahead ?? 0;
  const outputs = events.records.filter((r) => r.record === "output");
  const partials = outputs.filter((o) => o.kind === "partial");
  const final = outputs.find((o) => o.kind === "final");
  // Paced (schema 2) traces carry measured output times; cached-logit traces get the nominal schedule.
  const recorded = events.schema_version === 2;
  const frameMs = recorded ? (f) => partials[f].output_ready_ns / 1e6
    : (f) => (model.patch_size + f * model.patch_stride + lookahead) * 20;
  const endMs = recorded && final ? final.output_ready_ns / 1e6 : row.n_bins * 20;
  return { manifest, row, partials, final, frameMs, endMs, recorded, phonemes };
}

function revisions(partials) {
  const out = [];
  partials.forEach((p, f) => {
    const prev = f ? partials[f - 1].words : [];
    const isAppend = prev.every((w, i) => p.words[i] === w);
    if (!isAppend) out.push({ frame: f, before: prev.join(" "), after: p.words.join(" ") });
  });
  return out;
}

export async function trialView(main, runId, index, otherRunId) {
  const panes = await Promise.all([load(runId, index), otherRunId ? load(otherRunId, index) : null].filter(Boolean));
  const endMs = Math.max(...panes.map((p) => p.endMs));
  const clock = el("span", { class: "mono" });
  const slider = el("input", { type: "range", min: 0, max: endMs, step: 20, value: endMs, style: "flex: 1" });
  const views = panes.map((pane) => {
    const live = el("div", { class: "words" });
    const revs = revisions(pane.partials);
    const revRows = revs.map((r) => el("tr", {},
      el("td", { class: "num mono", text: `${fmt(pane.frameMs(r.frame) / 1000, 2)} s` }),
      el("td", {}, el("div", { class: "sub", text: r.before || "(empty)" }), el("div", { text: r.after || "(empty)" }))));
    const card = el("div", { class: "card" },
      el("div", {}, el("strong", { text: pane.manifest.pipeline.name }), el("span", { class: "sub", text: `  ${pane.manifest.run_id}` })),
      el("div", { class: "sub", text: "Output at this time (words changed since the previous frame are highlighted)" }), live,
      el("h2", { text: "Final vs reference" }), diffLine(pane.row.ref, pane.row.hyp),
      pane.phonemes ? el("h2", { text: "Phonemes vs words, per reference word" }) : null, wordColumns(pane.phonemes),
      el("h2", { text: `Revisions (${revs.length})` }),
      el("div", { class: "table-wrap" }, el("table", {}, el("tbody", {}, revRows))));
    return { pane, live, card };
  });

  function show(ms) {
    clock.textContent = `t = ${fmt(ms / 1000, 2)} s (${panes[0].recorded ? "recorded paced clock" : "nominal input clock"})`;
    for (const { pane, live } of views) {
      let f = -1;
      while (f + 1 < pane.partials.length && pane.frameMs(f + 1) <= ms) f++;
      const atFinal = ms >= pane.endMs && pane.final;
      const words = atFinal ? pane.final.words : f >= 0 ? pane.partials[f].words : [];
      // Highlight what the latest update changed (finalization is compared with the last partial).
      const prev = atFinal ? (f >= 0 ? pane.partials[f].words : []) : f > 0 ? pane.partials[f - 1].words : [];
      live.replaceChildren(...(words.length ? words.map((w, i) => [el("span", { class: `w${prev[i] === w ? "" : " new"}`, text: w }), " "]).flat()
        : [el("span", { class: "muted", text: "(no output yet)" })]));
    }
  }
  let timer = null;
  const play = el("button", { class: "primary", onclick: () => {
    if (timer) { cancelAnimationFrame(timer); timer = null; play.textContent = "Play"; return; }
    const start = performance.now() - (Number(slider.value) >= endMs ? 0 : Number(slider.value));
    play.textContent = "Pause";
    const tick = () => {
      const ms = Math.min(endMs, performance.now() - start);
      slider.value = ms; show(ms);
      if (ms < endMs) timer = requestAnimationFrame(tick); else { timer = null; play.textContent = "Play"; }
    };
    timer = requestAnimationFrame(tick);
  } }, "Play");
  slider.addEventListener("input", () => show(Number(slider.value)));
  const row = panes[0].row;
  main.replaceChildren(
    el("div", { class: "sub" }, el("a", { href: `#/run/${runId}`, text: "← run" })),
    el("h1", { text: `${row.session} · ${row.trial_key}` }),
    el("div", { class: "sub", text: `${fmt(row.n_bins * 0.02, 2)} s of features · ${row.n_frames} frames · `
      + `source: ${row.corpus || "unknown"} (block ${row.block_num}) · reference: ${row.ref}` }),
    el("div", { class: "sub", text: panes[0].recorded ? "Paced run: outputs appear at their measured times; dataset trial end is the (oracle) endpoint."
      : "Cached-logit replay on the nominal input schedule: compute time is excluded; dataset trial end is the (oracle) endpoint." }),
    el("div", { class: "filters" }, play, slider, clock),
    diffLegend(),
    el("div", { style: `display: grid; grid-template-columns: repeat(${views.length}, minmax(0, 1fr)); gap: 12px;` }, views.map((v) => v.card)));
  show(endMs);
  return () => { if (timer) cancelAnimationFrame(timer); };
}
