// One run: headline tiles, per-session WER, timing check, configuration, and trials.
import { diffLegend, diffLine, el, fmt, fmtCI, flagBadges, FLAG_TEXT, getJSON, kv, shortTime, tile } from "../ui.js";
import { cdfPlot, dotPlot, lineLegend } from "../charts.js";

export async function runView(main, runId) {
  const { manifest, summary } = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  const spec = manifest.pipeline;
  const header = [
    el("h1", { text: spec.name }),
    el("div", { class: "sub", text: `${manifest.run_id} · ${manifest.tier} · ${shortTime(manifest.created)} UTC · `
      + `git ${manifest.code.head.slice(0, 8)}${manifest.code.dirty ? " (dirty)" : ""}` }),
    el("div", { class: "row" }, el("span", { class: `status-${manifest.status}`, text: manifest.status }), flagBadges(manifest.flags)),
    manifest.error ? el("div", { class: "card error", text: manifest.error }) : null,
  ];
  if (!summary) { main.replaceChildren(...header.filter(Boolean)); return; }
  const accuracy = summary.accuracy;
  const wer = accuracy.wer;
  const su = accuracy.seen_unseen;
  const sections = [
    el("div", { class: "tiles" },
      tile("WER (pooled)", `${fmt(wer.percent, 2)}%`, `95% CI ${fmtCI(wer.ci95)} · ${wer.edits}/${wer.ref_words} words`),
      tile("Seen / unseen text WER", `${fmt(su.seen.percent, 1)} / ${fmt(su.unseen.percent, 1)}`,
        `${su.seen.n_trials} / ${su.unseen.n_trials} trials`),
      tile("Revisions per 100 final words", fmt(accuracy.revisions.revision_edits_per_100_final_words, 1),
        "ins + del + sub between partial outputs"),
      tile("Greedy PER (diagnostic)", `${fmt(accuracy.per_diagnostic.percent, 2)}%`, "acoustic logits only"),
      tile("Scope", `${summary.scope.n_trials} trials`, `${summary.scope.n_sessions} sessions · ${summary.scope.name}`)),
    el("div", { class: "sub", text: wer.method }),
    el("h2", { text: "WER by session" }),
    el("div", { class: "card" }, dotPlot(accuracy.per_session.map((s) => ({ label: s.session, value: s.percent, row: s })), {
      valueLabel: "WER %", reference: wer.percent, referenceLabel: `pooled ${fmt(wer.percent, 2)}%`,
      tooltip: (r) => [[`${fmt(r.value, 2)}%`, r.label], [`${r.row.edits}/${r.row.ref_words}`, "edits / words"], [`${r.row.n_trials}`, "trials"]],
    })),
  ];
  if (summary.paced) sections.push(el("h2", { text: "Paced 1× timing (measured)" }), pacedSection(summary, runId));
  else sections.push(el("h2", { text: "Timing check" }), timingSection(summary.timing_check, runId));
  sections.push(el("h2", { text: "Configuration" }), configSection(manifest, accuracy));
  const trialsCard = el("div", { class: "card" }, el("div", { class: "empty", text: "Loading trials…" }));
  const errorsCard = el("div", { class: "card" }, el("div", { class: "sub", text: "Aligning phonemes and words…" }));
  const replayable = manifest.tier === "standard" || manifest.tier === "sweep";
  if (replayable) sections.push(el("h2", { text: "Where word errors come from: phonemes vs language model" }), errorsCard);
  const rescoreCard = el("div", { class: "card" });
  if (replayable) {
    sections.push(el("h2", { text: "Finalization: neural rescoring at trial end" }), rescoreCard);
    getJSON(`/api/runs/${encodeURIComponent(runId)}/rescore`).then((r) => renderRescore(rescoreCard, r))
      .catch(() => rescoreCard.replaceChildren(el("div", { class: "sub", text: `Not rescored yet: .venv/bin/python -m harness rescore ${runId}` })));
  }
  const commitCard = el("div", { class: "card" });
  if (replayable) {
    sections.push(el("h2", { text: "Word commitment: when can a word be shown as final?" }), commitCard);
    loadCommitment(commitCard, runId, false);
  }
  sections.push(el("h2", { text: "Trials" }), trialsCard);
  main.replaceChildren(...header.filter(Boolean), ...sections);
  const attribution = replayable
    ? getJSON(`/api/runs/${encodeURIComponent(runId)}/errors`).catch(() => null) : Promise.resolve(null);
  attribution.then((result) => renderAttribution(errorsCard, result));
  renderTrials(trialsCard, runId, attribution);
}

function timingSection(timing, runId) {
  if (!timing || timing.skipped) {
    return el("div", { class: "card sub", text: `Skipped: ${timing ? timing.skipped : "not run"}` });
  }
  const sd = timing.structural_delay, svc = timing.service_ms, sim = timing.simulated_lag, eq = timing.equivalence;
  const exceed = Object.entries(sim.exceedances).map(([b, v]) => `${v.n}/${v.denominator} > ${Number(b)} ms`).join(" · ");
  const card = el("div", { class: "card" },
    el("div", { class: "sub", text: sd.label }),
    el("div", { class: "tiles" },
      tile("First possible output", `${fmt(sd.first_output_ms, 0)} ms`, `patch ${sd.patch.window_bins} bins + lookahead ${fmt(sd.lookahead_ms, 0)} ms`),
      tile("Update cadence", `${fmt(sd.update_cadence_ms, 0)} ms`, `frame-entry wait ${fmt(sd.frame_entry_wait_ms.min, 0)}–${fmt(sd.frame_entry_wait_ms.max, 0)} ms`),
      tile("Smoothing effective delay", `${fmt(sd.smoothing.centroid_delay_ms, 1)} ms`, `${sd.smoothing.support_bins}-bin kernel centroid`),
      tile("Step time p99 / max", `${fmt(svc.per_bin.p99, 2)} / ${fmt(svc.per_bin.max, 1)} ms`,
        `${fmt(100 * svc.frac_bins_over_bin, 2)}% of bins > ${svc.bin_budget_ms} ms`),
      tile("Frame cycle p99", `${fmt(svc.per_frame_cycle.p99, 2)} ms`,
        `${fmt(100 * svc.frac_cycles_over_cycle, 2)}% of cycles > ${svc.cycle_budget_ms} ms`),
      tile("Simulated lag p95 / max", `${fmt(sim.frame_window_to_output_ms.p95, 2)} / ${fmt(sim.frame_window_to_output_ms.max, 1)} ms`, exceed)),
    el("div", { class: "sub", text: `${sim.label}. Equivalence: max streamed-vs-batched logit error ${eq.max_logit_error?.toExponential(2)} over ${eq.n_trials} trials; `
      + `${eq.final_text_mismatches.length}/${eq.final_text_compared} final texts differ (near-ties).` }));
  const chart = el("div", {}, el("div", { class: "sub", text: "Loading lag distribution…" }));
  card.append(el("h2", { text: "Simulated frame-window-to-output lag (CDF)" }), chart);
  getJSON(`/api/runs/${encodeURIComponent(runId)}/timing`).then((q) => {
    chart.replaceChildren(cdfPlot([{ label: "simulated", quantiles: q.frame_window_to_output_ms, color: "var(--series-1)" }],
      { valueLabel: "lag ms (simulated)", budgets: [200, 500, 1500] }),
      el("div", { class: "sub", text: "Budget lines (200/500/1500 ms) are frame processing lag thresholds, not word latency; they appear only when in range." }));
  }).catch((e) => chart.replaceChildren(el("div", { class: "error", text: e.message })));
  return card;
}

function pacedSection(summary, runId) {
  const paced = summary.paced, cal = summary.sim_calibration, sd = paced.structural_delay;
  const lag = paced.frame_window_to_output_ms;
  const exceed = Object.entries(paced.exceedances).map(([b, v]) => `${v.n}/${v.denominator} > ${Number(b)} ms`).join(" · ");
  const card = el("div", { class: "card" },
    el("div", { class: "sub", text: paced.label }),
    el("div", { class: "tiles" },
      tile("Measured lag p50 / p95", `${fmt(lag.p50, 2)} / ${fmt(lag.p95, 2)} ms`, "frame window available → coordinator output"),
      tile("Measured lag p99 / max", `${fmt(lag.p99, 2)} / ${fmt(lag.max, 1)} ms`, exceed),
      tile("Endpoint → final p95", `${fmt(paced.endpoint_to_final_ms.p95, 2)} ms`, "oracle trial end → final text"),
      tile("Busy per bin p99", `${fmt(paced.busy_per_bin_ms.p99, 2)} ms`, "step start → last output"),
      tile("Input queue p95", `${fmt(paced.input_queue_ms.p95, 3)} ms`, "scheduled arrival → processing start"),
      tile("First possible output", `${fmt(sd.first_output_ms, 0)} ms`, `cadence ${fmt(sd.update_cadence_ms, 0)} ms (structural)`)));
  if (cal && !cal.skipped) {
    const keys = ["p50", "p95", "p99", "max"];
    card.append(el("h2", { text: "Simulation calibration" }),
      el("div", { class: "sub" }, "Measured here vs the queue simulation of standard run ",
        el("a", { href: `#/run/${cal.standard_run}`, text: cal.standard_run }), " on the same trials. ", cal.note),
      el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, ["", ...keys].map((k, i) => el("th", { class: i ? "num" : "", text: k })))),
        el("tbody", {}, [["Measured lag (paced)", cal.measured], ["Simulated lag (standard)", cal.simulated],
          ["Measured − simulated", cal.measured_minus_simulated_ms], ["Busy per bin (paced)", cal.paced_busy_per_bin_ms],
          ["Service per bin (unpaced)", cal.unpaced_service_per_bin_ms]].map(([label, values]) =>
          el("tr", {}, el("td", { text: label }), keys.map((k) => el("td", { class: "num", text: `${fmt(values[k], 2)} ms` }))))))));
  } else if (cal) {
    card.append(el("div", { class: "sub", text: `Calibration skipped: ${cal.skipped}` }));
  }
  const chart = el("div", {}, el("div", { class: "sub", text: "Loading lag distribution…" }));
  card.append(el("h2", { text: "Frame-window-to-output lag (CDF)" }), chart);
  const series = [{ label: "measured (paced)", color: "var(--series-1)", url: `/api/runs/${encodeURIComponent(runId)}/timing` }];
  if (cal && !cal.skipped) series.push({ label: "simulated (standard run)", color: "var(--series-2)",
                                         url: `/api/runs/${encodeURIComponent(cal.standard_run)}/timing` });
  Promise.all(series.map((s) => getJSON(s.url))).then((results) => {
    const withData = series.map((s, i) => ({ ...s, quantiles: results[i].frame_window_to_output_ms }));
    chart.replaceChildren(lineLegend(withData), cdfPlot(withData, { valueLabel: "lag ms", budgets: [200, 500, 1500] }));
  }).catch((e) => chart.replaceChildren(el("div", { class: "error", text: e.message })));
  return card;
}

function configSection(manifest, accuracy) {
  const spec = manifest.pipeline;
  const pre = spec.preprocess;
  const decode = Object.entries(spec.decode).map(([k, v]) => `${k}=${v}`).join("  ");
  return el("div", { class: "card" },
    kv([
      ["Acoustic", `${spec.acoustic.name} (${spec.acoustic.checkpoint_dir})`],
      ["LM graph", `${spec.lm.name} (${spec.lm.graph_dir})`],
      ["Decode", decode],
      ["Decode tuned on", spec.decode_tuned_on ? JSON.stringify(spec.decode_tuned_on) : "unrecorded"],
      ["Preprocess (trained)", JSON.stringify(pre.trained)],
      ["Preprocess (effective)", JSON.stringify(pre.effective)],
      ["Pipeline hash", manifest.identity.pipeline_hash.slice(0, 16)],
      ["Endpoint / commitment", `${manifest.constants.endpoint} · ${manifest.constants.commitment_policy}`],
      ["LM compute", accuracy.lm_compute ? `per frame p95 ${fmt(accuracy.lm_compute.per_frame_ms.p95, 3)} ms · finalize p95 ${fmt(accuracy.lm_compute.finalize_ms.p95, 2)} ms (${accuracy.lm_compute.label})` : "see paced timing"],
      ["Flags", manifest.flags.map((f) => `${f}: ${FLAG_TEXT[f] || ""}`).join("; ") || "none"],
    ]),
    el("details", {}, el("summary", { text: "Full manifest" }), el("pre", { class: "json", text: JSON.stringify(manifest, null, 2) })));
}

export const CATEGORY_TEXT = {
  both_correct: ["both correct", "greedy phonemes exact and LM word correct"],
  lm_fixed: ["LM fixed", "greedy phonemes wrong, but the LM chose the right word"],
  lm_introduced: ["LM introduced", "greedy phonemes exactly right, but the LM chose a different word"],
  both_wrong: ["both wrong", "greedy phonemes wrong and the LM word wrong"],
  lm_inserted: ["LM inserted", "an extra word with no reference word"],
};

function renderAttribution(card, result) {
  if (!result) { card.replaceChildren(el("div", { class: "sub", text: "Attribution unavailable for this run." })); return; }
  const a = result.summary, c = a.counts;
  const acousticWrong = c.lm_fixed + c.both_wrong, errors = c.lm_introduced + c.both_wrong + c.lm_inserted;
  card.replaceChildren(
    el("div", { class: "tiles" },
      tile("Words with exact greedy phonemes", `${fmt(a.pct_words_greedy_phonemes_exact, 1)}%`, `${a.reference_words - acousticWrong}/${a.reference_words} reference words`),
      tile("Acoustic word errors the LM fixed", `${fmt(a.pct_acoustic_word_errors_fixed_by_lm, 1)}%`, `${c.lm_fixed}/${acousticWrong} words with wrong phonemes`),
      tile("Word errors despite exact phonemes", `${fmt(a.pct_word_errors_with_exact_phonemes, 1)}%`, `${c.lm_introduced}/${errors} word errors (LM introduced)`),
      tile("Word errors where both were wrong", `${fmt(100 * c.both_wrong / errors, 1)}%`, `${c.both_wrong}/${errors}; plus ${c.lm_inserted} insertions`)),
    el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, ["Category", "Words", "Meaning"].map((h, i) => el("th", { class: i === 1 ? "num" : "", text: h })))),
      el("tbody", {}, Object.entries(CATEGORY_TEXT).map(([key, [name, meaning]]) => el("tr", {},
        el("td", {}, el("span", { class: `cat cat-${key}`, text: name })), el("td", { class: "num", text: c[key] }),
        el("td", { class: "sub", text: meaning })))))),
    el("div", { class: "sub", text: `${a.label}. ${a.trials_skipped ? `${a.trials_skipped} trials skipped (reference words and phoneme groups differ in count). ` : ""}`
      + "Open a trial to see phonemes and words side by side." }));
}

function renderRescore(card, { record, summary }) {
  const d = summary.delta_vs_one_best, w = summary.wer;
  card.replaceChildren(
    el("div", { class: "tiles" },
      tile("Rescored WER (utterance-final)", `${fmt(w.percent, 2)}%`, `95% CI ${fmtCI(w.ci95)} · ${record.rescorer.model}`),
      tile("Change vs 1-best", `${d.points >= 0 ? "+" : ""}${fmt(d.points, 2)} pts`, `95% CI ${fmtCI(d.ci95)} · ${d.trials_better} better / ${d.trials_worse} worse`),
      tile("N-best oracle WER", `${fmt(summary.oracle_wer, 2)}%`, `best candidate in each ${record.rescorer.nbest}-best list`),
      tile("Neural scoring per utterance", `${fmt(summary.latency.neural_scoring_ms_per_utterance.p50, 0)} / ${fmt(summary.latency.neural_scoring_ms_per_utterance.p95, 0)} ms`,
        `p50 / p95; n-best extraction p95 ${fmt(summary.latency.nbest_finalize_ms.p95, 1)} ms (not summed)`)),
    el("div", { class: "sub", text: `${record.label}. Weights ${JSON.stringify(record.rescorer.weights)}; unseen text ${fmt(summary.seen_unseen.unseen.percent, 2)}%. `
      + `Self-check: identity weights reproduce the run's 1-best on ${summary.self_check.identity_weights_reproduce_run_1best}/${summary.self_check.n_trials} trials.` }),
    flagBadges(record.flags.filter((f) => f.startsWith("rescorer"))));
}

async function loadCommitment(card, runId, compute) {
  card.replaceChildren(el("div", { class: "sub", text: compute ? "Evaluating policies on the recorded partial outputs (about 20 s)…" : "Loading…" }));
  let result;
  try {
    result = await getJSON(`/api/runs/${encodeURIComponent(runId)}/commitment${compute ? "?compute=1" : ""}`);
  } catch {
    card.replaceChildren(el("div", { class: "row" },
      el("span", { class: "sub", text: "Not evaluated yet. Replays each online commitment policy over this run's recorded partial outputs (CPU only)." }),
      el("button", { class: "primary", onclick: () => loadCommitment(card, runId, true) }, "Evaluate commitment policies")));
    return;
  }
  const rows = result.policies.filter((p) => p.online && p.policy !== "none" && p.delta_wer_vs_none)
    .sort((a, b) => a.waiting_ms.p50 - b.waiting_ms.p50);
  const none = result.policies.find((p) => p.policy === "none");
  card.replaceChildren(
    el("div", { class: "sub", text: "Online policies see only the partial outputs so far; committed words are never retracted. "
      + "Cost is the paired WER change against showing nothing until the (oracle) trial end. Waiting is measured from a word's first appearance "
      + "on the nominal input clock (a proxy; the release has no word timings)." }),
    dotPlot(rows.map((p) => ({ label: `${p.policy} · ${fmt(p.waiting_ms.p50 / 1000, 1)} s`, value: p.delta_wer_vs_none.points,
                               interval: p.delta_wer_vs_none.ci95, row: p })), {
      valueLabel: "Δ WER points vs no commitment (95% CI, sessions)", reference: 0, referenceLabel: "no cost",
      tooltip: (r) => [[`${r.value >= 0 ? "+" : ""}${fmt(r.value, 2)} pts`, r.row.policy],
                       [`${fmt(r.row.commit_errors.pct, 2)}%`, "committed words the decoder later changed"],
                       [`${fmt(r.row.waiting_ms.p50, 0)} ms`, "median wait after first appearance"]] }),
    el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, ["Policy", "WER %", "Δ vs none [95% CI]", "Commit errors", "Committed early", "Wait p50 / p95", "Final before trial end (p50)", "Revisions /100w"]
        .map((h, i) => el("th", { class: i ? "num" : "", text: h })))),
      el("tbody", {}, [none, ...rows, result.policies.find((p) => !p.online)].filter(Boolean).map((p) => el("tr", {},
        el("td", { text: p.policy }), el("td", { class: "num", text: fmt(p.wer.percent, 2) }),
        el("td", { class: "num", text: p.delta_wer_vs_none ? `${p.delta_wer_vs_none.points >= 0 ? "+" : ""}${fmt(p.delta_wer_vs_none.points, 2)} [${fmt(p.delta_wer_vs_none.ci95[0], 2)}, ${fmt(p.delta_wer_vs_none.ci95[1], 2)}]` : "–" }),
        el("td", { class: "num", text: p.commit_errors.pct == null ? "–" : `${fmt(p.commit_errors.pct, 2)}%` }),
        el("td", { class: "num", text: `${fmt(p.early_commit_pct, 1)}%` }),
        el("td", { class: "num", text: `${fmt(p.waiting_ms.p50 / 1000, 2)} / ${fmt(p.waiting_ms.p95 / 1000, 2)} s` }),
        el("td", { class: "num", text: `${fmt(p.committed_before_trial_end_ms.p50 / 1000, 2)} s` }),
        el("td", { class: "num", text: fmt(p.visible_revisions_per_100_words, 1) })))))),
    el("div", { class: "sub", text: `Evaluated ${result.created.slice(0, 16).replace("T", " ")} UTC. The hindsight row is not an online policy: it commits each word when it last changed. ${result.definitions.exposure}.` }));
}

function sourceTable(rows) {
  const by = new Map();
  for (const r of rows) {
    const key = r.corpus || "unknown";
    const a = by.get(key) || { trials: 0, edits: 0, words: 0, perfect: 0 };
    a.trials++; a.edits += r.edits; a.words += r.ref_words; a.perfect += r.edits === 0 ? 1 : 0;
    by.set(key, a);
  }
  const sorted = [...by.entries()].sort((x, y) => y[1].trials - x[1].trials);
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {}, ["Sentence source", "Trials", "Words", "WER %", "Error-free trials"].map((h, i) =>
      el("th", { class: i ? "num" : "", text: h })))),
    el("tbody", {}, sorted.map(([name, a]) => el("tr", {},
      el("td", { text: name }), el("td", { class: "num", text: a.trials }), el("td", { class: "num", text: a.words }),
      el("td", { class: "num", text: fmt(100 * a.edits / a.words, 2) }),
      el("td", { class: "num", text: `${fmt(100 * a.perfect / a.trials, 1)}%` }))))));
}

async function renderTrials(card, runId, attributionPromise) {
  const rows = await getJSON(`/api/runs/${encodeURIComponent(runId)}/trials`);
  const attribution = await attributionPromise;
  const counts = attribution ? new Map(attribution.trials.map((t) => [t.i, t.counts])) : new Map();
  const state = { errorsOnly: true, query: "", source: "", category: "" };
  const table = el("div", { class: "table-wrap" });
  const sources = [...new Set(rows.map((r) => r.corpus || "unknown"))].sort();
  function render() {
    const q = state.query.toLowerCase();
    const shown = rows.filter((r) => (!state.errorsOnly || r.edits > 0)
      && (!state.source || (r.corpus || "unknown") === state.source)
      && (!state.category || (counts.get(r.i) || {})[state.category] > 0)
      && (!q || r.session.includes(q) || r.ref.toLowerCase().includes(q) || r.hyp.includes(q)))
      .sort((a, b) => b.edits - a.edits || a.i - b.i).slice(0, 400);
    table.replaceChildren(el("table", {},
      el("thead", {}, el("tr", {}, ["#", "Session / trial", "Edits", "Reference and decoded (diff)", "Error sources", "Rev"].map((h, i) =>
        el("th", { class: i === 2 || i === 5 ? "num" : "", text: h })))),
      el("tbody", {}, shown.map((r) => el("tr", { class: "clickable", onclick: () => { location.hash = `#/trial/${runId}/${r.i}`; } },
        el("td", { class: "num", text: r.i }),
        el("td", {}, el("div", { text: r.session }), el("div", { class: "sub", text: `${r.trial_key}${r.seen ? " · seen text" : ""}` }),
          el("span", { class: "badge", title: "Sentence source corpus", text: r.corpus || "source unknown" })),
        el("td", { class: "num", text: `${r.edits}/${r.ref_words}` }),
        el("td", {}, el("div", { class: "sub", text: r.ref }), diffLine(r.ref, r.hyp)),
        el("td", {}, ["lm_fixed", "lm_introduced", "both_wrong", "lm_inserted"].filter((k) => (counts.get(r.i) || {})[k] > 0)
          .map((k) => el("div", {}, el("span", { class: `cat cat-${k}`, text: `${CATEGORY_TEXT[k][0]} ×${counts.get(r.i)[k]}` })))),
        el("td", { class: "num", text: r.revision_edits }))))));
  }
  card.replaceChildren(
    el("div", { class: "sub", text: "WER by sentence source (corpus of each recording block, from the release's block description). "
      + "Pooled within each source; small groups are noisy." }),
    sourceTable(rows),
    el("div", { class: "filters" },
      el("label", {}, el("input", { type: "checkbox", checked: true, onchange: (e) => { state.errorsOnly = e.target.checked; render(); } }), " errors only"),
      attribution ? el("select", { onchange: (e) => { state.category = e.target.value; render(); } },
        el("option", { value: "", text: "any error source" }),
        ["lm_introduced", "lm_fixed", "both_wrong", "lm_inserted"].map((k) => el("option", { value: k, text: `has: ${CATEGORY_TEXT[k][0]}` }))) : null,
      el("select", { onchange: (e) => { state.source = e.target.value; render(); } },
        el("option", { value: "", text: "all sentence sources" }), sources.map((c) => el("option", { value: c, text: c }))),
      el("input", { type: "search", placeholder: "filter session or text", oninput: (e) => { state.query = e.target.value; render(); } }),
      el("span", { class: "sub", text: `${rows.length} trials · sorted by edits · first 400 shown` })),
    diffLegend(), table);
  render();
}
