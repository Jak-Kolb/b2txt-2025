// Paired comparison of two runs on an identical scope (the server refuses anything else).
import { diffLegend, diffLine, el, fmt, flagBadges, getJSON, shortTime, tile } from "../ui.js";
import { dotPlot } from "../charts.js";

export async function compareView(main, a, b) {
  const runs = await getJSON("/api/runs");
  const complete = runs.filter((r) => r.status === "complete" && r.wer != null);
  const byId = new Map(complete.map((r) => [r.run_id, r]));
  // Group runs by trial set: only runs on the identical trials (same scope hash) can be paired.
  const groups = new Map();
  for (const r of complete) {
    if (!groups.has(r.scope.hash)) groups.set(r.scope.hash, []);
    groups.get(r.scope.hash).push(r);
  }
  const scopeText = (r) => `${r.scope.name}${r.flags.includes("smoke_limited") ? " (smoke subset)" : ""} · ${r.scope.n_trials} trials`;
  const runText = (r) => `${r.pipeline} · ${r.tier} · ${shortTime(r.created)} · WER ${fmt(r.wer, 2)}`;
  const pick = (value, label) => el("select", { "aria-label": label },
    el("option", { value: "", text: `${label}…` }),
    [...groups.values()].map((group) => el("optgroup", { label: `${scopeText(group[0])} (trial set ${group[0].scope.hash.slice(0, 8)})` },
      group.map((r) => el("option", { value: r.run_id, selected: r.run_id === value, text: runText(r) })))));
  const selectA = pick(a, "Baseline A"), selectB = pick(b, "Candidate B");
  const hint = el("span", { class: "sub" });
  function restrictB() {
    // B may only be a run on the same trials as A (and not A itself).
    const runA = byId.get(selectA.value);
    for (const option of selectB.querySelectorAll("option[value]")) {
      const r = byId.get(option.value);
      if (!r) continue;
      option.disabled = Boolean(runA) && (r.scope.hash !== runA.scope.hash || r.run_id === runA.run_id);
    }
    if (selectB.selectedOptions[0]?.disabled) selectB.value = "";
    const n = runA ? (groups.get(runA.scope.hash) || []).length - 1 : null;
    hint.textContent = runA ? `${n} other run${n === 1 ? "" : "s"} on the same ${runA.scope.n_trials} trials` : "Pick A first; B is limited to runs on the same trials.";
  }
  selectA.addEventListener("change", restrictB);
  restrictB();
  const go = el("button", { class: "primary", onclick: () => {
    if (selectA.value && selectB.value) location.hash = `#/compare/${selectA.value}/${selectB.value}`;
  } }, "Compare");
  const content = el("div");
  main.replaceChildren(el("h1", { text: "Compare runs" }),
    el("div", { class: "sub", text: "Paired on identical trials. Negative ΔWER means B makes fewer errors. "
      + "The interval resamples sessions; it is not seed variability, and one paired run is not a noise floor." }),
    el("div", { class: "filters" }, selectA, selectB, go, hint), content);
  if (!a || !b) return;
  const pairA = byId.get(a), pairB = byId.get(b);
  if (pairA && pairB && pairA.scope.hash !== pairB.scope.hash) {
    content.append(el("div", { class: "card error", text: `These runs used different trials, so they can't be paired: `
      + `A ran on ${scopeText(pairA)}, B on ${scopeText(pairB)}. Pick two runs from the same group.` }));
    return;
  }

  let result;
  try {
    result = await getJSON(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  } catch (error) {
    content.append(el("div", { class: "card error", text: error.message }));
    return;
  }
  const runA = runs.find((r) => r.run_id === result.a), runB = runs.find((r) => r.run_id === result.b);
  const ci = result.ci95 ? `95% CI [${fmt(result.ci95[0], 2)}, ${fmt(result.ci95[1], 2)}]` : "no interval";
  content.append(
    el("div", { class: "card" },
      el("div", {}, el("strong", { text: "A  " }), runA.pipeline, "  ", flagBadges(result.flags_a)),
      el("div", {}, el("strong", { text: "B  " }), runB.pipeline, "  ", flagBadges(result.flags_b)),
      el("div", { class: "sub", text: `scope ${result.scope} · ${result.scope_hash.slice(0, 8)} · ${result.ref_words} reference words` })),
    el("div", { class: "tiles" },
      tile("ΔWER (B − A)", `${result.delta_wer_points >= 0 ? "+" : ""}${fmt(result.delta_wer_points, 2)} pts`, ci),
      tile("WER A → B", `${fmt(runA.wer, 2)} → ${fmt(runB.wer, 2)}%`, `${result.delta_edits >= 0 ? "+" : ""}${result.delta_edits} edits`),
      tile("Trials better / worse", `${result.trials.improved} / ${result.trials.worsened}`,
        `${result.trials.unchanged_edits} same edit count · ${result.trials.different_text} differ in text`)),
    el("h2", { text: "ΔWER by session (B − A)" }),
    el("div", { class: "legend" },
      el("span", {}, el("span", { class: "key", style: "background: var(--div-neg)" }), "B better"),
      el("span", {}, el("span", { class: "key", style: "background: var(--div-pos)" }), "B worse")),
    el("div", { class: "card" }, dotPlot(result.per_session.map((s) => ({ label: s.session, value: s.delta_wer_points, row: s })), {
      valueLabel: "Δ WER points", diverging: true, reference: 0, referenceLabel: "",
      tooltip: (r) => [[`${r.value >= 0 ? "+" : ""}${fmt(r.value, 2)} pts`, r.label], [`${r.row.delta_edits >= 0 ? "+" : ""}${r.row.delta_edits}`, `edits of ${r.row.ref_words} words`]],
    })),
    el("h2", { text: `Trials with different output (${result.differing.length})` }),
    diffLegend(),
    el("div", { class: "card table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, ["#", "Session / trial", "A edits", "B edits", "Reference / A / B"].map((h, i) =>
        el("th", { class: i === 2 || i === 3 ? "num" : "", text: h })))),
      el("tbody", {}, result.differing.sort((x, y) => (y.edits_a - y.edits_b) - (x.edits_a - x.edits_b)).map((d) =>
        el("tr", { class: "clickable", onclick: () => { location.hash = `#/trial/${result.a}/${d.i}/${result.b}`; } },
          el("td", { class: "num", text: d.i }),
          el("td", {}, el("div", { text: d.session }), el("div", { class: "sub", text: d.trial_key })),
          el("td", { class: "num", text: d.edits_a }), el("td", { class: "num", text: d.edits_b }),
          el("td", {}, el("div", { class: "sub", text: d.ref }),
            el("div", { class: "row" }, el("span", { class: "sub", text: "A" }), diffLine(d.ref, d.hyp_a)),
            el("div", { class: "row" }, el("span", { class: "sub", text: "B" }), diffLine(d.ref, d.hyp_b)))))))));
}
