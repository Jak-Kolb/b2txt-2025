// Leaderboard: runs grouped by scope hash (only same-scope runs are comparable).
import { el, fmt, fmtCI, flagBadges, getJSON, shortTime } from "../ui.js";
import { intervalGlyph } from "../charts.js";

export async function runsView(main) {
  const runs = await getJSON("/api/runs");
  const state = { showSmoke: false, tier: "all", selected: [] };
  const compareButton = el("button", { class: "primary", disabled: true, onclick: () => {
    location.hash = `#/compare/${state.selected[0]}/${state.selected[1]}`;
  } }, "Compare selected");
  const hint = el("span", { class: "sub", text: "Select two runs with the same scope to compare." });
  const body = el("div");

  function render() {
    const visible = runs.filter((r) => (state.showSmoke || !r.flags.includes("smoke_limited"))
      && (state.tier === "all" || r.tier === state.tier));
    const groups = new Map();
    for (const run of visible) {
      const key = run.scope.hash;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(run);
    }
    body.replaceChildren();
    if (!visible.length) {
      body.append(el("div", { class: "empty", text: "No runs yet. Try: .venv/bin/python -m harness bench la0_gen4g" }));
    }
    for (const [hash, group] of groups) {
      const first = group[0].scope;
      const wers = group.flatMap((r) => [r.wer, ...(r.ci95 || [])]).filter((v) => v != null);
      const lo = Math.max(0, Math.min(...wers) - 0.5), hi = Math.max(...wers) + 0.5;
      body.append(el("div", { class: "card" },
        el("div", { class: "row" }, el("strong", { text: first.name }),
          el("span", { class: "sub", text: `${first.n_trials} trials · ${first.partition} · scope ${hash.slice(0, 8)}` })),
        el("div", { class: "table-wrap" }, el("table", {},
          el("thead", {}, el("tr", {}, ["", "Pipeline", "Decode ac / bp", "WER %", "95% CI (sessions)", "",
            "Seen / unseen", "Rev /100w", "LM fixed / errors w/ exact phonemes", "First out ms", "Step p99 ms", "Lag p95 ms", "Status", "Flags"]
            .map((h, i) => el("th", { class: i >= 2 && i <= 11 && i !== 5 ? "num" : "", text: h,
              title: i === 8 ? "Share of words with wrong greedy phonemes that the LM still got right / share of word errors where the greedy phonemes were exactly right" : null })))),
          el("tbody", {}, group.map((r) => row(r, lo, hi)))))));
    }
  }

  function row(r, lo, hi) {
    const box = el("input", { type: "checkbox", checked: state.selected.includes(r.run_id),
      onclick: (e) => {
        e.stopPropagation();
        state.selected = e.target.checked ? [...state.selected, r.run_id].slice(-2)
                                          : state.selected.filter((id) => id !== r.run_id);
        const pair = state.selected.map((id) => runs.find((x) => x.run_id === id));
        compareButton.disabled = !(pair.length === 2 && pair[0].scope.hash === pair[1].scope.hash);
        render();
      } });
    return el("tr", { class: "clickable", onclick: () => { location.hash = `#/run/${r.run_id}`; } },
      el("td", {}, box),
      el("td", { title: r.run_id }, el("div", {}, el("strong", { text: r.pipeline || "–" })),
        el("div", { class: "sub", text: `${r.acoustic} + ${r.lm} · ${r.tier} · ${shortTime(r.created)}` })),
      el("td", { class: "num", text: r.decode ? `${r.decode.acoustic_scale} / ${r.decode.blank_penalty}` : "–" }),
      el("td", { class: "num" }, el("strong", { text: fmt(r.wer, 2) })),
      el("td", { class: "num", text: fmtCI(r.ci95) }),
      el("td", {}, r.wer != null ? intervalGlyph(r.wer, r.ci95, lo, hi) : null),
      el("td", { class: "num", text: `${fmt(r.seen_wer, 1)} / ${fmt(r.unseen_wer, 1)}` }),
      el("td", { class: "num", text: fmt(r.revisions_per_100, 1) }),
      el("td", { class: "num", text: r.lm_fixed_pct == null ? "–" : `${fmt(r.lm_fixed_pct, 0)}% / ${fmt(r.exact_phoneme_error_pct, 0)}%` }),
      el("td", { class: "num", text: r.timing_skipped ? "n/a" : fmt(r.first_output_ms, 0) }),
      el("td", { class: "num", text: fmt(r.step_p99_ms, 2) }),
      el("td", { class: "num", title: r.lag_kind === "measured" ? "measured on the paced 20 ms schedule"
        : "simulated from unpaced service times (optimistic)" },
        fmt(r.lag_p95_ms, 2), r.lag_kind ? el("span", { class: "sub", text: r.lag_kind === "measured" ? " meas" : " sim" }) : null),
      el("td", { class: `status-${r.status}`, text: r.status }),
      el("td", { class: "flags" }, flagBadges(r.flags)));
  }

  const tierSelect = el("select", { onchange: (e) => { state.tier = e.target.value; render(); } },
    ["all", "standard", "sweep", "paced"].map((t) => el("option", { value: t, text: `Tier: ${t}` })));
  const smoke = el("label", {}, el("input", { type: "checkbox", onchange: (e) => { state.showSmoke = e.target.checked; render(); } }),
    " show smoke runs");
  main.replaceChildren(
    el("h1", { text: "Runs" }),
    el("div", { class: "sub", text: "WER pooled over the scope; intervals resample sessions (not seed variability). "
      + "Lag columns are simulated from unpaced service times unless the tier is paced." }),
    el("div", { class: "filters" }, tierSelect, smoke, compareButton, hint),
    body);
  render();
}
