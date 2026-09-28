// Harness-launched training runs (status only; launch with `python -m harness train`).
import { el, fmt, getJSON, shortTime } from "../ui.js";

export async function trainView(main) {
  const rows = await getJSON("/api/train");
  main.replaceChildren(
    el("h1", { text: "Training" }),
    el("div", { class: "sub", text: "Launch from the CLI: .venv/bin/python -m harness train --from causal_la0 --name NAME "
      + "--set model.n_units=1024 --question \"...\" [--then-bench la0_gen4g]. Checkpoints are selected by the trainer's "
      + "val PER over all enabled val sessions (recorded on the registered model)." }),
    rows.length ? el("div", { class: "card table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, ["Run", "Model", "Overrides", "Question", "State", "Last val PER", "Benchmark"].map((h) => el("th", { text: h })))),
      el("tbody", {}, rows.map((r) => el("tr", {},
        el("td", {}, el("div", { class: "mono", text: r.id }), el("div", { class: "sub", text: shortTime(r.updated) })),
        el("td", {}, el("strong", { text: r.name }), el("div", { class: "sub", text: `from ${r.forked_from}` })),
        el("td", { class: "mono", text: r.overrides.join(" ") || "–" }),
        el("td", { text: r.question }),
        el("td", { class: `status-${r.state}`, text: r.state }),
        el("td", { class: "num", text: r.log ? fmt(r.log.last_val_per, 4) : "–" }),
        el("td", {}, r.benchmark_run ? el("a", { href: `#/run/${r.benchmark_run}`, text: "view run" }) : "–"))))))
      : el("div", { class: "empty", text: "No harness training runs yet." }));
}
