// Hash router: #/runs, #/run/<id>, #/compare[/<a>/<b>], #/trial/<run>/<i>[/<otherRun>], #/live, #/jobs
import { el } from "./ui.js";
import { runsView } from "./views/runs.js";
import { runView } from "./views/run.js";
import { compareView } from "./views/compare.js";
import { trialView } from "./views/trial.js";
import { liveView } from "./views/live.js";
import { jobsView } from "./views/jobs.js";

const routes = { runs: runsView, run: runView, compare: compareView, trial: trialView, live: liveView, jobs: jobsView };
let cleanup = null;

async function route() {
  const [name = "runs", ...args] = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  const view = routes[name] || runsView;
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === name));
  if (cleanup) { cleanup(); cleanup = null; }
  const main = document.getElementById("view");
  main.replaceChildren(el("div", { class: "empty", text: "Loading…" }));
  try {
    cleanup = (await view(main, ...args)) || null;
  } catch (error) {
    main.replaceChildren(el("div", { class: "empty error", text: String(error.message || error) }));
  }
}

window.addEventListener("hashchange", route);
route();
