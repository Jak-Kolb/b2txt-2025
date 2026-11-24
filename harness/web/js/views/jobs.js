// Jobs: queue scripts (results/harness/queue/) and harness training runs: running, waiting, recently finished.
// Data: /api/jobs and /api/gpu, refreshed every 10 s. Log tails come from /api/jobs/<id>/log (sentence-like lines
// are hidden server-side); this page never shows sentence text.
import { el, fill, getJSON, shortTime } from "../ui.js";

const REFRESH_MS = 10000;
const FINISHED_SHOWN = 50;

function duration(seconds) {
  if (seconds == null) return "–";
  const s = Math.max(0, Math.round(seconds));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

const localTime = (iso) => {
  if (!iso) return "–";
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? shortTime(iso)
    : t.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
};

const mib = (v) => (v == null ? "–" : v >= 1024 ? `${(v / 1024).toFixed(1)} GB` : `${Math.round(v)} MB`);

function stateBadge(job) {
  const label = job.state === "done" && job.failed_stages ? `done · ${job.failed_stages} stage${job.failed_stages > 1 ? "s" : ""} failed` : job.state;
  return el("span", { class: `jstate jstate-${job.state}${job.state === "done" && job.failed_stages ? " partial" : ""}`, text: label });
}

function stageChips(job) {
  if (!job.stages || !job.stages.length) return null;
  return el("div", { class: "chips" }, job.stages.map((s) => el("span", {
    class: `chip chip-${s.state}`,
    title: [s.detail || s.state, s.started ? `start ${localTime(s.started)}` : null, s.ended ? `end ${localTime(s.ended)}` : null,
            s.error ? `error: ${s.error}` : null].filter(Boolean).join("\n"),
    text: s.name,
  })));
}

function links(job, jobsById) {
  const out = [];
  const runs = (job.links && job.links.runs) || [];
  if (runs.length > 4) out.push(el("span", { class: "muted", text: `${runs.length} runs, last 4:` }));
  runs.slice(-4).forEach((r) => out.push(el("a", { href: `#/run/${encodeURIComponent(r)}`, class: "mono", title: r, text: runLabel(r) })));
  if (runs.length >= 2) {
    const [a, b] = runs.slice(-2);
    out.push(el("a", { href: `#/compare/${encodeURIComponent(a)}/${encodeURIComponent(b)}`, text: "compare last two" }));
  }
  const trains = (job.links && job.links.train) || [];
  if (trains.length) out.push(el("a", { href: "#/train", title: trains.join("\n"), text: trains.length > 1 ? `training page (${trains.length} runs)` : "training page" }));
  ((job.links && job.links.jobs) || []).slice(-3).forEach((j) => out.push(el("a", {
    href: "#/jobs", title: j, text: `uses ${j}`, onclick: (e) => { e.preventDefault(); openJob(j); } })));
  return out.length ? el("div", { class: "jlinks" }, out) : null;
}

function waitingText(job, jobsById) {
  const w = job.waiting_on;
  if (!w) return "–";
  if (w.job && jobsById[w.job]) {
    return el("span", {}, "after ", el("a", { href: "#/jobs", text: jobsById[w.job].name, onclick: (e) => { e.preventDefault(); openJob(w.job); } }),
      el("span", { class: "muted", text: ` (PID ${w.pid})` }));
  }
  return w.label || (w.pid ? `PID ${w.pid}` : "–");
}

// ---------------------------------------------------------------- log panel (kept open across refreshes)

const openLogs = new Map();   // job id -> selected log name

function openJob(id) {
  const node = document.querySelector(`[data-job="${CSS.escape(id)}"]`);
  if (node) { node.scrollIntoView({ behavior: "smooth", block: "center" }); node.classList.add("flash"); setTimeout(() => node.classList.remove("flash"), 1200); }
}

function logPanel(job) {
  const box = el("div", { class: "logbox" });
  const pre = el("pre", { class: "logtail", text: "Loading…" });
  const picker = job.logs.length > 1 ? el("select", {
    onchange: (e) => { openLogs.set(job.id, e.target.value); load(); },
  }, job.logs.map((name) => el("option", { value: name, selected: openLogs.get(job.id) === name, text: name }))) : null;
  const meta = el("span", { class: "sub" });
  async function load() {
    const which = openLogs.get(job.id);
    const query = `tail=200${which ? `&log=${encodeURIComponent(which)}` : ""}`;
    try {
      const data = await getJSON(`/api/jobs/${encodeURIComponent(job.id)}/log?${query}`);
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
      pre.textContent = data.lines.join("\n") || "(empty)";
      if (atBottom || pre.dataset.loaded !== "1") pre.scrollTop = pre.scrollHeight;
      pre.dataset.loaded = "1";
      if (picker && !which) picker.value = data.log;
      meta.textContent = `${data.file} · last ${data.lines.length} lines${data.hidden_lines ? ` · ${data.hidden_lines} hidden (may contain sentence text)` : ""}`;
    } catch (error) {
      pre.textContent = String(error.message || error);
    }
  }
  load();
  box.append(el("div", { class: "row" }, picker, meta), pre);
  return box;
}

// Returns [button, panel]: the caller places the panel (below a card, or in a full-width row under a table row).
function logToggle(job, panel = el("div")) {
  if (!job.logs || !job.logs.length) return [el("span", { class: "muted small-note", text: "no log" }), panel];
  const button = el("button", { class: "small", text: openLogs.has(job.id) ? "Hide log" : "Show log" });
  const show = (on) => {
    button.textContent = on ? "Hide log" : "Show log";
    panel.hidden = !on;
    if (on) fill(panel.querySelector("[data-slot]") || panel, logPanel(job));
    else fill(panel.querySelector("[data-slot]") || panel);
  };
  button.addEventListener("click", () => {
    if (openLogs.has(job.id)) openLogs.delete(job.id); else openLogs.set(job.id, null);
    show(openLogs.has(job.id));
  });
  show(openLogs.has(job.id));
  return [button, panel];
}

function logRow(job, columns) {
  const slot = el("div", { "data-slot": "1" });
  const row = el("tr", { class: "logrow" }, el("td", { colspan: columns }, slot));
  const [button] = logToggle(job, row);
  return [button, row];
}

const RUN_STAMP = /^(\d{4})(\d\d)(\d\d)T(\d\d)(\d\d)/;
function runLabel(id) {
  const m = RUN_STAMP.exec(id);
  return m ? `run ${m[2]}-${m[3]} ${m[4]}:${m[5]}` : `run ${id.slice(0, 16)}`;
}

// ---------------------------------------------------------------- sections

function gpuCard(gpu) {
  if (!gpu) return null;
  const g = gpu.gpus && gpu.gpus[0];
  const lock = gpu.locks && gpu.locks.job;
  const holder = lock && lock.holders[0];
  const tiles = [
    el("div", { class: "tile" }, el("div", { class: "label", text: "GPU" }),
      el("div", { class: "value", text: g ? `${Math.round(g.util_pct ?? 0)}%` : "–" }),
      el("div", { class: "detail", text: g ? g.name : gpu.error || "nvidia-smi unavailable" })),
    el("div", { class: "tile" }, el("div", { class: "label", text: "GPU memory" }),
      el("div", { class: "value", text: g ? mib(g.mem_used_mib) : "–" }),
      el("div", { class: "detail", text: g ? `of ${mib(g.mem_total_mib)}${g.temp_c != null ? ` · ${Math.round(g.temp_c)}°C` : ""}` : "" })),
    el("div", { class: "tile" }, el("div", { class: "label", text: "GPU job lock" }),
      el("div", { class: "value", text: lock && lock.held ? "held" : "free" }),
      el("div", { class: "detail", text: holder ? `${holder.job || holder.command || ""} (PID ${holder.pid})${lock.waiters.length ? ` · ${lock.waiters.length} waiting` : ""}` : "benchmarks and training take it" })),
  ];
  const apps = gpu.apps || [];
  return el("div", {},
    el("div", { class: "tiles" }, tiles),
    apps.length ? el("div", { class: "sub", style: "margin-top:6px" }, "On the GPU: ", apps.map((a, i) => [i ? " · " : "",
      `${a.job || a.process} (PID ${a.pid}, ${mib(a.mem_mib)})`])) :
      gpu.apps_visible === false ? el("div", { class: "sub", style: "margin-top:6px",
        text: "GPU memory is in use but WSL hides which process holds it; per-job GPU memory is unavailable." }) : null);
}

function gpuMemFor(job, gpu) {
  if (!gpu || !gpu.apps) return null;
  const total = gpu.apps.filter((a) => a.job === job.id).reduce((t, a) => t + (a.mem_mib || 0), 0);
  return total ? mib(total) : null;
}

function runningCard(job, gpu, jobsById) {
  const mem = gpuMemFor(job, gpu);
  const detail = job.detail || {};
  return el("div", { class: "card jobcard", "data-job": job.id },
    el("div", { class: "jhead" },
      el("div", {}, el("strong", { class: "jname", text: job.name }), " ", el("span", { class: "badge", text: job.kind }), " ", stateBadge(job),
        el("div", { class: "sub mono", text: job.id })),
      el("div", { class: "jfacts" },
        fact("Stage", job.stage || "–"),
        fact("Elapsed", duration(job.elapsed_s)),
        fact("Started", localTime(job.started)),
        fact("PID", job.pid ?? "–"),
        fact("GPU mem", mem || "–"),
        job.kind === "train" ? fact("Last val PER", detail.last_val_per != null ? Number(detail.last_val_per).toFixed(4) : "–") : null)),
    stageChips(job),
    job.kind === "train" && detail.question ? el("div", { class: "sub", text: `Question: ${detail.question}` }) : null,
    links(job, jobsById),
    (() => { const [button, panel] = logToggle(job); return el("div", { class: "logwrap" }, button, panel); })());
}

function fact(label, value) {
  return el("div", { class: "fact" }, el("div", { class: "label", text: label }), el("div", { class: "v", text: String(value) }));
}

function jobCell(j, button) {
  return el("td", { class: "jobcell" }, el("strong", { text: j.name }), " ", el("span", { class: "badge", text: j.kind }),
    el("div", { class: "sub mono id", text: j.id }), el("div", { class: "jbtn" }, button));
}

function waitingTable(jobs, jobsById) {
  return el("div", { class: "card table-wrap" }, el("table", { class: "jobs" },
    el("thead", {}, el("tr", {}, ["Job", "Waiting on", "Queued", "Waited", "PID"].map((h) => el("th", { text: h })))),
    el("tbody", {}, jobs.map((j) => {
      const [button, row] = logRow(j, 5);
      return [el("tr", { "data-job": j.id },
        jobCell(j, button),
        el("td", {}, waitingText(j, jobsById)),
        el("td", { class: "nowrap", text: localTime(j.started) }),
        el("td", { class: "num", text: duration(j.elapsed_s) }),
        el("td", { class: "num mono", text: j.pid ?? "–" })), row];
    }))));
}

function finishedTable(jobs, jobsById) {
  return el("div", { class: "card table-wrap" }, el("table", { class: "jobs" },
    el("thead", {}, el("tr", {}, ["Result", "Job", "Ended", "Took", "Stages", "Links"].map((h) => el("th", { text: h })))),
    el("tbody", {}, jobs.map((j) => {
      const [button, row] = logRow(j, 6);
      return [el("tr", { "data-job": j.id },
        el("td", {}, stateBadge(j)),
        jobCell(j, button),
        el("td", { class: "nowrap", text: localTime(j.ended || j.updated) }),
        el("td", { class: "num", text: duration(j.elapsed_s) }),
        el("td", { class: "stagescell" }, stageChips(j) || el("span", { class: "muted", text: j.kind === "train" ? j.stage : "–" })),
        el("td", { class: "linkscell" }, links(j, jobsById) || "–")), row];
    }))));
}

// ---------------------------------------------------------------- view

export async function jobsView(main) {
  let timer = null, stopped = false;
  const body = el("div");
  const stamp = el("span", { class: "sub" });
  const refreshButton = el("button", { class: "small", text: "Refresh", onclick: () => render() });

  async function render() {
    try {
      const [data, gpu] = await Promise.all([getJSON(`/api/jobs?limit=${FINISHED_SHOWN}`), getJSON("/api/gpu").catch(() => null)]);
      if (stopped) return;
      const jobsById = Object.fromEntries(data.jobs.map((j) => [j.id, j]));
      const running = data.jobs.filter((j) => j.state === "running");
      const waiting = data.jobs.filter((j) => j.state === "waiting" || j.state === "queued");
      const finished = data.jobs.filter((j) => !["running", "waiting", "queued"].includes(j.state));
      const scrollY = window.scrollY;
      fill(body,
        gpuCard(gpu),
        el("h2", { text: `Running (${running.length})` }),
        running.length ? running.map((j) => runningCard(j, gpu, jobsById)) : el("div", { class: "card empty", text: "Nothing running." }),
        el("h2", { text: `Waiting (${waiting.length})` }),
        waiting.length ? waitingTable(waiting, jobsById) : el("div", { class: "card empty", text: "Nothing queued." }),
        el("h2", { text: `Finished (last ${finished.length} of ${data.total_finished})` }),
        finished.length ? finishedTable(finished, jobsById) : el("div", { class: "card empty", text: "No finished jobs yet." }));
      window.scrollTo(0, scrollY);
      stamp.textContent = `Updated ${new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" })} · refreshes every 10 s`;
    } catch (error) {
      stamp.textContent = `Refresh failed: ${error.message || error}`;
    }
  }

  main.replaceChildren(
    el("div", { class: "jtitle" }, el("h1", { text: "Jobs" }), el("div", { class: "row" }, stamp, refreshButton)),
    el("div", { class: "sub", text: "Queue scripts in results/harness/queue/ and harness training runs. Queue scripts report stages through "
      + "harness.jobs (job.json), training queues through their status.json; older scripts are read from their queue.log. One GPU job at a time: training and benchmarks share results/harness/locks/job.lock." }),
    body);
  await render();
  timer = setInterval(() => { if (!document.hidden) render(); }, REFRESH_MS);
  return () => { stopped = true; clearInterval(timer); };
}
