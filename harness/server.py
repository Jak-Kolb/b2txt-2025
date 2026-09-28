"""Local web viewer (runs, compare, trial replay) and live-player relay. Binds 127.0.0.1 only.

This process never imports torch: model work happens in benchmark CLIs and the live engine
subprocess (harness/live_engine.py), so the UI cannot perturb timing or hold GPU memory.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web

from . import metrics, registry
from .bench import compare_runs, load_run
from .errors import analyze_run, cached_summary
from .scopes import DATA_DIR, corpus_of
from .store import DEFAULT_ROOT, list_runs, resolve_run

WEB = Path(__file__).with_name("web")


def run_row(manifest):
    """Leaderboard row: manifest essentials plus headline summary numbers."""
    spec = manifest.get("pipeline") or {}
    row = dict(run_id=manifest["run_id"], tier=manifest["tier"], status=manifest["status"],
               created=manifest.get("created"), pipeline=spec.get("name"),
               acoustic=(spec.get("acoustic") or {}).get("name"), lm=(spec.get("lm") or {}).get("name"),
               decode=spec.get("decode"), flags=manifest.get("flags", []),
               scope=dict(name=manifest["scope"]["name"], hash=manifest["scope"]["hash"],
                          n_trials=manifest["scope"]["n_trials"], partition=manifest["scope"]["partition"]),
               pipeline_hash=(manifest.get("identity") or {}).get("pipeline_hash"),
               git=dict(head=(manifest.get("code") or {}).get("head"), dirty=(manifest.get("code") or {}).get("dirty")))
    path = Path(manifest["run_dir"]) / "summary.json"
    if path.exists():
        summary = json.loads(path.read_text())
        accuracy = summary.get("accuracy") or {}
        if accuracy:
            row.update(wer=accuracy["wer"]["percent"], ci95=accuracy["wer"].get("ci95"),
                       seen_wer=accuracy["seen_unseen"]["seen"]["percent"],
                       unseen_wer=accuracy["seen_unseen"]["unseen"]["percent"],
                       per=accuracy["per_diagnostic"]["percent"],
                       revisions_per_100=accuracy["revisions"]["revision_edits_per_100_final_words"])
            attribution = accuracy.get("error_attribution") or cached_summary(
                manifest["run_dir"], Path(manifest["run_dir"]).parents[1])
            if attribution:
                row.update(lm_fixed_pct=attribution["pct_acoustic_word_errors_fixed_by_lm"],
                           exact_phoneme_error_pct=attribution["pct_word_errors_with_exact_phonemes"])
        timing = summary.get("timing_check") or {}
        paced = summary.get("paced")
        if timing.get("simulated_lag"):
            row.update(first_output_ms=timing["structural_delay"]["first_output_ms"],
                       step_p99_ms=timing["service_ms"]["per_bin"]["p99"],
                       lag_p95_ms=timing["simulated_lag"]["frame_window_to_output_ms"]["p95"], lag_kind="simulated")
        elif paced:
            row.update(first_output_ms=paced["structural_delay"]["first_output_ms"],
                       step_p99_ms=paced["busy_per_bin_ms"]["p99"],
                       lag_p95_ms=paced["frame_window_to_output_ms"]["p95"], lag_kind="measured")
        elif timing.get("skipped"):
            row["timing_skipped"] = timing["skipped"]
    return row


class TraceIndex:
    """Byte offsets of each trial's records in a trace, rebuilt when the file changes."""

    def __init__(self):
        self.cache = {}

    def events(self, path, trial_index):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.cache:
            offsets, header = {}, None
            with path.open("rb") as handle:
                position = 0
                current = None
                for line in handle:
                    if line.startswith(b'{"record": "trial_start"'):
                        record = json.loads(line)
                        current = record.get("cache_index", record.get("trial_index"))
                        offsets[current] = [position, position + len(line)]
                    elif line.startswith(b'{"record": "run_start"'):
                        header = json.loads(line)
                    elif current is not None:
                        offsets[current][1] = position + len(line)
                    position += len(line)
            self.cache = {key: (offsets, header)}
        offsets, header = self.cache[key]
        start, end = offsets[trial_index]
        with path.open("rb") as handle:
            handle.seek(start)
            lines = handle.read(end - start).splitlines()
        return header, [json.loads(line) for line in lines]


def create_app(root=DEFAULT_ROOT, registry_root=registry.REGISTRY_DIR, live=None, data_dir=DATA_DIR):
    root = Path(root)
    traces = TraceIndex()
    app = web.Application()

    def run_dir(request):
        try:
            return resolve_run(root, request.match_info["run_id"])
        except KeyError as exc:
            raise web.HTTPNotFound(text=str(exc))

    async def index(_):
        return web.FileResponse(WEB / "index.html")

    async def runs(request):
        rows = [run_row(m) for m in list_runs(root)]
        return web.json_response(rows)

    async def run(request):
        manifest, summary, _ = load_run(run_dir(request))
        return web.json_response(dict(manifest=manifest, summary=summary))

    async def trials(request):
        _, _, rows = load_run(run_dir(request))
        if rows is None:
            raise web.HTTPNotFound(text="Run has no trials (incomplete or failed)")
        for row in rows:  # sentence source is derived at view time, so older runs get it too
            row["corpus"] = corpus_of(row["session"], row.get("block_num"), data_dir)
        return web.json_response(rows)

    async def events(request):
        directory = run_dir(request)
        name = {"accuracy": "output_trace.jsonl", "timing": "timing_trace.jsonl"}.get(
            request.query.get("trace", "accuracy"))
        if name is None or not (directory / name).exists():
            raise web.HTTPNotFound(text="No such trace")
        try:
            index = int(request.match_info["index"])
            header, records = traces.events(directory / name, index)
        except (KeyError, ValueError):
            raise web.HTTPNotFound(text="No such trial in trace")
        return web.json_response(dict(schema_version=header["schema_version"], records=records))

    timing_cache = {}

    async def timing(request):
        directory = run_dir(request)
        manifest = json.loads((directory / "manifest.json").read_text())
        name, simulated = (("output_trace.jsonl", False) if manifest["tier"] == "paced"
                           else ("timing_trace.jsonl", True))
        path = directory / name
        if not path.exists():
            raise web.HTTPNotFound(text="Run has no timing trace")
        key = (str(path), path.stat().st_mtime_ns)
        if key not in timing_cache:
            timing_cache[key] = metrics.timing_quantiles(path, simulated)
        return web.json_response(timing_cache[key])

    async def attribution(request):
        directory = run_dir(request)
        try:
            return await asyncio.to_thread(analyze_run, directory, data_dir, root)
        except ValueError as exc:
            raise web.HTTPNotFound(text=str(exc))

    async def errors(request):
        result = await attribution(request)
        return web.json_response(dict(summary=result["summary"], trials=[
            dict(i=t["i"], status=t["status"], counts=t["counts"]) for t in result["trials"]]))

    async def phonemes(request):
        result = await attribution(request)
        try:
            return web.json_response(result["trials"][int(request.match_info["index"])])
        except (IndexError, ValueError):
            raise web.HTTPNotFound(text="No such trial")

    async def compare(request):
        try:
            a = resolve_run(root, request.query["a"])
            b = resolve_run(root, request.query["b"])
            return web.json_response(compare_runs(a, b))
        except KeyError as exc:
            raise web.HTTPNotFound(text=str(exc))
        except ValueError as exc:
            raise web.HTTPConflict(text=str(exc))

    async def training(_):
        from .train import status
        return web.json_response(status(root))

    async def registry_view(_):
        entries = registry.load_all(registry_root)
        return web.json_response(entries)

    app.router.add_get("/", index)
    app.router.add_get("/api/runs", runs)
    app.router.add_get("/api/runs/{run_id}", run)
    app.router.add_get("/api/runs/{run_id}/trials", trials)
    app.router.add_get("/api/runs/{run_id}/trials/{index}/events", events)
    app.router.add_get("/api/runs/{run_id}/timing", timing)
    app.router.add_get("/api/runs/{run_id}/errors", errors)
    app.router.add_get("/api/runs/{run_id}/trials/{index}/phonemes", phonemes)
    app.router.add_get("/api/compare", compare)
    app.router.add_get("/api/registry", registry_view)
    app.router.add_get("/api/train", training)
    if live is not None:
        live.install(app)
    app.router.add_static("/static", WEB)
    return app


def main(options):
    from .live import LiveRelay
    live = LiveRelay(root=Path(options.root), data_dir=options.data_dir, lm_python=options.lm_python)
    app = create_app(options.root, live=live, data_dir=options.data_dir or DATA_DIR)
    print(f"Harness viewer on http://127.0.0.1:{options.port}  (from the Mac: "
          f"ssh -L {options.port}:127.0.0.1:{options.port} jakpc)")
    web.run_app(app, host="127.0.0.1", port=options.port, print=None)
