"""CLI: python -m harness {bench,runs,compare,serve,...} (run from the repo root with .venv)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

from . import registry
from .store import DEFAULT_ROOT, list_runs, resolve_run


def key_values(pairs):
    out = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"Expected KEY=VALUE, got {pair!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def cmd_bench(options):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from . import bench
    spec = registry.resolve(options.pipeline, acoustic=options.acoustic, lm=options.lm,
                            decode=key_values(options.decode), preprocess=key_values(options.preprocess))
    common = dict(root=options.root, limit=options.limit, device=options.device, lm_python=options.lm_python)
    if options.data_dir:
        common["data_dir"] = options.data_dir
    try:
        if options.tier == "standard":
            bench.run_standard(spec, scope_name=options.scope or "val-dev-full",
                               allow_exposed=options.allow_exposed,
                               timing_check=not options.no_timing_check, **common)
        else:
            bench.run_paced(spec, scope_name=options.scope or "val-dev-sample",
                            allow_exposed=options.allow_exposed, **common)
    except BlockingIOError:
        raise SystemExit("Another harness benchmark holds the job lock; wait for it to finish")


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def cmd_runs(options):
    rows = []
    for manifest in list_runs(options.root):
        if not options.all and "smoke_limited" in manifest.get("flags", []):
            continue
        summary_path = os.path.join(manifest["run_dir"], "summary.json")
        wer = ci = lag = "-"
        if os.path.exists(summary_path):
            summary = json.load(open(summary_path))
            acc = summary.get("accuracy") or {}
            if acc:
                wer = fmt(acc["wer"]["percent"])
                ci = "-" if not acc["wer"].get("ci95") else "[" + ", ".join(fmt(v) for v in acc["wer"]["ci95"]) + "]"
            timing = summary.get("timing_check") or {}
            if timing.get("simulated_lag"):
                lag = fmt(timing["simulated_lag"]["frame_window_to_output_ms"]["p95"], 1) + " sim"
            elif summary.get("paced"):
                lag = fmt(summary["paced"]["frame_window_to_output_ms"]["p95"], 1) + " measured"
        rows.append([manifest["run_id"], manifest["tier"], manifest["status"], manifest["scope"]["name"],
                     manifest["scope"]["hash"][:8], wer, ci, lag, ",".join(manifest.get("flags", []))])
    header = ["run_id", "tier", "status", "scope", "hash", "WER%", "95% CI (sessions)", "lag p95 ms", "flags"]
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for row in [header] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def cmd_compare(options):
    from .bench import compare_runs
    result = compare_runs(resolve_run(options.root, options.a), resolve_run(options.root, options.b))
    ci = result["ci95"]
    print(f"A = {result['a']}\nB = {result['b']}\nscope {result['scope']} ({result['scope_hash'][:8]})")
    print(f"delta WER (B - A): {result['delta_wer_points']:+.3f} points "
          f"({result['delta_edits']:+d} edits / {result['ref_words']} words); "
          f"95% CI {('[' + ', '.join(f'{v:+.3f}' for v in ci) + ']') if ci else 'n/a'} ({result['method']})")
    print(f"trials: {result['trials']}")
    for row in result["per_session"]:
        print(f"  {row['session']}: {row['delta_wer_points']:+.2f} points ({row['delta_edits']:+d} edits)")
    print(f"flags A: {result['flags_a']}\nflags B: {result['flags_b']}\n{result['note']}")


def cmd_train(options):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from . import train
    train_dir = train.fork(options.base, options.name, options.set or [], options.question, root=options.root,
                           smoke_batches=options.smoke_batches, additions=options.add or [])
    print(f"Forked {options.base} -> {options.name}: {train_dir}")
    print(f"Smoke run ({options.smoke_batches} batches, 2 validations) ...", flush=True)
    passed, result = train.run_smoke(train_dir)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(f"Smoke failed; nothing launched. See {train_dir}/smoke.log")
    if options.smoke_only:
        print("Smoke passed (--smoke-only: not launching the full run)")
        return
    pid = train.launch(train_dir, then_bench=options.then_bench, root=options.root)
    print(f"Launched detached supervisor pid {pid}; follow with: python -m harness train-status")


def cmd_train_status(options):
    from . import train
    for row in train.status(options.root):
        log = row["log"] or {}
        print(f"{row['id']}  {row['state']:<17} {row['name']} (from {row['forked_from']}; {' '.join(row['overrides'])})"
              f"  val PER {log.get('last_val_per')}  bench {row['benchmark_run'] or '-'}")


def cmd_train_supervise(options):
    from . import train
    raise SystemExit(train.supervise(options.train_dir, root=options.root))


def cmd_sweep(options):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from . import bench
    from .store import utc_stamp, write_new_json
    spec = registry.resolve(options.pipeline, acoustic=options.acoustic, lm=options.lm)
    grid = {}
    for item in options.grid:
        key, _, values = item.partition("=")
        grid[key.strip()] = [yaml.safe_load(v) for v in values.split(",")]
    common = dict(root=options.root, device=options.device, lm_python=options.lm_python)
    if options.data_dir:
        common["data_dir"] = options.data_dir
    try:
        runs = bench.run_sweep(spec, grid, **common)
    except BlockingIOError:
        raise SystemExit("Another harness benchmark holds the job lock; wait for it to finish")
    rows = []
    for run_dir in runs:
        manifest, summary, _ = bench.load_run(run_dir)
        acc = summary["accuracy"]
        rows.append(dict(run_id=manifest["run_id"], decode={k: manifest["pipeline"]["decode"][k] for k in grid},
                         wer=acc["wer"]["percent"], ci95=acc["wer"]["ci95"], attribution=acc.get("error_attribution")))
    path = options.root and os.path.join(options.root, "sweeps", f"{utc_stamp()}_{spec['name']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_new_json(path, dict(pipeline=spec["name"], acoustic=spec["acoustic"]["name"], lm=spec["lm"]["name"],
                              grid=grid, points=rows, note="decode settings explored on val-dev (exposed)"))
    for r in sorted(rows, key=lambda r: r["wer"]):
        a = r["attribution"] or {}
        print(f"{r['decode']}  WER {fmt(r['wer'])} [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]  "
              f"LM fixed {fmt(a.get('pct_acoustic_word_errors_fixed_by_lm'), 1)}%  "
              f"errors w/ exact phonemes {fmt(a.get('pct_word_errors_with_exact_phonemes'), 1)}%  {r['run_id']}")
    print(f"saved {path}")


def cmd_rescore(options):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from .rescore import rescore_run
    rescore_run(resolve_run(options.root, options.run), options.rescorer, root=options.root,
                device=options.device, lm_python=options.lm_python)


def cmd_commit(options):
    from . import commitment
    result = commitment.evaluate_run(resolve_run(options.root, options.run))
    path = commitment.save(result, options.root)
    print(f"{result['pipeline']} · {result['scope']['name']} ({result['scope']['n_trials']} trials) · {result['run_id']}")
    header = ["policy", "WER%", "dWER vs none [95% CI]", "commit err%", "early%", "wait p50/p95 ms",
              "before end p50 ms", "revisions/100w"]
    rows = []
    for r in result["policies"]:
        d = r.get("delta_wer_vs_none")
        rows.append([r["policy"], fmt(r["wer"]["percent"]),
                     "-" if not d else f"{d['points']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]",
                     fmt(r["commit_errors"]["pct"]), fmt(r["early_commit_pct"], 1),
                     f"{fmt(r['waiting_ms']['p50'], 0)}/{fmt(r['waiting_ms']['p95'], 0)}",
                     fmt(r["committed_before_trial_end_ms"]["p50"], 0), fmt(r["visible_revisions_per_100_words"], 1)])
    widths = [max(len(str(x[i])) for x in rows + [header]) for i in range(len(header))]
    for row in [header] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    print(f"saved {path}\n" + "\n".join(f"  {k}: {v}" for k, v in result["definitions"].items()))


def cmd_serve(options):
    from .server import main as serve
    serve(options)


def parser():
    top = argparse.ArgumentParser(prog="python -m harness", description=__doc__)
    top.add_argument("--root", default=str(DEFAULT_ROOT), help="Harness results root (ignored by git)")
    sub = top.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="Benchmark a pipeline (standard: accuracy + timing check)")
    bench.add_argument("pipeline", nargs="?", help="Pipeline preset name (harness/registry/pipelines)")
    bench.add_argument("--acoustic", help="Acoustic model name (overrides the preset)")
    bench.add_argument("--lm", help="LM name (overrides the preset)")
    bench.add_argument("--decode", action="append", metavar="KEY=VALUE", help="Decode override")
    bench.add_argument("--preprocess", action="append", metavar="KEY=VALUE",
                       help="Smoothing override (flags a train/inference mismatch)")
    bench.add_argument("--tier", choices=["standard", "paced"], default="standard")
    bench.add_argument("--scope", help="Scope name (default: val-dev-full; paced: val-dev-sample)")
    bench.add_argument("--limit", type=int, help="First N scope trials only (smoke; flagged)")
    bench.add_argument("--allow-exposed", metavar="REASON", help="Permit the former val-test scope")
    bench.add_argument("--no-timing-check", action="store_true")
    bench.add_argument("--device", default="cuda")
    bench.add_argument("--data-dir")
    bench.add_argument("--lm-python", default="/home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python")
    bench.set_defaults(func=cmd_bench)

    runs = sub.add_parser("runs", help="List harness runs")
    runs.add_argument("--all", action="store_true", help="Include smoke-limited runs")
    runs.set_defaults(func=cmd_runs)

    compare = sub.add_parser("compare", help="Paired comparison of two runs on the same scope")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.set_defaults(func=cmd_compare)

    train = sub.add_parser("train", help="Fork a model config, smoke-test it, launch detached, auto-register")
    train.add_argument("--from", dest="base", required=True, help="Registered acoustic model to fork")
    train.add_argument("--name", required=True, help="Name for the new acoustic model")
    train.add_argument("--set", action="append", metavar="KEY=VALUE", help="Override an existing config key")
    train.add_argument("--add", action="append", metavar="KEY=VALUE",
                       help="Introduce a new config key (e.g. model.type=transformer)")
    train.add_argument("--question", required=True, help="What this run is meant to answer")
    train.add_argument("--smoke-batches", type=int, default=50)
    train.add_argument("--smoke-only", action="store_true")
    train.add_argument("--then-bench", metavar="PRESET", help="Benchmark preset to run with the new model")
    train.set_defaults(func=cmd_train)

    status = sub.add_parser("train-status", help="Harness-launched training runs")
    status.set_defaults(func=cmd_train_status)

    supervise = sub.add_parser("train-supervise", help=argparse.SUPPRESS)
    supervise.add_argument("train_dir")
    supervise.set_defaults(func=cmd_train_supervise)

    sweep = sub.add_parser("sweep", help="Accuracy-only runs over a grid of decode settings (cached logits)")
    sweep.add_argument("pipeline", nargs="?")
    sweep.add_argument("--acoustic")
    sweep.add_argument("--lm")
    sweep.add_argument("--grid", action="append", required=True, metavar="KEY=V1,V2,...")
    sweep.add_argument("--device", default="cuda")
    sweep.add_argument("--data-dir")
    sweep.add_argument("--lm-python", default="/home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python")
    sweep.set_defaults(func=cmd_sweep)

    rescore = sub.add_parser("rescore", help="Neural n-best rescoring at trial end (finalization) for a run")
    rescore.add_argument("run")
    rescore.add_argument("--rescorer", default="gpt2large")
    rescore.add_argument("--device", default="cuda")
    rescore.add_argument("--lm-python", default="/home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python")
    rescore.set_defaults(func=cmd_rescore)

    commit = sub.add_parser("commit", help="Evaluate word-commitment policies on a standard run's partial outputs")
    commit.add_argument("run")
    commit.set_defaults(func=cmd_commit)

    serve = sub.add_parser("serve", help="Local web viewer and live player (127.0.0.1)")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--data-dir")
    serve.add_argument("--lm-python", default="/home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python")
    serve.set_defaults(func=cmd_serve)
    return top


def main(argv=None):
    options = parser().parse_args(argv)
    options.func(options)


if __name__ == "__main__":
    sys.exit(main())
