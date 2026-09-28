"""Finalization: neural rescoring of the n-best at trial end, as a derived record of a run.

Reuses the run's cached logits: `stream_lm.py nbest` (LM environment) dumps each trial's n-best
with acoustic and 4-gram scores, a causal LM scores every candidate (.venv, GPU), and the
registered frozen weights pick
    score(h) = acoustic_scale * ac(h) + gamma * lm_4gram(h) + alpha * logP_neural(h) + beta * |h|.
This is utterance-final processing after the (oracle) trial end; it never changes the partial
text shown while the trial runs, and its WER is reported separately from the run's 1-best.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from model_training.benchmark.stream_lm import norm_words

from . import REPO, metrics, registry
from .bench import STREAM_LM, decode_flags, load_run
from .store import DEFAULT_ROOT, fresh_dir, ram_guard, replace_json, utc_stamp, write_new_json

LABEL = ("utterance-final rescoring after the dataset trial end (oracle endpoint); not low-latency committed "
         "text. Frozen weights were tuned on val-dev with causal_la0 (exposed)")


def nbest(run_manifest, out_dir, lm_python, n, log=print):
    spec = run_manifest["pipeline"]
    ram_guard(spec["lm"]["expected_rss_gb"], f"LM {spec['lm']['name']} n-best")
    path = out_dir / "nbest.json"
    command = [lm_python, "-u", str(STREAM_LM), "nbest", "--cache",
               str(Path(run_manifest["logits_cache"]["dir"]) / "logits.npz"), "--lm", spec["lm"]["graph_dir"],
               "--split", "all", "--nbest", str(n), "--progress", "0", "--out", str(path),
               *decode_flags(spec["decode"])]
    with (out_dir / "nbest.log").open("x") as handle:
        done = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO, env=os.environ.copy())
    if done.returncode != 0:
        raise RuntimeError(f"n-best stage failed; see {out_dir / 'nbest.log'}")
    return json.loads(path.read_text())


def neural_scores(lists, model_name, device, log=print):
    """Per-candidate log P under the causal LM; one batch per trial so latency is per utterance."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_training.benchmark.neural_rescore import score_sentences
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    scores, latency = [], []
    for index, candidates in enumerate(lists):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        values = score_sentences(model, tokenizer, [c["s"] for c in candidates], device) if candidates else []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency.append((time.perf_counter() - start) * 1000.0)
        scores.append([float(v) for v in values])
        if (index + 1) % 250 == 0:
            log(f"  neural scoring {index + 1}/{len(lists)}")
    return scores, latency, sum(p.numel() for p in model.parameters())


def choose(candidates, neural, acoustic_scale, weights):
    if not candidates:
        return ""
    best = max(range(len(candidates)), key=lambda i: acoustic_scale * candidates[i]["ac"]
               + weights["gamma"] * candidates[i]["lm"] + weights["alpha"] * neural[i]
               + weights["beta"] * len(norm_words(candidates[i]["s"])))
    return candidates[best]["s"]


def rescore_run(run_dir, rescorer_name="gpt2large", *, root=DEFAULT_ROOT, device="cuda",
                lm_python="/home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python", log=print,
                registry_root=registry.REGISTRY_DIR):
    import torch
    run_dir = Path(run_dir)
    manifest, summary, rows = load_run(run_dir)
    if manifest["tier"] not in ("standard", "sweep") or rows is None:
        raise ValueError("Rescoring needs a complete standard or sweep run (with cached logits)")
    rescorer = registry.load_kind("rescorer", registry_root)[rescorer_name]
    out_dir = fresh_dir(Path(root) / "rescore" / f"{run_dir.name}__{rescorer_name}__{utc_stamp()}")
    record = dict(schema="harness_rescore_v1", run_id=run_dir.name, rescorer=rescorer, status="running",
                  created=datetime.now(timezone.utc).isoformat(), label=LABEL, pid=os.getpid(),
                  flags=sorted(set(manifest["flags"]) | ({"rescorer_tuned_for_other_acoustic"}
                               if rescorer["tuned_on"]["acoustic"] != manifest["pipeline"]["acoustic"]["name"] else set())))
    write_new_json(out_dir / "record.json", record)
    try:
        log(f"n-best {rescorer['nbest']} decode for {run_dir.name}")
        dump = nbest(manifest, out_dir, lm_python, rescorer["nbest"], log)
        if len(dump["nbest"]) != len(rows):
            raise AssertionError("n-best trials differ from the run's trials")
        log(f"{rescorer['model']} scoring {sum(len(c) for c in dump['nbest'])} candidates")
        scores, latency, n_params = neural_scores(dump["nbest"], rescorer["model"], torch.device(device), log)
        acoustic_scale = manifest["pipeline"]["decode"]["acoustic_scale"]
        identity = dict(gamma=1.0, alpha=0.0, beta=0.0)
        rescored, first_best_matches = [], 0
        for row, candidates, neural in zip(rows, dump["nbest"], scores):
            one_best = choose(candidates, neural, acoustic_scale, identity)
            first_best_matches += norm_words(one_best) == norm_words(row["hyp"])
            hyp = choose(candidates, neural, acoustic_scale, rescorer["weights"])
            edits, words = metrics.word_errors(row["ref"], hyp)
            rescored.append(dict(row, hyp=hyp, edits=edits, ref_words=words))
        with (out_dir / "trials.jsonl").open("x") as handle:
            for row in rescored:
                handle.write(json.dumps(row) + "\n")
        oracle = sum(min([metrics.word_errors(r["ref"], c["s"])[0] for c in cands] or [r["ref_words"]])
                     for r, cands in zip(rows, dump["nbest"]))
        paired = metrics.paired_compare(rows, rescored)
        result = dict(
            wer=dict(metrics.pooled(rescored), **metrics.session_bootstrap(rescored)),
            one_best_wer=summary["accuracy"]["wer"]["percent"],
            delta_vs_one_best=dict(points=paired["delta_wer_points"], ci95=paired["ci95"],
                                   trials_better=paired["trials"]["improved"], trials_worse=paired["trials"]["worsened"]),
            oracle_wer=100.0 * oracle / sum(r["ref_words"] for r in rows),
            seen_unseen=metrics.seen_unseen(rescored),
            self_check=dict(identity_weights_reproduce_run_1best=first_best_matches, n_trials=len(rows)),
            nbest=dict(list_size=dump["list_size"], finalize_ms=dump["finalize_ms"]),
            latency=dict(neural_scoring_ms_per_utterance=metrics.distribution(latency),
                         nbest_finalize_ms=dump["finalize_ms"],
                         note="n-best extraction and neural scoring are measured separately and are not summed; "
                              "GPU contention during this run is not controlled"),
            model_params=n_params)
        write_new_json(out_dir / "summary.json", result)
        record.update(status="complete", finished=datetime.now(timezone.utc).isoformat())
        replace_json(out_dir / "record.json", record)
        log(f"Rescored WER {result['wer']['percent']:.3f}% vs 1-best {result['one_best_wer']:.3f}% "
            f"(oracle {result['oracle_wer']:.2f}%) -> {out_dir}")
        return out_dir
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        replace_json(out_dir / "record.json", record)
        raise


def latest(run_id, root=DEFAULT_ROOT):
    """The newest complete rescoring record for a run: (record, summary) or None."""
    for path in sorted((Path(root) / "rescore").glob(f"{run_id}__*"), reverse=True):
        record = json.loads((path / "record.json").read_text())
        if record.get("status") == "complete":
            return record, json.loads((path / "summary.json").read_text())
    return None
