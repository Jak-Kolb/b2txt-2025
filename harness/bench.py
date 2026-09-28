"""Standard benchmark: full-scope accuracy plus an unpaced streaming timing check.

Accuracy: batched fp32 logits (cached) -> `stream_lm.py decode` in the LM environment, which
steps the native WFST one frame at a time and records every partial output (schema-1 trace).
Timing check: the true streaming path (paced_replay.replay_trial + AcousticAdapter + lm_worker)
on a stratified sample with a compressed 1 ns arrival schedule; measured service times drive
a queue simulation against the 20 ms schedule. Paced (1x) measurement is `run_paced`.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np

from model_training.benchmark.output_trace import OutputTraceWriter, summarize_trace
from model_training.benchmark.paced_replay import DEFAULT_LM_PYTHON
from model_training.benchmark.stream_lm import norm_words

from . import REPO, metrics, registry
from .scopes import build_scope, scope_record, train_sentence_set, DATA_DIR
from .store import (DEFAULT_ROOT, HashCache, env_info, flocked, git_provenance, list_runs, lock_path,
                    new_run_dir, ram_guard, replace_json, sha256_path, write_new_json)

STREAM_LM = REPO / "model_training" / "benchmark" / "stream_lm.py"
LM_WORKER = REPO / "model_training" / "benchmark" / "lm_worker.py"
EQUIVALENCE_TOLERANCE = 1e-3
CONSTANTS = dict(endpoint="provided_trial_end (oracle endpointing)", commitment_policy="none",
                 text_normalization="stream_lm.norm_words: lowercase, punctuation to space, keep apostrophes")


def now():
    return datetime.now(timezone.utc).isoformat()


def decode_flags(decode):
    flags = []
    for key in registry.DECODE_KEYS:
        flags += [f"--{key}", str(decode[key])]
    return flags


def lm_worker_command(spec, lm_python):
    return [lm_python, "-u", str(LM_WORKER), "--lm", spec["lm"]["graph_dir"],
            "--config", json.dumps(spec["decode"]), "--n_classes", str(spec["model"]["n_classes"])]


def run_lm_stage(spec, cache_dir, run_dir, lm_python, timeout_s=7200):
    ram_guard(spec["lm"]["expected_rss_gb"], f"LM {spec['lm']['name']}")
    command = [lm_python, "-u", str(STREAM_LM), "decode", "--cache", str(cache_dir / "logits.npz"),
               "--lm", spec["lm"]["graph_dir"], "--split", "all", "--capture_partials",
               "--trace_out", str(run_dir / "output_trace.jsonl"), "--out", str(run_dir / "lm_decode.json"),
               "--progress", "0", *decode_flags(spec["decode"])]
    with (run_dir / "lm_stage.log").open("x") as log:
        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s,
                              cwd=REPO, env=os.environ.copy())
    if done.returncode != 0:
        tail = (run_dir / "lm_stage.log").read_text()[-2000:]
        raise RuntimeError(f"LM stage failed (exit {done.returncode}); log tail:\n{tail}")
    return json.loads((run_dir / "lm_decode.json").read_text())


def read_finals(trace_path, index_key):
    finals = {}
    with Path(trace_path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["record"] == "output" and record["kind"] == "final":
                finals[record[index_key]] = record["words"]
    return finals


def run_timing_check(runtime, spec, scope, run_dir, lm_python, accuracy_words, log=print):
    """Unpaced streaming through the exact live/paced code path; returns the summary section."""
    from model_training.benchmark.lm_worker import LMClient
    from model_training.benchmark.paced_replay import AcousticAdapter, replay_trial, synthetic_warmup
    ram_guard(spec["lm"]["expected_rss_gb"], f"LM worker {spec['lm']['name']}")
    trace_path = run_dir / "timing_trace.jsonl"
    metadata = dict(
        replay_mode="unpaced_streaming_service_time", input_clock="compressed_schedule_bin_ns_1",
        timing_boundary="start-to-start service times of the synchronous coordinator; the lag fields "
                        "of this trace are NOT lag (arrivals are compressed to 1 ns)",
        endpoint=CONSTANTS["endpoint"], commitment_policy="none", precision="fp32_tf32_disabled",
        pipeline=spec["name"], config=spec["decode"], budgets_ms=[])
    max_errors, mismatched, compared = [], [], 0
    with LMClient(lm_worker_command(spec, lm_python), run_dir / "lm_worker.log", 30.0, 300.0,
                  shutdown_seconds=60.0) as lm:
        if lm.ready["n_classes"] != spec["model"]["n_classes"]:
            raise ValueError("Acoustic and LM class counts differ")
        warmup = synthetic_warmup(runtime.model, runtime.args, runtime.device,
                                  runtime.day_index(scope["trials"][0].session), lm)
        with OutputTraceWriter(trace_path, dict(metadata, warmup=warmup), schema_version=2) as writer:
            for index, trial in enumerate(scope["trials"]):
                features, _ = runtime.features(trial)
                day = runtime.day_index(trial.session)
                adapter = AcousticAdapter(runtime.model, runtime.args, day, runtime.device)
                runtime.sync()
                result = replay_trial(features, adapter, lm, writer, trial_index=index, day_index=day,
                                      patch_size=runtime.patch_size, patch_stride=runtime.patch_stride,
                                      bin_ns=1, source=dict(session=trial.session, trial_key=trial.trial_key),
                                      keep_logits=True)
                offline = runtime.offline(features, trial.session)
                streamed = np.stack(result["logits"]) if result["logits"] else np.empty_like(offline)
                if streamed.shape != offline.shape:
                    raise AssertionError(f"Streamed/offline logit shapes differ for {trial}")
                error = float(np.max(np.abs(streamed - offline))) if offline.size else 0.0
                if not error < EQUIVALENCE_TOLERANCE:
                    raise AssertionError(f"Streamed/offline max logit error {error} for {trial}")
                max_errors.append(error)
                key = (trial.session, trial.trial_key)
                if key in accuracy_words:
                    compared += 1
                    if accuracy_words[key] != result["words"]:
                        mismatched.append(dict(session=trial.session, trial_key=trial.trial_key))
                if (index + 1) % 10 == 0:
                    log(f"  timing check {index + 1}/{scope['n_trials']}")
        worker_rss = lm.peak_rss_kib
    summary = metrics.timing_check_summary(trace_path)
    summary.update(
        structural_delay=metrics.structural_delay(spec["preprocess"]["effective"],
                                                  runtime.patch_size, runtime.patch_stride),
        equivalence=dict(max_logit_error=max(max_errors) if max_errors else None,
                         tolerance=EQUIVALENCE_TOLERANCE, n_trials=len(max_errors),
                         final_text_compared=compared, final_text_mismatches=mismatched,
                         note="mismatches are near-ties between streamed and batched fp32 logits"),
        scope=dict(name=scope["name"], hash=scope["hash"], n_trials=scope["n_trials"],
                   n_sessions=scope["n_sessions"]),
        warmup=warmup, lm_worker_peak_rss_kib=worker_rss)
    return summary


def score_rows(cache_meta, cache_dir, run_dir, lm_decode, seen_set):
    finals = read_finals(run_dir / "output_trace.jsonl", "cache_index")
    references = (cache_dir / "logits.txt").read_text().split("\n")
    stability = {t["cache_index"]: t for t in lm_decode["output_stability"]["trials"]}
    rows = []
    for index, (trial, reference) in enumerate(zip(cache_meta["trials"], references)):
        hyp = " ".join(finals[index])
        edits, words = metrics.word_errors(reference, hyp)
        revision = stability[index]
        rows.append(dict(i=index, session=trial["session"], split=trial["split"], trial_key=trial["trial_key"],
                         block_num=trial["block_num"], trial_num=trial["trial_num"],
                         day_index=trial["day_index"], n_bins=trial["n_bins"], n_frames=trial["n_frames"],
                         ref=reference, hyp=hyp, ref_words=words, edits=edits,
                         seen=" ".join(norm_words(reference)) in seen_set,
                         phone_edits=trial["phone_edits"], phone_len=trial["phone_len"],
                         n_final_words=revision["n_final_words"], revision_edits=revision["revision_edits"],
                         finalization_edit_counts=revision["finalization_edit_counts"]))
    if len(rows) != len(finals):
        raise AssertionError("Final outputs and cached trials differ in count")
    return rows


def base_manifest(run_dir, tier, spec, identity, scope, flags, lm_python):
    return dict(schema="harness_run_v1", run_id=run_dir.name, tier=tier, status="running", pid=os.getpid(),
                created=now(), pipeline=spec, identity=identity, scope=scope_record(scope), flags=flags,
                constants=CONSTANTS, code=git_provenance(run_dir), env=env_info(), lm_python=lm_python)


def finish(run_dir, manifest, summary, rows):
    with (run_dir / "trials.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    write_new_json(run_dir / "summary.json", summary)
    manifest.update(status="complete", finished=now(), files_sha256={
        p.name: sha256_path(p) for p in sorted(run_dir.iterdir())
        if p.is_file() and p.name != "manifest.json"})
    replace_json(run_dir / "manifest.json", manifest)


def fail(run_dir, manifest, exc):
    (run_dir / "traceback.txt").write_text(traceback.format_exc())
    manifest.update(status="failed", finished=now(), error=f"{type(exc).__name__}: {exc}")
    replace_json(run_dir / "manifest.json", manifest)


def run_standard(spec, *, root=DEFAULT_ROOT, data_dir=DATA_DIR, scope_name="val-dev-full", limit=None,
                 allow_exposed=None, timing_check=True, device="cuda", lm_python=DEFAULT_LM_PYTHON,
                 log=print):
    root = Path(root)
    hasher = HashCache(root)
    identity = registry.identity(spec, hasher)
    scope = build_scope(scope_name, data_dir, limit, allow_exposed, hasher)
    timing_scope = None
    skip_reason = None
    if not timing_check:
        skip_reason = "disabled (--no-timing-check)"
    elif not spec["streaming_capable"]:
        skip_reason = "offline_noncausal pipeline cannot stream"
    else:
        timing_scope = build_scope("val-dev-sample", data_dir, limit, None, hasher)
    flags = registry.flags(spec, scope["partition"], limited=limit is not None)
    with flocked(lock_path(root, "job"), blocking=False):
        run_dir = new_run_dir(root, spec["name"], "standard")
        manifest = base_manifest(run_dir, "standard", spec, identity, scope, flags, lm_python)
        manifest["timing_scope"] = scope_record(timing_scope) if timing_scope else None
        write_new_json(run_dir / "manifest.json", manifest)
        log(f"Run {run_dir.name}: {scope['n_trials']} trials ({scope['name']}); flags {flags}")
        try:
            from .acoustic import Runtime, ensure_logits_cache
            runtime = Runtime(spec, device)
            manifest["env"] = env_info()
            cache_dir, cache_meta, hit = ensure_logits_cache(runtime, identity, scope, root, hasher, log)
            manifest["logits_cache"] = dict(key=cache_meta["key"], dir=str(cache_dir), hit=hit)
            replace_json(run_dir / "manifest.json", manifest)
            log(f"Acoustic logits {'reused' if hit else 'computed'}: {cache_dir.name}")
            lm_decode = run_lm_stage(spec, cache_dir, run_dir, lm_python)
            log(f"LM stage done: {lm_decode['n_trials']} trials")
            rows = score_rows(cache_meta, cache_dir, run_dir, lm_decode,
                              train_sentence_set(data_dir, root))
            accuracy = metrics.accuracy_summary(rows)
            accuracy["cross_checks"] = dict(
                lm_stage_edits=lm_decode["edits"], lm_stage_ref_words=lm_decode["ref_words"],
                harness_edits=accuracy["wer"]["edits"], harness_ref_words=accuracy["wer"]["ref_words"])
            if (lm_decode["edits"], lm_decode["ref_words"]) != (accuracy["wer"]["edits"], accuracy["wer"]["ref_words"]):
                raise AssertionError("Harness scoring disagrees with the LM stage's own WER")
            accuracy["lm_compute"] = dict(
                label="cached-logit native decoder compute per call; not integrated latency",
                per_frame_ms=lm_decode["per_frame_ms"], finalize_ms=lm_decode["finalize_ms"],
                partial_extract_ms=lm_decode.get("partial_extract_ms"), wall_seconds=lm_decode["wall_sec"])
            if timing_scope is not None:
                from .train import active_training
                words = {(r["session"], r["trial_key"]): r["hyp"].split() for r in rows}
                with flocked(lock_path(root, "paced")):
                    training = active_training(root)
                    timing = run_timing_check(runtime, spec, timing_scope, run_dir, lm_python, words, log)
                    training = sorted(set(training) | set(active_training(root)))
                timing["contention"] = dict(harness_training_running=training,
                                            note="only harness-launched jobs are detected")
            else:
                timing = dict(skipped=skip_reason)
            from .errors import analyze
            accuracy["error_attribution"] = analyze(rows, cache_dir / "logits.npz", spec, data_dir)["summary"]
            summary = dict(schema="harness_summary_v1", run_id=run_dir.name, tier="standard",
                           pipeline=spec["name"], pipeline_hash=identity["pipeline_hash"],
                           scope=dict(name=scope["name"], hash=scope["hash"], partition=scope["partition"],
                                      n_trials=scope["n_trials"], n_sessions=scope["n_sessions"],
                                      n_participants=scope["n_participants"], limit=limit),
                           flags=flags, accuracy=accuracy, timing_check=timing, constants=CONSTANTS)
            finish(run_dir, manifest, summary, rows)
        except BaseException as exc:
            fail(run_dir, manifest, exc)
            raise
    log(f"Complete: WER {accuracy['wer']['percent']:.3f}% -> {run_dir}")
    return run_dir


def matching_standard_run(root, identity, scope):
    """Most recent complete standard run whose timing check used this pipeline and trial list."""
    for manifest in list_runs(root):
        timing_scope = manifest.get("timing_scope") or {}
        if (manifest["tier"] == "standard" and manifest["status"] == "complete"
                and manifest["identity"]["pipeline_hash"] == identity["pipeline_hash"]
                and timing_scope.get("hash") == scope["hash"]):
            return Path(manifest["run_dir"])
    return None


def run_paced(spec, *, root=DEFAULT_ROOT, data_dir=DATA_DIR, scope_name="val-dev-sample", limit=None,
              allow_exposed=None, device="cuda", lm_python=DEFAULT_LM_PYTHON, log=print):
    """1x paced replay (the paced_replay.py path) with measured lag and simulation calibration."""
    if not spec["streaming_capable"]:
        raise ValueError("offline_noncausal pipelines cannot be replayed in real time")
    root = Path(root)
    hasher = HashCache(root)
    identity = registry.identity(spec, hasher)
    scope = build_scope(scope_name, data_dir, limit, allow_exposed, hasher)
    flags = registry.flags(spec, scope["partition"], limited=limit is not None)
    from .train import active_training
    if active_training(root):
        raise RuntimeError(f"Harness-launched training is running ({active_training(root)}); paced timing would be contended")
    with flocked(lock_path(root, "job"), blocking=False), flocked(lock_path(root, "paced")):
        run_dir = new_run_dir(root, spec["name"], "paced")
        manifest = base_manifest(run_dir, "paced", spec, identity, scope, flags, lm_python)
        write_new_json(run_dir / "manifest.json", manifest)
        log(f"Run {run_dir.name}: paced 1x replay of {scope['n_trials']} trials (real time); flags {flags}")
        try:
            from model_training.benchmark.common import TrialSpec
            from model_training.benchmark.lm_worker import LMClient
            from model_training.benchmark.paced_replay import replay_with_checks, synthetic_warmup
            from .acoustic import Runtime
            runtime = Runtime(spec, device)
            manifest["env"] = env_info()
            if runtime.device.type == "cuda":
                runtime.torch.cuda.reset_peak_memory_stats(runtime.device)
            selected = [TrialSpec(t.session, runtime.day_index(t.session), Path(t.hdf5_path), t.trial_key,
                                  int(t.trial_key.split("_")[-1])) for t in scope["trials"]]
            ram_guard(spec["lm"]["expected_rss_gb"], f"LM worker {spec['lm']['name']}")
            metadata = dict(
                timing_boundary="scheduled_released_feature_to_coordinator_output",
                replay_mode="paced_synchronous_two_process", input_clock="bin_end_schedule_20_ms",
                source_feature_availability="simulated_replay_schedule", endpoint=CONSTANTS["endpoint"],
                commitment_policy="none", precision="fp32_tf32_disabled", budgets_ms=list(metrics.BUDGETS_MS),
                budget_definition="output_ready_minus_frame_window_right_edge_availability",
                excluded=["raw_acquisition", "feature_extraction", "display_rendering", "speech_word_alignment"],
                pipeline=spec["name"], config=spec["decode"])
            trace_path = run_dir / "output_trace.jsonl"
            with LMClient(lm_worker_command(spec, lm_python), run_dir / "lm_worker.log", 30.0, 300.0,
                          shutdown_seconds=60.0) as lm:
                warmup = synthetic_warmup(runtime.model, runtime.args, runtime.device, selected[0].day_idx, lm)
                with OutputTraceWriter(trace_path, dict(metadata, warmup=warmup), schema_version=2) as writer:
                    checked = replay_with_checks(selected, runtime.model, runtime.args, runtime.device, lm, writer,
                                                 EQUIVALENCE_TOLERANCE, log)
                worker_rss = lm.peak_rss_kib
            report = summarize_trace(trace_path)
            seen = train_sentence_set(data_dir, root)
            rows = []
            for index, (trial, check, per_trial) in enumerate(zip(scope["trials"], checked, report["trials"])):
                hyp = " ".join(check["words"])
                edits, words = metrics.word_errors(check["reference"], hyp)
                rows.append(dict(i=index, session=trial.session, split=trial.split, trial_key=trial.trial_key,
                                 day_index=check["day_index"], n_bins=per_trial["n_bins"], n_frames=check["n_frames"],
                                 ref=check["reference"], hyp=hyp, ref_words=words, edits=edits,
                                 seen=" ".join(norm_words(check["reference"])) in seen, phone_edits=0, phone_len=0,
                                 n_final_words=per_trial["n_final_words"], revision_edits=per_trial["revision_edits"],
                                 finalization_edit_counts=per_trial["finalization_edit_counts"],
                                 paced=per_trial["paced"], max_logit_error=check["max_logit_error"],
                                 offline_native_text_match=check["offline_native_text_match"]))
            accuracy = metrics.accuracy_summary(rows)
            accuracy["per_diagnostic"] = dict(percent=None, label="not computed in paced runs")
            paced = metrics.paced_summary(trace_path)
            paced.update(structural_delay=metrics.structural_delay(spec["preprocess"]["effective"],
                                                                   runtime.patch_size, runtime.patch_stride),
                         warmup=warmup, resources=dict(
                             worker_peak_rss_kib=worker_rss,
                             peak_cuda_allocated_bytes=runtime.torch.cuda.max_memory_allocated(runtime.device)
                             if runtime.device.type == "cuda" else None))
            standard = matching_standard_run(root, identity, scope)
            calibration = (metrics.sim_calibration(paced, load_run(standard)[1], standard.name) if standard
                           else dict(skipped="no complete standard run of this pipeline used these trials "
                                             "for its timing check"))
            summary = dict(schema="harness_summary_v1", run_id=run_dir.name, tier="paced",
                           pipeline=spec["name"], pipeline_hash=identity["pipeline_hash"],
                           scope=dict(name=scope["name"], hash=scope["hash"], partition=scope["partition"],
                                      n_trials=scope["n_trials"], n_sessions=scope["n_sessions"],
                                      n_participants=scope["n_participants"], limit=limit),
                           flags=flags, accuracy=accuracy, paced=paced, sim_calibration=calibration,
                           constants=CONSTANTS)
            finish(run_dir, manifest, summary, rows)
        except BaseException as exc:
            fail(run_dir, manifest, exc)
            raise
    log(f"Complete: measured lag p95 {paced['frame_window_to_output_ms']['p95']:.2f} ms -> {run_dir}")
    return run_dir


def load_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else None
    rows = None
    if (run_dir / "trials.jsonl").exists():
        with (run_dir / "trials.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
    return manifest, summary, rows


def compare_runs(dir_a, dir_b):
    manifest_a, _, rows_a = load_run(dir_a)
    manifest_b, _, rows_b = load_run(dir_b)
    if manifest_a["scope"]["hash"] != manifest_b["scope"]["hash"]:
        raise ValueError("Runs used different scopes (scope hashes differ); refusing to compare")
    if rows_a is None or rows_b is None:
        raise ValueError("Both runs must be complete")
    result = metrics.paired_compare(rows_a, rows_b)
    result.update(a=manifest_a["run_id"], b=manifest_b["run_id"], scope=manifest_a["scope"]["name"],
                  scope_hash=manifest_a["scope"]["hash"],
                  flags_a=manifest_a["flags"], flags_b=manifest_b["flags"])
    return result
