"""Paced released-feature -> streaming GRU -> native WFST replay on JakPC.

A synchronous two-process baseline with fixed scheduled arrivals: if either stage
falls behind, later bins retain their original arrival times and expose backlog.
Raw acquisition, feature extraction, and intended speech-word timing are excluded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

try:
    from .lm_worker import LMClient
    from .output_trace import OutputTraceWriter, summarize_trace
    from .stream_lm import norm_words, levenshtein, VAL_TEST_DAYS
except ImportError:
    from lm_worker import LMClient
    from output_trace import OutputTraceWriter, summarize_trace
    from stream_lm import norm_words, levenshtein, VAL_TEST_DAYS


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LM_PYTHON = "/home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python"
DEFAULT_LM = REPO / "results/lm_gen_p1e-8/data/lang_test"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_geometry(n_bins, patch_size, patch_stride):
    window, stride = (patch_size, patch_stride) if patch_size > 0 else (1, 1)
    if n_bins < 0 or window < 1 or stride < 1:
        raise ValueError("Invalid frame geometry")
    return max(0, 1 + (n_bins - window) // stride), window, stride


def reorder_logits(logits):
    row = np.asarray(logits, dtype=np.float32)
    if row.ndim != 1 or row.size < 2 or not np.isfinite(row).all():
        raise ValueError("Expected one finite acoustic logit vector")
    # Acoustic [blank, phones..., silence] -> WFST [blank, silence, phones...].
    return np.concatenate((row[:1], row[-1:], row[1:-1])).tolist()


def wait_until(target_ns, clock=time.monotonic_ns, sleep=time.sleep):
    """Fixed absolute schedule; never move arrivals forward to hide processing lag."""
    while True:
        remaining = target_ns - clock()
        if remaining <= 0:
            return
        sleep(remaining / 1e9)


class AcousticAdapter:
    """The existing streaming decoder, with device-to-host completion included."""

    def __init__(self, model, args, day_index, device):
        try:
            from .streaming_infer import StreamingDecoder
        except ImportError:
            from streaming_infer import StreamingDecoder
        self.decoder = StreamingDecoder(model, args, day_index, device,
                                        enable_timing=False, keep_logits=False)
        self.device = device

    def _cpu(self, frames):
        import torch
        rows = [frame.detach().cpu().numpy().copy() for frame in frames]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return rows

    def process_bin(self, row):
        return self._cpu(self.decoder.process_bin(row))

    def finish(self):
        return self._cpu(self.decoder.finish())


def replay_trial(features, acoustic, lm, writer, *, trial_index, day_index,
                 patch_size, patch_stride, bin_ns=20_000_000, source=None,
                 clock=time.monotonic_ns, sleep=time.sleep, keep_logits=False):
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("features must have shape [bins, features]")
    if type(bin_ns) is not int or bin_ns <= 0:
        raise ValueError("bin_ns must be a positive integer")
    n_bins = int(features.shape[0])
    n_frames, window, stride = frame_geometry(n_bins, patch_size, patch_stride)
    lm.request("reset")
    writer.start_trial(trial_index, day_index, n_frames, n_bins=n_bins, bin_ns=bin_ns,
                       window_bins=window, stride_bins=stride, source=source)
    origin = clock()
    frame_index = 0
    saved = []
    last_words = []

    def emit(rows, stage, phase):
        nonlocal frame_index, last_words
        for row in rows:
            if keep_logits:
                saved.append(np.asarray(row).copy())
            response = lm.request("frame", frame_index=frame_index, logits=reorder_logits(row))
            pipeline = {k: stage[k] for k in ("input_available_ns", "acoustic_start_ns", "acoustic_ready_ns")}
            pipeline.update(
                worker_received_ns=response["worker_received_ns"] - origin,
                worker_output_ready_ns=response["worker_output_ready_ns"] - origin,
                frame_window_available_ns=(window + frame_index * stride) * bin_ns, phase=phase)
            last_words = response["words"]
            writer.output(kind="partial", frame_index=frame_index, words=last_words,
                          processing_start_ns=response["processing_start_ns"] - origin,
                          output_ready_ns=response["output_ready_ns"] - origin, pipeline=pipeline)
            frame_index += 1

    for index in range(n_bins):
        available = (index + 1) * bin_ns
        wait_until(origin + available, clock, sleep)
        start = clock() - origin
        if not np.isfinite(features[index]).all():
            raise ValueError(f"Nonfinite features at bin {index}")
        rows = acoustic.process_bin(features[index])
        ready = clock() - origin
        stage = dict(input_available_ns=available, acoustic_start_ns=start, acoustic_ready_ns=ready)
        writer.input(bin_index=index, n_emitted=len(rows), **stage)
        emit(rows, stage, "stream")

    wait_until(origin + n_bins * bin_ns, clock, sleep)
    start = clock() - origin
    tail = acoustic.finish()
    stage = dict(input_available_ns=n_bins * bin_ns, acoustic_start_ns=start,
                 acoustic_ready_ns=clock() - origin)
    writer.input(endpoint=True, n_emitted=len(tail), **stage)
    emit(tail, stage, "endpoint_flush")
    if frame_index != n_frames:
        raise RuntimeError(f"Streaming frame count {frame_index} differs from expected {n_frames}")
    response = lm.request("finish")
    writer.output(
        kind="final", frame_index=frame_index - 1 if frame_index else None,
        words=response["words"], processing_start_ns=response["processing_start_ns"] - origin,
        output_ready_ns=response["output_ready_ns"] - origin,
        pipeline=dict(**stage, worker_received_ns=response["worker_received_ns"] - origin,
                      worker_output_ready_ns=response["worker_output_ready_ns"] - origin,
                      frame_window_available_ns=None, phase="finalization"))
    return dict(words=response["words"], last_partial_words=last_words, logits=saved,
                n_frames=frame_index, duration_seconds=n_bins * bin_ns / 1e9)


def unpaced_decode(lm, logits):
    lm.request("reset")
    for i, row in enumerate(logits):
        lm.request("frame", frame_index=i, logits=reorder_logits(row))
    return lm.request("finish")["words"]


def synthetic_warmup(model, args, device, day_index, lm):
    """Warm both stages before timing; no data trial or reference is used."""
    n_bins = max(int(args["model"]["patch_size"]), 1) + 12
    acoustic = AcousticAdapter(model, args, day_index, device)
    lm.request("reset")
    index = 0
    for _ in range(n_bins):
        for row in acoustic.process_bin(np.zeros(int(args["model"]["n_input_features"]), dtype=np.float32)):
            lm.request("frame", frame_index=index, logits=reorder_logits(row))
            index += 1
    for row in acoustic.finish():
        lm.request("frame", frame_index=index, logits=reorder_logits(row))
        index += 1
    lm.request("finish")
    return dict(kind="synthetic_zero_features", n_bins=n_bins, n_frames=index)


def select_development_trials(args, data_dir, n_trials, max_trial_seconds):
    """Deterministic first N eligible validation trials; never includes old val-test."""
    try:
        from .common import discover_trials
    except ImportError:
        from common import discover_trials
    import h5py
    selected = []
    for trial in discover_trials(args, data_dir, split="val"):
        if trial.day_idx in VAL_TEST_DAYS:
            continue
        with h5py.File(trial.hdf5_path, "r") as handle:
            n_bins = int(handle[trial.trial_key]["input_features"].shape[0])
        if n_bins * 0.020 > max_trial_seconds:
            raise ValueError("Selected trial exceeds max_trial_seconds; choose an explicit revised limit")
        selected.append(trial)
        if len(selected) == n_trials:
            return selected
    raise ValueError(f"Requested {n_trials} development trials; only {len(selected)} eligible trials found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(REPO / "results/causal_la0/checkpoint"))
    parser.add_argument("--data_dir", default=str(REPO / "data/hdf5_data_final"))
    parser.add_argument("--lm", default=str(DEFAULT_LM))
    parser.add_argument("--lm_python", default=DEFAULT_LM_PYTHON)
    parser.add_argument("--out_dir", required=True, help="Fresh directory under ignored results/")
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--max_trial_seconds", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budgets_ms", default="200,500,1500",
                        help="Diagnostic frame-window lag thresholds, not word-latency guarantees")
    parser.add_argument("--acoustic_scale", type=float, default=0.4)
    parser.add_argument("--blank_penalty", type=float, default=90.0)
    parser.add_argument("--beam", type=float, default=17.0)
    parser.add_argument("--lattice_beam", type=float, default=8.0)
    parser.add_argument("--max_active", type=int, default=7000)
    parser.add_argument("--min_active", type=int, default=200)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--blank_skip_thresh", type=float, default=1.0)
    parser.add_argument("--worker_timeout", type=float, default=30.0)
    parser.add_argument("--startup_timeout", type=float, default=120.0)
    parser.add_argument("--shutdown_timeout", type=float, default=60.0)
    parser.add_argument("--equivalence_tolerance", type=float, default=1e-3)
    options = parser.parse_args()
    if options.n_trials < 1:
        parser.error("n_trials must be positive")
    budgets = [float(x) for x in options.budgets_ms.split(",")]
    for value in [*budgets, options.max_trial_seconds, options.worker_timeout,
                  options.startup_timeout, options.shutdown_timeout, options.equivalence_tolerance, options.blank_penalty]:
        if not math.isfinite(value) or value <= 0:
            parser.error("Budgets, bounds, tolerance, and blank penalty must be positive and finite")
    out_dir = Path(options.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {out_dir}")

    # Native acoustic imports stay out of Python 3.9 and lightweight protocol tests.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import h5py
    try:
        from .common import (args_to_container, configure_fp32_inference, get_feature_subset,
                             load_args, load_model, load_trial_features, offline_logits,
                             select_device, sync_device)
    except ImportError:
        from common import (args_to_container, configure_fp32_inference, get_feature_subset,
                            load_args, load_model, load_trial_features, offline_logits,
                            select_device, sync_device)
    setup_start = time.monotonic()
    configure_fp32_inference()
    device = select_device(options.device)
    checkpoint_dir = Path(options.checkpoint).resolve()
    args = load_args(checkpoint_dir)
    selected = select_development_trials(args, options.data_dir, options.n_trials, options.max_trial_seconds)
    config = {k: getattr(options, k) for k in (
        "acoustic_scale", "blank_penalty", "beam", "lattice_beam", "max_active", "min_active",
        "length_penalty", "blank_skip_thresh")}
    sources = [checkpoint_dir / "best_checkpoint", checkpoint_dir / "args.yaml",
               Path(options.lm) / "TLG.fst", Path(options.lm) / "words.txt"]
    sources += sorted({t.hdf5_path for t in selected})
    code_files = ["paced_replay.py", "lm_worker.py", "output_trace.py", "streaming_infer.py", "common.py"]
    sources += [Path(__file__).with_name(n) for n in code_files]
    identities = {str(p.resolve()): sha256(p) for p in sources}
    out_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        timing_boundary="scheduled_released_feature_to_coordinator_output",
        replay_mode="paced_synchronous_two_process", input_clock="bin_end_schedule_20_ms",
        source_feature_availability="simulated_replay_schedule", endpoint="provided_trial_end",
        commitment_policy="none", precision="fp32_tf32_disabled", budgets_ms=budgets,
        budget_definition="output_ready_minus_frame_window_right_edge_availability",
        excluded=["raw_acquisition", "feature_extraction", "display_rendering", "speech_word_alignment"],
        warmup=None, config=config, acoustic_config=args_to_container(args), source_sha256=identities,
        date=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        acoustic_python=sys.version.split()[0], torch=torch.__version__, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        provenance_warning="released-feature preprocessing unresolved; dev data, not an independent test",
        selection="first_N_enabled_validation_trials_excluding_former_val_test_days",
        n_trials=len(selected), diagnostic_only=True, status="running",
        worker_timeouts_seconds=dict(request=options.worker_timeout, startup=options.startup_timeout,
                                     shutdown=options.shutdown_timeout))
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    model, args, _ = load_model(checkpoint_dir, device, args)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    command = [options.lm_python, "-u", str(Path(__file__).with_name("lm_worker.py")),
               "--lm", str(Path(options.lm).resolve()), "--config", json.dumps(config),
               "--n_classes", str(args["dataset"]["n_classes"])]
    scores = []
    trace_path = out_dir / "output_trace.jsonl"
    with LMClient(command, out_dir / "lm_worker.log", options.worker_timeout, options.startup_timeout,
                  shutdown_seconds=options.shutdown_timeout) as lm:
        if lm.ready["n_classes"] != int(args["dataset"]["n_classes"]):
            raise ValueError("Acoustic and LM class counts differ")
        metadata["lm_python"] = lm.ready["python"]
        metadata["warmup"] = synthetic_warmup(model, args, device, selected[0].day_idx, lm)
        metadata["setup_seconds"] = time.monotonic() - setup_start
        metadata["equivalence_tolerance"] = options.equivalence_tolerance
        with OutputTraceWriter(trace_path, metadata, schema_version=2) as writer:
            for index, trial in enumerate(selected):
                features, _attrs = load_trial_features(trial, get_feature_subset(args))
                acoustic = AcousticAdapter(model, args, trial.day_idx, device)
                sync_device(device)
                result = replay_trial(
                    features, acoustic, lm, writer, trial_index=index, day_index=trial.day_idx,
                    patch_size=int(args["model"]["patch_size"]), patch_stride=int(args["model"]["patch_stride"]),
                    source=dict(session=trial.session, trial_key=trial.trial_key), keep_logits=True)
                # References and the offline comparison are accessed only after the timed trial.
                with h5py.File(trial.hdf5_path, "r") as handle:
                    encoded = handle[trial.trial_key]["transcription"][:]
                reference = bytes(encoded[encoded > 0].astype(np.uint8)).decode("ascii").strip()
                with torch.inference_mode():
                    offline = offline_logits(model, torch.as_tensor(features, device=device).unsqueeze(0),
                                             trial.day_idx, args, device)[0].float().cpu().numpy()
                streamed = np.stack(result["logits"]) if result["logits"] else np.empty_like(offline)
                if offline.shape != streamed.shape:
                    raise AssertionError("Streaming/offline logit shapes differ")
                max_error = float(np.max(np.abs(offline - streamed))) if offline.size else 0.0
                if not max_error < options.equivalence_tolerance:
                    raise AssertionError(f"Streaming/offline max error {max_error} exceeds tolerance")
                offline_words = unpaced_decode(lm, offline)
                if offline_words != result["words"]:
                    raise AssertionError("Native LM final text differs between streamed and offline logits")
                rw, hw = norm_words(reference), norm_words(" ".join(result["words"]))
                scores.append(dict(trial_index=index, day_index=trial.day_idx, n_frames=result["n_frames"],
                                   duration_seconds=result["duration_seconds"], ref_words=len(rw),
                                   edits=levenshtein(rw, hw), max_logit_error=max_error,
                                   offline_native_text_match=True))
                print(f"Trial {index + 1}/{len(selected)} complete: {result['n_frames']} frames, equivalence passed",
                      flush=True)
        worker_rss = lm.peak_rss_kib
    report = summarize_trace(trace_path)
    ref_words = sum(s["ref_words"] for s in scores)
    report.update(
        diagnostic_only=True, selection=metadata["selection"], accuracy=dict(
            edits=sum(s["edits"] for s in scores), ref_words=ref_words,
            wer_percent=100 * sum(s["edits"] for s in scores) / ref_words if ref_words else None,
            per_trial=scores, seen_unseen="not stratified for this bounded plumbing smoke"),
        resources=dict(worker_peak_rss_kib=worker_rss,
                       coordinator_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None))
    with (out_dir / "summary.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    # Update only this new run's manifest after successful completion.
    metadata.update(status="complete", output_trace_sha256=sha256(trace_path),
                    summary_sha256=sha256(out_dir / "summary.json"))
    manifest_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Completed bounded replay: {len(scores)} trials; results in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
