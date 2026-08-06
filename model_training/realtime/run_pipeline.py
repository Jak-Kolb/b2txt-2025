#!/usr/bin/env python3
"""CLI for the grok_recs real-time pipeline experiments (branch grok_test).

Subcommands:
  ensemble     — R1 multi-seed logit average, report greedy PER
  phase        — R3 phase-stagger vs single-phase PER + L_buf meta
  peak-delay   — S6 CTC peak delay measurement + kill/go decision
  adaptive     — R2 blank-skip rates + entropy-trigger curve (dry-run rescoring)
  stability    — stability metrics on greedy partial streams
  decode-sweep — eval-only blank threshold / entropy threshold grid
  all-smoke    — short n_trials run of the above (CI / laptop smoke)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import editdistance
import numpy as np
import torch

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"
MODEL_TRAINING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MODEL_TRAINING_DIR.parent
for p in (str(BENCHMARK_DIR), str(MODEL_TRAINING_DIR), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import (  # type: ignore  # noqa: E402
    configure_fp32_inference,
    discover_trials,
    get_feature_subset,
    greedy_ctc_decode_tensor,
    load_args,
    load_model,
    load_trial_features,
    offline_logits,
    resolve_path,
    select_device,
    write_json,
)

from ensemble import EnsembleDecoder  # noqa: E402
from phase_stagger import PhaseStaggerConfig, phase_stagger_logits  # noqa: E402
from peak_delay import measure_peak_delays, summarize_peak_delays  # noqa: E402
from streaming_decode import (  # noqa: E402
    AdaptiveRescoreConfig,
    greedy_ctc_with_blank_skip,
    run_adaptive_policy_over_logits,
)
from stability import compute_stability_metrics  # noqa: E402
from streaming_decode import stream_greedy_partials  # noqa: E402


def _load_seq_labels(trial) -> Optional[np.ndarray]:
    import h5py

    with h5py.File(trial.hdf5_path, "r") as h5:
        g = h5[trial.trial_key]
        if "seq_class_ids" not in g:
            return None
        ids = g["seq_class_ids"][:].astype(np.int64)
        seq_len = int(g.attrs["seq_len"]) if "seq_len" in g.attrs else None
        if seq_len is not None:
            ids = ids[:seq_len]
        return ids


def _per(pred: np.ndarray, true: np.ndarray) -> float:
    if len(true) == 0:
        return 0.0
    return editdistance.eval(list(pred), list(true)) / len(true)


def _collect_trials(args, data_dir, split, n_trials, include_disabled_days):
    return discover_trials(
        args,
        data_dir,
        split=split,
        n_trials=n_trials,
        include_disabled_days=include_disabled_days,
    )


def cmd_ensemble(ns: argparse.Namespace) -> Dict[str, Any]:
    device = select_device(ns.device)
    dirs = [resolve_path(d) for d in ns.checkpoint_dirs]
    # Allow single dir repeated? No — if one dir, N=1 baseline.
    ensemble = EnsembleDecoder.from_checkpoint_dirs(dirs, device, average_mode=ns.average_mode)
    ref_args = ensemble.args
    trials = _collect_trials(
        ref_args, ns.data_dir, ns.split, ns.n_trials, ns.include_disabled_days
    )
    feature_subset = get_feature_subset(ref_args)

    # Also eval first member alone for delta.
    single_model, single_args, _ = load_model(dirs[0], device)

    ens_ed = 0
    ens_len = 0
    single_ed = 0
    single_len = 0
    t0 = time.perf_counter()
    n_done = 0
    for trial in trials:
        feats, _ = load_trial_features(trial, feature_subset)
        labels = _load_seq_labels(trial)
        if labels is None:
            continue
        raw = torch.as_tensor(feats, device=device, dtype=torch.float32)
        with torch.inference_mode():
            ens_logits = ensemble.forward_offline(raw, trial.day_idx)[0]
            single_logits = offline_logits(
                single_model, raw.unsqueeze(0), trial.day_idx, single_args, device
            )[0]
        ens_pred = greedy_ctc_decode_tensor(ens_logits)
        single_pred = greedy_ctc_decode_tensor(single_logits)
        ens_ed += editdistance.eval(list(ens_pred), list(labels))
        ens_len += len(labels)
        single_ed += editdistance.eval(list(single_pred), list(labels))
        single_len += len(labels)
        n_done += 1

    elapsed = time.perf_counter() - t0
    result = {
        "mode": "ensemble",
        "n_members": ensemble.n_members,
        "average_mode": ns.average_mode,
        "checkpoint_dirs": [str(d) for d in dirs],
        "n_trials": n_done,
        "ensemble_per": 100.0 * ens_ed / ens_len if ens_len else float("nan"),
        "single_per": 100.0 * single_ed / single_len if single_len else float("nan"),
        "delta_per_pp": (
            100.0 * ens_ed / ens_len - 100.0 * single_ed / single_len
            if ens_len and single_len
            else float("nan")
        ),
        "wall_sec": elapsed,
        "info": ensemble.info(),
    }
    return result


def cmd_phase(ns: argparse.Namespace) -> Dict[str, Any]:
    device = select_device(ns.device)
    ckpt = resolve_path(ns.checkpoint_dir)
    model, args, _ = load_model(ckpt, device)
    trials = _collect_trials(args, ns.data_dir, ns.split, ns.n_trials, ns.include_disabled_days)
    feature_subset = get_feature_subset(args)
    config = PhaseStaggerConfig(n_phases=ns.n_phases, merge_mode=ns.average_mode)

    base_ed = phase_ed = 0
    base_len = phase_len = 0
    metas: List[Dict[str, Any]] = []
    n_done = 0
    for trial in trials:
        feats, _ = load_trial_features(trial, feature_subset)
        labels = _load_seq_labels(trial)
        if labels is None:
            continue
        raw = torch.as_tensor(feats, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            base_logits = offline_logits(model, raw, trial.day_idx, args, device)[0]
            phase_logits, meta = phase_stagger_logits(
                model, raw, trial.day_idx, args, device, config=config
            )
            phase_logits = phase_logits[0]
        base_pred = greedy_ctc_decode_tensor(base_logits)
        phase_pred = greedy_ctc_decode_tensor(phase_logits)
        base_ed += editdistance.eval(list(base_pred), list(labels))
        phase_ed += editdistance.eval(list(phase_pred), list(labels))
        base_len += len(labels)
        phase_len += len(labels)
        metas.append(meta)
        n_done += 1

    l_buf = metas[0]["l_buf_effective_ms"] if metas else float("nan")
    return {
        "mode": "phase_stagger",
        "n_phases": ns.n_phases,
        "n_trials": n_done,
        "baseline_per": 100.0 * base_ed / base_len if base_len else float("nan"),
        "phase_per": 100.0 * phase_ed / phase_len if phase_len else float("nan"),
        "delta_per_pp": (
            100.0 * phase_ed / phase_len - 100.0 * base_ed / base_len
            if base_len and phase_len
            else float("nan")
        ),
        "l_buf_effective_ms": l_buf,
        "l_buf_baseline_ms": metas[0]["l_buf_baseline_ms"] if metas else 80.0,
        "meta_example": metas[0] if metas else {},
    }


def cmd_peak_delay(ns: argparse.Namespace) -> Dict[str, Any]:
    device = select_device(ns.device)
    ckpt = resolve_path(ns.checkpoint_dir)
    model, args, _ = load_model(ckpt, device)
    trials = _collect_trials(args, ns.data_dir, ns.split, ns.n_trials, ns.include_disabled_days)
    feature_subset = get_feature_subset(args)
    patch_stride = int(args["model"]["patch_stride"])

    trial_results = []
    for trial in trials:
        feats, _ = load_trial_features(trial, feature_subset)
        labels = _load_seq_labels(trial)
        if labels is None:
            continue
        raw = torch.as_tensor(feats, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            logits = offline_logits(model, raw, trial.day_idx, args, device)[0]
        trial_results.append(
            measure_peak_delays(logits, labels, patch_stride=patch_stride)
        )

    summary = summarize_peak_delays(trial_results)
    summary["mode"] = "peak_delay"
    summary["checkpoint_dir"] = str(ckpt)
    return summary


def cmd_adaptive(ns: argparse.Namespace) -> Dict[str, Any]:
    device = select_device(ns.device)
    ckpt = resolve_path(ns.checkpoint_dir)
    model, args, _ = load_model(ckpt, device)
    trials = _collect_trials(args, ns.data_dir, ns.split, ns.n_trials, ns.include_disabled_days)
    feature_subset = get_feature_subset(args)
    patch_stride = int(args["model"]["patch_stride"])
    config = AdaptiveRescoreConfig(
        entropy_threshold=ns.entropy_threshold,
        blank_threshold=ns.blank_threshold,
    )

    trigger_rates = []
    skip_rates = []
    ents = []
    per_base = []
    per_skip = []
    n_done = 0
    for trial in trials:
        feats, _ = load_trial_features(trial, feature_subset)
        labels = _load_seq_labels(trial)
        if labels is None:
            continue
        raw = torch.as_tensor(feats, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            logits = offline_logits(model, raw, trial.day_idx, args, device)[0]
        pol = run_adaptive_policy_over_logits(
            logits, config=config, rescorer=None, patch_stride=patch_stride
        )
        trigger_rates.append(pol["trigger_rate"])
        skip_rates.append(pol["blank_skip_rate"])
        ents.append(pol["mean_entropy"])
        pred_base = greedy_ctc_decode_tensor(logits)
        pred_skip = greedy_ctc_with_blank_skip(
            logits, blank_threshold=ns.blank_threshold, use_skip=True
        )
        if len(labels):
            per_base.append(editdistance.eval(list(pred_base), list(labels)) / len(labels))
            per_skip.append(editdistance.eval(list(pred_skip), list(labels)) / len(labels))
        n_done += 1

    return {
        "mode": "adaptive",
        "n_trials": n_done,
        "mean_trigger_rate": float(np.mean(trigger_rates)) if trigger_rates else float("nan"),
        "mean_blank_skip_rate": float(np.mean(skip_rates)) if skip_rates else float("nan"),
        "mean_entropy": float(np.mean(ents)) if ents else float("nan"),
        "greedy_per": 100.0 * float(np.mean(per_base)) if per_base else float("nan"),
        "blank_skip_greedy_per": 100.0 * float(np.mean(per_skip)) if per_skip else float("nan"),
        "config": {
            "entropy_threshold": ns.entropy_threshold,
            "blank_threshold": ns.blank_threshold,
        },
    }


def cmd_stability(ns: argparse.Namespace) -> Dict[str, Any]:
    device = select_device(ns.device)
    ckpt = resolve_path(ns.checkpoint_dir)
    model, args, _ = load_model(ckpt, device)
    trials = _collect_trials(args, ns.data_dir, ns.split, ns.n_trials, ns.include_disabled_days)
    feature_subset = get_feature_subset(args)
    patch_stride = int(args["model"]["patch_stride"])

    metrics = []
    for trial in trials:
        feats, _ = load_trial_features(trial, feature_subset)
        raw = torch.as_tensor(feats, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            logits = offline_logits(model, raw, trial.day_idx, args, device)[0]
        state = stream_greedy_partials(logits, patch_stride=patch_stride)
        m = compute_stability_metrics(
            state.partial_history, state.emission_times_ms, state.partial_text
        )
        metrics.append(m)

    if not metrics:
        return {"mode": "stability", "n_trials": 0}

    return {
        "mode": "stability",
        "n_trials": len(metrics),
        "mean_ttf_ms": float(np.nanmean([m["mean_ttf_ms"] for m in metrics])),
        "mean_rpv": float(np.mean([m["mean_rpv"] for m in metrics])),
        "mean_prefix_freeze_rate": float(np.mean([m["prefix_freeze_rate"] for m in metrics])),
        "median_ttf_ms": float(np.nanmedian([m["median_ttf_ms"] for m in metrics])),
    }


def cmd_decode_sweep(ns: argparse.Namespace) -> Dict[str, Any]:
    device = select_device(ns.device)
    ckpt = resolve_path(ns.checkpoint_dir)
    model, args, _ = load_model(ckpt, device)
    trials = _collect_trials(args, ns.data_dir, ns.split, ns.n_trials, ns.include_disabled_days)
    feature_subset = get_feature_subset(args)

    blank_grid = [float(x) for x in ns.blank_thresholds.split(",")]
    # Precompute logits once
    cache = []
    for trial in trials:
        feats, _ = load_trial_features(trial, feature_subset)
        labels = _load_seq_labels(trial)
        if labels is None:
            continue
        raw = torch.as_tensor(feats, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            logits = offline_logits(model, raw, trial.day_idx, args, device)[0]
        cache.append((logits.cpu(), labels))

    rows = []
    for thr in blank_grid:
        ed = length = 0
        for logits_cpu, labels in cache:
            pred = greedy_ctc_with_blank_skip(
                logits_cpu, blank_threshold=thr, use_skip=True
            )
            ed += editdistance.eval(list(pred), list(labels))
            length += len(labels)
        rows.append(
            {
                "blank_threshold": thr,
                "per": 100.0 * ed / length if length else float("nan"),
                "n_trials": len(cache),
            }
        )
    # Also baseline no-skip
    ed = length = 0
    for logits_cpu, labels in cache:
        pred = greedy_ctc_decode_tensor(logits_cpu)
        ed += editdistance.eval(list(pred), list(labels))
        length += len(labels)
    baseline = 100.0 * ed / length if length else float("nan")
    return {
        "mode": "decode_sweep",
        "baseline_greedy_per": baseline,
        "blank_threshold_sweep": rows,
    }


def cmd_all_smoke(ns: argparse.Namespace) -> Dict[str, Any]:
    """Compact multi-check for CI / laptop."""
    ns_local = argparse.Namespace(**vars(ns))
    ns_local.n_trials = ns.n_trials or 5
    results = {}
    # peak delay + adaptive + stability + decode sweep need one ckpt
    results["peak_delay"] = cmd_peak_delay(ns_local)
    results["adaptive"] = cmd_adaptive(ns_local)
    results["stability"] = cmd_stability(ns_local)
    results["decode_sweep"] = cmd_decode_sweep(ns_local)
    # phase
    results["phase"] = cmd_phase(ns_local)
    # ensemble if multiple dirs provided
    if ns.checkpoint_dirs and len(ns.checkpoint_dirs) >= 1:
        ens_ns = argparse.Namespace(**vars(ns_local))
        ens_ns.checkpoint_dirs = ns.checkpoint_dirs
        ens_ns.average_mode = ns.average_mode
        results["ensemble"] = cmd_ensemble(ens_ns)
    results["mode"] = "all_smoke"
    return results


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--checkpoint_dir",
        default="results/trained_models/causal_la0/checkpoint",
        help="Primary causal checkpoint directory.",
    )
    p.add_argument(
        "--checkpoint_dirs",
        nargs="*",
        default=None,
        help="Ensemble member checkpoint dirs (R1). Defaults to primary if omitted.",
    )
    p.add_argument("--data_dir", default="data/hdf5_data_final")
    p.add_argument("--split", default="val")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    p.add_argument("--include_disabled_days", action="store_true")
    p.add_argument(
        "--average_mode",
        default="prob_mean",
        choices=["prob_mean", "logit_mean", "logprob_mean"],
    )
    p.add_argument("--n_phases", type=int, default=4)
    p.add_argument("--entropy_threshold", type=float, default=1.5)
    p.add_argument("--blank_threshold", type=float, default=0.7)
    p.add_argument(
        "--blank_thresholds",
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated blank thresholds for decode-sweep.",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="grok_test real-time pipeline runner")
    sub = p.add_subparsers(dest="command", required=True)
    for name in (
        "ensemble",
        "phase",
        "peak-delay",
        "adaptive",
        "stability",
        "decode-sweep",
        "all-smoke",
    ):
        sp = sub.add_parser(name)
        _add_shared_args(sp)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    configure_fp32_inference()

    if ns.checkpoint_dirs is None:
        ns.checkpoint_dirs = [ns.checkpoint_dir]

    dispatch = {
        "ensemble": cmd_ensemble,
        "phase": cmd_phase,
        "peak-delay": cmd_peak_delay,
        "adaptive": cmd_adaptive,
        "stability": cmd_stability,
        "decode-sweep": cmd_decode_sweep,
        "all-smoke": cmd_all_smoke,
    }
    result = dispatch[ns.command](ns)
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    print(text)
    if ns.out:
        out_path = Path(ns.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        write_json(out_path, result)
        print(f"\nWrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
