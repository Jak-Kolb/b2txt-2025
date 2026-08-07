"""C9 — per-phase greedy PER. The deciding test for C16 (phase staggering).

The patcher takes P=14 bins with stride s=4, so a frame's right edge lands at bin 4f+13. Four
phase-shifted copies of the same stream (offsets p in {0,1,2,3}) therefore emit on a common 20 ms
grid rather than an 80 ms one -- which is the only mechanism in the plan that converts idle compute
into *latency* (L_buf 60 -> 15 ms worst case) rather than into accuracy.

That only works if the model is actually phase-invariant. `random_cut: 3` draws {0,1,2}, so
training covers 3 of the 4 phases; phase 3 is never seen. This measures what that costs.

Gate G1-c: per-phase greedy PER spans <= 0.5 points across p in {0,1,2,3}. If it fails, C16 waits
for Group 2's `random_cut: 4` rather than spending a run on it.

    cd model_training && ../.venv/bin/python -m benchmark.phase_ensemble
"""
import argparse
import json
import pathlib
import sys

import h5py
import numpy as np
import torch

try:
    from .common import (
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
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from common import (  # type: ignore
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
    )

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from splits import is_val_test  # noqa: E402


def rearrange(logits):
    """[BLANK, phonemes..., SIL] -> [BLANK, SIL, phonemes...], matching
    evaluate_model_helpers.rearrange_speech_logits_pt (:79-83) and stream_lm.py's export."""
    return np.concatenate((logits[:, 0:1], logits[:, -1:], logits[:, 1:-1]), axis=-1)


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def merge_interleaved(streams, phases):
    """C16(a) — one 20 ms grid instead of four staggered 80 ms ones.

    Frame f of phase p has its right edge at original bin p + 4f + 13, so sorting every phase's
    frames by right edge yields a single stream at 4x the frame rate. This is the LATENCY half of
    C16: worst-case L_buf falls 60 -> 15 ms because a frame closes every bin rather than every
    fourth. It quadruples the LM's frame rate, which is the cost side.
    """
    tagged = []
    for p, s in zip(phases, streams):
        for f in range(s.shape[0]):
            tagged.append((p + 4 * f + 13, p, s[f]))
    tagged.sort(key=lambda r: (r[0], r[1]))
    return np.stack([r[2] for r in tagged], axis=0)


def merge_phase_averaged(streams):
    """C16(b) — the ACCURACY half, reported separately because it is a different mechanism.

    Averages in log-probability space (a geometric mean of posteriors), not in logit space: each
    phase carries its own log-partition, so averaging raw logits mixes unnormalized quantities.
    DecodeNumpy's internal log_softmax is idempotent on log-probs, so this passes through cleanly.

    Frame f of phase p sits at bin p+4f+13, so a common f smears the four members over 3 bins
    (60 ms). That smear is inherent to the alignment and is why (b) buys no latency.
    """
    n = min(s.shape[0] for s in streams)
    return np.mean(np.stack([log_softmax(s[:n]) for s in streams], axis=0), axis=0)


def write_cache(path, trials_logits, texts, days, tag):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_trials=np.int64(len(trials_logits)),
        lengths=np.array([a.shape[0] for a in trials_logits], dtype=np.int64),
        days=np.array(days, dtype=np.int64),
        flat=np.concatenate(trials_logits, axis=0).astype(np.float32),
        avg_PER=np.float64(np.nan),
    )
    path.with_suffix(".txt").write_text("\n".join(t.replace("\n", " ") for t in texts))
    frames = sum(a.shape[0] for a in trials_logits)
    print(f"  {tag:<12} {len(trials_logits)} trials / {frames} frames -> {path} "
          f"({path.stat().st_size/1e6:.1f} MB)")


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_labels(trial):
    with h5py.File(trial.hdf5_path, "r") as h5:
        return np.asarray(h5[trial.trial_key]["seq_class_ids"][:], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default="results/causal_la0/checkpoint")
    ap.add_argument("--data_dir", default="data/hdf5_data_final")
    ap.add_argument("--split", default="val")
    ap.add_argument("--phases", default="0,1,2,3")
    ap.add_argument("--n_trials", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=0.5, help="G1-c spread threshold, PER points")
    ap.add_argument("--out", default="results/c9_phase_per.json")
    ap.add_argument("--dump", action="store_true",
                    help="C16: also write interleaved and phase-averaged logit caches for the LM")
    ap.add_argument("--dump_prefix", default="results/la0_phase")
    a = ap.parse_args()

    configure_fp32_inference()
    device = select_device(a.device)
    ckpt = resolve_path(a.checkpoint_dir)
    args = load_args(ckpt)
    model, args, _ = load_model(ckpt, device, args=args)
    feature_subset = get_feature_subset(args)

    trials = discover_trials(args=args, data_dir=a.data_dir, split=a.split,
                             n_trials=a.n_trials, include_disabled_days=False)
    trials = [t for t in trials if not is_val_test(t.day_idx)]   # val-dev only
    phases = [int(p) for p in a.phases.split(",")]
    print(f"{len(trials)} val-dev trials, phases {phases}, checkpoint {ckpt}\n")

    edits = {p: 0 for p in phases}
    lengths = {p: 0 for p in phases}
    per_trial_frames = {p: 0 for p in phases}
    interleaved, phaseavg, texts, days = [], [], [], []

    for index, trial in enumerate(trials):
        raw, attrs = load_trial_features(trial, feature_subset)
        labels = load_labels(trial)
        labels = labels[labels > 0]                              # strip padding
        streams = []
        for p in phases:
            shifted = raw[p:]
            if shifted.shape[0] < 1:
                continue
            tensor = torch.as_tensor(shifted, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.inference_mode():
                logits = offline_logits(model, tensor, trial.day_idx, args, device)[0].float()
            if a.dump:
                streams.append(rearrange(logits.cpu().numpy().astype(np.float32)))
            hyp = greedy_ctc_decode_tensor(logits)
            # int(): comparing numpy scalars inside levenshtein promotes the running cost to
            # np.int64, which json cannot serialize.
            edits[p] += int(levenshtein([int(x) for x in labels], [int(x) for x in hyp]))
            lengths[p] += int(len(labels))
            per_trial_frames[p] += int(logits.shape[0])
        if a.dump and len(streams) == len(phases):
            interleaved.append(merge_interleaved(streams, phases))
            phaseavg.append(merge_phase_averaged(streams))
            texts.append(str(attrs["sentence_label"]).strip())
            days.append(int(trial.day_idx))
        if (index + 1) % 200 == 0:
            run = {p: 100.0 * edits[p] / max(lengths[p], 1) for p in phases}
            print(f"  {index+1}/{len(trials)}  " +
                  "  ".join(f"p{p} {run[p]:.2f}%" for p in phases), flush=True)

    # Micro-averaged, matching rnn_trainer validation: total edits / total label length.
    per = {p: 100.0 * edits[p] / max(lengths[p], 1) for p in phases}
    spread = max(per.values()) - min(per.values())

    print("\n" + "=" * 58)
    print(f"{'phase':>6} {'greedy PER %':>14} {'frames':>10} {'vs p0':>9}")
    print("-" * 58)
    for p in phases:
        print(f"{p:>6} {per[p]:>14.3f} {per_trial_frames[p]:>10} {per[p]-per[phases[0]]:>+9.3f}")
    print("-" * 58)
    ok = spread <= a.tol
    print(f"  spread across phases : {spread:.3f} pts")
    print(f"  G1-c (<= {a.tol} pts)     : {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  -> C16 defers behind Group 2's random_cut: 4; do not spend a run on it now.")
    print("=" * 58)

    if a.out:
        out = resolve_path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint_dir": str(ckpt), "split": "val-dev", "n_trials": len(trials),
            "per_percent": per, "edits": edits, "label_tokens": lengths,
            "frames": per_trial_frames, "spread_points": spread,
            "gate_G1c_pass": bool(ok),
        }, indent=2))
        print(f"wrote {out}")

    if a.dump:
        print("\nC16 logit caches (decode these with `stream_lm.py decode --split all`):")
        write_cache(f"{a.dump_prefix}_interleaved.npz", interleaved, texts, days, "interleaved")
        write_cache(f"{a.dump_prefix}_avg.npz", phaseavg, texts, days, "phase-avg")
        print("  NOTE: the interleaved stream runs at 50 Hz, not 12.5 Hz. acoustic_scale and")
        print("  blank_penalty were tuned at 12.5 Hz, so re-sweep before comparing its WER.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
