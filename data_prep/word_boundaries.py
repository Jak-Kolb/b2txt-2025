"""Between-word cut points for every training trial, for phrase-splicing augmentation (dataset.splice_trials;
docs/PLAN.md step 3). Prepared 2026-11-09.

A registered acoustic model's log-probs on each TRAINING trial are force-aligned to the trial's own phoneme labels
(CTC Viterbi, learning/live_eval/clock_a.viterbi_label_ends). For each ' | ' label that separates two words, the cut
is the input bin at the centre of that label's aligned frames: bin = 4 * (first + last) / 2 + 7 for patch 14 /
stride 4. With the bidirectional aligner (default bigru768_pall_fall_20k_s10) emissions sit near the evidence; a causal
aligner would cut late. This is an alignment proxy, not measured word timing. The aligner was trained on these
trials, which is fine for cutting training data and is never used for evaluation.
Output: results/word_boundaries/MODEL.json = {session: {trial index: [[cut_bin, n_labels_before_cut], ...]}}.
usage: word_boundaries.py [--model NAME] [--device cuda|cpu] [--limit N_TRIALS_PER_SESSION] [--sessions S ...]
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model_training")); sys.path.insert(0, str(REPO / "learning/live_eval"))
from clock_a import SIL, viterbi_label_ends
from harness.scopes import list_trials
from model_training.benchmark.common import configure_fp32_inference, load_model, offline_logits


def boundaries(lp, ids, patch=14, stride=4):
    """[[cut_bin, n_labels_before_cut]] for each ' | ' followed by another word, or [] when alignment fails."""
    ends = viterbi_label_ends(lp.astype(np.float32), ids)
    if ends is None:
        return []
    out = []
    for j in range(1, len(ids) - 1):
        if ids[j] == SIL and ids[j - 1] != SIL:
            centre = (ends[j - 1] + 1 + ends[j]) / 2.0
            out.append([int(round(stride * centre + patch / 2)), j + 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bigru768_pall_fall_20k_s10")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    configure_fp32_inference()
    device = torch.device(a.device)
    entry = yaml.safe_load((REPO / "harness/registry/acoustic" / f"{a.model}.yaml").read_text())
    model, args, _ = load_model(str(REPO / entry["checkpoint_dir"]), device)
    sessions = list(args["dataset"]["sessions"])
    result, n, failed = {}, 0, 0
    for t in list_trials(REPO / "data/hdf5_data_final", "train", a.sessions):
        index = int(t.trial_key.split("_")[-1])
        if t.session not in sessions or (a.limit and index >= a.limit):
            continue
        with h5py.File(t.hdf5_path, "r") as h:
            g = h[t.trial_key]
            x = torch.as_tensor(g["input_features"][:], device=device).unsqueeze(0)
            ids = g["seq_class_ids"][: int(g.attrs["seq_len"])].astype(np.int64)
        with torch.inference_mode():
            lp = offline_logits(model, x, sessions.index(t.session), args, device)[0].float().log_softmax(-1).cpu().numpy()
        b = [c for c in boundaries(lp, ids) if 0 < c[0] < x.shape[1]]
        result.setdefault(t.session, {})[str(index)] = b
        n += 1; failed += not b
    out = Path(a.out) if a.out else REPO / "results/word_boundaries" / f"{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    print(f"{a.model}: {n} training trials, {failed} without a usable boundary -> {out}", flush=True)


if __name__ == "__main__":
    main()
