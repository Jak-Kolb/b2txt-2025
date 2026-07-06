"""
Check whether the pre-extracted HDF5 neural features were normalized using
whole-block statistics (which leaks future timesteps within a block and is
therefore non-causal for real-time decoding).

The released files only contain the already-normalized `input_features` per
trial -- there is no separate raw/normalized pair, so we cannot reproduce the
exact raw -> norm transform the way the original template did. Instead we test
for the *signature* of block-level z-scoring:

    If features were z-scored per (block, feature) over the whole block, then
    concatenating every trial in a block and computing the per-feature mean/std
    over that concatenation must give mean ~= 0 and std ~= 1. A single trial
    will NOT satisfy this (it is only a slice of the block).

Blocks can be split across the train/val/test files, so we regather all trials
sharing a block_num across all three splits before computing block statistics.

Usage:
    python check_hdf5_causality.py [--data_dir ../data/hdf5_data_final] [--session t15.2023.08.27]
    (omit --session to scan every session directory)
"""

import os
import argparse
from collections import defaultdict

import h5py
import numpy as np


def gather_blocks(session_dir):
    """Return {block_num: [(trial_num, features[T, C]), ...]} across all splits."""
    blocks = defaultdict(list)
    for split in ("data_train.hdf5", "data_val.hdf5", "data_test.hdf5"):
        path = os.path.join(session_dir, split)
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            for key in f.keys():
                g = f[key]
                block_num = int(g.attrs["block_num"])
                trial_num = int(g.attrs["trial_num"])
                blocks[block_num].append((trial_num, g["input_features"][:]))
    # order trials within each block by trial_num (chronological)
    for b in blocks:
        blocks[b].sort(key=lambda t: t[0])
    return blocks


def analyze_session(session_dir, atol=1e-2):
    session = os.path.basename(session_dir)
    blocks = gather_blocks(session_dir)
    if not blocks:
        print(f"[{session}] no hdf5 files found, skipping")
        return None

    block_mean_absmax = []   # max |per-feature mean| over the block
    block_std_dev = []       # max |per-feature std - 1| over the block
    trial_mean_absmax = []   # same, but for individual trials (contrast)

    for block_num, trials in blocks.items():
        feats = np.concatenate([x for _, x in trials], axis=0).astype(np.float64)  # [sum_T, C]
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        block_mean_absmax.append(np.max(np.abs(mean)))
        block_std_dev.append(np.max(np.abs(std - 1.0)))
        for _, x in trials:
            trial_mean_absmax.append(np.max(np.abs(x.astype(np.float64).mean(axis=0))))

    block_mean_absmax = np.array(block_mean_absmax)
    block_std_dev = np.array(block_std_dev)
    trial_mean_absmax = np.array(trial_mean_absmax)

    # A block is "z-scored over the whole block" if every feature's mean ~= 0 and std ~= 1.
    is_block_zscored = bool(np.all(block_mean_absmax < atol) and np.all(block_std_dev < atol))

    print(f"\n=== {session} ({len(blocks)} blocks) ===")
    print(f"  block-level  max|mean|:  median={np.median(block_mean_absmax):.4f}  worst={block_mean_absmax.max():.4f}")
    print(f"  block-level  max|std-1|:  median={np.median(block_std_dev):.4f}  worst={block_std_dev.max():.4f}")
    print(f"  trial-level  max|mean|:   median={np.median(trial_mean_absmax):.4f}  (for contrast; should be >> block if block-normed)")
    print(f"  --> whole-block z-score signature (mean~=0 & std~=1 per block, atol={atol}): {is_block_zscored}")
    if is_block_zscored:
        print("      NON-CAUSAL: normalization used full-block statistics -> future leak within each block.")
    else:
        print("      No clean whole-block z-score signature. Normalization is NOT a simple full-block z-score")
        print("      (could be causal/running, an external reference block, or a different transform).")
    return is_block_zscored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data/hdf5_data_final")
    ap.add_argument("--session", default=None, help="single session name, e.g. t15.2023.08.27; omit to scan all")
    ap.add_argument("--atol", type=float, default=1e-2)
    args = ap.parse_args()

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        )

    verdicts = {}
    for s in sessions:
        v = analyze_session(os.path.join(args.data_dir, s), atol=args.atol)
        if v is not None:
            verdicts[s] = v

    if verdicts:
        n_leak = sum(verdicts.values())
        print(f"\n=== SUMMARY: {n_leak}/{len(verdicts)} sessions show a whole-block z-score (future-leak) signature ===")


if __name__ == "__main__":
    main()
