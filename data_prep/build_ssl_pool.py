"""Build a label-free pool for masked (BERT-style) pretraining in the T15 HDF5 layout.

- B2T '25 sessions: train and test trials merged into one data_train.hdf5 per session with labels blanked
  (seq_len 0, empty sentence). Val trials are never copied (val-dev is the evaluation split; former val-test is
  exposed). Test recordings carry no labels and are never evaluated here.
- Extra sessions (e.g. data/hdf5_data_t15v_harm, data/hdf5_data_t12_6v_xvaldev): linked as they are; any labels in
  them are ignored by pretraining.
Session names are kept, so pretrained day layers can be matched to fine-tuning sessions by name.

    .venv/bin/python -m data_prep.build_ssl_pool --b2t-dir data/hdf5_data_final --out-dir data/ssl_pool_v1 \\
        --extra-dirs data/hdf5_data_t15v_harm data/hdf5_data_t12_6v_xvaldev
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

MAX_SEQ_ELEMENTS = 500
BIN_SECONDS = 0.02


def merge_unlabelled(session_dir, out_path, splits=("train", "test")):
    """Copy the features of every trial in the given splits into out_path with blank labels; return counts."""
    out_path.parent.mkdir(parents=True, exist_ok=False)
    counts, bins, index = {}, 0, 0
    with h5py.File(out_path, "x") as out:
        for split in splits:
            path = session_dir / f"data_{split}.hdf5"
            if not path.exists():
                continue
            with h5py.File(path, "r") as src:
                keys = sorted((k for k in src if k.startswith("trial_")), key=lambda k: int(k.split("_")[-1]))
                for key in keys:
                    g = src[key]
                    x = g["input_features"][:]
                    o = out.create_group(f"trial_{index:04d}")
                    o.create_dataset("input_features", data=x, compression="lzf")
                    o.create_dataset("seq_class_ids", data=np.zeros(MAX_SEQ_ELEMENTS, np.int32), compression="lzf")
                    o.create_dataset("transcription", data=np.zeros(MAX_SEQ_ELEMENTS, np.int32), compression="lzf")
                    o.attrs["n_time_steps"] = np.int64(x.shape[0])
                    o.attrs["seq_len"] = np.int64(0)
                    o.attrs["block_num"] = np.int64(g.attrs["block_num"])
                    o.attrs["trial_num"] = np.int64(g.attrs["trial_num"])
                    o.attrs["session"] = session_dir.name
                    o.attrs["sentence_label"] = ""
                    o.attrs["source_split"] = split
                    index += 1
                    bins += x.shape[0]
                counts[split] = len(keys)
    return counts, bins


def count_bins(path):
    with h5py.File(path, "r") as h:
        return len(h), sum(int(h[k].attrs["n_time_steps"]) for k in h)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--b2t-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="new directory (must not exist)")
    parser.add_argument("--extra-dirs", nargs="*", default=[], help="directories of sessions to link as they are")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a fresh output directory")
    out.mkdir(parents=True)
    sources, total_bins = [], 0
    for session_dir in sorted(p for p in Path(args.b2t_dir).iterdir() if p.is_dir() and p.name.startswith("t15.")):
        counts, bins = merge_unlabelled(session_dir, out / session_dir.name / "data_train.hdf5")
        sources.append(dict(session=session_dir.name, source="b2t train+test (labels blanked)", trials=counts,
                            hours=round(bins * BIN_SECONDS / 3600, 3)))
        total_bins += bins
    for extra in args.extra_dirs:
        for session_dir in sorted(p for p in Path(extra).iterdir() if p.is_dir()):
            if (out / session_dir.name).exists():
                raise SystemExit(f"duplicate session {session_dir.name} from {extra}")
            os.symlink(os.path.relpath(session_dir.resolve(), out.resolve()), out / session_dir.name)
            n, bins = count_bins(session_dir / "data_train.hdf5")
            sources.append(dict(session=session_dir.name, source=str(extra), trials=dict(train=n),
                                hours=round(bins * BIN_SECONDS / 3600, 3)))
            total_bins += bins
    summary = dict(b2t_dir=args.b2t_dir, extra_dirs=args.extra_dirs, n_sessions=len(sources),
                   n_trials=sum(sum(s["trials"].values()) for s in sources), hours=round(total_bins * BIN_SECONDS / 3600, 2),
                   sessions=sources)
    (out / "pool_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{summary['n_sessions']} sessions, {summary['n_trials']} trials, {summary['hours']} h -> {out}")
    return summary


if __name__ == "__main__":
    main()
