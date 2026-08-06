#!/usr/bin/env python3
"""Launch one causal training seed for the R1 ensemble (isolates seed only).

Copies a base YAML (default: causal_la0 args), sets seed + output paths, trains.

Usage (from repo root, GPU box):
  python model_training/realtime/launch_seed_train.py --seed 0
  python model_training/realtime/launch_seed_train.py --seed 1 --base results/trained_models/causal_la0/checkpoint/args.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
MODEL_TRAINING = REPO / "model_training"
sys.path.insert(0, str(MODEL_TRAINING))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--base",
        default=str(REPO / "results/trained_models/causal_la0/checkpoint/args.yaml"),
        help="Base args.yaml with smooth_lookahead=0.",
    )
    p.add_argument(
        "--out_root",
        default=str(REPO / "results/trained_models/ensemble_causal_la0"),
        help="Root directory for seed_* outputs.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Write config only; do not train.",
    )
    ns = p.parse_args()

    base_path = Path(ns.base)
    if not base_path.is_absolute():
        base_path = (REPO / base_path).resolve()
    args = OmegaConf.load(base_path)

    out_dir = Path(ns.out_root)
    if not out_dir.is_absolute():
        out_dir = (REPO / out_dir).resolve()
    seed_dir = out_dir / f"seed_{ns.seed}"
    ckpt_dir = seed_dir / "checkpoint"

    # Isolates seed only — do not change architecture or lookahead.
    args.seed = int(ns.seed)
    if "dataset" in args and "seed" in args.dataset:
        # Keep dataset split seed fixed so all ensemble members see same train/val split.
        # Only the model init / dropout RNG uses args.seed.
        pass
    args.output_dir = str(seed_dir)
    args.checkpoint_dir = str(ckpt_dir)
    args.mode = "train"
    # Guard: must remain causal
    la = args.dataset.data_transforms.get("smooth_lookahead", None)
    if la is None or int(la) != 0:
        print(
            f"WARNING: base config smooth_lookahead={la}; R1 requires fully causal L=0",
            file=sys.stderr,
        )

    seed_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = seed_dir / "args_train.yaml"
    OmegaConf.save(args, cfg_path)
    print(f"Wrote {cfg_path}")
    print(f"  seed={args.seed} output_dir={args.output_dir}")
    print(f"  smooth_lookahead={args.dataset.data_transforms.smooth_lookahead}")

    if ns.dry_run:
        return 0

    from rnn_trainer import BrainToTextDecoder_Trainer

    trainer = BrainToTextDecoder_Trainer(args)
    metrics = trainer.train()
    print(f"Seed {ns.seed} done. metrics keys: {list(metrics.keys()) if metrics else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
