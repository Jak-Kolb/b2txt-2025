"""C24 — average the last k validation checkpoints, and evaluate any checkpoint through the
EXACT path that produced the numbers we compare against.

Why this reuses the trainer instead of `benchmark/common.offline_logits`
------------------------------------------------------------------------
The WER figures this is compared against (la0 8.49 %, G2 9.10 %) come from `val_metrics.pkl`,
written by `BrainToTextDecoder_Trainer.validation()`, which runs the forward pass under
**bfloat16 autocast** (`rnn_trainer.py:730`) when `use_amp` is true. `benchmark/common` runs fp32
with TF32 disabled. Those are different numerics, so generating logits through the benchmark path
would fold a precision difference into the C24 delta and there would be no way to separate them.

So: instantiate the trainer from the run's own saved `args.yaml`, swap the weights, and call its
`validation()`. Downstream (`stream_lm.py export` -> decode) is then byte-identical to how the
baselines were produced, and `--checkpoint best_checkpoint` is the control that proves it.

    # control: reproduce G2's 9.10 % through this path
    cd model_training && ../.venv/bin/python average_checkpoints.py \
        --checkpoint_dir trained_models/g2_masking/checkpoint \
        --checkpoint best_checkpoint --out ../results/c24_g2_best.pkl

    # the k=10 average
    cd model_training && ../.venv/bin/python average_checkpoints.py \
        --checkpoint_dir trained_models/g2_masking/checkpoint \
        --k 10 --save_avg ../results/c24_g2_avg10.pt --out ../results/c24_g2_avg10.pkl
"""
import argparse
import os
import pathlib
import pickle
import re
import sys

import torch
from omegaconf import OmegaConf

REPO = pathlib.Path(__file__).resolve().parent.parent


def last_k_checkpoints(checkpoint_dir, k):
    """The k highest-batch `checkpoint_batch_N` files, oldest first."""
    found = []
    for p in pathlib.Path(checkpoint_dir).iterdir():
        m = re.fullmatch(r"checkpoint_batch_(\d+)", p.name)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        raise FileNotFoundError(f"no checkpoint_batch_* under {checkpoint_dir}")
    found.sort()
    if len(found) < k:
        raise ValueError(f"asked for {k} checkpoints but only {len(found)} exist")
    return found[-k:]


def average_state_dicts(paths):
    """Uniform mean of the float tensors; non-float entries come from the last checkpoint.

    Averaging integer state (step counters, and any future int buffer) is meaningless, and
    silently truncating it to an int mean is the kind of thing that produces a model that loads
    fine and is subtly wrong.
    """
    acc, n_float, n_copied = None, 0, 0
    for path in paths:
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        if acc is None:
            acc = {k: (v.double().clone() if v.is_floating_point() else v.clone())
                   for k, v in sd.items()}
            n_float = sum(1 for v in sd.values() if v.is_floating_point())
            n_copied = len(sd) - n_float
            continue
        if sd.keys() != acc.keys():
            raise ValueError(f"state_dict keys differ at {path}")
        for k, v in sd.items():
            if v.is_floating_point():
                acc[k] += v.double()
            else:
                acc[k] = v.clone()          # last one wins
    out = {}
    for k, v in acc.items():
        out[k] = (v / len(paths)) if v.is_floating_point() else v
    print(f"  averaged {n_float} float tensors over {len(paths)} checkpoints "
          f"({n_copied} non-float taken from the last)")
    return out


def build_eval_trainer(args_path, scratch_dir):
    """A trainer configured to do nothing but validate — no dirs created, no run artifacts touched.

    `mode != 'train'` skips the output_dir makedirs; turning the three save flags off skips the
    checkpoint_dir makedirs (`rnn_trainer.py:62-64`), which would otherwise crash on the existing
    directory. `output_dir` is redirected to scratch because `:176` writes `train_val_trials.json`
    into it unconditionally and we must not mutate a finished run.
    """
    a = OmegaConf.load(args_path)
    a.mode = "eval"
    a.save_best_checkpoint = False
    a.save_all_val_steps = False
    a.save_final_model = False
    a.num_training_batches = 1          # the train dataset is built but never iterated
    a.output_dir = str(scratch_dir)
    a.checkpoint_dir = str(scratch_dir / "checkpoint")
    os.makedirs(scratch_dir, exist_ok=True)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from rnn_trainer import BrainToTextDecoder_Trainer
    return BrainToTextDecoder_Trainer(OmegaConf.to_container(a, resolve=True)), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--k", type=int, default=10, help="how many trailing checkpoints to average")
    ap.add_argument("--checkpoint", default="",
                    help="evaluate this single checkpoint instead of averaging (the control)")
    ap.add_argument("--save_avg", default="", help="also write the averaged checkpoint here")
    ap.add_argument("--scratch", default="/tmp/c24_eval")
    ap.add_argument("--out", required=True, help="val_metrics.pkl to write")
    a = ap.parse_args()

    ckpt_dir = pathlib.Path(a.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (pathlib.Path.cwd() / ckpt_dir)

    if a.checkpoint:
        src = ckpt_dir / a.checkpoint
        print(f"control: evaluating {src}")
        state = torch.load(src, map_location="cpu", weights_only=False)["model_state_dict"]
    else:
        picked = last_k_checkpoints(ckpt_dir, a.k)
        print(f"averaging the last {a.k} checkpoints: "
              f"batch {picked[0][0]} .. {picked[-1][0]}")
        state = average_state_dicts([p for _, p in picked])
        if a.save_avg:
            torch.save({"model_state_dict": state}, a.save_avg)
            print(f"  wrote {a.save_avg}")

    trainer, cfg = build_eval_trainer(ckpt_dir / "args.yaml", pathlib.Path(a.scratch))
    missing, unexpected = trainer.model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # The checkpoints were saved from a torch.compile'd model, so keys carry `_orig_mod.`.
        # The trainer compiles too, so they should line up exactly; anything else is a real bug.
        raise RuntimeError(f"state_dict mismatch: missing={list(missing)[:5]} "
                           f"unexpected={list(unexpected)[:5]}")

    print("running validation (bf16 autocast, same as training) ...")
    metrics = trainer.validation(trainer.val_loader, return_logits=True, return_data=False)
    print(f"  avg_PER {metrics['avg_PER']:.5f}   avg_loss {metrics['avg_loss']:.4f}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(metrics, fh)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
