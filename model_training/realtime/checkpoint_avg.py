"""Eval-only checkpoint averaging / SWA-style weight merge (run ledger #2).

Averages state_dict floating tensors across multiple checkpoint files.
Does not retrain. Useful for last-k epoch averaging when those checkpoints
were saved; also works for averaging independent seeds into a single weight
file (distinct from logit-space ensemble).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import torch


def average_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("no state dicts")
    keys = state_dicts[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        vals = [sd[key] for sd in state_dicts]
        if not torch.is_floating_point(vals[0]):
            out[key] = vals[0].clone()
            continue
        acc = vals[0].float().clone()
        for v in vals[1:]:
            acc += v.float()
        acc /= len(vals)
        out[key] = acc.to(dtype=vals[0].dtype)
    return out


def _clean_key(key: str) -> str:
    return key.replace("module.", "").replace("_orig_mod.", "")


def load_checkpoint_state(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        # raw state dict
        sd = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format: {path}")
    return {_clean_key(k): v for k, v in sd.items()}


def average_checkpoints(
    checkpoint_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    template_checkpoint: Union[str, Path, None] = None,
) -> Path:
    """Average weights and write a new checkpoint file loadable by the trainer."""
    sds = [load_checkpoint_state(p) for p in checkpoint_paths]
    averaged = average_state_dicts(sds)

    template: Dict[str, Any] = {}
    if template_checkpoint is not None:
        template = torch.load(template_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(template, dict):
            template = {}
    elif checkpoint_paths:
        template = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
        if not isinstance(template, dict):
            template = {}

    out = dict(template) if isinstance(template, dict) else {}
    out["model_state_dict"] = averaged
    out["averaged_from"] = [str(p) for p in checkpoint_paths]
    out["n_averaged"] = len(checkpoint_paths)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Average model checkpoints (eval-only).")
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="Paths to best_checkpoint (or any torch checkpoint with model_state_dict).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for averaged checkpoint.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Optional template checkpoint to copy non-weight metadata from.",
    )
    args = parser.parse_args()
    out = average_checkpoints(args.checkpoints, args.out, template_checkpoint=args.template)
    print(f"Wrote averaged checkpoint ({len(args.checkpoints)} members) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
