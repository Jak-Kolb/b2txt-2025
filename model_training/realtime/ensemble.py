"""R1 — Causal deep ensemble with pre-beam logit averaging.

All members must be causal (smooth_lookahead=0) and share the same day index
space / n_classes. Logits are averaged in probability space by default
(mean of softmax, then log), which is numerically safer than raw-logit mean
when members have different scales; log-prob mean is available as an option.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"
MODEL_TRAINING_DIR = Path(__file__).resolve().parent.parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
if str(MODEL_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_DIR))

from common import (  # type: ignore  # noqa: E402
    load_args,
    load_model,
    offline_logits,
    resolve_path,
    smooth_features,
)
from rnn_model import GRUDecoder  # type: ignore  # noqa: E402


def average_logits(
    logit_list: Sequence[torch.Tensor],
    mode: str = "prob_mean",
) -> torch.Tensor:
    """Average a list of logit tensors with matching shapes.

    mode:
      - "prob_mean": mean of softmax probabilities, then log (default)
      - "logit_mean": arithmetic mean of logits
      - "logprob_mean": mean of log_softmax, then renorm via logsumexp
    """
    if not logit_list:
        raise ValueError("logit_list is empty")
    stacked = torch.stack([t.float() for t in logit_list], dim=0)
    if mode == "logit_mean":
        return stacked.mean(dim=0)
    if mode == "prob_mean":
        probs = F.softmax(stacked, dim=-1).mean(dim=0)
        return torch.log(probs.clamp_min(1e-12))
    if mode == "logprob_mean":
        log_probs = F.log_softmax(stacked, dim=-1).mean(dim=0)
        # renorm so rows sum to 1 in prob space
        return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    raise ValueError(f"Unknown average mode: {mode}")


def load_ensemble(
    checkpoint_dirs: Sequence[Union[str, Path]],
    device: torch.device,
) -> Tuple[List[GRUDecoder], List[Any], List[Path]]:
    """Load N models. Returns (models, args_list, resolved_dirs)."""
    models: List[GRUDecoder] = []
    args_list: List[Any] = []
    resolved: List[Path] = []
    for raw in checkpoint_dirs:
        path = resolve_path(raw)
        model, args, _ckpt = load_model(path, device)
        models.append(model)
        args_list.append(args)
        resolved.append(path)
    _validate_ensemble_compatible(args_list)
    return models, args_list, resolved


def _validate_ensemble_compatible(args_list: Sequence[Any]) -> None:
    if not args_list:
        raise ValueError("No ensemble members")
    ref = args_list[0]
    for idx, args in enumerate(args_list[1:], start=1):
        if int(args["dataset"]["n_classes"]) != int(ref["dataset"]["n_classes"]):
            raise ValueError(f"Member {idx}: n_classes mismatch")
        if int(args["model"]["n_input_features"]) != int(ref["model"]["n_input_features"]):
            raise ValueError(f"Member {idx}: n_input_features mismatch")
        if int(args["model"]["patch_size"]) != int(ref["model"]["patch_size"]):
            raise ValueError(f"Member {idx}: patch_size mismatch")
        if int(args["model"]["patch_stride"]) != int(ref["model"]["patch_stride"]):
            raise ValueError(f"Member {idx}: patch_stride mismatch")


class EnsembleDecoder:
    """Offline + streaming-compatible multi-member acoustic ensemble."""

    def __init__(
        self,
        models: Sequence[GRUDecoder],
        args_list: Sequence[Any],
        device: torch.device,
        average_mode: str = "prob_mean",
    ):
        if len(models) != len(args_list):
            raise ValueError("models and args_list length mismatch")
        if not models:
            raise ValueError("empty ensemble")
        self.models = list(models)
        self.args_list = list(args_list)
        self.device = device
        self.average_mode = average_mode
        # Reference args for smoothing / patching: use first member.
        self.args = args_list[0]
        self.n_members = len(models)

    @classmethod
    def from_checkpoint_dirs(
        cls,
        checkpoint_dirs: Sequence[Union[str, Path]],
        device: torch.device,
        average_mode: str = "prob_mean",
    ) -> "EnsembleDecoder":
        models, args_list, _ = load_ensemble(checkpoint_dirs, device)
        return cls(models, args_list, device, average_mode=average_mode)

    def forward_offline(
        self,
        raw_features: torch.Tensor,
        day_idx: int,
    ) -> torch.Tensor:
        """raw_features: [T, F] or [1, T, F]. Returns ensemble logits [1, T', C]."""
        if raw_features.dim() == 2:
            raw_features = raw_features.unsqueeze(0)
        member_logits: List[torch.Tensor] = []
        with torch.inference_mode():
            for model, args in zip(self.models, self.args_list):
                logits = offline_logits(model, raw_features, day_idx, args, self.device)
                member_logits.append(logits[0])
            averaged = average_logits(member_logits, mode=self.average_mode)
        return averaged.unsqueeze(0)

    def forward_members_offline(
        self,
        raw_features: torch.Tensor,
        day_idx: int,
    ) -> List[torch.Tensor]:
        if raw_features.dim() == 2:
            raw_features = raw_features.unsqueeze(0)
        out: List[torch.Tensor] = []
        with torch.inference_mode():
            for model, args in zip(self.models, self.args_list):
                logits = offline_logits(model, raw_features, day_idx, args, self.device)
                out.append(logits[0])
        return out

    def info(self) -> Dict[str, Any]:
        return {
            "n_members": self.n_members,
            "average_mode": self.average_mode,
            "patch_size": int(self.args["model"]["patch_size"]),
            "patch_stride": int(self.args["model"]["patch_stride"]),
            "n_classes": int(self.args["dataset"]["n_classes"]),
            "smooth_lookahead": int(
                self.args["dataset"]["data_transforms"].get("smooth_lookahead", -1)
            ),
        }
