"""R3 — Phase-staggered patch emission for sub-cadence resolution.

With patch_size P and stride s, the default pipeline emits every s bins
(L_buf = s * 20 ms). Evaluating the same causal model at K phase offsets
{0, 1, ..., K-1} with K=s yields a collective emission grid of 1 bin
(L_buf = 20 ms) while acting as a free ensemble over phases.

Important: day layers and the causal smoother are phase-agnostic. The patch
embedding (flatten of P consecutive bins) is phase-sensitive; random phase
augmentation at train time improves multi-phase eval. Without that training,
eval-only phase stagger may hurt — measure before shipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"
MODEL_TRAINING_DIR = Path(__file__).resolve().parent.parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
if str(MODEL_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_DIR))

from common import smooth_features  # type: ignore  # noqa: E402
try:
    from .ensemble import average_logits
except ImportError:  # script-style execution
    from ensemble import average_logits  # type: ignore  # noqa: E402


@dataclass(frozen=True)
class PhaseStaggerConfig:
    """K phase offsets in bins. Default K=patch_stride uses every offset."""

    n_phases: Optional[int] = None  # None -> use model patch_stride
    merge_mode: str = "prob_mean"  # same modes as ensemble.average_logits
    # How to align shorter phase streams onto the finest grid:
    # "hold" repeats the last emitted frame; "interp" is not implemented yet.
    align: str = "hold"


def phase_stagger_logits(
    model: GRUDecoder,
    raw_features: torch.Tensor,
    day_idx: int,
    args: Any,
    device: torch.device,
    config: Optional[PhaseStaggerConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Evaluate model at multiple patch phases and merge.

    raw_features: [T, F] or [1, T, F]
    Returns (merged_logits [1, T_out, C], meta dict).

    Implementation: for phase offset o, left-pad the *smoothed* sequence with
    o zero bins, run the normal model forward (which unfolds patches with
    stride s starting at 0 of the padded sequence), then drop the first
    ceil(o/s) frames that only see pad so outputs align near the true right edge.
    Finally merge on the shortest common length via average_logits.
    """
    config = config or PhaseStaggerConfig()
    if raw_features.dim() == 2:
        raw_features = raw_features.unsqueeze(0)
    raw_features = raw_features.to(device=device, dtype=torch.float32)

    patch_size = int(args["model"]["patch_size"])
    patch_stride = int(args["model"]["patch_stride"])
    if patch_size <= 0 or patch_stride <= 0:
        raise ValueError("phase stagger requires patch_size > 0 and patch_stride > 0")

    n_phases = config.n_phases if config.n_phases is not None else patch_stride
    if n_phases < 1:
        raise ValueError("n_phases must be >= 1")
    if n_phases > patch_stride:
        # Offsets beyond stride are redundant modulo stride for regular grids.
        n_phases = patch_stride

    day_tensor = torch.tensor([day_idx], device=device, dtype=torch.long)
    member_logits: List[torch.Tensor] = []
    n_frames_per_phase: List[int] = []

    with torch.inference_mode():
        smoothed = smooth_features(raw_features, args, device)  # [1, T, F]
        t_len = int(smoothed.shape[1])

        for phase in range(n_phases):
            if phase == 0:
                padded = smoothed
            else:
                pad = torch.zeros(
                    1,
                    phase,
                    smoothed.shape[2],
                    device=device,
                    dtype=smoothed.dtype,
                )
                padded = torch.cat([pad, smoothed], dim=1)

            logits = model(padded, day_tensor)  # [1, n_patches, C]
            # Drop patches whose right edge is still inside the left pad region.
            # Patch i covers bins [i*stride, i*stride + patch_size) of padded.
            # Require right edge > phase so the patch sees at least one real bin
            # at the end: i*stride + patch_size - 1 >= phase
            # => i >= ceil((phase - patch_size + 1) / stride) — usually 0 for
            # phase < patch_size. Keep all patches whose right edge maps into
            # original time: right_edge - phase < t_len
            keep = []
            n_patches = int(logits.shape[1])
            for i in range(n_patches):
                right_edge_padded = i * patch_stride + patch_size - 1
                right_edge_orig = right_edge_padded - phase
                left_edge_padded = i * patch_stride
                # Keep if patch ends on a real (non-pad) index and fully defined.
                if right_edge_orig >= 0 and right_edge_orig < t_len and left_edge_padded + patch_size - 1 >= phase:
                    keep.append(i)
            if not keep:
                continue
            cropped = logits[:, keep, :]
            member_logits.append(cropped[0])
            n_frames_per_phase.append(int(cropped.shape[1]))

    if not member_logits:
        raise RuntimeError("phase stagger produced no frames")

    min_len = min(t.shape[0] for t in member_logits)
    trimmed = [t[:min_len] for t in member_logits]
    if n_phases == 1:
        merged = trimmed[0]
    else:
        merged = average_logits(trimmed, mode=config.merge_mode)

    # Effective buffering: with K=stride phases, collective cadence is 1 bin.
    bin_ms = 20.0
    l_buf_baseline_ms = patch_stride * bin_ms
    l_buf_effective_ms = bin_ms if n_phases >= patch_stride else (patch_stride / n_phases) * bin_ms

    meta = {
        "n_phases": n_phases,
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "merge_mode": config.merge_mode,
        "n_frames_per_phase": n_frames_per_phase,
        "merged_frames": min_len,
        "l_buf_baseline_ms": l_buf_baseline_ms,
        "l_buf_effective_ms": l_buf_effective_ms,
        "l_algo_ms": 0.0,
    }
    return merged.unsqueeze(0), meta
