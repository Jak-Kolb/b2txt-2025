"""R2 — Streaming decode helpers: blank-frame skip + adaptive rescoring gates.

These operate on acoustic-model logits (and optional hypothesis strings).
They intentionally do not depend on Redis/LLM wiring so they can be unit-tested
and used as a gate in front of any external rescoring backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

BLANK_CLASS = 0


def blank_skip_mask(
    logits: torch.Tensor,
    blank_threshold: float = 0.7,
    blank_index: int = BLANK_CLASS,
) -> torch.Tensor:
    """Boolean mask [T] — True where frame should be skipped (high blank mass).

    logits: [T, C] or [1, T, C]
    """
    if logits.dim() == 3:
        logits = logits[0]
    probs = F.softmax(logits.float(), dim=-1)
    blank_p = probs[:, blank_index]
    return blank_p >= blank_threshold


def filter_frames_by_blank_skip(
    logits: torch.Tensor,
    blank_threshold: float = 0.7,
    blank_index: int = BLANK_CLASS,
    min_keep: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop high-blank frames for cheaper beam expansion.

    Returns (kept_logits [T', C], keep_indices [T']).
    Always keeps at least min_keep frames (first/last preference) so CTC
    never sees an empty sequence.
    """
    if logits.dim() == 3:
        logits = logits[0]
    mask_skip = blank_skip_mask(logits, blank_threshold, blank_index)
    keep = ~mask_skip
    indices = torch.nonzero(keep, as_tuple=False).flatten()
    if indices.numel() < min_keep:
        # Fall back to lowest-blank frames
        probs = F.softmax(logits.float(), dim=-1)
        blank_p = probs[:, blank_index]
        _, order = torch.sort(blank_p)
        indices = order[: max(min_keep, 1)].sort().values
    return logits[indices], indices


def frame_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-frame Shannon entropy in nats. logits [T, C] or [1, T, C] -> [T]."""
    if logits.dim() == 3:
        logits = logits[0]
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def hypothesis_entropy_proxy(logits: torch.Tensor) -> float:
    """Scalar uncertainty proxy: mean frame entropy over non-blank-dominated frames."""
    if logits.dim() == 3:
        logits = logits[0]
    if logits.numel() == 0:
        return 0.0
    ent = frame_entropy(logits)
    skip = blank_skip_mask(logits, blank_threshold=0.7)
    kept = ent[~skip]
    if kept.numel() == 0:
        kept = ent
    return float(kept.mean().item())


@dataclass
class AdaptiveRescoreConfig:
    """Entropy-triggered rescoring policy (S5 / R2).

    When partial-hypothesis entropy exceeds `entropy_threshold`, call the
    expensive rescorer. Expected latency = f_trigger * cost_resc + always_on_cost.
    """

    entropy_threshold: float = 1.5  # nats; sweep this
    min_partial_frames: int = 5
    cooldown_frames: int = 4  # min frames between rescoring fires
    always_rescore_final: bool = True
    blank_threshold: float = 0.7


@dataclass
class StreamingBeamState:
    """Minimal streaming hypothesis state for partial emission + stability."""

    partial_tokens: List[int] = field(default_factory=list)
    partial_text: str = ""
    frozen_prefix: str = ""
    last_rescore_frame: int = -10**9
    rescore_count: int = 0
    skip_count: int = 0
    expand_count: int = 0
    entropy_history: List[float] = field(default_factory=list)
    partial_history: List[str] = field(default_factory=list)
    emission_times_ms: List[float] = field(default_factory=list)


def should_trigger_rescore(
    state: StreamingBeamState,
    frame_idx: int,
    entropy: float,
    config: AdaptiveRescoreConfig,
    is_final: bool = False,
) -> bool:
    if is_final and config.always_rescore_final:
        return True
    if frame_idx < config.min_partial_frames:
        return False
    if frame_idx - state.last_rescore_frame < config.cooldown_frames:
        return False
    return entropy >= config.entropy_threshold


def greedy_ctc_with_blank_skip(
    logits: torch.Tensor,
    blank_threshold: float = 0.7,
    blank_index: int = BLANK_CLASS,
    use_skip: bool = True,
) -> np.ndarray:
    """Greedy CTC collapse, optionally after dropping high-blank frames."""
    if logits.dim() == 3:
        logits = logits[0]
    if use_skip:
        logits, _ = filter_frames_by_blank_skip(
            logits, blank_threshold=blank_threshold, blank_index=blank_index
        )
    decoded = torch.argmax(logits.detach(), dim=-1)
    decoded = torch.unique_consecutive(decoded, dim=-1)
    return np.array([int(t) for t in decoded.cpu().numpy() if int(t) != blank_index], dtype=np.int64)


def stream_greedy_partials(
    logits: torch.Tensor,
    bin_ms: float = 20.0,
    patch_stride: int = 4,
    blank_index: int = BLANK_CLASS,
    token_to_str: Optional[Callable[[int], str]] = None,
) -> StreamingBeamState:
    """Produce cumulative partial hypotheses frame-by-frame (greedy CTC).

    Emission time for frame i is approximated as (i+1) * patch_stride * bin_ms
    (right-edge emission under patching).
    """
    if logits.dim() == 3:
        logits = logits[0]
    state = StreamingBeamState()
    prev: Optional[int] = None
    tokens: List[int] = []
    for i in range(int(logits.shape[0])):
        argmax = int(torch.argmax(logits[i], dim=-1).item())
        if argmax != prev and argmax != blank_index:
            tokens.append(argmax)
        prev = argmax
        state.partial_tokens = list(tokens)
        if token_to_str is not None:
            state.partial_text = " ".join(token_to_str(t) for t in tokens)
        else:
            state.partial_text = " ".join(str(t) for t in tokens)
        state.partial_history.append(state.partial_text)
        state.emission_times_ms.append((i + 1) * patch_stride * bin_ms)
        state.entropy_history.append(float(frame_entropy(logits[i : i + 1]).item()))
        state.expand_count += 1
    return state


def run_adaptive_policy_over_logits(
    logits: torch.Tensor,
    config: Optional[AdaptiveRescoreConfig] = None,
    rescorer: Optional[Callable[[str, torch.Tensor], str]] = None,
    patch_stride: int = 4,
    bin_ms: float = 20.0,
    token_to_str: Optional[Callable[[int], str]] = None,
) -> Dict[str, Any]:
    """Walk frames, apply blank-skip accounting, fire rescorer on entropy gate.

    If `rescorer` is None, triggers are counted but text is unchanged (dry-run).
    """
    config = config or AdaptiveRescoreConfig()
    if logits.dim() == 3:
        logits = logits[0]

    state = stream_greedy_partials(
        logits,
        bin_ms=bin_ms,
        patch_stride=patch_stride,
        token_to_str=token_to_str,
    )
    skip = blank_skip_mask(logits, blank_threshold=config.blank_threshold)
    state.skip_count = int(skip.sum().item())

    trigger_frames: List[int] = []
    rescored_text = state.partial_text
    for i, ent in enumerate(state.entropy_history):
        is_final = i == len(state.entropy_history) - 1
        if should_trigger_rescore(state, i, ent, config, is_final=is_final):
            trigger_frames.append(i)
            state.last_rescore_frame = i
            state.rescore_count += 1
            if rescorer is not None:
                rescored_text = rescorer(state.partial_history[i], logits[: i + 1])
                state.partial_history[i] = rescored_text

    n_frames = max(len(state.entropy_history), 1)
    return {
        "state": state,
        "trigger_frames": trigger_frames,
        "trigger_rate": state.rescore_count / n_frames,
        "blank_skip_rate": state.skip_count / int(logits.shape[0]) if logits.shape[0] else 0.0,
        "mean_entropy": float(np.mean(state.entropy_history)) if state.entropy_history else 0.0,
        "final_text": rescored_text,
        "config": {
            "entropy_threshold": config.entropy_threshold,
            "blank_threshold": config.blank_threshold,
            "cooldown_frames": config.cooldown_frames,
            "always_rescore_final": config.always_rescore_final,
        },
    }
