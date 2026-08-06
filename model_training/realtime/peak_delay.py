"""S6 — Measure CTC peak delay relative to ground-truth phoneme timing.

Gate for emission-latency regularizers (FastEmit, delay-penalized CTC, etc.).
If median peak delay is already small (≤40 ms), those runs should be killed.

Phoneme midpoints are estimated by linear force-alignment: distribute phoneme
labels evenly across non-blank-ish time, or use equal duration over T frames
when no forced aligner is available. This is a defensible order-of-magnitude
measurement, not a gold forced-alignment — flag as estimated in reports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

BLANK_CLASS = 0
BIN_MS = 20.0  # feature bin
# After patching with stride s, one logit frame ≈ s bins.
DEFAULT_PATCH_STRIDE = 4


def _ctc_peak_positions(
    logits: torch.Tensor,
    blank_index: int = BLANK_CLASS,
) -> List[Tuple[int, int, float]]:
    """Return list of (frame_idx, class_id, confidence) for non-blank peaks.

    Peak = frame where argmax is non-blank and either first frame or differs
    from previous argmax (CTC collapse boundary).
    """
    if logits.dim() == 3:
        logits = logits[0]
    probs = F.softmax(logits.float(), dim=-1)
    argmax = torch.argmax(probs, dim=-1).cpu().numpy()
    peaks: List[Tuple[int, int, float]] = []
    prev = None
    for t, c in enumerate(argmax):
        c = int(c)
        if c == blank_index:
            prev = c
            continue
        if prev is None or c != prev:
            conf = float(probs[t, c].item())
            peaks.append((t, c, conf))
        prev = c
    return peaks


def _linear_phoneme_midpoints(
    phone_ids: Sequence[int],
    n_frames: int,
    blank_index: int = BLANK_CLASS,
) -> List[Tuple[int, float]]:
    """Equal-duration placement of non-blank phonemes over [0, n_frames).

    Returns list of (phone_id, midpoint_frame).
    seq_class_ids are phoneme labels (1..40); 0 is padding / blank — strip it.
    """
    phones = [int(p) for p in phone_ids if int(p) != blank_index]
    if not phones or n_frames <= 0:
        return []
    slot = n_frames / len(phones)
    return [(phones[i], (i + 0.5) * slot) for i in range(len(phones))]


def measure_peak_delays(
    logits: torch.Tensor,
    phone_ids: Sequence[int],
    patch_stride: int = DEFAULT_PATCH_STRIDE,
    bin_ms: float = BIN_MS,
    blank_index: int = BLANK_CLASS,
) -> Dict[str, Any]:
    """Estimate peak delay distribution for one trial.

    Delay_ms = (peak_frame - phone_mid_frame) * patch_stride * bin_ms
    Matched greedily by phoneme class identity in order.
    """
    if logits.dim() == 3:
        logits = logits[0]
    n_frames = int(logits.shape[0])
    peaks = _ctc_peak_positions(logits, blank_index=blank_index)
    mids = _linear_phoneme_midpoints(phone_ids, n_frames, blank_index=blank_index)

    delays_ms: List[float] = []
    matches: List[Dict[str, Any]] = []
    pi = 0
    for frame_idx, class_id, conf in peaks:
        # advance mid pointer to matching class if possible
        matched = False
        for j in range(pi, len(mids)):
            mid_id, mid_frame = mids[j]
            if mid_id == class_id:
                delay_frames = frame_idx - mid_frame
                delay_ms = float(delay_frames * patch_stride * bin_ms)
                delays_ms.append(delay_ms)
                matches.append(
                    {
                        "peak_frame": frame_idx,
                        "mid_frame": mid_frame,
                        "class_id": class_id,
                        "confidence": conf,
                        "delay_ms": delay_ms,
                    }
                )
                pi = j + 1
                matched = True
                break
        if not matched:
            # unmatched peak — skip
            continue

    frame_ms = patch_stride * bin_ms
    return {
        "n_frames": n_frames,
        "n_peaks": len(peaks),
        "n_targets": len(mids),
        "n_matched": len(matches),
        "delays_ms": delays_ms,
        "matches": matches,
        "frame_ms": frame_ms,
        "alignment": "linear_equal_duration_estimated",
    }


def summarize_peak_delays(trial_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate across trials; apply pre-registered S6 kill/go thresholds."""
    all_delays: List[float] = []
    total_matched = 0
    total_peaks = 0
    for trial in trial_results:
        all_delays.extend(trial.get("delays_ms", []))
        total_matched += int(trial.get("n_matched", 0))
        total_peaks += int(trial.get("n_peaks", 0))

    if not all_delays:
        return {
            "n_trials": len(trial_results),
            "n_delays": 0,
            "median_ms": float("nan"),
            "mean_ms": float("nan"),
            "p90_ms": float("nan"),
            "decision": "INCONCLUSIVE",
            "rule": "no matched peaks — check labels / logits",
        }

    arr = np.asarray(all_delays, dtype=np.float64)
    median = float(np.median(arr))
    mean = float(np.mean(arr))
    p90 = float(np.percentile(arr, 90))

    # Pre-registered from grok_recs R3/S6
    if median <= 40.0:
        decision = "KILL_EMISSION_REGULARIZERS"
        rule = "median peak delay <= 40 ms → FastEmit / delay-CTC / Peak-First / TrimTail killed"
    elif median >= 80.0:
        decision = "AUTHORIZE_DELAY_CTC"
        rule = "median peak delay >= 80 ms → authorize 1-seed delay-penalized CTC"
    else:
        decision = "BORDERLINE"
        rule = "40 < median < 80 ms → optional single probe run only"

    return {
        "n_trials": len(trial_results),
        "n_delays": int(arr.size),
        "total_matched": total_matched,
        "total_peaks": total_peaks,
        "median_ms": median,
        "mean_ms": mean,
        "p90_ms": p90,
        "p10_ms": float(np.percentile(arr, 10)),
        "decision": decision,
        "rule": rule,
        "alignment_note": "linear equal-duration midpoints — estimated, not gold forced-align",
    }
