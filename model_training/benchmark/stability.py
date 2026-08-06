"""Streaming stability metrics for a partial-hypothesis decoder.

WER measures the final transcript. A streaming user reads the *partial* transcript, so
the quantity they actually experience is how much it flickers before settling. No
brain-to-text paper currently reports this.

Two levels, because they behave completely differently:

ACOUSTIC LEVEL (greedy CTC, no LM)
    The online collapse rule in common.update_online_ctc only ever appends, so the
    partial hypothesis is monotone in the prefix order: hyp_t is always a prefix of
    hyp_{t+1}. Revisions are therefore ZERO BY CONSTRUCTION, and time-to-final equals
    time-to-first-emission. verify_append_only() checks this empirically rather than
    trusting the argument.

LM LEVEL (WFST beam search, partial best path)
    CtcWfstBeamSearch::Search calls GetBestPath(use_final=false) after every frame, so
    the partial hypothesis CAN change retroactively: a later frame may make a different
    lattice path best. This is where all user-visible flicker lives. Requires the
    compiled lm_decoder module; see language_model/README.md.

Metrics reported per emitted word w:
    revisions(w)      number of times the token at w's position changed after the
                      position was first occupied
    ttf(w)            time-to-final: (frame w reached its final value)
                      - (frame w's position was first occupied), in ms
    ttfe(w)           time-to-first-emission: frame w's position was first occupied,
                      relative to utterance start, in ms

and in aggregate:
    flicker_rate      total revisions / total emitted words
    ttf_p50/p95       the honest "when can the user trust this word" latency
    unstable_frac     fraction of words revised at least once

Usage
    # acoustic-level, from saved validation logits (no GPU, no LM)
    python stability.py --val_metrics results/causal_la0/checkpoint/val_metrics.pkl

    # compare emission timing of two checkpoints (the S6 emission-delay test)
    python stability.py --val_metrics results/causal_la0/checkpoint/val_metrics.pkl \
                        --compare_to results/causal_la4/checkpoint/val_metrics.pkl \
                        --expected_shift_frames 0.306
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .common import BLANK_CLASS, resolve_path, write_json
except ImportError:  # pragma: no cover - supports direct script execution
    from common import BLANK_CLASS, resolve_path, write_json  # type: ignore

BIN_SECONDS = 0.020


def percentile(values, pct: float) -> float:
    """common.percentile truth-tests its argument, so it rejects numpy arrays."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.percentile(array, pct))


def frame_to_ms(frame_idx: int, patch_size: int = 14, patch_stride: int = 4) -> float:
    """Wall-clock ms at which frame `frame_idx` can first be emitted.

    Frame f spans smoothed bins [s*f, s*f + P - 1] and is emitted the moment its last
    bin arrives (right-edge emission, see rnn_model.py:112 and streaming_infer.py:218).
    """
    return (patch_stride * frame_idx + patch_size - 1) * BIN_SECONDS * 1000.0


# --------------------------------------------------------------------------------------
# Word-level stability (needs an LM that emits partial hypotheses)
# --------------------------------------------------------------------------------------


@dataclass
class WordTrace:
    """History of one position in the partial transcript."""

    position: int
    first_frame: int
    final_frame: int
    revisions: int
    final_word: str
    values: List[str] = field(default_factory=list)


def trace_partials(partials: Sequence[Sequence[str]]) -> List[WordTrace]:
    """Turn a per-frame sequence of partial hypotheses into per-position traces.

    partials[t] is the full word list the decoder would have displayed after frame t.
    """
    traces: Dict[int, WordTrace] = {}
    for frame, hypothesis in enumerate(partials):
        for position, word in enumerate(hypothesis):
            trace = traces.get(position)
            if trace is None:
                traces[position] = WordTrace(
                    position=position,
                    first_frame=frame,
                    final_frame=frame,
                    revisions=0,
                    final_word=word,
                    values=[word],
                )
            elif word != trace.final_word:
                trace.revisions += 1
                trace.final_word = word
                trace.final_frame = frame
                trace.values.append(word)

    final = partials[-1] if partials else []
    return [traces[p] for p in sorted(traces) if p < len(final)]


def stability_metrics(
    traces_per_trial: Sequence[Sequence[WordTrace]],
    patch_size: int = 14,
    patch_stride: int = 4,
) -> Dict[str, Any]:
    revisions: List[int] = []
    ttf_ms: List[float] = []
    ttfe_ms: List[float] = []

    for traces in traces_per_trial:
        for trace in traces:
            revisions.append(trace.revisions)
            first = frame_to_ms(trace.first_frame, patch_size, patch_stride)
            final = frame_to_ms(trace.final_frame, patch_size, patch_stride)
            ttf_ms.append(final - first)
            ttfe_ms.append(first)

    n_words = len(revisions)
    if n_words == 0:
        return {"n_words": 0}

    return {
        "n_words": n_words,
        "n_trials": len(traces_per_trial),
        "flicker_rate": float(np.sum(revisions) / n_words),
        "unstable_frac": float(np.mean(np.asarray(revisions) > 0)),
        "revisions_max": int(np.max(revisions)),
        "ttf_ms": {
            "mean": float(np.mean(ttf_ms)),
            "p50": percentile(ttf_ms, 50),
            "p95": percentile(ttf_ms, 95),
            "max": float(np.max(ttf_ms)),
        },
        "ttfe_ms": {
            "mean": float(np.mean(ttfe_ms)),
            "p50": percentile(ttfe_ms, 50),
            "p95": percentile(ttfe_ms, 95),
        },
    }


# --------------------------------------------------------------------------------------
# Acoustic-level analysis (works from saved logits; no GPU, no LM)
# --------------------------------------------------------------------------------------


def load_val_logits(path: str | Path) -> List[np.ndarray]:
    """Per-trial [n_frames, n_classes] logits from a saved val_metrics.pkl.

    rnn_trainer.py:751 stores adjusted_lens under the key 'n_time_steps', so that field
    is a FRAME count, not a bin count.
    """
    with open(resolve_path(path), "rb") as handle:
        payload = pickle.load(handle)

    trials: List[np.ndarray] = []
    for batch_logits, batch_lens in zip(payload["logits"], payload["n_time_steps"]):
        for row in range(batch_logits.shape[0]):
            trials.append(batch_logits[row, : batch_lens[row], :].astype(np.float64))
    return trials


def collapse_with_frames(logits: np.ndarray) -> Tuple[List[int], List[int]]:
    """Online CTC collapse, returning tokens and the frame each was emitted at."""
    argmax = logits.argmax(-1)
    tokens: List[int] = []
    frames: List[int] = []
    previous: Optional[int] = None
    for frame, value in enumerate(argmax):
        if value != previous and value != BLANK_CLASS:
            tokens.append(int(value))
            frames.append(frame)
        previous = int(value)
    return tokens, frames


def verify_append_only(trials: Sequence[np.ndarray], limit: Optional[int] = None) -> Dict[str, int]:
    """Confirm the greedy partial hypothesis is monotone in the prefix order."""
    violations = 0
    emitted = 0
    for logits in trials[: limit or len(trials)]:
        argmax = logits.argmax(-1)
        hypothesis: List[int] = []
        history: List[List[int]] = []
        previous: Optional[int] = None
        for value in argmax:
            if value != previous and value != BLANK_CLASS:
                hypothesis.append(int(value))
            previous = int(value)
            history.append(list(hypothesis))
        for index in range(1, len(history)):
            if history[index][: len(history[index - 1])] != history[index - 1]:
                violations += 1
        emitted += len(hypothesis)
    return {"prefix_violations": violations, "emitted_tokens": emitted}


def posterior_stats(trials: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Blank skippability and confidence structure.

    Blank skippability predicts the gain from ctc_blank_skip_threshold, which is
    implemented at ctc_wfst_beam_search.cc:79 and disabled by default.
    Confidence structure is the gating signal an adaptive-compute decoder would use.
    """
    blank_prob: List[np.ndarray] = []
    entropy: List[np.ndarray] = []
    margin: List[np.ndarray] = []
    is_nonblank: List[np.ndarray] = []

    for logits in trials:
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        blank_prob.append(probs[:, BLANK_CLASS])
        entropy.append(-(probs * np.log(probs + 1e-12)).sum(-1))
        ordered = np.sort(probs, axis=-1)
        margin.append(ordered[:, -1] - ordered[:, -2])
        is_nonblank.append(logits.argmax(-1) != BLANK_CLASS)

    blank_prob = np.concatenate(blank_prob)
    entropy = np.concatenate(entropy)
    margin = np.concatenate(margin)
    is_nonblank = np.concatenate(is_nonblank)

    return {
        "n_frames": int(blank_prob.size),
        "blank_argmax_frac": float(np.mean(~is_nonblank)),
        "skippable_frac": {
            str(threshold): float(np.mean(blank_prob > threshold))
            for threshold in (0.9, 0.99, 0.999)
        },
        "entropy": {
            "median": float(np.median(entropy)),
            "mean": float(np.mean(entropy)),
            "p95": percentile(entropy, 95),
            "max_possible": float(np.log(trials[0].shape[-1])),
        },
        "entropy_nonblank": {
            "median": float(np.median(entropy[is_nonblank])),
            "mean": float(np.mean(entropy[is_nonblank])),
            "p95": percentile(entropy[is_nonblank], 95),
        },
        "top1_margin_nonblank": {
            "median": float(np.median(margin[is_nonblank])),
            "p25": percentile(margin[is_nonblank], 25),
            "p05": percentile(margin[is_nonblank], 5),
        },
        "high_entropy_frac": {
            str(threshold): {
                "all": float(np.mean(entropy > threshold)),
                "nonblank": float(np.mean(entropy[is_nonblank] > threshold)),
            }
            for threshold in (0.5, 1.0, 1.5)
        },
    }


def emission_delay(
    trials_a: Sequence[np.ndarray],
    trials_b: Sequence[np.ndarray],
    expected_shift_frames: float = 0.0,
) -> Dict[str, Any]:
    """Emission-timing shift of A relative to B on trials where both agree.

    Restricting to identical token sequences makes the comparison paired and removes
    any confound from the two models decoding differently.

    `expected_shift_frames` is the shift predicted by signal processing alone (for
    lookahead=0 vs lookahead=4 the smoother's group delay differs by 1.224 bins =
    0.306 frames). The residual is the LEARNED emission delay, which is what the
    delay-penalty literature exists to remove.
    """
    shifts: List[int] = []
    matched = 0
    for logits_a, logits_b in zip(trials_a, trials_b):
        tokens_a, frames_a = collapse_with_frames(logits_a)
        tokens_b, frames_b = collapse_with_frames(logits_b)
        if tokens_a and tokens_a == tokens_b:
            matched += 1
            shifts.extend(int(x - y) for x, y in zip(frames_a, frames_b))

    if not shifts:
        return {"matched_trials": 0}

    shift_array = np.asarray(shifts, dtype=float)
    mean_shift = float(shift_array.mean())
    return {
        "matched_trials": matched,
        "total_trials": len(trials_a),
        "paired_tokens": len(shifts),
        "mean_shift_frames": mean_shift,
        "mean_shift_ms": mean_shift * 80.0,
        "median_shift_frames": float(np.median(shift_array)),
        "sd_shift_frames": float(shift_array.std()),
        "same_frame_frac": float(np.mean(shift_array == 0)),
        "within_one_frame_frac": float(np.mean(np.abs(shift_array) <= 1)),
        "expected_shift_frames": expected_shift_frames,
        "excess_learned_delay_frames": mean_shift - expected_shift_frames,
        "excess_learned_delay_ms": (mean_shift - expected_shift_frames) * 80.0,
    }


def trial_geometry(trials: Sequence[np.ndarray], patch_size: int = 14, patch_stride: int = 4) -> Dict[str, Any]:
    n_frames = np.asarray([t.shape[0] for t in trials])
    n_bins = patch_stride * (n_frames - 1) + patch_size

    first_ms: List[float] = []
    last_ms: List[float] = []
    n_tokens: List[int] = []
    for logits in trials:
        nonblank = np.nonzero(logits.argmax(-1) != BLANK_CLASS)[0]
        if nonblank.size == 0:
            continue
        first_ms.append(frame_to_ms(int(nonblank[0]), patch_size, patch_stride))
        last_ms.append(frame_to_ms(int(nonblank[-1]), patch_size, patch_stride))
        n_tokens.append(len(collapse_with_frames(logits)[0]))

    return {
        "n_trials": len(trials),
        "bins_per_trial_median": float(np.median(n_bins)),
        "seconds_per_trial_median": float(np.median(n_bins) * BIN_SECONDS),
        "frames_per_trial_median": float(np.median(n_frames)),
        "tokens_per_trial_median": float(np.median(n_tokens)),
        "first_token_ms": {"p05": percentile(first_ms, 5), "p50": percentile(first_ms, 50)},
        "last_token_ms": {"p50": percentile(last_ms, 50)},
        "cold_start_ms": frame_to_ms(0, patch_size, patch_stride),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming stability and emission-timing metrics.")
    parser.add_argument(
        "--val_metrics",
        default="results/causal_la0/checkpoint/val_metrics.pkl",
        help="Saved val_metrics.pkl containing validation logits.",
    )
    parser.add_argument(
        "--compare_to",
        default=None,
        help="Second val_metrics.pkl for the paired emission-delay test.",
    )
    parser.add_argument(
        "--expected_shift_frames",
        type=float,
        default=0.0,
        help="Group-delay shift predicted by the smoother alone, in frames.",
    )
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--patch_stride", type=int, default=4)
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parsed = parser.parse_args()

    trials = load_val_logits(parsed.val_metrics)

    report: Dict[str, Any] = {
        "source": str(parsed.val_metrics),
        "geometry": trial_geometry(trials, parsed.patch_size, parsed.patch_stride),
        "append_only": verify_append_only(trials),
        "posterior": posterior_stats(trials),
    }

    if parsed.compare_to:
        report["emission_delay_vs"] = str(parsed.compare_to)
        report["emission_delay"] = emission_delay(
            trials,
            load_val_logits(parsed.compare_to),
            expected_shift_frames=parsed.expected_shift_frames,
        )

    append_only = report["append_only"]
    posterior = report["posterior"]
    print(f"trials: {report['geometry']['n_trials']}  frames: {posterior['n_frames']}")
    print(
        f"append-only: {append_only['prefix_violations']} prefix violations over "
        f"{append_only['emitted_tokens']} tokens "
        f"-> acoustic revisions/word = 0"
    )
    print(f"blank argmax: {posterior['blank_argmax_frac'] * 100:.2f}%")
    for threshold, fraction in posterior["skippable_frac"].items():
        print(f"  skippable at p(blank) > {threshold}: {fraction * 100:.2f}%")
    print(
        f"entropy median {posterior['entropy']['median']:.4f} / "
        f"max possible {posterior['entropy']['max_possible']:.3f}"
    )
    if "emission_delay" in report:
        delay = report["emission_delay"]
        print(
            f"emission shift: {delay['mean_shift_frames']:+.4f} frames "
            f"({delay['mean_shift_ms']:+.2f} ms); "
            f"excess over predicted: {delay['excess_learned_delay_ms']:+.2f} ms"
        )

    if parsed.out:
        write_json(Path(parsed.out), report)
        print(f"Wrote {parsed.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
