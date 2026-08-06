"""Word-level streaming stability metrics (grok_recs §6.3).

No published brain-to-text paper currently reports these. They are the
user-visible quantities under a streaming decoder:

  - Time-to-final (TTF): ms from first emission involving word j until it freezes
  - Revisions per word (RPV): how many times the surface form changed before freeze
  - Prefix freeze rate: fraction of emissions that only extend a frozen prefix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _tokenize_words(text: str) -> List[str]:
    return [w for w in text.strip().split() if w]


@dataclass
class StabilityTracker:
    """Online tracker fed successive partial hypotheses."""

    partials: List[str] = field(default_factory=list)
    times_ms: List[float] = field(default_factory=list)

    def update(self, partial_text: str, time_ms: float) -> None:
        self.partials.append(partial_text)
        self.times_ms.append(float(time_ms))

    def finalize(self, final_text: Optional[str] = None) -> Dict[str, Any]:
        if final_text is None:
            final_text = self.partials[-1] if self.partials else ""
        return compute_stability_metrics(self.partials, self.times_ms, final_text)


def compute_stability_metrics(
    partials: Sequence[str],
    times_ms: Sequence[float],
    final_text: str,
) -> Dict[str, Any]:
    """Compute TTF / RPV / prefix-freeze stats from a stream of partials.

    Alignment is left-to-right by word index against the final word list
    (not full edit-distance alignment — sufficient for streaming prefix UIs).
    """
    if len(partials) != len(times_ms):
        raise ValueError("partials and times_ms length mismatch")

    final_words = _tokenize_words(final_text)
    n_words = len(final_words)
    if n_words == 0:
        return {
            "n_words": 0,
            "mean_ttf_ms": 0.0,
            "median_ttf_ms": 0.0,
            "p90_ttf_ms": 0.0,
            "mean_rpv": 0.0,
            "prefix_freeze_rate": 1.0,
            "per_word": [],
        }

    # For each final word index j, track surface forms seen and freeze time.
    forms_seen: List[List[str]] = [[] for _ in range(n_words)]
    first_appear_ms: List[Optional[float]] = [None] * n_words
    last_change_ms: List[Optional[float]] = [None] * n_words

    prev_words: List[str] = []
    prefix_freeze_events = 0
    total_events = 0

    for text, t in zip(partials, times_ms):
        words = _tokenize_words(text)
        total_events += 1
        # Prefix freeze: all words that exist in prev and are within final length
        # match final's corresponding words and were already present.
        if prev_words:
            k = min(len(prev_words), len(words), n_words)
            if k > 0 and words[:k] == prev_words[:k] and words[:k] == final_words[:k]:
                # only extended or unchanged
                if len(words) >= len(prev_words):
                    prefix_freeze_events += 1
        for j in range(min(len(words), n_words)):
            w = words[j]
            if first_appear_ms[j] is None:
                first_appear_ms[j] = t
            if not forms_seen[j] or forms_seen[j][-1] != w:
                forms_seen[j].append(w)
                last_change_ms[j] = t
        prev_words = words

    # Finalization time: last time the slot changed to the final correct form,
    # or last emission if never correct mid-stream.
    ttf_list: List[float] = []
    rpv_list: List[float] = []
    per_word: List[Dict[str, Any]] = []
    final_t = times_ms[-1] if times_ms else 0.0

    for j in range(n_words):
        forms = forms_seen[j]
        # revisions = number of changes after first appearance
        rpv = max(len(forms) - 1, 0)
        rpv_list.append(float(rpv))
        t0 = first_appear_ms[j]
        if t0 is None:
            ttf = float("nan")
        else:
            # freeze when form becomes final word and never changes after
            t_final = last_change_ms[j] if last_change_ms[j] is not None else final_t
            # if last form is wrong, freeze at end
            if forms and forms[-1] != final_words[j]:
                t_final = final_t
            ttf = float(t_final - t0)
        ttf_list.append(ttf)
        per_word.append(
            {
                "index": j,
                "final": final_words[j],
                "forms": forms,
                "ttf_ms": ttf,
                "rpv": rpv,
                "first_appear_ms": first_appear_ms[j],
            }
        )

    import numpy as np

    ttf_arr = np.asarray([x for x in ttf_list if x == x], dtype=np.float64)  # drop nan
    return {
        "n_words": n_words,
        "mean_ttf_ms": float(np.mean(ttf_arr)) if ttf_arr.size else float("nan"),
        "median_ttf_ms": float(np.median(ttf_arr)) if ttf_arr.size else float("nan"),
        "p90_ttf_ms": float(np.percentile(ttf_arr, 90)) if ttf_arr.size else float("nan"),
        "mean_rpv": float(np.mean(rpv_list)) if rpv_list else 0.0,
        "prefix_freeze_rate": float(prefix_freeze_events / total_events) if total_events else 1.0,
        "per_word": per_word,
    }


def partials_from_token_streams(
    token_partials: Sequence[Sequence[int]],
    times_ms: Sequence[float],
    token_to_str: Optional[Any] = None,
) -> Tuple[List[str], List[float]]:
    """Convert cumulative token-id partials to text partials."""
    texts: List[str] = []
    for tokens in token_partials:
        if token_to_str is None:
            texts.append(" ".join(str(int(t)) for t in tokens))
        else:
            texts.append(" ".join(token_to_str(int(t)) for t in tokens))
    return texts, list(times_ms)
