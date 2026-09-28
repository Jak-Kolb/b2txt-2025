"""Word-commitment policies evaluated on a run's recorded partial outputs (no GPU, no decoding).

A policy sees only the partial hypotheses produced so far (never references or future frames)
and commits words that are then never retracted. At the (oracle) trial end the output is the
committed words followed by the final hypothesis's remaining positions. Metrics are defined in
evaluate_trial(); times use the nominal input clock of each partial output (compute excluded)
and word delays are measured from a word's first appearance in the partials, a proxy because
the release has no word-level alignments.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from model_training.benchmark.output_trace import word_edits
from model_training.benchmark.stream_lm import levenshtein, norm_words

from . import metrics
from .store import write_new_json, utc_stamp

VERSION = 1
DEFINITIONS = dict(
    clock="nominal input availability of each partial output, (window + frame*stride + lookahead) * 20 ms; "
          "decoder compute excluded (paced median frame lag for la0 is about 9.5 ms)",
    output="committed words, then the final hypothesis from the next position on (position splice)",
    commit_errors="committed-before-endpoint words that differ from the final hypothesis at that position",
    waiting="commit time minus the first time the partials showed that word at that position",
    revisions="non-append word edits between consecutive displayed texts (committed + tentative tail)",
    endpoint="dataset trial end (oracle endpointing)",
    exposure="val-dev selected checkpoints and tuned decode settings; policy parameters are explored on the same data")


class Policy:
    """Online policy: step(partial_words) -> number of leading positions to commit (monotone)."""

    online = True

    def __init__(self):
        self.history = []

    def step(self, words):
        self.history.append(tuple(words))
        return self.commit_count(words)


class StableK(Policy):
    def __init__(self, k):
        super().__init__()
        self.k, self.name = k, f"stable-{k}"

    def commit_count(self, words):
        recent = self.history[-self.k:]
        if len(recent) < self.k:
            return 0
        count = 0
        for i in range(len(words)):
            if all(len(h) > i and h[:i + 1] == tuple(words[:i + 1]) for h in recent):
                count = i + 1
            else:
                break
        return count


class LagN(Policy):
    def __init__(self, n):
        super().__init__()
        self.n, self.name = n, f"lag-{n}"

    def commit_count(self, words):
        return max(0, len(words) - self.n)


class Never(Policy):
    name = "none"

    def commit_count(self, words):
        return 0


def simulate(policy, partials, final):
    """Run one policy over a trial; returns committed words with their commit frame (None = endpoint)."""
    committed, frames, displayed = [], [], []
    for f, words in enumerate(partials):
        target = policy.step(words)
        if target > len(committed):
            committed.extend(words[len(committed):target])
            frames.extend([f] * (target - len(frames)))
        displayed.append(committed + list(words[len(committed):]))
    output = committed + list(final[len(committed):])
    frames = frames + [None] * (len(output) - len(frames))
    displayed.append(output)
    return output, frames, displayed


def retrospective(partials, final):
    """Hindsight reference: each final word committed when its prefix last changed (not online)."""
    frames = []
    for i in range(len(final)):
        last = None
        for f in range(len(partials) - 1, -1, -1):
            if tuple(partials[f][:i + 1]) != tuple(final[:i + 1]):
                break
            last = f
        frames.append(last)
    # a prefix is committed only after all earlier words are
    for i in range(1, len(frames)):
        if frames[i] is not None and (frames[i - 1] is None or frames[i - 1] > frames[i]):
            frames[i] = frames[i - 1]
    displayed = [list(p) for p in partials] + [list(final)]
    return list(final), frames, displayed


def evaluate_trial(output, frames, displayed, partials, final, reference, frame_ms, end_ms):
    edits, words = metrics.word_errors(reference, " ".join(output))
    early = [(i, f) for i, f in enumerate(frames) if f is not None]
    commit_errors = sum(1 for i, _ in early if i >= len(final) or output[i] != final[i])
    first_seen = {}
    for f, p in enumerate(partials):
        for i, w in enumerate(p):
            first_seen.setdefault((i, w), f)
    waits, before_end = [], []
    for i, (w, f) in enumerate(zip(output, frames)):
        commit_ms = frame_ms(f) if f is not None else end_ms
        seen = first_seen.get((i, w))
        seen_ms = frame_ms(seen) if seen is not None else end_ms
        waits.append(max(0.0, commit_ms - seen_ms))
        before_end.append(end_ms - commit_ms)
    revisions = 0
    for a, b in zip(displayed, displayed[1:]):
        e = word_edits(a, b)
        revisions += e["insertions"] + e["deletions"] + e["substitutions"]
    return dict(edits=edits, ref_words=words, output_words=len(output), early=len(early),
                commit_errors=commit_errors, waits=waits, before_end=before_end, revisions=revisions)


def load_trials(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["tier"] not in ("standard", "sweep"):
        raise ValueError("Commitment evaluation uses a standard run's accuracy trace")
    with (run_dir / "trials.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    partials = {r["i"]: [] for r in rows}
    finals = {}
    with (run_dir / "output_trace.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["record"] != "output":
                continue
            if record["kind"] == "partial":
                partials[record["cache_index"]].append(record["words"])
            else:
                finals[record["cache_index"]] = record["words"]
    return manifest, rows, partials, finals


def default_policies():
    return ([lambda: Never()] + [lambda k=k: StableK(k) for k in (1, 2, 3, 4, 6, 8, 12, 16)]
            + [lambda n=n: LagN(n) for n in (1, 2, 3)])


def evaluate_run(run_dir, policies=None, n_boot=10_000):
    manifest, rows, partials, finals = load_trials(run_dir)
    model = manifest["pipeline"]["model"]
    lookahead = manifest["pipeline"]["preprocess"]["effective"].get("smooth_lookahead") or 0
    frame_ms = lambda f: (model["patch_size"] + f * model["patch_stride"] + lookahead) * 20.0
    makers = (policies or default_policies()) + [None]  # None = retrospective reference
    results, baseline_rows = [], None
    for make in makers:
        per_trial = []
        for row in rows:
            p, final = partials[row["i"]], finals[row["i"]]
            if make is None:
                output, frames, displayed = retrospective(p, final)
                name, online = "retrospective (hindsight)", False
            else:
                policy = make()
                output, frames, displayed = simulate(policy, p, final)
                name, online = policy.name, True
            result = evaluate_trial(output, frames, displayed, p, final, row["ref"], frame_ms, row["n_bins"] * 20.0)
            per_trial.append(dict(result, session=row["session"], split=row["split"], trial_key=row["trial_key"],
                                  ref=row["ref"], hyp=" ".join(output)))
        early = sum(t["early"] for t in per_trial)
        out_words = sum(t["output_words"] for t in per_trial)
        waits = [w for t in per_trial for w in t["waits"]]
        before_end = [b for t in per_trial for b in t["before_end"]]
        summary = dict(policy=name, online=online,
                       wer=dict(metrics.pooled(per_trial), **metrics.session_bootstrap(per_trial, n_boot=n_boot)),
                       commit_errors=dict(n=sum(t["commit_errors"] for t in per_trial), early_commits=early,
                                          pct=100.0 * sum(t["commit_errors"] for t in per_trial) / early if early else None),
                       early_commit_pct=100.0 * early / out_words if out_words else None,
                       waiting_ms=metrics.distribution(waits),
                       committed_before_trial_end_ms=metrics.distribution(before_end),
                       visible_revisions_per_100_words=100.0 * sum(t["revisions"] for t in per_trial) / out_words
                       if out_words else None)
        if name == "none":
            baseline_rows = per_trial
        elif baseline_rows is not None:
            paired = metrics.paired_compare(baseline_rows, per_trial, n_boot=n_boot)
            summary["delta_wer_vs_none"] = dict(points=paired["delta_wer_points"], ci95=paired["ci95"],
                                                trials_worse=paired["trials"]["worsened"],
                                                trials_better=paired["trials"]["improved"])
        results.append(summary)
    return dict(version=VERSION, run_id=manifest["run_id"], pipeline=manifest["pipeline"]["name"],
                scope=dict(name=manifest["scope"]["name"], hash=manifest["scope"]["hash"],
                           n_trials=manifest["scope"]["n_trials"]),
                created=datetime.now(timezone.utc).isoformat(), definitions=DEFINITIONS, policies=results)


def save(result, root):
    path = Path(root) / "commitment" / f"{result['run_id']}_{utc_stamp()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(path, result)
    return path


def latest(run_id, root):
    """The most recent saved evaluation for a run, or None."""
    paths = sorted((Path(root) / "commitment").glob(f"{run_id}_*.json"))
    return json.loads(paths[-1].read_text()) if paths else None
