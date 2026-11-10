"""Live-word (clock A) evaluation of recorded partial outputs: when each displayed word becomes final, measured from a
model-estimated word end.

Clock A (user decision 2026-10-18): a displayed word must be final within the budget (200 / 500 / 1,500 ms).
- Word ends: CTC Viterbi forced alignment of the reference phonemes (' | ' after every word) on a frozen aligner's
  log-probs. A word ends at the last frame of its last phoneme; frame f's input window ends at (14 + 4 f) * 20 ms.
  Two aligners bracket the unknown true timing: bidirectional anc_bi768 (fires early: strict) and causal la0 (fires
  late: lenient; user 2026-10-18: lenient for now). Evaluation only; no online policy sees them.
- Partial outputs: the run's recorded partials (output_trace.jsonl). Partial f becomes available at the run's
  nominal clock (patch + (f + R) * stride) * 20 ms, R = look-ahead frames (decoder compute excluded).
- Policies. Online: stable-k and lag-n (as harness commit); sil+D: commit the first n words of the partial D frames
  after the model's own n-th greedy word-boundary (' | ') emission; conf-tau+D: as sil+D, but a word whose greedy
  phonemes all have confidence >= tau (max posterior over each phoneme's frames) commits at its boundary (words commit
  in order). Not online: oracle B commits the first n words once n reference word ends + B have passed, a bound on
  what a commit rule can get from this decoder within B.
- Delay of a committed word = commit time - end of the reference word it aligns to (word-level edit alignment;
  insertions have no reference end). Words not committed before the trial end are committed at the (oracle) end.
- A word the decoder has not displayed by its deadline cannot be committed in time; the oracle then commits it at
  first display. causal_la0 is also the lenient aligner, so its own lenient delays are circular.
usage: clock_a.py OUT_JSON MODEL [MODEL ...]   (registered acoustic models with a +gen4g_standard run and cached
                                                log-probs in learning/per_eval)
"""
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model_training"))
from harness.commitment import load_trials
from harness.scopes import list_trials
from model_training.benchmark.stream_lm import levenshtein, norm_words

LP = REPO / "learning/per_eval"
RUNS = REPO / "results/harness/runs"
SIL = 40
ALIGNERS = {"strict": "anc_bi768_s10_final", "lenient": "causal_la0"}
BUDGETS = (200, 500, 1500)
STABLE = (1, 2, 3, 4, 6, 8, 12, 16)
LAGS = (1, 2)
DEADLINES = (0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18)
GATES = ((0.99, 4), (0.99, 8), (0.99, 15), (0.999, 4), (0.999, 8), (0.999, 15))
RUN_NAMES = dict(causal_la0="la0_gen4g_standard", g2_masking="g2_gen4g_standard", g3_objective="g3_gen4g_standard")


def viterbi_label_ends(lp, labels):
    """CTC forced alignment of labels on log-probs [T, C]; the last frame of each label's segment, or None."""
    T, L = lp.shape[0], len(labels)
    S = 2 * L + 1
    ext = np.zeros(S, dtype=np.int64); ext[1::2] = labels
    emit = lp[:, ext]
    skip = np.zeros(S, dtype=bool); skip[3::2] = labels[1:] != labels[:-1]
    neg = -1e30
    delta = np.full(S, neg); delta[0] = emit[0, 0]; delta[1] = emit[0, 1]
    back = np.zeros((T, S), dtype=np.int8)
    cols = np.arange(S)
    for t in range(1, T):
        one = np.concatenate(([neg], delta[:-1]))
        two = np.where(skip, np.concatenate(([neg, neg], delta[:-2])), neg)
        cand = np.stack([delta, one, two])
        arg = cand.argmax(0)
        delta = cand[arg, cols] + emit[t]
        back[t] = arg
    end = S - 1 if delta[S - 1] >= delta[S - 2] else S - 2
    if delta[end] < neg / 2:
        return None
    ends, s = [None] * L, end
    for t in range(T - 1, -1, -1):
        if s % 2 == 1 and ends[(s - 1) // 2] is None:
            ends[(s - 1) // 2] = t
        s -= int(back[t, s])
    return ends


def word_end_frames(lp, ids):
    """Aligned end frame of each reference word (its last phoneme before ' | '), or None."""
    ends = viterbi_label_ends(lp.astype(np.float32), ids)
    if ends is None:
        return None
    return [ends[j - 1] for j in range(1, len(ids)) if ids[j] == SIL and ids[j - 1] != SIL]


def word_boundaries(lp):
    """The model's own word ends: frames where the greedy path emits ' | ' after at least one phoneme, each with the
    lowest confidence (max posterior over the segment's frames) among that word's greedy phonemes."""
    probs = np.exp(lp.astype(np.float32))
    arg = probs.argmax(-1)
    out, prev, start, confs = [], 0, 0, []
    for f, c in enumerate(list(arg) + [0]):
        if c != prev:
            if prev not in (0, SIL):
                confs.append(float(probs[start:f, prev].max()))
            if c == SIL and confs:
                out.append((f, min(confs)))
                confs = []
            start = f
        prev = c
    return out


class Stable:
    def __init__(self, k):
        self.k, self.hist, self.name = k, [], f"stable-{k}"

    def __call__(self, f, words):
        self.hist.append(tuple(words))
        recent = self.hist[-self.k:]
        if len(recent) < self.k:
            return 0
        n = 0
        for i in range(len(words)):
            if all(len(h) > i and h[:i + 1] == tuple(words[:i + 1]) for h in recent):
                n = i + 1
            else:
                break
        return n


class Lag:
    def __init__(self, n):
        self.n, self.name = n, f"lag-{n}"

    def __call__(self, f, words):
        return max(0, len(words) - self.n)


class AtTimes:
    """Commit the first n displayed words once n trigger frames have passed (triggers made non-decreasing)."""

    def __init__(self, name, triggers):
        self.name = name
        self.triggers = np.maximum.accumulate(np.asarray(triggers)) if len(triggers) else np.array([])

    def __call__(self, f, words):
        return min(len(words), int(np.searchsorted(self.triggers, f, side="right")))


def simulate(policy, partials, final):
    committed, frames = [], []
    for f, words in enumerate(partials):
        target = policy(f, words)
        if target > len(committed):
            committed.extend(words[len(committed):target])
            frames.extend([f] * (target - len(frames)))
    output = committed + list(final[len(committed):])
    return output, frames + [None] * (len(output) - len(frames))


def word_pairs(hyp, ref):
    """(hyp index, ref index) for matched or substituted words in a minimum edit alignment."""
    n, m = len(hyp), len(ref)
    d = np.zeros((n + 1, m + 1), dtype=np.int32); d[:, 0] = range(n + 1); d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (hyp[i - 1] != ref[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (hyp[i - 1] != ref[j - 1]):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs


def run_dir(model):
    name = RUN_NAMES.get(model, f"{model}+gen4g_standard")
    runs = sorted(p for p in RUNS.iterdir() if re.fullmatch(rf"\d{{8}}T\d{{6}}Z_{re.escape(name)}", p.name))
    return runs[-1]


class References:
    """Normalized reference words and their aligned end times (ms) per aligner, cached by val trial index."""

    def __init__(self):
        self.trials = list_trials(REPO / "data/hdf5_data_final", "val")
        self.index = {(t.session, t.trial_key): i for i, t in enumerate(self.trials)}
        self.aligner_lp = {k: np.load(LP / f"logprobs_{v}.npz") for k, v in ALIGNERS.items()}
        self.cache = {}

    def get(self, row):
        ti = self.index[(row["session"], row["trial_key"])]
        if ti not in self.cache:
            with h5py.File(self.trials[ti].hdf5_path, "r") as h:
                g = h[self.trials[ti].trial_key]
                ids = g["seq_class_ids"][: int(g.attrs["seq_len"])].astype(np.int64)
            ref = norm_words(row["ref"])
            ends = {}
            for k, z in self.aligner_lp.items():
                frames = word_end_frames(z[f"t{ti}"], ids)
                ends[k] = [(14 + 4 * f) * 20.0 for f in frames] if frames is not None and len(frames) == len(ref) else None
            self.cache[ti] = (ti, ref, ends)
        return self.cache[ti]


def evaluate(manifest, rows, partials, finals, refs, own_lp):
    """Per-policy WER and delay distributions for one decoder's recorded partials."""
    spec = manifest["pipeline"]["model"]
    right = spec.get("lookahead_frames") or 0
    smooth = manifest["pipeline"]["preprocess"]["effective"].get("smooth_lookahead") or 0
    frame_ms = lambda f: (spec["patch_size"] + (f + right) * spec["patch_stride"] + smooth) * 20.0
    acc, n_timed = {}, 0
    for row in rows:
        ti, ref, ends = refs.get(row)
        n_timed += ends["strict"] is not None and ends["lenient"] is not None
        p, final = partials[row["i"]], finals[row["i"]]
        bounds = word_boundaries(own_lp[f"t{ti}"])
        events = [e for e, _ in bounds]
        end_ms = row["n_bins"] * 20.0
        policies = ([Stable(k) for k in STABLE] + [Lag(n) for n in LAGS]
                    + [AtTimes(f"sil+{d}", [e + d for e in events]) for d in DEADLINES]
                    + [AtTimes(f"conf{tau}+{d}", [e if c >= tau else e + d for e, c in bounds]) for tau, d in GATES])
        for k in ALIGNERS:
            if ends[k] is not None:
                for b in BUDGETS:
                    # last partial frame available by each reference word end + B (frame 0 if none is)
                    trig = [max([f for f in range(len(p)) if frame_ms(f) <= e + b], default=0) for e in ends[k]]
                    policies.append(AtTimes(f"oracle-{k}-{b}", trig))
        for policy in policies:
            output, frames = simulate(policy, p, final)
            hyp = norm_words(" ".join(output))
            a = acc.setdefault(policy.name, dict(edits=0, words=0, endpoint=0, out=0, delays={k: [] for k in ALIGNERS}))
            a["edits"] += levenshtein(ref, hyp); a["words"] += len(ref)
            a["out"] += len(output); a["endpoint"] += sum(f is None for f in frames)
            if len(hyp) != len(output):
                continue                                    # normalization changed word count; skip timing
            for k in ALIGNERS:
                if ends[k] is None:
                    continue
                for hi, ri in word_pairs(hyp, ref):
                    commit = frame_ms(frames[hi]) if frames[hi] is not None else end_ms
                    a["delays"][k].append(commit - ends[k][ri])
    summary = {}
    for name, a in acc.items():
        s = dict(wer=100.0 * a["edits"] / a["words"], endpoint_pct=100.0 * a["endpoint"] / a["out"])
        for k, ds in a["delays"].items():
            ds = np.array(ds)
            if len(ds):
                s[k] = dict(p50=float(np.percentile(ds, 50)), p95=float(np.percentile(ds, 95)), max=float(ds.max()),
                            **{f"within_{b}": float(100 * np.mean(ds <= b)) for b in BUDGETS})
        summary[name] = s
    return dict(R_ms=right * spec["patch_stride"] * 20, n_trials=len(rows), n_timed=n_timed, policies=summary)


def best_under_budget(summary, anchor, budget, prefix=None):
    ok = [(s["wer"], n) for n, s in summary.items() if not n.startswith("oracle") and anchor in s
          and s[anchor]["p95"] <= budget and (prefix is None or n.startswith(prefix))]
    return min(ok) if ok else None


def print_summary(label, report):
    summary = report["policies"]
    print(f"\n== {label} (R {report['R_ms']} ms; {report['n_timed']}/{report['n_trials']} trials timed by both aligners)",
          flush=True)
    for k in ALIGNERS:
        for b in BUDGETS:
            best = best_under_budget(summary, k, b)
            orc = summary.get(f"oracle-{k}-{b}", {}).get("wer")
            print(f"  {k:<8} {b:>5} ms: best online (p95 delay <= budget) "
                  f"{f'{best[0]:.2f} ({best[1]})' if best else 'none'} | timing-oracle bound "
                  f"{f'{orc:.2f}' if orc is not None else '-'}", flush=True)


def main(out_path, models):
    refs, report = References(), {}
    for model in models:
        manifest, rows, partials, finals = load_trials(run_dir(model))
        report[model] = evaluate(manifest, rows, partials, finals, refs, np.load(LP / f"logprobs_{model}.npz"))
        print_summary(model, report[model])
    Path(out_path).write_text(json.dumps(dict(aligners=ALIGNERS, budgets=BUDGETS, models=report), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
