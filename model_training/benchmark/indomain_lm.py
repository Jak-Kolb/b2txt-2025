"""C28-A — in-domain n-gram trained on the TRAINING transcriptions, used to rescore the n-best.

Why this exists
---------------
C13/C15 established that the shipped 1-gram tops out at ~35 % WER, and that the failure is purely
lexical: "part ways" decodes as "parte weighs", "see" as "sci". Measuring the n-best oracle showed
why — at nbest=100 on val-dev the list contains a candidate averaging **15.17 % WER** while the
decoder returns one averaging 36.04 %, because the 1-gram assigns near-identical scores to every
candidate and the 1-best is therefore close to arbitrary. **20.87 WER points are sitting in the
list**, unreachable without a sequential LM.

This is the cheapest possible test of "does any sequential LM recover them": train a trigram on the
8,072 TRAINING transcriptions and use it to re-rank. No FST rebuild, no corpus download, no
arpa2fst — the existing decoder produces the candidates, this only reorders them.

Honest limits, stated up front
------------------------------
* 50,648 word tokens / 3,745 types is a tiny corpus. Coverage on val will be poor. This is a LOWER
  BOUND on what a real n-gram buys, not a deployable LM.
* Rescoring can only pick among candidates the 1-gram surfaced. The oracle at nbest=100 (15.17 %)
  is a hard ceiling; shallow fusion inside the beam could go below it, re-ranking cannot.
* Trained on TRAIN transcriptions only. Including val would leak and invalidate the result.
* **6.94 % of val-dev sentences appear verbatim in the training transcriptions** — the prompts are
  drawn from overlapping pools. `stream_lm.py rescore` scores that subset separately; report the
  unseen row.

Result (val-dev, nbest=100, acoustic_scale 1.0 / blank_penalty 30): 36.04 -> 18.33 % WER overall,
**36.24 -> 18.83 % on the 1,166 sentences that never appear in train**. Interior optimum at
alpha=1.0.
"""
import argparse
import glob
import json
import math
import pathlib
import re
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
_PUNCT = re.compile(r"[^\w\s']")


def norm(text):
    return _PUNCT.sub(" ", text.lower()).split()


def read_sentences_file(path):
    """Load the exported TRAIN sentences. Used by the LM env, which has no h5py."""
    return [l for l in pathlib.Path(path).read_text().split("\n") if l.strip()]


def read_train_sentences(data_dir):
    """TRAIN split only — reading val here would leak into every number downstream."""
    import h5py
    import numpy as np
    out = []
    for path in sorted(glob.glob(str(pathlib.Path(data_dir) / "*" / "data_train.hdf5"))):
        with h5py.File(path, "r") as h:
            for key in h.keys():
                if "transcription" not in h[key]:
                    continue
                raw = h[key]["transcription"][:]
                out.append(bytes(raw[raw > 0].astype(np.uint8)).decode("ascii").strip())
    return out


class Trigram:
    """Interpolated absolute discounting (simplified Kneser-Ney).

    p(w|u,v) = max(c(uvw)-D, 0)/c(uv) + lambda(uv) * p(w|v),  recursing to a unigram backed off
    to a uniform floor. Chosen over full modified KN because with 50k words the discount schedule
    is not what limits us — coverage is — and this form is easy to verify by hand.
    """

    def __init__(self, discount=0.75):
        self.D = discount
        self.c1, self.c2, self.c3 = defaultdict(int), defaultdict(int), defaultdict(int)
        self.ctx2, self.ctx1 = defaultdict(int), defaultdict(int)
        self.follow2, self.follow1 = defaultdict(set), defaultdict(set)
        self.total = 0
        self.vocab = set()

    def train(self, sentences):
        for s in sentences:
            w = ["<s>", "<s>"] + norm(s) + ["</s>"]
            self.vocab.update(w)
            for i in range(2, len(w)):
                u, v, x = w[i - 2], w[i - 1], w[i]
                self.c1[x] += 1
                self.total += 1
                self.c2[(v, x)] += 1
                self.ctx1[v] += 1
                self.follow1[v].add(x)
                self.c3[(u, v, x)] += 1
                self.ctx2[(u, v)] += 1
                self.follow2[(u, v)].add(x)
        self.V = max(len(self.vocab), 1)
        return self

    def _p1(self, x):
        c = self.c1.get(x, 0)
        if self.total == 0:
            return 1.0 / self.V
        # back off to a uniform floor so OOV never yields probability zero
        return (max(c - self.D, 0.0) / self.total) + (self.D * self.V / self.total) * (1.0 / self.V) \
            if c else 0.5 / self.total

    def _p2(self, v, x):
        cv = self.ctx1.get(v, 0)
        if cv == 0:
            return self._p1(x)
        lam = self.D * len(self.follow1[v]) / cv
        return max(self.c2.get((v, x), 0) - self.D, 0.0) / cv + lam * self._p1(x)

    def _p3(self, u, v, x):
        cuv = self.ctx2.get((u, v), 0)
        if cuv == 0:
            return self._p2(v, x)
        lam = self.D * len(self.follow2[(u, v)]) / cuv
        return max(self.c3.get((u, v, x), 0) - self.D, 0.0) / cuv + lam * self._p2(v, x)

    def logprob(self, words):
        """Total log10 probability of a word sequence, plus its length in tokens."""
        w = ["<s>", "<s>"] + list(words) + ["</s>"]
        lp = 0.0
        for i in range(2, len(w)):
            lp += math.log10(max(self._p3(w[i - 2], w[i - 1], w[i]), 1e-12))
        return lp, len(w) - 2

    def stats(self):
        return {"sentences_seen": None, "tokens": self.total, "vocab": len(self.vocab),
                "bigrams": len(self.c2), "trigrams": len(self.c3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(REPO / "data/hdf5_data_final"))
    ap.add_argument("--out", default=str(REPO / "results/indomain_trigram.json"))
    ap.add_argument("--discount", type=float, default=0.75)
    a = ap.parse_args()

    sents = read_train_sentences(a.data_dir)
    lm = Trigram(a.discount).train(sents)
    st = lm.stats()
    st["sentences_seen"] = len(sents)
    print(f"trained on {len(sents)} TRAIN sentences")
    for k, v in st.items():
        print(f"  {k:<16} {v}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    txt = out.with_name("train_sentences.txt")
    txt.write_text("\n".join(x.replace("\n", " ") for x in sents))
    print(f"wrote {txt} ({len(sents)} sentences) — the LM env has no h5py, so it reads this")
    out.write_text(json.dumps({
        "discount": a.discount, "stats": st,
        "c1": {k: v for k, v in lm.c1.items()},
        "c2": {"\t".join(k): v for k, v in lm.c2.items()},
        "c3": {"\t".join(k): v for k, v in lm.c3.items()},
    }))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
