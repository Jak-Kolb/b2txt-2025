"""C29 — rescore the n-best with a pretrained causal LM.

What this is testing
--------------------
With the C28 4-gram in the beam, val-dev 1-best is 8.49 % while the ORACLE over the 100-best is
2.88 % (`results/c29_oracle_4gram.json`). The published full-stack figure is 2.66 %. So the correct
hypothesis is already in the list at essentially published accuracy, and the remaining 5.62 points
is a *selection* problem. An in-domain trigram recovers 0.00 of it — a well-estimated 4-gram
already subsumes what a small n-gram knows — so the question is what a model of a different KIND
claims.

Why rescoring rather than shallow fusion, for now
-------------------------------------------------
All ~30 candidates of a trial are independent short sequences, so rescoring is ONE batched forward
per utterance. Shallow fusion has to score extensions as the beam advances and cannot batch across
time, which is where the plan's "+2-10 ms/word" comes from. Rescoring is also capped by the oracle
(2.88 %), and that cap is not binding while 2.88 % is already ~the published number.

Scoring
-------
The decoder ranks by `lm_score + acoustic_scale * ac_score` (brain_speech_decoder.cc:123-124). This
keeps that as the base and adds the neural term, so alpha=beta=0 must reproduce the 1-best exactly
— the harness's own self-check.

    score(h) = lm_4gram(h) + acoustic_scale*ac(h) + alpha*logP_neural(h) + beta*|h|

    cd model_training && ../.venv/bin/python -m benchmark.neural_rescore
"""
import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
_PUNCT = re.compile(r"[^\w\s']")


def norm_words(text):
    return _PUNCT.sub(" ", text.lower()).split()


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.inference_mode()
def score_sentences(model, tok, sentences, device, batch_size=256):
    """Total log P(sentence) under a causal LM, one value per sentence.

    Each sequence is prefixed with the EOS token so the first real token also gets a conditional
    probability; without it the first word is unscored and short candidates gain an unfair edge.
    """
    out = np.zeros(len(sentences), dtype=np.float64)
    order = np.argsort([-len(s) for s in sentences])       # length-sort to minimise padding
    bos = tok.eos_token_id

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = [[bos] + tok.encode(" " + sentences[i].strip()) for i in idx]
        width = max(len(e) for e in enc)
        ids = torch.full((len(enc), width), bos, dtype=torch.long)
        mask = torch.zeros((len(enc), width), dtype=torch.bool)
        for r, e in enumerate(enc):
            ids[r, :len(e)] = torch.tensor(e)
            mask[r, 1:len(e)] = True                        # score real tokens only, not the BOS
        ids, mask = ids.to(device), mask.to(device)

        # cross_entropy fuses log_softmax+gather, so the [B, T, V] logits are never duplicated.
        # That matters: Qwen's vocab is 151,936 against GPT-2's 50,257, and materialising a second
        # copy OOMs a 16 GB card at any useful batch size.
        logits = model(ids).logits
        tgt = ids[:, 1:]
        lp = -torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            tgt.reshape(-1), reduction="none").view(tgt.shape)
        lp = lp * mask[:, 1:]
        out[idx] = lp.sum(-1).double().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", default=str(REPO / "results/c29_oracle_4gram.json"))
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--alphas", default="0,0.25,0.5,1,2,4,8")
    ap.add_argument("--betas", default="0,2,4")
    ap.add_argument("--gammas", default="1,0.5,0.25,0",
                    help="weight on the 4-gram score. 1=add neural on top (double-counts the LM), "
                         "0=replace it, matching BrainSpeechDecoder::Rescore")
    ap.add_argument("--top_k", type=int, default=0,
                    help="rescore only the top-k candidates per trial (0 = all). The neural forward "
                         "is the dominant finalization cost, and it scales with list length, so "
                         "this is the accuracy/latency knob for larger models.")
    ap.add_argument("--train_sentences", default=str(REPO / "results/train_sentences.txt"))
    ap.add_argument("--out", default=str(REPO / "results/c29_neural_rescore.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = json.loads(pathlib.Path(a.nbest).read_text())
    refs, lists = d["refs"], d["nbest"]
    a_scale = d["config"]["acoustic_scale"]
    print(f"{len(lists)} trials from {a.nbest}")
    print(f"  1-best {d['wer_1best']:.2f} %   oracle {d['wer_oracle']:.2f} %   "
          f"headroom {d['wer_1best']-d['wer_oracle']:.2f} pts")

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model).to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {a.model}: {n_par/1e6:.0f} M params on {device}")

    if a.top_k:
        # The list is already ranked by the decoder, so truncation keeps the best candidates.
        lists = [c[: a.top_k] for c in lists]
        print(f"  top_k={a.top_k}: mean list {sum(len(c) for c in lists)/len(lists):.1f}")

    flat, owner = [], []
    for t, cands in enumerate(lists):
        for c in cands:
            flat.append(c["s"])
            owner.append(t)
    print(f"  scoring {len(flat)} candidates (mean {len(flat)/len(lists):.1f}/trial)")

    t0 = time.time()
    neural = score_sentences(model, tok, flat, device, a.batch_size)
    torch.cuda.synchronize() if device.type == "cuda" else None
    total_s = time.time() - t0
    print(f"  scored in {total_s:.1f} s  ({1000*total_s/len(lists):.1f} ms/trial amortized)")

    # Per-utterance latency is the deployable number: one batched forward over one trial's list.
    per_trial = []
    for t in range(min(50, len(lists))):
        s = [c["s"] for c in lists[t]]
        if not s:
            continue
        t1 = time.time()
        score_sentences(model, tok, s, device, a.batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_trial.append((time.time() - t1) * 1000)
    per_trial = np.array(per_trial)
    print(f"  per-utterance forward: p50 {np.percentile(per_trial,50):.1f} / "
          f"p95 {np.percentile(per_trial,95):.1f} / max {per_trial.max():.1f} ms  (n=50)")

    train = {" ".join(norm_words(l)) for l in
             pathlib.Path(a.train_sentences).read_text().split("\n") if l.strip()}
    seen = [" ".join(norm_words(r)) in train for r in refs]

    scored, ref_words = [], []
    k = 0
    for t, (ref, cands) in enumerate(zip(refs, lists)):
        rw = norm_words(ref)
        ref_words.append(rw)
        rows = []
        for c in cands:
            hw = norm_words(c["s"])
            # Keep the acoustic and the 4-gram terms SEPARATE so gamma can down-weight or remove
            # the n-gram. Adding a neural score on top of a full-weight n-gram double-counts the
            # language model; BrainSpeechDecoder::Rescore (brain_speech_decoder.cc:65-72) subtracts
            # the original LM before adding the new one, and gamma=0 reproduces that.
            rows.append((a_scale * c["ac"], float(c["lm"]), float(neural[k]), float(len(hw)),
                         levenshtein(rw, hw)))
            k += 1
        scored.append(rows)

    def wer_at(alpha, beta, gamma=1.0, subset=None):
        e = w = 0
        for i, (rows, rw) in enumerate(zip(scored, ref_words)):
            if subset is not None and not subset[i]:
                continue
            w += len(rw)
            if not rows:
                e += len(rw)
                continue
            e += max(rows, key=lambda r: r[0] + gamma * r[1] + alpha * r[2] + beta * r[3])[4]
        return 100.0 * e / max(w, 1)

    base = wer_at(0.0, 0.0, 1.0)
    ok = abs(base - d["wer_1best"]) < 1e-6
    print(f"\nself-check alpha=beta=0 -> {base:.4f} % vs dumped {d['wer_1best']:.4f} % "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        return 1

    rows = []
    print(f"\n{'gamma':>7} {'alpha':>7} {'beta':>7} {'WER %':>8} {'vs 1-best':>10}")
    print("-" * 43)
    for ga in [float(x) for x in a.gammas.split(",")]:
        for al in [float(x) for x in a.alphas.split(",")]:
            for be in [float(x) for x in a.betas.split(",")]:
                w = wer_at(al, be, ga)
                rows.append({"gamma": ga, "alpha": al, "beta": be, "wer_percent": w})
                print(f"{ga:>7.2f} {al:>7.2f} {be:>7.2f} {w:>8.2f} {w-base:>+10.2f}", flush=True)

    best = min(rows, key=lambda r: r["wer_percent"])
    unseen = [not s for s in seen]
    ab, bb, gb = best["alpha"], best["beta"], best["gamma"]
    gap = base - d["wer_oracle"]
    print("\n" + "=" * 62)
    print(f"  1-best (4-gram only)  : {base:.2f} %")
    print(f"  + {a.model} rescoring : {best['wer_percent']:.2f} %  (gamma={gb}, alpha={ab}, beta={bb})")
    print(f"  oracle (ceiling)      : {d['wer_oracle']:.2f} %")
    print(f"  recovered             : {base-best['wer_percent']:.2f} of {gap:.2f} pts "
          f"= {100*(base-best['wer_percent'])/max(gap,1e-9):.0f} % of the headroom")
    print("-" * 62)
    for tag, sub in (("unseen", unseen), ("seen", seen)):
        print(f"  {tag:>6} ({sum(sub):>4} trials): {wer_at(0,0,1.0,sub):.2f} -> "
              f"{wer_at(ab,bb,gb,sub):.2f} %")
    print("=" * 62)

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps({
            "model": a.model, "n_params": n_par, "nbest_source": str(a.nbest),
            "wer_1best": base, "wer_oracle": d["wer_oracle"], "best": best, "grid": rows,
            "wer_unseen_1best": wer_at(0, 0, 1.0, unseen),
            "wer_unseen_rescored": wer_at(ab, bb, gb, unseen),
            "latency_ms_per_utterance": {"p50": float(np.percentile(per_trial, 50)),
                                         "p95": float(np.percentile(per_trial, 95)),
                                         "max": float(per_trial.max())},
        }, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
