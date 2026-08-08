"""C29-analysis — why 71 % of the oracle headroom survives every rescorer.

The question this answers
------------------------
At the deliverable operating point (C28 4-gram in the beam, gpt2-large rescoring at gamma=0.25,
alpha=1.0) val-dev is 6.89 % against a 2.88 % oracle over the same 100-best. Every lever tried on
the rescorer -- 6x the parameters, in-domain fine-tuning -- plateaus at ~29 % of that headroom.
Before spending on shallow fusion, establish WHICH of two worlds we are in:

    (A) the neural LM prefers the correct candidate but is out-voted by the acoustic term
        -> a re-weighting problem; a rescorer can still win, and fusion inherits the same fix
    (B) the correct and incorrect candidates are ~equiprobable under English
        -> no rescorer of any size can win; the acoustic scores are the binding constraint

These make opposite predictions about the score decomposition of the surviving errors, so one
pass over the cached n-best settles it. No decoding, no training -- one gpt2-large forward over
38 k candidates (36 s), cached to .npz so re-analysis is free.

Method
------
The decoder ranks by `acoustic_scale*ac + lm` (brain_speech_decoder.cc:123-124); rescoring adds a
neural term with the 4-gram down-weighted by gamma (neural_rescore.py:191):

    S(c) = acoustic_scale*ac(c) + gamma*lm(c) + alpha*neural(c) + beta*|c|

For every trial, `pick` is the argmax of S and `best` is the minimum-edit-distance candidate
(ties broken by max S, so the gap reported is the SMALLEST one that would have to be closed --
a lower bound, which is the conservative direction for world (A)). Where edit(pick) > edit(best)
the trial is *selection-limited*: the fix is in the list and the ranking missed it. Splitting
S(pick) - S(best) into its three terms says which term did the damage.

    cd model_training && ../.venv/bin/python -m benchmark.rescore_analysis
"""
import argparse
import difflib
import json
import pathlib
import sys

import numpy as np

from benchmark.neural_rescore import levenshtein, norm_words, score_sentences

REPO = pathlib.Path(__file__).resolve().parents[2]

# A candidate pair whose neural log-prob differs by less than this PER DIFFERING WORD is one the
# LM has no real opinion about: e^1 = 2.7x is inside the noise of any sentence-level LM score.
INDIFFERENT_NATS = 1.0


def neural_scores(nbest_path, model_name, device, batch_size, cache_path):
    """Total log P(candidate) under `model_name`, one per candidate, flattened trial-major."""
    d = json.loads(pathlib.Path(nbest_path).read_text())
    flat = [c["s"] for cands in d["nbest"] for c in cands]

    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if len(z["neural"]) == len(flat) and str(z["model"]) == model_name:
            print(f"  neural scores from cache {cache_path.name}")
            return d, z["neural"]
        print(f"  cache {cache_path.name} stale ({len(z['neural'])} vs {len(flat)}) -- rescoring")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(dev).eval()
    print(f"  scoring {len(flat)} candidates with {model_name} on {dev}")
    neural = score_sentences(model, tok, flat, dev, batch_size)
    np.savez(cache_path, neural=neural, model=np.array(model_name))
    print(f"  cached -> {cache_path}")
    return d, neural


def word_diff(a_words, b_words):
    """The (pick_span, best_span) pairs where two hypotheses disagree."""
    out = []
    sm = difflib.SequenceMatcher(a=a_words, b=b_words, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            out.append((" ".join(a_words[i1:i2]) or "-", " ".join(b_words[j1:j2]) or "-"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", default=str(REPO / "results/c29_oracle_4gram.json"))
    ap.add_argument("--model", default="gpt2-large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.25, help="deliverable operating point")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--n_examples", type=int, default=25)
    ap.add_argument("--out", default=str(REPO / "results/c29_error_analysis.json"))
    a = ap.parse_args()

    # Key the cache on BOTH the n-best file and the model: two n-best files of different depth
    # otherwise collide on the model name and evict each other on every switch.
    nb = pathlib.Path(a.nbest)
    cache = nb.parent / f"c29_neural_{nb.stem}_{a.model.replace('/', '_')}.npz"
    legacy = nb.parent / f"c29_neural_{a.model.replace('/', '_')}.npz"
    if legacy.exists() and not cache.exists() and nb.stem == "c29_oracle_4gram":
        legacy.rename(cache)
    d, neural = neural_scores(a.nbest, a.model, a.device, a.batch_size, cache)
    refs, lists, a_scale = d["refs"], d["nbest"], d["config"]["acoustic_scale"]
    print(f"{len(lists)} trials  1-best {d['wer_1best']:.2f} %  oracle {d['wer_oracle']:.2f} %")

    # ---- per-trial tables: acoustic term, 4-gram term, neural term, length, edits vs reference
    trials, k = [], 0
    for ref, cands in zip(refs, lists):
        rw = norm_words(ref)
        rows = []
        for c in cands:
            hw = norm_words(c["s"])
            rows.append({"s": c["s"], "w": hw, "ac": a_scale * c["ac"], "lm": float(c["lm"]),
                         "neu": float(neural[k]), "n": len(hw), "ed": levenshtein(rw, hw)})
            k += 1
        trials.append({"ref": ref, "rw": rw, "rows": rows})
    assert k == len(neural), f"{k} rows consumed vs {len(neural)} scores"

    def score(r, g, al, be):
        return r["ac"] + g * r["lm"] + al * r["neu"] + be * r["n"]

    def wer(g, al, be):
        e = w = 0
        for t in trials:
            w += len(t["rw"])
            e += max(t["rows"], key=lambda r: score(r, g, al, be))["ed"] if t["rows"] else len(t["rw"])
        return 100.0 * e / w

    base, op = wer(1.0, 0.0, 0.0), wer(a.gamma, a.alpha, a.beta)
    assert abs(base - d["wer_1best"]) < 1e-6, f"self-check {base} vs {d['wer_1best']}"
    print(f"  self-check gamma=1,alpha=0 -> {base:.4f} % OK")
    print(f"  operating point (gamma={a.gamma}, alpha={a.alpha}) -> {op:.2f} %")

    total_w = sum(len(t["rw"]) for t in trials)

    # ---- 1. is the operating point a coarse-grid artifact? fine 2-D sweep -------------------
    print("\n1. IS THE OPERATING POINT REAL? fine (gamma, alpha) sweep, beta=0")
    gammas = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0]
    grid = {}
    print("   g\\a  " + "".join(f"{al:>6.2f}" for al in alphas))
    for g in gammas:
        row = [wer(g, al, 0.0) for al in alphas]
        for al, w in zip(alphas, row):
            grid[f"{g},{al}"] = w
        print(f"  {g:>4.2f}  " + "".join(f"{w:>6.2f}" for w in row))
    gb, ab = min(grid, key=grid.get).split(",")
    print(f"   grid optimum: gamma={gb} alpha={ab} -> {grid[f'{gb},{ab}']:.2f} % "
          f"(operating point {op:.2f} %)")

    # ---- 2. partition the residual error ----------------------------------------------------
    # `best` = min-edit candidate, ties broken by max S: the cheapest one to have picked.
    for t in trials:
        if not t["rows"]:
            t["pick"], t["best"] = None, None
            continue
        t["pick"] = max(t["rows"], key=lambda r: score(r, a.gamma, a.alpha, a.beta))
        m = min(r["ed"] for r in t["rows"])
        t["best"] = max((r for r in t["rows"] if r["ed"] == m),
                        key=lambda r: score(r, a.gamma, a.alpha, a.beta))

    floor_e = sum(t["best"]["ed"] if t["best"] else len(t["rw"]) for t in trials)
    pick_e = sum(t["pick"]["ed"] if t["pick"] else len(t["rw"]) for t in trials)
    sel = [t for t in trials if t["pick"] and t["pick"]["ed"] > t["best"]["ed"]]
    print(f"\n2. WHERE THE RESIDUAL {op:.2f} % LIVES")
    print(f"   list-limited floor (oracle)     {100*floor_e/total_w:>6.2f} pts  "
          f"-- no rescorer can touch this; needs fusion or a deeper list")
    print(f"   selection-limited (recoverable) {100*(pick_e-floor_e)/total_w:>6.2f} pts  "
          f"over {len(sel)} of {len(trials)} trials")

    # ---- 3. decompose the score gap on the selection-limited trials -------------------------
    for t in sel:
        p, b = t["pick"], t["best"]
        t["d_ac"] = p["ac"] - b["ac"]
        t["d_lm"] = a.gamma * (p["lm"] - b["lm"])
        t["d_neu_raw"] = p["neu"] - b["neu"]          # unweighted: alpha-independent
        t["d_neu"] = a.alpha * t["d_neu_raw"]
        t["n_diff"] = max(1, levenshtein(p["w"], b["w"]))
        t["per_word"] = t["d_neu_raw"] / t["n_diff"]  # nats per differing word, pick minus best
        t["cost"] = p["ed"] - b["ed"]
        # alpha at which the neural term alone would flip this trial, holding gamma fixed
        t["alpha_flip"] = ((t["d_ac"] + t["d_lm"]) / -t["d_neu_raw"]
                           if t["d_neu_raw"] < 0 < t["d_ac"] + t["d_lm"] else np.inf)

    def bucket(t):
        if t["per_word"] < -INDIFFERENT_NATS:
            return "lm_right"        # LM prefers the better candidate, decisively
        if t["per_word"] > INDIFFERENT_NATS:
            return "lm_wrong"        # LM prefers the worse candidate, decisively
        return "lm_indifferent"      # within e^1 per differing word

    buckets = {name: [t for t in sel if bucket(t) == name]
               for name in ("lm_right", "lm_wrong", "lm_indifferent")}
    print(f"\n3. WHAT THE NEURAL LM THINKS ABOUT THE {len(sel)} SELECTION FAILURES")
    print(f"   (log P(pick) - log P(best), per differing word; |.| < {INDIFFERENT_NATS} nat = no opinion)")
    print(f"   {'bucket':<17}{'trials':>7}{'WER pts':>9}{'med nats/word':>15}{'med alpha_flip':>16}")
    summary = {}
    for name, ts in buckets.items():
        pts = 100 * sum(t["cost"] for t in ts) / total_w
        pw = float(np.median([t["per_word"] for t in ts])) if ts else float("nan")
        af = [t["alpha_flip"] for t in ts if np.isfinite(t["alpha_flip"])]
        afm = float(np.median(af)) if af else float("inf")
        summary[name] = {"trials": len(ts), "wer_pts": pts, "median_nats_per_word": pw,
                         "median_alpha_flip": afm, "n_flippable": len(af)}
        print(f"   {name:<17}{len(ts):>7}{pts:>9.2f}{pw:>15.2f}"
              f"{(f'{afm:.1f}' if np.isfinite(afm) else 'inf'):>16}")

    # ---- 3b. is the "better" candidate actually CORRECT, or merely edit-closer? -------------
    # The oracle is defined by minimum edit distance, which awards partial credit for candidates
    # that are nearer the reference while being worse English ("grable stag beetles"). No language
    # model can be asked to prefer those, so headroom sitting on such pairs is not LM-reachable.
    exact = [t for t in sel if t["best"]["ed"] == 0]
    both_wrong = [t for t in sel if t["best"]["ed"] > 0]
    print("\n3b. IS THE MISSED CANDIDATE ACTUALLY CORRECT, OR JUST EDIT-CLOSER?")
    print(f"   {'':<24}{'trials':>7}{'WER pts':>9}{'lm_right':>10}{'lm_wrong':>10}{'lm_indiff':>11}")
    tab = {}
    for name, ts in (("exact match available", exact), ("both hyps wrong", both_wrong)):
        pts = 100 * sum(t["cost"] for t in ts) / total_w
        by = {b: sum(t["cost"] for t in ts if bucket(t) == b) * 100 / total_w
              for b in ("lm_right", "lm_wrong", "lm_indifferent")}
        tab[name] = {"trials": len(ts), "wer_pts": pts, "by_bucket_pts": by}
        print(f"   {name:<24}{len(ts):>7}{pts:>9.2f}{by['lm_right']:>10.2f}"
              f"{by['lm_wrong']:>10.2f}{by['lm_indifferent']:>11.2f}")

    # Which term is actually carrying the wrong decision?
    ac_dom = [t for t in sel if t["d_ac"] > t["d_lm"] + t["d_neu"]]
    print(f"\n   acoustic term is the largest contributor to the wrong pick in "
          f"{len(ac_dom)}/{len(sel)} ({100*len(ac_dom)/max(len(sel),1):.0f} %) of failures")
    print(f"   median |d_ac| {np.median([abs(t['d_ac']) for t in sel]):.2f} vs "
          f"|d_neu| {np.median([abs(t['d_neu']) for t in sel]):.2f} "
          f"(alpha={a.alpha}) -- the acoustic term outweighs the neural one by "
          f"{np.median([abs(t['d_ac']) for t in sel])/max(np.median([abs(t['d_neu']) for t in sel]),1e-9):.1f}x")

    # ---- 4. what a perfect rescorer could claim, and what raising alpha actually does -------
    print("\n4. THE CEILING FOR *ANY* RESCORER ON THIS LIST")
    # An oracle over the neural term alone: pick by neural, ignore acoustics entirely.
    print(f"   pick by neural only (alpha->inf)   {wer(0.0, 1.0, 0.0):>6.2f} %")
    print(f"   pick by 4-gram only                {wer(1.0, 0.0, 0.0):>6.2f} %  (= 1-best)")
    print(f"   oracle over the list               {d['wer_oracle']:>6.2f} %")
    print("\n   raising alpha: trials fixed vs broken against the operating point")
    print(f"   {'alpha':>7}{'WER %':>8}{'fixed':>8}{'broken':>8}")
    op_ed = [t["pick"]["ed"] if t["pick"] else len(t["rw"]) for t in trials]
    alpha_trace = []
    for al in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0]:
        fixed = broken = 0
        e = 0
        for t, base_ed in zip(trials, op_ed):
            if not t["rows"]:
                e += len(t["rw"])
                continue
            ed = max(t["rows"], key=lambda r: score(r, a.gamma, al, a.beta))["ed"]
            e += ed
            fixed += ed < base_ed
            broken += ed > base_ed
        w = 100.0 * e / total_w
        alpha_trace.append({"alpha": al, "wer": w, "fixed": fixed, "broken": broken})
        print(f"   {al:>7.1f}{w:>8.2f}{fixed:>8}{broken:>8}")

    # ---- 5. qualitative sample ---------------------------------------------------------------
    print(f"\n5. THE {a.n_examples} COSTLIEST SELECTION FAILURES")
    ex = sorted(sel, key=lambda t: -t["cost"])[:a.n_examples]
    dump = []
    for t in ex:
        diffs = word_diff(t["pick"]["w"], t["best"]["w"])
        dump.append({"ref": t["ref"], "pick": t["pick"]["s"], "best": t["best"]["s"],
                     "cost_words": t["cost"], "d_ac": t["d_ac"], "d_lm": t["d_lm"],
                     "d_neu": t["d_neu"], "nats_per_word": t["per_word"],
                     "bucket": bucket(t), "diffs": diffs})
        print(f"\n   ref  : {t['ref']}")
        print(f"   pick : {t['pick']['s']}")
        print(f"   best : {t['best']['s']}")
        print(f"   +{t['cost']} err  d_ac {t['d_ac']:+8.2f}  d_lm {t['d_lm']:+7.2f}  "
              f"d_neu {t['d_neu']:+7.2f}  {t['per_word']:+.2f} nats/word  [{bucket(t)}]")
        print(f"   diff : {'; '.join(f'{p} -> {b}' for p, b in diffs)}")

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps({
            "model": a.model, "nbest_source": a.nbest,
            "operating_point": {"gamma": a.gamma, "alpha": a.alpha, "beta": a.beta, "wer": op},
            "wer_1best": base, "wer_oracle": d["wer_oracle"],
            "grid": grid, "grid_optimum": {"gamma": float(gb), "alpha": float(ab),
                                           "wer": grid[f"{gb},{ab}"]},
            "residual": {"list_limited_pts": 100 * floor_e / total_w,
                         "selection_limited_pts": 100 * (pick_e - floor_e) / total_w,
                         "n_selection_failures": len(sel), "n_trials": len(trials)},
            "buckets": summary, "reachability": tab,
            "acoustic_dominant_frac": len(ac_dom) / max(len(sel), 1),
            "median_abs_d_ac": float(np.median([abs(t["d_ac"]) for t in sel])),
            "median_abs_d_neu": float(np.median([abs(t["d_neu"]) for t in sel])),
            "wer_neural_only": wer(0.0, 1.0, 0.0),
            "alpha_trace": alpha_trace, "examples": dump,
        }, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
