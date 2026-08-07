"""C13 — incremental WFST decoding: the constrained baseline and its per-frame cost.

Answers the two measurements that gate every WER number in PLAN.md:

    Q1  what does the incremental WFST beam search cost per frame?
    Q2  what WER does incremental-n-gram-only decoding reach, with Rescore()/augment_nbest()/OPT
        all deleted?

The acoustic model never runs here — everything reads cached val logits.

TWO ENVIRONMENTS, BY NECESSITY
------------------------------
`lm_decoder` is built for python3.9 / numpy 1.24 (setup_lm.sh pins torch==1.13.1, which caps
numpy below 2). The val_metrics.pkl written by training uses numpy 2.x and cannot be unpickled
there. The .npy *format* is stable across both, so the two stages talk through a cache file:

    # stage 1 — in .venv (numpy 2.x)
    ../.venv/bin/python benchmark/stream_lm.py export \
        --val_metrics results/causal_la0/checkpoint/val_metrics.pkl \
        --out results/la0_logits.npz

    # stage 2 — in the LM env (numpy 1.24, has lm_decoder)
    $(conda info --base)/envs/b2txt25_lm/bin/python benchmark/stream_lm.py decode \
        --cache results/la0_logits.npz --out results/stream_lm_la0.json

Stage 2 deliberately does NOT pass --rescore or --do_opt and uses nbest=1: those three stages are
the 620-830 ms batch wall this measurement exists to remove.
"""
import argparse
import json
import math
import re
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_LM = REPO / "language_model/pretrained_language_models/openwebtext_1gram_lm_sil"


def rp(path):
    """Resolve a path against the cwd first, then the repo root — so the commands work
    whether you run them from model_training/ or from the repo root."""
    p = pathlib.Path(path)
    if p.is_absolute():
        return p
    return p if p.exists() else (REPO / p)


# ----------------------------------------------------------------------------------------
# stage 1 — export (runs where numpy 2.x can read the pickle)
# ----------------------------------------------------------------------------------------
def cmd_export(args):
    import pickle

    with open(rp(args.val_metrics), "rb") as fh:
        m = pickle.load(fh)

    logits, texts, days = [], [], []
    for batch_logits, batch_lens, batch_tx, batch_day in zip(
            m["logits"], m["n_time_steps"], m["transcription"], m["day_indicies"]):
        bd = np.asarray(batch_day).ravel()
        for i in range(batch_logits.shape[0]):
            days.append(int(bd[i]))
            x = batch_logits[i, : batch_lens[i], :].astype(np.float32)
            # [BLANK, phonemes..., SIL] -> [BLANK, SIL, phonemes...], matching
            # evaluate_model_helpers.rearrange_speech_logits_pt (:79-83). Applied ONCE, here.
            logits.append(np.concatenate((x[:, 0:1], x[:, -1:], x[:, 1:-1]), axis=-1))
            texts.append(bytes(batch_tx[i][batch_tx[i] > 0].astype(np.uint8)).decode("ascii").strip())

    out = rp(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        n_trials=np.int64(len(logits)),
        lengths=np.array([a.shape[0] for a in logits], dtype=np.int64),
        days=np.array(days, dtype=np.int64),
        flat=np.concatenate(logits, axis=0),
        avg_PER=np.float64(m.get("avg_PER", np.nan)),
    )
    # Transcriptions go to a plain sidecar, NOT into the npz: a dtype=object array is
    # serialized by pickle, which reintroduces the numpy 2.x / 1.24 incompatibility this
    # cache exists to avoid. One line per trial, newlines stripped.
    out.with_suffix(".txt").write_text("\n".join(t.replace("\n", " ") for t in texts))
    frames = sum(a.shape[0] for a in logits)
    print(f"exported {len(logits)} trials / {frames} frames -> {out}  "
          f"({out.stat().st_size/1e6:.1f} MB)")
    print(f"  source avg_PER {m.get('avg_PER', float('nan')):.5f}")


def load_cache(path):
    path = pathlib.Path(path)
    z = np.load(path)                      # no allow_pickle: the cache is pure arrays
    lens = z["lengths"]
    flat = z["flat"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    trials = [flat[offs[i]:offs[i + 1]] for i in range(len(lens))]
    texts = path.with_suffix(".txt").read_text().split("\n")
    if len(texts) != len(trials):
        raise ValueError(f"cache mismatch: {len(trials)} trials but {len(texts)} transcriptions")
    days = z["days"] if "days" in z.files else np.full(len(trials), -1, dtype=np.int64)
    return trials, texts, days


# ----------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------
VAL_TEST_DAYS = (39, 40, 41, 42, 43, 44)   # see model_training/splits.py


def apply_split(trials, texts, days, split):
    """dev = tune here | test = touch ONCE per group | all = both (do not tune on this)."""
    if split == "all":
        return trials, texts
    keep = [i for i in range(len(trials))
            if (days[i] in VAL_TEST_DAYS) == (split == "test")]
    if days[0] == -1:
        raise ValueError("cache has no day indices — re-run `export` to add them")
    return [trials[i] for i in keep], [texts[i] for i in keep]


_PUNCT = re.compile(r"[^\w\s']")


def norm_words(text):
    """Standard ASR text normalization: lowercase, drop punctuation (keeping apostrophes so
    contractions stay one token), collapse whitespace.

    This matters more than it looks. 98% of the reference transcriptions carry punctuation while
    the decoder emits none, so scoring on `.lower().split()` alone made "well." vs "well" a miss
    and inflated WER by 9.2 points (51.65 -> 42.42 on the 1-gram baseline).
    """
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


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


# ----------------------------------------------------------------------------------------
# stage 2 — incremental decode (runs where lm_decoder is importable)
# ----------------------------------------------------------------------------------------
def run_decode(lm_decoder, lm_dir, trials, texts, *, acoustic_scale, blank_penalty, beam,
               lattice_beam, max_active, min_active, length_penalty, blank_skip_thresh,
               capture_partials=False, progress=0, temperature=1.0):
    """One full incremental decode pass. Returns a metrics dict. No rescore, no OPT, nbest=1.

    C10: `temperature` divides the logits before DecodeNumpy's internal log_softmax
    (lm_decoder.cc:31). It is NOT a reparameterization of acoustic_scale -- scaling log-probs
    shifts every path by the same per-frame constant, while scaling logits renormalizes and
    genuinely reshapes the posterior. The posterior here is near one-hot (mean entropy 0.9 % of
    maximum), which starves the beam of alternatives, so this is the free test of whether
    calibration is the binding problem.
    """
    import numpy as _np
    opts = lm_decoder.DecodeOptions(max_active, min_active, beam, lattice_beam,
                                    acoustic_scale, blank_skip_thresh, length_penalty, 1)
    res = lm_decoder.DecodeResource(str(pathlib.Path(lm_dir) / "TLG.fst"), "", "",
                                    str(pathlib.Path(lm_dir) / "words.txt"), "")
    dec = lm_decoder.BrainSpeechDecoder(res, opts)
    log_bp = float(_np.log(blank_penalty))

    frame_ms, result_ms, finish_ms, hyps = [], [], [], []
    edits = words = 0
    t_start = time.time()
    for idx, (lg, ref) in enumerate(zip(trials, texts)):
        if temperature != 1.0:
            lg = lg / temperature
        dec.Reset()
        for f in range(lg.shape[0]):
            row = _np.ascontiguousarray(lg[f:f + 1])
            t0 = time.perf_counter()
            lm_decoder.DecodeNumpy(dec, row, _np.zeros_like(row), log_bp)
            frame_ms.append((time.perf_counter() - t0) * 1000.0)
            if capture_partials:
                t0 = time.perf_counter(); dec.result()
                result_ms.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        dec.FinishDecoding(); r = dec.result()
        finish_ms.append((time.perf_counter() - t0) * 1000.0)
        hyp = r[0].sentence.strip() if r else ""
        hyps.append(hyp)
        rw, hw = norm_words(ref), norm_words(hyp)
        edits += levenshtein(rw, hw); words += len(rw)
        if progress and (idx + 1) % progress == 0:
            print(f"  {idx+1}/{len(trials)} trials  running WER "
                  f"{100.0*edits/max(words,1):.2f}%", flush=True)

    fm = _np.array(frame_ms); fin = _np.array(finish_ms)
    out = {
        "config": {"acoustic_scale": acoustic_scale, "blank_penalty": blank_penalty,
                   "beam": beam, "max_active": max_active,
                   "blank_skip_thresh": blank_skip_thresh, "temperature": temperature,
                   "nbest": 1, "rescore": False, "opt": False},
        "n_trials": len(trials), "n_frames": int(fm.size),
        "wer_percent": 100.0 * edits / max(words, 1), "edits": edits, "ref_words": words,
        "per_frame_ms": {"mean": float(fm.mean()), "p50": pct(fm, 50), "p95": pct(fm, 95),
                         "p99": pct(fm, 99), "max": float(fm.max())},
        "finalize_ms": {"p50": pct(fin, 50), "p95": pct(fin, 95), "max": float(fin.max())},
        "wall_sec": time.time() - t_start,
        "hyps": hyps,
    }
    if result_ms:
        rm = _np.array(result_ms)
        out["partial_extract_ms"] = {"p50": pct(rm, 50), "p95": pct(rm, 95)}
    return out


def cmd_sweep(args):
    """C15 — coordinate-wise decode sweep. acoustic_scale x blank_penalty first (they interact
    strongly), then beam/max_active at the winner. Tunes on val-dev only."""
    import lm_decoder
    trials, texts, days = load_cache(rp(args.cache))
    trials, texts = apply_split(trials, texts, days, args.split)
    if args.n_trials:
        trials, texts = trials[: args.n_trials], texts[: args.n_trials]
    print(f"sweeping on split={args.split}, {len(trials)} trials\n")

    scales = [float(x) for x in args.acoustic_scales.split(",")]
    penalties = [float(x) for x in args.blank_penalties.split(",")]
    rows = []
    print(f"{'ac_scale':>9} {'blank_pen':>10} {'WER %':>8} {'p95 ms':>8} {'wall s':>7}")
    print("-" * 46)
    for a in scales:
        for b in penalties:
            r = run_decode(lm_decoder, rp(args.lm), trials, texts,
                           acoustic_scale=a, blank_penalty=b, beam=args.beam,
                           lattice_beam=args.lattice_beam, max_active=args.max_active,
                           min_active=args.min_active, length_penalty=args.length_penalty,
                           blank_skip_thresh=args.blank_skip_thresh)
            r.pop("hyps")
            rows.append(r)
            print(f"{a:>9.3f} {b:>10.1f} {r['wer_percent']:>8.2f} "
                  f"{r['per_frame_ms']['p95']:>8.3f} {r['wall_sec']:>7.1f}", flush=True)

    rows.sort(key=lambda r: r["wer_percent"])
    best = rows[0]["config"]
    print("\n" + "=" * 46)
    print(f"BEST: acoustic_scale={best['acoustic_scale']}  blank_penalty={best['blank_penalty']}"
          f"  -> {rows[0]['wer_percent']:.2f} % WER")
    print(f"spread across grid: {rows[0]['wer_percent']:.2f} - {rows[-1]['wer_percent']:.2f} %"
          f"  ({rows[-1]['wer_percent'] - rows[0]['wer_percent']:.2f} pts)")
    print("=" * 46)
    if args.out:
        q = rp(args.out); q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps({"split": args.split, "n_trials": len(trials),
                                 "results": rows}, indent=2))
        print(f"wrote {q}")


def cmd_decode(args):
    import lm_decoder

    trials, texts, days = load_cache(rp(args.cache))
    trials, texts = apply_split(trials, texts, days, args.split)
    if args.n_trials:
        trials, texts = trials[: args.n_trials], texts[: args.n_trials]

    opts = lm_decoder.DecodeOptions(
        args.max_active, args.min_active, args.beam, args.lattice_beam,
        args.acoustic_scale, args.blank_skip_thresh, args.length_penalty, 1,  # nbest=1
    )
    res = lm_decoder.DecodeResource(
        str(rp(args.lm) / "TLG.fst"), "", "", str(rp(args.lm) / "words.txt"), "")
    decoder = lm_decoder.BrainSpeechDecoder(res, opts)
    log_bp = float(np.log(args.blank_penalty))

    frame_ms, result_ms, finish_ms = [], [], []
    edits = words = 0
    hyps = []
    t_start = time.time()

    for idx, (lg, ref) in enumerate(zip(trials, texts)):
        decoder.Reset()
        for f in range(lg.shape[0]):
            row = np.ascontiguousarray(lg[f:f + 1])
            t0 = time.perf_counter()
            lm_decoder.DecodeNumpy(decoder, row, np.zeros_like(row), log_bp)
            frame_ms.append((time.perf_counter() - t0) * 1000.0)
            if args.capture_partials:
                t0 = time.perf_counter()
                r = decoder.result()
                result_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        decoder.FinishDecoding()
        r = decoder.result()
        finish_ms.append((time.perf_counter() - t0) * 1000.0)

        hyp = r[0].sentence.strip() if r else ""
        hyps.append(hyp)
        rw, hw = norm_words(ref), norm_words(hyp)
        edits += levenshtein(rw, hw)
        words += len(rw)

        if args.progress and (idx + 1) % args.progress == 0:
            print(f"  {idx+1}/{len(trials)} trials  running WER {100.0*edits/max(words,1):.2f}%",
                  flush=True)

    frame_ms = np.array(frame_ms)
    finish_ms = np.array(finish_ms)
    wer = 100.0 * edits / max(words, 1)

    out = {
        "config": {
            "lm": str(args.lm), "acoustic_scale": args.acoustic_scale,
            "blank_penalty": args.blank_penalty, "beam": args.beam,
            "max_active": args.max_active, "blank_skip_thresh": args.blank_skip_thresh,
            "nbest": 1, "rescore": False, "opt": False,
        },
        "n_trials": len(trials), "n_frames": int(frame_ms.size),
        "wer_percent": wer, "edits": edits, "ref_words": words,
        "per_frame_ms": {"mean": float(frame_ms.mean()), "p50": pct(frame_ms, 50),
                         "p95": pct(frame_ms, 95), "p99": pct(frame_ms, 99),
                         "max": float(frame_ms.max())},
        "finalize_ms": {"p50": pct(finish_ms, 50), "p95": pct(finish_ms, 95),
                        "max": float(finish_ms.max())},
        "wall_sec": time.time() - t_start,
    }
    if result_ms:
        rm = np.array(result_ms)
        out["partial_extract_ms"] = {"p50": pct(rm, 50), "p95": pct(rm, 95)}

    budget = 79.0  # ms left for the LM after acoustic (BUDGET.md)
    print("\n" + "=" * 66)
    print(f"  Q2  constrained-baseline WER : {wer:.2f} %   ({len(trials)} trials)")
    print(f"  Q1  per-frame WFST cost      : p50 {out['per_frame_ms']['p50']:.3f}  "
          f"p95 {out['per_frame_ms']['p95']:.3f}  max {out['per_frame_ms']['max']:.3f} ms")
    print(f"      vs {budget:.0f} ms budget        : "
          f"{'PASS' if out['per_frame_ms']['p95'] <= budget else 'FAIL'} "
          f"({budget/max(out['per_frame_ms']['p95'],1e-9):.0f}x margin at p95)")
    print(f"      finalize per trial       : p50 {out['finalize_ms']['p50']:.2f}  "
          f"p95 {out['finalize_ms']['p95']:.2f} ms")
    print("=" * 66)

    if args.out:
        p = rp(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        if args.save_hyps:
            rp(args.save_hyps).write_text(
                "\n".join(f"{r}\t{h}" for r, h in zip(texts, hyps)))
        print(f"wrote {p}")


def cmd_temp(args):
    """C10 — temperature sweep on the cached logits. Free: no retrain, no FST change."""
    import lm_decoder
    trials, texts, days = load_cache(rp(args.cache))
    trials, texts = apply_split(trials, texts, days, args.split)
    if args.n_trials:
        trials, texts = trials[: args.n_trials], texts[: args.n_trials]
    print(f"temperature sweep on split={args.split}, {len(trials)} trials, "
          f"acoustic_scale={args.acoustic_scale} blank_penalty={args.blank_penalty}\n")

    rows = []
    print(f"{'T':>6} {'WER %':>8} {'p95 ms':>8} {'wall s':>7}")
    print("-" * 33)
    for t in [float(x) for x in args.temperatures.split(",")]:
        r = run_decode(lm_decoder, rp(args.lm), trials, texts,
                       acoustic_scale=args.acoustic_scale, blank_penalty=args.blank_penalty,
                       beam=args.beam, lattice_beam=args.lattice_beam,
                       max_active=args.max_active, min_active=args.min_active,
                       length_penalty=args.length_penalty,
                       blank_skip_thresh=args.blank_skip_thresh, temperature=t)
        r.pop("hyps")
        rows.append(r)
        print(f"{t:>6.2f} {r['wer_percent']:>8.2f} {r['per_frame_ms']['p95']:>8.3f} "
              f"{r['wall_sec']:>7.1f}", flush=True)

    base = next(r for r in rows if r["config"]["temperature"] == 1.0)
    best = min(rows, key=lambda r: r["wer_percent"])
    print("\n" + "=" * 52)
    print(f"  T=1.0 (uncalibrated) : {base['wer_percent']:.2f} %")
    print(f"  best T={best['config']['temperature']:<12.2f}: {best['wer_percent']:.2f} %  "
          f"({best['wer_percent'] - base['wer_percent']:+.2f} pts)")
    print("=" * 52)
    if args.out:
        q = rp(args.out); q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps({"split": args.split, "n_trials": len(trials),
                                 "results": rows}, indent=2))
        print(f"wrote {q}")


# ----------------------------------------------------------------------------------------
# C28-A — n-best capture, and re-ranking with the in-domain trigram
# ----------------------------------------------------------------------------------------
def cmd_nbest(args):
    """Decode at nbest=N and dump the lists with their scores.

    Split from `rescore` on purpose: the decode is the expensive half and the re-ranking sweep
    should cost nothing to repeat. Also prices the n-best finalization, which is a real latency
    item — nbest>1 builds a lattice in FinishDecoding that nbest=1 skips entirely.
    """
    import lm_decoder

    trials, texts, days = load_cache(rp(args.cache))
    trials, texts = apply_split(trials, texts, days, args.split)
    if args.n_trials:
        trials, texts = trials[: args.n_trials], texts[: args.n_trials]
    print(f"nbest={args.nbest} on split={args.split}, {len(trials)} trials\n")

    opts = lm_decoder.DecodeOptions(
        args.max_active, args.min_active, args.beam, args.lattice_beam,
        args.acoustic_scale, args.blank_skip_thresh, args.length_penalty, args.nbest)
    res = lm_decoder.DecodeResource(
        str(rp(args.lm) / "TLG.fst"), "", "", str(rp(args.lm) / "words.txt"), "")
    decoder = lm_decoder.BrainSpeechDecoder(res, opts)
    log_bp = float(np.log(args.blank_penalty))

    lists, finish_ms, sizes = [], [], []
    edits_1best = edits_oracle = words = 0
    t_start = time.time()

    for idx, (lg, ref) in enumerate(zip(trials, texts)):
        decoder.Reset()
        for f in range(lg.shape[0]):
            row = np.ascontiguousarray(lg[f:f + 1])
            lm_decoder.DecodeNumpy(decoder, row, np.zeros_like(row), log_bp)

        t0 = time.perf_counter()
        decoder.FinishDecoding()
        r = decoder.result()
        finish_ms.append((time.perf_counter() - t0) * 1000.0)

        cands = [{"s": h.sentence.strip(), "ac": float(h.ac_score), "lm": float(h.lm_score)}
                 for h in r]
        lists.append(cands)
        sizes.append(len(cands))

        rw = norm_words(ref)
        d = [levenshtein(rw, norm_words(c["s"])) for c in cands] or [len(rw)]
        edits_1best += d[0]
        edits_oracle += min(d)
        words += len(rw)

        if args.progress and (idx + 1) % args.progress == 0:
            print(f"  {idx+1}/{len(trials)} trials  1-best {100.0*edits_1best/max(words,1):.2f}%"
                  f"  oracle {100.0*edits_oracle/max(words,1):.2f}%", flush=True)

    fin = np.array(finish_ms)
    wer_1best = 100.0 * edits_1best / max(words, 1)
    wer_oracle = 100.0 * edits_oracle / max(words, 1)

    print("\n" + "=" * 66)
    print(f"  1-best WER   : {wer_1best:.2f} %")
    print(f"  oracle WER   : {wer_oracle:.2f} %   (headroom {wer_1best - wer_oracle:.2f} pts)")
    print(f"  list size    : mean {np.mean(sizes):.1f}  min {min(sizes)}  max {max(sizes)}")
    print(f"  finalize/trial: p50 {pct(fin,50):.2f}  p95 {pct(fin,95):.2f}  "
          f"max {fin.max():.2f} ms")
    print(f"  wall         : {time.time()-t_start:.1f} s")
    print("=" * 66)

    out = rp(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"lm": str(args.lm), "acoustic_scale": args.acoustic_scale,
                   "blank_penalty": args.blank_penalty, "beam": args.beam,
                   "lattice_beam": args.lattice_beam, "max_active": args.max_active,
                   "blank_skip_thresh": args.blank_skip_thresh, "nbest": args.nbest},
        "split": args.split, "n_trials": len(trials),
        "wer_1best": wer_1best, "wer_oracle": wer_oracle, "ref_words": words,
        "list_size": {"mean": float(np.mean(sizes)), "min": int(min(sizes)),
                      "max": int(max(sizes))},
        "finalize_ms": {"p50": pct(fin, 50), "p95": pct(fin, 95), "max": float(fin.max())},
        "refs": texts, "nbest": lists,
    }))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


def cmd_rescore(args):
    """C28-A — re-rank the dumped n-best with a trigram trained on the TRAIN transcriptions.

    The decoder ranks by `lm_score + acoustic_scale * ac_score` (brain_speech_decoder.cc:123-124
    against ctc_wfst_beam_search.cc:156). Keeping that as the base and adding the trigram means
    alpha=beta=0 must reproduce the 1-best exactly — which is the harness's own self-check.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from indomain_lm import Trigram, read_sentences_file

    d = json.loads(rp(args.nbest).read_text())
    refs, lists = d["refs"], d["nbest"]
    a_scale = d["config"]["acoustic_scale"]
    print(f"loaded {len(lists)} lists from {args.nbest} "
          f"(1-best {d['wer_1best']:.2f} %, oracle {d['wer_oracle']:.2f} %)")

    sents = read_sentences_file(rp(args.sentences))
    lm = Trigram(args.discount).train(sents)
    print(f"trigram: {len(sents)} TRAIN sentences, {lm.total} tokens, {len(lm.vocab)} types")

    # Coverage — the honest limit on this whole experiment. A trigram over 50k words cannot
    # score what it has never seen, so report the miss rate before reporting the WER.
    ref_toks = [w for r in refs for w in norm_words(r)]
    oov = sum(w not in lm.vocab for w in ref_toks)
    print(f"reference-token OOV under the TRAIN trigram: {100.0*oov/max(len(ref_toks),1):.2f} % "
          f"({oov}/{len(ref_toks)})")

    # The validity check that decides whether the headline number means anything: some val
    # sentences are prompted from the same pool as train, and for those the trigram has simply
    # memorized the answer. Score the unseen subset separately — that is the reportable number.
    train_set = {" ".join(norm_words(s)) for s in sents}
    seen = [" ".join(norm_words(r)) in train_set for r in refs]
    print(f"val sentences appearing VERBATIM in train: {sum(seen)}/{len(refs)} "
          f"({100.0*sum(seen)/max(len(refs),1):.2f} %) — scored separately below")

    # Score every candidate once; the sweep is then pure arithmetic.
    LN10 = math.log(10.0)
    scored, ref_words = [], []
    for ref, cands in zip(refs, lists):
        rw = norm_words(ref)
        ref_words.append(rw)
        rows = []
        for c in cands:
            hw = norm_words(c["s"])
            lp, n = lm.logprob(hw)
            rows.append((c["lm"] + a_scale * c["ac"], lp * LN10, float(n),
                         levenshtein(rw, hw)))
        scored.append(rows)

    def wer_at(alpha, beta, subset=None):
        e = w = 0
        for i, (rows, rw) in enumerate(zip(scored, ref_words)):
            if subset is not None and not subset[i]:
                continue
            w += len(rw)
            if not rows:
                e += len(rw)
                continue
            e += max(rows, key=lambda r: r[0] + alpha * r[1] + beta * r[2])[3]
        return 100.0 * e / max(w, 1)

    base = wer_at(0.0, 0.0)
    ok = abs(base - d["wer_1best"]) < 1e-6
    print(f"self-check alpha=beta=0 -> {base:.4f} % vs dumped 1-best {d['wer_1best']:.4f} % "
          f"{'OK' if ok else 'MISMATCH — the base score is not the decoder ranking'}")
    if not ok:
        return 1

    alphas = [float(x) for x in args.alphas.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    rows = []
    print(f"\n{'alpha':>7} {'beta':>7} {'WER %':>8} {'vs 1-best':>10}")
    print("-" * 35)
    for a in alphas:
        for b in betas:
            w = wer_at(a, b)
            rows.append({"alpha": a, "beta": b, "wer_percent": w})
            print(f"{a:>7.2f} {b:>7.2f} {w:>8.2f} {w - base:>+10.2f}", flush=True)

    best = min(rows, key=lambda r: r["wer_percent"])
    unseen = [not s for s in seen]
    a_b, b_b = best["alpha"], best["beta"]
    split_wer = {
        "all": {"1best": base, "rescored": best["wer_percent"]},
        "unseen": {"1best": wer_at(0.0, 0.0, unseen), "rescored": wer_at(a_b, b_b, unseen),
                   "n_trials": sum(unseen)},
        "seen": {"1best": wer_at(0.0, 0.0, seen), "rescored": wer_at(a_b, b_b, seen),
                 "n_trials": sum(seen)},
    }
    print("\n" + "=" * 60)
    print(f"  1-best (1-gram only)   : {base:.2f} %")
    print(f"  best rescored          : {best['wer_percent']:.2f} %  "
          f"(alpha={a_b}, beta={b_b})")
    print(f"  recovered              : {base - best['wer_percent']:.2f} pts of the "
          f"{base - d['wer_oracle']:.2f}-pt oracle gap")
    print(f"  oracle (ceiling)       : {d['wer_oracle']:.2f} %")
    print("-" * 60)
    for k in ("unseen", "seen"):
        v = split_wer[k]
        print(f"  {k:>6} sentences ({v['n_trials']:>4} trials): "
              f"{v['1best']:.2f} -> {v['rescored']:.2f} %  "
              f"({v['rescored'] - v['1best']:+.2f})")
    print("  ^ report the UNSEEN row. `seen` is memorization, not language modelling.")
    print("=" * 60)

    if args.out:
        p = rp(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "nbest_source": str(args.nbest), "split": d["split"], "n_trials": d["n_trials"],
            "trigram": {"sentences": len(sents), "tokens": lm.total, "vocab": len(lm.vocab),
                        "discount": args.discount,
                        "ref_oov_percent": 100.0 * oov / max(len(ref_toks), 1),
                        "verbatim_overlap_percent": 100.0 * sum(seen) / max(len(refs), 1)},
            "wer_1best": base, "wer_oracle": d["wer_oracle"],
            "best": best, "by_overlap": split_wer, "grid": rows,
        }, indent=2))
        print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="val_metrics.pkl -> logits cache (run in .venv)")
    e.add_argument("--val_metrics", default="results/causal_la0/checkpoint/val_metrics.pkl")
    e.add_argument("--out", default="results/la0_logits.npz")
    e.set_defaults(func=cmd_export)

    d = sub.add_parser("decode", help="incremental decode (run in the b2txt25_lm env)")
    d.add_argument("--cache", default="results/la0_logits.npz")
    d.add_argument("--lm", default=str(DEFAULT_LM))
    d.add_argument("--acoustic_scale", type=float, default=0.3)
    d.add_argument("--blank_penalty", type=float, default=9.0)
    d.add_argument("--beam", type=float, default=17.0)
    d.add_argument("--lattice_beam", type=float, default=8.0)
    d.add_argument("--max_active", type=int, default=7000)
    d.add_argument("--min_active", type=int, default=200)
    d.add_argument("--length_penalty", type=float, default=0.0)
    d.add_argument("--blank_skip_thresh", type=float, default=1.0,
                   help="1.0 = disabled (can never fire). C14 sweeps 0.999/0.99/0.9")
    d.add_argument("--capture_partials", action="store_true",
                   help="call result() every frame for stability metrics; adds real cost")
    d.add_argument("--split", choices=["dev", "test", "all"], default="dev",
                   help="dev = 35 sessions/1253 trials (tune here); "
                        "test = 6 held-out sessions/173 trials (touch once per group)")
    d.add_argument("--n_trials", type=int, default=0)
    d.add_argument("--progress", type=int, default=200)
    d.add_argument("--out", default="")
    d.add_argument("--save_hyps", default="")
    d.set_defaults(func=cmd_decode)

    w = sub.add_parser("sweep", help="C15 decode-parameter sweep (run in the b2txt25_lm env)")
    w.add_argument("--cache", default="results/la0_logits.npz")
    w.add_argument("--lm", default=str(DEFAULT_LM))
    w.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    w.add_argument("--acoustic_scales", default="0.20,0.25,0.30,0.325,0.40,0.50")
    w.add_argument("--blank_penalties", default="1,3,9,30,90")
    w.add_argument("--beam", type=float, default=17.0)
    w.add_argument("--lattice_beam", type=float, default=8.0)
    w.add_argument("--max_active", type=int, default=7000)
    w.add_argument("--min_active", type=int, default=200)
    w.add_argument("--length_penalty", type=float, default=0.0)
    w.add_argument("--blank_skip_thresh", type=float, default=1.0)
    w.add_argument("--n_trials", type=int, default=0)
    w.add_argument("--out", default="")
    w.set_defaults(func=cmd_sweep)

    t = sub.add_parser("temp", help="C10: temperature sweep (run in the b2txt25_lm env)")
    t.add_argument("--cache", default="results/la0_logits.npz")
    t.add_argument("--lm", default=str(DEFAULT_LM))
    t.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    t.add_argument("--temperatures", default="1.0,1.25,1.5,2.0,3.0")
    t.add_argument("--acoustic_scale", type=float, default=1.0, help="C15 optimum")
    t.add_argument("--blank_penalty", type=float, default=30.0, help="C15 optimum")
    t.add_argument("--beam", type=float, default=17.0)
    t.add_argument("--lattice_beam", type=float, default=8.0)
    t.add_argument("--max_active", type=int, default=7000)
    t.add_argument("--min_active", type=int, default=200)
    t.add_argument("--length_penalty", type=float, default=0.0)
    t.add_argument("--blank_skip_thresh", type=float, default=1.0)
    t.add_argument("--n_trials", type=int, default=0)
    t.add_argument("--out", default="results/c10_temperature.json")
    t.set_defaults(func=cmd_temp)

    n = sub.add_parser("nbest", help="C28-A: decode at nbest=N, dump lists (b2txt25_lm env)")
    n.add_argument("--cache", default="results/la0_logits.npz")
    n.add_argument("--lm", default=str(DEFAULT_LM))
    n.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    n.add_argument("--nbest", type=int, default=100)
    n.add_argument("--acoustic_scale", type=float, default=1.0, help="C15 optimum")
    n.add_argument("--blank_penalty", type=float, default=30.0, help="C15 optimum")
    n.add_argument("--beam", type=float, default=17.0)
    n.add_argument("--lattice_beam", type=float, default=8.0)
    n.add_argument("--max_active", type=int, default=7000)
    n.add_argument("--min_active", type=int, default=200)
    n.add_argument("--length_penalty", type=float, default=0.0)
    n.add_argument("--blank_skip_thresh", type=float, default=1.0)
    n.add_argument("--n_trials", type=int, default=0)
    n.add_argument("--progress", type=int, default=200)
    n.add_argument("--out", default="results/nbest_la0_1gram.json")
    n.set_defaults(func=cmd_nbest)

    s = sub.add_parser("rescore", help="C28-A: re-rank the n-best with the in-domain trigram")
    s.add_argument("--nbest", default="results/nbest_la0_1gram.json")
    s.add_argument("--sentences", default="results/train_sentences.txt")
    s.add_argument("--discount", type=float, default=0.75)
    s.add_argument("--alphas", default="0,0.25,0.5,1,2,4,8")
    s.add_argument("--betas", default="0,2,4")
    s.add_argument("--out", default="results/rescore_indomain_trigram.json")
    s.set_defaults(func=cmd_rescore)

    a = ap.parse_args()
    return a.func(a) or 0


if __name__ == "__main__":
    sys.exit(main())
