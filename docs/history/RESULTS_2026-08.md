> Historical snapshot preserved 2026-09-18. Not current instructions. Claims may be superseded by ../../RESEARCH_ASSESSMENT.md. Original text and relative references follow unchanged.

# Real-time brain-to-text: measured results

This fork asks one question: **how accurate can this system be if the language-model stage is
required to run incrementally, inside a real-time latency budget?**

The published stack reaches 2.66 % WER, but its language-model stage costs **620–830 ms of
finalization** — 4.4–5.9× the entire ~140 ms end-to-end budget — because it runs n-best
rescoring and a 6.7B-parameter LLM after speech ends. Everything below runs **frame-by-frame with
no batch stage**, and every number is measured on this repository's data.

> **Status:** the LM stage is characterized end to end. The original causality question is
> answered (2026-08-09): at matched bandwidth, removing 80 ms of lookahead costs +0.030 PER,
> under the 0.041 replicate floor. The remaining validity threat is the normalizer — released
> features are still whole-block z-scored. `val-test` (173 trials, 6 held-out sessions) was
> **spent once, on 2026-08-08**, against the frozen stack with nothing re-tuned — see
> [Held-out evaluation](#held-out-evaluation) below. Unless a row says otherwise, WER figures
> are `val-dev` (1,253 trials, 35 sessions).

---

## Headline

| Stage | val-dev WER | What changed |
|---|---|---|
| Shipped 1-gram, default decode parameters | 42.42 % | the honest starting point |
| + decode-parameter sweep | 35.51 % | free; no retraining |
| + **4-gram language model** (built here) | **8.49 %** | the single largest change in the project |
| + **neural n-best rescoring** (gpt2-large) | **6.89 %** | 29 % of the remaining oracle headroom |

**42.42 % → 6.89 %, incrementally, with no batch stage.** Per-frame LM cost is p50 0.067 ms /
p95 0.641 ms against a 79 ms per-frame budget — a 123× margin — versus 620–830 ms of batch
finalization for the shipped stack.

End-of-utterance finalization is ~96 ms p95 on val-dev, **but that figure does not survive the
harder held-out sessions** (~158 ms p95, over budget) because finalization scales with candidate-list
length. **The evaluated configuration fails its own latency constraint on held-out data** — see
[Held-out evaluation](#held-out-evaluation). Truncating the n-best to 10 is the obvious mitigation
and returns finalization to ~83 ms, but its held-out accuracy is unverified for the reason given
there.

The oracle over the 100-best is **2.88 %**, essentially the published 2.66 %. **The correct
hypothesis is already in the list**; the remaining 4.0 points is a selection problem, not a search
problem.

---

## Held-out evaluation

The six held-out sessions were decoded **once**, with every hyperparameter frozen at its
val-dev-selected value and the rescoring grid pinned to a single point, so no selection could occur.

| | val-dev (1,253) | **val-test (173)** |
|---|---|---|
| incremental 4-gram, 1-best | 8.49 % | 18.84 % |
| + gpt2-large rescoring | 6.89 % | 16.10 % |
| — **unseen** (the reportable row) | 7.09 % | **19.31 %**  *95 % CI [15.17, 23.49]* |
| — seen | 4.01 % | 11.88 % |
| headroom recovered by the rescorer | 29 % | **26 %** |

**The rescorer transfers** — 26 % of oracle headroom recovered against 29 % on val-dev. The LM
stage is not overfit to the tuning split.

**The two splits are not comparable, and the difference is measured, not speculative:**

1. **The held-out sessions are 2.03× harder acoustically** — per-session PER 21.06 % vs 10.39 %.
   The WER/PER ratio is unchanged across splits, so this is harder data, not a tuning failure.
2. **43.4 % of val-test references appear verbatim in the training transcriptions, against 6.9 % of
   val-dev.** The aggregate is memorization-inflated; the unseen row is the only honest headline.
3. The unseen subset is **98 trials / 663 words**, so the error bar is roughly **±4 points**.
   Do not read val-test at finer resolution than that.

**The headline held-out finding is a failure, not a pass: the evaluated operating point does not
hold its own real-time constraint.** Harder data produces longer candidate lists (mean 52.1 vs
30.6), and finalization scales with list length — full n-best reaches **~158 ms p95 on val-test,
against the 140 ms budget**, where val-dev had suggested ~96 ms. The *accuracy* transferred; the
*latency* did not. An accuracy-only evaluation would have reported a clean success.

**A caveat we are obliged to state, because it limits what the mitigation is worth.** Truncating to
the top-10 candidates returns finalization to ~83 ms and costs 0.15 points on val-test. But that
number came from evaluating a **second** configuration on a **one-shot** split, which is precisely
what a held-out set is not for. So:

- **19.31 % unseen (full n-best) is a clean held-out result.** Nothing was tuned; it stands.
- **19.46 % (top-10) is descriptive only** — a mitigation supported by val-dev whose held-out
  accuracy is unverified, with no held-out set left to verify it.

The underlying process error is that this evaluation pre-registered accuracy criteria and **no
latency criterion**, even though the necessary scaling behaviour had already been measured on
val-dev beforehand. It is recorded here rather than smoothed over.

---

## What the measurements overturned

Most of what was expected to help did not, and the reasons generalize.

### Causality is free; the 0.17-pt "gain" was bandwidth

A bandwidth-matched causal control (R-D1a, 2026-08-09) decomposes the old L=4 → L=0 comparison:

```
observed L=4 → L=0    −0.170 PER  =  bandwidth −0.200  +  causality +0.030
```

Removing 80 ms of lookahead, at matched σ_eff ≈ 1.85, costs **+0.030 PER — under the 0.041
replicate floor.** The previously quoted 0.17-pt advantage of the causal kernel was the truncation
narrowing σ_eff by 37 % (1.853 → 1.161), not the latency property. That narrowing is worth
**0.61 unseen WER at zero added latency**, because the extra kernel tail is all past.

The smoother is therefore causal. The released features are still whole-block z-scored, so
end-to-end causality of the *normalizer* is not yet shown.

### Perplexity does not predict WER here — four times over

| Change | Perplexity | WER effect |
|---|---|---|
| In-domain 4-gram (8k sentences) vs general | 4× **worse** | −19 pts **better** |
| Interpolating in-domain into the general LM | better (91.5 vs 103.6) | **worse** on unseen sentences |
| In-domain trigram re-ranking the 4-gram's n-best | — | **0.00 pts** |
| Fine-tuning gpt2-large on in-domain text | **14× better** (395 → 27) | **worse** (7.09 → 7.42 unseen) |

Restricting an LM's vocabulary acts as a domain prior that perplexity rewards and decoding does
not. **Do not select a language model for this task on perplexity.**

### In-domain adaptation buys memorization, not generalization

6.94 % of `val-dev` sentences appear **verbatim** in the training transcriptions. Every in-domain
technique tried looks good until that overlap is split out, at which point the gain lands almost
entirely on the memorized sentences. All results here report **seen** and **unseen** subsets
separately; the unseen row is the reportable one.

### Better phoneme accuracy made word accuracy worse

A regularization bundle (time masking + channel masking + full phase coverage) improved val PER
**10.04 % → 9.43 %** — a strong pass against its ≤ 9.50 % gate — and made WER **worse**,
8.49 % → 9.10 %. The masking made the posterior *sharper* (mean entropy 0.89 % → 0.86 % of
maximum), leaving the beam fewer alternatives for the LM to repair. **PER is the wrong gate for a
system whose deliverable is WER.**

### Most acoustic *capacity* changes did not help. Smoothing bandwidth did.

| Change | WER effect |
|---|---|
| Temperature scaling | −0.05 |
| Phase merging, interleaved 50 Hz | +0.46 (but buys L_buf 60 → 15 ms) |
| Phase merging, phase-averaged | +0.69 |
| Regularization bundle (masking) | +0.61 (PER improved; WER got worse) |
| **Smoothing bandwidth** (σ_eff 1.853 → 1.161, lookahead held at 0) | **−0.61 unseen** |

The acoustic model occupies 60.3 ms of a 140 ms budget and is 40× under its compute ceiling. The
LM axis moved WER 33 points; every capacity or regularization change on the GRU was neutral or
negative. The exception is the smoother: a bandwidth-matched causal control (R-D1a, 2026-08-09)
showed that the old L=0 vs L=4 gap was a 37 % kernel narrowing, not lookahead, and that narrowing
is worth **0.61 unseen WER at zero added latency**. Unlike masking, PER and WER moved the same
way (3× amplification). The remaining validity question is the normalizer, not the GRU.

---

## The accuracy/latency frontier

Rescoring is one batched forward pass per utterance, so it is cheap; the knobs are model size and
how deep into the n-best list you score.

| Configuration | val-dev WER | finalization p95 (val-dev) |
|---|---|---|
| no rescoring | 8.49 % | 6.1 ms |
| gpt2-large, top-5 | 7.45 % | ~35 ms |
| gpt2-large, top-10 | 7.35 % | ~37 ms |
| **gpt2-large, full n-best — the evaluated point** | **6.89 %** | ~96 ms |

Every point sits inside the 140 ms budget **on val-dev**. The oracle gain is spread *through* the
list — going from ~30 candidates to 10 costs 9 points of headroom — and model size and list depth
trade off against each other (gpt2-large@top-10 ≈ gpt2-small@full on both axes).

**This whole table is a val-dev frontier, and its latency column does not transfer.** On the harder
held-out sessions the lists grow (mean 52.1 vs 30.6) and every row's finalization cost rises with
them — full n-best goes over budget there. Choosing a row on the basis of held-out latency is what
compromised the top-10 number; see [Held-out evaluation](#held-out-evaluation).

Scaling stops paying past ~800M parameters: Qwen2.5-1.5B is *worse* than gpt2-large at twice the
parameters and 2.5× the latency (p95 190.9 ms, over budget).

---

## Reproducing

Environment setup is unchanged from upstream (`setup.sh`, `setup_lm.sh`). Beyond that, the LM
decoder's command-line tools are **not** built by `setup_lm.sh` — see `language_model/build_tlg.sh`,
which wires up SRILM and the six Kaldi FST binaries and is the entry point for building a language
model.

```bash
# Build a 4-gram + TLG decoding graph from a text corpus
cd language_model
python fetch_corpus.py --target_gb 2            # stream an OpenWebText subset
./build_tlg.sh <out_dir> <corpus.txt> 4 1e-8    # count, prune, compose

# Measure it (two environments by necessity — see benchmark/stream_lm.py)
cd ../model_training
../.venv/bin/python benchmark/stream_lm.py export --val_metrics <val_metrics.pkl> --out results/logits.npz
$(conda info --base)/envs/b2txt25_lm/bin/python benchmark/stream_lm.py \
    decode --cache results/logits.npz --lm <out_dir>/data/lang_test --split dev

# Neural rescoring
../.venv/bin/python -m benchmark.neural_rescore --model gpt2-large --gammas 0.5,0.25,0.1
```

**Two traps that produce plausible-looking wrong answers**, both documented in the scripts:

1. The LM recipe is **uppercase end to end**. Lowercase text fed to `ngram-count` maps every token
   to `<unk>`, exits 0 in under two seconds, and yields three bigrams.
2. When combining a neural LM with the n-gram, **sweep the n-gram's weight**. Adding the neural
   score on top of a full-weight n-gram double-counts the language model; correcting it was worth
   more than a 6× increase in parameters.

---

## Layout of this fork's additions

```
model_training/
  benchmark/stream_lm.py         incremental WFST driver; decode/sweep/nbest/rescore/temperature
  benchmark/neural_rescore.py    n-best rescoring with any HF causal LM
  benchmark/finetune_rescorer.py in-domain adaptation of a rescorer (measured, rejected)
  benchmark/phase_ensemble.py    per-phase PER and the two phase-merge modes
  benchmark/indomain_lm.py       trigram over the training transcriptions
  causal_normalize.py            causal rolling normalization + its identity check
  splits.py                      frozen val-dev / val-test session split
language_model/
  build_tlg.sh                   SRILM + Kaldi FST wiring; builds an n-gram and its TLG graph
  fetch_corpus.py                streams and normalizes an LM corpus
  make_tlg_from_arpa.sh          compose TLG from an existing ARPA
  interpolate_lm.sh              LM interpolation with held-out lambda selection
brainstorm/                      planning, audit and research-proposal documents
```

`brainstorm/` holds the exploratory material: the original optimization plan, a causality audit of
the released features, a literature review, a candidate-change catalog, and `PLAN.md`, which is the
working record of what was implemented, what was measured, and which predictions failed. It is kept
because the failed predictions are more instructive than the successful ones.
