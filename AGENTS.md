# AGENTS.md — b2txt-2025

Fork of `Neuroprosthetics-Lab/nejm-brain-to-text` (Card et al. 2024, NEJM). Upstream line numbers
still match; edits are additive.

## 1. The project

Convert the Card et al. pipeline into a **fully causal, real-time streaming decoder**, then find the
best accuracy achievable *subject to* a hard real-time constraint.

The original question ("what does causality cost?") is **not** answered — the L=0 vs L=4 comparison
is confounded, and the released features turned out to be non-causally normalized. The live
questions are **"what is the causal-constrained accuracy optimum?"** and **"what does end-to-end
causality actually cost?"**

Write-up targets: ICASSP, SLT, EMBC, NeurIPS neuro tracks. Not a leaderboard entry.

**Open decision — do not assume either way.** Whether the deliverable is (a) an accuracy-vs-latency
**frontier** paper, (b) **best accuracy at fixed latency**, or (c) a **systems-and-measurement**
paper on what real-time costs once the LM is priced. It changes experiment ordering. If a task's
sequencing depends on it, ask rather than picking.

### The constraint that defines everything

| | |
|---|---|
| End-to-end latency budget | ~140 ms = `L_algo` + `L_buf` (0–60 ms, mean 30) + `L_comp` |
| Acoustic side, worst case | **60.3 ms of 140** — leaves ~79 ms per frame for the LM |
| LM stack, as shipped | **620–830 ms finalization = 4.4–5.9× the whole budget** |
| Emission | streaming; no full-utterance right context anywhere |

`L_algo = 20·L` ms where `L = smooth_lookahead` in 20 ms bins. **80 ms is the update cadence, not
the emission latency** — conflating them overstates `L_buf` and understates the LM's share.

**Ledger rule.** A latency claim is only real if it prices *every* stage, including LM decode and
rescoring. An acoustic-only ledger is incomplete — say so rather than accepting it.

---

## 2. Current state — 2026-08-07

**WER chain** (val-dev, incremental, no batch stage): 42.42 % shipped 1-gram → 35.51 % after decode
sweep → **8.49 %** with the C28 4-gram → **6.89 %** with C29 gpt2-large rescoring. Oracle over the
100-best is **2.88 %**, against a published full-stack 2.66 % **[unverified]**.

Per-frame LM cost p95 **0.641 ms** against 79 ms; finalization ~96 ms p95 against 140 ms.

**Deliverable operating point:** `results/lm_gen_p1e-8/data/lang_test` (3.94 GB TLG) + gpt2-large
rescoring at **γ=0.25, α=1.0**.

**Acoustic anchors:** `causal_la0` 10.04 % val PER, `causal_la4` 10.21 %, against a **0.041-pt**
replicate floor. Checkpoints live in `results/`, not `trained_models/`.

**Done:** Group 0; Group 1 (all gates, 0 training runs); Group 2 (1 run — **rejected on WER**);
C28; C29. Training runs spent: **1**.

**`val-test` (173 trials, 6 held-out sessions) is UNSPENT.** Spend it once, on the final stack.

**The strategic consequence.** The acoustic model is 40× under its compute budget while the LM was
4–6× over it. **The LM/decoding stage is the entire remaining budget.** Any optimization aimed at
the GRU before pricing the decoder is aimed at the wrong stage.

---

## 3. Next

**Immediate, cheap and decisive.** Error-analyse the 71 % of oracle headroom that survives every
rescorer: sample trials where the 1-best is wrong but a better candidate exists, and determine
whether the right and wrong candidates are genuinely equiprobable under English, or whether the
acoustic term is overwhelming the LM. No GPU time, and it decides whether *any* rescorer can win.

Then, in priority order:

1. **Shallow fusion (C29b)**, gated on that analysis — the only mechanism that can beat the 2.88 %
   oracle, since it changes which candidates exist rather than reordering a fixed list.
   ~+10 ms/word plus decoder integration.
2. **`val-test`** on the final stack.
3. **Group 3** — C20 (label smoothing) downgraded; C10 and G2 both undercut its mechanism.
   C21/C24 untouched by that argument.
4. **Group 4 validity runs** (R-D1/R-D2/R-D3). C8 left the variance floor for sparse channels as an
   open design choice — decide it before the run, not after.
5. **Larger C28 corpus** — blocked on RAM, not method.

### Do not re-propose without new evidence

Each was killed by measurement in this repo, not by argument.

1. **Perplexity does not predict WER here (4 for 4).** The in-domain 4-gram had 4× worse ppl and won
   by 19 pts; fine-tuning gpt2-large improved ppl 14× and made unseen WER *worse*.
2. **In-domain adaptation buys memorization, not generalization (4 for 4).** 6.94 % of val-dev
   sentences appear verbatim in train — always split seen/unseen and report unseen.
3. **PER is the wrong gate.** G2 passed its PER gate (10.04 → 9.43) and lost on WER (8.49 → 9.10):
   masking sharpened the posterior and starved the beam.
4. **Bigger rescorers.** Plateaus at ~29 % of oracle headroom; Qwen2.5-1.5B is worse than gpt2-large
   at 2× the parameters and 2.5× the latency.
5. **Delay-penalized / Bayes-Risk / Peak-First CTC, TrimTail, Align-With-Purpose** — excess learned
   emission delay is **+4.64 ms**, 6 % of one frame. Nothing to remove.
6. **Reduced `patch_size` for cold start** — median time-to-first-token is 3,380 ms vs a 280 ms fill.
7. **Left-padding the patcher; deep ensembles at N=10; wider/deeper GRU.** Full reasoning and the
   rest of the excluded list: `brainstorm/PLAN.md` §9.

---

## 4. Working norms

- **No execution without approval.** No training launches, no long-running experiments without an
  explicit go.
- **Source-first verification.** Grep the source before asserting any pipeline property; cite
  `file:line`. Never assert from memory of a paper or from the shape of the code.
- **Push back.** If a request rests on a wrong premise, fix the premise first. Do not implement a
  plausible-looking version of a bad idea.
- **Traceable deltas only.** Every claimed improvement needs a mechanism and a magnitude.
- **Grouped batching, not one-variable-per-run** — but only bundle mechanistically distinct,
  **same-signed** changes, and never bundle opposed ones. Validity runs stay single-variable.
- **Pre-register the gate**: config, metric, threshold, and what each outcome implies, *before*
  launching.
- **A measurement beats any entry in these files.** When one does, correct the entry in the same
  commit rather than leaving both versions in play.

**Response style.** Lead with the answer; justify only where it changes the conclusion. No preamble,
no restatement of the question, no closing summary. Short prose by default; bullets for 3+ discrete
comparisons or ordered steps. Assume fluency in code, math, and papers. Read the docs rather than
reciting APIs from memory, and flag uncertainty explicitly.

---

## 5. Where everything else lives

| Need | Where |
|---|---|
| Plan, change list, results ledger, excluded list | `brainstorm/PLAN.md` |
| Public results summary | `RESULTS.md` |
| Evidence archive (audit, literature, candidates) | `brainstorm/audit/` |
| Acoustic pipeline facts — smoothing, patching, normalization | auto-loads on `model_training/**` |
| LM traps, build limits, decode tuning | auto-loads on `language_model/**` |
| Splits, seen/unseen, PER-vs-WER, latency accounting | auto-loads on `benchmark/**` |
| Running a training job | skill `train-model` |
| Building or measuring an LM | skill `build-lm` |

**Environment:** use `.venv` (uv, py3.10) for everything training and benchmark side. conda exists
for exactly one thing — `setup_lm.sh`, which builds the py3.9 env holding `lm_decoder`. Export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for benchmark entrypoints; without it the
allocator pages over PCIe and batches go 0.169 → ~1.5 s.
