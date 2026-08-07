# OPEN_QUESTIONS.md

Everything I could not resolve, each with the specific evidence that would resolve it. Ordered by
how much downstream work depends on the answer.

---

## Tier 1 — blocks a headline claim

### Q1. What does the incremental WFST beam search actually cost per frame?

**Status:** completely unmeasured. **Blocks:** R1 in its entirety, and every WER number in
`candidates.json`.

The incremental path exists and works (`ctc_wfst_beam_search.cc:98` advances Kaldi one frame at a
time; `:113` extracts a partial best path), but nobody has ever timed it. The only anchors are the
repo's own end-to-end figure — **620–830 ms per trial [D]** for the full 5-gram + OPT stack,
derived from `evaluate_model.py:215` and 1,450 test trials — and an external offline number,
**17 ± 11 ms/trial** for a 3-gram at beam width 18 (arXiv:2507.02800). Neither is a per-frame
incremental cost, and [LITERATURE §B5](LITERATURE.md) found **no published per-frame incremental
LM cost for any brain-to-text system**.

**Resolves it:** build `lm_decoder` (`./setup_lm.sh`, needs CMake ≥ 3.14 and gcc ≥ 10.1 — not
present on this machine), drive `DecodeNumpy` one frame at a time over the cached val logits, and
time `DecodeNumpy` and `result()` separately at each of `beam ∈ {12,15,17,20}` ×
`max_active ∈ {2000,4000,7000}`. Run #2 in the ledger.

---

### Q2. What is the constrained baseline WER?

**Status:** estimated at **~5.5% (range 4–8) [E]**, never measured. **Blocks:** every ΔWER in the
catalog is a prediction against this number.

The estimate comes from scaling arXiv:2507.02800's offline-vs-LLM-rescored ratio (12.17 → 5.68%,
53% relative) onto this project's ~2.66% unconstrained figure. That is a cross-participant,
cross-LM extrapolation and could easily be wrong by a factor of two.

**Resolves it:** run #2. Same measurement as Q1.

---

### Q3. Which LM produced the quoted 2.66% local WER?

**Status:** **[UNVERIFIED]**, and it matters more than it looks.

The only LM artifact present locally is the **1-gram** (`openwebtext_1gram_lm_sil/TLG.fst`,
13.4 MB). The repo's README pairs it with OPT-6.7B rescoring at `nbest=100` plus `augment_nbest`.
If 2.66% came from a 1-gram + OPT, then essentially **all** the lexical work is being done by the
LLM — squarely inside the 620–830 ms batch wall — and R1's recoverable fraction is much smaller
than estimated. If it came from a 3-gram or 5-gram, the n-gram is doing real work and R1 is on
firmer ground.

Cross-check that deepens the doubt: an independent T15 within-subject baseline reports **7.34%
WER** (J. Neural Eng. 23(4) 2026), nearly 3× the 2.66% figure. Different LM stacks, but the gap is
large enough to want the provenance nailed down.

**Resolves it:** locate the eval run's `--lm_path` and whether `--do_opt` / `--rescore` were set;
or re-run the eval with the 1-gram and compare.

---

### Q4. Ensemble size, combination operator, and the baseline-error dependence in the deep-ensembles preprint

**Status:** **[UNVERIFIED — bioRxiv returns HTTP 403 to automated fetch on `.full`, `.full.pdf`,
the ResearchGate mirror and Europe PMC].** **Blocks:** any quantitative claim about S1.

I have the abstract, full author list (Willett senior), date (2026-06-04) and the headline
33.7% → 26.0% WER. I do **not** have: N tested, whether members are combined by logit averaging /
ROVER / LLM merge, what augmentation the "pseudoensembling" uses, compute topology, or measured
latency. Most importantly, the abstract states the paper assesses "how these gains depend on
baseline error rate" — **that figure is exactly what is needed to know whether a 23% relative gain
at 33.7% baseline transfers to a ~3–5% baseline**, and it is the basis on which I declined N=10.

**Resolves it:** manual retrieval of doi 10.64898/2026.06.02.729705 in a browser. Ten minutes of
human time; the single highest-value unresolved item in this document.

---

## Tier 2 — changes a recommendation

### Q5. Was the upstream normalization exactly a whole-block z-score?

**Status:** block-*scale* normalization is established to within 1–26% by the 1/√W scaling test
across four sessions ([AUDIT F1](AUDIT.md#f1)). The exact functional form is not.

Unknown: whether the +10 clip was applied before or after normalization, and whether the
statistics were computed over exactly the released trials or a superset including unreleased
inter-trial periods (the residual 0.0085 median is consistent with the latter).

This matters because R2's exact-cancellation argument — that rolling-normalizing the released data
recovers the causally-normalized raw features identically — assumes a pure affine block transform.
If clipping came after, the 0.004% of samples at the rail are unrecoverable (negligible); if the
statistics came from a superset, the cancellation is still exact (the constants still cancel), so
R2 largely survives either way.

**Resolves it:** the identity check pre-registered in R2 — feed the rolling normalizer constant
block statistics and confirm it reproduces the released features to 1e-4. Failure means the
transform is misidentified.

---

### Q6. Is the L=0 advantage a causality effect or a bandwidth effect?

**Status:** confounded, unresolved. **Changes:** the project's central claim.

`smooth_lookahead` changes lookahead **and** effective smoothing bandwidth together: σ_eff goes
1.852 → 1.161 bins (−37%) and peak weight rises 66% going L=4 → L=0 **[M]**. The observed
+0.163-point advantage for L=0 has an obvious alternative explanation that has nothing to do with
causality.

**Resolves it:** run #12, the bandwidth-matched causal control — a lookahead=0 kernel with σ_eff
matched to 1.852 bins. Lands near 10.21% ⇒ bandwidth. Lands near 10.04% ⇒ something else. One run.

---

### Q7. Does the la0-vs-la4 difference survive replication?

**Status:** n=1 per config. The only same-config replicate available (`baseline_rnn` vs
`causal_la4`, algorithmically identical) differs by **0.041 points [M]**, against a la4→la0 effect
of **0.163 points** on the last-10-validation mean.

The brief asserts the difference is "inside seed noise." The available evidence says the opposite —
roughly 4× the one observed same-config difference, with a clean and much larger val-loss gap
(21.74 vs 22.60). But one replicate on a different GPU is a very weak variance estimate.

**Resolves it:** runs #18–19, two more seeds on `causal_la0`, plus the existing la4 run.

---

### Q8. Is the patch embedding phase-sensitive?

**Status:** unknown; decides whether R3 costs 1.5 h or 7.9 h.

`random_cut = 3` covers 3 of 4 phases during training **[M]**; the smoother is shift-equivariant
and the day layer is per-bin, so only the patch embedding sees phase.

**Resolves it:** run #3 — per-phase greedy PER on the existing checkpoint. 30 minutes, no training.

---

### Q9. Can a 4-gram be pruned under 32 GB without destroying it?

**Status:** one external data point (**19 GB**, 7th-place B2T'25), never reproduced. The repo's own
README says the shipped 3-gram needs **~60 GB** and the 5-gram **~300 GB**, so neither shipped
artifact fits. The brief's "pruned 3-gram (~6–8 GB)" figure is **unsourced** and matches nothing
retrieved.

**Resolves it:** build it. `language_model/srilm-1.7.3/` and `language_model/tools/fst/` have the
toolchain. Note peak RAM *during compilation* can exceed the deployment target even when the
artifact fits.

---

## Tier 3 — affects interpretation, not the plan

### Q10. What GPU produced the existing benchmark numbers?

`metrics.json` records only `device: cuda`. `model_training/CLAUDE.md` says RunPod A100; the brief
says a single RTX 3090. The three completed runs took 372–396 min, consistent with the 6.4 h
budgeting unit, but on an unidentified device. All GPU-derived headroom figures in
[BUDGET §5](BUDGET.md) inherit this uncertainty. The CPU column is measured here and is the safe
floor.

**Resolves it:** `nvidia-smi` on the training pod, or re-run `offline_benchmark.py` with the device
name recorded.

---

### Q11. Training-time VRAM

Never measured — no CUDA device on this machine. Needed before any capacity recommendation
(`wider_deeper_gru`, `larger_patch`, `causal_transformer`) can be sized against 24 GB. The one
external anchor is arXiv:2507.02800's 5.55 GiB for a 56.7 M GRU baseline, which suggests this
repo's 44.3 M model has ample room.

**Resolves it:** `torch.cuda.max_memory_allocated()` around one training step.

---

### Q12. Competition placements 1st (~1.5%) and 4th (~1.78%)

**[UNVERIFIED]** — Kaggle returns HTTP 403 to automated fetch on the leaderboard, competition
overview and discussion pages. I corroborated only the **2.787%** single-model figure, and that
corroboration reframes it: it is the **7th-place** team's single Mamba model, not a separate entry.

B2T'25 also produced far less public writeup material than B2T'24 — one findable solution writeup
versus a full organizer retrospective plus webinar.

**Resolves it:** manual browser access to the Kaggle leaderboard and writeups.

---

### Q13. Is online day-layer adaptation in scope?

The brief excludes "calibration-efficiency work (few-shot day adaptation, minimizing calibration
data)." Online adaptation *during use* is a different thing from reducing calibration data, but
the boundary is thin enough to want confirmed before spending on it. Scored at 0.30 and tagged
`scope-check` in the catalog.

Separately, [BUDGET §4](BUDGET.md) suggests the obvious unsupervised objective for it —
entropy minimization — will likely **no-op**, because mean posterior entropy is already 0.9% of
maximum and the gradient is nearly vanishing.

**Resolves it:** a scope decision, plus a gradient-norm check on the entropy objective.

---

### Q14. Lookahead convolution has no primary citation

Treated as a classical technique (Deep Speech 2 lineage) but **[UNVERIFIED]** — no single modern
citable paper retrieved. It is mechanically the cleanest instrument for building a genuine
accuracy-vs-lookahead curve (exactly 20k ms of L_algo for k future frames), which is what
[AUDIT F2](AUDIT.md#f2) shows the project lacks, so it deserves a proper citation before being
proposed in a paper.

---

## Things I deliberately did not resolve

- **`augment_nbest`'s contribution to the 620–830 ms.** It is O(n²) over the top-20 candidates and
  clearly a batch stage; splitting it from `Rescore()` and OPT would be nice but changes no
  decision, since all three are deleted in the constrained configuration.
- **The exact identity of `feat[0:256]` vs `feat[256:512]`.** Measurement shows one block is
  discrete-ish (170 unique values) and one continuous (32,935 unique), which is enough to identify
  them as threshold crossings and spike-band power respectively. Which is which does not affect
  any recommendation.
- **`create_attention_mask`** (`rnn_trainer.py:408-434`) is dead code with a no-op
  `torch.where(mask, True, False)`. Noted; not worth a decision.
