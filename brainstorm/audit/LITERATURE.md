> Historical evidence, not current instructions. Read ../../RESEARCH_ASSESSMENT.md for corrections. Links to removed proposal files are archival references recoverable from Git or .git/context-cleanup/2026-09-18/.

# LITERATURE.md — Phase B recon

Retrieval date: **2026-08-04**. Every entry carries a URL and a date. Staleness is flagged
relative to a ~12-month window (i.e. anything before ~2025-08 is flagged).

Transfer verdicts are for **this** constraint set: causal decoder, ~140 ms end-to-end, RTF < 1
on one commodity GPU, 32 GB deployment RAM, single participant T15, 45 sessions.

**Access failures (stated up front, not hidden):** `biorxiv.org` and `kaggle.com` both return
HTTP 403 to automated fetches (Cloudflare). Two high-value sources are therefore recorded from
their abstract/metadata and search-indexed summaries only, and their internals are marked
**[UNVERIFIED — needs manual retrieval]**: the deep-ensembles preprint (B3.1) and the B2T'25
leaderboard (B1.1). This is a real gap in the recon and it lands on the highest-priority
recommendation.

---

## Cross-cutting conclusions

1. **The '24 retrospective's architecture verdict is superseded.** Its "transformers and deep SSMs
   do not beat the GRU" conclusion is 20 months old and has since been contradicted on the sibling
   benchmark by a **causal** transformer that beats the unidirectional GRU by 20–26% relative with
   **83% fewer parameters** (B3.4). Citing 2412.17227 for that claim in 2026 would be an error.
2. **The field moved toward smaller + heavily regularized, not bigger.** This is direct evidence
   **against** reviewer prior S2 ("capacity is under-spent"). The strongest recent causal result on
   this benchmark family used 9.4 M parameters against a 56.7 M GRU baseline.
3. **Time masking is the single largest missing regularizer in this repo.** ~53% of every trial
   masked, with learnable MASK tokens (B3.4). This repo has **no** time masking at all ([AUDIT
   §A2](AUDIT.md)).
4. **The ensembling evidence is real but measured in the wrong error regime.** The closed-loop
   result is 33.7% → 26.0% WER (B3.1). This project sits near 2.7%. Ensemble gains compress as
   baseline error falls, and the paper itself studies that dependence — so the headline relative
   gain must **not** be transferred at face value.
5. **The causal-normalization fix has a published precedent from the same lab and participant.**
   Wairagkar et al. normalize with "rolling means and s.d. from the past 10 s" in their real-time
   system (B3.5), while offline datasets in this literature are conventionally "z-scored per block"
   (B3.8). That gap is exactly [AUDIT F1](AUDIT.md#f1).
6. **The emission-latency literature probably does not apply here** — see §B4 verdict. This
   sharpens reviewer prior S6 from "measure first" to "expect a null, and free four runs."

---

## B1 — Brain-to-Text '25 competition

### B1.1 — Competition and leaderboard
- URL: https://www.kaggle.com/competitions/brain-to-text-25/overview ,
  https://www.kaggle.com/competitions/brain-to-text-25/leaderboard
- Status: **closed**. Retrieval **[BLOCKED — HTTP 403]**.

The brief cites "winning WER ≈ 1.5%, 4th place 1.78%, strong single model ≈ 2.79%." Of these I
could independently corroborate **only the 2.79%**, and the corroboration reframes it: the
2.787% public-LB figure is the *single Mamba model* belonging to the **7th-place** team
(B1.2), not a separate entry. **1.5% and 1.78% remain [UNVERIFIED].**

Notably, B2T'25 has generated far less public writeup material than B2T'24 — one findable
solution writeup versus a full organizer retrospective plus webinar for '24. Plan the recon
budget accordingly; the '24 evidence base is much richer and is the one to lean on.

### B1.2 — 7th place: Mamba + GRU + KenLM (Jackson Cheng)
- URL: https://medium.com/@jackson3b04/7th-place-solution-mamba-gru-kenlm-with-code-brain-to-text-25-00f1c69dcd0d
- Date: 2025 (competition closed 2025) — **STALE FLAG: borderline, ~12 months**

The most informative public '25 artifact. Extracted:

| Item | Value |
|---|---|
| Single-model public LB WER | **0.02787** |
| Architecture | **Bidirectional** Mamba w/ "soft sliding window" + GRU variants |
| GRU patch/stride variants | **34/4 and 22/4** (this repo: 14/4) |
| Regularization | dropout, **stochastic depth (drop path)**, weight decay **0.005** (this repo: 0.001) |
| Novel loss | **Temporal Consistency Loss** — squared L2 between adjacent day matrices |
| Ensemble merge | **LISA**: intra-family logit averaging → **context-aware gating on n-gram confidence (threshold −3.76)** → LLM rescoring |
| Rescorer | **Mistral-7B** (not OPT-6.7B) |
| LM | **custom 4-gram KenLM, 19 GB RAM** (from 300 GB), 80% Wiki/news + 20% conversational, custom flashlight decoder, `kTrieMaxLabel=14` |
| Hardware | single A100 (Colab), 120k batches |
| Author credits gains to | Mamba long-range + GRU short-range being **"highly uncorrelated"**; memory-optimized LM; gating that prevents LLM over-correction |

**Three findings that change the plan:**

- **The 19 GB 4-gram is the only real datapoint on the RAM question.** It sits between the brief's
  optimistic "6–8 GB pruned 3-gram" and the repo README's "~60 GB 3-gram". 19 GB *fits* 32 GB with
  the acoustic model alongside — but it was **built**, not downloaded. Budget it as work.
- **The context-aware gate is S5.** "Route on n-gram confidence, only invoke the LLM when
  uncertain" is a deployed version of reviewer prior S5 (adaptive-compute decoding). S5's novelty
  claim must be narrowed accordingly — see [RECOMMENDATIONS](RECOMMENDATIONS.md) when written.
- **Bidirectional.** Latency and memory are unreported, as they are for every entry. That absent
  column is this project's opening.

**Transfer verdict: PARTIAL.** The LM compression recipe and the regularization set transfer.
The bidirectional Mamba and the whole-utterance LLM rescoring do not.

---

## B2 — The '24 retrospective

### B2.1 — Willett et al., *Brain-to-Text Benchmark '24: Lessons Learned*
- URL: https://arxiv.org/abs/2412.17227 — **2024-12-23**, v1 only
- **STALE FLAG: 20 months old.** Its architecture conclusion is now contradicted (B3.4).

Verbatim conclusions from the abstract: largest gains from "an ensembling approach, where the
output of multiple independent decoders was merged using a fine-tuned large language model (an
approach used by all 3 top entrants)"; further gains from "optimizing learning rate scheduling and
by using a diphone training objective"; and "attempts to use deep state space models or
transformers not yet appearing to offer a benefit over the RNN baseline."

The abstract contains **no numbers**. The commonly quoted 9.7% → 5.8% WER improvement appears in
the body/secondary coverage; treat it as **[UNVERIFIED from primary source]** until the PDF is read
in full. 1st place: Yue Chen, Xin Zheng, Tatsuo S. Okubo (Beijing Institute for Brain Research),
using a **diphone** objective.

**Transfer verdicts, split:**
- Ensembling → **TRANSFERS**, but see B3.1 on the error-regime caveat.
- Diphone objective → **TRANSFERS**, causality-neutral, cheap. High-value candidate.
- LR-schedule tuning → **TRANSFERS**. Reinforced by this repo's untuned `eps=0.1` ([AUDIT §A4](AUDIT.md)).
- LLM merge of multiple decoders → **DOES NOT TRANSFER** as specified (batch, end-of-utterance).
- "SSMs/transformers don't help" → **SUPERSEDED. Do not cite.** The result was obtained in an
  unconstrained, effectively non-causal setting; B3.4 shows the opposite in the causal regime,
  which is exactly the regime this project cares about.

---

## B3 — Recent primary literature

### B3.1 — Yoon et al., *Neural decoding of speech using deep neural ensembles*
- URL: https://www.biorxiv.org/content/10.64898/2026.06.02.729705v1 — bioRxiv **2026-06-04**
- DOI: 10.64898/2026.06.02.729705. Fresh.
- Authors incl. Yoon, Avansino, Madugula, Fan, Card, Fogg, Wairagkar, Nason-Tomaszewski, Deo,
  Hochberg, Brandman, Stavisky, Pandarinath, Henderson, **Willett (senior)**.
- Full text **[BLOCKED — HTTP 403 on .full, .full.pdf, ResearchGate mirror, Europe PMC]**.

From the abstract: **first closed-loop test of deep ensembles** in a participant with **bilateral**
intracortical arrays; **WER 33.7% → 26.0%** on a large-vocabulary task. Assesses how gains depend
on "baseline error rate, training dataset size, and ensemble size … including the resource-accuracy
tradeoffs most relevant for real-world deployment," using additional data from **three
participants**. Introduces "a computationally efficient **pseudoensembling** approach based on
**test-time augmentation** that improves decoding accuracy while requiring only a single base
decoder."

**[UNVERIFIED — needs manual retrieval]:** ensemble sizes N, the combination operator (logit
averaging vs. ROVER vs. LLM merge), what augmentation the pseudoensemble uses, compute topology,
and any measured latency. These are precisely the numbers reviewer prior S1 depends on.

**Two observations that matter more than the headline:**

- **The error regime is wrong for this project.** 33.7% → 26.0% is a 23% relative gain measured at
  a baseline **12× higher** than this project's ~2.7%. Ensemble gains shrink as baseline error
  falls — and the paper explicitly studies that dependence, which means the data to correct the
  extrapolation exists in a figure I could not retrieve. **Any S1 prediction that quotes ~23%
  relative is unsupported.** Getting this figure is the single highest-value retrieval left.
- **Pseudoensembling via TTA is the same family as reviewer prior S3.** Phase-staggered replicas
  *are* test-time augmentation over patch phase, using one base decoder. This is strong
  independent support for S3's mechanism — and simultaneously a **novelty threat**: S3 must be
  positioned as "TTA over patch phase, which additionally buys sub-cadence temporal resolution,"
  not as "TTA for BCI ensembling," which is now published.

**Transfer verdict: TRANSFERS (mechanism), DOES NOT TRANSFER (magnitude).**

### B3.2 — Zamora Vera & Gonzalez-Lopez, *Phoneme- vs. Character-Level Targets and Selective State-Space Models for Intracortical Brain-to-Text*
- URL: https://arxiv.org/abs/2607.26751 — **2026-07-29**. Fresh.
- 2×2 grid (GRU vs. hybrid Mamba) × (phonetic vs. character), CTC, on B2T'25.
- Best phonetic GRU: **12.62% PER / 21.19% WER**. Character: 13.39% CER / 26.28% WER after LM
  rescoring. Mamba hybrid competitive, **not better**.

**Why their PER is 2.6 points worse than this repo's — resolved.** The brief flagged this as
needing explanation before citing. It is not a protocol subtlety: **their baseline is simply
below the field's.** The independent T15 baseline in B3.3 is **10.2% PER**, matching this repo's
10.21% almost exactly. So 12.62% is ~2.4 points off the established baseline, and their 21.19% WER
is ~3× the 7.34% WER that B3.3 reports for a comparable baseline. Their absolute numbers reflect
an under-trained or under-tuned reproduction plus a weak LM.

**Transfer verdict: USE FOR DIRECTION ONLY, NEVER FOR ABSOLUTE COMPARISON.** The useful content is
the controlled 2×2 (phoneme targets beat character targets; Mamba ≈ GRU) and the error analysis
(articulatory phoneme confusions vs. lexical/word-boundary errors). Do not quote their numbers as
a reference point.

### B3.3 — *Cross-subject decoding of human neural data for speech BCIs*
- URLs: https://www.biorxiv.org/content/10.64898/2026.02.27.708564v1 (bioRxiv **2026-03-02**);
  published **J. Neural Eng. 23(4), 2026-07-14**, https://iopscience.iop.org/article/10.1088/1741-2552/ae8576
- Fresh.

First neural-to-phoneme decoder trained jointly on **both** Willett 2023 (T12) and Card 2024 (T15),
with day- **and dataset**-specific affine transforms `ỹ = W_{d,s}x + b_{d,s}` aligning subjects into
a shared latent space.

| Condition | Card / T15 PER | Card / T15 WER |
|---|---|---|
| Within-subject baseline | **10.2%** | **7.34%** |
| Cross-subject + hierarchical CTC | **9.1%** | **6.67%** |

On Willett/T12, joint training gives 19.7% → 17.6% (plain CTC) → **16.1%** (hierarchical CTC).
The authors note the joint-training gain is "notably larger for the Willett dataset than for the
Card dataset" — i.e. **T15, this project's dataset, benefits least.**

Architecture: three-block hierarchical GRU, **hidden 2048**, first two blocks **bidirectional**,
intermediate CTC supervision after blocks 1 and 2 whose phoneme predictions are "projected back
and added to their hidden states, guiding deeper layers."

**This is the most directly comparable external result.** Its 10.2% within-subject T15 PER
validates that this repo's 10.21% is a correct baseline reproduction, not an outlier.

**Transfer verdict: SPLIT — and the split is the whole point.**
- Cross-dataset pretraining with day+dataset affine → **TRANSFERS**, causality-neutral. But the
  measured T15 gain is ~1.1 PER points, the authors say T15 gains least, and it requires ingesting
  a second dataset. Moderate value, non-trivial cost.
- Intermediate/hierarchical CTC with feedback → **TRANSFERS AND IS CHEAP.** Nothing about
  auxiliary CTC heads at intermediate depths is non-causal. This is an attractive candidate.
- Bidirectional GRU at hidden 2048 → **DOES NOT TRANSFER.** ~7× this repo's hidden width and
  non-causal. Their 9.1% PER is not a target this project can chase directly.
- Their 7.34% baseline WER vs. this repo's claimed 2.66% shows the two use very different LM
  stacks. **Never compare WER across these papers without naming the LM.**

### B3.4 — Feghhi, Kaasyap, Hadidi & Kao, *Time-Masked Transformers with Lightweight Test-Time Adaptation for Neural Speech Decoding*
- URL: https://arxiv.org/abs/2507.02800 — v1 **2025-07-03**, v2 **2025-11-02**. Fresh (v2).
- Dataset: **Brain-to-Text Benchmark '24 — T12, Willett 2023**, ALS participant, area 6v, 256
  features, 10,850 sentences, 24 days. **Not** T15/B2T'25.

**The most transferable paper found.** It targets this project's exact framing: prior gains
"increased computational costs and were not demonstrated in a real-time decoding setting."

| Component | Detail |
|---|---|
| Time masking | **53% of each trial masked**; N=20 masks/trial, M=0.075 max length fraction; contiguous spans, start + duration uniformly sampled, may overlap; **learnable MASK tokens** |
| Architecture | 5 transformer blocks, d=384, 6 heads, **9.4 M params** (baseline GRU: 56.7 M) |
| Directionality | **Unidirectional causal attention**, T5 relative positions |
| Patching | **non-overlapping 100 ms windows** (5 bins × 256), logits every 100 ms |
| TTA (DietCORP) | Z=64 time-masked/noised augmentations, **one gradient step per trial**, updates **patch embedding only**; 1.33 GiB, **18.21 ± 5.34 ms/trial** |

| Model | 3-gram WER | 5-gram WER | dir. |
|---|---|---|---|
| Baseline GRU | 15.25 ± 0.16 | 11.12 ± 0.13 | uni |
| **Time-masked Transformer** | **12.17 ± 0.22** | **8.18 ± 0.22** | **causal** |
| Linderman Lab GRU | — | 8.0 | bi |
| Diphone + GRU | — | 8.39 ± 0.22 | bi |
| Transformer + Llama-3.1-8B | — | **5.68** | causal |

Relative improvement over the unidirectional GRU: **20% (3-gram) / 26% (5-gram)**.
Efficiency: training VRAM 2.66 vs 5.55 GiB; epoch 10.81 s vs 25.88 s (58% faster); **364.2 vs
634.8 MFLOPs (43% fewer)**. **Beam search decoding: 0.017 ± 0.011 s per trial**, 3× faster than
the GRU's. Beam width 18, α = 0.8, blank penalty log(2) or log(7).

**Four consequences for this project:**

1. **A causal model matched bidirectional SOTA** (8.18 vs 8.0/8.39). This is the strongest
   published evidence that this project's constraint is not as costly as assumed — and it is the
   natural citation for the framing.
2. **It refutes S2 as stated.** The gain came with **83% fewer parameters**. "Capacity is
   under-spent" is contradicted by the closest available evidence; the binding constraint was
   regularization, exactly as the '24 retrospective framed it.
3. **Time masking is the highest-expected-value single change available.** Absent from this repo,
   worth up to 20% relative in the closest published comparison, one config-level change, no
   latency cost.
4. **The first LM-cost anchor: 17 ms/trial for a 3-gram beam search at width 18** — offline, whole
   utterance. Order-of-magnitude useful for Phase C, but *not* an incremental per-frame number.

**Transfer caveats:** different participant (T12, area 6v) and different benchmark; their
100 ms non-overlapping patch is coarser than this repo's 14/4; their reported WERs are far above
this project's because the LM stack differs.

**Transfer verdict: TRANSFERS — highest confidence of anything in this review.**

### B3.5 — Wairagkar et al., *An instantaneous voice-synthesis neuroprosthesis*
- **Citation correction:** the brief gives "Nature 2025, PMC11370360". PMC11370360 is the
  **bioRxiv preprint** (**2024-08-19**, v2 2024-09-20). The journal version is
  **Nature 644:145–152 (2025)**, https://www.nature.com/articles/s41586-025-09127-3 ,
  PubMed 40506548. Cite the Nature version.
- **Same participant lineage (T15) and same lab** (Card, Wairagkar, Brandman, Stavisky).
- Code: https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025

**The closest published latency accounting, and the direct precedent for [AUDIT F1](AUDIT.md#f1).**

Pipeline: 30 kHz sampling → threshold crossings + spike-band power from 1 ms segments → **10 ms
non-overlapping bins** → normalize → smooth. Explicitly:

- **"normalized using rolling means and s.d. from the past 10 s for that feature"** — causal.
- **"causal smoothing using a sigmoid kernel of length 1.5 s of the past activity"** — 75 bins of
  past at 20 ms, versus this repo's 5-tap (100 ms) Gaussian. **Far longer causal integration.**
- log-transform before normalization; per-block recomputation of RMS thresholds.
- **Full pipeline, neural acquisition → speech sample reconstruction, < 10 ms per frame.**
  Extended Data Fig. 2 gives the cumulative per-stage breakdown; brain-to-voice inference is minor
  relative to audio-driver processing.

**Transfer verdict: TRANSFERS DIRECTLY, and it is the fix for F1.** The same lab, on the same
participant, runs causal rolling normalization when the system is actually real-time. Adopting it
converts this project's weakest point into a defensible measurement. Their 1.5 s causal smoothing
kernel is also a strong hint that this repo's 100 ms smoother may be far too short — an
independent reason to sweep `smooth_kernel_std` ([AUDIT F2](AUDIT.md#f2)).

### B3.6 — Khanday et al., *End-to-End Intracortical Speech Decoding from Neural Activity*
- URL: https://arxiv.org/abs/2605.24313 — **2026-05-23**. Fresh.
- Conformer, **no external LM**, **23.80% CER** on held-out validation. ALS participant.
- Motivation stated as reducing "memory, computation, and latency" from external LMs.
- Same group as B3.2 (Gonzalez-Lopez).

**Value: it prices the LM.** 23.80% CER with no LM versus single-digit WER with one. Even
allowing for CER-vs-WER, the LM is worth the overwhelming majority of end-to-end accuracy in this
task. **This is the argument against any "drop the LM to save latency" proposal**, and it should be
cited whenever the temptation arises. Directionally it also cautions that B3.2/B3.6's absolute
numbers run well behind the field.

**Transfer verdict: TRANSFERS AS A BOUND, not as a method.**

### B3.7 — Other 2026 work surfaced (lower priority, logged for completeness)
- *Decoding the decoder: Contextual sequence-to-sequence modeling for intracortical speech
  decoding* — https://arxiv.org/abs/2603.20246 (2026-03). Not yet read.
- *Generalizable, real-time neural decoding with hybrid state-space models* —
  https://arxiv.org/abs/2506.05320. States the three requirements framing: robust predictions,
  **"causal, low-latency inference viable in an online setting"**, cross-subject generalization.
  Useful framing citation.
- *A cross-species neural foundation model for end-to-end speech decoding* —
  https://arxiv.org/abs/2511.21740.
- *Long-term independent use of an intracortical BCI for speech and cursor control* —
  Nature Medicine 2026, https://www.nature.com/articles/s41591-026-04414-6.

### B3.8 — Normalization convention in this literature
Corroborating [AUDIT F1](AUDIT.md#f1) from secondary sources: offline intracortical speech datasets
conventionally have "neural features **z-scored per block**," while closed-loop systems apply
"feature normalization … to account for neural nonstationarities (drifts in mean firing rate)
which could arise over the course of a block." The general ML-hygiene principle is also stated
explicitly in this literature: "estimating preprocessing statistics on training data only to
prevent test-set leakage and optimistic bias."

So the block z-scoring measured in `data/hdf5_data_final` is **the field's normal offline
practice** — this project is not uniquely at fault. But it *is* incompatible with a real-time
latency claim, and the same lab's own real-time system (B3.5) shows what to do instead.

---

## B4 — Streaming-ASR emission-latency mechanisms

**Verdict for the whole family, stated first because it determines whether ~6 candidate runs are
worth spending.**

Every method below attacks the same problem: CTC/transducer models learn to **delay** emission to
absorb future context, producing 100–300 ms of emission latency on top of the architectural
lookahead. All were developed for 10–40 ms frame rates in acoustic ASR.

**This system very likely does not have that problem, for three independent reasons:**
1. It emits at an **80 ms cadence** with `L_algo = 0`. There are only ~12.5 output frames/s; a
   1-frame emission delay is 80 ms and would be conspicuous.
2. The **causal model already matches or beats the symmetric one** (10.04% vs 10.21%). A model
   exploiting emission delay heavily would degrade when the future context is removed.
3. Each frame already integrates a **280 ms patch** of past context, so there is little pressure
   to defer.

**This upgrades reviewer prior S6 from "measure first" to "expect a null result."** Measure the CTC
peak delay of `causal_la0` against ground-truth phoneme timing before spending anything here. If
the delay is ≲1 frame, **B4.1–B4.6 are all solving a problem this system does not have**, and those
runs should be reallocated — most obviously to time masking (B3.4).

| # | Method | URL | Date | Stale? | Mechanism (1 sentence) | Reported tradeoff | Open impl. | Verdict |
|---|---|---|---|---|---|---|---|---|
| B4.1 | **FastEmit** | https://arxiv.org/abs/2010.11148 | 2020-10-21 (rev 2021-02-03) | **YES — 5.8 yr** | Sequence-level emission regularization on per-sequence probability in **transducers**; no alignment needed | LibriSpeech 4.4/8.9 → 3.1/7.5 WER; **p90 latency 210 → 30 ms** | Yes (Lingvo/ESPnet) | **DOES NOT APPLY** — transducer-only; no transducer here |
| B4.2 | **Delay-penalized CTC via FST** | https://arxiv.org/abs/2305.11539 | 2023-05-19, INTERSPEECH'23 | **YES — 3.2 yr** | Binary attribute on CTC topology marks first non-blank frames; adds frame offsets to log-probs via differentiable FST | "balances delay-accuracy trade-off"; no numbers in abstract | **Yes — k2** (github.com/k2-fsa/k2) | **CONDITIONAL** — the right tool *if* S6 finds delay; otherwise skip |
| B4.3 | **Delay-penalized transducer** (Kang et al.) | https://arxiv.org/abs/2211.00490 | 2022-11-01, ICASSP'23 | **YES — 3.7 yr** | Penalize symbol delay in transducer without external alignments | "significantly reduce symbol delay with acceptable degradation" | Yes — k2 | **DOES NOT APPLY** — transducer-only |
| B4.4 | **Bayes-Risk CTC** | https://arxiv.org/abs/2210.07499 | 2022-10-14 | **YES — 3.8 yr** | Bayes-risk function weights all CTC paths, making alignment preference controllable | controllable alignment; task-dependent | Yes (ESPnet) | **CONDITIONAL** — most general of the family |
| B4.5 | **Peak-First CTC** | https://arxiv.org/abs/2211.03284 | 2022-11-07 | **YES — 3.7 yr** | Frame-wise KD term forces the CTC distribution to shift **left** along time | peak-latency reduction | Partial | **CONDITIONAL** |
| B4.6 | **TrimTail** | https://arxiv.org/abs/2211.00522 | 2022-11-01 | **YES — 3.8 yr** | Trims trailing frames of the **input**; loss- and architecture-agnostic, no alignment | **100–200 ms** latency reduction at equal/better accuracy; 400 ms USD gain at <0.2 WER cost | Yes (WeNet) | **CONDITIONAL — cheapest to try.** Loss-agnostic, ~10 lines, no architecture change |
| B4.7 | **Align-With-Purpose** | https://arxiv.org/abs/2307.01715 | 2023-07-04, ICLR 2024 | **YES — 3.1 yr** | Plug-and-play auxiliary CTC loss term ranking alignment pairs by any desired property | property-dependent | Yes | **CONDITIONAL** — most flexible; needs a property to optimize |
| B4.8 | **U2++** | https://arxiv.org/abs/2106.05642 | 2021-06-10 | **YES — 5.1 yr** | Unified two-pass; dynamic chunk training gives one model for streaming and non-streaming | one model, both modes | **Yes — WeNet** | **PARTIAL** — dynamic-chunk *training* is a real regularizer; the two-pass attention rescoring is a batch stage |
| B4.9 | **Fast-U2++** | https://arxiv.org/abs/2211.00941 | 2022-11-02 | **YES — 3.7 yr** | Small chunk in bottom encoder layers for early partials, large chunk on top to recover accuracy | **latency 320 → 80 ms**, CER 5.06% streaming | Yes — WeNet | **PARTIAL** — the layer-wise-chunk idea is elegant but presumes a chunked encoder |
| B4.10 | **CUSIDE-T** | https://arxiv.org/abs/2407.10255 | 2024-07-14, SLT'24 | **YES — 2.1 yr** | Context-sensitive chunking + a lightweight module that **simulates** future context from history, jointly trained | matches/beats U2++ at lower latency | Yes | **INTERESTING** — "simulate the future instead of waiting for it" is conceptually the right shape for a hard latency cap; transducer-oriented |
| B4.11 | **Zipformer** | https://arxiv.org/abs/2310.11230 | 2023-10-17 | **YES — 2.8 yr** | U-Net encoder at mixed frame rates, reused attention weights, BiasNorm, SwooshR/L | faster + better + less memory than Conformer | **Yes — icefall** | **CANDIDATE ENCODER** — but see B3.4: a plain causal transformer already suffices |
| B4.12 | **Emformer** | ICASSP 2021 | 2021 | **YES — 5 yr** | Memory-augmented transformer with cached summary vectors for low-latency streaming | low-latency streaming AM | Yes (torchaudio) | **SUPERSEDED** by Zipformer / B3.4 |
| B4.13 | **Delayed-KD** | https://arxiv.org/abs/2505.22069 | 2025-05-28 | Fresh-ish (14 mo) | Non-streaming **teacher** → streaming **student** via CTC posteriors, with a **Temporal Alignment Buffer** defining a relative delay range to align outputs | **5.42% CER at 40 ms** latency vs U2++ needing **320 ms** for similar accuracy (AISHELL-1) | Not found | **CLOSEST TO S4** — see below |

**B4.13 vs. reviewer prior S4 (predictive distillation).** Delayed-KD is S4, already published:
non-streaming teacher, streaming student, KD on CTC posteriors, with an explicit delay-alignment
mechanism. S4's proposed differentiator — deriving the shift from *measured* CTC peak positions
rather than a hyperparameter — is a genuine but **narrow** delta against a Temporal Alignment
Buffer that already sweeps a delay range. S4 should be re-scoped or demoted; it is not the
novel contribution the brief hoped for. It is also doubly exposed: if S6 finds no emission delay,
there is nothing for the alignment shift to correct.

**Lookahead convolution layers** (explicit, exactly-quantifiable future-context purchase): this is
a classical technique (Deep Speech 2 lineage) rather than a single citable modern paper.
**Mechanically it is the cleanest way to buy a *tunable* amount of lookahead** — k future frames
at exactly 20k ms of L_algo — and it is the natural instrument for building a genuine
accuracy-vs-latency curve, which is what [AUDIT F2](AUDIT.md#f2) shows the project currently lacks.
**[UNVERIFIED — no primary citation retrieved.]**

---

## B5 — Incremental LM decoding

**The headline is in [AUDIT F4](AUDIT.md#f4), not in the literature: the incremental WFST beam
search already exists in this repository and works.** `ctc_wfst_beam_search.cc:98` advances Kaldi
one frame at a time and `:113` extracts a partial best path after every call. The brief's premise
that this is "the least-explored area" is inverted — it is the most-already-built area. What is
missing is **measurement**, not implementation.

| # | Method | URL | Date | Stale? | Extracted numbers | Verdict |
|---|---|---|---|---|---|---|
| B5.1 | **Blank Collapse** (Jung et al., INTERSPEECH'23) | https://arxiv.org/abs/2210.17017 | 2022-10-31 | **YES — 3.8 yr** | θ ∈ {0.9, 0.99, 0.999}; **43–46% of frames removed**; RTF 0.291 → 0.163 (**~44% faster**) at WER 1.783 → 1.781 (unchanged); **"~33% decoding-time reduction with WFST-based decoding"**; up to 78% speedup in optimal conditions | **TRANSFERS.** Directly validates enabling `blank_skip_thresh` ([AUDIT F12](AUDIT.md)) — a **config change, zero retrain**. Note the WFST figure (33%) is the relevant one and is lower than the end-to-end figure |
| B5.2 | **WeNet 2.0** | https://arxiv.org/abs/2203.15455 | 2022-03-29 | **YES — 4.4 yr** | Compiles n-gram (G) + lexicon (L) + CTC topology (T) into the TLG WFST; blank frame skipping "adopted to speed up CTC WFST beam search" | **THIS IS THE STACK IN THIS REPO.** `language_model/wenet/` is a vendored copy. Its docs are the authoritative reference for the decoder's behaviour |
| B5.3 | **Spike Window Decoding** | https://arxiv.org/abs/2501.03257 | 2025-01-06 | Fresh-ish (19 mo) | Makes WFST-decoded frame count **linearly related to spiking frames** rather than total frames | **TRANSFERS** — a stronger version of B5.1 |
| B5.4 | Blank-regularized CTC for frame skipping in neural transducer | (INTERSPEECH'23) | 2023 | **YES** | Trains the model to produce *more* blanks so more frames are skippable | **CONDITIONAL** — requires a retrain; only worth it if B5.1 shows the LM is the bottleneck |
| B5.5 | IOO / KOO blank handling | (secondary source) | — | — | Collapse blank runs + dedup non-blank spikes; **2.4× speedup, no accuracy loss** | **[UNVERIFIED — no primary source retrieved]** |

**The "blank probability stabilizes near 70%" figure** quoted in the brief is **not corroborated by
any retrieved source** — B5.1's published figure is 43–46% of frames removable at θ = 0.99, on
acoustic ASR at a much finer frame rate — **but it is confirmed by direct measurement.**
[BUDGET §5](BUDGET.md) measures **71.83% of frames with p(blank) > 0.999** and a 75.36% blank
argmax fraction on `causal_la0` over the full val split.

*(Correction: this entry originally predicted the blank fraction would be **lower** than acoustic
ASR's, reasoning that 80 ms patches carry more content per frame. That was wrong. The dominant
factor is the opposite one — this task has long pre-speech and trailing silence, so blank frames
dominate more heavily, not less. The brief's 70% figure was right.)*

**Not found despite targeted search, and therefore genuinely open:** any published measurement of
**per-frame incremental** n-gram/WFST decoding cost for a brain-to-text system. The only LM timing
anchor retrieved anywhere in this review is B3.4's **17 ± 11 ms per trial** for an offline 3-gram
beam search at width 18. Nobody reports the streaming number. That absence is an opportunity, and
it is the measurement Phase C must produce.

---

## Sources

- [Brain-to-Text Benchmark '24: Lessons Learned (arXiv:2412.17227)](https://arxiv.org/abs/2412.17227)
- [Phoneme- vs. Character-Level Targets and Selective SSMs (arXiv:2607.26751)](https://arxiv.org/abs/2607.26751)
- [End-to-End Intracortical Speech Decoding (arXiv:2605.24313)](https://arxiv.org/abs/2605.24313)
- [Time-Masked Transformers with Lightweight TTA (arXiv:2507.02800)](https://arxiv.org/abs/2507.02800)
- [Neural decoding of speech using deep neural ensembles (bioRxiv 2026.06.02.729705)](https://www.biorxiv.org/content/10.64898/2026.06.02.729705v1)
- [Cross-subject decoding of human neural data for speech BCIs (J. Neural Eng. 2026)](https://iopscience.iop.org/article/10.1088/1741-2552/ae8576)
- [An instantaneous voice-synthesis neuroprosthesis (Nature 644:145–152, 2025)](https://www.nature.com/articles/s41586-025-09127-3)
- [Wairagkar et al. preprint (PMC11370360)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11370360/)
- [brain-to-voice-2025 code](https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025)
- [7th place solution — Mamba + GRU + KenLM](https://medium.com/@jackson3b04/7th-place-solution-mamba-gru-kenlm-with-code-brain-to-text-25-00f1c69dcd0d)
- [Brain-to-text '25 competition](https://www.kaggle.com/competitions/brain-to-text-25/overview)
- [FastEmit (arXiv:2010.11148)](https://arxiv.org/abs/2010.11148)
- [Delay-penalized CTC via FST (arXiv:2305.11539)](https://arxiv.org/abs/2305.11539)
- [Delay-penalized transducer (arXiv:2211.00490)](https://arxiv.org/abs/2211.00490)
- [Bayes risk CTC (arXiv:2210.07499)](https://arxiv.org/abs/2210.07499)
- [Peak-First CTC (arXiv:2211.03284)](https://arxiv.org/abs/2211.03284)
- [TrimTail (arXiv:2211.00522)](https://arxiv.org/abs/2211.00522)
- [Align With Purpose (arXiv:2307.01715)](https://arxiv.org/abs/2307.01715)
- [U2++ (arXiv:2106.05642)](https://arxiv.org/abs/2106.05642)
- [Fast-U2++ (arXiv:2211.00941)](https://arxiv.org/abs/2211.00941)
- [CUSIDE-T (arXiv:2407.10255)](https://arxiv.org/abs/2407.10255)
- [Zipformer (arXiv:2310.11230)](https://arxiv.org/abs/2310.11230)
- [Delayed-KD (arXiv:2505.22069)](https://arxiv.org/abs/2505.22069)
- [Blank Collapse (arXiv:2210.17017)](https://arxiv.org/abs/2210.17017)
- [WeNet 2.0 (arXiv:2203.15455)](https://arxiv.org/abs/2203.15455)
- [Spike Window Decoding (arXiv:2501.03257)](https://arxiv.org/abs/2501.03257)
- [Generalizable, real-time neural decoding with hybrid SSMs (arXiv:2506.05320)](https://arxiv.org/abs/2506.05320)
