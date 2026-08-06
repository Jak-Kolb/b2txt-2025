# Design Review: Best Achievable Accuracy for a Real-Time Intracortical Brain-to-Text Decoder

**Author role:** independent external architect (literature + constraints; no code).  
**Date context:** 2026-08-04. Sources verified via live search; claims that could not be confirmed are marked **UNVERIFIED**.

---

## 1. Verdict up front

**Ship the existing causal GRU/CTC acoustic stack almost unchanged; spend the 76× compute headroom on causal deep ensembling + phase-staggered emission + streaming n-gram fusion with entropy-triggered rescoring. Do not re-architect the encoder first.** The accuracy–latency frontier this project was designed to map (smooth lookahead) is flat (Fact A); the binding levers are (i) variance reduction across independently trained causal members, (ii) L_buf reduction via phase-stagger without increasing L_algo, and (iii) moving language modeling from an end-of-utterance wall into the 60 ms of unused budget per cadence. Predicted real-time-feasible operating point on the reference full-stack metric: **test WER 3.2–4.5%** (conservative ensemble + streaming 3-gram, no EOU LLM) and **2.4–3.5%** with adaptive partial-hypothesis LLM rescoring inside the 140 ms finalization budget on a 3090; acoustic val PER **9.4–9.9%** under N=8 logit-average (estimated). Headline recommendations:

- **R1 — Causal deep ensemble (N=6–8) with pre-beam logit average:** Δval PER **−0.4 to −0.9 pp**; Δtest WER **−0.6 to −1.4 pp** vs single causal seed; L_algo +0, L_buf +0, L_comp ≈ N×0.34 ms ≪ 80 ms.
- **R2 — Streaming n-gram / incremental beam + blank-frame skip; reserve LLM for high-entropy partials only (adaptive-compute decoding):** closes most of the unconstrained–real-time gap on *finalization* WER while keeping expected emission latency inside budget; predicted **+0.3–0.8 pp WER** vs full EOU LLM stack (estimated), not the multi-point penalty of dropping LM entirely.
- **R3 — Phase-staggered replicas (K=4) before larger encoders:** cuts L_buf 80→20 ms and simultaneously acts as a free ensemble; predicted emission latency **~25–40 ms** (derived) with ΔPER comparable to small N ensemble (estimated −0.2 to −0.6 pp if phase diversity is real).

**Strongest disagreement with Section 1:** the objective is stated as test WER under latency, but for a streaming clinical decoder the *user-visible* quantity is word-level **stability** (time-to-final and revisions per word). If PER is flat and final WER is nearly flat under causal AM changes, the paper is the non-degenerate **stability–latency** curve produced by adaptive rescoring thresholds—not another 0.2 pp PER chase. Second disagreement: Section 1 underweights that **the LM/beam stage, not the acoustic model, is the real-time bottleneck and the contribution surface**; Fact B+C already imply this.

---

## 2. Where the gains are

### 2.1 The frontier moved

Fact A is a genuine negative result for *this* control surface: truncated half-Gaussian smoother lookahead L∈{0,4} does not trade accuracy for latency under an otherwise-identical protocol (val PER 10.04% vs 10.21%, n=1 each, gap inside seed noise). Emission-delay regularization tools from streaming ASR (FastEmit, delay-penalized CTC, Peak-First, TrimTail, Bayes-Risk early-emission variants) only pay when the model is *actually* exploiting future context via late peaks. **Until CTC peak delay is measured on the causal checkpoint (S6), any of those runs is gambling on a problem that may not exist.** Fact A makes that measurement the highest information-per-hour experiment available.

### 2.2 Compute is free; data is not

Streaming RTF ≈ 0.0132 (measured) means an 80 ms cadence window has ~79.7 ms idle after the AM forward. On a 3090, that is order **~230–250 model-forward-equivalents per cadence** at the measured 0.34 ms/patch (derived: 80/0.34 ≈ 235). Ensembling, phase-stagger, larger beams, and even a small causal neural LM shallow-fusion pass are all *inside* that budget long before RTF approaches 1. Capacity scaling of a single model (S2) is the wrong way to spend headroom first: the '24 retrospective and every controlled 2025–2026 architecture study on this benchmark family conclude the regime is **regularization- and data-limited**, not capacity-limited (Willett et al. arXiv:2412.17227, Dec 2024; Feghhi et al. arXiv:2507.02800, Jul 2025; Zamora et al. arXiv:2607.26751, Jul 2026).

### 2.3 Where competitions actually won

Unconstrained SOTA on Brain-to-Text '25 is ~**1.5% WER** (winner) with 4th near **1.78%** and strong single-model public entries near **~2.79%** (competition framing; detailed per-team writeups with latency/memory columns are sparse—see §7 negative results). The '24 retrospective is explicit: **ensembling independent decoders + fine-tuned LLM merge was the largest gain and used by all top-3** (baseline 9.7% → top 5.8% WER). Architecture swaps (Transformer, deep SSM/Mamba) did not beat the GRU under unconstrained conditions; diphone auxiliary and LR schedule did help secondarily. The June 2026 closed-loop ensemble paper (Yoon et al.) is the decisive transfer proof: deep ensembles work in real-time in a participant, **33.7% → 26.0% WER**, with a single-decoder pseudoensemble (TTA) as the compute-efficient fallback.

The real-time-feasible optimum is therefore **not** “rebuild a bidirectional Conformer ensemble with GPT rescoring.” It is: take the causal AM that already matches the non-causal smoother, **multiply independent causal seeds**, keep L_algo=0, compress L_buf with phase-stagger, and redesign the LM path so finalization is not a batch wall. Gap framing: unconstrained ~1.5% WER vs released full-stack ~2.66% WER leaves ~1.1 pp of “competition magic” that is mostly non-causal diversity + heavy EOU LLM. Expect to close **half to two-thirds** of that gap inside 140 ms; closing all of it requires violating the constraint.

### 2.4 What not to optimize first

- Encoder rewrites (Conformer, Mamba hybrid, deeper GRU) as primary lever: repeatedly null or worse at fixed protocol (Linderman '24; Zamora '26; Willett discussion). Feghhi’s time-masked Transformer *did* win on T12 data—but primarily via **time-masking regularization + training recipe**, and their ablation shows time-masked GRU nearly matches (4% relative WER gap). Transfer the recipe, not the architecture brand.
- Delay-penalized CTC family before measuring peak delay (S6).
- Unpruned 5-gram (~300 GB per public competition writeup lore; Feghhi reports ~60 GB even for a 3-gram setup on '24)—ruled out by 32 GB deployment RAM.
- Bundled multi-knob retrains (already rejected correctly).

---

## 3. Recommended pipeline

Running latency total is tracked as **Σ = L_algo + L_buf + L_comp** for *emission-relevant* path. Cold-start is separate. All L_comp numbers: **measured** for current AM; **derived/estimated** elsewhere as labeled.

### 3.1 Stage-by-stage (Conservative — **ship this**)

| Stage | Specification | L_algo | L_buf | L_comp | Params / mem | Replaces | Independent? | Running Σ (emission) |
|---|---|---:|---:|---:|---|---|---|---:|
| Feature extraction | Keep: 20 ms bins, threshold crossings + spike-band power, 512-D | 0 | 0 | ~0 (online) | — | nothing | yes | 0 |
| Normalization | Keep: causal rolling past-10 s (verified no block z-score); optional log1p pre-zscore as in Feghhi if ablation free | 0 | 0 | ~0 | — | nothing | yes | 0 |
| Temporal smoothing | **Causal only:** smooth_lookahead=0 half-Gaussian (peak at current bin). Do not return to L=4 for production | 0 | 0 | ≪0.1 ms | — | L=4 smoother | yes | 0 |
| Patching | Keep P=14, s=4, right-edge emission. Cold start 280 ms once. Do **not** left-pad | 0 | **80** | 0 | — | nothing | package w/ phase-stagger optional | 80 |
| Encoder | Keep ~45 M unidirectional GRU stack + day affine layers. **Do not widen first.** Optionally add LayerNorm stack after GRU as post-comp Linderman tweak (one isolated retrain) | 0 | 0 | **0.34** ms (p95; measured) | ~45 M; ≲2 GB train activations est. | nothing | yes | 80.3 |
| Ensemble (new) | **N=6–8 independent seeds**, phase-aligned, **mean logits** before beam. All causal. Optional SWA within each seed | 0 | 0 | **2.0–2.7** ms (derived N×0.34) | N×45 M weights on GPU; fits 24 GB easily | single seed | package with shared beam | 82–83 |
| Output head | Linear → phoneme+blank(+silence) CTC | 0 | 0 | in AM | small | nothing | yes | 82–83 |
| CTC decoding | Streaming beam over WFST/lexicon; **blank-frame skip** when P(blank)>τ_b≈0.7; beam width sweep first | 0 | 0 | **5–25** ms/update **estimated** (depends on beam, CPU vs GPU, blank skip) | beam state small | offline full-pass beam | yes | 87–108 |
| LM fusion | **Pruned 3-gram in ~6–8 GB RAM** (deployment target). Acoustic scale, LM scale, blank penalty: eval-only sweep. Optional 4-gram aggressively pruned if ≤16 GB resident | 0 | 0 | in beam | 6–8 GB CPU | unpruned 5-gram | yes | 87–108 |
| Rescoring | **Not EOU-default.** Entropy/gate-triggered mini-LLM rescoring over top-B partial or finalized hypotheses only (R2). Target expected L_comp rescoring **≪20** ms amortized | 0–20 (policy) | 0 | **0–40** ms **estimated** when triggered | 7–8 B QLoRA LLM optional; or 100–300 M causal neural LM | always-on EOU LLM | works alone | 87–128 |
| Endpointing | Energy/blank-run + LM endpoint; freeze words after T_stable ms without revision | 0 | **0–40** policy | ~0 | — | hard EOU wait | yes | ≤140 |
| Text emission | Partial hypotheses every cadence; freeze-stable prefix; never wait for full utterance for first paint | 0 | 0 | ~0 | — | batch emission | yes | ≤140 |

**Conservative total (typical emission):** L_algo=0 + L_buf=80 + L_comp≈8–30 ≈ **88–110 ms**, slack **30–52 ms**.  
**Conservative finalization (word freeze):** +0–40 ms policy + rare rescoring; design so p95 finalization ≤140 ms.

**What it replaces and why it wins under the constraint:** does not throw away a validated ~10% PER causal AM; spends idle GPU on the only lever competitions and closed-loop work agree on (ensembling); removes the EOU LLM wall that zeros the latency score of a 1.5% WER pipeline.

### 3.2 Aggressive variant (willing to restructure)

| Delta | Spec | Latency effect | Risk |
|---|---|---|---|
| Phase-stagger K=4 | One trained model evaluated at patch phase offsets {0,20,40,60} ms **if** embedding is phase-agnostic; else train K phase-conditioned adapters or K full seeds | **L_buf 80→20**; L_comp ×4 still ~1.4 ms; collective ensemble | High if patch embed is phase-locked |
| Streaming neural LM shallow fusion | Small causal Transformer-LM (≤100 M) on phoneme or word pieces, fused every emission | +2–8 ms L_comp **est.**; L_algo=0 | Med; may not beat n-gram at this data scale |
| Time-masked training recipe | N=20 masks, M≈0.075 (~50% avg mask), longer schedule, LR step (Feghhi) on GRU or compact causal Transformer | 0 latency | Low–med |
| Hierarchical intermediate CTC | Boccato-style multi-block GRU with intermediate CTC + feedback (unidirectional) | ~0 | Med |
| Cross-subject pretrain | Init from joint T12+T15 hierarchical model, freeze body, train day affines | 0 latency | Med (data access, protocol mismatch) |
| Adaptive dual path | Always: cheap ensemble+3-gram. Trigger: full beam widen + LLM when H(partial)>θ | **Expected** L ≪ worst-case; worst-case must still be capped | Med (threshold calibration) |

**Aggressive emission total if phase-stagger works:** L_algo=0 + L_buf=20 + L_comp≈5–20 ≈ **25–40 ms**, large slack for smarter LM.

### 3.3 Ship verdict

**Ship Conservative.** Aggressive phase-stagger is the first structural experiment *after* ensemble N-scaling curves, not instead of them. Replacing the GRU with Conformer/Mamba/Transformer is **not** clearly better under the constraint and should not consume the first 6.4-hour runs. The existing structure is close to right; the pipeline’s failure mode is **under-use of idle compute and batch-mode LM**, not wrong inductive bias in the recurrent core.

If Conservative exceeds budget, cut first: (1) LLM rescoring entirely, (2) beam width, (3) ensemble N from 8→4, never the causal AM.

---

## 4. Recommendations (≥3)

### R1 — Causal deep ensemble with logit averaging (primary contribution)

1. **Claim:** An ensemble of N=8 independently trained fully causal GRU seeds, mean-pooled in logit space before streaming beam search, reduces val PER by **0.5–0.9 pp** (to ~9.1–9.5%) and full-stack test WER by **0.6–1.4 pp** relative to the best single causal seed, with **L_algo = L_buf = 0 incremental** and streaming RTF still ≪ 1.  
2. **Why under constraint:** L_comp ≈ 8×0.34 = 2.72 ms (derived) vs 80 ms cadence → RTF_AM ≈ 0.034; literature’s largest gain source (Willett '24; all top-3 '24; Yoon '26 closed-loop 33.7→26.0% WER). Causal members are more correlated than bi-directional ones—**expected gain is smaller than competition writeups**, hence the conservative Δ range, not rubber-stamped 2× competition deltas.  
3. **Implementation spec:** Train N seeds with identical protocol (smooth_lookahead=0), seeds {0…N−1}. Checkpoint by val PER. At inference: run N forwards (batch or sequential), average log-probs (or probs—pre-register which; prefer log-prob mean then renorm). Single shared CTC beam + pruned 3-gram. Verify streaming equivalence: max |logit_ens_stream − logit_ens_offline| < 1e−3 on 10 trials. Optional: diversify with mild hyperparameter jitter (dropout ±0.05, mask rate) for 2 of N members only after plain multi-seed baseline.  
4. **Pre-registered decision rule:** H1 supported if mean val PER over the N members’ ensemble ≤ **9.60%** on ≥2 of 3 ensemble constructions (different seed partitions), and streaming equivalence maintained at <1e−3; WER secondary: ensemble test WER ≤ single-seed test WER − 0.4 pp on the fixed full stack (or streaming stack if that becomes primary).  
5. **Compute plan:** 8×6.4 h = **51.2 GPU-h** for members (can parallelize); 1 eval-only suite for N∈{1,2,4,8} curve (no retrain). Dependency: after seed-variance baseline (R0/S7).  
6. **Failure mode / fallback:** If multi-seed ΔPER < 0.2 pp → members too correlated; fallback to **pseudoensemble TTA** (Yoon; Feghhi DietCORP-style multi-mask at test time, single weights) and/or diversify with diphone-auxiliary member + time-masked member.  
7. **Novelty:** Ensemble gains known offline; **closed-loop deep ensemble exists (Yoon 2026)**; what is still paper-grade is **causal-only, logit-average, streaming-equivalence-guaranteed ensemble on the Card/T15 stack with explicit 140 ms budget accounting**—especially N-scaling under correlation.

### R2 — Incremental LM decoding + adaptive-compute rescoring (relocates the contribution)

1. **Claim:** Replacing always-on end-of-utterance LLM rescoring with (a) continuous pruned 3-gram shallow fusion in the beam and (b) LLM rescoring only when partial-hypothesis entropy H_t > θ yields test WER within **+0.3–0.8 pp** of the full EOU LLM stack while keeping **p95 finalization latency ≤140 ms** and producing a **non-degenerate WER-vs-latency curve** over θ.  
2. **Why under constraint:** Fact C + Feghhi’s note that even 3-gram setups dominate memory; beam/LM is where latency lives. Emission path never waits on LLM; finalization path pays only on uncertain words. Arithmetic: AM+ensemble ≤3 ms; blank-skipped beam **estimated** 5–25 ms; LLM fire rate f(θ) if f=0.1 and cost 50 ms → amortized 5 ms → total ~20–35 ms + L_buf.  
3. **Implementation spec:** (i) Port/ensure KenLM or equivalent supports **incremental** query as frames arrive; (ii) blank skip: if max blank prob > 0.7 for consecutive frames, advance beam time without expansion (competition writeups report ~70% blank mass—treat as **estimated** until measured here); (iii) emit partial argmax path each cadence; (iv) compute token-level or hypothesis-level entropy; (v) if H>θ, rescore top-B with frozen LLM (start with small open model, not GPT API); (vi) endpoint: freeze prefix when unchanged for T_stable∈{120,200,280} ms. Sweep θ, B, beam, acoustic/LM scales **eval-only**.  
4. **Pre-registered decision rule:** H1 if there exists θ such that test WER ≤ EOU-LLM WER + 0.8 pp **and** p95 finalization latency ≤140 ms **and** mean revisions/word ≤ 0.15 (metric §6). Secondary: area under WER–latency curve better than hard EOU baseline at the 140 ms vertical line.  
5. **Compute plan:** **0 retrain** for core; 1–2 days eng + eval GPU for sweeps. Optional: 1×6.4 h if training a small causal neural LM for shallow fusion.  
6. **Failure mode / fallback:** If incremental beam quality collapses vs batch beam → fix blank-skip/beam pruning bugs first; fallback to **cadence-synchronous full beam** every 80 ms without LLM (accept WER floor). If LLM never triggers usefully → drop LLM, invest in better n-gram order under 16 GB.  
7. **Novelty:** Streaming ASR incremental decoding is old; **brain-to-text papers almost never report partial-hypothesis stability or adaptive rescoring under a hard clinical latency cap**. High paper potential if the stability metric is primary.

### R3 — Measure CTC peak delay; only then consider emission regularization (gate + possible contribution)

1. **Claim:** The causal model’s mean CTC peak delay relative to forced-aligned phoneme midpoints is **already ≤40 ms** (predicted from Fact A); if true, FastEmit / delay-penalized CTC / Peak-First / TrimTail each yield **ΔPER ∈ [−0.1, +0.3] pp** (null) and should not consume retrain budget. If mean delay **>80 ms**, delay-penalized CTC (k2) is expected to cut emission latency **30–100 ms** (ASR literature order) at **≤0.3 pp PER cost**.  
2. **Why under constraint:** Fact A implies the model may not be “waiting” for future context; emission regularizers solve a different problem than flat smoother lookahead. Measuring is free of retrain.  
3. **Implementation spec:** Forced-align phonemes (e.g., from transcripts + lexicon) to time; for each non-blank CTC peak after collapse-aware peak picking, record t_peak − t_phoneme_mid; report mean/median/p90 in ms on val. Secondary: blank run lengths, emission cadence vs phoneme rate.  
4. **Pre-registered decision rule:** If median peak delay ≤40 ms → **kill** FastEmit, Peak-First, TrimTail, delay-CTC runs (do not train). If median ≥80 ms → authorize 1 seed delay-penalized CTC (k2) with λ delay grid.  
5. **Compute plan:** **0 GPU-h train**; hours of analysis. Contingent: 1×6.4 h only if gate opens.  
6. **Failure mode:** Alignment quality poor → use diphone or energy proxy; still report distribution.  
7. **Novelty:** Negative result (no delay to penalize) is publishable as a **boundary condition** on transferring streaming-ASR alignment shaping to intracortical CTC; positive result enables a clean constrained paper.

### R4 — Phase-staggered replicas for sub-cadence resolution (novel structure)

1. **Claim:** Evaluating K=4 phase offsets of one causal model reduces effective L_buf from 80 to 20 ms and yields ensemble-like ΔPER **−0.2 to −0.6 pp** if phase diversity is nontrivial; total L_comp ≈1.4 ms.  
2. **Why under constraint:** Converts Fact B headroom into **latency**, un-flattening the frontier Fact A left flat. Emission Σ ≈ 20 + 1.4 + beam ≈ **30–50 ms**.  
3. **Implementation spec:** Define patch start offsets o∈{0,1,2,3} bins. Prefer **single weights**, evaluate on shifted streams (day layer + smoother phase-agnostic—**assumption**). If patch embedding encodes absolute phase within the 14-bin window, either (a) train with random phase offset augmentation, or (b) K small phase adapters, or (c) K full seeds at fixed phases. Merge: average logits on the common 20 ms grid (interpolate/hold).  
4. **Pre-registered decision rule:** H1 if L_buf_effective = 20 ms verified in streaming harness **and** val PER ≤ single-phase PER + 0.1 (no damage) **and** preferably ≤ single-phase − 0.2; streaming equivalence retained.  
5. **Compute plan:** 0–1 retrain if phase aug needed; else eval-only. After R1 N-curve.  
6. **Failure mode:** Phase-locked embed → PER worsens when shifted → train with random phase offsets (1 run) or abandon for pure N-ensemble.  
7. **Novelty:** **No known prior** in brain-to-text (author’s S3; this review agrees). High paper value if it works; clean negative if not.

### R5 — Seed protocol (methodological, mandatory)

1. **Claim:** The 0.17 pp PER gap (L=0 vs L=4) is **not distinguishable** from seed noise; headline configs require **n=3 seeds**. Expected seed SD for val PER on this stack: **~0.15–0.35 pp** (estimated from Benster/LISA '24 multi-seed WERs and Feghhi SEMs).  
2. **Why:** Without error bars every “improvement” is fiction.  
3. **Spec:** 3 seeds for: causal baseline, any config within 0.3 pp of baseline, and final ship ensemble members (members *are* seeds).  
4. **Decision rule:** Report mean±SD; claim improvement only if meanΔ ≥ 0.3 pp PER **and** same sign on ≥2/3 seeds.  
5. **Compute:** +2×6.4 h on baseline causal; built into R1.  
6. **Fallback:** If budget tight, n=2 on headlines only.  
7. **Novelty:** none; necessary science.

---

## 5. Candidate catalog (≥20)

Score = expected accuracy gain per GPU-hour **conditional on staying inside 140 ms**. Higher is better. Sort descending.

| # | Name | Mechanism (2–3 sentences) | Source | E[ΔPER] | E[ΔWER] | L_algo/L_buf/L_comp | Mem train/inf | Train cost (6.4h runs) | Risk / failure | Do not bundle with | Kill criterion | Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Eval sweeps (beam, scales, blank pen.) | Exhaust decoding hyperparameters on fixed logits | ASR standard | 0 to −0.1 | **−0.2 to −0.8** | 0/0/var | 0 / +beam | **0** | Low | nothing | no WER gain at equal latency | **10** |
| 2 | Checkpoint avg / SWA | Average last-k or SWA weights of one run | standard | −0.1 to −0.3 | −0.1 to −0.4 | 0/0/0 | 0 | **0–0.2** | Low | multi-seed confusion | ΔPER > −0.05 | **9** |
| 3 | CTC peak delay measurement | Quantify emission delay vs forced align | S6; novel as gate | n/a (info) | n/a | 0 | 0 | **0** | Low | delay-CTC family | n/a | **9** |
| 4 | Blank-frame skip in beam | Skip expansion when P(blank) high (~70%) | public writeups; ASR | 0 | 0 to −0.1 (speed enables wider beam) | 0/0/**−50–80% beam time est.** | 0 | **0** | Med (quality) | broken incremental beam | WER↑ >0.3 pp | **8** |
| 5 | Causal deep ensemble N=8 logit avg | Independent causal seeds; mean logits | Willett'24; Yoon'26; S1 | **−0.4 to −0.9** | **−0.6 to −1.4** | 0/0/+2.4 ms | N×weights | **8** | Med correlation | bi-dir members | ΔPER > −0.2 at N=8 | **7.5** |
| 6 | Incremental 3-gram beam | Streaming KenLM/WFST partial emission | ASR; underexplored in B2T | 0 | **−0.5 to −2 vs greedy**; vs EOU LLM +0.3–1.0 | 0/0/+5–25 ms est. | 6–8 GB | **0** | Med eng | unpruned 5-gram | cannot fit 32 GB or RTF>1 | **7.5** |
| 7 | Adaptive LLM rescoring (θ) | Rescore only high-entropy partials | S5; novel curve | 0 | **−0.3 to −1.0 vs no LLM** | 0/0/amortized | LLM RAM | **0** | Med | always-on EOU | no θ meets 140 ms & WER | **7** |
| 8 | Phase-stagger K=4 | Multi-phase patch eval → 20 ms grid | S3 novel | −0.2 to −0.6 | −0.2 to −0.7 | **0/−60 ms**/+1 ms | ×K compute | 0–1 | **High** phase lock | left-pad patcher | PER worsens or L_buf not reduced | **6.5** |
| 9 | Time-masking heavy | SpecAugment-style ~50% temporal mask | Feghhi'25 | −0.3 to −0.8 | −0.5 to −1.5 | 0/0/0 | 0 | **1–3** | Low–med | capacity 20× | ΔPER > −0.15 | **6** |
| 10 | Pseudoensemble TTA | Multi-mask forward, average logits, 1 weights | Yoon'26 | −0.2 to −0.5 | −0.3 to −0.8 | 0/0/×Z | 1×weights | **0** | Low | full N=10 if VRAM tight | ΔPER > −0.1 | **6** |
| 11 | Pruned/quantized 4-gram | Higher-order LM under 16 GB | competition mem lore | 0 | −0.2 to −0.6 | 0/0/+small | 8–16 GB | **0** | Med quality loss | 300 GB 5-gram | WER↑ vs 3-gram | **5.5** |
| 12 | Small causal neural LM fusion | Shallow fusion vs n-gram | streaming ASR | 0 | −0.2 to −0.8 uncertain | 0/0/+2–8 ms | +0.2–2 GB | **1–2** | Med | huge LLM EOU | no beat 3-gram | **5** |
| 13 | Cross-subject pretrain (T12+T15) | Joint hierarchical GRU + day/dataset affines | Boccato'26 | −0.2 to −0.7 | −0.3 to −1.0 | 0/0/0 | similar | **2–4** | Med protocol | same-day calibration work | no transfer on T15 val | **4.5** |
| 14 | Hierarchical intermediate CTC | Multi-block GRU + mid CTC + feedback, causal | Boccato'26 | −0.2 to −0.6 | −0.2 to −0.8 | 0/0/~0 | ~same | **1–2** | Med | Mamba rewrite | ΔPER > −0.15 | **4** |
| 15 | Diphone auxiliary / DCoND | Context-dependent diphone then marginalize | Li/DCoND'24 | −0.3 to −0.8 PER | −0.3 to −0.7 WER | 0/0/0 | +classes | **1–2** | Med | character head | no PER gain | **4** |
| 16 | Predictive distillation (bi→causal) | Teacher bi-dir; KD with delay-aligned frames | S4; Delayed-KD arXiv:2505.22069 May 2025 | −0.2 to −0.7 | −0.2 to −0.6 | 0/0/0 student | 2× train | **2–3** | High align | delay-CTC simultaneous | student ≱ teacher−ε | **3.5** |
| 17 | LayerNorm + LR schedule on GRU | Linderman post-comp recipe | Willett'24 §2.3 | −0.1 to −0.4 | −0.2 to −0.5 | 0 | 0 | **1** | Low | architecture swap | ΔPER > −0.1 | **3.5** |
| 18 | DietCORP-style TTA | 1-step patch-embed adapt on LM pseudo-label | Feghhi'25 | day-shift WER | large on held-out days | +18 ms/trial Feghhi | 1.3 GiB | **0** (adapt) | Med drift | offline-only eval | hurts same-day WER | **3** |
| 19 | Width/depth scale 5–20× | Spend compute on one bigger GRU | S2 | −0.2 to +0.5 (overfit) | uncertain | 0/0/×5–20 still <10 ms | VRAM | **2–4** | **High overfit** | weak reg | val PER worsens | **2** |
| 20 | Delay-penalized CTC (k2) | FST latency penalty on alignments | arXiv:2305.11539 | 0 to +0.3 if no delay | latency not PER | −L_algo if any | 0 | **1** | High null (Fact A) | Peak-First same run | median delay <40 ms | **1.5** |
| 21 | FastEmit | Sequence-level early emission regularizer | arXiv:2010.11148; Linderman null | ~0 (Linderman) | ~0 | −L if any | 0 | **1** | High null | delay-CTC | median delay <40 ms | **1** |
| 22 | Peak-First / TrimTail / BRCTC early | Peak latency regularizers | arXiv:2211.03284; 2211.00522; 2210.07499 | ~0 here | ~0 | −L | 0 | **1 each** | High null | each other | S6 gate | **1** |
| 23 | Streaming Conformer / Emformer / Zipformer | Chunked attention encoders | WeNet/ESPnet/NeMo | −0.3 to +1.0 | uncertain | +R right ctx if any | var | **2–4** | High | GRU ensemble first | no beat causal GRU | **1.5** |
| 24 | Mamba / ConvMamba hybrid | Selective SSM core | Zamora'26; Linderman'24 | ~0 to +0.5 worse | worse WER risk | 0 | similar | **2** | High | character targets | no beat GRU | **1** |
| 25 | End-to-end char Conformer no LM | Direct CER optimize | Khanday arXiv:2605.24313 May 2026 | n/a (CER 23.8%) | **much worse** without LM | 0 | ~47 M | **2** | Wrong objective | WER claims | WER ≫ phoneme+LM | **0.5** |
| 26 | U2++/Fast-U2++ dual-mode | Dynamic chunk train stream+offline | WeNet | may help dual deploy | — | dual L | — | **3+** | High complexity | pure causal product | no need dual | **1** |
| 27 | Left-pad patcher | F.pad (P−1,0) | rejected thrice | 0 | 0 | 0 | breaks CTC frames | 1 | **Reject** | everything | n/a | **0** |

### Top-ten notes (short)

**1–4** are nearly free and must precede any retrain. **5–7** are the accuracy/latency product surface. **8** is the only novel structure that simultaneously attacks L_buf and variance. **9–10** are regularization/ensemble efficiency. Everything below 15 is either secondary architecture or gated by S6.

**S1 adversarial verdict:** **Endorse with reduced Δ.** VRAM: 8×45 M fp16 ≈ 0.7 GB weights—fits. Logit average preserves streaming equivalence if each member does and average is frame-synchronous—**yes**. Correlation among causal members **will** shrink gains vs bi-dir competition ensembles—do not promise 5.8%→1.5% style jumps.

**S2 adversarial verdict:** **Mostly reject as primary.** Turn-over of capacity/reg curve is early; heavy time-masking + ensemble beats raw width.

**S3 adversarial verdict:** **Endorse as high-upside experiment after N-curve.** Prefer single-model multi-phase with phase-aug training; separate trains only if needed.

**S4:** Conditional on teacher–student gap existing under *causal* student; Delayed-KD (May 2025) is the right citation. After S6 and after ensemble.

**S5:** **Endorse strongly**—this is how you un-flatten the frontier.

**S6:** **Mandatory gate.** Linderman already saw FastEmit null for WER.

**S7:** **Mandatory.**

---

## 6. Latency & compute budget

### 6.1 Three latencies (do not blend)

| Name | Definition | Current system (causal L=0) | Notes |
|---|---|---|---|
| **Emission latency** | Neural event → first appearance of corresponding text (may still revise) | ≈ **L_buf + L_comp_path ≈ 80 + (0.3–30) ≈ 80–110 ms** (derived) | Dominated by stride-4 buffering |
| **Finalization latency** | Neural event → text for that content ceases to change | Emission + stability wait + optional rescoring; **often EOU today** (seconds) | Clinical UX; must enter the 140 ms budget or be explicitly exempted for “committed” words only |
| **Cold-start latency** | First emission only: patch fill | **280 ms** (P=14×20 ms) measured-config | One-time; exclude from steady-state 140 ms |

Decomposition identity for steady-state emission:  
**L_emit = L_algo + L_buf + L_comp_AM + L_comp_beam + L_comp_resc_amortized**.

Current AM-only: L_algo≈0, L_buf=80, L_comp_AM≈0.34 → **~80.3 ms** before language search.

### 6.2 Analytic stage table

| Stage | L_algo (ms) | L_buf (ms) | L_comp (ms) | Basis |
|---|---:|---:|---:|---|
| Binning 20 ms | 0 | 0–20 (sample phase) | ~0 | config |
| Causal smoother L=0 | 0 | 0 | ≪0.1 | config; peak at t |
| Patch stride s=4 | 0 | **80** | 0 | 4×20 |
| GRU AM forward | 0 | 0 | **0.32–0.34** | **measured** p50/p95 |
| N=8 ensemble | 0 | 0 | **2.6–2.7** | **derived** |
| Phase-stagger K=4 | 0 | **20** | **1.3–1.4** | **derived** |
| CTC beam+3-gram | 0 | 0 | **5–40** | **estimated**; blank-skip lowers |
| LLM rescoring (EOU) | 0 | **utterance** | **50–500+** | **estimated**; batch wall |
| Adaptive LLM (f=0.1) | 0 | 0 | **~5–20 amort.** | **estimated** |

### 6.3 Word-level stability metric (propose; no paper reports this in B2T)

For a stream of partial hypotheses H_0, H_1, … at emission times t_i:

- Align each H_i to final H_∞ with word-level edit alignment.  
- For each reference word w_j with finalization time t_final(j) = min { t_i : the aligned word identity equals H_∞’s word j and remains unchanged for all i'≥i }.  
- **Time-to-final (TTF_j)** = t_final(j) − t_neural_onset(j) [ms]. Report mean/median/p90 TTF.  
- **Revisions per word (RPV)** = number of times the aligned slot’s surface form changes before t_final, averaged over words.  
- **Prefix freeze rate** = fraction of emissions that only extend a frozen prefix (desirable).  
- **Formula sketch:** RPV = (1/|W|) Σ_j |{ i : word_j(H_i) ≠ word_j(H_{i−1}) }|.

If PER frontier is flat but median TTF and RPV improve under adaptive rescoring, **that is the paper**.

### 6.4 Headroom (per 80 ms cadence)

| Platform | AM budget used | Idle ms | Model-forward-equivalents (at 0.34 ms) | Feasible ensemble N (AM only, RTF_AM<0.5) |
|---|---:|---:|---:|---:|
| RTX 3090 (measured class) | 0.34 ms | **~79.7** | **~235** | **N≫50** AM-only; N=8–16 trivial |
| Mid-range laptop GPU (est. 3–5× slower) | ~1–1.7 ms | ~78 | ~45–80 | N=8–16 still fine |
| CPU only (est. 20–50× slower AM) | ~7–17 ms | ~63–73 | ~5–11 | N=2–4 AM; beam may dominate |

**Gating statement:** Every ensembling recommendation in this document is feasible on 3090 and laptop GPU for N≤16 AM-side; CPU-only may be beam-bound before AM-bound. RTF must include LM: **estimate** beam carefully; blank-skip is mandatory on CPU.

---

## 7. Literature

### 7.1 Competitions

**Brain-to-Text Benchmark '24 — Willett et al., arXiv:2412.17227, Dec 23, 2024.**  
https://arxiv.org/abs/2412.17227 · https://arxiv.org/html/2412.17227v1  
Baseline 9.7% WER → top 5.8%. Top-3 all used multi-decoder ensemble + fine-tuned LLM merge. Diphone (DCoND) + training recipe helped. Mamba matched PER but worse WER; Transformers no clear win. FastEmit did not improve WER (Linderman ablation).  
**Transfer:** Ensemble + LLM merge is the proven gain; architecture swap is not. **Caveat:** unconstrained, non-causal, offline LLM—**do not chase on those terms**.

**Brain-to-Text '25 — Kaggle, UC Davis Neuroprosthetics Lab.**  
https://www.kaggle.com/competitions/brain-to-text-25  
Closed. Winner ~**1.5%** WER; user-stated 4th ~1.78%; strong single-model ~2.79% public. Detailed public writeups with **latency and memory columns are scarce** (negative result of search: many leaderboard references, few complete system cards). Roth (2026 thesis) notes best entrants ~1.5% and contrasts pure end-to-end ~10–11% WER.  
**Transfer:** treat 1.5% as **unconstrained ceiling**, not real-time target.

### 7.2 2025–2026 primary

**Yoon et al., Neural decoding of speech using deep neural ensembles, bioRxiv, Jun 4, 2026.**  
https://www.biorxiv.org/content/10.64898/2026.06.02.729705v1  
First closed-loop deep ensemble: **WER 33.7% → 26.0%**. Studies scaling with baseline error, data size, ensemble size; introduces **pseudoensembling via TTA** (single base decoder).  
**Transfer: high.** Closest existence proof for S1/S5. Exact N and ms latency per forward: partial extraction from abstract/full-text snippets; full GPU topology numbers **partially UNVERIFIED** beyond “real-time capable” claim—still sufficient to endorse causal ensembles under Fact B.

**Zamora & Gonzalez-Lopez, Phoneme- vs. Character-Level Targets and SSMs, arXiv:2607.26751, Jul 29, 2026.**  
https://arxiv.org/abs/2607.26751  
2×2 on B2T'25: best phonetic GRU **12.62% PER / 21.19% WER**; char GRU **13.39% CER / 26.28% WER**; Mamba hybrid competitive, not better.  
**Why PER ≫ this project’s 10.04%:** different training recipe/length, likely weaker LM stack and less-optimized baseline loop—not evidence their architecture is worse in absolute terms. **Do not cite their absolute PER as comparable SOTA.** Transfer of *relative* finding (GRU ≥ Mamba; phoneme > char for WER): **high**.

**Boccato et al., Cross-subject decoding…, bioRxiv Feb 2026 / J. Neural Eng. 2026.**  
https://www.biorxiv.org/content/10.64898/2026.02.27.708564v1  
Hierarchical GRU + intermediate CTC + feedback; day/dataset affines; joint Willett(T12)+Card(T15). Joint hierarchical reports **Card 9.1% PER / 6.67% WER**; Willett gains PER 19.7→16.1 path.  
**Transfer: serious free-lunch candidate** for pretrain→finetune; keep unidirectional for streaming. Protocol WER not directly comparable to this project’s full stack.

**Feghhi et al., Time-Masked Transformers…, arXiv:2507.02800, Jul 2025.**  
https://arxiv.org/abs/2507.02800 · https://arxiv.org/html/2507.02800v2  
Causal time-masked Transformer vs uni GRU on '24 data: WER **15.25→12.17%** (3-gram), **11.12→8.18%** (5-gram); −83% params; DietCORP **18 ms/trial, 1.3 GiB**. Ablation: **time-masking is the big lever**; time-masked GRU nearly matches Transformer. **3-gram ~60 GB** memory callout—critical for 32 GB deploy.  
**Transfer: high for masking + TTA; architecture secondary.**

**Khanday et al., End-to-End Intracortical Speech Decoding, arXiv:2605.24313, May 23, 2026.**  
https://arxiv.org/abs/2605.24313  
Conformer, char CTC, **no LM: 23.80% CER**. Errors dominated by **word-boundary** confusions. ~47 M params, P=14 s=4 (same patching family).  
**Transfer:** LM is worth many absolute points; dropping LM to chase “end-to-end purity” fails the WER objective. Boundary errors reinforce lexicon/WFST value.

**Wairagkar et al., Instantaneous voice-synthesis neuroprosthesis, Nature 2025 (PMC12369848; bioRxiv 2024.08.14.607690).**  
https://www.nature.com/articles/s41586-025-09127-3 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12369848/  
Same participant T15. Continuous causal synthesis; public materials: **~25 ms** audio delay (also “1/40 s”); input **600 ms** window of 10 ms bins into Transformer with week embeddings.  
**Latency accounting:** closest published low-latency analogue; their product is voice not text, so LM finalization differs, but **causal continuous emission with short hop** is the design rhyme. Use as existence proof that **≪140 ms** neural→output is achievable on this implant when the stack is built for it.

**Card et al. NEJM 2024** (system under study): high-accuracy rapidly calibrating speech neuroprosthesis; reference stack this repo reproduces.

### 7.3 Streaming ASR mechanisms (CTC-phoneme, 20 ms bin, 80 ms cadence)

| Mechanism | One-line | Latency–accuracy | Open impl | Transfer here |
|---|---|---|---|---|
| FastEmit | Regularize to emit labels earlier at sequence level | Lower emission delay; Linderman: **no WER gain** on B2T'24 | limited | **does not transfer** until S6 says delay exists |
| Delay-penalized CTC (k2 FST) | Latency penalty on alignments via differentiable FST | Controls delay without external align | **k2/icefall** | **untested**; gate on S6 |
| Delay-penalized transducer | Same idea for RNN-T | Strong in ASR streaming | icefall | **does not transfer** (system is CTC) |
| Bayes-Risk CTC | Risk weights on delay groups | Controllable early/late | research | **untested**; gate S6 |
| Peak-First CTC | Prefer early peaks | −100–200 ms peak latency ASR | research | **untested**; gate S6 |
| TrimTail | Truncate trailing blank-heavy tails | Latency cut | research | **untested**; gate S6 |
| Align-With-Purpose | Plug-in optimize desired alignment property | Flexible | research | **untested** |
| Delayed-KD | Distill non-stream→stream with temporal shift | Helps low-latency stream students | research (arXiv:2505.22069, May 2025) | **transfers** as S4 formalization |
| U2/U2++/Fast-U2++ | Dual-mode chunk train; Fast-U2++ ~80 ms CER≈offline | Dynamic latency | **WeNet** | **untested**; complexity high for pure causal product |
| CUSIDE | Chunk + simulate future context | Streaming context | research | **untested** |
| Lookahead conv / Emformer / Zipformer / stream Conformer | Limited right context, memory, multi-rate | Classic ASR stream tradeoff | ESPnet/NeMo/WeNet/icefall | **does not transfer as first lever** (Fact A; GRU strong) |

### 7.4 Incremental LM

Public competition commentary and Feghhi: n-gram memory is the elephant (3-gram tens of GB; unpruned 5-gram hundreds). Blank mass ~70% → skip. Incremental KenLM/WFST is standard in production ASR but **under-documented in brain-to-text writeups**—exactly the gap R2 exploits. Shallow fusion with small neural LMs is open and attractive under 32 GB.

### 7.5 Negative search results

- Complete Brain-to-Text '25 1st-place system card with **measured end-to-end latency and RAM**: **not found** in public discussions/writeups searchable as of this review.  
- Prior art on **phase-staggered patch ensembles** for neural speech BCIs: **not found**.  
- Brain-to-text papers reporting **revisions-per-word / time-to-final**: **not found**.

---

## 8. Assumptions & open questions

| Assumption | If true → recommend | If false → recommend |
|---|---|---|
| Day affine + smoother are phase-agnostic; patch embed may not be | Phase-stagger with phase-aug training (R4) | Train K phase-specific models or drop R4 |
| Streaming beam can match batch beam WER within 0.3 pp with blank-skip | R2 primary | Fix decoder parity before any LM paper claims |
| Released 2.66% WER uses non-causal or heavy EOU LLM components | Real-time gap analysis as written | Re-baseline full-stack causal WER immediately |
| Val PER 10.04% causal is within seed noise of 10.21% | Do not claim causal “wins”; use as equal | If n=3 confirms causal better, publish Fact A as finding |
| Pruned 3-gram fits 6–8 GB at acceptable WER | Deploy 3-gram; optional pruned 4-gram | Prioritize neural LM or on-disk LM with mmap (eng risk) |
| Ensemble correlation among causal seeds is moderate (not near 1) | N=8 pays | Stop at N=2–4; use TTA diversity |
| LM stage dominates user-visible latency | Relocate contribution to R2; fewer encoder runs | If beam is already <5 ms, spend on ensemble/AM only |
| Code’s CTC peaks are late (>80 ms) | Open delay-CTC runs | Kill emission regularizers (S6) |
| Cross-subject T12 data accessible under license | Boccato pretrain path | Skip; stay single-subject |
| 140 ms budget includes finalization of each word, not only first paint | Optimize TTF metric | Split SLAs: emit@80, finalize@140 |

---

## 9. Run ledger

Budget ~20 runs ≈ 128 GPU-h. Sort by information per hour. **Cut line after row 14.**

| Order | Hypothesis / action | Isolated variable | Cost (h) | Info gain / h | Notes |
|---|---|---:|---:|---:|---|
| 1 | Beam width, acoustic/LM scale, blank penalty grid on fixed causal logits | decoding only | **0 train** | **max** | Exhaust first |
| 2 | SWA / last-k checkpoint average | weight average | **0** | max | |
| 3 | Measure CTC peak delay distribution | measurement | **0** | max | Gates 15–18 |
| 4 | Blank-skip + incremental beam parity vs batch | decoder eng | **0** | high | |
| 5 | Entropy-θ adaptive rescoring curve (WER, TTF, RPV vs latency) | θ, B | **0** | high | Defines paper metric |
| 6 | Causal baseline n=3 seeds | seed variance | **12.8** | high | S7 |
| 7 | Ensemble N∈{1,2,4,8} from seeds | N | **0 extra** if seeds from 6–8 | high | Logit avg |
| 8 | Train remaining seeds to N=8 | multi-seed | **~32–45** | high | R1 |
| 9 | Phase-stagger eval-only on one seed | phase offsets | **0** | high | Kill/go for R4 |
| 10 | Time-masking recipe on causal GRU | mask hyperparams | **6.4–12.8** | med-high | Feghhi transfer |
| 11 | Pseudoensemble TTA Z-grid | Z masks | **0** | med | Yoon fallback |
| 12 | LayerNorm+LR schedule GRU | recipe | **6.4** | med | Linderman |
| 13 | Phase-aug retrain if 9 failed | phase training | **6.4** | med | |
| 14 | Hierarchical intermediate CTC causal | architecture | **6.4–12.8** | med | Boccato |
| ——— | **CUT LINE (~128 h cumulative if all above)** | | | | |
| 15 | Delay-penalized CTC | only if S6 opens | 6.4 | low if closed | |
| 16 | FastEmit | only if S6 | 6.4 | low | |
| 17 | Peak-First | only if S6 | 6.4 | low | |
| 18 | Predictive distillation bi→causal | teacher+student | 12.8–19 | med | After ensemble |
| 19 | Cross-subject pretrain+finetune | data regime | 12.8–25 | med | License-dependent |
| 20 | Width×4 GRU + heavy mask | capacity | 6.4–12.8 | low | S2 last |
| 21 | Mamba hybrid | architecture | 12.8 | low | Literature nulls |
| 22 | Streaming Conformer | architecture | 12.8 | low | |

**Ledger opens with zero-retrain work by design.** First retrain is seed variance (scientifically mandatory), not a speculative architecture.

---

## 10. What I am not recommending

1. **Replacing the unidirectional GRU with a streaming Conformer/Mamba as the flagship retrain.** Multiple independent 2024–2026 studies on this exact problem family failed to beat a well-trained GRU on architecture alone; Feghhi’s win is mostly masking. Spending 2–4 runs here before ensemble N-curves is how projects burn 128 GPU-h for a null.

2. **Chasing unconstrained ~1.5% WER with bidirectional ensembles + always-on large LLM rescoring.** Scores zero under the stated constraint. Report the **gap** honestly instead.

3. **Emission-delay regularizer suite (FastEmit, Peak-First, TrimTail, delay-CTC) as a package before S6.** Linderman already saw FastEmit null; Fact A suggests peaks may not be late. Running all four is correlational noise.

4. **Left-padding the patcher.** Correctly rejected; still zero causality value.

5. **Character-level end-to-end without LM as a WER strategy.** Khanday’s 23.8% CER with boundary-dominated errors shows what you get when you remove the linguistic stage—the stage that currently holds most of the WER.

6. **5–20× single-model capacity without first proving underfit under heavy time-masking and ensembles.** Data volume (45 sessions, one participant) predicts overfit.

---

*End of review. Single deliverable: this file.*
