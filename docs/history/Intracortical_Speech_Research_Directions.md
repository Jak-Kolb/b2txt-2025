# Research directions for an efficient streaming intracortical speech decoder

Research checked 11 October 2026. The measurements in the prompt are treated as supplied experimental observations; I did not inspect the harness, rerun the models, or independently validate those measurements.

**Recommendation:** retain the causal GRU as the experimental backbone. First measure a bounded-future accuracy curve, improve selection from its existing hypothesis set, and learn a commitment policy that distinguishes neural evidence from language-model preference. These offer a stronger, cheaper research program than another unrestricted architecture search.

The scientific question is: **how much additional neural evidence must arrive before a correct word can be selected and safely committed?** Separate the ability to generate a good candidate, select it, and know when to stop revising it.

Your fixed n-best pool has 5.61 WER points between the live result and its 2.90% oracle. GPT-2 recovers about 29% of that gap: (8.51−6.87)/(8.51−2.90). That does not make 2.90% attainable without an oracle, but it identifies an unusually inexpensive experimental target. These are calculations from your supplied rounded numbers, not published effect sizes.

## Evidence and scope

“Established” below means demonstrated in the cited setting, not guaranteed for this pipeline. “Extrapolation” means an existing method transferred to a different setting. “Hypothesis” marks a proposed mechanism or experiment. Novelty statements are bounded by the literature reviewed; none is a claim of having proved a first-ever result.

Several relevant BCI sources—LightBeam, BrainWhisperer, ALIGN, and the time-masked Transformer paper—have versions older than six months as of this report. Their findings remain useful, but should not be described as the latest frontier. I checked BIT v5 (13 May 2026) and the newer Fogg et al. multi-user preprint (27 July 2026). Preprints remain preliminary.

Four literature updates change the framing:

- **Cross-participant transfer is now a direct speech-BCI precedent.** Fogg et al. report joint training across six participants and adaptation to held-out users. This is promising evidence for T12 transfer, not a predicted effect size for your two-person GRU experiment. Conversely, BIT v5 Appendix M reports that its supervised cross-subject training hurt performance. The disagreement makes architecture, participant projections, sampling, and adaptation important experimental variables. [Fogg et al.](https://www.biorxiv.org/content/10.64898/2026.07.23.739430v1.full), [BIT, Appendix M](https://arxiv.org/html/2511.21740v5)
- **LightBeam includes a causal encoder and incremental LLM fusion.** It is not exclusively a whole-sentence technique. However, final sentence WER and real-time factor do not establish stable-word latency. Its published decoding also has a final sentence-ending score pass. [LightBeam](https://arxiv.org/html/2603.14002v1)
- **Pretraining provenance matters.** BIT explicitly includes unlabeled evaluation/holdout recordings in self-supervised learning. That is transductive evaluation, distinct from a model applied to recordings it has never encountered. [BIT, Appendix A.4](https://arxiv.org/html/2511.21740v5)
- **Do not blindly inherit language-model checkpoints.** LightBeam’s main text says LM adaptation uses training and validation sentences; Appendix F says training sentences. Resolve that discrepancy in the implementation/checkpoint provenance before using those weights to evaluate val-dev. This is a reporting ambiguity, not evidence of concealed misuse. [LightBeam, §3.4.2 and Appendix F](https://arxiv.org/html/2603.14002v1)

## Ranked experiments

Ranks reflect expected information gained and practical value relative to cost. GPU-hour ranges are engineering estimates anchored to your 1–2 hour, 20k-batch run; they are not measured RTX 5070 Ti benchmarks. Code effort assumes familiarity with the existing harness. CPU decoding and out-of-fold data generation can dominate elapsed time. Use one seed for screening, then repeat the shortlisted comparison; a narrow confidence interval across sessions does not replace seed replication.

| Rank | Direction | First GPU budget | Main result to measure |
|---|---|---:|---|
| 1 | Bounded-future correction above the existing GRU | 2–4 h | WER versus audited right context |
| 2 | Calibrated candidate scoring and minimum-risk selection | 0–1 h | Recoverable n-best errors actually recovered |
| 3 | Commitment based on evidence and predicted risk | 0–1 h, plus optional out-of-fold models | Wrong locks versus coverage and waiting |
| 4 | Equal-budget schedules, calibration, and checkpoint selection | 4–8 h | Whether the comparison survives fair optimization |
| 5 | Shared, low-rank session adapters | 3–6 h | Accuracy and adaptation with fewer session parameters |
| 6 | Small diverse ensembles with sequence-level fusion | 0–1 h cached; 2–4 h for another model | Complementary errors per added compute |
| 7 | Alignment-aware distillation from a stronger teacher | 3–6 h | Better causal WER without delayed emissions |
| 8 | T12 transfer through participant-specific front ends | 4–8 h | T15 gains beyond transcript-only exposure |
| 9 | Small LLM rescoring of live prefixes | 1–3 h | Better live ranking without missed deadlines |
| 10 | Diphone or articulatory auxiliary targets | 2–4 h | WER gains at fixed right context |
| 11 | CTC emission-delay regularization | 2–4 h | Accuracy–emission–commitment frontier |
| 12 | Word-risk fine-tuning with the deployed decoder | 2–5 h | Improvements aligned with final WER |
| 13 | Conservative chronological input adaptation | 1–3 h replay | New-session drift and adaptation failures |
| 14 | Explicit rate and fine/coarse temporal paths | 3–6 h | Robustness to speaking-rate variation |
| 15 | Small, leakage-resistant self-supervised pretraining | 4–10 h | Downstream gains, not reconstruction alone |
| 16 | Session invariance conditional on linguistic content | 3–6 h | Drift resistance without deleting phonetic information |
| 17 | Cross-fitted correction and controlled pseudo-labeling | 8–16 h | Generalization beyond memorized training errors |
| 18 | Continuous-stream and endpointing evaluation | 0–2 h initially | False activations, truncations, and deadline failures |

### 1. Add a bounded-future correction head to the frozen causal GRU

**Idea.** Cache causal hidden states and train a small temporal head that refines the logits for frame t using only h[t:t+H]. Keep the ordinary causal output available immediately; allow revisions only within an explicitly bounded suffix.

**Mechanism.** This asks whether a little future evidence resolves ambiguities already represented by the GRU. It isolates temporal context much more cheaply than retraining a 79M-parameter bidirectional network. Compare H=0, 2, 6, and 18 encoder frames: 0, 160, 480, and 1,440 ms on the current 80 ms grid. The zero-context head is the essential capacity control.

**Evidence.** Latency-controlled bidirectional recurrent models and dynamic-chunk two-pass ASR show that bounded right context is a workable design. They do not establish the best horizon for neural speech. [Zhang et al., latency-controlled BLSTM](https://arxiv.org/abs/1510.08983), [U2](https://arxiv.org/abs/2012.05481)

**Novelty.** The general approach is established in ASR. The hypothesis here is that a small refinement module recovers a useful fraction of this participant’s bidirectional gain while preserving an audited streaming path.

**Cost.** Roughly 1–2 days of code and 2–4 GPU h for short head-training runs; existing training recordings and cached hidden states suffice. Cache per-session shards on disk rather than all activations in RAM. Joint fine-tuning is a later 2–4 h experiment if the frozen-head test is promising.

**Test.** Fit equal-capacity heads with identical data and optimization. Compare both fixed decoder settings and a small, equally sized tuning grid. A repeatable gain at 160–480 ms supports the route. If improvement appears only near a full word or sentence, deprioritize it for the tightest operating point. A negative frozen-head result does not rule out future information in the original features.

**Latency class.** Live with H×80 ms explicit right context, plus measured compute/queue delay. A prefix second pass remains live when its inputs stop at the current wall clock. Its cost must still be charged to the revised words.

**Risks/confounds.** Never compute h[t] with a bidirectional backbone. Multi-layer local attention can silently expand receptive fields. Right padding and end-of-trial handling must be identical across conditions. CTC can shift emissions independently of architectural look-ahead.

### 2. Recover more of the n-best oracle with calibrated scoring and minimum-risk decoding

**Idea.** Keep the acoustic model fixed. Decompose candidate scores, fit a small discriminative ranker, and compare maximum-score selection with minimum expected word-edit loss.

**Mechanism.** Use acoustic score, n-gram score, GPT-2 score, length, blank/nonblank statistics, score margins, and phonetic support as separate features. Calibrate on held-out training predictions. For candidates h_i, approximate minimum Bayes risk as argmin_h sum_i q_i EditDistance(h,h_i), where q_i are normalized, calibrated candidate weights. This needs no reference transcript at inference. A word confusion network can combine supported words from different candidates, although then the fixed-list 2.90% oracle is no longer the relevant bound.

**Evidence.** Confusion-network ASR demonstrated that the most probable whole sentence need not minimize word errors. [Mangu, Brill and Stolcke](https://arxiv.org/abs/cs/0010012)

**Novelty.** Standard ASR machinery; a useful speech-BCI contribution would explain how calibration, blank handling, and LM overrides produce the PER/WER disconnect.

**Cost.** About 1–2 days; mostly CPU, 0–1 GPU h with cached candidates and scores. Realistic supervised ranker training may require extra out-of-fold acoustic predictions; count those separately.

**Test.** Compare current scores, calibrated linear ranking, candidate-list minimum risk, and a small confusion-network implementation on exactly the same pool. First ask whether improvements transfer across held-out sessions and sources. Stop if the ranker only improves its tuning folds or rescues common sentences while damaging unusual ones.

Before fitting anything, audit transcript normalization, out-of-vocabulary words, pronunciation coverage, homophones, and word-boundary errors. A scoring model cannot select a spelling the lexicon never permits. Keep fixes to evaluation-text normalization symmetric between predictions and references and record them separately from decoder improvements.

**Latency class.** Finalization for full-sentence candidates. Live only when applied to candidates generated from the available prefix; regenerate prefix candidate sets rather than cropping final n-best lists.

**Risks/confounds.** Search scores are not automatically calibrated probabilities. Candidate duplicates distort posterior mass. Beam pruning creates missing support. Correct phonemes do not determine spelling uniquely, especially for homophones. The oracle is a reference-dependent diagnostic, not a deployable selector.

### 3. Commit using predicted error risk and sensitivity to LM pressure

**Idea.** Estimate whether a word is safe to lock using current beam evidence, not just unchanged duration. Include whether it survives modest changes in LM weight.

**Mechanism.** A word stable across high-posterior beams, with strong neural support and agreement under several acoustic/LM balances, is different from one stable only because the LM dominates. Candidate features include posterior mass sharing the prefix, top-two margin, recent neural-score increments, acoustic-only disagreement, age, and diversity of competing continuations. Fit separate targets for “will remain unchanged” and “is correct”; a permanently wrong word satisfies the former.

**Evidence.** ASR already estimates per-word stability from features and studies the lag/stability trade-off. [McGraw and Gruenstein](https://www.isca-archive.org/interspeech_2012/mcgraw12_interspeech.html)

**Novelty.** The specific hypothesis is that counterfactual LM-weight agreement plus neural evidence predicts irreversible-error risk in this strongly LM-dependent speech BCI.

**Cost.** About 2–3 days; 0–1 GPU h if logs contain beam scores, with CPU feature extraction. Add 4–8 h if training-only out-of-fold acoustic logs must be generated.

**Test.** First cross-fit a simple logistic model on development sessions as an explicitly exploratory feasibility test. Then fit the selected rule on out-of-fold training predictions. Compare the entire wrong-lock-rate versus coverage/delay curve with the three-newer-word baseline. A policy that attains low risk by locking almost nothing fails.

**Latency class.** Live, with adaptive commitment waiting and a specified maximum deadline or abstention rule.

**Risks/confounds.** Frames and consecutive partials are highly correlated. Confidence must transfer across sessions and sentence sources. For online features, exclude the eventual final hypothesis and all future logits. Post-hoc scoring may use them only as labels. Replaying locks on unconstrained partials measures a display policy; feeding locks back into beam search requires a new replay because the hypotheses change.

### 4. Run a fair short-budget schedule and calibration experiment before declaring the model exhausted

**Idea.** Compare complete 20k schedules with 20k prefixes of the existing 120k schedule, using WER and calibration rather than PER alone for limited checkpoint selection.

**Mechanism.** With the stated cosine schedule, the learning rate at 20k is still approximately 0.00467, close to its 0.005 start. This is an early high-rate checkpoint, not a converged low-budget model. At 20k, 64-trial batches represent about 159 nominal passes through 8,072 trials; at 120k, about 951. Actual exposure differs under session sampling. Rising validation CTC loss while PER improves can also reflect confidence changes, so test temperature and blank calibration on old checkpoints before assuming all deterioration is representation overfitting.

**Evidence.** The B2T ’24 lessons paper identifies training schedules as a useful lever and documents difficulty beating the RNN through architecture changes alone. [Willett et al.](https://arxiv.org/abs/2412.17227)

**Novelty.** Primarily an essential control, not a new algorithm. The research value is explaining why cheap exploration does or does not predict full-run rankings.

**Cost.** Less than a day of code; 4–8 GPU h for matched short schedules and seeds, plus CPU decoding. Re-scoring existing checkpoints is cheaper.

**Test.** For the causal model and one challenger, compare the original schedule prefix with a 20k cosine schedule using the same initial rate and warmup convention. Preselect a few checkpoints. If the architecture ranking reverses, update the screening protocol. Also compare likelihood reductions on identical unaugmented examples and identical length normalization.

**Latency class.** Offline training; no added inference delay.

**Risks/confounds.** A faster schedule may hurt eventual full-run performance. Equal steps do not mean equal GPU time or parameter updates. Do not expand into an unrestricted schedule search on exposed val-dev.

### 5. Replace independent day matrices with a shared projection and small session residuals

**Idea.** Use a common front end plus low-rank session corrections, for example W_d = W_shared + U_d V_d^T, with a bias/gain term and shrinkage toward the shared mapping.

**Mechanism.** A full 512×512 matrix per day gives sparse sessions considerable freedom to memorize. At rank 8, the factor matrices contain 2×512×8 = 8,192 parameters per day versus 262,144 for a dense matrix, excluding biases. Shared training can learn reusable channel interactions while session residuals correct smaller shifts. Do not assume drift is smooth across every date; allow a change-point reset.

**Evidence.** BrainWhisperer already uses hierarchical month/day low-rank projections for neural speech. [BrainWhisperer](https://arxiv.org/html/2603.13321v1)

**Novelty.** Low-rank day adaptation is not new for speech BCIs. Your contribution would be its causal-GRU sample-efficiency, calibration, and new-session behavior at matched compute.

**Cost.** Roughly 1–2 days and 3–6 GPU h. Existing data suffice; a chronological session split is necessary for a calibration claim.

**Test.** Compare shared-only, shared plus rank-8 residual, and dense day layers. Include a low-data curve per day. Initialize a completely held-out day with the shared map, then evaluate zero, small supervised, and past-only unlabeled adaptation. Improvements only on known-day validation do not demonstrate calibration-free use.

**Latency class.** Live with negligible additional buffering; actual compute must be profiled.

**Risks/confounds.** Low-rank changes may miss arbitrary electrode failures. Calendar hierarchy can overfit sentence mix. Dense and factored models need compatible initialization and regularization. Keep hardware channel ordering explicit.

### 6. Try two complementary models before considering large ensembles

**Idea.** Start with existing independently trained checkpoints. Fuse at the candidate or sequence-score level and compare with an ensemble chosen for complementary errors rather than best individual PER.

**Mechanism.** Your roughly 1 ms acoustic-model update suggests room to investigate a second small model, subject to actual simultaneous inference profiling. Distinct seeds, feature views, or day-adapter designs may disagree usefully. Existing Transformer outputs can be informative even when their standalone WER is worse.

**Evidence.** B2T ’24 ensemble/LLM-merging results motivate diversity. Guided-CTC work shows that independent models can have incompatible posterior spike timing, making naive framewise fusion unreliable. [B2T lessons](https://arxiv.org/abs/2412.17227), [Kurata and Audhkhasi](https://arxiv.org/abs/1904.08311)

**Novelty.** Ensembling is established in speech BCIs. A small, latency-accounted ensemble with an accuracy-per-compute justification is the useful distinction.

**Cost.** Roughly a day; 0–1 GPU h using cached logits or 2–4 h for another short model. Independent streaming decoders may multiply CPU costs even when GPU costs are small.

**Test.** Form the union of two n-best lists and recompute sequence scores under both models. Compare error overlap, oracle coverage, WER, and p99 update time. If a second model adds negligible complementary coverage or worsens tail latency, stop.

**Latency class.** Live if both component models are causal and prefix scoring is used; finalization if either uses a whole sentence.

**Risks/confounds.** Never average unaligned CTC spikes as the only ensemble test. Full-sequence CTC scoring of candidate transcripts is safe offline; online prefix scoring must account for unfinished words and alignments. Select model pairs on training-derived development predictions, not by exhaustively optimizing validation pairs.

### 7. Distill the bidirectional teacher without copying impossible emission times

**Idea.** Train a causal or short-look-ahead student from the stronger teacher using sequence-level candidate distributions or explicitly aligned soft targets, alongside ordinary supervised CTC.

**Mechanism.** The teacher may provide better uncertainty over phonemes and alternatives than a single hard transcript. But its framewise predictions can depend on information the student has not seen. Start by computing teacher scores for reference and competing phoneme sequences and teaching their relative ranking. If using framewise distillation, align spike timing or shift the target by the allowed student delay.

**Evidence.** ASR studies show that CTC posterior timing is arbitrary across models and that guiding alignment improves fusion and distillation. [Kurata and Audhkhasi](https://arxiv.org/abs/1904.08311), [Kim et al.](https://arxiv.org/abs/2406.07909)

**Novelty.** Distillation is established in ASR. Capturing this particular whole-sentence neural teacher’s gain in a strictly measured streaming student is an extrapolation, not an established BCI effect.

**Cost.** About 2–3 days and 3–6 GPU h including teacher inference and student training. Cache teacher outputs only on permitted training data; the teacher is not needed at inference.

**Test.** Compare ordinary CTC, naive framewise distillation, and aligned or sequence-level distillation at the same compute budget. Require improved WER without shifting words later. If improvement comes only from delayed emission, report it as an accuracy–delay exchange rather than a free causal improvement.

**Latency class.** Live with the student’s own context limit; no runtime teacher cost.

**Risks/confounds.** Training on future context is permitted for a teacher, but student gains cannot create information absent from a prefix. The teacher can transfer sentence priors and mistakes. Its checkpoint selection history also belongs in the exposure ledger.

### 8. Transfer T12 phonetic structure through a separate participant front end

**Idea.** Compare T15-only training, mixed T12/T15 training with separate input projections, and T12 pretraining followed by T15 fine-tuning. Share phoneme representations, not electrode identities.

**Mechanism.** Additional attempted-speech trials can constrain the recurrent decoder’s phonetic dynamics. Separate front ends absorb electrode placement and feature-scale differences. Keep the phoneme inventory consistent and make participant/task identity explicit where needed. Start with T12 attempted speech; add imagined speech as a separately tagged task only after that works.

**Evidence.** Fogg et al.’s July 2026 preprint reports successful multi-user speech models with participant projections, whereas BIT Appendix M reports harmful supervised cross-subject training in its setup. These are direct but conflicting speech-BCI precedents; transfer is conditional on design. [Fogg et al.](https://www.biorxiv.org/content/10.64898/2026.07.23.739430v1.full), [BIT](https://arxiv.org/html/2511.21740v5)

**Novelty.** Cross-user speech decoding itself is not new. The question is whether two-person transfer benefits a small causal GRU under fixed training and latency budgets.

**Cost.** Roughly 2–4 days for schema, normalization, and split checks; 4–8 GPU h for the first three-arm screen. The downloaded T12 data suffice.

**Test.** Preserve T15 trial exposure, and separately report total GPU cost. Include a transcript-only control that adapts the LM with T12 text while leaving the neural model unchanged. Require benefit on T15 sentences absent from the auxiliary transcripts. If transcript-only exposure explains the gain, the neural-transfer claim fails.

**Latency class.** Live; the T12 front end is not needed when decoding T15.

**Risks/confounds.** Audit the actual T12 release and loaded feature width, not just implanted electrode count. Record feature types, thresholds, channel ordering, bin sizes, normalizers, clipping, tasks, and session overlaps. For 10→20 ms conversion, aggregate according to the feature’s definition; sums for counts and appropriate averaging for power, not one rule for every channel. No per-trial future-derived normalization. Duplicate transcripts across participants can masquerade as transfer.

For the later ECoG track, initially share phonetic targets, decoding/search code, and evaluation definitions. Keep modality-specific feature extraction and front ends. Speech perception, overt speech, attempted speech, and imagined speech labels should remain distinct unless a tested transfer objective justifies combining them.

### 9. Rescore only the live ambiguous suffix with a small LLM

**Idea.** Add prefix-only LLM scoring on word-boundary or uncertainty events, initially with a roughly 1B-class model. Reuse the committed prefix and deduplicate candidate text; reserve the 8B model for a later measured comparison.

**Mechanism.** At this participant’s speech rate, the language model need not run every 80 ms. Candidate scoring can improve homophone/word selection before the sentence ends. Score alternatives under one explicit formula, such as acoustic + weighted n-gram + weighted LLM + length terms. When replacing an n-gram score with an LLM score, subtract the old contribution rather than accidentally counting both at full weight.

**Evidence.** LightBeam directly demonstrates delayed LLM fusion in a speech-BCI beam and uses Llama 3.2 1B; its resource measurements are on different hardware. [LightBeam](https://arxiv.org/html/2603.14002v1)

**Novelty.** In-beam LLM fusion is already done in speech BCIs. Event-triggered scoring, controlled LM pressure, and its contribution to irreversible-word delay are the proposed extensions.

**Cost.** Roughly 2–4 days and 1–3 GPU h to replay and profile; training is optional. Start with an unadapted checkpoint or training-text-only adaptation of verified provenance.

**Test.** Replay the same prefix hypotheses with fixed-interval versus event-triggered rescoring. Measure live WER, correct-candidate survival, LM override errors, number of LLM calls, and deadline misses with both models resident. Compare against cheap calibrated ranking from idea 2.

**Latency class.** Live prefix rescoring, with measured queue and completion time. Sentence-end punctuation or whole-sentence generation remains finalization.

**Risks/confounds.** Mean real-time factor hides GPU contention and p99 stalls. KV-cache memory grows with beam count and sentence length. A scheduler must prevent an LLM call from blocking the 80 ms decoder. Do not use known source labels unless deployment really provides them. Stronger LM scores can worsen random strings and rare names.

### 10. Add context-sensitive phonetic supervision without replacing the 41-class interface

**Idea.** Add a diphone or articulatory-feature auxiliary head while retaining the existing monophone CTC output and decoder. A diphone represents a neighboring phoneme pair.

**Mechanism.** An auxiliary task can encourage the hidden state to preserve transitions that a single phoneme target ignores. A factorized pair representation avoids treating every possible pair as a well-supported independent class. Articulatory supervision can share statistical strength across confusable sounds, but target categories need a checked mapping rather than ad hoc definitions.

**Evidence.** DConD-LIFT uses context-aware diphone representations while preserving tractable phoneme decoding and reports a gain on the B2T ’24 benchmark. [Li et al.](https://pubmed.ncbi.nlm.nih.gov/40795874/)

**Novelty.** Diphone supervision is already a speech-BCI method. The new comparison would test whether it substitutes for right context or merely causes later emission in this causal T15 model.

**Cost.** About 1–2 days and 2–4 GPU h; existing transcripts provide pair targets. External articulatory annotations are optional.

**Test.** Start with one auxiliary loss weight and one short run; repeat only if WER or the accuracy–delay curve improves. Stratify transition errors and monitor rare pairs. A PER improvement without WER/latency benefit is not a success for the stated goal.

**Latency class.** Live if the decoder uses only causal hidden states. The auxiliary head can be dropped at inference.

**Risks/confounds.** Predicting the next phoneme can induce waiting or exploit sentence predictability. Pair counts are sparse and boundary conventions matter. Do not infer physiological coarticulation merely because diphone supervision helps.

### 11. Penalize delayed CTC emissions explicitly

**Idea.** Add a mild penalty to CTC alignment paths that emit nonblank labels late. Sweep only a few penalty strengths together with the bounded-context condition.

**Mechanism.** Causality constrains which inputs the model sees; it does not require the model to emit a phoneme promptly. The current blank penalty changes inference scores, while an alignment-time penalty changes what the model learns to wait for. Their effects should be separated.

**Evidence.** Delay-penalized CTC implements this trade-off directly in a differentiable finite-state formulation and demonstrates an accuracy/delay trade-off in ASR. [Yao et al.](https://arxiv.org/abs/2305.11539)

**Novelty.** Established ASR approach; applying it to slow attempted speech and judging correct, irreversible words is an extrapolation. FastEmit is a transducer method, so it should not be treated as a drop-in CTC loss.

**Cost.** Roughly 2–4 days if loss integration is needed, 2–4 GPU h for a small fine-tuning screen; an existing differentiable-CTC/FST implementation can reduce code effort, after version verification.

**Test.** Compare no penalty with two mild penalties. Report the change in relative emission times under a fixed offline aligner and the lock-risk curve. Reject settings that merely produce more early guesses, deletions, or revisions.

**Latency class.** Live; no added future context. Any apparent timing improvement remains model-relative unless independently aligned.

**Risks/confounds.** Without true word times, low alignment cost is not proof of low physiological latency. A language-heavy model may guess early. Aggressive penalties can damage accuracy, especially for low-probability utterances.

### 12. Fine-tune toward the deployed word-error objective

**Idea.** Use a small fixed or periodically refreshed candidate set to optimize expected word-edit loss, while retaining CTC as an anchor.

**Mechanism.** PER, likelihood, and WER are different objectives. For each training sentence, score candidate word strings using their pronunciation paths and the deployed LM, normalize within the candidate pool, and minimize expected word errors. This targets errors that the actual search makes instead of assuming every phoneme error has the same downstream cost.

**Evidence.** Minimum-WER training is established in ASR; incorporating LM fusion during this training has been investigated explicitly. Those demonstrations are not evidence of an equivalent gain for phoneme-CTC BCI data. [Prabhavalkar et al.](https://arxiv.org/abs/1712.01818), [Meng et al.](https://arxiv.org/abs/2106.02302)

**Novelty.** The proposed extension is decoder-aware word-risk optimization for this small-data pipeline, particularly whether it explains the masking model’s better PER but worse WER.

**Cost.** Roughly 3–5 days; 2–5 GPU h for short fine-tuning, plus candidate-generation CPU time. Keep lists small initially and avoid rebuilding a huge LM.

**Test.** First fine-tune only the output head on hard examples; then the upper GRU layers if helpful. Compare with equal-step ordinary CTC fine-tuning. If gains disappear with frozen decoding weights or hurt unusual-source sentences, the model may only be adapting to the LM’s preferences.

**Latency class.** Live if the base model is causal; training changes no architectural look-ahead.

**Risks/confounds.** Near-zero training loss leaves few realistic in-sample errors: use noisy/early-checkpoint candidates or preferably out-of-fold candidates. Training references are allowed in the loss, but validation references are not. Candidate-set mismatch, rare pronunciations, and spelling ambiguity need explicit treatment.

### 13. Adapt inputs conservatively in chronological order

**Idea.** Evaluate input-only adaptation on a chronological trial stream, using high-confidence pseudo-labels, a frozen reference model, and a small replay/regularization anchor. Make adaptation optional when evidence is weak.

**Mechanism.** A few input parameters may track channel gain, offset, and recording drift without rewriting the phonetic model. The model first emits and is scored on trial n; learning from that trial affects n+1 onward. A slowly updated normalizer and a trainable adapter are separate ablations, not a single bundle.

**Evidence.** DietCORP adapts the input embedding from augmented pseudo-labeled speech trials; CORP’s earlier long-term demonstration was handwriting, not speech. [Feghhi et al.](https://arxiv.org/html/2507.02800v2), [CORP](https://arxiv.org/abs/2311.03611)

**Novelty.** Speech-BCI test-time adaptation is established. Combining small session residuals, conservative neural-support gates, and a fully chronological failure analysis is the proposed contribution.

**Cost.** Roughly 2–4 days; 1–3 GPU h for an initial replay, with retraining costs if the current checkpoint has already trained on all prospective test days.

**Test.** Compare frozen, running-statistics-only, adapter-only, and confidence-gated adapter conditions after one predefined session cutoff. Track the first trials of each new day and cumulative WER, not only the final adapted score. A method that sometimes collapses on difficult sessions should not win on mean WER alone.

**Latency class.** Live background adaptation for subsequent predictions; include GPU contention. It is label-free adaptation, not absence of calibration or adaptation.

**Risks/confounds.** LM mistakes can reinforce themselves. Blank-entropy minimization can prefer all-blank output. A rollback trigger cannot use unavailable reference WER. Whole-day normalization, pseudo-labels from future trials, and an already-trained day layer invalidate a new-day causal claim. Causal re-normalization cannot undo upstream future-dependent normalization or clipping in released features.

### 14. Separate speaking rate from fine temporal evidence

**Idea.** First try modest temporal stretch augmentation. Separately test a channel bottleneck before patching or fine/coarse causal paths, so a slow utterance does not force the model to discard short neural events.

**Mechanism.** “1.6 seconds per word” is not evidence that discriminative neural changes are slow. A small 20 ms path can retain transitions while the existing 80 ms path carries longer history. A projection from 512 features to a smaller latent dimension before the 14-bin patch reduces the first GRU layer’s very large input matrix without increasing look-ahead. These are separate experiments.

**Evidence.** ALIGN investigates temporal-stretch augmentation for neural speech. Time-masked Transformer work motivates reducing redundant overlapping inputs. Neither establishes the proposed two-rate GRU front end. [ALIGN](https://arxiv.org/html/2603.18299v1), [Feghhi et al.](https://arxiv.org/html/2507.02800v2)

**Novelty.** Rate augmentation has direct BCI precedent. The factorized/two-rate front end is a hypothesis aimed at this unusually wide patch input.

**Cost.** Roughly 1–3 days; 3–6 GPU h for unbundled tests. No new data needed.

**Test.** Test a small predeclared speed range, then one bottleneck width. Compare WER by trial duration per reference phoneme and by sentence source, with duration statistics used only for offline analysis. Prefer a simpler model if accuracy is maintained and memory/latency improve.

**Latency class.** Live and past-only; finer update rates increase compute rather than mandatory future context.

**Risks/confounds.** Time warping changes the physical interpretation of counts and power; check feature semantics. Do not normalize or resample each inference trial using its eventual duration. A channel bottleneck can remove weak but valuable units. Source and rate are correlated.

### 15. Pretrain a small model with masking that cannot be solved by overlapping patches

**Idea.** Start with T15/T12 training recordings and a modest denoising or masked-prediction task. Scale to monkey/handwriting data only if the small experiment improves downstream decoding.

**Mechanism.** Your adjacent patches overlap by 10/14, approximately 71%. A masked patch can be reconstructed from neighboring unmasked patches containing the same raw bins. Mask before patching and cover every overlapping copy, including smoothing support; use temporal guard bands. Predict held-out channels or genuinely hidden spans rather than rewarding copy shortcuts.

**Evidence.** BIT demonstrates neural self-supervised pretraining across tasks/species, with a transductive exposure protocol. The proposed overlap audit is derived from your front end, not a measured BIT defect. [BIT](https://arxiv.org/html/2511.21740v5)

**Novelty.** Cross-task neural pretraining is established. Testing leakage-resistant masking and causal downstream benefit on a one-GPU budget is the proposed contribution.

**Cost.** About 2–4 days and 4–10 GPU h for short pretraining plus fine-tuning. Begin with downloaded data. Treat a 367-hour pretraining run as outside the cheap exploratory budget.

**Test.** Compare random initialization, naive patch masking, and raw-bin masking with guard bands under equal compute. Judge WER and new-session robustness. Good reconstruction with no downstream gain refutes the practical benefit, even if the pretraining loss looks excellent.

**Latency class.** Offline training. A bidirectional pretraining objective is allowable, but the deployed model must still meet its causal/context specification.

**Risks/confounds.** Use regression likelihoods suited to normalized continuous features; do not apply a Poisson count likelihood to clipped z-scores. Task identity, normalization, bad channels, and motor-task dynamics can dominate the representation. Keep evaluation recordings out for the inductive arm and disclose them for any transductive arm.

### 16. Remove session nuisance only after conditioning on linguistic content

**Idea.** If adding domain-adversarial training, suppress session identity conditional on phonetic/task content rather than indiscriminately making all days look alike.

**Mechanism.** Because sentence mix explains much of your session variation, session identity is correlated with useful linguistic information. An unconditional adversary could erase task-relevant structure. Begin with source-balanced batches and session probes within matched phoneme/source strata; only then introduce a conditional session loss or low-weight adversary.

**Evidence.** ALIGN reports cross-session speech-BCI gains from adversarial alignment and uses unlabeled target-session data. The conditional variant proposed here responds to your measured source confound. [ALIGN](https://arxiv.org/html/2603.18299v1)

**Novelty.** Session adversarial adaptation is already a speech-BCI method. Preserving content while removing recording-specific variation in this source-confounded dataset is the hypothesis.

**Cost.** About 2–3 days and 3–6 GPU h. A probe-only diagnostic costs much less.

**Test.** Compare source-balanced sampling, unconditional adversarial training, and a weak conditional variant on the same chronological split. Require lower WER beyond the benefit of balancing alone. If an adversary reduces session predictability but worsens phonetic/word decoding, abandon that objective.

**Latency class.** Live; the adversarial head is dropped at inference.

**Risks/confounds.** Weakly inferred phonetic strata are noisy. Perfect invariance is not necessarily desirable: knowledge of a day’s recording state can help decoding. Access to all future target-day recordings makes an experiment transductive, not a simulation of a first encounter with that day.

### 17. Build correction and pseudo-labeling data from genuinely held-out predictions

**Idea.** Train a constrained error corrector on out-of-fold training predictions before attempting free-form 8B rewriting. Test unlabeled trials in a separate pseudo-labeling/transductive arm.

**Mechanism.** A decoder with near-zero in-sample loss produces unrealistically clean training hypotheses. Cross-fitted predictions expose the corrector to errors closer to deployment. Give the model a short candidate list and phonetic evidence; first predict candidate rank or supported edits. Mix clean examples and unusual word strings so it learns to preserve accurate, unlikely text. For unlabeled trials, accept pseudo-labels only when independent evidence checks agree and retain a labeled-data anchor.

**Evidence.** DConD-LIFT and B2T competition work establish LLM-assisted correction; CORP/DietCORP establish pseudo-label adaptation. The leakage-control protocol and evidence-constrained correction proposed here are methodological extensions. [Li et al.](https://pubmed.ncbi.nlm.nih.gov/40795874/), [B2T lessons](https://arxiv.org/abs/2412.17227), [DietCORP](https://arxiv.org/html/2507.02800v2)

**Novelty.** Neither LLM correction nor pseudo-labeling is new. The contribution would be demonstrating benefit without evaluation-text training and quantifying when linguistic correction overrides neural intent.

**Cost.** About 3–5 days and 8–16 GPU h including approximately three short acoustic folds, candidate generation, and a small corrector/student run. A roughly 1–3B parameter-efficient model is the practical first target; an 8B 4-bit model fitting at inference does not guarantee a comfortable training setup.

**Test.** Make folds match the deployment claim: block-held-out trials for known-day decoding, or whole future sessions for new-day use. Remove exact and near-duplicate transcripts crossing folds. Fit only on training-derived predictions. Compare candidate ranking, constrained edits, and free rewriting on frozen exploratory evaluation, including a random-string/rare-word slice. For pseudo-labeling, first hide labels in a permitted training subset so accuracy of accepted labels can be measured.

**Latency class.** Full-sentence correction is finalization-only. A separately trained prefix corrector can be live. Pseudo-label pretraining is offline; chronological adaptation affects only later predictions.

**Risks/confounds.** Using B2T ’25 test recordings for training is transductive, even without labels; their labels cannot later provide an untouched inductive test. T12 held-out recordings require the same accounting. Keep inductive and transductive models distinct. Do not train an error corrector on val-dev transcripts. Public-text pretraining contamination is not fully auditable; use novel/compositional text tests where possible.

### 18. Evaluate continuous operation and endpointing before calling the pipeline complete

**Idea.** Remove oracle sentence resets in a controlled replay and evaluate speech activity, pauses, end-of-utterance decisions, and recovery after false starts.

**Mechanism.** Sentence boundaries provide valuable information. A system that works on presegmented speech can hallucinate words during rest, stop too early during slow speech, or accumulate recurrent/search-state errors. CTC blank is “emit no label here,” not a calibrated detector of silence or intent.

**Evidence.** Littlejohn et al. demonstrate continuously streaming ECoG speech synthesis and examine implicit speech detection; this is direct streaming-BCI precedent, with a different output modality. [Littlejohn et al.](https://pubmed.ncbi.nlm.nih.gov/40164740/)

**Novelty.** Continuous neural decoding is not new. The value here is exposing how much of the text decoder’s accuracy/delay frontier relies on supplied boundaries.

**Cost.** Roughly 1–3 days and 0–2 GPU h for replay; real rest/no-speech data may be the limiting resource. Artificially concatenated trials are a stress test, not a substitute for continuous recordings.

**Test.** Replay long streams with actual available pre/post speech periods, if they exist. Score false words per no-speech minute, false endpoints, truncation errors, long-pause recovery, and sentence-final delay. Distinguish a fixed timeout from a learned endpoint detector. If only segmented speech exists, publish that limitation explicitly.

**Latency class.** Live, with a separately reported endpoint waiting budget.

**Risks/confounds.** Zero padding is not real rest. Trial-concatenation discontinuities are artificial. Attempted speech, inner speech, speech perception, and silence are not interchangeable negative classes. The lab ECoG track needs its own acquisition, reference, label, timing, and consent/task audit before importing any inputs or endpoint model.

## What the future-context result can and cannot mean

The current result supports this narrow claim: **under the tested 20k checkpoints and separately tuned acoustic scales, this bidirectional system has lower exploratory val-dev WER than the causal system.** It does not yet identify how much gain comes from future input, extra capacity, optimization, alignment behavior, or selection variability.

Use a staged experiment:

1. Compare the equal-capacity correction heads in idea 1. This is the cheapest bounded-context test.
2. Repeat the causal and bidirectional comparison with multiple seeds. Include a parameter-matched control and report both shared parameters and inference cost; do not assume equal hidden width implies equal capacity.
3. Use the same smoothing, patch anchoring, trial handling, minibatch policy, training schedule, and decoder grid. Report a frozen common decoder configuration and each model’s equally budgeted tuned configuration.
4. Evaluate WER, greedy phonemes, neural likelihood margins for competing words, oracle n-best coverage, and score calibration. A small PER difference with a large WER difference can result from changing which errors occur, confidence, or beam survival.
5. Plot a right-context curve using trained bounded models. Stratify by source, within-word versus boundary confusions, word predictability, and session. Audit layer-by-layer support with two inputs sharing a prefix but differing in the suffix: outputs whose declared support excludes the changed suffix must remain identical within numerical tolerance.

| Pattern observed | Plausible explanation | What it does not prove |
|---|---|---|
| Most gain arrives within a few hundred milliseconds and improves phonetic contrasts | Local dynamics or later useful evidence | Physiological coarticulation specifically |
| Gains grow around a word-scale interval and especially on predictable sentences | Following-word evidence and learned linguistic context | That future samples contain an explicit semantic code |
| Gains mainly improve score calibration or candidate survival | Better alignment/confidence helps the LM/search | Better phoneme representations in a neurophysiological sense |
| A larger causal model recovers most of the gain | Capacity/optimization explanation | That future context never helps |
| Gain is largest near supplied sentence endings | Endpoint, length, padding, or final alignment effects | Deployable continuous-stream benefit |

Offline suffix masking/replacement, lag-specific probes, and array ablations can localize sensitivity. They remain exploratory: replacement creates distribution shift, CTC time is model-dependent, and array differences also reflect signal quality and coverage. To distinguish motor preparation, coarticulation, delayed neural evidence, and residual overt speech feedback convincingly requires better timing/behavioral annotations or a targeted recording experiment. Your present observation cannot settle that neuroscience question.

## A defensible evaluation design

### Confirmation with an exposed validation set

There is no statistical trick that turns repeatedly inspected val-dev into an independent test. Session bootstrapping quantifies a component of variability conditional on the chosen models and development process; it does not correct adaptive model/settings selection. Repartitioning already exposed data does not erase that history. [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)

Useful work is still possible:

- Freeze one primary comparison, one practical target, a small tuning protocol, a fixed seed budget, and stopping rules before the next runs. Preserve a dated analysis plan and all results, including failures. An example target is at least a chosen WER reduction at unchanged wrong-lock rate and lower waiting; the threshold must be selected before seeing those runs, not justified afterward.
- Do grouped/chronological nested reanalysis from scratch, with preprocessing, adaptation, checkpoint choice, decoder tuning, and policy calibration confined to the appropriate training folds. Describe it as robustness analysis on an exposed corpus, not restored independence.
- An unused T12 partition or genuinely untouched lab data can test a frozen transferable method, after preventing label/text and feature leakage. That supports transfer to that setting, not an independent T15 estimate.
- The strongest T15 confirmation requires new recordings or an organizer-mediated, one-shot hidden-label evaluation if still available and permitted. Check availability; do not assume the closed competition accepts submissions. Keep those recordings out of self-supervised training for an inductive claim. Even a hidden label set loses value if repeatedly queried through leaderboard tuning.
- Independent training seeds can confirm reproducibility of an effect on the fixed exposed corpus. They cannot confirm participant/population generalization.

For uncertainty, use the same resampled sessions for both models and recompute total errors/total reference words in each replicate. Also show equal-session summaries, source-stratified effects, and seed-to-seed contrasts. Sessions contain correlated blocks and may share repeated sentences; use a session/block or repeated-sentence sensitivity analysis rather than treating 8,277 words as independent. All participant-level conclusions remain conditional on T15 unless separately tested.

### Define the clocks before the 200/500/1,500 ms conditions

Do not publish those numbers as “word latency” without an anchor. Specify at least:

- **Feature availability:** when each released bin would become available. Upstream raw-signal filtering, normalization, and clipping are unresolved and sit outside the currently validated replay boundary.
- **Startup:** 280 ms to assemble the first patch in the present implementation. It is not automatically a recurring 280 ms delay if frames are anchored to their right edge.
- **Explicit right context:** the latest neural input used for a prediction relative to the prediction frame’s declared anchor.
- **Emission:** when a candidate word first appears; CTC may wait despite causal architecture.
- **Compute and queue:** model, beam, LLM, scheduling, transfer, and any backlog; report distributions and deadline misses.
- **Commitment:** extra waiting until a displayed word is irreversible.
- **Endpointing:** waiting to decide the utterance is over; whole-sentence rescoring follows as a separate stage.

For the first paper-quality sweep, use two independently labeled axes rather than hiding them in one number:

| Axis | Suggested settings | Defensible interpretation |
|---|---|---|
| Input right-context cap R | 0, 200, 500, 1,500 ms, full sentence | Amount of future released-feature data the model may use |
| Maximum post-display revision window C | 200, 500, 1,500 ms, unconstrained | How long a word candidate may remain revisable after first display |

The 80 ms output grid gives actual context choices of 0, 160, 480, and 1,440 ms under those caps. Report actual values. If a cap is meant to include compute/queue time, subtract measured tail overhead first and reduce context as necessary: 480 ms context plus more than 20 ms overhead misses a 500 ms deadline. A nominal 200 ms context cap does not promise first output within 200 ms of speech onset, given startup.

For a fixed revision-window policy, declare what happens at expiry: commit, withhold/abstain, or flag failure. Count uncommitted and deleted words; do not make a low-risk policy look fast by reporting only easy committed words. First-display clocks can be gamed by delaying first display, so report emission and commitment separately and use a common offline timing anchor for comparisons when possible.

Do not simply add medians from separate latency components; calculate the realized end-to-end timestamps per event.

### Word timing without word alignments

You can measure display age, revision events, and lock times exactly in replay. You cannot recover exact attempted-word onset/end times from text and neural bins alone without additional assumptions.

For an explicitly model-based timing analysis:

1. Freeze a common offline aligner that is not the model under test. Convert reference words to permitted phoneme pronunciations and use CTC forward–backward alignment to obtain distributions over boundary frames, not just an argmax path.
2. Report timing uncertainty and sensitivity to at least one different aligner. Bidirectional and causal aligners can place boundaries differently. Intervals from these models are alignment uncertainty, not validated physiological confidence intervals.
3. Use references and whole trials only in this offline scoring component. They must never influence an online commitment, adaptive LM weight, or endpoint decision.
4. If residual speech audio, video, EMG, task events, or suitable lab annotation exists, manually inspect a stratified subset and estimate systematic alignment bias. Acoustic alignment of severe dysarthria may itself be uncertain.
5. Without such external checks, label the measure “delay relative to model-estimated word end” or “post-first-display commitment delay.” Do not call it true speech latency. Include errors, missing words, and coverage rather than only correctly recognized words.

CTC’s arbitrary spike timing is a documented reason for this caution. [Kurata and Audhkhasi](https://arxiv.org/abs/1904.08311)

### The smallest useful reporting panel

- Final live WER and finalization WER, with paired session uncertainty and seed contrasts.
- Irreversible wrong-word rate, definition of its denominator, committed coverage, deletion/missed-deadline rate, and correction burden.
- Revision count or revised-word burden, not just eventual stability. ASR has established metrics for this distinction. [Shangguan et al.](https://arxiv.org/abs/2006.01416)
- Emission/commitment delay distributions, including upper tails, with the timestamp anchor stated.
- p50/p95/p99 processing time, worst observed backlog, peak RAM/VRAM, and deadline failures with the full system running.
- Session/source breakdowns, rare words and random strings, and a prespecified check on meaning-changing errors such as negation, numbers, and names.
- Whether inputs, labels, text, or normalizer statistics from each evaluation partition were seen at any training/adaptation stage.

## Critique of the current interpretation

**The 582/93/29 breakdown is descriptive, not a causal decomposition.** A wrong greedy phoneme need not make the reference word unavailable in the beam. Conversely, correct phonemes can map to multiple spellings or word segmentations. Reclassify errors by whether the reference pronunciation is supported, whether the correct word sequence survives the search, and which scoring term changes the ranking. The n-best oracle shows that candidate selection has substantial headroom.

**The bidirectional model’s lower preferred acoustic scale does not establish a mechanism.** A scale reflects logit calibration, score conventions, blank handling, and sequence length as well as the acoustic/LM balance. It is not direct evidence that the model contains less neural information or more useful linguistic context.

**An unchanged word is not a correct word, and a word finalized eight seconds before sentence end need not have been recognized promptly.** Sentence-final waiting can be large even when physiological word latency is poor. Audit whether “0.4% locked words wrong” means wrong against the reference or merely different from the eventual final output, and report coverage. An offline policy replay that does not constrain the beam is not evidence for the same policy inside search.

**Your current pipeline is causal relative to released features, not yet proven causal from raw acquisition.** If per-block statistics were computed using later data, the future entered before your smoother. Re-normalizing released z-scores cannot reliably remove that dependence, especially after clipping. Verify acquisition/preprocessing code or obtain provenance before making a raw end-to-end claim.

**Day-layer validation is not new-day generalization.** A validation trial from a session with a supervised day layer tests interpolation within that recording context. Calibration-free decoding requires a new session whose labeled trials and future statistics were never used.

**The 61% sentence-mix result needs a diagnostic, not a subtraction.** It does not establish that exactly 39% is neural drift. Fit source-standardized and within-source session comparisons, inspect overlap in source distributions, and quantify uncertainty. If sources occur in disjoint sessions, some separation is not identifiable. A text-only prediction baseline or neural-input shuffle is an offline diagnostic of prompt predictability, not a useful decoder.

**A single failed causal Transformer run does not settle architecture.** The broader literature includes successful causal Transformer speech models and streaming ECoG systems. A fair negative result about this particular architecture, training budget, and dataset remains valuable. [Feghhi et al.](https://arxiv.org/abs/2507.02800), [Littlejohn et al.](https://pubmed.ncbi.nlm.nih.gov/40164740/)

**Deployment evidence is broader than competition WER.** Card et al.’s June 2026 Nature Medicine paper documents long-term independent home use, making continuous operation, recovery, and calibration burden directly relevant comparison axes. Its private home-use neural recordings/transcripts are not an assumed source of additional training data. [Card et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13375540/)

**Protect the distinction between wording and intent.** Strong LMs can improve standard WER while replacing rare but intended words with plausible alternatives. Preserve a provisional/correctable UI and measure the cost of a wrong irreversible output. Semantic similarity is a supplementary diagnostic, not permission to rewrite intended wording.

**A reproducibility failure is also an experimental confound.** For WSL/CUDA interruptions, log completed optimizer steps, actual sampling exposure, optimizer/scheduler/random-number states, and any recovery. Runs silently restarted with different schedules or data streams are not matched comparisons. Evaluate deadlines with real-time pacing; fast offline replay misses queue effects.

## If only a few experiments are possible

1. **Best next GPU experiment: the bounded-future head.** Train the same correction architecture with 0 and 480 ms right context on the same causal checkpoint. If useful, add 160 and 1,440 ms and a seed repeat. This turns the most interesting result into an actionable context–accuracy curve without another large model.
2. **Best next CPU experiment: score and reselect the existing n-best list.** Start with a calibrated linear score and minimum-risk selection. It directly tests whether the 2.90% oracle is useful headroom or largely inaccessible under realistic scoring.
3. **Best interaction experiment: risk-based commitment.** Compare neural-supported/LM-robust commitment with the three-newer-word rule at matched wrong-lock rate and coverage. It addresses the observed seconds-long waiting cost rather than merely speeding up an already fast GPU model.
4. **Best validity experiment: freeze and repeat the future-context comparison under a completed short schedule.** Use equal decoder-tuning budgets, seed replication, support auditing, and a parameter-matched control. This makes either a positive or negative result credible on the exposed corpus.
5. **Best additional-data experiment: T12 transfer with separate front ends and a text-only control.** Defer 367-hour heterogeneous pretraining and 8B free-form correction until this much cheaper supervised transfer test earns them.

Do not execute all eighteen ideas as an unstructured search. A coherent first study can be limited to one acoustic backbone, a small right-context sweep, and two commitment policies. Its strongest claim would be a reproducible, explicitly exploratory accuracy–revision–delay frontier, followed by a frozen transfer or genuinely unseen-data test when available.

