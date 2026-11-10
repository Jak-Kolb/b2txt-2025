# Making the Transformer competitive: assessment and experiment plan

27 October 2026

Based on `current_knowledge.md` dated 26 October, your new results, and the exact configuration and training curves you supplied for `tfm_conv_fix_r2_60k_s10`. This document contains the complete recommendation. I reviewed the reported experiments and primary research; I have not inspected your local training implementation, run these experiments, or modified the pipeline. New hyperparameters below are proposed starting points, not established optima.

## 1. My recommendation

**Yes—making a Transformer work better is a feasible research direction, and I would continue it. I would keep your current convolutional Transformer and test a more useful initialization before building a larger model or repeating the heterogeneous SSL run.**

Your current model is already a credible design: pre-normalized attention, local causal convolutions, relative positions, AdamW, warmup, and carefully restricted future context. This is no longer a case of applying the GRU training recipe unchanged to a Transformer. There is no clear architectural mistake in the configuration you supplied.

However, “Transformers eventually win with more data” is a hypothesis here, not a guarantee. Additional independent neural recordings, repeatedly revisiting the same recordings, and adding recordings from different observation processes are three different things. The literature supports pursuing Transformers; it does not establish that more updates on these 8,072 trials will close the gap.

My preferred sequence is:



1. Establish the best result already present in your saved checkpoints, including acoustic scales below 0.25.
2. Run one focused supervised experiment sharing the large day-specific input matrices.
3. Test whether the successful GRU can teach the Transformer a useful **hidden speech representation**, then remove the teacher and fine-tune with ordinary CTC.
4. Include controls that tell us whether masking helps and whether the gain exceeds simply spending that time on supervised training.
5. Expand to genuinely new data and broader self-supervised pretraining only after establishing that the transfer mechanism works.

The main new bet is **GRU-to-Transformer representation pretraining**. It uses something you already know works, fits the available hardware, and adds nothing to inference. It is distinct from both the unsuccessful raw-reconstruction pretraining and the unsuccessful CTC-logit distillation. It is also unproven: the experiments below are designed to make a negative result informative.

## 2. What your new evidence says

### The performance gap is real, but the final-WER search is unfinished

| Model                 |   Greedy PER | Best measured final WER | Acoustic future context |
| --------------------- | -----------: | ----------------------: | ----------------------: |
| Conv Transformer, 60k |       10.17% |       9.45%, scale 0.25 |                  160 ms |
| `la1344_r2_60k` GRU   |        8.93% |                   7.85% |                  160 ms |
| Transformer minus GRU | +1.24 points |            +1.60 points | Matched nominal context |

The Transformer needs about a **12.2% relative reduction in PER** and **16.9% relative reduction in WER** to match those GRU numbers. These are substantial gaps, but small enough to justify a bounded research campaign.

Because 0.25 was the lowest acoustic scale tested, 9.45% is the best _measured_ WER, not a demonstrated optimum. Better scaling could recover part of the WER gap. It cannot explain away the greedy PER gap.

These are final-transcript WERs. They do not establish the Transformer's live-word or deadline-respecting performance.

### The architecture should remain the experimental base

| Component         | Actual configuration                                                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Input             | Per-day 512×512 affine layer with softsign; input dropout 0.2                                                                      |
| Patches           | 14 trailing 20 ms bins; stride four bins; 80 ms output interval                                                                    |
| Encoder           | Eight blocks, width 384, six heads, FFN width 1,536                                                                                |
| Block order       | Pre-norm attention → causal convolution module → FFN                                                                               |
| Convolution       | Pointwise/GLU, depthwise kernel nine, per-frame LayerNorm, SiLU, pointwise                                                         |
| Attention         | ALiBi; current frame plus previous 63; only the final attention layer has two future frames                                        |
| Regularization    | Dropout 0.2; stochastic depth rising to 0.1; shared weight decay 0.05, exempting biases/LayerNorm; no day-layer decay              |
| Optimization      | AdamW, peak LR 0.001 for shared/day parameters; 1k warmup; cosine to 0.0001 at 60k; clip norm 10                                   |
| Augmentation      | White noise std 1.0; constant offset std 0.2; random cut up to three bins; causal Gaussian smoothing std two bins; no time masking |
| Training          | Batch 64 from four sessions; bf16; model seed 10; data-order seed 1; 189 minutes                                                   |
| Parameters        | About 20.6M shared + 11.8M day-specific                                                                                            |
| Deployment status | Full masked forward only; cached streaming implementation not yet built                                                            |

Two details are favorable. First, the two future frames occur only once, so the stated architecture does **not** multiply 160 ms by eight layers. The causal convolution after that attention layer does not add future context. Second, 64 frames is a per-layer attention window, not a five-second limit on the entire stacked network's history.

Keep both properties. There is no reason to begin by rewriting this as Emformer or replacing every block.

### The loss curve calls for diagnosis, not an automatic longer run

| Update                      |   10k |   20k |   30k |   40k |   50k |   54k |   58k | 59,999 |
| --------------------------- | ----: | ----: | ----: | ----: | ----: | ----: | ----: | -----: |
| Trainer validation PER      | 12.93 | 12.75 | 12.40 | 11.64 | 10.97 | 11.08 | 10.80 |  11.09 |
| Trainer validation CTC loss |  23.2 |  26.7 |  27.7 |  28.6 |  28.9 |  29.8 |  29.4 |   30.0 |

The trainer evaluates a different aggregate from the headline val-dev result. Reconcile the example lists, weighting, preprocessing, and checkpoint identity before comparing 11.09 with 10.17; the difference is not itself evidence of a bug.

The rising loss does **not** directly prove that individual wrong frames became more confident. CTC loss is the negative log probability of the reference sequence, summing over valid alignments. Sequence probability and greedy edit distance can move in opposite directions. Confidence concentration, alignment changes, and generalization failure are possibilities that require additional measurements. [Original CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

The last 10k updates look approximately flat in PER. Reaching the LR floor does not mean the model has exhausted its capacity, but neither does it justify assuming that another 60k updates will help. At batch 64, 60k updates already correspond to about **476 nominal passes through 8,072 trials**, before accounting for nonuniform session sampling. More updates would predominantly add repeated exposure.

## 3. What the research supports—and what transfers to this project

**LightBeam supports a strong supervised Transformer baseline.** Its reported private-test Transformer WER is 4.81% with the baseline decoding pipeline versus 6.72% for its GRU reproduction, averaged over ten seeds. These are final-transcript results with different evaluation and language-model conditions from yours. They are not a prediction of your attainable WER. The encoder was trained with supervised CTC, so successful Transformer decoding does not inherently require SSL. [LightBeam paper](https://arxiv.org/html/2603.14002)

Its public implementation, inspected at commit `90e0ac086d4dee643be154f446a6c374600575d7`, differs materially from your model: four-bin nonoverlapping patches with LayerNorm–linear–LayerNorm, five plain attention blocks, no day affine layer in the released configuration, much smaller weight decay, and weaker additive noise. This public revision is not a verified archive of the paper run. Its centered smoothing also consumes approximately 80 ms of future input despite causal attention. These differences make “our LightBeam-style run failed” weaker evidence than a faithful reproduction failure. They provide hypotheses, not a reason to change all settings together. [Pinned configuration](https://github.com/ebrahimfeghhi/brainaudio/blob/90e0ac086d4dee643be154f446a6c374600575d7/src/brainaudio/training/utils/custom_configs/neurips_b2t_25_chunked_transformer.yaml), [smoothing implementation](https://github.com/ebrahimfeghhi/brainaudio/blob/90e0ac086d4dee643be154f446a6c374600575d7/src/brainaudio/training/utils/augmentations.py)

**BIT provides direct neural-speech pretraining evidence, with important limits.** In its latest version, T15 validation PER improves from 8.87% for a scratch Transformer to 7.61% with 97.8 hours of human pretraining, and 7.12% with 366.8 hours including monkeys. The objective reconstructs normalized neural features with masked MSE. The encoder is bidirectional, and the main protocol uses unlabeled evaluation recordings. An exclusion experiment still reports T15 cascaded WER of 4.36% versus 4.06% when speech data are included; the gains therefore should not be dismissed as merely transductive exposure. None of these numbers proves a benefit at your 160 ms future limit or immutable-word deadlines. [BIT v5, especially Appendices A, C, O and P](https://arxiv.org/html/2511.21740v5)

The inspected BIT repository exposes model and supervised training code, but I did not find a documented complete SSL launcher or downloadable pretrained encoder. Treat reproduction as engineering work, not a ready checkpoint to load. [Official BIT repository](https://github.com/yzhang511/bit)

**The longitudinal T15 deployment supports causal Transformers and simpler session adaptation.** It uses a causal Transformer with day-specific additive offsets. Its offline pretraining uses prior labeled and participant-confirmed utterances; that is supervised training, not masked SSL. Its much larger longitudinal dataset and ongoing adaptation are central differences from your public-data experiment. The authors did not perform a controlled repeated evaluation of the architecture switch. This is evidence for feasibility and a useful adaptation hypothesis, not proof that switching to twelve layers will reproduce its accuracy. [Card et al., Nature Medicine 2026](https://www.nature.com/articles/s41591-026-04414-6)

**Audio SSL supports matching pretraining to deployment context.** BEST-RQ compares causal and noncausal pretraining of a streaming architecture and finds similar final WER but different emission timing. Its large audio models and RNN-T task are not neural-speech evidence. The transferable lesson is to train the student with its intended context restriction and measure timing, rather than assuming a full-context pretraining gain survives streaming conversion. [BEST-RQ](https://proceedings.mlr.press/v162/chiu22a/chiu22a.pdf)

**Latent targets are a reasonable alternative to raw reconstruction.** data2vec predicts teacher representations from masked input; DistilHuBERT transfers hidden representations between speech models. These motivate the experiment below, but neither validates this particular causal GRU-to-Transformer transfer. [data2vec](https://proceedings.mlr.press/v162/baevski22a.html), [DistilHuBERT](https://arxiv.org/abs/2110.01900)

NDT2 and NDT3 further support causal transfer across neural recording sessions, but their motor-decoding tasks, feature conventions, and tokenization differ from your phoneme decoder. Their public checkpoints are research references, not drop-in weights for your 512-feature input. I would not make adapting those checkpoints the first experiment. [NDT2 paper and code](https://github.com/joel99/context_general_bci), [NDT3 paper and code](https://github.com/joel99/ndt3)

## 4. First: evaluate the work already completed

This stage needs inference and diagnostics, not another training run.

### A. Compare the saved checkpoints consistently

Evaluate 20k, 40k, the best-PER checkpoint, and the fixed final checkpoint on identical example lists. Record both val-dev and the trainer's full evaluation aggregate. Keep the distinction between a fixed endpoint and a validation-selected checkpoint explicit.

Use an initial acoustic-scale grid of **0.10, 0.15, 0.20, 0.25, 0.35, 0.50**. If the minimum remains at 0.10, extend to 0.05. Keep other decoder settings fixed. Give the GRU the same tuning opportunity.

Use the existing session-disjoint validation halves for tuning/reporting, reversing them for a cross-fitted estimate if practical. These halves are already historically exposed; this reduces immediate tuning bias but does not create a fresh test set.

Log substitutions, deletions, insertions, candidate coverage, and word-boundary timing. A lower acoustic scale that merely replaces neural evidence with stronger language priors may improve final WER without improving live candidate availability.

### B. Find out whether the Transformer fits the training data

On a fixed training subset, evaluate both Transformer and GRU with augmentation and dropout disabled. Compare PER, target-length-normalized CTC loss, and per-session performance with validation. Check that CTC/log-softmax calculations and reductions are numerically appropriate under bf16; a float32 diagnostic pass is inexpensive.

| Observation                                       | Interpretation to investigate                                                        |
| ------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Transformer fits train much worse than GRU        | Optimization, excessive regularization, input representation, or capacity allocation |
| Transformer fits train well but validation poorly | Generalization and session adaptation become stronger suspects                       |
| Similar PER but much worse WER                    | Calibration, blank/word-boundary behavior, and decoding deserve more attention       |
| Lower PER accompanies later boundaries            | An accuracy gain may be unusable at the intended deadline                            |

If confidence looks problematic, a small temperature diagnostic can help. Temperature scaling preserves frame argmax and hence greedy PER; any resulting WER gain is a decoding/probability effect. Do not use it to declare the acoustic representation fixed, and do not launch a large temperature-by-scale sweep.

### C. Check the context contract before expensive pretraining

With dropout disabled, perturb raw input after frame `t`'s allowed endpoint plus 160 ms. Output `t` should not change. Include smoothing, patching, normalization, and padding in this test.

Inspect the final layer's ALiBi behavior for future positions. A causal-only signed-distance implementation extended to future keys can accidentally reward them rather than penalizing distance. This is an implementation check, not a claim that your code has that problem. [ALiBi paper and implementation](https://github.com/ofirpress/attention_with_linear_biases)

The eventual cached implementation must match the full masked forward, including startup and trial-end handling. Do not charge only attention's declared lookahead while omitting preprocessing or output buffering.

## 5. One focused supervised change before the pretraining comparison

**Test sharing the day-input matrix while retaining day-specific offsets.** About 36% of the current parameters are in the 45 independent input matrices. Sharing them could improve statistical efficiency for the Transformer. The fact that these matrices work well for the GRU does not establish that the same allocation is best here.

Use this specific partial-sharing variant:

```text
Current:  u = softsign(A_day x + b_day)
Trial:    u = softsign(A_shared x + b_day)
```

Keep the 512-dimensional output, dropout, patching, all eight blocks, ALiBi, 160 ms future context, augmentation, batch sampling, and complete 60k schedule unchanged. Preserve the input-layer optimizer treatment so this experiment primarily tests sharing. This is a proposed intermediate design; it is not an exact reproduction of the published offset-only model.

It reduces the input matrices from about 11.8M parameters to about 0.26M, plus small day offsets. Compare fixed final checkpoints first. Report the separately selected best-PER checkpoints as secondary results. If it helps, repeat it with another seed before calling it established.

If it is neutral or worse, keep the current input layers for pretraining. Do not turn a negative result into an obligation to sweep every adapter design.

**One conditional alternative:** if the training-set diagnostics show that the Transformer substantially underfits relative to the GRU, a shared-weight-decay ablation from **0.05 to 0.005**, with everything else fixed, is more directly motivated. Your existing dropout, stochastic depth, and neural noise already provide substantial regularization. This is a single diagnostic alternative, not an instruction to run a full optimizer grid. Do not change dropout, noise, LR, and weight decay together and attribute the result to one of them.

## 6. The main experiment: pretrain on useful speech representations

### Why this is the first pretraining bet

The old SSL runs fine-tuned **`la_r18`, a GRU**. They do not establish that Transformer pretraining failed. They do establish that repeating that objective/data mixture unchanged is poorly motivated.

Raw reconstruction asks the model to preserve every predictable aspect of the neural features. That can include recording-day differences, smoothing, and nuisance variability. Your successful GRU already extracts features that support phoneme decoding. Transferring those features is a more targeted way to test whether the Transformer is missing a useful inductive bias or initialization.

Call this **supervised representation transfer**, not pure SSL. The teacher learned from transcripts, even though the pretraining loss itself uses hidden states. Success would establish a useful training route; it would not establish a neural foundation model or prove the benefit of extra unlabeled data.

### Teacher

Start with the existing **20k `la1344_r2`** model. Your record gives it 7.74% final WER, while longer training has tended to delay emissions. This is a practical initial choice, not proof that it is the optimal teacher.

Freeze it in evaluation mode. Extract hidden states from causal layers four and five, **before the backward/lookahead head**. Verify that these states do not depend on future input. Use separate target heads because layers have different representation coordinates.

Use clean, consistently preprocessed T15 training input for teacher targets. Normalize each target vector across its feature dimension at the same timestamp, or with fixed training-only statistics. Do not normalize over an entire future-containing trial. Keep the physical timestamp mapping explicit.

### Student and objective

Keep the selected eight-block Transformer. Its student forward uses the same 160 ms total future limit as deployment. Attach two small training-only projection heads to its final hidden output, each predicting one teacher layer's normalized state at the **same input timestamp**. The teacher target is causal; the student may use its allowed two future frames. This is a deliberate bounded-context transfer experiment, not an exact data2vec implementation.

```text
For each selected output position t:
    target_4 = stop_gradient(feature_normalize(teacher_layer_4[t]))
    target_5 = stop_gradient(feature_normalize(teacher_layer_5[t]))

    prediction_4 = feature_normalize(project_4(student[t]))
    prediction_5 = feature_normalize(project_5(student[t]))

Pretraining loss = mean squared prediction error
                  averaged over features, selected positions, and both targets
```

Exclude padding. Use full trials initially; independently cropping the student's history while caching a full-history teacher target can create an impossible prediction task.

This loss replaces CTC during the pretraining stage. It avoids repeating the earlier situation where a nominally small distillation term was 5–50 times larger than CTC. Afterward, discard both projection heads and fine-tune the whole student with CTC. The teacher is absent at inference.

The earlier KL experiments asked a student to match sharp blank/phoneme probabilities and inherited teacher timing. This proposal transfers dense states before that output head, then releases the student from the teacher during supervised fine-tuning. It may still inherit delayed or recurrence-specific representations; hidden-state transfer is not automatically immune to either problem. Fine-tuning can move beyond the teacher's representation, but surpassing the teacher is an experimental question, not an expected consequence of using a Transformer.

### Masking: modest, explicit, and tested

For the masked arm, start with **10% raw-time coverage in 320 ms spans**. These are trial settings, not a published optimum. They are intentionally modest given your history of masking-induced delay.

Mask normalized raw bins **before Gaussian smoothing and before overlapping patches**. Replace them with the training mean. Otherwise, a later visible smoothed bin or overlapping patch can carry the supposedly hidden values. Check this mechanically: changing only the hidden raw values must leave the student's entire processed input unchanged when the mask is applied.

Compute the hidden loss at output positions whose trailing patch lies entirely inside a masked span. Use the same selected positions in the unmasked control. Align span starts to the frame grid so that the 320 ms span actually contains eligible 280 ms patches; count eligible targets instead of assuming a coverage percentage translates directly into useful loss samples. If too few survive, increase the number of spans before increasing their duration.

Use the teacher's clean version of the same physical crop. Cache targets only when preprocessing and crop offsets can be mapped exactly. Do not match tensors by array index when their timestamps differ.

During CTC fine-tuning, return to the ordinary unmasked training recipe. Masked pretraining may still teach delayed representations; measure the resulting word timing rather than assuming removal of masks removes that risk.

### The three arms that make the result interpretable

| Arm                   | First 20k updates                               | Next 60k updates | Question answered                              |
| --------------------- | ----------------------------------------------- | ---------------- | ---------------------------------------------- |
| C: supervised control | CTC                                             | CTC              | Does ordinary extra training explain the gain? |
| U: unmasked transfer  | Predict clean GRU states without input masking  | CTC              | Does transferring the representation help?     |
| M: masked transfer    | Predict the same GRU states with the mask above | CTC              | Does masked prediction improve transfer?       |

Keep the existing 60k scratch model as another reference. For C/U/M, use the same architecture, input layers, trial sampling policy, downstream augmentation, and checkpoint selection rule. Reset the optimizer at the stage boundary in all three arms. The CTC control may retain its already learned output head; give the control that advantage rather than deliberately weakening it.

A concrete initial schedule for the controlled comparison is 1k warmup and cosine decay in each stage: pretraining/shared peak **3e-4**, ending at **3e-5**; downstream shared peak **3e-4** and fresh CTC-head peak **1e-3**, each ending at one tenth of peak. Apply the same parameter-group schedule to the supervised control, retaining the selected day/shared LR ratio. This protects transferred features from an immediate large update while allowing the fresh head to learn. These values are experimental choices, not inferred optimal settings.

Equal updates are not equal compute. Record processed trials, frames, GPU time, and teacher-cache cost. If the transfer arms cost materially more, give the CTC control a comparable time budget before claiming pretraining efficiency. A gain requiring more computation may still be useful; report that distinction.

Run the first comparison with the same seed policy as your current experiment. Repeat the winning arm and its control with two additional model seeds if the initial gain is meaningful. Do not confirm only the winning seed.

Before scheduling the full comparison, run a short plumbing pilot: verify timestamp correspondence, mask isolation, nonconstant teacher targets, noncollapsed student predictions, finite gradients, and decreasing training loss. This checks that the proposed objective is being learned; it does not select a winner from a short-run PER, since your present Transformer improved substantially after 30k updates.

### How to interpret the outcome

| Result                                    | Decision                                                                                              |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| U beats C; M does not beat U              | Keep representation transfer; drop masking                                                            |
| M beats both U and C                      | Evidence that masked representation pretraining adds value                                            |
| C improves as much as U/M                 | Extra supervised training or the revised schedule explains the improvement                            |
| Hidden loss improves but PER/WER does not | Reject this target/objective as useful initialization under the tested conditions                     |
| PER improves but deadline WER worsens     | Reject it as a live-decoder improvement; inspect boundary delay                                       |
| All three remain clearly behind the GRU   | Stop this transfer branch and test one input-representation hypothesis rather than scaling it blindly |

## 7. What to do about more data and pure SSL

Use **only the original T15 training recordings** in the first transfer comparison. That isolates the objective. Reusing them without their labels is not new independent data, but it can still test whether a different learning objective produces a better initialization.

Do not reuse the historical 74.65-hour pool without rebuilding its manifest:

- It included unlabeled test recordings. The primary new comparison should exclude validation/test recordings from every training stage. A deliberately transductive experiment is a separate result.
- The voice release contains substantial recording overlap. Count unique recording duration, not filenames or nominal hours.
- Its log transform, rolling normalization, long causal smoothing, and crop differences make it a different observation process. These transformations are not generally invertible; z-scoring does not recreate the missing fast temporal information.
- T12 requires its own input mapping. Feeding T12 channels directly through a T15 teacher does not produce meaningful speech targets merely because tensor shapes match.

**If representation transfer works**, the next data experiment should add a small, verified increment of genuinely distinct and compatible T15 recordings, with an equal-compute control on the original data. This tests data value separately from objective value. If only the heavily processed voice release is available, treat source adaptation as its own experiment and acknowledge that it may not improve fast word timing.

**If the aim is specifically pure SSL**, use a causal/bounded-context normalized-feature masked-MSE baseline inspired by BIT/NDT2, with separate source input mappings and training-only normalization. Do not use a Poisson count loss on clipped z-scored features. Start with a clean, small pool and a matched scratch control before adding multiple subjects or monkey data. Published gains justify testing this; your previous failures justify controlling it carefully.

I would postpone EMA self-distillation, large neural-foundation checkpoints, and a 367-hour corpus rebuild. They add moving targets or major domain-adaptation work before answering whether this Transformer benefits from a useful speech representation at all.

## 8. Keep the original real-time objective in the experiment

The first deployment target for this 160 ms model should be **500 ms**, with 1.5 s reported alongside it. At 200 ms, 160 ms of acoustic future context leaves very little room for emission delay and computation. A causal variant is the more sensible subsequent branch for that budget.

Before promoting a model, implement cached streaming and compare it with the full masked forward. Cache the lower causal layers, wait for the final layer's permitted two future frames, and finalize each output once. Include convolution state, ALiBi positions, startup, and trial-end flush behavior. Measure integrated inference with the intended decoder/ranker load; isolated encoder timing is insufficient.

For every serious finalist, report:

| Measure                                       | Why it matters                                               |
| --------------------------------------------- | ------------------------------------------------------------ |
| Greedy PER                                    | Whether acoustic discrimination improved                     |
| Final WER with equally tuned decoder settings | Whether acoustic gains survive word decoding                 |
| Live WER at 500 ms and 1.5 s                  | Actual irreversible output quality                           |
| Deadline-respecting WER                       | Prevents late correct words from masquerading as success     |
| Word-boundary delay distribution              | Detects the failure mode seen with masking and long training |
| Candidate coverage at the deadline            | Separates missing evidence from poor candidate ranking       |
| Peak memory and integrated compute time       | Determines whether the result is deployable                  |

First compare with a fixed downstream pipeline to isolate the acoustic replacement. Then allow the finalist the same model-specific scale/ranker fitting procedure as the GRU. Keep those two comparisons separate.

Use the existing timing anchors consistently and retain their limitation: they are model-derived, not measured physical word completion. Do not regenerate easier anchors from the new Transformer.

## 9. Budget, milestones, and fallback

At your observed rate, 60k updates take 3.15 hours. An 80k-update arm would take roughly 4.2 hours **if throughput stayed the same**. One 60k supervised ablation plus three 80k arms therefore represents roughly **16 GPU-hours**, before teacher target generation, decoding, and seed confirmation. Pretraining throughput may differ, so this is a planning estimate rather than a runtime promise.

Cache teacher states in bounded shards so the GRU and student need not occupy GPU memory together. Measure peak memory on long trials. Keep the large decoder out of the training process, consistent with the instability already observed.

Use these decision gates:

1. **Baseline established:** saved-checkpoint comparisons and scale search no longer leave the present Transformer result ambiguous.
2. **Promising direction:** roughly 0.5 WER point or more improvement over the re-evaluated Transformer baseline, accompanied by a compatible PER result and no material deadline regression. This is an engineering screening threshold, not statistical significance.
3. **Credible gain:** the direction repeats across seeds and sessions. Report paired session uncertainty and per-session results; your repeatedly tuned validation set still limits claims.
4. **Transformer competitive:** within about 0.2 points of the GRU on matched final and deadline WER, followed by a reproducible advantage before declaring it the replacement. Matching the GRU is a milestone; it does not fulfill the eventual below-6% live-WER goal.

If transfer fails, my next single architectural experiment would be a **shorter, normalized input patch representation**: four-bin nonoverlapping patches at the same 80 ms output rate, with LayerNorm–linear–LayerNorm, retaining the current backbone and future limit. This tests whether the 280 ms flattened overlapping input is an unfavorable representation for attention. It changes available local input history and must be labeled accordingly. It is a fallback hypothesis, not another change to bundle into the first pretraining run.

The commitment I would make now is a small, interpretable Transformer campaign: qualify the model you have, test shared session adaptation, and run the three-way representation-pretraining comparison. That is enough to learn whether this route has traction without betting the project on generic scaling arguments.

## 10. Experiment record to preserve

Store the exact executed configuration and code diff, immutable checkpoint identity, train/pretraining recording manifests, evaluation example IDs, seed policy, optimizer stages, selection rule, decoder-scale curve, and metric definitions with every result. Record whether an input source has already been smoothed or normalized using information beyond the allowed timestamp.

The supplied reference run is:

```text
results/harness/train/20261027T000841Z_tfm_conv_fix_r2_60k_s10/config.yaml
results/harness/train/20261027T000841Z_tfm_conv_fix_r2_60k_s10/git_diff.patch
model_training/trained_models/tfm_conv_fix_r2_60k_s10_20261027T000841Z/checkpoint/
harness/registry/acoustic/tfm_conv_fix_r2_60k_s10_final.yaml
harness/registry/acoustic/tfm_conv_fix_r2_60k_s10.yaml
model_training/transformer_model.py
```

These are the project-relative paths you supplied, not files I have independently opened in this workspace. Preserve the executed diff because the implementation is currently uncommitted on `model_development`.
