# LightBeam reproduction gaps and prepared fixes, 2026-11-10 (moved unchanged from `current_knowledge.md` 4.8 on 2026-11-17; summary and results in its 4.9)

- **Reproduction gaps found 2026-11-10** (their public code at commit 90e0ac0, read file by file by a web-fetch agent;
  `brainstorm/PLAN.md` has the follow-up). The 2026-11-03 run was not faithful in these respects:

  | Setting | Their code | Our run |
  |---|---|---|
  | Time masks | 20 masks, each up to **5% of the trial's own patches** (`max_mask_pct: 0.05`), zeros on raw patches before the embedding | 20 masks of up to **5 frames**, learned token; realized coverage 22% (training log) |
  | Left context in training | sampled per batch, uniform **5-20 s**; 20 s at evaluation | fixed 20 s |
  | Positions | T5-style learned relative bias (shared across heads, distances clamped at 200) | ALiBi |
  | Gaussian smoothing | kernel 20, std 2, `padding='same'`: **centered, so it sees 4 bins (80 ms) ahead** | causal |
  | Other | random_cut 0; batches shuffled across all days; CTC loss divided by target length; dropout also on attention weights; checkpoint = best val PER | random_cut 3; 4 days per batch; per-trial CTC sum; fixed final / average |

  - The earlier reading "masks of <= 5 frames" was a misreading of `max_mask_pct: 0.05`. For the median training trial
    (~200 patches) their masks average ~5 patches each, about twice our masking (coverage not computed exactly).
  - Their decoder settings were tuned on the GRU and reused unchanged for the Transformer (acoustic scale 0.325,
    blank 90; Tables A3-A5), and it still won. So their Transformer was usable at the GRU's scale, while ours needs
    0.2: the overconfidence is probably a property of our run, not of the recipe. Untested.
  - Feghhi et al. 2025 (arXiv 2507.02800v2, B2T '24, 3-gram, validation): removing time masking costs +18% WER;
    replacing the relative bias with sinusoidal positions +10%; a time-masked GRU with the same recipe is only 4%
    behind the Transformer, the unmasked baseline GRU 20% behind. Most of the published Transformer-vs-GRU gap is
    the training recipe, and their GRU baseline is the organizers' untuned one.

- **More gaps found in their code on the second pass, 2026-11-10** (`training/trainer.py`,
  `models/transformer_chunking.py`, `training/utils/loss.py`, read directly):

  | Setting | Their code | Our 2026-11-03 run |
  |---|---|---|
  | Batch composition | shuffled passes over all trials, each trial once per epoch | 4 random days per batch, every day equally likely, trials with replacement: a trial of the smallest day (59 trials) is drawn 3.0x as often as trial-uniform, of the largest (364) 0.49x; the 10 smallest days hold 10.5% of trials and get 22% of the samples |
  | Dropout at the input | 0.2 on the frame embeddings only | 0.2 on the raw bins and 0.35 on the embeddings |
  | Feed-forward | dropout after the activation and after the block | after the block only |
  | Initialization | PyTorch defaults | Xavier-uniform weights (3x the default variance for square layers), zero biases |
  | Smoothing | centred, 9 taps: reads 4 bins = 80 ms ahead (not "a few bins" / 120 ms as first noted) | causal, 5 taps |
  | Warmup | 10 epochs (~1260 batches), stepped per epoch | 1000 batches |
  | Weight decay | none on LayerNorm, embeddings, biases | 1e-5 everywhere (negligible) |

  Their zero-filled masked patches become a constant learned vector after their LayerNorm-Linear-LayerNorm embedding,
  so our learned mask token is equivalent in function. Their data preparation for B2T '25 applies no transform to the
  released features.

- **Fixes prepared 2026-11-10, not yet run** (`results/harness/queue/tfm_lbfix_20261111.py`). New options, all off
  by default, unit-tested, and checked on CPU under `torch.compile` (compiled gradients equal eager to 1e-13; a
  combined masked + CR-CTC training step on a real batch runs for both arm shapes); no GPU run yet:

  | Option | What it does |
  |---|---|
  | `model.transformer.pos_bias=relative` | Learned bias per layer and relative distance, shared by heads, clamp 200, N(0, 1) init (their `rel_pos_bias`); replaces ALiBi. No streaming path |
  | `model.transformer.window_min=W` | Training only: attention window drawn per forward from [W, window] frames; evaluation uses `window` |
  | `time_mask_max_frac=0.05` without `time_mask_max_len` | Spans up to 5% of the trial's frames. Simulated coverage at 200 frames: 39.5% (the 2026-11-03 run: 22%) |
  | `ctc_length_norm=true` | Training CTC loss per trial divided by target length (their `normalize_ctc_len`) |
  | `dataset.sampling=trials` | Shuffled passes over all trials instead of day-balanced batches. Batch loading time unchanged (~0.3 s per batch per worker either way) |
  | `model.transformer.embed_dropout`, `ffn_dropout`, `init=torch` | Their dropout placement and PyTorch-default initialization |
  | `cr_ctc_weight=w` | CR-CTC: each batch run as two independently augmented copies; loss adds w * per-trial sum over frames of 0.5 [KL(sg(b) \|\| a) + KL(sg(a) \|\| b)] (`rnn_trainer.consistency_loss`). Doubles the cost per batch; works for the GRU too |

  - Arm A `tfm384x5_lb_p20000_f80_sm80_noday_v2_63k_s10`: every fix above except CR-CTC, their centred smoothing
    (`smooth_lookahead=4`, 80 ms of future input), no attention look-ahead: the reproduction control. Arm B
    `tfm384x5_lb_p20000_f160_noday_v2_63k_s10`: the same with causal smoothing and R = 2, usable in the live pipeline as is. Arm C
    `tfm384x5_lb_p20000_f160_noday_v2_crctc_32k_s10`: B + `cr_ctc_weight=0.2` on 31.5k batches (same compute). About 100 min each.
  - Still different on purpose: fixed final snapshot / average (theirs: best validation PER); bias on the attention
    projections.
  - Read-out set before training: best-temperature CTC loss (reference 0.442 average / 0.452 final), best acoustic
    scale (reference 0.2), best WER over 0.2-0.5 (reference 9.15; GRU 7.77). Arm A counts as reproduced at a best
    4-gram WER of 7.0 or below (0.81 x A2's 8.51) with the best scale at 0.3 or above; at 8.5 or above the
    next step is their code run unmodified. Keep rule for B and C: average model at least 0.3 below 9.15. The
    fixes are bundled, so a gain is not attributed to any one.
  - Arm A cannot join the deadline ranker until `ctc_features.lookahead` counts smoothing look-ahead (it asserts
    none); the harness benchmark already counts it.
  - Risk noted for arm C: the last confidence regularizer tried here (uniform label smoothing) left PER unchanged and
    cost 2.4 WER. CR-CTC smooths toward the model's own second view, not toward uniform, and its paper reports no
    results with an external LM.
  - Heavy masking has only been tried here on the GRU with its full noise augmentation (white noise 1.0, offset 0.2;
    +5.17 WER, 4.1). Their recipe pairs heavy masking with noise 0.2 / 0.05; that pairing is untested on our GRU.
