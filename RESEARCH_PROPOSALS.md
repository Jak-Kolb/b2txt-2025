# Research Proposals: Causality-Preserving T15 Decoder Improvements

Date: 2026-07-08

Scope: proposal only. No retraining was run, and no model/training files were
modified. Predicted PER effects below are hypotheses against the current causal
single-seed reference, `smooth_lookahead=0`, validation PER 10.04%. Negative
delta PER means lower/better PER. Because this is a single seed, deltas below
about 0.3 absolute PER points should be treated as noise until the baseline seed
spread is measured.

## Ranked Table

| Rank | Proposal | Predicted delta PER | Delta latency / compute | Causal? | Cost to test | Priority |
|---:|---|---|---|---|---|---|
| 1 | Past-only smoother widening: `smooth_kernel_std=2.5`, optionally `3.0` | `2.5`: 0.00 to -0.25 pts, confidence medium-low. `3.0`: -0.10 to +0.20 pts, confidence low | No future latency. Offline/streaming smoothing taps increase from 5 causal taps to 6 or 7 taps; streaming FIR cost is negligible vs GRU | yes | 1 full retrain per std; plan as 1.0x baseline GPU-run each, roughly 10-20 GPU-h/run on 3090-class hardware until measured | Do first |
| 2 | Day-layer regularization screen: lower day LR and/or add small day weight decay | -0.05 to -0.40 pts if high-PER days are day-layer overfit; confidence medium-low | No inference change. Training dynamics only | yes | 2-3 full retrains, same GPU-h/run as baseline | Do first after or alongside smoother |
| 3 | Small regularization/capacity screen, max 6 configs | Best case -0.10 to -0.50 pts; confidence low until seed variance is known | Inference unchanged except capacity variants: `n_units=896` adds compute/params; still well inside latency headroom | yes | 6 screening runs if budget allows; promote only if full-val gain exceeds seed noise | Do maybe |
| 4 | Patch geometry as PER/cadence knob: `patch_stride=2` and/or `patch_size=18` | `stride=2`: 0.00 to -0.35 pts, confidence low. `patch_size=18`: -0.10 to +0.20 pts, confidence low | `stride=2` doubles output frames and GRU steps, RTF likely still far below 1. `patch_size=18` adds about 80 ms algorithmic patch latency and about 4.7M first-layer weights | yes | 1 full retrain per geometry; requires checking output-length formula remains correct | Do maybe |
| 5 | Bidirectional/noncausal teacher distillation into same causal student | -0.20 to -0.80 pts possible, confidence low | No inference latency change if only student is deployed. Training cost high | yes | At least teacher training plus student retrain; likely 2-4x baseline GPU budget | Do later |
| 6 | Delay-penalized CTC / FastEmit-style emission regularization | PER likely -0.10 to +0.30 pts; confidence low. Mostly an emission-latency objective, not a PER lever | No inference compute change; may shift CTC emissions earlier. Latency is already solved here | yes | New loss implementation plus retrain; >1x baseline and higher engineering risk | Skip for now |
| 7 | RNN-T or streaming chunked/attention architecture | Unknown; could improve with enough engineering, but not predictable from this code | RNN-T adds prediction/joint network and decoder complexity; chunked attention must use zero right context to remain strictly causal | yes, for causal encoder / zero-right-context variants | Multi-week rewrite plus retraining | Future only |

## Code Facts Used

- Smoothing is centralized in `model_training/data_augmentations.py:6-43`.
  With `lookahead=L`, the code keeps offsets `-p..+L`, renormalizes, pads
  `(p, L)`, and uses grouped `conv1d`; `smooth_lookahead=0` is fully causal.
- Training calls the smoother through `model_training/rnn_trainer.py:473-482`.
  Evaluation uses the saved model args in
  `model_training/evaluate_model_helpers.py:83-101`, so train/eval smoother
  consistency is structural.
- The GRU is unidirectional: `bidirectional=False` in
  `model_training/rnn_model.py:65-72`.
- The day layer is a per-timestep affine plus `Softsign` at
  `model_training/rnn_model.py:94-103`; it does not mix time.
- Current patching uses `unfold` in `model_training/rnn_model.py:105-119`.
  With right-edge emission, patch output `i` depends on a completed input window,
  not on future bins.
- Current CTC input length is
  `floor((n_time_steps - patch_size) / patch_stride) + 1`, implemented by the
  config-driven formula at `model_training/rnn_trainer.py:533` and `:708`.
- The day affine has about `45 * (512 * 512 + 512) = 11,819,520` parameters,
  about 27% of the approximately 44.3M parameter model. The saved split has
  8,072 training trials across 45 days; per-day training counts range from 59 to
  364, so low-data days can repeatedly update a large day-specific matrix.
- The released data description says neural features are 20 ms bins and are
  normalized from the preceding 20 trials, which is compatible with real-time use
  if reproduced online with only prior trials.

## 1. Past-Only Smoother Widening

Recommendation: first test `smooth_kernel_std: 2.5`, then test `3.0` only if
`2.5` is neutral or better. Do not change `smooth_lookahead`; it stays `0`.

Causal: yes — with `smooth_lookahead=0`, output[t] is a normalized weighted sum
of raw inputs from offsets `[-p, 0]`, so it depends only on inputs at or before t.

### Kernel Analysis

The current implementation trims Gaussian taps whose pre-trim impulse response is
`<= 0.01`, then renormalizes. Using the same discrete Gaussian rule as
`scipy.ndimage.gaussian_filter1d`, the one-sided causal kernels are:

| `smooth_kernel_std` | retained full offsets | causal offsets | mean past lag | peak weight | mass at lag >=3 | mass at lag >=4 | mass at lag >=5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 current | `-4..4` | `-4..0` | 1.224 bins / 24.5 ms | 0.339 | 0.156 | 0.046 | 0.000 |
| 2.5 | `-5..5` | `-5..0` | 1.585 bins / 31.7 ms | 0.282 | 0.254 | 0.117 | 0.038 |
| 3.0 | `-6..6` | `-6..0` | 1.945 bins / 38.9 ms | 0.241 | 0.338 | 0.192 | 0.093 |

Interpretation: `2.5` adds one additional past bin and shifts about 10% more
mass into lags >=3, while still keeping the current bin as the largest tap.
`3.0` is a stronger denoiser but lowers the current-bin weight enough that
phoneme boundary smearing becomes the main risk. Because the current causal
single seed is already slightly better than the symmetric single seed
(`10.04%` vs `10.21%`), any observed improvement below 0.3 points should be
treated as provisional.

### Ready-To-Apply Config Diffs

Run these as separate experiment copies with unique output directories.

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-output_dir: trained_models/baseline_rnn # directory to save the trained model and logs
-checkpoint_dir: trained_models/baseline_rnn/checkpoint # directory to save checkpoints during training
+output_dir: trained_models/causal_std25 # directory to save the trained model and logs
+checkpoint_dir: trained_models/causal_std25/checkpoint # directory to save checkpoints during training
@@
-    smooth_kernel_std: 2 # standard deviation of the smoothing kernel applied to the data
+    smooth_kernel_std: 2.5 # wider past-only causal smoother candidate
     smooth_lookahead: 0    # future taps; 0 = causal/real-time, 4 = symmetric baseline
```

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-output_dir: trained_models/baseline_rnn # directory to save the trained model and logs
-checkpoint_dir: trained_models/baseline_rnn/checkpoint # directory to save checkpoints during training
+output_dir: trained_models/causal_std30 # directory to save the trained model and logs
+checkpoint_dir: trained_models/causal_std30/checkpoint # directory to save checkpoints during training
@@
-    smooth_kernel_std: 2 # standard deviation of the smoothing kernel applied to the data
+    smooth_kernel_std: 3.0 # stronger past-only causal smoother candidate
     smooth_lookahead: 0    # future taps; 0 = causal/real-time, 4 = symmetric baseline
```

Optional longer-tail code hook, only if `std=2.5/3.0` helps and the human wants a
cleaner way to sweep the trim threshold. The default preserves existing behavior.

```diff
diff --git a/model_training/data_augmentations.py b/model_training/data_augmentations.py
--- a/model_training/data_augmentations.py
+++ b/model_training/data_augmentations.py
@@
-def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,
-                 padding='same', lookahead=None):
+def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,
+                 padding='same', lookahead=None, smooth_kernel_min_weight=0.01):
@@
-    validIdx = np.argwhere(gaussKernel > 0.01)
+    validIdx = np.argwhere(gaussKernel > smooth_kernel_min_weight)
diff --git a/model_training/rnn_trainer.py b/model_training/rnn_trainer.py
--- a/model_training/rnn_trainer.py
+++ b/model_training/rnn_trainer.py
@@
                 smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                 smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                 lookahead = self.transform_args['smooth_lookahead'],
+                smooth_kernel_min_weight = self.transform_args.get('smooth_kernel_min_weight', 0.01),
                 )
diff --git a/model_training/evaluate_model_helpers.py b/model_training/evaluate_model_helpers.py
--- a/model_training/evaluate_model_helpers.py
+++ b/model_training/evaluate_model_helpers.py
@@
             smooth_kernel_std = model_args['dataset']['data_transforms']['smooth_kernel_std'],
             smooth_kernel_size = model_args['dataset']['data_transforms']['smooth_kernel_size'],
             lookahead = model_args['dataset']['data_transforms']['smooth_lookahead'],
+            smooth_kernel_min_weight = model_args['dataset']['data_transforms'].get('smooth_kernel_min_weight', 0.01),
             # eval reads lookahead from saved model_args -> train/eval consistency is structural
         )
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
     smooth_kernel_size: 100 # size of the smoothing kernel applied to the data
+    smooth_kernel_min_weight: 0.01 # trim threshold; default preserves current smoother
     smooth_data: true # whether to smooth the data
```

Predicted effect: `std=2.5` is the best PER-per-GPU-hour candidate because it is
config-only and specifically targets causal denoising. `std=3.0` is worth one run
only if `2.5` is not clearly worse.

## 2. Day-Layer Behavior

Recommendation: test a small day-layer screen before broad model tuning:

1. `lr_max_day: 0.0025`, `lr_min_day: 0.00005`, `weight_decay_day: 0`.
2. `lr_max_day: 0.005`, `lr_min_day: 0.0001`, `weight_decay_day: 0.0001`.
3. `lr_max_day: 0.0025`, `lr_min_day: 0.00005`, `weight_decay_day: 0.0001`.

Causal: yes — the day layer is a fixed per-session affine applied independently
to each timestep, so output[t] depends only on input[t] and learned day
parameters.

Reasoning: the current day layer has a full 512x512 matrix plus bias for each
day, initialized to identity and trained with the same peak LR as the main model
and zero day weight decay. This is a large number of session-specific parameters
for low-data sessions. The stated per-day PER spread, combined with training
counts ranging from 59 to 364 trials/day, makes day-layer overfit or unstable
adaptation more plausible than a pure global-capacity issue.

`days_per_batch` is less clearly a first knob. With `batch_size=64` and uniform
day sampling, the expected examples per day per training batch are approximately
`64 / 45` regardless of `days_per_batch`; changing it mostly changes update
frequency and per-update variance. Keep `days_per_batch: 4` for the first three
runs. If day-layer regularization helps but high-PER low-data days remain
unstable, then try `days_per_batch: 2` with the best LR/decay setting to give
each selected day more examples per update.

### Ready-To-Apply Config Diff

Run one hunk at a time with unique directories.

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-output_dir: trained_models/baseline_rnn # directory to save the trained model and logs
-checkpoint_dir: trained_models/baseline_rnn/checkpoint # directory to save checkpoints during training
+output_dir: trained_models/causal_day_lr0025 # directory to save the trained model and logs
+checkpoint_dir: trained_models/causal_day_lr0025/checkpoint # directory to save checkpoints during training
@@
-lr_max_day: 0.005 # maximum learning rate for the day specific input layers
-lr_min_day: 0.0001 # minimum learning rate for the day specific input layers
+lr_max_day: 0.0025 # lower day-layer LR candidate
+lr_min_day: 0.00005 # keep same min/max ratio for day-layer schedule
```

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-output_dir: trained_models/baseline_rnn # directory to save the trained model and logs
-checkpoint_dir: trained_models/baseline_rnn/checkpoint # directory to save checkpoints during training
+output_dir: trained_models/causal_day_wd1e4 # directory to save the trained model and logs
+checkpoint_dir: trained_models/causal_day_wd1e4/checkpoint # directory to save checkpoints during training
@@
-weight_decay_day: 0 # weight decay for the day specific input layers
+weight_decay_day: 0.0001 # light regularization for per-day affine matrices
```

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-output_dir: trained_models/baseline_rnn # directory to save the trained model and logs
-checkpoint_dir: trained_models/baseline_rnn/checkpoint # directory to save checkpoints during training
+output_dir: trained_models/causal_day_lr0025_wd1e4 # directory to save the trained model and logs
+checkpoint_dir: trained_models/causal_day_lr0025_wd1e4/checkpoint # directory to save checkpoints during training
@@
-lr_max_day: 0.005 # maximum learning rate for the day specific input layers
-lr_min_day: 0.0001 # minimum learning rate for the day specific input layers
+lr_max_day: 0.0025 # lower day-layer LR candidate
+lr_min_day: 0.00005 # keep same min/max ratio for day-layer schedule
@@
-weight_decay_day: 0 # weight decay for the day specific input layers
+weight_decay_day: 0.0001 # light regularization for per-day affine matrices
```

Predicted effect: if the worst sessions are dominated by day-affine overfit, this
could produce a real aggregate gain. If the spread is driven by neural signal
quality, corpus mix, or validation denominator noise, expect no reliable
aggregate change.

## 3. Model Capacity / Regularization Screen

Recommendation: do not run a broad sweep. Use at most these six configs, all
with `smooth_lookahead: 0`, unique directories, and the same validation gate.
Promote only configs whose full-validation PER improves by more than the measured
seed spread or at least 0.3 absolute points.

Causal: yes — dropout, white-noise augmentation, layer count, and hidden width
change training or per-timestep recurrent state updates only; the deployed
unidirectional GRU output[t] still depends only on inputs up to t.

| Config | Change | Hypothesis | Predicted delta PER |
|---|---|---|---|
| R1 | `rnn_dropout: 0.3` | Current `0.4` may be high for a strong denoised causal input | 0.00 to -0.25, low |
| R2 | `rnn_dropout: 0.5` | If low-data days overfit globally, more recurrent dropout helps | -0.10 to +0.30, low |
| R3 | `input_layer_dropout: 0.1` | Day layer may need more channel stability after causal smoothing | 0.00 to -0.25, low |
| R4 | `input_layer_dropout: 0.3` | If day matrices memorize noisy channels, stronger dropout helps | -0.10 to +0.30, low |
| R5 | `white_noise_std: 0.5` | Current white noise may be too strong relative to already noisy neural features | 0.00 to -0.30, low |
| R6 | `n_units: 896` | Latency headroom allows a capacity probe for hard sessions | -0.10 to -0.40, low |

I would not test `n_layers > 5` first. Deeper GRUs increase optimization risk and
do not directly address the observed day variance. I would also avoid compact
models in this project phase because latency is already solved and compactness is
not a PER lever.

### Ready-To-Apply Config Diffs

Use the following single-field substitutions per run.

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-  rnn_dropout: 0.4 # dropout rate for the GRU layers
+  rnn_dropout: 0.3 # regularization screen R1
```

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-  rnn_dropout: 0.4 # dropout rate for the GRU layers
+  rnn_dropout: 0.5 # regularization screen R2
```

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-    input_layer_dropout: 0.2 # dropout rate for the input layer
+    input_layer_dropout: 0.1 # regularization screen R3
```

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-    input_layer_dropout: 0.2 # dropout rate for the input layer
+    input_layer_dropout: 0.3 # regularization screen R4
```

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-    white_noise_std: 1.0 # standard deviation of the white noise added to the data
+    white_noise_std: 0.5 # augmentation screen R5
```

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-  n_units: 768 # number of units per GRU layer
+  n_units: 896 # capacity screen R6
```

Each run also needs a unique `output_dir` and `checkpoint_dir`; I did not repeat
that hunk six times.

## 4. Patch Geometry as Accuracy / Cadence Knob

Recommendation: only test patch geometry after smoother and day-layer screens,
unless the human specifically wants finer output cadence. Do not adopt
left-padded patching as an accuracy proposal; it is a cold-start/latency change.

Causal: yes — with right-edge emission, each patch output is emitted only after
the patch window has been observed, so output[t] depends only on completed input
bins at or before t.

Candidate configs:

1. `patch_size: 14`, `patch_stride: 2`: same 280 ms patch context, 40 ms output
   cadence. This doubles GRU time steps and CTC alignment opportunities. PER
   might improve if the current 80 ms cadence is too coarse for phoneme timing.
2. `patch_size: 18`, `patch_stride: 4`: 360 ms context, same 80 ms output
   cadence. This adds past context but increases first-layer input width from
   `14*512` to `18*512`, adding about 4.7M first-layer GRU weights. It may help
   slow articulatory context but can smear transitions.
3. `patch_size: 10`, `patch_stride: 2`: 200 ms context, 40 ms cadence. This is a
   fallback if stride 2 helps but `patch_size=14` over-contextualizes or costs
   too much.

The current formula remains correct for non-left-padded patch geometry because
it is already config driven:

```python
adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
```

The equivalent exact formula is
`floor((T - patch_size) / patch_stride) + 1` for positive lengths. Confirm both
training and validation copies at `rnn_trainer.py:533` and `:708`.

If and only if the human separately chooses left-padded patching for cold-start
latency, the formula must change to `floor((T - 1) / patch_stride) + 1`, but that
is not recommended here because latency is already solved and the task forbids
presenting left-padding as a latency win.

### Ready-To-Apply Config Diffs

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-output_dir: trained_models/baseline_rnn # directory to save the trained model and logs
-checkpoint_dir: trained_models/baseline_rnn/checkpoint # directory to save checkpoints during training
+output_dir: trained_models/causal_stride2 # directory to save the trained model and logs
+checkpoint_dir: trained_models/causal_stride2/checkpoint # directory to save checkpoints during training
@@
-  patch_stride: 4 # stride for the input patches (4 time steps)
+  patch_stride: 2 # cadence/PER screen: 40 ms output cadence
```

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-output_dir: trained_models/baseline_rnn # directory to save the trained model and logs
-checkpoint_dir: trained_models/baseline_rnn/checkpoint # directory to save checkpoints during training
+output_dir: trained_models/causal_patch18 # directory to save the trained model and logs
+checkpoint_dir: trained_models/causal_patch18/checkpoint # directory to save checkpoints during training
@@
-  patch_size: 14 # size of the input patches (14 time steps)
+  patch_size: 18 # context/PER screen: 360 ms right-edge patch
```

```diff
diff --git a/model_training/rnn_args.yaml b/model_training/rnn_args.yaml
--- a/model_training/rnn_args.yaml
+++ b/model_training/rnn_args.yaml
@@
-output_dir: trained_models/baseline_rnn # directory to save the trained model and logs
-checkpoint_dir: trained_models/baseline_rnn/checkpoint # directory to save checkpoints during training
+output_dir: trained_models/causal_patch10_stride2 # directory to save the trained model and logs
+checkpoint_dir: trained_models/causal_patch10_stride2/checkpoint # directory to save checkpoints during training
@@
-  patch_size: 14 # size of the input patches (14 time steps)
-  patch_stride: 4 # stride for the input patches (4 time steps)
+  patch_size: 10 # shorter context, finer cadence screen
+  patch_stride: 2 # 40 ms output cadence
```

## 5. Literature-Informed Future Ideas

These are causality-compatible but are not first-line recommendations for this
codebase.

### 5.1 Noncausal Teacher Distillation Into Causal Student

Causal: yes — the teacher may use future context during training only, but the
deployed student remains the same causal GRU and output[t] uses only student
state from inputs <= t.

Applicability: streaming ASR literature uses non-streaming teachers to improve
streaming students; Shim et al. propose encoder distillation from a non-streaming
teacher to a streaming ASR encoder. This maps conceptually to training a stronger
offline T15 teacher and distilling logits or hidden states into the existing
causal GRU. It fits the causality rule only if the teacher is never used at
inference.

Engineering fit: medium. Add a teacher checkpoint loader and a KL or
representation loss in `rnn_trainer.py`. Hard parts are alignment mismatch
between teacher and student frames and not letting teacher-only future context
creep into runtime features.

### 5.2 Delay-Penalized CTC

Causal: yes — it changes the training objective over the same causal encoder
outputs; inference still consumes only causal logits.

Applicability: Yao et al. propose delay-penalized CTC with a differentiable FST
implementation to encourage lower-delay CTC alignments. This is more directly
relevant than FastEmit because the current code uses CTC, not RNN-T. However, the
objective is primarily about emission timing; since latency is already solved and
PyTorch `CTCLoss` is currently simple, this is a lower PER-per-engineering-hour
candidate.

Engineering fit: medium-low. It likely requires replacing or augmenting
`torch.nn.CTCLoss` with an FST-based differentiable loss or a carefully verified
approximation.

### 5.3 FastEmit / RNN-T Emission Regularization

Causal: yes — FastEmit-style regularization can be used with a streaming
transducer whose encoder is causal, so output[t] uses only encoder frames <= t
and previously emitted labels.

Applicability: FastEmit is designed for sequence transducer models and applies
latency regularization directly to transducer forward-backward probabilities.
It is not directly applicable to the current CTC head. Treat it as evidence that
emission regularization can help streaming ASR, not as a patch for this repo.

Engineering fit: low for current code. It becomes relevant only if the project
moves to RNN-T.

### 5.4 Streaming / Chunked Training Tricks

Causal: yes — restricted to zero-future/right-context chunking, output[t]
depends only on past cached state and current-or-earlier inputs; any positive
right context is disqualified for the strict real-time default.

Applicability: Emformer and dynamic chunk convolution papers show how streaming
ASR models train with limited context and cached memory. The current architecture
is already a unidirectional GRU with cached state, so these ideas mostly matter
for a future Transformer/Conformer rewrite. They do not offer a small safe patch
to the existing GRU.

Engineering fit: low now, higher only for a new architecture track.

### 5.5 RNN-T Future Direction

Causal: yes — an RNN-T with a causal acoustic encoder and prediction network over
previous labels can produce streaming outputs without future neural input.

Applicability: Graves' RNN Transducer is a natural streaming alternative to CTC
because it models output history inside the neural decoder instead of relying on
only frame-independent CTC logits plus external sentence-final language-model
rescoring. It may eventually reduce substitutions or deletions that require label
history. It is not a scoped improvement here because it changes the head, loss,
decoder, and evaluation path.

Engineering fit: low for this phase. Save for a future branch after the causal
CTC frontier is better characterized.

## Do First / Do Maybe / Skip

### Do First

1. Establish baseline seed variance or run promoted candidates with at least
   matching seeds. The 10.04% result is a single seed; without a seed spread,
   any single-run difference under about 0.3 points is not actionable.
2. Test `smooth_kernel_std: 2.5` with `smooth_lookahead: 0`. This is the most
   direct causal denoising lever and has the best expected PER-per-GPU-hour.
3. Test one day-layer regularization run, preferably
   `lr_max_day: 0.0025`, `lr_min_day: 0.00005`, `weight_decay_day: 0.0001`.
   If it helps aggregate PER or narrows high-PER sessions, test the other
   day-layer variants.

### Do Maybe

1. Test `smooth_kernel_std: 3.0` if `2.5` is neutral or better.
2. Run the six-config regularization/capacity screen only after seed variance is
   known. The predicted effects are mostly within single-seed noise.
3. Try `patch_stride: 2` if CTC alignment density looks like a bottleneck. The
   compute increase is acceptable, but the PER prediction is weak.
4. Plan teacher distillation only after cheaper config-level levers are exhausted.

### Skip

1. Do not set `smooth_lookahead > 0` as a default improvement. It is a deliberate
   latency/PER tradeoff for the human to sweep, not a causal real-time default.
2. Do not add bidirectional RNNs, noncausal convolutions, or attention with future
   receptive field to the deployed model.
3. Do not present left-padded patching as an accuracy or latency improvement in
   this phase. Current patching is already causal under right-edge emission, and
   latency has 76x headroom.
4. Do not add whole-trial, whole-session, or future-statistic normalization at
   inference. The released data's preceding-20-trial normalization is acceptable
   only if the online implementation also uses preceding trials only.
5. Skip delay-penalized CTC and RNN-T until config-level causal PER is bounded;
   they are real research tracks, not small follow-up diffs.

## Considered And Rejected For Causality

- Bidirectional GRU/LSTM layers in the deployed acoustic model: rejected because
  hidden state at output[t] would depend on future neural bins.
- Symmetric Gaussian smoothing or `smooth_lookahead > 0` as a default: rejected
  because smoothing output[t] would use raw inputs after t. It can remain an
  explicit latency tradeoff sweep only.
- Centered or noncausal temporal convolutions: rejected unless the kernel is
  strictly one-sided past-to-current.
- Attention over a full trial or chunk attention with positive right context:
  rejected because keys/values after t influence output[t].
- Per-trial or whole-session normalization at inference: rejected because the
  mean/std would include future bins or future trials.
- Offline LM/LLM rescoring as a streaming acoustic improvement: not part of the
  acoustic real-time decoder. Sentence-final rescoring can be product-valid, but
  it must not be counted as per-frame real-time acoustic output.
- Left-padded patching as a latency win: rejected for this project phase. It is
  causal, but it is a cold-start/algorithmic-latency change, not a PER-first
  improvement.

## Sources

- Local code: `model_training/data_augmentations.py`,
  `model_training/rnn_model.py`, `model_training/rnn_trainer.py`,
  `model_training/evaluate_model_helpers.py`, `model_training/rnn_args.yaml`,
  `results/baseline_rnn/train_val_trials.json`.
- Card et al. T15 data description, including 20 ms bins and preceding-20-trial
  normalization: [Dryad dataset](https://datadryad.org/dataset/doi%3A10.5061/dryad.dncjsxm85).
- Card et al. NEJM dataset citation: [NEJM DOI](https://doi.org/10.1056/nejmoa2314132).
- Willett et al. 2023 speech neuroprosthesis background:
  [Nature DOI](https://doi.org/10.1038/s41586-023-06377-x) and
  [WashU profile page](https://profiles.wustl.edu/en/publications/a-high-performance-speech-neuroprosthesis/).
- CTC original paper: Graves et al., 2006,
  [PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf).
- RNN-T original paper: Graves, 2012,
  [arXiv:1211.3711](https://arxiv.org/abs/1211.3711).
- FastEmit: Yu et al., 2021,
  [arXiv:2010.11148](https://arxiv.org/abs/2010.11148).
- Delay-penalized CTC: Yao et al., Interspeech 2023,
  [ISCA PDF](https://www.isca-archive.org/interspeech_2023/yao23b_interspeech.pdf).
- Non-streaming to streaming ASR distillation: Shim et al., 2023,
  [arXiv:2308.16415](https://arxiv.org/abs/2308.16415).
- Emformer streaming ASR: Shi et al., 2021,
  [arXiv:2010.10759](https://arxiv.org/abs/2010.10759).
- Dynamic chunk convolution: Li et al., 2023,
  [arXiv:2304.09325](https://arxiv.org/abs/2304.09325).
