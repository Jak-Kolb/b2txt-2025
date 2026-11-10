# Model and Dataset Design: B2T '25 Real-Time Speech Decoder

How the Brain-to-Text '25 dataset is organized, and how the reference real-time pipeline turns
intracortical activity into text. All numbers are traced to code, configs or recorded runs in this
repository; results are on **val-dev** and are **exploratory** (see §7.4).

Written 2026-10-07. Reference model: `causal_la0` (`model_training/rnn_args.yaml`,
`harness/registry/acoustic/causal_la0.yaml`).

---

## 1. The pipeline at a glance

```
 4 microelectrode arrays (256 electrodes)
        │  20 ms bins, 512 features (threshold crossings + spike-band power)
        ▼
 ┌──────────────────────────── neural network ("acoustic model") ───────────────────────────┐
 │ 1. Causal Gaussian smoothing     past 4 bins + current, per channel                       │
 │ 2. Day layer                     per-session 512×512 linear map + softsign                │
 │ 3. Patching                      14 bins (280 ms) stacked → 7,168 numbers, every 4 bins   │
 │ 4. 5-layer GRU, 768 units        one 768-vector per 80 ms frame                            │
 │ 5. Output head                   768 → 41 class probabilities (blank, 39 phonemes, "|")   │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
        │  one probability distribution per 80 ms frame
        ▼
 ┌──────────────────────────── language-model search (live) ────────────────────────────────┐
 │ WFST beam search over T (CTC rules) ∘ L (pronunciation dictionary) ∘ G (4-gram LM)        │
 │ Dials: acoustic scale 0.4, blank penalty 90                                              │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
        │  partial transcript updated every 80 ms (may revise earlier words)
        ▼
 Optional, after the sentence ends: GPT-2 Large re-ranks the search's n-best list
```

| Stage | Reference result (val-dev, 1,253 sentences) |
|---|---|
| Network alone, greedy phonemes | 8.85% phoneme error rate |
| Network + 4-gram search (live) | **8.51% WER** [95% CI 6.62–10.65] |
| + GPT-2 re-ranking at sentence end | **6.87% WER** [5.09–8.92] |

---

## 2. The dataset

### 2.1 Participant and recording

- **Source:** Card et al., *"An Accurate and Rapidly Calibrating Speech Neuroprosthesis"*,
  NEJM 2024; released as the Brain-to-Text '25 benchmark (`README.md`).
- **Participant:** one person (T15) with ALS, implanted with **four 64-electrode microelectrode
  arrays** in the left speech motor cortex: ventral 6v, area 4, 55b and dorsal 6v.
- **Timeline:** 45 recording sessions from 2023-08-11 (post-implant day 25) to 2025-04-13
  (day 636), about 20 months. One session is one recording day.
- **Task:** "copy task": a sentence appears on screen and the participant attempts to say it after
  a cue. Most blocks are attempted vocalized speech, some attempted silent speech; the strategy per
  block is intentionally not released.

### 2.2 Features: what the network actually sees

The release contains **processed features**, not raw voltage.

| Property | Value |
|---|---|
| Bin size | 20 ms (50 bins per second) |
| Features per bin | 512 = 2 per electrode × 256 electrodes |
| Feature 1, channels 0–255 | **Threshold crossings (TX):** count of voltage dips below −4.5 × RMS in the bin, a proxy for nearby spikes |
| Feature 2, channels 256–511 | **Spike-band power (SBP):** energy in the high-frequency band where spikes live; a smoother measure of the same activity |
| Channel order | ventral 6v, area 4, 55b, dorsal 6v (64 each), first all TX then all SBP |
| Normalization | z-scored per recording block by the dataset authors |
| Observed range | TX takes a few discrete levels per channel (scaled counts); both features appear capped at 10 |

Because both features are z-scored, they are on comparable scales and zero is each channel's
mean. TX and SBP from the same electrode are strongly correlated (median within-electrode
correlation 0.77 in one example trial).

### 2.3 Splits

| Split | Sessions | Trials | Hours | Median trial | Labels |
|---|---|---|---|---|---|
| train | 45 | 8,072 | 39.2 | 16.7 s, 6 words, 26 phonemes | sentence + phoneme sequence |
| val | 41 | 1,426 | 7.3 | 17.8 s | sentence + phoneme sequence |
| test | 41 | 1,450 | 7.4 | 17.9 s | **none** (`input_features` only; competition-held) |

- Splits are by **block**: each block of sentences goes entirely to train or to val/test.
- Four sessions are train-only: 2023-08-11, 2024-03-03, 2024-04-25, 2024-04-28.
- Every val/test trial comes from a session that also has training data, so every evaluated trial
  has a trained day layer (§3.3).

### 2.4 Sentence sources

Each block draws sentences from one corpus (`data/t15_copyTaskData_description.csv`). The mix is
very uneven across days, which matters for evaluation (§7.3).

| Corpus | Blocks (train / val-test) | Character | la0 WER on val-dev |
|---|---|---|---|
| Switchboard | 135 / 47 | conversational English | 6.9% |
| Freq words | 38 / 10 | sentences of common words | 3.5% |
| 50-Word | 25 / 0 | closed 50-word vocabulary (early sessions) | — |
| OpenWebText | 0 / 5 | web text | 12.9% |
| Harvard | 0 / 3 | phonetically balanced test sentences | 28.6% |
| Random | 0 / 2 | random word strings (no grammar to exploit) | 38.5% |

### 2.5 Labels

Each labeled trial has its sentence and a phoneme sequence over **39 phonemes + a word-boundary
token `|`**, with `|` after every word. The network adds a CTC **blank**, giving **41 classes**
(`model_training/evaluate_model_helpers.py:9`):

```
"Which is most unfortunate ..."  →  W IH CH | IH Z | M OW S T | AH N F AO R CH AH N AH T | ...
class order:  0 = BLANK, 1–39 = AA … ZH, 40 = "|"
```

The labels say *which* phonemes occurred, not *when*. There is no per-word or per-phoneme timing,
which is why CTC is used (§4.1) and why word latency cannot be measured (§7.5).

### 2.6 The project's evaluation split

The public validation trials were split **by session** in `model_training/splits.py`:

| Name | Sessions | Trials | Words | Intended use |
|---|---|---|---|---|
| **val-dev** | 35 earlier sessions | 1,253 | 8,277 | develop and tune |
| **former val-test** | 6 most recent (2025-01-10 … 2025-04-13) | 173 | | look once, at the end |

The former val-test is **exposed**: the training code selects checkpoints using all validation
trials, including these (§4.5). It is not an independent holdout.

---

## 3. The neural network

### 3.1 Causal smoothing

Each channel is low-pass filtered with a Gaussian kernel (std 2 bins) truncated to the past
(`model_training/data_augmentations.py:12-60`, `smooth_lookahead: 0`):

```
smoothed[t] = 0.339·x[t] + 0.299·x[t−1] + 0.206·x[t−2] + 0.110·x[t−3] + 0.046·x[t−4]
```

- **Per channel, identical weights.** The filter is linear with weights summing to 1, so scale and
  offset differences between TX and SBP pass straight through.
- **What it does:** turns spiky per-bin TX counts into a firing-rate estimate and removes jitter
  from SBP. Gain: 0.96 at 2 Hz, 0.76 at 5 Hz, 0.35 at 10 Hz.
- **Cost:** no look-ahead, but the average lag of the output is about 24 ms.
- **Contrast:** the authors' released model uses the symmetric 9-tap kernel (offsets −4…+4), which
  needs 80 ms of future data and smooths more heavily (gain 0.02 at 10 Hz).
- **Streaming:** the same weights run as a circular-buffer FIR filter, one matrix multiply per bin
  (`model_training/benchmark/streaming_infer.py:86-115`). Training, offline evaluation and
  streaming use identical weights; a test checks this (`tests/test_harness_metrics.py:70`).

### 3.2 Augmentation (training only, before smoothing)

From `rnn_args.yaml` and `rnn_trainer.py:430-466`:

| Augmentation | Setting | Purpose |
|---|---|---|
| White noise | std 1.0, independent per bin and channel | robustness to noise |
| Constant offset | std 0.2, one draw per trial and channel | robustness to baseline drift |
| Random cut | drop 0–2 bins from the start of the batch | robustness to alignment |

Static gain and random-walk noise exist in the code but are off.

### 3.3 Day layers

Neural recordings drift from day to day: electrodes shift, neurons appear and disappear, and
baselines wander. Each session therefore has its own input transform:

```
x' = softsign(W_day · x + b_day)      W_day: 512×512 (initialized to identity), b_day: 512
```

- 45 day layers, one per session: 262,656 parameters each, 11.8M in total.
- At training and inference, each trial uses its own session's layer; the model is told which
  session a trial comes from.
- Dropout 0.2 is applied after the day layer during training.
- **Implication:** results measure accuracy **with a calibrated day layer**. A brand-new day has
  no trained layer; the clinical system recalibrated with ~7.5 minutes of prompted sentences on
  request and fine-tuned on confirmed conversational sentences (Card et al. 2024). New-day
  performance is not measured in this repo.

### 3.4 Patching

After the day layer, 14 consecutive bins are concatenated into one vector, and the window moves
4 bins at a time (`rnn_model.py:104-117`):

```
smoothed bins:  |b0|b1|...|b13|b14|b15|b16|b17|...
frame 0:        [b0 ......... b13]
frame 1:                [b4 ......... b17]
frame 2:                        [b8 ......... b21]
```

| Property | Value |
|---|---|
| Patch | 14 bins × 512 = **7,168 numbers** (concatenated, not averaged) |
| Frame rate | every 4 bins = **80 ms** (12.5 Hz) |
| Raw data behind one frame | 18 bins = 360 ms (280 ms patch + 80 ms smoothing reach) |
| Frames per trial | (bins − 14) // 4 + 1, e.g. 910 bins → 225 frames |

Patching makes the recurrent layers run 4× less often and lets the first layer learn its own
temporal filters across the 280 ms window. It also fixes the update cadence and makes the first
output possible only 280 ms into a trial.

### 3.5 GRU

Five stacked unidirectional GRU layers, 768 units each (`rnn_model.py:63-85`).

Per layer and frame, with input x and previous hidden state h:

```
r  = σ(W_ir·x + b_ir + W_hr·h + b_hr)            reset gate
z  = σ(W_iz·x + b_iz + W_hz·h + b_hz)            update gate
n  = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn))   candidate
h' = (1 − z) ⊙ n + z ⊙ h                         new hidden state = the layer's output
```

- A GRU's output **is** its hidden state: the same 768-vector goes up to the next layer and
  forward to the next frame.
- Learned initial hidden state `h0`; orthogonal init for recurrent weights, Xavier for input weights.
- Dropout 0.4 between layers during training.
- Strictly causal: frame t depends only on frames ≤ t.

### 3.6 Output head

A linear layer maps the top GRU layer's 768-vector to 41 scores; softmax turns them into a
probability distribution per frame. **The network outputs distributions, not a phoneme string.**
The decoder consumes the full distribution. Example from val-dev trial 3, frame 132:
`CH 0.56, JH 0.44`. Keeping both lets the search pick "just" (JH AH S T) even though CH was
slightly more likely.

### 3.7 Shapes and parameters

| Component | Input → output per frame | Calculation | Parameters |
|---|---|---|---|
| Day layers (45) | 512 → 512 | 45 × (512² + 512) | 11,819,520 |
| GRU layer 0 | 7,168 → 768 | 3 × (768·7168 + 768² + 2·768) | 18,289,152 |
| GRU layers 1–4 | 768 → 768 | 4 × 3 × (768·768 + 768² + 2·768) | 14,174,208 |
| Output head | 768 → 41 | 768·41 + 41 | 31,529 |
| `h0` | | 768 | 768 |
| **Shared total** | | | **32,495,657** |
| **Grand total** | | | **44,315,177** |

- The factor 3 is the three gates; each has an input matrix, a recurrent matrix and two biases.
  PyTorch stacks them (`weight_ih_l0` is 2,304 × 7,168).
- Layer 0's input matrix alone (16.5M) is about half of the shared parameters, because patching
  makes its input 7,168 wide.
- One forward pass uses 32.5M shared + 0.26M for the trial's day layer.

---

## 4. Training

### 4.1 Objective: CTC loss

The labels have no timing, so the loss must not depend on which frame each phoneme falls on.
**Connectionist Temporal Classification (CTC)** handles this:

- An **alignment** assigns one class per frame. It **collapses** to a phoneme sequence by merging
  repeats, then removing blanks: `A A B _ B C C D A _ B → A B B C D A B`. A blank is required
  between repeated phonemes.
- **Loss** = −log P(target | input), where P sums the probabilities of **every** alignment that
  collapses to the target; each alignment's probability is the product of its per-frame
  probabilities.
- The sum is computed exactly by a dynamic-programming forward algorithm in
  O(frames × target length), not by enumeration (41^frames paths).
- A target needs at least (length + adjacent repeats) frames; the trainer counts infeasible
  targets (`rnn_trainer.py:546`).

### 4.2 Optimization

| Setting | Value |
|---|---|
| Batch | 64 trials drawn from **4 random sessions** |
| Steps | 120,000 batches (≈7.7M trial presentations) |
| Optimizer | AdamW, β = (0.9, 0.999), ε = 0.1 |
| Learning rate | cosine 0.005 → 0.0001, 1,000 warmup steps (same for day layers) |
| Weight decay | 0.001 on shared weights; 0 on biases and day layers |
| Gradient clipping | global norm 10 |
| Precision | bfloat16 autocast (no gradient scaler, so no skipped steps) |

### 4.3 What one step updates

Backpropagation computes every parameter's gradient in one backward pass (chain rule; about 2–3×
the cost of the forward pass). AdamW then moves each parameter against its gradient.

| Group | Parameters | Updated each step? |
|---|---|---|
| Shared (GRU, head, `h0`) | 32,495,657 | all |
| Day layers of the 4 sessions in the batch | 1,050,624 | all |
| Day layers of the other 41 sessions | 10,768,896 | none (no gradient, skipped by AdamW) |

### 4.4 Validation during training

Every 2,000 batches (60 times per run), the model is scored on all validation trials with greedy
phoneme decoding. The snapshot with the lowest phoneme error rate is kept (ties broken by loss;
`rnn_trainer.py:662`). Newer runs also save late snapshots so selection can be redone.

### 4.5 Consequence for evaluation

Checkpoint selection looks at **all** validation trials, including the former val-test. Decoder
dials are tuned on val-dev. Reported val-dev numbers are therefore **exposed**: fine for comparing
methods treated identically, likely somewhat optimistic in absolute terms (§7.4).

---

## 5. Decoding: language-model search

### 5.1 The search graph

The decoder is a weighted finite-state transducer (WFST) beam search over one composed graph,
`TLG.fst` (3.9 GB on disk, ≈8.5 GB in memory):

| Component | Role |
|---|---|
| **T**: CTC token graph | how frames map to phonemes (repeats, blanks) |
| **L**: pronunciation dictionary | which phoneme sequences spell which words, with `|` after each word |
| **G**: 4-gram language model (`gen4g`) | how likely each word sequence is; built from OpenWebText, pruned (1e-8) to fit in memory; **125,074-word vocabulary** |

### 5.2 How a candidate is scored

As each frame's distribution arrives, the search extends thousands of partial candidates (a word
sequence plus a position within the current word), adds scores, and prunes those too far behind
the leader (beam 17, lattice beam 8, 200–7,000 active states).

```
score = acoustic_scale × (network evidence) + (language-model and lexicon cost from the graph)
```

**Network evidence** for a candidate is the log-probability of its single best frame-by-frame
alignment (Viterbi), summed over frames. Recomputed from la0's raw outputs for val-dev trial 3,
it matches the decoder's reported values exactly (−757.19, −782.11, −770.48 for three candidates).
For the top candidate, 164 of 238 frames are blank and contribute −737.98, almost entirely the
blank penalty (164 × 4.50); candidates differ only where their phonemes differ.

### 5.3 The two dials

Both are applied in code read for this document: `language_model/runtime/server/x86/python/lm_decoder.cc:30-36`
and `language_model/runtime/core/decoder/ctc_wfst_beam_search.cc:27-33`.

**Blank penalty (90).** Before the search, `log(90) = 4.50` is subtracted from the blank's
log-probability on every frame; nothing is renormalized.
- *Why:* CTC networks output blank on most frames, and staying blank costs no LM probability while
  emitting a word does, so short or weakly signalled words get dropped. The penalty makes each
  phoneme frame relatively cheaper, like a word-insertion bonus or a prior correction for blank.
- Too low → deletions; too high → spurious insertions. Near the optimum it matters little
  (9 / 30 / 90 → 7.03 / 6.91 / 6.87% on a 400-trial val-dev subset).

**Acoustic scale (0.4).** Every log-probability the search reads, including the penalized blank, is
multiplied by this value; LM costs enter unscaled. So it is the only network-vs-LM balance, and
the effective blank penalty in the search is 0.4 × 4.50 = 1.8 per frame.
- *Why below 1:* overlapping patches (each bin feeds 3.5 frames) and multi-frame phonemes make
  summed frame log-probabilities double-count evidence; CTC outputs are overconfident; the network
  already learned some language from its training sentences.
- *Example:* "the jury in **the us**" (evidence −782.1, LM −42.6) beats "the jury in **a just**"
  (−757.2, −51.3) whenever the scale is below 0.35.
- Lower values also slow the search (more candidates survive the beam): 0.13 ms/frame at 0.5 vs
  ≈0.5 ms at 0.25.

| Acoustic scale (400-trial sweep) | 0.25 | 0.35–0.4 | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|---|
| WER | 7.6–7.7% | **6.87%** | 7.2–7.3% | 9.4% | 11.5% |

The best pair depends on the network and LM (the Transformer's best scale is 0.2), so each model
is swept on val-dev.

### 5.4 Live output and revisions

The search always has a current best guess. It may revise earlier words when later evidence makes
another reading more likely: 27.9 revision edits per 100 final words for la0. Commitment rules
lock words so they never change, trading delay for locked-in mistakes. Replayed on la0's live
outputs: locking a word once it is unchanged for 80 ms gives 14.56% WER; locking it once 3 newer
words have appeared gives 8.47% but waits about 5.4 s, since this participant takes about 1.6 s per
word. With that rule, 0.4% of locked words differ from what the unconstrained final output would
have been. That measures stability, not correctness against the reference.

### 5.5 Sentence-end rescoring (finalization, not live)

When a trial ends, the search lists up to 100 complete candidates ranked by its own score (mean
30.6 per trial; 38,298 total on val-dev). GPT-2 Large (774M parameters) scores each one, and the
final pick maximizes

```
0.4 × network evidence + 0.25 × 4-gram score + 1.0 × GPT-2 log-probability
```

- Applied to **every** trial; the 100 caps the candidates per sentence.
- **Oracle** (best candidate per trial): 2.90% WER, so the right sentence is usually on the list.
- Result: 8.51% → **6.87%** (paired −1.63 [−2.25, −1.02]); 170 trials better, 68 worse.
  Example: the correct "the jury and a judge …" was ranked 17th of 91 by the search; GPT-2 picked it.
- Cost: 14.7 ms per sentence typical, 98.7 ms at the 95th percentile.

### 5.6 Vocabulary size

A smaller vocabulary does not help here. On the training transcripts, a top-15,000 vocabulary
would leave 6.4% of sentences with an unproducible word. On la0's val-dev output, a top-15k cut
could remove at most 33 errors (0.40 points) while making at least 141 reference words impossible
(1.70 points). The 4-gram already disfavors rare words softly.

---

## 6. Latency structure

| Delay | Value (reference GRU pipeline) |
|---|---|
| First possible output | 280 ms after trial start (14 bins) |
| Update cadence | every 80 ms |
| Wait for a new bin to enter a frame | 0–60 ms |
| Smoothing lag (average) | ≈24 ms, no look-ahead |
| Processing lag, measured at real speed (70 trials) | typical 9.5 ms; 95th percentile 13.5 ms; worst 51.7 ms |
| End of trial to final text | 95th percentile 12.7 ms (GPT-2 adds ≈15–100 ms) |
| Word latency (attempt → text) | **not measurable**: no word timing in the data |

Trial ends come from the dataset (oracle endpointing). Released-feature replay does not establish
raw-signal causality: the authors' preprocessing and normalization run upstream of the release.

---

## 7. Evaluation design

### 7.1 Metrics

- **WER:** minimum word substitutions + deletions + insertions to reach the reference, divided by
  reference words; pooled over trials.
- **PER:** the same over phonemes from greedy decoding (most likely class per frame, collapsed).
  Measures the network alone; diagnostic only.
- **Seen / unseen:** whether the exact sentence also appears in the training split. Unseen means
  unseen *text*, not an unseen day or participant. la0: 5.41% seen vs 8.74% unseen.

### 7.2 Uncertainty: session bootstrap

- Resample the 35 val-dev sessions **with replacement** 10,000 times and recompute pooled WER;
  the middle 95% is the CI. Whole sessions are resampled because trials from one day share that
  day's difficulty.
- Each resample has 35 draws but on average 22.4 distinct sessions; each session appears in about
  64% of resamples. 10,000 resamples make the CI endpoints stable to about ±0.05 points.
- A direct cluster standard-error formula gives nearly the same interval ([6.47, 10.54] vs
  [6.62, 10.65] for la0).
- **Paired comparison:** both systems use the same resampled sessions; the CI of the difference is
  much narrower (la4 − la0: [−0.29, 0.88] paired vs [−2.55, 3.13] unpaired). If it excludes 0 the
  difference is "resolved".
- Not covered: retraining (seed) variability, other participants, selection optimism.

### 7.3 Why sessions differ

Per-session la0 WER ranges from 0% to about 30%.
- **Sentence mix** explains about 60% of the between-session variance (Harvard and Random are much
  harder than Freq words).
- **The day itself** still matters: on Switchboard sentences alone, session WER ranges 1.2–15.5%;
  after removing sentence-sampling noise (≈2.2 points), the day-to-day spread is ≈2.8 points.
- Harvard and Random appear on only 2–3 days, so their corpus and day effects are confounded.

### 7.4 Exposure and the missing independent test

- The test split has no labels, so validation is the only labeled evaluation data.
- Checkpoints were selected on all validation trials, and dials tuned on val-dev; reported val-dev
  numbers are exploratory and likely somewhat optimistic.
- **Open decision:** split val-dev by session into a "choosing" half and a "reporting" half
  (cleaner, wider intervals), or keep all of val-dev and label results exposed.

### 7.5 Known limitations

- One participant; results do not generalize to other people without new data.
- Every evaluated day has a trained day layer; new-day performance is unmeasured.
- No word timing, so word latency cannot be measured.
- Processed features only; preprocessing, clipping and normalization provenance are upstream.
- Single training replicate per configuration; seed variance unmeasured.

---

## 8. Results so far (val-dev, exposed)

| System (acoustic scale / blank penalty) | WER | 95% CI | Seen / unseen |
|---|---|---|---|
| **la0**, reference causal GRU (0.4 / 90) | **8.51%** | 6.62–10.65 | 5.41 / 8.74 |
| g3, restrained day layers + stronger weight decay (0.5 / 90) | 8.36% | 6.53–10.47 | 4.71 / 8.63 |
| Authors' non-causal model, same LM and dials (0.4 / 90) | 8.70% | 6.74–10.82 | 5.24 / 8.96 |
| la4, 80 ms look-ahead smoothing (0.4 / 90) | 8.80% | 6.87–10.94 | 4.54 / 9.11 |
| rd1a, wider causal smoother (0.3 / 30) | 9.06% | 7.15–11.19 | 5.58 / 9.32 |
| g2, weak masking bundle (0.5 / 90) | 9.11% | 7.08–11.32 | 3.84 / 9.50 |
| Transformer (0.2 / 90) | 10.37% | 8.15–12.80 | 5.24 / 10.75 |
| Masked Transformer (0.4 / 90) | 10.21% | 7.75–13.05 | 3.66 / 10.70 |
| **la0 + GPT-2 re-ranking** (finalization) | **6.87%** | 5.09–8.92 | |

Paired readings:
- Causal vs the authors' non-causal model: +0.19 [−0.26, 0.62]. Not distinguishable in this
  pipeline; the dials were tuned for la0 and the authors' selection procedure is unrecorded.
- 80 ms look-ahead (la4 vs la0): +0.29 [−0.29, 0.88]. No measurable help. One run (rd1a) suggests
  la0's narrower smoothing, not causality itself, explains the ordering.
- Transformer vs GRU: +1.86 [+1.10, +2.65] at its best dials; worse in PER too.
- LM choice was the largest single lever: 1-gram 35.5% (tuned) → 4-gram 8.5%.
- In-domain LMs memorized: superb on seen sentences, worse on unseen.

---

## 9. Where things live

| Topic | Location |
|---|---|
| Model definition | `model_training/rnn_model.py` |
| Training loop, augmentation, CTC, checkpoint selection | `model_training/rnn_trainer.py` |
| Smoothing kernel | `model_training/data_augmentations.py` |
| Reference config | `model_training/rnn_args.yaml` |
| val-dev / val-test split | `model_training/splits.py` |
| Streaming decoder | `model_training/benchmark/streaming_infer.py` |
| LM worker and dial handling | `model_training/benchmark/lm_worker.py`, `stream_lm.py` |
| Bootstrap and paired comparison | `harness/metrics.py` |
| GPT-2 rescoring | `harness/rescore.py`, `harness/registry/rescorer/gpt2large.yaml` |
| Run records | `results/harness/runs/` |
| Block-level corpus table | `data/t15_copyTaskData_description.csv` |
| Project status and decisions | `STATUS_2026-09-29.md`, `brainstorm/PLAN.md`, `results/RESULTS.md` |

Figures for slides can be regenerated locally from the gitignored `learning/` scripts
(`.venv/bin/python learning/<script>.py`):

| Script | Figure |
|---|---|
| `smoother_demo.py` | raw vs smoothed TX/SBP, kernel weights, frequency response |
| `ctc_demo.py` | CTC brute force vs forward algorithm (text) |
| `gradient_demo.py` | backprop vs finite differences, step size (text) |
| `bootstrap_demo.py` | per-session WER, bootstrap distributions, paired vs unpaired |
| `session_difficulty.py` | session WER over time by corpus, corpus-predicted vs actual |
| `network_evidence.py` | per-frame outputs with best alignment, per-frame evidence |
| `vocab_coverage.py` | vocabulary coverage and error analysis (text) |
