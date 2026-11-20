# Model names, recipe IDs and run IDs

Every acoustic model has these labels.

- **Full name**, for example `gru1344_head256_pall_f160_20k_s10`. This is the model's actual name: the name of its
  registry file and what harness commands take. It never contains an ID.
- **Recipe ID**, for example `G2`: one configuration (everything in the full name but the seed and `run<N>`).
- **Run ID**, for example `G2_1`: one training run of that recipe. Plots, tables and writing use these IDs.

The registry was renamed to this convention on 2026-11-10. Each entry lists its old name under `aliases`, so an
old name still works in harness commands. Saved runs and training records show the current names. Checkpoint
folders and run folders were not renamed and still carry the old names ("Old name" in the tables below).

## Naming convention

A name reads left to right: the network, what it sees, how it was trained.

```
<network>_p<past>_f<future>[_<tags>]_<batches>k_s<seed>[_run<N>][_<weights>]

gru768_pall_f0_120k_s10                         A2_1
gru1344_head256_pall_f160_20k_s10               G2_1
tfm384x8_conv_p5120_f160_shared_20k_s10_avg     X6_1 avg (averaged weights)
```

| Part | Meaning |
|---|---|
| network | The architecture. It decides the letter. See "Network" below. |
| `p` | Past: how far back the model can use the signal, in ms. `pall` means no limit. |
| `f` | Future: how much signal after the current moment the model waits for, in ms. `f0` means none. `fall` means the rest of the sentence. |
| tags | Only what differs from the plain recipe, from the tag table below. A model with no tags is the plain recipe. |
| batches | Training batches, in thousands. |
| seed | Model seed. |
| `run<N>` | Present only when the same configuration was trained again (`run2`, `run3`). |
| weights | Which saved weights of the run. No suffix means the final snapshot. |

Times are in ms. The models step through the signal in frames 80 ms apart.

### Network

| Written as | Meaning |
|---|---|
| `gru768` | Causal GRU with 768 units. Every GRU has 5 layers. |
| `bigru448` | Bidirectional GRU with 448 units |
| `gru1344_head256` | Causal GRU with 1344 units and a 256-unit look-ahead head |
| `tfm512x8` | Transformer, width 512, 8 layers |
| `tfm384x8_conv` | Transformer, width 384, 8 layers, with a causal convolution (9 frames) in every block |
| `tfm384x5_lb` | Transformer, width 384, 5 layers, LightBeam style: frames of 80 ms that do not overlap and a learned mask token |

### Past: `pall` or a number

- **`pall`**: no cut-off. A normal GRU reads the sentence once from its start and carries a running memory forward,
  so at any moment it can still use anything since the sentence began. 58 of the 60 GRU runs are `pall`.
- **A number on a GRU** (`p280`, `p760`): a hard cut-off. The GRU is restarted at every 80 ms step and shown only a
  short window, so it knows nothing older. `p280` is the current frame alone (a frame is built from 280 ms of
  signal). `p760` adds the 6 frames before it. Only C3 and C4 work this way.
- **A number on a Transformer** (`p2560`, `p5120`, `p20000`): the attention window. Each layer attends to the last
  32, 64 or 250 frames. Stacked layers can pass on older information, so this is the reach of one layer, not a
  hard cut-off.

### Future: `f` is the total, and the name says where it comes from

`f` counts all signal after the end of the current frame, from every source. It is the acoustic look-ahead part
of the 200 ms, 500 ms and 1.5 s budgets, which also cover emission delay, commit wait and compute.

| Source of the future signal | Shown in the name by | Models |
|---|---|---|
| Look-ahead head. The main GRU stays causal, so its own output never depends on the future. A second, 256-unit GRU reads the main GRU's outputs for the next frames, backwards, and the output layer gets both. All of `f` comes through the head. | `head256` in the network | F, G |
| Input smoothing. The smoothing kernel reaches into future bins. | tag `sm<ms>`, for example `sm80` | A0, A1 |
| Attention. The Transformer's last layer attends to the next frames. | nothing. It is the source in a Transformer unless `sm` says otherwise. | X, Y |
| Sliding window. A bidirectional GRU is rerun on a short window. | `bigru` with a number after `p` | C3, C4 |
| Whole sentence. A bidirectional GRU runs over the full sentence. | `fall` | B, C1_1, C1_2 |

### Tags

Tags follow `f`. A smoothing tag comes first.

| Tag | Meaning |
|---|---|
| `sm80` | 80 ms of the future comes from input smoothing |
| `rawin`, `pal` | Look-ahead head reads the input frames instead of the GRU outputs; `pal` runs it forward and back with shared weights |
| `bn128` | Each 20 ms bin compressed from 512 to 128 values before frames are built |
| `shared` | One input matrix shared by all sessions |
| `noday` | No session-specific parameters at all |
| `mask`, `mask2x`, `mask3x` | Time and channel masking, at 1, 2 or 3 times the first strength |
| `tmask` | Time masking only |
| `t12joint` | Trained on T15 and T12 together |
| `t12init` | Started from a model pretrained on T12 only |
| `voice` | Adds 12 sessions from the T15 voice release |
| `sslcon`, `sslgnll` | Started from masked self-supervised pretraining: contrastive, or noise-aware reconstruction |
| `kd50`, `kd10` | Distilled from a teacher model, at weight 0.5 or 0.1 |
| `daylr` | Day-layer learning rate and weight decay changed |
| `lowlr` | Shared-layer learning rate 0.0015 instead of 0.005 |
| `bwmatch` | Wider causal smoothing (std 3.24 bins) |
| `reg` | Heavier regularisation: dropout 0.3, drop-path 0.2, weight decay 0.1 |
| `ent12` | Entropy term, weight 0.12 |
| `ictc4` | Intermediate CTC loss at layer 4, weight 0.3 |
| `ls05` | Label smoothing, weight 0.05 |
| `mask5f` | LightBeam-style time masks of at most 5 frames each (as Y4), not 5% of the trial's length |
| `sf0`-`sf4` | Out-of-fold model: the training sentences of fold K of 5 held out (`dataset.train_fold`, by sentence), for decoding them |
| `tx`, `sbp` | Only the 256 threshold-crossing features, or only the 256 spike-band-power features |
| `upstream` | The released model from the dataset's authors, not trained here |

A new tag gets a row here before its first run is named.

### Weights

| Suffix | Weights | Use |
|---|---|---|
| none | Final snapshot, at a batch fixed before training | The one results are reported on |
| `_avg` | Average of the last snapshots, also fixed before training | Reported when stated in advance |
| `_b20000` | Snapshot at that batch | Comparisons at equal training |
| `_bestval` | Checkpoint with the best validation phoneme error | Picked on validation data, so not for reported results |

Under the old names this was the other way round: the plain name was the best-validation checkpoint and `_final`
was the final snapshot. Runs before 2026-10-11, and the pilots X1 and X2, saved no final snapshot. They have no
plain name and exist only as `_bestval` or `_b20000`.

## Recipe and run IDs (since 2026-11-18)

- The **recipe ID** is a letter and a number, for example `Y6`. The letter is the network part of the name (type,
  size, number of layers, block style, the look-ahead head with its size); a change in training data, augmentation,
  schedule, seed or look-ahead length keeps the letter. GRU architectures take letters from **A upward**,
  Transformers from **Z downward**, each in order of first training; I and O are skipped. The number counts the
  recipes of the letter in order of first training; an upstream release is number 0.
- The **run ID** is the recipe ID, an underscore and the run number in order of training: `Y6_1`, `Y6_2`. A second
  seed or a retrain of the same configuration is the next run of the recipe, not a new recipe. Every row below is
  a run. Write the run ID whenever a result comes from one run; a recipe with a single run may be written with the
  recipe ID alone.
- A **recipe's result** is the mean over its runs, given with each run's value, for example "Y6 avg 6.96 (Y6_1 avg
  6.97, Y6_2 avg 6.95)". A recipe is never one set of weights: weights of separately trained runs cannot be averaged.
- **Weights** follow the run ID after a space: `Y6_1` is the final snapshot, `Y6_1 avg` the average of the run's
  late snapshots, `Y6_1 bestval` the best-validation checkpoint. They are different weight files of the same run and
  score differently (Y6_1 7.27, Y6_1 avg 6.97); the live deciders use `avg`.
- **Data variants that are not seeds**, such as the out-of-fold models, are recipes of their own, named by the base
  recipe and the tag: `Y6sf0`, run `Y6sf0_1`.
- Recipe and run IDs are never reused. Pipeline presets keep the recipe ID (`A2_gen4g` uses run A2_1's weights).

### Renamed on 2026-11-18

Before this date every run had its own letter ID. Seeds and retrains of one configuration became runs of one recipe;
every other ID `X` became run `X_1` of recipe `X`. The freed numbers are retired. Documents in `docs/history/` keep
the old IDs.

| Old ID | Run ID | Old ID | Run ID | Old ID | Run ID |
|---|---|---|---|---|---|
| A2 | A2_1 | D1 | D1_1 | G15 | G2_4 |
| A6 | A2_2 | D2 | D1_2 | Y6 | Y6_1 |
| A7 | A7_1 | D4 | D1_3 | Y10 | Y6_2 |
| A8 | A7_2 | D5 | D1_4 | Y7 | Y7_1 |
| A9 | A7_3 | G2 | G2_1 | Y9 | Y7_2 |
| C1 | C1_1 | G13 | G2_2 | Y14 | Y6sf0_1 |
| C2 | C1_2 | G14 | G2_3 | | |

Retired recipe numbers: A6, A8, A9, C2, D2, D4, D5, G13, G14, G15, Y9, Y10, Y14.

## Adding a model

1. Before training, work out the full name from the convention and pass it as `harness train --name`. If it needs
   a tag that is not in the tag table, add the tag there first.
2. Training registers the best-validation checkpoint as `<name>_bestval`. The queue helpers in
   `results/harness/queue/` register the final snapshot under the plain name and the average as `<name>_avg`.
3. Give it its ID. A new configuration gets the next free recipe number of its letter and run number 1; a new
   seed or a retrain of an existing recipe gets that recipe's next run number. It gets a new letter only if the
   network part is new; then add the letter to the Letters table. Next free recipe numbers: A12, B3, C5, D6, E2, F22,
   G18, X10, Y15, Z3.
4. In the same step as registering it, add its row to the table of its letter: run ID, full name, weights, start date
   and a short note. Remove it from "Planned runs" if it was listed there.

## Letters

| Letter | Name starts with | Architecture | First trained |
|---|---|---|---|
| A | `gru768` | Causal GRU, 768 units, 5 layers | 2026-07-07 |
| B | `bigru768` | Bidirectional GRU, 768 units, 5 layers | 2026-10-07 |
| C | `bigru448` | Bidirectional GRU, 448 units, 5 layers | 2026-10-11 |
| D | `gru1344` | Causal GRU, 1344 units, 5 layers | 2026-10-11 |
| E | `gru1024` | Causal GRU, 1024 units, 5 layers | 2026-10-12 |
| F | `gru768_head256` | Causal GRU, 768 units, 5 layers, with the 256-unit look-ahead head | 2026-10-12 |
| G | `gru1344_head256` | Causal GRU, 1344 units, 5 layers, with the 256-unit look-ahead head | 2026-10-14 |
| X | `tfm384x8_conv` | Transformer, width 384, 8 layers, a causal convolution in every block | 2026-10-26 |
| Y | `tfm384x5_lb` | Transformer, width 384, 5 layers, LightBeam style | 2026-10-16 |
| Z | `tfm512x8` | Transformer, width 512, 8 layers | 2026-09-28 |

The next new GRU architecture is H. The next new Transformer architecture is W.

## Pipeline presets

A preset is one model with one language model and its decode settings (`harness/registry/pipelines/`), for
example `harness bench A2_gen4g`. A preset is named `<recipe ID>_<language model>`. This is the only place where an
ID is part of a name; the preset file gives the model's full name and so fixes the weights.

| Preset | Model | Language model | Decode settings | Old name |
|---|---|---|---|---|
| `A0_gen4g` | `gru768_pall_f80_sm80_upstream_120k_s10_bestval` | general 4-gram | language-model defaults | `baseline_gen4g` |
| `A1_gen4g` | `gru768_pall_f80_sm80_120k_s10_bestval` | general 4-gram | language-model defaults (tuned on A2_1) | `la4_gen4g` |
| `A2_gen4g` | `gru768_pall_f0_120k_s10_bestval` | general 4-gram | language-model defaults (tuned on A2_1) | `la0_gen4g` |
| `A2_indomain4g` | `gru768_pall_f0_120k_s10_bestval` | in-domain 4-gram | language-model defaults | `la0_indomain4g` |
| `A2_mixed4g` | `gru768_pall_f0_120k_s10_bestval` | interpolated 4-gram | language-model defaults | `la0_mixed4g` |
| `A2_owt1gram` | `gru768_pall_f0_120k_s10_bestval` | shipped 1-gram | language-model defaults | `la0_owt1gram` |
| `A3_gen4g` | `gru768_pall_f0_mask_120k_s10_bestval` | general 4-gram | own sweep | `g2_gen4g` |
| `A4_gen4g` | `gru768_pall_f0_daylr_120k_s10_bestval` | general 4-gram | own sweep | `g3_gen4g` |
| `A5_gen4g` | `gru768_pall_f0_bwmatch_120k_s10_bestval` | general 4-gram | own sweep | `rd1a_gen4g` |
| `Z1_gen4g` | `tfm512x8_p5120_f0_120k_s10_bestval` | general 4-gram | own sweep | `tfm_gen4g` |
| `Y6_gen4g` | `tfm384x5_lb_p20000_f80_sm80_noday_v2_63k_s10_avg` (Y6_1 avg) | general 4-gram | own sweep (scale 0.3) | - |
| `D1_gen4g` | `gru1344_pall_f0_20k_s10` (D1_1) | general 4-gram | language-model defaults (tuned on A2_1) | - |
| `G2_gen4g` | `gru1344_head256_pall_f160_20k_s10` (G2_1) | general 4-gram | language-model defaults (tuned on A2_1) | - |

`Y6_gen4g`, `D1_gen4g` and `G2_gen4g` (2026-11-19) carry `live_rank` 1-3: the live player lists them first, as the
current best (Y6 decides the 500 ms and 1.5 s live results, D1 the 200 ms ones, G2 is the GRU baseline). The viewer
shows every model's ID next to its full name, read from the tables below.

Most benchmarks do not use a preset of their own: they start from `A2_gen4g` and swap the model
(`bench A2_gen4g --acoustic <full name>`), and the run is then named `<full name>+gen4g`.

## All trained models

One row per training run, grouped by recipe. "Weights" lists what is registered for the run: `final` is the plain name, the others are the suffixes.

### A: Causal GRU, 768 units, 5 layers

| Run | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| A0_1 | `gru768_pall_f80_sm80_upstream_120k_s10` | bestval | `pretrained_baseline` | 2025-07 | Upstream released model, symmetric smoothing. Offline reference only. |
| A1_1 | `gru768_pall_f80_sm80_120k_s10` | bestval | `causal_la4` | 2026-07-07 | Our retrain of the upstream recipe: smoothing reaches 80 ms ahead. |
| A2_1 | `gru768_pall_f0_120k_s10` | bestval | `causal_la0` | 2026-07-08 | Reference recipe. Every later run is a fork of this one. |
| A2_2 | `gru768_pall_f0_30k_s11` | b20000, bestval | `la0_seed11_30k` | 2026-10-08 | Seed check, stopped at 30k of the 120k schedule. |
| A3_1 | `gru768_pall_f0_mask_120k_s10` | bestval | `g2_masking` | 2026-08-07 | Time and channel masking. |
| A4_1 | `gru768_pall_f0_daylr_120k_s10` | bestval | `g3_objective` | 2026-08-07 | Day-layer learning rate and weight decay changed. |
| A5_1 | `gru768_pall_f0_bwmatch_120k_s10` | bestval | `rd1a_bwmatched` | 2026-08-09 | Wider causal smoothing (std 3.24 bins), bandwidth-matched control. |
| A7_1 | `gru768_pall_f0_20k_s10` | final, bestval | `anc_causal_s10` | 2026-10-11 | 20k-batch anchor. |
| A7_2 | `gru768_pall_f0_20k_s11` | final, bestval | `anc_causal_s11` | 2026-10-11 | Seed 11. |
| A7_3 | `gru768_pall_f0_20k_s12` | final, bestval | `anc_causal_s12` | 2026-10-12 | Seed 12. |
| A10_1 | `gru768_pall_f0_tmask_20k_s10` | final, bestval | `anc_causal_tmask_s10` | 2026-10-12 | Time masking only. |
| A11_1 | `gru768_pall_f0_mask_20k_s10` | final, bestval | `g2_causal_20k_s10` | 2026-10-15 | Time and channel masking at 20k. |

### B: Bidirectional GRU, 768 units, 5 layers

| Run | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| B1_1 | `bigru768_pall_fall_30k_s10` | b20000 | `bigru768_30k_b20k` | 2026-10-07 | Crashed near batch 26,400; only the batch-20,000 snapshot is registered. |
| B2_1 | `bigru768_pall_fall_20k_s10` | final, bestval | `anc_bi768_s10` | 2026-10-11 | Whole-sentence anchor. Also the source of the word cut points. |

### C: Bidirectional GRU, 448 units, 5 layers

| Run | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| C1_1 | `bigru448_pall_fall_20k_s10` | final, bestval | `anc_bi448_s10` | 2026-10-11 | Whole-sentence, parameter-matched to the 768 causal GRU. |
| C1_2 | `bigru448_pall_fall_20k_s11` | final, bestval | `anc_bi448_s11` | 2026-10-12 | Seed 11. |
| C3_1 | `bigru448_p280_f160_20k_s10` | final, bestval | `win_p0_r2_s10` | 2026-10-12 | Sliding window: rerun every frame on the current frame and 2 future frames. |
| C4_1 | `bigru448_p760_f160_20k_s10` | final, bestval | `win_p6_r2_s10` | 2026-10-13 | Sliding window with 6 more past frames. |

### D: Causal GRU, 1344 units, 5 layers

| Run | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| D1_1 | `gru1344_pall_f0_20k_s10` | final, bestval | `anc_causal1344_s10` | 2026-10-11 | 20k-batch anchor. |
| D1_2 | `gru1344_pall_f0_20k_s11` | final, bestval | `anc_causal1344_s11` | 2026-10-12 | Seed 11. |
| D1_3 | `gru1344_pall_f0_20k_s12` | final, bestval | `anc_causal1344_s12` | 2026-10-20 | Seed 12. |
| D1_4 | `gru1344_pall_f0_20k_s13` | final, bestval | `anc_causal1344_s13` | 2026-10-20 | Seed 13. |
| D3_1 | `gru1344_pall_f0_60k_s10` | final, bestval | `causal1344_60k_s10` | 2026-10-18 | 60k schedule. |

### E: Causal GRU, 1024 units, 5 layers

| Run | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| E1_1 | `gru1024_pall_f0_20k_s10` | final, bestval | `anc_causal1024_s10` | 2026-10-12 | 20k-batch anchor. |

### F: Causal GRU, 768 units, 5 layers, with the 256-unit look-ahead head

| Run | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| F1_1 | `gru768_head256_pall_f0_20k_s10` | final, bestval | `la_r0_s10` | 2026-10-12 | Head with no future frames: control for the head's extra weights. |
| F2_1 | `gru768_head256_pall_f160_20k_s10` | final, bestval | `la_r2_s10` | 2026-10-12 |  |
| F3_1 | `gru768_head256_pall_f480_20k_s10` | final, bestval | `la_r6_s10` | 2026-10-12 |  |
| F4_1 | `gru768_head256_pall_f1440_20k_s10` | final, bestval | `la_r18_s10` | 2026-10-12 |  |
| F5_1 | `gru768_head256_pall_f960_20k_s10` | final, bestval | `la_r12_s10` | 2026-10-13 |  |
| F6_1 | `gru768_head256_pall_f1920_20k_s10` | final, bestval | `la_r24_s10` | 2026-10-13 |  |
| F7_1 | `gru768_head256_pall_f1440_rawin_20k_s10` | final, bestval | `laraw_r18_s10` | 2026-10-13 | Head reads the input frames instead of the GRU outputs. |
| F8_1 | `gru768_head256_pall_f1440_bn128_20k_s10` | final, bestval | `bn128_la_r18_s10` | 2026-10-14 | Each bin compressed 512 to 128 before framing. |
| F9_1 | `gru768_head256_pall_f1440_rawin_pal_20k_s10` | final, bestval | `lapal_raw_r18_s10` | 2026-10-14 | Head runs forward and back over the input frames with shared weights. |
| F10_1 | `gru768_head256_pall_f1440_t12joint_20k_s10` | final, bestval | `joint_la_r18_s10` | 2026-10-15 | Trained on T15 and T12 together. |
| F11_1 | `gru768_head256_pall_f480_t12joint_20k_s10` | final, bestval | `joint_la_r6_s10` | 2026-10-15 | Trained on T15 and T12 together. |
| F12_1 | `gru768_head256_pall_f1440_t12init_20k_s10` | final, bestval | `ft_t12_la_r18_s10` | 2026-10-15 | Started from T12-only pretraining. |
| F13_1 | `gru768_head256_pall_f1440_t12init_lowlr_20k_s10` | final, bestval | `ft_lowlr_t12_la_r18_s10` | 2026-10-15 | Started from T12-only pretraining, shared-layer learning rate 0.0015. |
| F14_1 | `gru768_head256_pall_f1440_voice_20k_s10` | final, bestval | `voice_rebin_la_r18_s10` | 2026-10-15 | Adds 12 T15 voice-release sessions. |
| F15_1 | `gru768_head256_pall_f1440_sslcon_20k_s10` | final, bestval | `ft_ssl3_con_la_r18_20k_s10` | 2026-10-16 | Started from contrastive masked pretraining. |
| F16_1 | `gru768_head256_pall_f1440_sslgnll_20k_s10` | final, bestval | `ft_ssl3_gnll_la_r18_20k_s10` | 2026-10-16 | Started from reconstruction masked pretraining. |
| F17_1 | `gru768_head256_pall_f1440_60k_s10` | final, bestval | `la_r18_60k_s10` | 2026-10-16 | 60k schedule. Teacher of G10. |
| F18_1 | `gru768_head256_pall_f1440_sslgnll_60k_s10` | final, bestval | `ft60k_ssl3_gnll_la_r18_s10` | 2026-10-16 | Reconstruction pretraining, 60k. |
| F19_1 | `gru768_head256_pall_f1440_sslcon_60k_s10` | final, bestval | `ft60k_ssl3_con_la_r18_s10` | 2026-10-16 | Contrastive pretraining, 60k. |
| F20_1 | `gru768_head256_pall_f1440_mask_60k_s10` | final, b20000, b40000, bestval | `g2_la_r18_60k_s10` | 2026-10-17 | Masking, 60k. |
| F21_1 | `gru768_head256_pall_f1440_mask_20k_s10` | final, bestval | `g2_la_r18_20k_s10` | 2026-10-17 | Masking, 20k. |

### G: Causal GRU, 1344 units, 5 layers, with the 256-unit look-ahead head

| Run | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| G1_1 | `gru1344_head256_pall_f1440_20k_s10` | final, bestval | `la1344_r18_s10` | 2026-10-14 |  |
| G2_1 | `gru1344_head256_pall_f160_20k_s10` | final, bestval | `la1344_r2_s10` | 2026-10-15 | First training of this configuration. |
| G2_2 | `gru1344_head256_pall_f160_20k_s11` | final, bestval | `la1344_r2_s11` | 2026-10-20 | Seed 11. |
| G2_3 | `gru1344_head256_pall_f160_20k_s10_run2` | final, bestval | `fix_la1344_r2_s10` | 2026-10-27 | G2_1 trained again after the torch.compile gradient fix of 2026-10-26. |
| G2_4 | `gru1344_head256_pall_f160_20k_s10_run3` | final, avg, bestval | `snap_la1344_r2_20k_s10` | 2026-10-28 | G2_1 trained a third time, saving the snapshots for averaging. |
| G3_1 | `gru1344_head256_pall_f480_20k_s10` | final, bestval | `la1344_r6_s10` | 2026-10-15 | Teacher of G11. |
| G4_1 | `gru1344_head256_pall_f1440_mask_20k_s10` | final, bestval | `all_la1344_r18_20k_s10` | 2026-10-15 | Masking. |
| G5_1 | `gru1344_head256_pall_f160_mask_20k_s10` | final, bestval | `g2_la1344_r2_20k_s10` | 2026-10-17 | Masking. |
| G6_1 | `gru1344_head256_pall_f480_mask_20k_s10` | final, bestval | `g2_la1344_r6_20k_s10` | 2026-10-17 | Masking. |
| G7_1 | `gru1344_head256_pall_f480_mask2x_20k_s10` | final, bestval | `m2x_la1344_r6_20k_s10` | 2026-10-17 | Masking at twice the strength. |
| G8_1 | `gru1344_head256_pall_f480_mask3x_20k_s10` | final, bestval | `m3x_la1344_r6_20k_s10` | 2026-10-18 | Masking at three times the strength. |
| G9_1 | `gru1344_head256_pall_f160_60k_s10` | final, bestval | `la1344_r2_60k_s10` | 2026-10-18 | 60k schedule. |
| G10_1 | `gru1344_head256_pall_f160_kd50_20k_s10` | final, bestval | `kd_la1344_r2_20k_s10` | 2026-10-19 | Distilled from F17, weight 0.5. |
| G11_1 | `gru1344_head256_pall_f160_kd10_20k_s10` | final, bestval | `kd2_la1344_r2_20k_s10` | 2026-10-19 | Distilled from G3, weight 0.1. |
| G12_1 | `gru1344_head256_pall_f80_20k_s10` | final, bestval | `la1344_r1_s10` | 2026-10-20 |  |
| G16_1 | `gru1344_head256_pall_f160_splice_20k_s10` | final, avg, bestval | - | 2026-11-11 | The G2 recipe with phrase splicing (probability 0.3). Best WER 7.70 / 7.71 (final / avg) vs G2_4 7.77 / 7.94: keep rule not met. |
| G17_1 | `gru1344_head256_pall_f160_diph_20k_s10` | final, avg, bestval | - | 2026-11-11 | The G2 recipe with the diphone auxiliary target (weight 0.3). Best WER 8.37 / 8.24: worse than G2_4. |

### X: Transformer, width 384, 8 layers, a causal convolution in every block

| Run | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| X1_1 | `tfm384x8_conv_p5120_f160_8k_s10` | bestval | `tfmp_conv_fix_s10` | 2026-10-26 | Pilot, stopped at 8k of a 20k schedule. |
| X2_1 | `tfm384x8_conv_p2560_f160_reg_8k_s10` | bestval | `tfmp_convreg_fix_s10` | 2026-10-26 | Pilot with heavier regularisation and a 32-frame window. |
| X3_1 | `tfm384x8_conv_p5120_f160_60k_s10` | final, bestval | `tfm_conv_fix_r2_60k_s10` | 2026-10-27 | 60k schedule. |
| X4_1 | `tfm384x8_conv_p5120_f160_20k_s10` | final, avg, bestval | `tfm_conv_r2_20k_s10` | 2026-10-27 | 20k baseline for the three screens below. |
| X5_1 | `tfm384x8_conv_p5120_f160_ent12_20k_s10` | final, avg, bestval | `tfm_conv_ent12_r2_20k_s10` | 2026-10-27 | Entropy term, weight 0.12. |
| X6_1 | `tfm384x8_conv_p5120_f160_shared_20k_s10` | final, avg, bestval | `tfm_conv_shared_r2_20k_s10` | 2026-10-27 | One input matrix shared by all sessions. |
| X7_1 | `tfm384x8_conv_p5120_f160_ictc4_20k_s10` | final, avg, bestval | `tfm_conv_ictc4_r2_20k_s10` | 2026-10-27 | Intermediate CTC loss at layer 4. |
| X8_1 | `tfm384x8_conv_p5120_f160_shared_splice_20k_s10` | final, avg, bestval | - | 2026-11-12 | X6's recipe with phrase splicing (probability 0.3). Best WER 8.72 / 8.52 vs X6 8.80 / 8.67: keep rule not met. |
| X9_1 | `tfm384x8_conv_p5120_f160_shared_diph_20k_s10` | final, avg, bestval | - | 2026-11-16 | X6's recipe with the diphone auxiliary target (weight 0.3). Best WER 8.96 / 8.81: worse than X6. |

### Y: Transformer, width 384, 5 layers, LightBeam style

| Run | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| Y1_1 | `tfm384x5_lb_p20000_f1440_20k_s10` | final, bestval | `lb_tfm_r18_20k_s10` | 2026-10-16 | First LightBeam-style run, with our per-session input layers. |
| Y2_1 | `tfm384x5_lb_p20000_f1440_60k_s10` | final, bestval | `lb_tfm_r18_60k_s10` | 2026-10-16 | 60k schedule. |
| Y3_1 | `tfm384x5_lb_p20000_f0_noday_63k_s10` | final, avg, bestval | `tfm_lb_r0_63k_s10` | 2026-11-03 | LightBeam reproduction: no per-session layers, step schedule. |
| Y4_1 | `tfm384x5_lb_p20000_f160_noday_63k_s10` | final, avg, bestval | `tfm_lb_r2_63k_s10` | 2026-11-03 | LightBeam reproduction with 2 frames of look-ahead. |
| Y5_1 | `tfm384x5_lb_p20000_f160_noday_ls05_63k_s10` | final, avg, bestval | `tfm_lb_ls05_r2_63k_s10` | 2026-11-04 | Adds label smoothing 0.05. |
| Y6_1 | `tfm384x5_lb_p20000_f80_sm80_noday_v2_63k_s10` | final, avg, bestval | - | 2026-11-11 | Corrected LightBeam recipe (v2) with their centred smoothing, no attention look-ahead. Reproduction control. |
| Y6_2 | `tfm384x5_lb_p20000_f80_sm80_noday_v2_63k_s11` | final, avg, bestval | - | 2026-11-17 | Second run of Y6 (seed 11, data seed 2). Best WER 6.95 / 6.95 at scale 0.3 (Y6_1 7.27 / 6.97). |
| Y6sf0_1 | `tfm384x5_lb_p20000_f80_sm80_noday_v2_sf0_63k_s10` | final, avg, bestval | - | 2026-11-18 | Out-of-fold model 0 of 5 on the Y6 recipe (training sentences of fold 0 held out, 6,469 of 8,072 trials kept), for the selector's held-out decodes. Full-val PER at the final batch 10.07 (training log; Y6_1 8.89). |
| Y6sf1_1 | `tfm384x5_lb_p20000_f80_sm80_noday_v2_sf1_63k_s10` | final, avg, bestval | - | 2026-11-18 | Out-of-fold model 1 of 5 on the Y6 recipe (training sentences of fold 1 held out, 6,436 of 8,072 trials kept). Full-val PER at the final batch 10.05 (training log; Y6_1 8.89). |
| Y6sf2_1 | `tfm384x5_lb_p20000_f80_sm80_noday_v2_sf2_63k_s10` | final, avg, bestval | - | 2026-11-18 | Out-of-fold model 2 of 5 on the Y6 recipe (training sentences of fold 2 held out, 6,456 of 8,072 trials kept). Full-val PER at the final batch 10.17 (training log; Y6_1 8.89). |
| Y6sf3_1 | `tfm384x5_lb_p20000_f80_sm80_noday_v2_sf3_63k_s10` | final, avg, bestval | - | 2026-11-18 | Out-of-fold model 3 of 5 on the Y6 recipe (training sentences of fold 3 held out, 6,497 of 8,072 trials kept). Full-val PER at the final batch 10.11 (training log; Y6_1 8.89). |
| Y6sf4_1 | `tfm384x5_lb_p20000_f80_sm80_noday_v2_sf4_63k_s10` | final, avg, bestval | - | 2026-11-19 | Out-of-fold model 4 of 5 on the Y6 recipe (training sentences of fold 4 held out, 6,430 of 8,072 trials kept). Full-val PER at the final batch 10.12 (training log; Y6_1 8.89). A first attempt (2026-11-18 23:24, run folder `20261118T232434Z`) was stopped at batch 22,600 of 63,000 to free the GPU for the 2026-11-19 presentation runs; this is the rerun from scratch. |
| Y7_1 | `tfm384x5_lb_p20000_f160_noday_v2_63k_s10` | final, avg, bestval | - | 2026-11-11 | Corrected LightBeam recipe (v2), causal smoothing, 160 ms of attention look-ahead. |
| Y7_2 | `tfm384x5_lb_p20000_f160_noday_v2_63k_s11` | final, avg, bestval | - | 2026-11-16 | Second run of Y7 (seed 11, data seed 2). Best WER 7.66 / 7.58 (Y7_1 7.44 / 7.43). |
| Y8_1 | `tfm384x5_lb_p20000_f160_noday_v2_crctc_32k_s10` | final, avg, bestval | - | 2026-11-12 | Y7_1 + consistency-regularized CTC (weight 0.2), half the batches. Best WER 8.64 / 8.57 at scale 0.6 / 0.5 (Y7_1 7.44 / 7.43): keep rule not met. |
| Y11_1 | `tfm384x5_lb_p20000_f80_sm80_noday_v2_tx_63k_s10` | final, avg, bestval | - | 2026-11-18 | The Y6 recipe on the 256 threshold-crossing features only, as an ensemble member. Greedy val-dev PER 11.80 / 11.62 (Y6_1 8.07 / 7.95). |
| Y12_1 | `tfm384x5_lb_p20000_f80_sm80_noday_v2_sbp_63k_s10` | final, avg, bestval | - | 2026-11-18 | The Y6 recipe on the 256 spike-band-power features only, as an ensemble member. Greedy val-dev PER 8.33 / 8.26. |
| Y13_1 | `tfm384x5_lb_p20000_f80_sm80_noday_v2_mask5f_63k_s10` | final, avg, bestval | - | 2026-11-18 | The Y6 recipe with Y4's time masks (at most 5 frames each). Best WER 8.03 / 8.01 at scale 0.25 (Y6_1 7.27 / 6.97); word ends earlier (own-boundary p95 120 vs 360 ms). Keep rule not met. |

### Z: Transformer, width 512, 8 layers

| Run | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| Z1_1 | `tfm512x8_p5120_f0_120k_s10` | bestval | `tfm_d512l8_adamw` | 2026-09-28 | First causal Transformer. |
| Z2_1 | `tfm512x8_p5120_f0_tmask_120k_s10` | bestval | `tfm_d512l8_tmask` | 2026-09-29 | Time masking with a learned mask token. |

## Planned runs

Prepared in `results/harness/queue/` and not trained. Each takes the next free number of its letter when its
training starts. New tags: `v2` (the 2026-11-11 corrections to the LightBeam reproduction), `crctc` (CR-CTC
consistency loss, weight 0.2), `splice` (phrase splicing), `diph` (diphone targets).

None at the moment.
