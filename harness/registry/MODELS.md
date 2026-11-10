# Model names and letter IDs

Every acoustic model has two labels.

- **Full name**, for example `gru1344_head256_pall_f160_20k_s10`. This is the model's actual name: the name of its
  registry file and what harness commands take. It never contains the letter ID.
- **Letter ID**, for example `G2`. This is a short label for plots, tables and writing. It points to one training run.

The registry was renamed to this convention on 2026-11-10. Each entry lists its old name under `aliases`, so an
old name still works in harness commands. Checkpoint folders, run folders and saved results were not renamed and
still carry the old names ("Old name" in the tables below).

## Naming convention

A name reads left to right: the network, what it sees, how it was trained.

```
<network>_p<past>_f<future>[_<tags>]_<batches>k_s<seed>[_run<N>][_<weights>]

gru768_pall_f0_120k_s10                         A2
gru1344_head256_pall_f160_20k_s10               G2
tfm384x8_conv_p5120_f160_shared_20k_s10_avg     X6, averaged weights
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
| Whole sentence. A bidirectional GRU runs over the full sentence. | `fall` | B, C1, C2 |

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

## Letter IDs

- The **letter** is the network part of the name: type, size, number of layers, block style, and the look-ahead
  head with its size. A change in training data, augmentation, schedule, seed or look-ahead length keeps the letter.
- GRU architectures take letters from **A upward**. Transformer architectures take letters from **Z downward**.
  Each side is in order of first training. I and O are skipped.
- The **number** is the order in which training started within the letter. An upstream release is number 0.
- The ID belongs to the training run. All saved weights of the run share it: write "X6" or, when it matters,
  "X6 avg".
- IDs cover the models in the registry. An ID is never changed or reused.

## Adding a model

1. Before training, work out the full name from the convention and pass it as `harness train --name`. If it needs
   a tag that is not in the tag table, add the tag there first.
2. Training registers the best-validation checkpoint as `<name>_bestval`. The queue helpers in
   `results/harness/queue/` register the final snapshot under the plain name and the average as `<name>_avg`.
3. Give the run the next free number of its letter. It gets a new letter only if the network part is new; then add
   the letter to the Letters table.
4. In the same step as registering it, add its row to the table of its letter: ID, full name, weights, start date
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

## All trained models

One row per training run. "Weights" lists what is registered for the run: `final` is the plain name, the others are the suffixes.

### A: Causal GRU, 768 units, 5 layers

| ID | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| A0 | `gru768_pall_f80_sm80_upstream_120k_s10` | bestval | `pretrained_baseline` | 2025-07 | Upstream released model, symmetric smoothing. Offline reference only. |
| A1 | `gru768_pall_f80_sm80_120k_s10` | bestval | `causal_la4` | 2026-07-07 | Our retrain of the upstream recipe: smoothing reaches 80 ms ahead. |
| A2 | `gru768_pall_f0_120k_s10` | bestval | `causal_la0` | 2026-07-08 | Reference recipe. Every later run is a fork of this one. |
| A3 | `gru768_pall_f0_mask_120k_s10` | bestval | `g2_masking` | 2026-08-07 | Time and channel masking. |
| A4 | `gru768_pall_f0_daylr_120k_s10` | bestval | `g3_objective` | 2026-08-07 | Day-layer learning rate and weight decay changed. |
| A5 | `gru768_pall_f0_bwmatch_120k_s10` | bestval | `rd1a_bwmatched` | 2026-08-09 | Wider causal smoothing (std 3.24 bins), bandwidth-matched control. |
| A6 | `gru768_pall_f0_30k_s11` | b20000, bestval | `la0_seed11_30k` | 2026-10-08 | Seed check, stopped at 30k of the 120k schedule. |
| A7 | `gru768_pall_f0_20k_s10` | final, bestval | `anc_causal_s10` | 2026-10-11 | 20k-batch anchor. |
| A8 | `gru768_pall_f0_20k_s11` | final, bestval | `anc_causal_s11` | 2026-10-11 | Seed 11. |
| A9 | `gru768_pall_f0_20k_s12` | final, bestval | `anc_causal_s12` | 2026-10-12 | Seed 12. |
| A10 | `gru768_pall_f0_tmask_20k_s10` | final, bestval | `anc_causal_tmask_s10` | 2026-10-12 | Time masking only. |
| A11 | `gru768_pall_f0_mask_20k_s10` | final, bestval | `g2_causal_20k_s10` | 2026-10-15 | Time and channel masking at 20k. |

### B: Bidirectional GRU, 768 units, 5 layers

| ID | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| B1 | `bigru768_pall_fall_30k_s10` | b20000 | `bigru768_30k_b20k` | 2026-10-07 | Crashed near batch 26,400; only the batch-20,000 snapshot is registered. |
| B2 | `bigru768_pall_fall_20k_s10` | final, bestval | `anc_bi768_s10` | 2026-10-11 | Whole-sentence anchor. Also the source of the word cut points. |

### C: Bidirectional GRU, 448 units, 5 layers

| ID | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| C1 | `bigru448_pall_fall_20k_s10` | final, bestval | `anc_bi448_s10` | 2026-10-11 | Whole-sentence, parameter-matched to the 768 causal GRU. |
| C2 | `bigru448_pall_fall_20k_s11` | final, bestval | `anc_bi448_s11` | 2026-10-12 | Seed 11. |
| C3 | `bigru448_p280_f160_20k_s10` | final, bestval | `win_p0_r2_s10` | 2026-10-12 | Sliding window: rerun every frame on the current frame and 2 future frames. |
| C4 | `bigru448_p760_f160_20k_s10` | final, bestval | `win_p6_r2_s10` | 2026-10-13 | Sliding window with 6 more past frames. |

### D: Causal GRU, 1344 units, 5 layers

| ID | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| D1 | `gru1344_pall_f0_20k_s10` | final, bestval | `anc_causal1344_s10` | 2026-10-11 | 20k-batch anchor. |
| D2 | `gru1344_pall_f0_20k_s11` | final, bestval | `anc_causal1344_s11` | 2026-10-12 | Seed 11. |
| D3 | `gru1344_pall_f0_60k_s10` | final, bestval | `causal1344_60k_s10` | 2026-10-18 | 60k schedule. |
| D4 | `gru1344_pall_f0_20k_s12` | final, bestval | `anc_causal1344_s12` | 2026-10-20 | Seed 12. |
| D5 | `gru1344_pall_f0_20k_s13` | final, bestval | `anc_causal1344_s13` | 2026-10-20 | Seed 13. |

### E: Causal GRU, 1024 units, 5 layers

| ID | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| E1 | `gru1024_pall_f0_20k_s10` | final, bestval | `anc_causal1024_s10` | 2026-10-12 | 20k-batch anchor. |

### F: Causal GRU, 768 units, 5 layers, with the 256-unit look-ahead head

| ID | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| F1 | `gru768_head256_pall_f0_20k_s10` | final, bestval | `la_r0_s10` | 2026-10-12 | Head with no future frames: control for the head's extra weights. |
| F2 | `gru768_head256_pall_f160_20k_s10` | final, bestval | `la_r2_s10` | 2026-10-12 |  |
| F3 | `gru768_head256_pall_f480_20k_s10` | final, bestval | `la_r6_s10` | 2026-10-12 |  |
| F4 | `gru768_head256_pall_f1440_20k_s10` | final, bestval | `la_r18_s10` | 2026-10-12 |  |
| F5 | `gru768_head256_pall_f960_20k_s10` | final, bestval | `la_r12_s10` | 2026-10-13 |  |
| F6 | `gru768_head256_pall_f1920_20k_s10` | final, bestval | `la_r24_s10` | 2026-10-13 |  |
| F7 | `gru768_head256_pall_f1440_rawin_20k_s10` | final, bestval | `laraw_r18_s10` | 2026-10-13 | Head reads the input frames instead of the GRU outputs. |
| F8 | `gru768_head256_pall_f1440_bn128_20k_s10` | final, bestval | `bn128_la_r18_s10` | 2026-10-14 | Each bin compressed 512 to 128 before framing. |
| F9 | `gru768_head256_pall_f1440_rawin_pal_20k_s10` | final, bestval | `lapal_raw_r18_s10` | 2026-10-14 | Head runs forward and back over the input frames with shared weights. |
| F10 | `gru768_head256_pall_f1440_t12joint_20k_s10` | final, bestval | `joint_la_r18_s10` | 2026-10-15 | Trained on T15 and T12 together. |
| F11 | `gru768_head256_pall_f480_t12joint_20k_s10` | final, bestval | `joint_la_r6_s10` | 2026-10-15 | Trained on T15 and T12 together. |
| F12 | `gru768_head256_pall_f1440_t12init_20k_s10` | final, bestval | `ft_t12_la_r18_s10` | 2026-10-15 | Started from T12-only pretraining. |
| F13 | `gru768_head256_pall_f1440_t12init_lowlr_20k_s10` | final, bestval | `ft_lowlr_t12_la_r18_s10` | 2026-10-15 | Started from T12-only pretraining, shared-layer learning rate 0.0015. |
| F14 | `gru768_head256_pall_f1440_voice_20k_s10` | final, bestval | `voice_rebin_la_r18_s10` | 2026-10-15 | Adds 12 T15 voice-release sessions. |
| F15 | `gru768_head256_pall_f1440_sslcon_20k_s10` | final, bestval | `ft_ssl3_con_la_r18_20k_s10` | 2026-10-16 | Started from contrastive masked pretraining. |
| F16 | `gru768_head256_pall_f1440_sslgnll_20k_s10` | final, bestval | `ft_ssl3_gnll_la_r18_20k_s10` | 2026-10-16 | Started from reconstruction masked pretraining. |
| F17 | `gru768_head256_pall_f1440_60k_s10` | final, bestval | `la_r18_60k_s10` | 2026-10-16 | 60k schedule. Teacher of G10. |
| F18 | `gru768_head256_pall_f1440_sslgnll_60k_s10` | final, bestval | `ft60k_ssl3_gnll_la_r18_s10` | 2026-10-16 | Reconstruction pretraining, 60k. |
| F19 | `gru768_head256_pall_f1440_sslcon_60k_s10` | final, bestval | `ft60k_ssl3_con_la_r18_s10` | 2026-10-16 | Contrastive pretraining, 60k. |
| F20 | `gru768_head256_pall_f1440_mask_60k_s10` | final, b20000, b40000, bestval | `g2_la_r18_60k_s10` | 2026-10-17 | Masking, 60k. |
| F21 | `gru768_head256_pall_f1440_mask_20k_s10` | final, bestval | `g2_la_r18_20k_s10` | 2026-10-17 | Masking, 20k. |

### G: Causal GRU, 1344 units, 5 layers, with the 256-unit look-ahead head

| ID | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| G1 | `gru1344_head256_pall_f1440_20k_s10` | final, bestval | `la1344_r18_s10` | 2026-10-14 |  |
| G2 | `gru1344_head256_pall_f160_20k_s10` | final, bestval | `la1344_r2_s10` | 2026-10-15 | First training of this configuration. |
| G3 | `gru1344_head256_pall_f480_20k_s10` | final, bestval | `la1344_r6_s10` | 2026-10-15 | Teacher of G11. |
| G4 | `gru1344_head256_pall_f1440_mask_20k_s10` | final, bestval | `all_la1344_r18_20k_s10` | 2026-10-15 | Masking. |
| G5 | `gru1344_head256_pall_f160_mask_20k_s10` | final, bestval | `g2_la1344_r2_20k_s10` | 2026-10-17 | Masking. |
| G6 | `gru1344_head256_pall_f480_mask_20k_s10` | final, bestval | `g2_la1344_r6_20k_s10` | 2026-10-17 | Masking. |
| G7 | `gru1344_head256_pall_f480_mask2x_20k_s10` | final, bestval | `m2x_la1344_r6_20k_s10` | 2026-10-17 | Masking at twice the strength. |
| G8 | `gru1344_head256_pall_f480_mask3x_20k_s10` | final, bestval | `m3x_la1344_r6_20k_s10` | 2026-10-18 | Masking at three times the strength. |
| G9 | `gru1344_head256_pall_f160_60k_s10` | final, bestval | `la1344_r2_60k_s10` | 2026-10-18 | 60k schedule. |
| G10 | `gru1344_head256_pall_f160_kd50_20k_s10` | final, bestval | `kd_la1344_r2_20k_s10` | 2026-10-19 | Distilled from F17, weight 0.5. |
| G11 | `gru1344_head256_pall_f160_kd10_20k_s10` | final, bestval | `kd2_la1344_r2_20k_s10` | 2026-10-19 | Distilled from G3, weight 0.1. |
| G12 | `gru1344_head256_pall_f80_20k_s10` | final, bestval | `la1344_r1_s10` | 2026-10-20 |  |
| G13 | `gru1344_head256_pall_f160_20k_s11` | final, bestval | `la1344_r2_s11` | 2026-10-20 | Seed 11. |
| G14 | `gru1344_head256_pall_f160_20k_s10_run2` | final, bestval | `fix_la1344_r2_s10` | 2026-10-27 | G2 trained again after the torch.compile gradient fix of 2026-10-26. |
| G15 | `gru1344_head256_pall_f160_20k_s10_run3` | final, avg, bestval | `snap_la1344_r2_20k_s10` | 2026-10-28 | G2 trained a third time, saving the snapshots for averaging. |

### X: Transformer, width 384, 8 layers, a causal convolution in every block

| ID | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| X1 | `tfm384x8_conv_p5120_f160_8k_s10` | bestval | `tfmp_conv_fix_s10` | 2026-10-26 | Pilot, stopped at 8k of a 20k schedule. |
| X2 | `tfm384x8_conv_p2560_f160_reg_8k_s10` | bestval | `tfmp_convreg_fix_s10` | 2026-10-26 | Pilot with heavier regularisation and a 32-frame window. |
| X3 | `tfm384x8_conv_p5120_f160_60k_s10` | final, bestval | `tfm_conv_fix_r2_60k_s10` | 2026-10-27 | 60k schedule. |
| X4 | `tfm384x8_conv_p5120_f160_20k_s10` | final, avg, bestval | `tfm_conv_r2_20k_s10` | 2026-10-27 | 20k baseline for the three screens below. |
| X5 | `tfm384x8_conv_p5120_f160_ent12_20k_s10` | final, avg, bestval | `tfm_conv_ent12_r2_20k_s10` | 2026-10-27 | Entropy term, weight 0.12. |
| X6 | `tfm384x8_conv_p5120_f160_shared_20k_s10` | final, avg, bestval | `tfm_conv_shared_r2_20k_s10` | 2026-10-27 | One input matrix shared by all sessions. |
| X7 | `tfm384x8_conv_p5120_f160_ictc4_20k_s10` | final, avg, bestval | `tfm_conv_ictc4_r2_20k_s10` | 2026-10-27 | Intermediate CTC loss at layer 4. |

### Y: Transformer, width 384, 5 layers, LightBeam style

| ID | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| Y1 | `tfm384x5_lb_p20000_f1440_20k_s10` | final, bestval | `lb_tfm_r18_20k_s10` | 2026-10-16 | First LightBeam-style run, with our per-session input layers. |
| Y2 | `tfm384x5_lb_p20000_f1440_60k_s10` | final, bestval | `lb_tfm_r18_60k_s10` | 2026-10-16 | 60k schedule. |
| Y3 | `tfm384x5_lb_p20000_f0_noday_63k_s10` | final, avg, bestval | `tfm_lb_r0_63k_s10` | 2026-11-03 | LightBeam reproduction: no per-session layers, step schedule. |
| Y4 | `tfm384x5_lb_p20000_f160_noday_63k_s10` | final, avg, bestval | `tfm_lb_r2_63k_s10` | 2026-11-03 | LightBeam reproduction with 2 frames of look-ahead. |
| Y5 | `tfm384x5_lb_p20000_f160_noday_ls05_63k_s10` | final, avg, bestval | `tfm_lb_ls05_r2_63k_s10` | 2026-11-04 | Adds label smoothing 0.05. |

### Z: Transformer, width 512, 8 layers

| ID | Full name | Weights | Old name | Started | Notes |
|---|---|---|---|---|---|
| Z1 | `tfm512x8_p5120_f0_120k_s10` | bestval | `tfm_d512l8_adamw` | 2026-09-28 | First causal Transformer. |
| Z2 | `tfm512x8_p5120_f0_tmask_120k_s10` | bestval | `tfm_d512l8_tmask` | 2026-09-29 | Time masking with a learned mask token. |

## Planned runs

Prepared in `results/harness/queue/` and not trained. Each takes the next free number of its letter when its
training starts. New tags: `v2` (the 2026-11-11 corrections to the LightBeam reproduction), `crctc` (CR-CTC
consistency loss, weight 0.2), `splice` (phrase splicing), `diph` (diphone targets).

| Letter | Full name | What it tests |
|---|---|---|
| Y | `tfm384x5_lb_p20000_f80_sm80_noday_v2_63k_s10` | Corrected LightBeam reproduction with their centred smoothing and no attention look-ahead |
| Y | `tfm384x5_lb_p20000_f160_noday_v2_63k_s10` | The same with causal smoothing and 160 ms of attention look-ahead |
| Y | `tfm384x5_lb_p20000_f160_noday_v2_crctc_32k_s10` | Adds the CR-CTC consistency loss, at half the batches |
| G | `gru1344_head256_pall_f160_splice_20k_s10` | Phrase splicing on G2's recipe |
| G | `gru1344_head256_pall_f160_diph_20k_s10` | Diphone targets on G2's recipe |
| X | `tfm384x8_conv_p5120_f160_shared_splice_20k_s10` | Phrase splicing on X6's recipe |
| X | `tfm384x8_conv_p5120_f160_shared_diph_20k_s10` | Diphone targets on X6's recipe |
