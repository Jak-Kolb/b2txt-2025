# Storage cleanup, 2026-11-24

The repository folder had grown to 460 GiB: 149 GiB of harness training and evaluation runs, 140 GiB of checkpoints
and 127 GiB of datasets. This cleanup deleted 130.9 GiB of files that nothing loads and that hold no result. Every file
deleted is listed with its size and modification time in `storage_cleanup_2026-11-24_files.tsv` beside this file.
All of them were in git-ignored folders; no tracked file was touched.

## What was deleted

| Group | Files | GiB | What it was |
|---|---:|---:|---|
| Smoke-test weights | 396 | 109.6 | `results/harness/train/*/smoke_model/checkpoint/{best_checkpoint, checkpoint_batch_*, val_metrics.pkl}`. Every `harness train` run first trains 50 batches to check that the launch works and keeps that model (about 1.5 GiB). Nothing reads it after the smoke step. |
| Duplicate dataset archives | 2 | 7.1 | `data/external/kunz2025/sentenceData.zip` and `data/external/willett2023/competitionData.tar.gz`. Both were verified identical to their extracted folders before deletion: every zip member's CRC-32 and every tar member's SHA-256 matched the extracted file (34 and 63 files). |
| Weights of unfinished, unregistered runs | 43 | 14.3 | Checkpoints and `val_metrics.pkl` in the 21 `model_training/trained_models/` folders below: runs that were killed, crashed or failed before their last batch, plus one dry run. None is in the registry, and no evaluation run, rescoring, live evaluation, figure or `current_knowledge.md` result uses them. |

## What was kept

- In every harness training folder: `config.yaml`, `request.json`, `state.json`, `smoke.log`, `train.log`,
  `git_diff.patch`, and `smoke_model/{training_log, train_val_trials.json, checkpoint/args.yaml}`. Registry entries
  point at these `config.yaml` files.
- In every unfinished run's folder: `training_log`, `train_val_trials.json` and `checkpoint/args.yaml`, so its
  validation curve and configuration remain.
- The extracted dataset folders, the Dryad manifests and the READMEs.
- Every file a registered model loads.

## Getting something back

- Smoke weights: any `harness train` run makes them again; the smoke result itself is in each folder's `smoke.log`.
- `sentenceData.zip` (Kunz et al. 2025): Dryad DOI 10.5061/dryad.gf1vhhn1j, SHA-256
  `a92c1e47f87858c1f17f9175c474eefaf78b2988ec513c8376eb1adfd26945c6` (matches `data/external/kunz2025/dryad_manifest.json`).
- `competitionData.tar.gz` (Willett et al. 2023): Dryad, see `data/external/willett2023/README.md`; SHA-256
  `5b5c9784d80d5286e69352621366be2e4ff468e043e082e7aa7cbc90d18336b3`.
- Unfinished runs: rerun from the harness training folder's `config.yaml`, or from `checkpoint/args.yaml` for the two
  August runs.

## The 21 unfinished runs

"Best val PER" is the lowest validation phoneme error rate in the run's training log, on the trainer's validation set
during training. These checkpoints were never scored.

| Folder in `model_training/trained_models/` | Stopped at / planned batches | Best val PER | Harness state | Replaced by |
|---|---|---:|---|---|
| `_aborted_causal_la0_retrain_24k` | 24,400 / 120,000 | 0.117 | (before the harness) | Aborted retrain of A2 (August); `docs/history/log_2026-08.md` |
| `_crashed_rd1a_83k` | 83,000 / 120,000 | 0.104 | (before the harness) | `rd1a_bwmatched` (A5). Kept in August as "the partial that first suggested" the R-D1a result (`docs/history/PLAN_2026-08.md`); its logits remain in `results/rd1a_logits.npz` and its sweeps in `results/rd1a_sweep*.json` |
| `dryrun_queue_20261011_20261011T164641Z` | 0 / 200 | 1.000 | registered (dry run) | Queue dry run; its two 30-trial evaluation runs in `results/harness/runs/` remain |
| `g2_la1344_r2_60k_s10_20261018T175120Z` | 2,400 / 60,000 | 0.241 | training (killed) | - |
| `g2_la_r18_120k_s10_20261015T200054Z` | 9,200 / 120,000 | 0.131 | training (killed) | - |
| `gru1344_head256_pall_f160_splice_20k_s10_20261111T085602Z` | 6,800 / 20,000 | 0.135 | training (killed) | `..._20261111T163353Z` (`gru1344_head256_pall_f160_splice_20k_s10`) |
| `kd_la1344_r2_20k_s10_20261019T050440Z` | 800 / 20,000 | - | training (killed) | `..._20261019T121825Z` (`gru1344_head256_pall_f160_kd50_20k_s10`) |
| `la0_seed11_30k_20261008T012418Z` | 1,600 / 30,000 | - | failed | `..._20261008T125326Z` (`gru768_pall_f0_30k_s11`) |
| `la1344_r2_60k_s10_20261018T180342Z` | 8,600 / 60,000 | 0.126 | failed | `..._20261018T190706Z` (`gru1344_head256_pall_f160_60k_s10`) |
| `la1344_r2_60k_s10_20261018T184459Z` | 3,400 / 60,000 | 0.220 | failed | `..._20261018T190706Z` (`gru1344_head256_pall_f160_60k_s10`) |
| `la_r18_s11_20261014T055611Z` | 6,000 / 20,000 | 0.142 | failed | - |
| `la_r18_s11_20261014T062230Z` | 1,600 / 20,000 | - | training (killed) | - |
| `la_r6_s10_20261012T194204Z` | 10,000 / 20,000 | 0.122 | failed | `..._20261012T201817Z` (`gru768_head256_pall_f480_20k_s10`) |
| `lapal_raw_r18_s10_20261014T014201Z` | 6,000 / 20,000 | 0.178 | failed | `..._20261014T021619Z` (`gru768_head256_pall_f1440_rawin_pal_20k_s10`) |
| `snap_la1344_r2_20k_s10_20261028T163425Z` | 8,000 / 20,000 | 0.126 | failed | `..._20261028T171947Z` (`gru1344_head256_pall_f160_20k_s10_run3`) |
| `tfm384x5_lb_conv_p20000_f80_sm80_noday_v2_63k_s11_20261121T125823Z` | 26,400 / 63,000 | 0.335 | training (killed) | - |
| `tfm384x5_lb_p20000_f160_noday_v2_63k_s11_20261116T183410Z` | 8,000 / 63,000 | 0.161 | training (killed) | `..._20261117T063511Z` (`tfm384x5_lb_p20000_f160_noday_v2_63k_s11`) |
| `tfm384x5_lb_p20000_f80_sm80_noday_v2_sf4_63k_s10_20261118T232434Z` | 22,600 / 63,000 | 0.119 | training (killed) | `..._20261119T002852Z` (`tfm384x5_lb_p20000_f80_sm80_noday_v2_sf4_63k_s10`) |
| `tfmp_conf_s10_20261026T221324Z` | 2,000 / 8,001 | 0.746 | training (killed) | - (Conformer pilot that stayed on the blank plateau; `docs/history/transformer_pilots_2026-10-26.md`) |
| `tfmp_conv_s10_20261026T224420Z` | 7,800 / 8,001 | 0.133 | failed | - (pre-fix Transformer pilot, died of a CUDA error; its curve is in the same file, and its recipe lives on as `PILOTS["tfmp_conv_s10"]` in the queue scripts) |
| `voice_harm_la_r18_s10_20261015T182659Z` | 18,400 / 20,000 | 0.112 | failed | - |

"training (killed)" means the supervisor stopped without updating `state.json`.

## Not done (left for a later decision)

| GiB | What | What deleting it would lose |
|---:|---|---|
| 40.9 | Intermediate `checkpoint_batch_*` snapshots already averaged into a saved `snapshot_avg`, and 55 of `g2_masking`'s (A3_1) 61 every-2,000-batch checkpoints (its last five and final would stay) | Building a different average later |
| 24.4 | `output_trace.jsonl` of 442 acoustic-scale sweep points that no code, figure or result reads | Rerunning the live-word evaluation at those scales without decoding again (`bench` regenerates a trace from the logits cache) |
| 9.2 | `val_metrics.pkl` in registered runs (validation logits of the best-validation checkpoint) | Analyses that read those logits; never used for scoring |
| 24.4 | `data/external/wairagkar2025/t15_neural_data.zip` | 63 files that were never extracted (2.3 GB: `.mat` files, the release's Keras `.h5` models, macOS metadata); Dryad DOI 10.5061/dryad.2280gb64f |
| 8.0 | Language-model build inputs: `results/lm_general_4gram/*.arpa`, `results/corpus/*.txt`, `results/lm_mixed/lm_mixed.arpa` | Rebuilding a language model without regenerating them |
