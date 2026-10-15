# T12 (Willett et al. 2023) as extra training data

`convert_t12.py` rewrites the T12 `competitionData` .mat files in the B2T '25 T15 HDF5 layout,
so T12 sessions can be listed in `dataset.sessions` like T15 days. T12 is **training data
only**. The converter writes no `data_val.hdf5`, and T12 never takes part in validation,
checkpoint selection, or evaluation.

## Running the converter

From the repo root, in the acoustic environment (CPU only, about 1 minute):

    .venv/bin/python -m data_prep.convert_t12 \
        --input-dir data/external/willett2023/extracted/competitionData \
        --out-dir data/hdf5_data_t12 \
        --combined-dir data/hdf5_data_t15_t12 --t15-dir data/hdf5_data_final

- `--out-dir` must not already exist. The converter writes `<session>/data_train.hdf5` for each
  of the 24 sessions, plus `conversion_summary.json` (counts, blocks, any excluded or failed trials).
- `--combined-dir` creates a new directory of relative symlinks to every T15 session in
  `data/hdf5_data_final` and every T12 session. Nothing is written into `hdf5_data_final`.
- `--exclude-sentences FILE` drops trials whose sentence matches a line of FILE. Both sides are
  matched after lowercasing, deleting everything except a-z, 0-9 and whitespace, and collapsing
  whitespace. By default nothing is excluded. An exclusion does not renumber `trial_num`, so
  `bad_trials_dict` entries stay valid.
- g2p_en needs the NLTK data `averaged_perceptron_tagger_eng` and `cmudict` (in `~/.nltk_data`).

## What is written

The layout matches the T15 files: groups `trial_0000…` (contiguous), lzf-compressed datasets
`input_features (T, 512) float32`, `seq_class_ids (500,) int32`, and `transcription (500,) int32`.
The int64 attributes are `block_num`, `n_time_steps`, `seq_len` and `trial_num`; the string
attributes are `session` and `sentence_label`. Two extra provenance attributes record the source:
`source_split` (`train`/`test`) and `source_index` (the row in the .mat file).

- **Features**: tx2 (-4.5 x RMS threshold crossings) for all 256 electrodes, then spikePow for all
  256. This follows the T15 convention of 256 crossings followed by 256 spike-band power columns.
  Electrodes keep T12's own order: 0-127 are area 6v and 128-255 are area 44. The two participants'
  electrodes do not correspond to each other. The day layer learns the input mapping.
- **Normalization**: each feature is z-scored within each block (`blockIdx`), using all labelled
  trials of that block. Zero-variance features become 0, as constant columns are in T15. Values are
  then clipped to ±10.
- **Labels**: `nejm_b2txt_utils.general_utils.sentence_to_phonemes` (g2p_en) mapped to the acoustic
  class order (BLANK=0, AA..ZH=1..39, SIL=40). `transcription` holds the character codes of the
  original `sentence_label`, as in T15. On the 9,498 labelled T15 train and val trials, this path
  reproduces 9,007 phoneme sequences exactly, with a 0.29% phoneme edit rate overall. The
  differences are alternative pronunciations, such as "interesting" and "awful".
- Labelled trials from `train/` and `test/` both go into `data_train.hdf5`. The unlabelled
  `competitionHoldOut` files are never read.

### Filtering compared with T15 (from the papers; the released files are already binned)

Both datasets come from Blackrock headstages that analog-filter at 0.3 Hz-7.5 kHz before 30 kHz digitization, use
linear regression referencing, -4.5 x RMS threshold crossings and mean-squared spike-band power in 20 ms bins.
- **T15:** 250-5000 Hz band-pass (4th-order zero-phase Butterworth); spike-band power clipped at 50,000
  (Wairagkar et al. 2025, same participant and lab).
- **T12, real-time feature pipeline:** non-causal 300 Hz high-pass (4 ms delay), no digital low-pass
  (Willett et al. 2023 supplement). The competitionData README says a 250 Hz cutoff for spikePow. The 250-5000 Hz
  band-pass in that paper applies to its spike-waveform figure, not the features.
- **Effect:** the upper band (5-7.5 kHz) and the 250-300 Hz edge change each channel's power scale and noise mix.
  Per-block z-scoring removes per-channel offset and scale; the per-session input layer absorbs further linear
  differences. Re-filtering is impossible without raw voltages, and the converter does not attempt it.

### Scaling compared with T15 (measured on the released files)

T15 features look **per-block z-scored and then clipped at +10**. Pooled over all trials of each
(session, block), the median |mean| per feature is 0.008 for crossings and 0.012 for SBP, and the
median SD is 1.00. The maximum is exactly 10.0 (6.8e-5 of values at the rail) and the minimum is
-4.2. As brainstorm/audit/AUDIT.md (F1) found, the T15 statistics appear to come from the whole
block, including periods that were not released. T12 provides only go-period snippets, so its
statistics use only the trials. The converted T12 blocks therefore have mean 0 and SD 1 exactly,
and slightly more values reach the rail (1.9e-4). Both datasets use the same family of
normalization, and both are non-causal at block scale. If the causal-normalizer work changes the
T15 inputs, regenerate T12 to match (`zscore_by_block`).

## Training configurations

The trainer already handles sessions without `data_val.hdf5`: they contribute no validation
trials. Four T15 days already work this way. Every T12 session must have
`dataset_probability_val = 0`. Session sampling stays uniform over listed sessions (the default
`days_per_batch` sampler).

### (a) T12-only pretraining

With no validation data, validation reports NaN. `rnn_trainer.validation` previously raised
ZeroDivisionError in this case. A NaN PER never counts as a new best, so `best_checkpoint` is never
written. Save the final model instead, and add `checkpoint_batches` if you want intermediate
snapshots. The harness `train` command cannot run this configuration, because its smoke and
registration require a finite val PER and a `best_checkpoint`. Write a config and run the trainer
directly, using a fresh output directory:

    cd model_training
    ../.venv/bin/python - <<'EOF'
    from omegaconf import OmegaConf
    c = OmegaConf.load('../results/causal_la0/checkpoint/args.yaml')
    c.dataset.dataset_dir = '../data/hdf5_data_t12'
    c.dataset.sessions = ['t12.2022.04.28','t12.2022.05.05','t12.2022.05.17','t12.2022.05.19','t12.2022.05.24',
        't12.2022.05.26','t12.2022.06.02','t12.2022.06.07','t12.2022.06.14','t12.2022.06.16','t12.2022.06.21',
        't12.2022.06.23','t12.2022.06.28','t12.2022.07.05','t12.2022.07.14','t12.2022.07.21','t12.2022.07.27',
        't12.2022.07.29','t12.2022.08.02','t12.2022.08.11','t12.2022.08.13','t12.2022.08.18','t12.2022.08.23',
        't12.2022.08.25']
    c.dataset.dataset_probability_val = [0] * 24
    c.save_best_checkpoint = False
    c.save_final_model = True
    c.output_dir = 'trained_models/<fresh_name>'
    c.checkpoint_dir = 'trained_models/<fresh_name>/checkpoint'
    OmegaConf.save(c, 'trained_models/<fresh_name>.yaml')
    EOF
    ../.venv/bin/python train_model.py trained_models/<fresh_name>.yaml

As dotlist overrides:

    dataset.dataset_dir=../data/hdf5_data_t12
    dataset.sessions=[t12.2022.04.28,t12.2022.05.05,t12.2022.05.17,t12.2022.05.19,t12.2022.05.24,t12.2022.05.26,t12.2022.06.02,t12.2022.06.07,t12.2022.06.14,t12.2022.06.16,t12.2022.06.21,t12.2022.06.23,t12.2022.06.28,t12.2022.07.05,t12.2022.07.14,t12.2022.07.21,t12.2022.07.27,t12.2022.07.29,t12.2022.08.02,t12.2022.08.11,t12.2022.08.13,t12.2022.08.18,t12.2022.08.23,t12.2022.08.25]
    dataset.dataset_probability_val=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    save_best_checkpoint=false
    save_final_model=true

Choose `num_training_batches` and the LR schedule explicitly. The trainer saves the final
checkpoint as `checkpoint/final_checkpoint_batch_<N-1>`.

### (b) Joint T15 + T12 training

Keep the 45 T15 sessions in their existing order and append the 24 T12 sessions. T15 day indices
stay 0-44, which matters for loading checkpoints and for harness benchmarking. Checkpoint selection
is unchanged: the best val PER over the enabled T15 val sessions (val-dev and the exposed former
val-test).

    .venv/bin/python -m harness train --from causal_la0 --name <name> --question "<question>" \
      --data-dir data/hdf5_data_t15_t12 \
      --set 'dataset.sessions=[t15.2023.08.11,t15.2023.08.13,t15.2023.08.18,t15.2023.08.20,t15.2023.08.25,t15.2023.08.27,t15.2023.09.01,t15.2023.09.03,t15.2023.09.24,t15.2023.09.29,t15.2023.10.01,t15.2023.10.06,t15.2023.10.08,t15.2023.10.13,t15.2023.10.15,t15.2023.10.20,t15.2023.10.22,t15.2023.11.03,t15.2023.11.04,t15.2023.11.17,t15.2023.11.19,t15.2023.11.26,t15.2023.12.03,t15.2023.12.08,t15.2023.12.10,t15.2023.12.17,t15.2023.12.29,t15.2024.02.25,t15.2024.03.03,t15.2024.03.08,t15.2024.03.15,t15.2024.03.17,t15.2024.04.25,t15.2024.04.28,t15.2024.05.10,t15.2024.06.14,t15.2024.07.19,t15.2024.07.21,t15.2024.07.28,t15.2025.01.10,t15.2025.01.12,t15.2025.03.14,t15.2025.03.16,t15.2025.03.30,t15.2025.04.13,t12.2022.04.28,t12.2022.05.05,t12.2022.05.17,t12.2022.05.19,t12.2022.05.24,t12.2022.05.26,t12.2022.06.02,t12.2022.06.07,t12.2022.06.14,t12.2022.06.16,t12.2022.06.21,t12.2022.06.23,t12.2022.06.28,t12.2022.07.05,t12.2022.07.14,t12.2022.07.21,t12.2022.07.27,t12.2022.07.29,t12.2022.08.02,t12.2022.08.11,t12.2022.08.13,t12.2022.08.18,t12.2022.08.23,t12.2022.08.25]' \
      --set 'dataset.dataset_probability_val=[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'

`--data-dir` sets `dataset.dataset_dir`; it is new and defaults to `data/hdf5_data_final`. The
smoke's streaming check still uses the T15 val-dev sample. With 69 days and `days_per_batch=4`,
about 24/69 of the sampled day slots are T12. The model gets 69 day layers, which adds 24 × 263k
parameters.

## Day-specific input layers when transferring T12 → T15

- Each listed session has one input layer: a 512×512 `day_weights.<i>` plus `day_biases.<i>`
  followed by softsign. The index is the session's position in `dataset.sessions`. A T12-only
  checkpoint has 24 of these layers, while a T15 config needs 45 different ones. The GRU and output
  layer are shared and transferable. The day layers are not, because the electrodes, arrays and
  brain areas differ (T12: 6v and 44; T15: 6v, 4 and 55b).
- The existing `init_from_checkpoint` path cannot do this transfer. It calls a strict
  `load_state_dict`, so a different day-layer count fails. It also restores the optimizer and
  scheduler, which would resume the old LR schedule, and `best_val_PER`. For a T12-only run that
  value is NaN, after which no checkpoint would ever count as a new best. The checkpoint keys also
  carry torch.compile's `_orig_mod.` prefix.
- Fine-tuning therefore needs a weights-only partial initialization, which is not implemented
  here. It would load every parameter except `day_*`, initialize the T15 day layers as in a fresh
  run (identity and zero) or from the mean of the T12 day layers, create a fresh
  optimizer/scheduler, and set `best_val_PER = inf`.
  `model.rnn_trainable` / `model.input_network.input_trainable` already allow a staged schedule:
  train the new day layers with the GRU frozen, then unfreeze it.
- Joint training (b) avoids the transfer problem, because every session keeps its own layer in one
  model. Fine-tuning a joint checkpoint on T15 alone means changing the session list, which runs
  into the same partial-load requirement.
