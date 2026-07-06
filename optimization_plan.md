# Optimization Plan: Causal GRU Training and Real-Time Acoustic Decoding

Date: 2026-07-06

## Goal

Optimize the causal GRU acoustic model so the pipeline can run approximately in real time while preserving or improving aggregate validation PER.

This plan treats the acoustic GRU path as the real-time target. The Redis/ngram/LLM rescoring path is sentence-final and should not be included in acoustic real-time latency measurements unless the product target requires final text latency rather than streaming phoneme/acoustic latency.

Assumptions from the current code:

- Neural bins are treated as 20 ms bins in `causal_audit.md`.
- Current model defaults are `n_layers=5`, `n_units=768`, `patch_size=14`, `patch_stride=4`, `smooth_kernel_std=2`, and `smooth_lookahead=0`.
- `patch_stride=4` means the acoustic model emits one GRU output every 80 ms.
- Current non-left-padded patching is causal but adds up to 13 bins / 260 ms of emission latency.
- The new smoothing control is `smooth_lookahead`; `0` is real-time causal and `4` is the symmetric baseline/control for the current kernel.

## Success Metrics

Track these for every run:

- Aggregate validation PER from the run's own validation metrics.
- Best validation batch and wall-clock training time.
- Training throughput: batches/sec and neural bins/sec.
- Acoustic inference real-time factor (RTF): processing time divided by input audio/neural duration.
- Streaming compute latency: p50/p95 wall time per new bin and per emitted patch.
- Algorithmic latency: `smooth_lookahead * 20 ms` plus patch emission latency.
- Output cadence: 80 ms with the current `patch_stride=4`.

Initial targets:

- Control sanity: `smooth_lookahead=4` should be near the prior ~10.25% aggregate val PER.
- Real-time frontier: `smooth_lookahead=0` should be measured before adding any other accuracy changes.
- Streaming acoustic RTF should be comfortably below 1.0, ideally below 0.25 on the target machine.
- If left-padded patching is adopted, algorithmic acoustic latency should drop by ~260 ms without adding future dependence.

## Phase 0: Measurement Harness

Create repeatable measurement before tuning.

1. Add a lightweight benchmark script for acoustic-only inference.
   - Load a trained checkpoint and `checkpoint/args.yaml`.
   - Run validation trials without Redis or LM.
   - Measure smoothing, patch/day-layer/GRU/head, greedy CTC collapse, and total acoustic time separately.
   - Save `metrics.json` beside each run.

2. Add a streaming benchmark path.
   - Feed one neural bin at a time.
   - Maintain the smoothing ring buffer, patch ring buffer, GRU hidden state, and CTC collapse state.
   - Run the GRU only when a new patch is emitted, not on every raw bin.
   - Report p50/p95 compute latency per bin and per emitted patch.

3. Standardize run directories.
   - Use names like `trained_models/rnn_lah4_control`, `trained_models/rnn_lah0_realtime`, and `trained_models/rnn_lah0_leftpad_patch`.
   - Always change both `output_dir` and `checkpoint_dir`.
   - Never compare models by shared or overwritten directories.

## Phase 1: Establish Correct Baselines

Run the smoothing lookahead sequence before changing anything else.

1. Control run: `smooth_lookahead: 4`.
   - Expected behavior: symmetric smoothing baseline.
   - Decision gate: if aggregate val PER is not near ~10.25%, stop and diagnose environment/data/config drift.

2. Real-time run: `smooth_lookahead: 0`.
   - Expected behavior: no future smoothing taps.
   - This is the first true real-time acoustic frontier point.

3. Optional interpolation: `smooth_lookahead: 1`, `2`, `3`.
   - Only run if the `0` versus `4` gap is meaningful.
   - Each lookahead bin costs 20 ms of future latency.

4. Confirm the impulse unit check inside the actual ML environment.
   - `lookahead=0`: zero output before impulse bin, peak at impulse bin.
   - `lookahead=4`: symmetric control, matching the old centered smoother.

## Phase 2: No-Risk Training Throughput Fixes

These should not change model math or PER except through run-to-run noise.

1. Data loading.
   - The dataset currently opens HDF5 files inside `__getitem__` for every batch/day.
   - Add an opt-in RAM cache or persistent per-worker HDF5 handles.
   - Use `persistent_workers=True` and tune `prefetch_factor` for the training DataLoader.
   - Keep validation deterministic and simple; optimize it only if validation time dominates.

2. Device transfer.
   - Use `.to(self.device, non_blocking=True)` for tensors loaded from pinned memory.
   - Keep `pin_memory=True` for CUDA training.

3. Optimizer step.
   - Change `self.optimizer.zero_grad()` to `self.optimizer.zero_grad(set_to_none=True)`.
   - Keep fused AdamW on CUDA; fall back explicitly if a non-CUDA environment cannot support `fused=True`.

4. Gaussian smoothing overhead.
   - `gauss_smooth` rebuilds and repeats the kernel every call.
   - Cache the Gaussian kernel per `(device, channels, smooth_kernel_std, smooth_kernel_size, lookahead)` or move it into the trainer as a reusable tensor.
   - For streaming inference, replace full-sequence `conv1d` with a fixed FIR ring buffer.

5. Validation overhead.
   - For hyperparameter search, set `save_val_logits: false` unless logits are needed.
   - Consider `log_individual_day_val_PER: false` for screening runs.
   - Use a fixed validation subset for quick screening, then full validation for promoted runs.
   - Keep final model selection on full aggregate validation PER.

6. Compile and cuDNN comparison.
   - Current code always calls `torch.compile`.
   - Benchmark `torch.compile` on/off for this GRU workload; cuDNN RNN kernels sometimes gain little from compile but pay startup cost.
   - Keep the faster option per environment, controlled by a config flag.

## Phase 3: Reduce Algorithmic Latency

This phase changes what the GRU sees and requires retraining.

1. Implement causal left-padded patching.
   - Current patch `i` covers `[i*stride, i*stride+patch_size)`, which is causal under right-edge emission but delays the output by `patch_size-1` bins.
   - Left-pad by `patch_size-1` before `unfold` so patch `i` covers `[i*stride-(patch_size-1), i*stride]`.
   - Update the train and validation `adjusted_lens` formula from `floor((T - patch_size) / stride) + 1` to `floor((T - 1) / stride) + 1`.

2. Train a direct comparison.
   - Baseline: `smooth_lookahead=0`, current patching.
   - Candidate: `smooth_lookahead=0`, left-padded patching.
   - Decision gate: keep left-padded patching unless PER regression is large, because it removes ~260 ms of acoustic emission latency.

3. Streaming implementation.
   - Cache selected day weight/bias for the active session.
   - Maintain GRU hidden state across emitted patches.
   - Maintain CTC last-token state so duplicate collapse can happen online.
   - Reset state only on utterance/session boundaries as required by the product behavior.

## Phase 4: Recover or Improve PER Under Causal Constraints

Run small, structured sweeps. Promote only configurations that improve the real-time frontier or materially reduce compute at comparable PER.

1. Smoothing and past context.
   - `smooth_kernel_std`: try `1.5`, `2.0`, `2.5`, `3.0` with `smooth_lookahead=0`.
   - Larger std with zero lookahead adds past context without future leakage.
   - Compare against `smooth_lookahead=1..3` only if small future latency is acceptable.

2. Patch geometry.
   - Sweep `patch_size`: `8`, `10`, `12`, `14`, `18`.
   - Sweep `patch_stride`: `2`, `4`.
   - Tradeoff: smaller stride improves output cadence but increases GRU steps; larger patch improves context but increases input size and compute.
   - Keep left-padded patching for real-time candidates.

3. GRU capacity.
   - Current model is 5 layers x 768 units.
   - Test compact candidates: 3x512, 4x512, 4x768.
   - Test accuracy candidates: 5x896 or 5x1024 only if training throughput and inference budget allow.
   - Track params, FLOPs proxy, and streaming p95 latency for every candidate.

4. Regularization and augmentation.
   - Tune `rnn_dropout`: `0.2`, `0.3`, `0.4`.
   - Tune `input_layer_dropout`: `0.1`, `0.2`, `0.3`.
   - Tune `white_noise_std`: `0.5`, `1.0`, `1.5`.
   - Keep `random_cut` compatible with streaming boundary behavior.

5. Learning schedule.
   - Keep cosine as the baseline.
   - Try shorter screening runs with early stopping enabled.
   - Promote promising configs to full `num_training_batches=120000`.
   - If full training is slow, use ASHA-style promotion: short run, medium run, full run.

6. Day-layer behavior.
   - Monitor per-day PER, not only aggregate PER.
   - If day-specific layers overfit or update noisily, tune `days_per_batch`, `lr_max_day`, and `weight_decay_day`.
   - Consider freezing day layers after convergence for final fine-tuning, but only if per-day PER improves.

## Phase 5: Runtime Optimization for the Final Candidate

Once a candidate has acceptable PER, optimize deployment behavior.

1. Acoustic-only realtime path.
   - Use `torch.inference_mode()`.
   - Use `model.eval()`.
   - Keep dropout inactive.
   - Use bf16/fp16 autocast only if it improves latency without changing PER materially.

2. Avoid sequence recomputation.
   - Do not rerun `gauss_smooth` over the entire growing sequence online.
   - Do not rerun the GRU over the entire growing sequence online.
   - Feed only the newest emitted patch plus the previous hidden state.

3. Cache static tensors.
   - Day index tensor.
   - Day weight and bias for the active session.
   - Smoothing taps.
   - Patch buffer layout.

4. Export path.
   - First target a clean PyTorch streaming module.
   - Then benchmark TorchScript or ONNX only if PyTorch overhead is the bottleneck.
   - Dynamic quantization is a CPU-only option to test if CPU realtime is required; otherwise prioritize CUDA/cuDNN.

## Phase 6: Final Acceptance

A final candidate is ready only when all of these are true:

- It was trained with saved args that include `smooth_lookahead`.
- Offline validation and streaming validation produce matching greedy CTC sequences for the same checkpoint, allowing for expected boundary handling.
- Aggregate validation PER is reported from the run's own `val_metrics.pkl` or training log.
- Acoustic RTF is below target on the deployment machine.
- Algorithmic latency is explicitly reported as smoothing lookahead plus patch latency.
- Redis/LM is either excluded from realtime claims or measured separately as sentence-final latency.

## Recommended First Runs

Run these in order:

1. `rnn_lah4_control`: `smooth_lookahead=4`, current patching.
2. `rnn_lah0_realtime`: `smooth_lookahead=0`, current patching.
3. `rnn_lah0_leftpad_patch`: `smooth_lookahead=0`, causal left-padded patching plus updated length formula.
4. `rnn_lah0_leftpad_patch_fastval`: same as #3 with training-throughput changes and reduced validation overhead for sweep screening.
5. Compact-model sweep from the best causal setup: 3x512, 4x512, 4x768.

Decision rule:

- If #2 is close to #1, optimize latency and throughput first.
- If #2 regresses materially, run `smooth_lookahead=1..3` and past-only smoothing/std sweeps.
- If #3 keeps PER close to #2, make left-padded patching the real-time default.
- If compact models keep PER close while reducing p95 streaming latency, prefer the smallest model that meets PER and latency targets.
