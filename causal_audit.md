# Causal Audit

| Row | Current behavior | Causal? | Fix needed |
|-----|-----------------|---------|------------|
| **Smoothing — train path** | `gauss_smooth` (`data_augmentations.py:6`), called from `rnn_trainer.py:476` with the default `padding='same'`. Kernel is nominally `smooth_kernel_size=100` but the `>0.01` trim (`data_augmentations.py:22`) shrinks it to a **9-bin** symmetric Gaussian (`std=2`, peak at ±0, ±4 bins each side). `'same'` leaks **4 bins (80ms) of future** into every smoothed sample. | **N** | Expose a `lookahead` arg on `gauss_smooth` and run this path with `lookahead=0` (keep offsets `-p..0`, renormalize, pad `(p,0)`, unpadded grouped `conv1d`; length-preserving). See Update 2026-07-06. |
| **Smoothing — real-time / eval path** | `runSingleDecodingStep` (`evaluate_model_helpers.py:87-98`) calls `gauss_smooth(..., padding='valid')`. **CORRECTION from prior audit: `'valid'` is NOT causal.** The kernel is still the symmetric 9-bin Gaussian; `'valid'` only trims edge samples and shifts indices. Verified empirically: an impulse at bin `t` still contributes to output samples representing times before `t` (center offset +4). So the eval path leaks **~4 bins (80ms) of future** too. | **N** | Same fix: `lookahead=0`, applied identically here. |
| **Train/inference mismatch** | Two separate problems: (a) *both* paths are acausal (above), and (b) they use *different* padding (`'same'` vs `'valid'`), so `'valid'` also yields a shorter, index-shifted sequence than training saw. The reported ~10.2% PER is therefore measured through a non-causal, train-mismatched path. | **N — bug + non-causal** | A single `lookahead=0` setting (length-preserving, backward-only), applied identically in train and eval — structurally enforced via the saved model args — fixes both at once. |
| **Patching (`unfold`)** | `rnn_model.py:112`: `x.unfold(3, patch_size, patch_stride)` with `patch_size=14`, `patch_stride=4`, no left-pad. Patch `i` covers bins `[i*stride, i*stride+patch_size)`; max index touched is `i*stride+patch_size−1`, so with a **right-edge emission convention this has NO future dependence — it is already causal.** The cost is *latency*: each output summarizes a 280ms window that leads its left edge, so the phoneme for activity at bin `i*stride` isn't emitted until 13 bins (260ms) later. Runs identically in train and eval (no mismatch). | **Y (causal), but 260ms latency** | *Optional, for latency only:* left-pad `patch_size−1` before `unfold` so each window sits entirely in the past. Requires retrain **and** updating the `adjusted_lens` formula (`rnn_trainer.py:532,707`) since it changes `num_patches`. |
| **Upstream normalization** | No per-feature mean/std step in this repo; features arrive pre-extracted in the `.hdf5` files. **Tested empirically** (`analyses/check_hdf5_causality.py`, all 45 sessions): features are **not** per-trial z-scored (per-trial max\|mean\|≈0.40) and do **not** match a whole-block z-score (per-block max\|mean\|≈0.10–0.15, max\|std−1\|≈0.1–0.3 — closer to standardized than trial-scale, but not the ≈0/≈1 a full-block z-score would produce). So the specific future-leak the check targets is **absent**. | **Likely OK (not confirmed)** | No whole-block leak detected. Stats operate at ~block scale or broader in a way consistent with a *causal running/adaptive* estimate or an *external reference block* — both causal-safe — but output alone can't prove it. Confirm against NEJM methods/upstream extraction code before relying on it for real-time. |
| **Day-specific input layer** | `rnn_model.py:94-99`: per-timestep affine (`einsum` over feature dim only) + `Softsign`. No mixing across time. | **Y** | None. |
| **GRU** | `rnn_model.py:65-72`: `bidirectional=False`, fed sequentially. Purely causal by construction. | **Y** | None. |
| **Output head / CTC greedy decode** | Per-timestep linear layer; greedy argmax decode has no look-ahead. | **Y** | None — streamable as-is. |
| **n-gram + LLM rescoring (remote LM)** | `evaluate_model_helpers.py:129-296`: `finalize_remote_lm` rescoring requires the full utterance's logits. This is *sentence-final* look-ahead, not per-frame — acceptable for a real-time system that streams partial greedy/n-gram hypotheses and only rescores at utterance end. | **Y** (by design) | None, as long as partial output isn't blocked on it. |

## Bottom line

The **only genuine future-leak in the model/feature path is the smoothing kernel** (symmetric Gaussian, leaking ~4 bins / 80ms in both train and eval — `'valid'` does *not* fix it). Everything else — day layer, GRU, head, greedy decode — is already causal. Patching is causal too but carries 260ms of *latency* that's optional to reduce. Upstream normalization is the one unknown.

## Step 1 — IMPLEMENTED (2026-07-03) — SUPERSEDED by Update 2026-07-06

> **Superseded.** The `padding='causal'` branch described below was replaced by the configurable `lookahead` argument (see Update 2026-07-06). `lookahead=0` is the exact equivalent of the causal branch here; `lookahead=4` reproduces the symmetric baseline. The problem analysis still holds — only the mechanism and call sites changed. Kept for history.

Smoothing is now causal and identical across train and eval. Three edits:

**1. `data_augmentations.py` (`gauss_smooth`, ~L35)** — added a `causal` branch:
```python
    if padding == 'causal':
        # left-pad by (K_eff - 1) so output[t] depends only on inputs <= t (no future leak)
        inputs = F.pad(inputs, (gaussKernel.shape[-1] - 1, 0))
        smoothed = F.conv1d(inputs, gaussKernel, padding=0, groups=C)
    else:
        smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
```
*Why:* the kernel is a symmetric 9-bin Gaussian. `'same'` centers it (±4 bins → 80ms future leak); `'valid'` only trims edges and keeps the same forward-looking center (still leaks). Left-padding `K_eff−1=8` zeros with `padding=0` shifts the entire kernel into the past, so `output[t]` is a weighted average of `inputs[t−8 … t]` only. Length is preserved (verified: 30→30; impulse@10 affects only outputs ≥10), so no downstream `adjusted_lens` changes are needed.

**2. `rnn_trainer.py:476`** — added `padding = 'causal'` to the training-time `gauss_smooth` call (was defaulting to `'same'`).
*Why:* training was smoothing with 80ms of future leak; this removes it.

**3. `evaluate_model_helpers.py:92`** — changed `padding = 'valid'` → `'causal'`.
*Why:* `'valid'` was not causal (corrected finding above) and also produced a shorter, index-shifted sequence than training saw. Now train and eval use the identical backward-only kernel, closing both the non-causality and the train/eval mismatch in one change.

**Net effect:** the model/feature path is now fully causal. Requires one retrain; measure the new causal PER against the 10.2% reference. Patch latency (Step 2) is still open.

## Plan (ordered; only step 1 is required for causality)

1. **[REQUIRED] Make smoothing causal.** *(Implemented via the `lookahead` arg — see Update 2026-07-06.)* Expose a `lookahead` argument on `gauss_smooth` (keep offsets `-p..+L`, renormalize, pad `(p,L)`, unpadded grouped `conv1d`; length-preserving) driven by `smooth_lookahead` in `rnn_args.yaml` and read by *both* `rnn_trainer.py` and `evaluate_model_helpers.py`. Set `smooth_lookahead: 0` for the causal/real-time path. Single knob, no length-formula edits, fixes both the non-causality and the train/eval mismatch. Requires one retrain.
2. **[OPTIONAL — latency] Left-pad the patch window.** See the copy-pasteable Step 2 below. Cuts ~260ms of emission latency; does not change causality. Bundle into the same retrain as Step 1 if you want it.
3. **Confirm baseline first, then retrain** (per `CLAUDE.md`). Measure PER on the now-causal `evaluate_model.py` path vs the current 10.2% reference. A small regression from losing the 80ms of smoothing look-ahead is expected; quantify it. To recover accuracy for free, backward context is cheap in a causal model — consider a slightly larger `smooth_kernel_std` or `patch_size` as a tuning lever.
4. **Startup transient.** First `~K_eff−1` (+ `patch_size−1` if Step 2) bins of a stream have zero-padded history. Training already zero-pads at trial boundaries, so train/eval agree if the online decoder also zero-pads the very first bins (only the first utterance of a continuous session pays it).
5. **Upstream normalization — tested, no leak found.** `analyses/check_hdf5_causality.py` regathers each block's trials across the train/val/test splits and checks for the whole-block z-score signature. Result: 0/45 sessions match, so the released features are not full-block z-scored. Remaining residual (block-scale stats near but not exactly 0/1) is consistent with a causal running estimate or an external reference block. Still worth a one-line confirmation against the NEJM methods; if it turns out non-causal, it needs a Welford running estimate at inference or a re-extraction.

## Step 2 — NOT YET APPLIED (latency reduction, ~260ms)

Only needed to reduce emission latency; the current `unfold` is already free of future dependence. Requires a retrain (changes what the GRU is fed) **and** a length-formula update because left-padding increases `num_patches`. Three edits:

**1. `rnn_model.py` imports (top of file)** — add:
```python
import torch.nn.functional as F
```

**2. `rnn_model.py` (`forward`, in the `if self.patch_size > 0:` block, before `unfold`)** — insert the pad:
```python
            x = x.unsqueeze(1)                      # [B, 1, T, D]
            x = x.permute(0, 3, 1, 2)               # [B, D, 1, T]

            x = F.pad(x, (self.patch_size - 1, 0))  # causal left-pad: each window sits entirely in the past

            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
```
*Why:* without the pad, patch `i` spans bins `[i*stride, i*stride+patch_size)`, so the phoneme for activity at bin `i*stride` isn't emitted until 13 bins (260ms) later. Padding `patch_size−1` zeros on the left makes patch `i` span `[i*stride−(patch_size−1), i*stride]` — the same 14-bin context, but entirely in the past, emitted as soon as bin `i*stride` arrives.

**3. `rnn_trainer.py:532` AND `:707`** — update the CTC length formula (both the train and val copies). Left-padding by `patch_size−1` changes the patch count from `floor((T − patch_size)/stride) + 1` to `floor((T − 1)/stride) + 1`:
```python
                adjusted_lens = ((n_time_steps - 1) / self.args['model']['patch_stride'] + 1).to(torch.int32)
```
*Why:* `adjusted_lens` must equal the model's output time dimension for CTC to align correctly. If you skip this, the loss/decode lengths will be wrong.

**Note (streaming vs training boundary):** training zero-pads the left edge. In the online decoder you can instead feed *real* preceding history across trial boundaries so only the very first bins of a session are zero-padded — but keep training and the offline `evaluate_model.py` consistent (both zero-pad) so the reported PER matches what the model was trained on.

## Update 2026-07-06 — configurable smoothing lookahead

This supersedes the Step 1 implementation detail above. The smoother no longer uses a special `padding='causal'` branch at the call sites. It now exposes an explicit `lookahead` argument:

- `model_training/data_augmentations.py`: `gauss_smooth(..., padding='same', lookahead=None)`.
- `lookahead=None`: preserves the original symmetric behavior and honors `padding`.
- `lookahead=L`: keeps kernel offsets `-p..+L`, renormalizes them, pads `(p, L)`, and runs an unpadded grouped `conv1d` so the output length stays equal to the input length.
- `model_training/rnn_args.yaml`: adds `smooth_lookahead: 0` as the default real-time setting.
- `model_training/rnn_trainer.py`: training passes `lookahead = self.transform_args['smooth_lookahead']`.
- `model_training/evaluate_model_helpers.py`: eval passes `lookahead = model_args['dataset']['data_transforms']['smooth_lookahead']`, so saved model args structurally enforce train/eval consistency.

For the current `smooth_kernel_std=2` trimmed Gaussian, `K_full=9` and `p=4`. Therefore:

- `smooth_lookahead: 0` is the real-time/causal setting. An impulse at bin `t` should produce zero response before `t` and peak at output `t`.
- `smooth_lookahead: 4` is the symmetric control setting and should reproduce the original smoothing baseline.
- `smooth_lookahead: 1..3` are optional latency/PER tradeoff points if the `0` versus `4` gap is large enough to justify resolving.

Verification status in this checkout:

- `python3 -m py_compile model_training/data_augmentations.py model_training/rnn_trainer.py model_training/evaluate_model_helpers.py` passed.
- `git diff --check` passed.
- `rg` confirms there are no remaining `padding = 'causal'` smoothing call sites under `model_training`.
- The exact Torch impulse unit check is still pending in the project ML environment. The available shell did not have `python`, `torch`, `numpy`, `scipy`, or `conda` available in a usable combination.

Recommended run order remains:

1. `smooth_lookahead: 4` with fresh `output_dir` and `checkpoint_dir` under `trained_models/` as the control. Confirm aggregate validation PER is near the previous ~10.25% reference; if not, assume environment or run drift before interpreting causal results.
2. `smooth_lookahead: 0` with a separate fresh output/checkpoint directory as the real-time frontier point.
3. Fill `smooth_lookahead: 1..3` only if the measured `0` versus `4` PER gap is large enough to justify the added future latency.
