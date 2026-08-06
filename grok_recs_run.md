# Testing the `grok_test` real-time pipeline

Branch: **`grok_test`** (do not merge assumptions into `main` until decision rules pass).  
Implements the **conservative ship pipeline** from `grok_recs.md`: keep the causal GRU; add ensemble averaging, phase-stagger eval, blank-skip / adaptive-rescoring hooks, CTC peak-delay gate, stability metrics, and checkpoint averaging.

---

## 0. Prerequisites

```bash
cd /path/to/nejm-brain-to-text
git checkout grok_test
source .venv/bin/activate   # or your env

# Data (validation hdf5 sessions)
ls data/hdf5_data_final/t15.*/data_val.hdf5 | head

# Causal reference checkpoint (smooth_lookahead=0)
ls results/trained_models/causal_la0/checkpoint/best_checkpoint
ls results/trained_models/causal_la0/checkpoint/args.yaml
```

Optional GPU: pass `--device cuda` (or `cuda:0`). Default is `cpu` for laptop smoke tests.

Package layout:

```
model_training/realtime/
  ensemble.py           # R1 logit-average ensemble
  phase_stagger.py      # R3 multi-phase patch eval
  streaming_decode.py   # R2 blank-skip + entropy-gated rescoring hooks
  stability.py          # word-level TTF / RPV metrics
  peak_delay.py         # S6 CTC peak-delay gate
  checkpoint_avg.py     # SWA / last-k weight average
  run_pipeline.py       # CLI entrypoint
  test_realtime.py      # unit tests
  launch_seed_train.py  # multi-seed train launcher
  train_ensemble_seeds.sh
```

---

## 1. Unit tests (no data / no GPU)

```bash
python model_training/realtime/test_realtime.py
```

**Pass criterion:** all tests `OK` (15 tests).

---

## 2. End-to-end smoke (3 val trials, CPU OK)

```bash
python model_training/realtime/run_pipeline.py all-smoke \
  --checkpoint_dir results/trained_models/causal_la0/checkpoint \
  --checkpoint_dirs results/trained_models/causal_la0/checkpoint \
  --data_dir data/hdf5_data_final \
  --n_trials 3 \
  --device cpu \
  --out results/grok_test/all_smoke.json
```

**What you should see:**

| Field | Sanity check |
|---|---|
| `decode_sweep.baseline_greedy_per` | ~10% on a few trials (not full val) |
| `adaptive.mean_blank_skip_rate` | ~0.6–0.8 (≈70% blank mass) |
| `phase.l_buf_effective_ms` | `20.0` with `n_phases=4` |
| `phase.delta_per_pp` | often **positive** (worse) without phase-aug training — expected |
| `ensemble.n_members` | `1` if only one checkpoint listed |
| `peak_delay.decision` | one of `KILL_…` / `AUTHORIZE_…` / `BORDERLINE` / `INCONCLUSIVE` |

---

## 3. Run ledger order (eval-only first)

All commands from repo root. Increase `--n_trials` (or omit limit for full val) for decision-quality numbers. Full val is ~1426 trials.

### 3.1 Decode hyperparameter sweep (0 retrain)

```bash
python model_training/realtime/run_pipeline.py decode-sweep \
  --checkpoint_dir results/trained_models/causal_la0/checkpoint \
  --data_dir data/hdf5_data_final \
  --n_trials 200 \
  --device cuda \
  --blank_thresholds 0.5,0.6,0.7,0.8,0.9 \
  --out results/grok_test/decode_sweep.json
```

**Decision:** pick blank threshold with PER ≤ baseline + 0.1 pp (or best PER). Prefer higher threshold if PER ties (more frames skipped → cheaper beam).

### 3.2 CTC peak delay — S6 gate (0 retrain)

```bash
python model_training/realtime/run_pipeline.py peak-delay \
  --checkpoint_dir results/trained_models/causal_la0/checkpoint \
  --data_dir data/hdf5_data_final \
  --n_trials 200 \
  --device cuda \
  --out results/grok_test/peak_delay.json
```

**Pre-registered rule (from `grok_recs.md`):**

| `decision` | Action |
|---|---|
| `KILL_EMISSION_REGULARIZERS` (median ≤ 40 ms) | Do **not** train FastEmit / delay-CTC / Peak-First / TrimTail |
| `AUTHORIZE_DELAY_CTC` (median ≥ 80 ms) | Authorize **one** delay-penalized CTC seed |
| `BORDERLINE` | Optional single probe only |

**Caveat:** midpoints use equal-duration linear placement (estimated, not gold forced-align). Treat magnitudes as order-of-magnitude; the kill/go threshold is still useful as a coarse gate.

### 3.3 Adaptive rescoring dry-run + blank-skip PER (0 retrain)

```bash
python model_training/realtime/run_pipeline.py adaptive \
  --checkpoint_dir results/trained_models/causal_la0/checkpoint \
  --data_dir data/hdf5_data_final \
  --n_trials 200 \
  --device cuda \
  --blank_threshold 0.7 \
  --entropy_threshold 1.5 \
  --out results/grok_test/adaptive.json
```

Sweep entropy thresholds for a latency curve (R2 / S5):

```bash
for th in 0.5 1.0 1.5 2.0 3.0; do
  python model_training/realtime/run_pipeline.py adaptive \
    --checkpoint_dir results/trained_models/causal_la0/checkpoint \
    --n_trials 200 --device cuda --entropy_threshold $th \
    --out results/grok_test/adaptive_th${th}.json
done
```

**Metrics:**

- `mean_trigger_rate` — fraction of frames that would call the expensive rescorer  
- `blank_skip_greedy_per` vs `greedy_per` — PER cost of blank skip  
- Wire a real LLM/n-gram rescorer later by passing a `rescorer` callback into `run_adaptive_policy_over_logits` (Python API)

### 3.4 Stability metrics (0 retrain)

```bash
python model_training/realtime/run_pipeline.py stability \
  --checkpoint_dir results/trained_models/causal_la0/checkpoint \
  --data_dir data/hdf5_data_final \
  --n_trials 200 \
  --device cuda \
  --out results/grok_test/stability.json
```

Reports mean **TTF** (ms), **RPV** (revisions/word), **prefix freeze rate**.  
Note: pure greedy cumulative partials rarely revise (RPV≈0). Metrics become informative once partial **beam** hypotheses can change prior words.

### 3.5 Phase-stagger (R3) — eval-only first

```bash
python model_training/realtime/run_pipeline.py phase \
  --checkpoint_dir results/trained_models/causal_la0/checkpoint \
  --data_dir data/hdf5_data_final \
  --n_phases 4 \
  --n_trials 200 \
  --device cuda \
  --out results/grok_test/phase_stagger.json
```

**Pre-registered decision rule:**

- H1 if `l_buf_effective_ms == 20` **and** `phase_per ≤ baseline_per + 0.1` (no damage)  
- Prefer `phase_per ≤ baseline_per − 0.2`  
- If phase hurts PER (common without phase-aug training): either  
  1. retrain one seed with random phase offset augmentation, or  
  2. abandon R3 and spend budget on N-seed ensemble  

### 3.6 Checkpoint weight averaging (0 retrain)

If you have multiple checkpoints (last-k epochs or seeds):

```bash
python model_training/realtime/checkpoint_avg.py \
  path/to/seed0/checkpoint/best_checkpoint \
  path/to/seed1/checkpoint/best_checkpoint \
  --out results/grok_test/avg_checkpoint \
  --template results/trained_models/causal_la0/checkpoint/best_checkpoint

# Then point a checkpoint_dir that contains best_checkpoint + args.yaml
# (copy args.yaml from causal_la0 and replace best_checkpoint with the average)
```

---

## 4. R1 — Causal deep ensemble (requires multi-seed training)

### 4.1 Train seeds (GPU, ~6.4 h each)

Dry-run config only:

```bash
python model_training/realtime/launch_seed_train.py --seed 0 --dry_run
```

Train N seeds (isolates `seed` only; keeps `smooth_lookahead=0` and fixed dataset split seed):

```bash
# Example: 3 seeds first (S7 error bars), then scale to 8
bash model_training/realtime/train_ensemble_seeds.sh 3 results/trained_models/ensemble_causal_la0
# later:
bash model_training/realtime/train_ensemble_seeds.sh 8 results/trained_models/ensemble_causal_la0
```

Or one seed:

```bash
python model_training/realtime/launch_seed_train.py \
  --seed 0 \
  --base results/trained_models/causal_la0/checkpoint/args.yaml \
  --out_root results/trained_models/ensemble_causal_la0
```

### 4.2 Evaluate ensemble N∈{1,2,4,8}

```bash
python model_training/realtime/run_pipeline.py ensemble \
  --checkpoint_dirs \
    results/trained_models/ensemble_causal_la0/seed_0/checkpoint \
    results/trained_models/ensemble_causal_la0/seed_1/checkpoint \
    results/trained_models/ensemble_causal_la0/seed_2/checkpoint \
    results/trained_models/ensemble_causal_la0/seed_3/checkpoint \
  --data_dir data/hdf5_data_final \
  --n_trials 500 \
  --device cuda \
  --average_mode prob_mean \
  --out results/grok_test/ensemble_N4.json
```

**Pre-registered decision rule (R1):**

- H1 supported if ensemble val PER ≤ **9.60%** on ≥2 of 3 ensemble constructions  
- Secondary: ensemble PER ≤ single-seed PER − 0.4 pp  
- Streaming equivalence: max |logit_stream − logit_offline| < 1e−3 on 10 trials for **each** member (existing tool):

```bash
python -m model_training.benchmark.test_equivalence \
  --checkpoint_dir results/trained_models/ensemble_causal_la0/seed_0/checkpoint \
  --data_dir data/hdf5_data_final \
  --n_trials 10 \
  --device cuda
```

*(Adjust import/path if you normally run `python model_training/benchmark/test_equivalence.py`.)*

**Latency check:** with N members, acoustic L_comp ≈ N × 0.34 ms. Confirm RTF ≪ 1:

```bash
python model_training/benchmark/streaming_infer.py \
  --checkpoint_dir results/trained_models/causal_la0/checkpoint \
  --data_dir data/hdf5_data_final \
  --n_trials 50 \
  --device cuda \
  --out streaming_metrics.json
```

---

## 5. Python API snippets

```python
from pathlib import Path
import torch
from model_training.realtime.ensemble import EnsembleDecoder
from model_training.realtime.phase_stagger import phase_stagger_logits, PhaseStaggerConfig
from model_training.realtime.streaming_decode import run_adaptive_policy_over_logits
from model_training.realtime.stability import compute_stability_metrics

device = torch.device("cpu")
ens = EnsembleDecoder.from_checkpoint_dirs(
    ["results/trained_models/causal_la0/checkpoint"],
    device,
)
# logits = ens.forward_offline(raw_features_TF, day_idx=0)
```

Entropy-triggered rescoring hook (plug in your LM):

```python
def my_rescorer(partial_text, logits_prefix):
    # call n-gram / LLM; return rescored string
    return partial_text

out = run_adaptive_policy_over_logits(logits, rescorer=my_rescorer)
print(out["trigger_rate"], out["final_text"])
```

---

## 6. Recommended experiment sequence (≈20-run budget)

| Order | Action | GPU-h | Gate |
|---|---|---:|---|
| 1 | Unit tests | 0 | must pass |
| 2 | `all-smoke` n=3 | 0 | plumbing |
| 3 | `decode-sweep` n=200+ | 0 | pick blank thr |
| 4 | `peak-delay` n=200+ | 0 | kill/go emission regs |
| 5 | `adaptive` θ grid | 0 | trigger-rate curve |
| 6 | `phase` n=200+ | 0 | keep/kill R3 |
| 7 | Train seeds 0–2 (S7) | ~19 | error bars |
| 8 | `ensemble` N=1,2,3 | 0 | ΔPER vs N |
| 9 | Train seeds 3–7 if N-curve pays | ~32 | R1 full |
| 10 | Streaming RTF + equivalence | 0 | RTF≪1, eq&lt;1e-3 |

Do **not** start FastEmit / Mamba / width×N until steps 4 and 8 resolve.

---

## 7. Known limitations of this implementation

1. **Ensemble quality** needs distinct trained seeds; N=1 is an identity check only.  
2. **Phase-stagger** without phase-aug training often **raises** PER; measure before shipping.  
3. **Peak-delay** uses linear equal-duration midpoints — estimated, not gold forced-alignment.  
4. **Adaptive rescoring** dry-run counts triggers; it does **not** call the Redis/LLM stack (that remains in `evaluate_model.py`). Hook your rescorer via the Python API.  
5. **Stability** on pure greedy partials understates RPV; use with beam partials for UX metrics.  
6. **Training launcher** assumes `BrainToTextDecoder_Trainer` + OmegaConf YAML layout matching `causal_la0`.

---

## 8. Branch hygiene

```bash
git checkout grok_test
git status
# all realtime work stays on grok_test
# main remains untouched
```

When decision rules pass, open a PR from `grok_test` → `main` with measured PER/WER/latency tables — not unmeasured “improvements.”
