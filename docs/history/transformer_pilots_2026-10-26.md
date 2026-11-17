# Conv-Transformer block tests and pilots, 2026-10-26 to 27

Moved unchanged from `current_knowledge.md` 4.8 on 2026-11-17, when that section was condensed. The results that
still matter (X3 scored, the diagnostics, the X4-X7 screen and the finishing runs) stay in `current_knowledge.md`
4.8. Pilot names: tfmp_conv_s10 (pre-fix code, not registered), X1, X2; X3 is the 60k run.

- **Macaron Conformer blocks (with the per-block output LayerNorm) stayed on the CTC blank plateau.** Train loss was
  ~75-90 through batch 2000, val PER 74.6% at 2k (d384, 8 layers, causal conv kernel 9, LayerNorm in the conv module).
- 800-batch diagnostics with the same recipe, val PER at batch 800: plain pre-norm block 25.4; pre-norm block plus a
  causal convolution module ("conv" block) 23.4; macaron without the output LayerNorm 31.9; macaron with it 92.8 (that
  run at weight decay 0.001). Single runs, scratchpad only.
- **Pilot tfmp_conv_s10** (pre-fix code; dropout 0.2, drop_path 0.1, weight decay 0.05, window 64): val PER 16.6 / 14.0
  / 13.3 at 2k / 4k / 6k (G2 21.9 / 15.6 / 13.6), so faster than every earlier model. Val loss still rises
  (18.9 -> 22.3), and train loss is ~0.6-1.1 by 7.6k, so it still memorizes.
  It died at batch 7800 (CUDA "unknown error" while GPU tests ran beside it), so it has no 8k value.
- **Fixed-code pilots** (8001 batches of the 20k schedule; trainer val PER / val CTC loss at 2k, 4k, 6k, 8k):

  | Pilot | 2k | 4k | 6k | 8k |
  |---|---|---|---|---|
  | tfmp_conv_s10 (pre-fix code) | 16.6 / 18.9 | 14.0 / 19.6 | 13.3 / 22.3 | (crashed) |
  | X1 (same configuration) | 16.2 / 18.7 | 14.5 / 21.6 | 13.7 / 22.7 | **12.75** / 22.8 |
  | X2 (dropout 0.3, drop_path 0.2, wd 0.1, window 32) | 16.3 / 18.7 | 14.1 / 19.9 | 13.4 / 21.6 | 12.78 / 23.2 |
  | G2 (GRU, pre-fix) | 21.9 / 21.9 | 15.6 / 17.4 | 13.6 / 17.5 | 12.63 / 17.3 |
  | Z1 (old transformer, 120k schedule) | 18.7 / 22.2 | 15.4 / 22.6 | 14.3 / 23.2 | 13.89 / 24.7 |

  - The fix made no clear difference for this transformer at 8k (differences within single-run noise).
  - Stronger regularization did not change val PER at 8k. Val loss still rises in both: the rise is not removed by
    dropout, stochastic depth or weight decay at these strengths.
  - Selected by the pre-set rule (lowest val PER at 8k): X1 -> **X3** (60k,
    started 2026-10-27 00:09 UTC).
- **X3 training** (189 min; trainer val PER / val loss): 10k 12.93 / 23.2, 20k 12.75 / 26.7, 30k
  12.40 / 27.7, 40k 11.64 / 28.6, 58k 10.80 / 29.4, final 59999 11.09 / 30.0. G9 (GRU, pre-fix): 20k
  11.48 / 21.3, 40k 10.35 / 21.0, final 10.05 / 20.6. The conv Transformer is ~1 point PER behind the GRU at 60k, and
  its val loss keeps rising (the GRU's falls).
- Queue: `results/harness/queue/tfm_20261026.py`, then `tfm2_20261026.py`.
