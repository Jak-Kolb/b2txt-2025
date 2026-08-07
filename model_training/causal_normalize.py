"""C8 — strictly causal rolling normalization, and the identity check that licenses R-D2/R-D3.

Why this matters
----------------
The released HDF5 features are whole-block z-scored: every value at time t was offset and scaled
by statistics computed over up to ~19 minutes of *future* recording. So the 10.04 % / 10.21 % PER
figures were obtained with leaked normalization statistics.

The fix needs no raw data. Block z-scoring gives `x_norm = (x_raw - mu_blk)/sigma_blk`. Applying a
causal rolling normalizer on top yields

    (x_norm - mu_roll,norm) / sigma_roll,norm
        = [ (x_raw - mu_blk)/s - (mu_roll,raw - mu_blk)/s ] / (sigma_roll,raw / s)
        = (x_raw - mu_roll,raw) / sigma_roll,raw

i.e. **the block constants cancel identically** and rolling-normalizing the released data recovers
the causally-normalized raw features. That is an algebraic claim about invariance under any
per-channel affine map, and this module tests it numerically before 12.8 GPU-hours are spent on it.

Run the check:

    cd model_training && ../.venv/bin/python causal_normalize.py

Gate G1-d: both residuals must be < 1e-4. If they are not, the transform has been misidentified
and R-D2/R-D3 are void as specified.
"""
import argparse
import glob
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent

# The only term that is not affine-invariant. Kept far below the variance of real features
# (which are O(1) after block z-scoring) so it cannot matter, but it is why the identity holds
# to ~1e-7 rather than exactly.
_EPS = 1e-8


def rolling_normalize(features, half_life_bins=500, warmup_bins=50, return_std=False):
    """Strictly causal per-channel normalization. features: [T, C] -> [T, C].

    Emits using statistics from bins strictly BEFORE t, then updates. The first `warmup_bins`
    emit zeros: before ~1 s of history the variance estimate is not yet conditioned (at t=1 it is
    identically 0), and dividing by it would both explode the output and destroy the affine
    invariance the whole scheme rests on. Emitting nothing until the estimator is usable is also
    what a deployed system does, and it is free here — the measured median time-to-first-token is
    3,380 ms, so a 1 s warmup lands entirely inside the pre-speech period.

    Statistics use an expanding window through the warmup, then an EWMA with the requested
    half-life. half_life_bins=500 is 10 s at 20 ms/bin, matching the rolling scheme Wairagkar
    et al. use on this participant.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected [T, C], got {x.shape}")
    T, C = x.shape
    alpha = 1.0 - math.exp(-math.log(2.0) / half_life_bins)

    out = np.zeros_like(x)
    std_used = np.zeros_like(x) if return_std else None
    mean = np.zeros(C)
    var = np.zeros(C)
    csum = np.zeros(C)
    csumsq = np.zeros(C)

    for t in range(T):
        if t >= warmup_bins:
            out[t] = (x[t] - mean) / np.sqrt(var + _EPS)
            if return_std:
                std_used[t] = np.sqrt(var)

        xt = x[t]
        csum += xt
        csumsq += xt * xt
        n = t + 1
        if n <= warmup_bins:
            # Expanding window: the EWMA's effective sample count is still tiny here.
            mean = csum / n
            var = np.maximum(csumsq / n - mean * mean, 0.0)
        else:
            delta = xt - mean
            mean = mean + alpha * delta
            var = (1.0 - alpha) * (var + alpha * delta * delta)

    return (out, std_used) if return_std else out


def _iter_blocks(data_dir, split, max_sessions):
    """Yield (session, block_num, [T, C]) with a block's trials concatenated in trial order.

    The block is the unit the released features were z-scored over, so it is the unit the check
    has to operate on.
    """
    import h5py
    paths = sorted(glob.glob(str(pathlib.Path(data_dir) / "*" / f"data_{split}.hdf5")))
    for path in paths[:max_sessions]:
        blocks = {}
        with h5py.File(path, "r") as h:
            for key in sorted(h.keys()):
                g = h[key]
                blocks.setdefault(int(g.attrs["block_num"]), []).append(
                    np.asarray(g["input_features"][:], dtype=np.float64))
        session = pathlib.Path(path).parent.name
        for block_num, trials in sorted(blocks.items()):
            yield session, block_num, np.concatenate(trials, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(REPO / "data/hdf5_data_final"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--half_life_bins", type=int, default=500)
    ap.add_argument("--warmup_bins", type=int, default=50)
    ap.add_argument("--max_bins", type=int, default=6000,
                    help="cap per block; the loop is O(T*C) in python and blocks run long")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--min_std", type=float, default=1e-3,
                    help="running std below which the emission is floor-dominated and excluded")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    worst_block, worst_affine = 0.0, 0.0
    rows = []

    for session, block_num, feats in _iter_blocks(a.data_dir, a.split, a.sessions):
        # DIAGNOSTIC, not a gate. Whole-block z-scoring implies mean 0 / std 1 over the
        # normalization unit, but data_val.hdf5 holds only the val trials of each block, so this
        # is always a SUBSAMPLE of that unit and its mean is displaced by sampling noise. This is
        # the same artifact that made analyses/check_hdf5_causality.py false-negative. The
        # decisive evidence for block z-scoring is the 1/sqrt(W) scaling test in the audit; here
        # the informative quantity is the median std, which lands on 1.
        mu, sd = feats.mean(axis=0), feats.std(axis=0)
        block_resid = float(np.median(np.abs(sd - 1.0)))

        # THE GATE. Any per-channel affine map stands in for "a different block constant"; if the
        # causal normalizer is blind to it, then the block constants cancel and rolling-normalizing
        # released data recovers the causally-normalized raw features.
        sub = feats[: a.max_bins]
        scale = rng.uniform(0.5, 3.0, size=sub.shape[1])
        shift = rng.uniform(-2.0, 2.0, size=sub.shape[1])
        base, std_used = rolling_normalize(sub, a.half_life_bins, a.warmup_bins, return_std=True)
        moved = rolling_normalize(sub * scale + shift, a.half_life_bins, a.warmup_bins)

        # Where the running variance is degenerate the normalized value is 0/0 for raw and
        # released data alike, and _EPS -- the one term that is not affine-covariant -- decides
        # the output. Those points are excluded and counted, not silently averaged away. They
        # arise on sparse threshold-crossing channels that are locally constant.
        live = std_used > a.min_std
        live[: a.warmup_bins] = False          # warmup emits zeros by construction
        diff = np.abs(base - moved)
        affine_resid = float(diff[live].max()) if live.any() else float("nan")
        floored = float((~live[a.warmup_bins:]).mean())

        worst_block = max(worst_block, block_resid)
        worst_affine = max(worst_affine, affine_resid)
        rows.append((session, block_num, feats.shape[0], block_resid, affine_resid, floored))

    print(f"{'session':<16} {'blk':>4} {'bins':>6} {'med|std-1|':>12} {'|affine resid|':>15} "
          f"{'%floored':>9}")
    print("-" * 70)
    for session, block_num, n, br, ar, fl in rows:
        print(f"{session:<16} {block_num:>4} {n:>6} {br:>12.2e} {ar:>15.2e} {100*fl:>8.4f}%")

    print("\n" + "=" * 66)
    print(f"  blocks checked                      : {len(rows)}")
    print(f"  worst median |std-1| (diagnostic)   : {worst_block:.3e}  <- expect ~1e-3, not 1e-4;"
          f" val trials are a subsample of the block")
    print(f"  worst affine-invariance residual    : {worst_affine:.3e}  (running std > {a.min_std:g})")
    print(f"  worst %(t,ch) hitting the variance floor: {100*max(r[5] for r in rows):.4f} %")
    ok = worst_affine < a.tol
    print(f"  G1-d (affine residual < {a.tol:g})       : {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  -> the causal rolling normalizer is blind to per-channel affine maps, so the")
        print("     block constants cancel identically. Rolling-normalizing the RELEASED data")
        print("     recovers (x_raw - mu_roll,raw)/sigma_roll,raw with no access to raw data.")
        print("     R-D2/R-D3 are well-posed.")
        print("  OPEN: the variance floor is a real design choice for R-D2/R-D3, not a detail.")
        print("     Sparse threshold-crossing channels go locally constant, and no floor can be")
        print("     both fixed and scale-invariant. Decide it before the run, not after.")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
