"""C7 — the frozen val-dev / val-test split.

The public validation split (1,426 trials, 41 sessions) is subdivided BY SESSION, not by trial.
Splitting by session preserves the day-layer generalization structure and avoids leaking
block-level normalization statistics between dev and test — which, given that the released
features are whole-block z-scored (AUDIT F1), would otherwise be a genuine leak.

    val-dev   35 sessions, 1,253 trials  — tune everything here
    val-test   6 sessions,   173 trials  — touch ONCE per group, at the gate

Held out are the six most recent sessions, all of which carry dataset_probability_val=1.
Listed as literals so the boundary cannot drift when rnn_args.yaml is edited.

NEVER call this "test WER". data_test.hdf5 contains only input_features — no labels of any
kind (AUDIT F7) — and the leaderboard is closed. The honest phrasing is
"held-out sessions from the public validation split".
"""

VAL_TEST_SESSIONS = (
    "t15.2025.01.10",
    "t15.2025.01.12",
    "t15.2025.03.14",
    "t15.2025.03.16",
    "t15.2025.03.30",
    "t15.2025.04.13",
)

# Indices into rnn_args.yaml's `dataset.sessions` list, which is also what val_metrics.pkl's
# `day_indicies` refers to. Verified 2026-08-06: these are exactly the sessions above.
VAL_TEST_DAY_INDICES = (39, 40, 41, 42, 43, 44)

# Measured 2026-08-06 from results/causal_la0/checkpoint/val_metrics.pkl
N_VAL_TRIALS = 1426
N_VAL_TEST_TRIALS = 173          # 12.1 % of the validation split
N_VAL_DEV_TRIALS = 1253


def is_val_test(day_index: int) -> bool:
    return day_index in VAL_TEST_DAY_INDICES


def verify(sessions) -> None:
    """Assert the literal indices still match the session names in a loaded config."""
    actual = tuple(i for i, s in enumerate(sessions) if s in VAL_TEST_SESSIONS)
    if actual != VAL_TEST_DAY_INDICES:
        raise ValueError(
            f"val-test split drifted: sessions {VAL_TEST_SESSIONS} are now at indices "
            f"{actual}, not {VAL_TEST_DAY_INDICES}. The session list in rnn_args.yaml changed."
        )
