"""Offline-equivalence acceptance gate.

No trial-boundary sequence differences are allowed here. The streaming decoder
flushes the saved lookahead with zero-valued future bins at utterance end, which
matches the right-side zero pad used by gauss_smooth offline.
"""

import argparse
from pathlib import Path
from typing import Any, List

import numpy as np
import torch

try:
    from .common import (
        configure_fp32_inference,
        discover_trials,
        get_feature_subset,
        greedy_ctc_decode_tensor,
        load_args,
        load_model,
        load_trial_features,
        offline_logits,
        resolve_path,
        select_device,
    )
    from .streaming_infer import run_streaming_trial
except ImportError:  # pragma: no cover - supports direct script execution
    from common import (  # type: ignore
        configure_fp32_inference,
        discover_trials,
        get_feature_subset,
        greedy_ctc_decode_tensor,
        load_args,
        load_model,
        load_trial_features,
        offline_logits,
        resolve_path,
        select_device,
    )
    from streaming_infer import run_streaming_trial  # type: ignore


def first_sequence_difference(left: np.ndarray, right: np.ndarray) -> str:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return f"token {index}: offline={int(left[index])}, streaming={int(right[index])}"
    if len(left) != len(right):
        return f"length differs: offline={len(left)}, streaming={len(right)}"
    return "no difference"


def assert_trial_equivalent(
    model: torch.nn.Module,
    args: Any,
    device: torch.device,
    raw_features,
    day_idx: int,
    tolerance: float,
    trial_label: str,
) -> None:
    raw_tensor = torch.as_tensor(raw_features, device=device, dtype=torch.float32).unsqueeze(0)

    with torch.inference_mode():
        offline = offline_logits(model, raw_tensor, day_idx, args, device)[0].float()
        offline_sequence = greedy_ctc_decode_tensor(offline)

    streamed = run_streaming_trial(
        model=model,
        args=args,
        device=device,
        raw_features=raw_features,
        day_idx=day_idx,
        enable_timing=False,
    )
    streamed_logits = streamed["logits"].float()
    streamed_sequence = streamed["decoded"]

    if tuple(streamed_logits.shape) != tuple(offline.shape):
        raise AssertionError(
            f"{trial_label}: logit shape mismatch offline={tuple(offline.shape)} "
            f"streaming={tuple(streamed_logits.shape)}"
        )

    if offline.numel() == 0:
        max_diff = 0.0
    else:
        diff = torch.abs(streamed_logits - offline)
        frame_diff = diff.amax(dim=1)
        max_diff = float(frame_diff.max().item())

    if not max_diff < tolerance:
        bad_frames = torch.nonzero(frame_diff >= tolerance, as_tuple=False)
        frame_idx = int(bad_frames[0, 0].item())
        class_idx = int(torch.argmax(diff[frame_idx]).item())
        raise AssertionError(
            f"{trial_label}: logit mismatch at frame {frame_idx}, class {class_idx}; "
            f"abs_diff={float(diff[frame_idx, class_idx].item()):.6g}, "
            f"max_abs_diff={max_diff:.6g}, tolerance={tolerance}"
        )

    if not np.array_equal(offline_sequence, streamed_sequence):
        raise AssertionError(
            f"{trial_label}: greedy CTC sequence mismatch "
            f"({first_sequence_difference(offline_sequence, streamed_sequence)})"
        )


def run_checkpoint(
    checkpoint_dir: Path,
    data_dir: str,
    split: str,
    n_trials: int | None,
    include_disabled_days: bool,
    device: torch.device,
    tolerance: float,
) -> int:
    args = load_args(checkpoint_dir)
    model, args, _checkpoint = load_model(checkpoint_dir, device, args=args)
    trials = discover_trials(
        args=args,
        data_dir=data_dir,
        split=split,
        n_trials=n_trials,
        include_disabled_days=include_disabled_days,
    )
    feature_subset = get_feature_subset(args)

    for index, trial in enumerate(trials):
        raw_features, _attrs = load_trial_features(trial, feature_subset)
        trial_label = (
            f"{checkpoint_dir} {trial.session}/{trial.trial_key} "
            f"(day_idx={trial.day_idx}, index={index})"
        )
        assert_trial_equivalent(
            model=model,
            args=args,
            device=device,
            raw_features=raw_features,
            day_idx=trial.day_idx,
            tolerance=tolerance,
            trial_label=trial_label,
        )

    return len(trials)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assert streaming logits match offline logits.")
    parser.add_argument(
        "--checkpoint_dirs",
        nargs="+",
        default=[
            "model_training/trained_models/causal_la0/checkpoint",
            "model_training/trained_models/causal_la4/checkpoint",
        ],
        help="Checkpoint directories to test.",
    )
    parser.add_argument(
        "--data_dir",
        default="data/hdf5_data_final",
        help="Directory containing per-session data_<split>.hdf5 files.",
    )
    parser.add_argument("--split", default="val", help="Dataset split to test.")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Number of trials per checkpoint to test; default is all.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Max logit abs diff.")
    parser.add_argument(
        "--include_disabled_days",
        action="store_true",
        help="Include sessions whose saved dataset_probability_val is 0.",
    )
    parsed = parser.parse_args()

    configure_fp32_inference()
    device = select_device(parsed.device)

    for checkpoint_dir_raw in parsed.checkpoint_dirs:
        checkpoint_dir = resolve_path(checkpoint_dir_raw)
        try:
            n_checked = run_checkpoint(
                checkpoint_dir=checkpoint_dir,
                data_dir=parsed.data_dir,
                split=parsed.split,
                n_trials=parsed.n_trials,
                include_disabled_days=parsed.include_disabled_days,
                device=device,
                tolerance=parsed.tolerance,
            )
        except Exception as exc:
            print(f"FAILED {checkpoint_dir}: {exc}")
            return 1
        print(f"PASSED {checkpoint_dir}: {n_checked} trial(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
