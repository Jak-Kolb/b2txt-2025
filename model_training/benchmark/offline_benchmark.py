import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from .common import (
        BIN_SECONDS,
        args_to_container,
        checkpoint_id,
        configure_fp32_inference,
        discover_trials,
        greedy_ctc_decode_tensor,
        load_args,
        load_model,
        load_trial_features,
        mean,
        resolve_output_path,
        resolve_path,
        rtf_stats,
        select_device,
        smooth_features,
        summarize,
        sync_device,
        timed_call,
        warmup_count,
        write_json,
        get_feature_subset,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from common import (  # type: ignore
        BIN_SECONDS,
        args_to_container,
        checkpoint_id,
        configure_fp32_inference,
        discover_trials,
        greedy_ctc_decode_tensor,
        load_args,
        load_model,
        load_trial_features,
        mean,
        resolve_output_path,
        resolve_path,
        rtf_stats,
        select_device,
        smooth_features,
        summarize,
        sync_device,
        timed_call,
        warmup_count,
        write_json,
        get_feature_subset,
    )


def run_trial(
    model: torch.nn.Module,
    args: Any,
    device: torch.device,
    raw_features_cpu,
    day_idx: int,
) -> Dict[str, Any]:
    raw_features = torch.as_tensor(raw_features_cpu, device=device, dtype=torch.float32).unsqueeze(0)
    day_tensor = torch.tensor([day_idx], device=device, dtype=torch.long)

    with torch.inference_mode():
        smoothed, smoothing_sec = timed_call(device, smooth_features, raw_features, args, device)
        logits, forward_sec = timed_call(device, model, smoothed, day_tensor)
        decoded, greedy_sec = timed_call(device, greedy_ctc_decode_tensor, logits[0])

    total_sec = smoothing_sec + forward_sec + greedy_sec
    return {
        "n_bins": int(raw_features.shape[1]),
        "n_frames": int(logits.shape[1]),
        "decoded_len": int(len(decoded)),
        "smoothing_sec": float(smoothing_sec),
        "forward_sec": float(forward_sec),
        "greedy_sec": float(greedy_sec),
        "total_acoustic_sec": float(total_sec),
        "rtf": float(total_sec / (raw_features.shape[1] * BIN_SECONDS)),
    }


def build_metrics(
    checkpoint_dir: Path,
    args: Any,
    device: torch.device,
    trial_metrics: List[Dict[str, Any]],
    n_warmup_trials: int,
) -> Dict[str, Any]:
    total_times = [trial["total_acoustic_sec"] for trial in trial_metrics]
    n_bins = [trial["n_bins"] for trial in trial_metrics]
    rtfs = [trial["rtf"] for trial in trial_metrics]

    transforms = args["dataset"]["data_transforms"]
    return {
        "mode": "offline",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_id": checkpoint_id(checkpoint_dir),
        "device": str(device),
        "n_trials": len(trial_metrics),
        "n_warmup_trials": n_warmup_trials,
        "bin_seconds": BIN_SECONDS,
        "smooth_lookahead": int(transforms.get("smooth_lookahead", -1)),
        "smooth_kernel_std": float(transforms["smooth_kernel_std"]),
        "smooth_kernel_size": int(transforms["smooth_kernel_size"]),
        "patch_size": int(args["model"]["patch_size"]),
        "patch_stride": int(args["model"]["patch_stride"]),
        "n_layers": int(args["model"]["n_layers"]),
        "n_units": int(args["model"]["n_units"]),
        "stage_sec": {
            "smoothing": summarize([trial["smoothing_sec"] for trial in trial_metrics]),
            "forward_day_patch_gru_head": summarize(
                [trial["forward_sec"] for trial in trial_metrics]
            ),
            "greedy_ctc": summarize([trial["greedy_sec"] for trial in trial_metrics]),
            "total_acoustic": summarize(total_times),
        },
        "rtf": {
            **rtf_stats(total_times, n_bins),
            "mean_from_trials": mean(rtfs),
        },
        "trials": trial_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Acoustic-only offline timing benchmark.")
    parser.add_argument(
        "--checkpoint_dir",
        default="results/causal_la0/checkpoint",
        help="Directory containing best_checkpoint and args.yaml.",
    )
    parser.add_argument(
        "--data_dir",
        default="data/hdf5_data_final",
        help="Directory containing per-session data_<split>.hdf5 files.",
    )
    parser.add_argument("--split", default="val", help="Dataset split to benchmark.")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Number of trials to read before warmup discard; default is all.",
    )
    parser.add_argument(
        "--warmup_trials",
        type=int,
        default=3,
        help="Number of leading trials to run without recording timings.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--out", default="metrics.json", help="Output JSON name or path.")
    parser.add_argument(
        "--include_disabled_days",
        action="store_true",
        help="Include sessions whose saved dataset_probability_val is 0.",
    )
    parsed = parser.parse_args()

    configure_fp32_inference()
    checkpoint_dir = resolve_path(parsed.checkpoint_dir)
    args = load_args(checkpoint_dir)
    device = select_device(parsed.device)
    model, args, _checkpoint = load_model(checkpoint_dir, device, args=args)
    trials = discover_trials(
        args,
        parsed.data_dir,
        split=parsed.split,
        n_trials=parsed.n_trials,
        include_disabled_days=parsed.include_disabled_days,
    )
    feature_subset = get_feature_subset(args)
    n_warmup = warmup_count(len(trials), parsed.warmup_trials)

    trial_metrics: List[Dict[str, Any]] = []
    for index, trial in enumerate(trials):
        raw_features, attrs = load_trial_features(trial, feature_subset)
        result = run_trial(model, args, device, raw_features, trial.day_idx)
        if index < n_warmup:
            continue

        result.update(
            {
                "session": trial.session,
                "day_idx": trial.day_idx,
                "trial_key": trial.trial_key,
                "trial_index": trial.trial_index,
                "block_num": attrs.get("block_num"),
                "trial_num": attrs.get("trial_num"),
            }
        )
        trial_metrics.append(result)

    if not trial_metrics:
        raise RuntimeError("No measured trials remained after warmup")

    sync_device(device)
    metrics = build_metrics(checkpoint_dir, args, device, trial_metrics, n_warmup)
    out_path = resolve_output_path(checkpoint_dir, parsed.out)
    metrics["saved_args"] = args_to_container(args)
    write_json(out_path, metrics)

    print(f"Wrote {out_path}")
    if metrics["rtf"]["any_ge_1"]:
        print("WARNING: offline acoustic RTF is >= 1.0 on at least one measured trial")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
