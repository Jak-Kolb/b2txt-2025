import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from .common import (
        BIN_SECONDS,
        args_to_container,
        checkpoint_id,
        configure_fp32_inference,
        discover_trials,
        get_feature_subset,
        load_args,
        load_model,
        load_trial_features,
        resolve_output_path,
        resolve_path,
        rtf_stats,
        select_device,
        smoothing_kernel_from_args,
        summarize,
        sync_device,
        update_online_ctc,
        warmup_count,
        write_json,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from common import (  # type: ignore
        BIN_SECONDS,
        args_to_container,
        checkpoint_id,
        configure_fp32_inference,
        discover_trials,
        get_feature_subset,
        load_args,
        load_model,
        load_trial_features,
        resolve_output_path,
        resolve_path,
        rtf_stats,
        select_device,
        smoothing_kernel_from_args,
        summarize,
        sync_device,
        update_online_ctc,
        warmup_count,
        write_json,
    )


class StreamingDecoder:
    """One-bin-at-a-time acoustic decoder matching the offline GRU pipeline."""

    def __init__(
        self,
        model: torch.nn.Module,
        args: Any,
        day_idx: int,
        device: torch.device,
        enable_timing: bool = False,
    ):
        self.model = model
        self.args = args
        self.device = device
        self.enable_timing = enable_timing
        self.feature_dim = int(args["model"]["n_input_features"])
        self.n_classes = int(args["dataset"]["n_classes"])
        self.patch_size = int(args["model"]["patch_size"])
        self.patch_stride = int(args["model"]["patch_stride"])
        if self.patch_size > 0 and self.patch_stride <= 0:
            raise ValueError("patch_stride must be positive when patch_size > 0")

        transforms = args["dataset"]["data_transforms"]
        self.smooth_data = bool(transforms.get("smooth_data", True))
        if self.smooth_data:
            self.kernel, self.past_taps, self.lookahead = smoothing_kernel_from_args(args, device)
        else:
            self.kernel = torch.ones(1, dtype=torch.float32, device=device)
            self.past_taps = 0
            self.lookahead = 0

        self.day_idx = day_idx
        self.day_weight: torch.Tensor
        self.day_bias: torch.Tensor
        self.reset(day_idx=day_idx)

    def reset(self, day_idx: Optional[int] = None) -> None:
        if day_idx is not None:
            self.day_idx = int(day_idx)

        self.day_weight = self.model.day_weights[self.day_idx].detach()
        self.day_bias = self.model.day_biases[self.day_idx].detach().squeeze(0)
        self.zero_raw = torch.zeros(self.feature_dim, dtype=torch.float32, device=self.device)

        self.raw_buffer = deque(maxlen=int(self.kernel.numel()))
        for idx in range(-self.past_taps, 0):
            self.raw_buffer.append((idx, self.zero_raw))
        self.raw_index = 0
        self.n_real_bins = 0
        self.next_smooth_index = 0

        self.transformed_buffer: List[torch.Tensor] = []
        self.next_patch_start = 0
        self.hidden = self.model.h0.expand(
            self.model.n_layers,
            1,
            self.model.n_units,
        ).contiguous()

        self.logit_frames: List[torch.Tensor] = []
        self.collapsed_tokens: List[int] = []
        self.previous_argmax: Optional[int] = None
        self.finished = False

        self.per_bin_compute_sec: List[float] = []
        self.per_patch_compute_sec: List[float] = []
        self.finish_compute_sec = 0.0

    def process_bin(self, raw_bin) -> List[torch.Tensor]:
        if self.finished:
            raise RuntimeError("StreamingDecoder.process_bin called after finish")

        raw_tensor = torch.as_tensor(raw_bin, device=self.device, dtype=torch.float32)
        if raw_tensor.shape != (self.feature_dim,):
            raise ValueError(f"Expected raw bin shape {(self.feature_dim,)}, got {tuple(raw_tensor.shape)}")

        start = self._timer_start()
        with torch.inference_mode():
            self._append_raw(raw_tensor, is_real=True)
            emitted = self._emit_ready_smoothed()
        self._record_elapsed(start, self.per_bin_compute_sec)
        return emitted

    def finish(self) -> List[torch.Tensor]:
        if self.finished:
            return []

        start = self._timer_start()
        emitted: List[torch.Tensor] = []
        with torch.inference_mode():
            max_target = self.n_real_bins - 1
            for _ in range(self.lookahead):
                self._append_raw(self.zero_raw, is_real=False)
                emitted.extend(self._emit_ready_smoothed(max_target=max_target))

            if self.next_smooth_index != self.n_real_bins:
                raise RuntimeError(
                    "Streaming smoother did not emit the same number of bins as offline "
                    f"({self.next_smooth_index} != {self.n_real_bins})"
                )

        self.finish_compute_sec = self._elapsed(start)
        self.finished = True
        return emitted

    def logits(self) -> torch.Tensor:
        if not self.logit_frames:
            return torch.empty((0, self.n_classes), dtype=torch.float32, device=self.device)
        return torch.stack(self.logit_frames, dim=0)

    def decoded_sequence(self) -> np.ndarray:
        return np.asarray(self.collapsed_tokens, dtype=np.int64)

    def total_compute_sec(self) -> float:
        return float(sum(self.per_bin_compute_sec) + self.finish_compute_sec)

    def timing_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            "per_bin_sec": summarize(self.per_bin_compute_sec),
            "per_emitted_patch_sec": summarize(self.per_patch_compute_sec),
        }

    def _append_raw(self, raw_tensor: torch.Tensor, is_real: bool) -> None:
        self.raw_buffer.append((self.raw_index, raw_tensor.detach()))
        self.raw_index += 1
        if is_real:
            self.n_real_bins += 1

    def _emit_ready_smoothed(self, max_target: Optional[int] = None) -> List[torch.Tensor]:
        emitted: List[torch.Tensor] = []
        while self.next_smooth_index + self.lookahead <= self.raw_index - 1:
            if max_target is not None and self.next_smooth_index > max_target:
                break
            smoothed = self._compute_smoothed(self.next_smooth_index)
            emitted.extend(self._consume_smoothed_bin(smoothed))
            self.next_smooth_index += 1
        return emitted

    def _compute_smoothed(self, target_idx: int) -> torch.Tensor:
        smoothed = torch.zeros_like(self.zero_raw)
        for kernel_idx, weight in enumerate(self.kernel):
            raw_idx = target_idx - self.past_taps + kernel_idx
            smoothed = smoothed + self._raw_for_index(raw_idx) * weight
        return smoothed

    def _raw_for_index(self, raw_idx: int) -> torch.Tensor:
        if raw_idx < 0 or raw_idx >= self.n_real_bins:
            return self.zero_raw
        for idx, value in self.raw_buffer:
            if idx == raw_idx:
                return value
        available = [idx for idx, _value in self.raw_buffer]
        raise RuntimeError(f"Raw index {raw_idx} is no longer in smoother ring buffer {available}")

    def _consume_smoothed_bin(self, smoothed: torch.Tensor) -> List[torch.Tensor]:
        transformed = self.model.day_layer_activation(smoothed @ self.day_weight + self.day_bias)

        if self.patch_size <= 0:
            return [self._step_gru(transformed.view(1, 1, -1))]

        self.transformed_buffer.append(transformed.detach())
        emitted: List[torch.Tensor] = []
        while len(self.transformed_buffer) >= self.next_patch_start + self.patch_size:
            emitted.append(self._emit_patch(self.next_patch_start))
            self.next_patch_start += self.patch_stride
        return emitted

    def _emit_patch(self, start_idx: int) -> torch.Tensor:
        start = self._timer_start()
        patch_bins = self.transformed_buffer[start_idx : start_idx + self.patch_size]
        patch = torch.stack(patch_bins, dim=0).reshape(1, 1, -1)
        frame_logits = self._step_gru(patch)
        self._record_elapsed(start, self.per_patch_compute_sec)
        return frame_logits

    def _step_gru(self, gru_input: torch.Tensor) -> torch.Tensor:
        output, self.hidden = self.model.gru(gru_input, self.hidden)
        logits = self.model.out(output)
        frame_logits = logits[0, 0].float()
        self.logit_frames.append(frame_logits.detach().clone())
        self.previous_argmax = update_online_ctc(
            frame_logits,
            self.previous_argmax,
            self.collapsed_tokens,
        )
        return frame_logits

    def _timer_start(self) -> Optional[float]:
        if not self.enable_timing:
            return None
        sync_device(self.device)
        return time.perf_counter()

    def _elapsed(self, start: Optional[float]) -> float:
        if start is None:
            return 0.0
        sync_device(self.device)
        return time.perf_counter() - start

    def _record_elapsed(self, start: Optional[float], target: List[float]) -> None:
        elapsed = self._elapsed(start)
        if start is not None:
            target.append(float(elapsed))


def run_streaming_trial(
    model: torch.nn.Module,
    args: Any,
    device: torch.device,
    raw_features,
    day_idx: int,
    enable_timing: bool,
) -> Dict[str, Any]:
    decoder = StreamingDecoder(
        model=model,
        args=args,
        day_idx=day_idx,
        device=device,
        enable_timing=enable_timing,
    )

    for raw_bin in raw_features:
        decoder.process_bin(raw_bin)
    decoder.finish()

    logits = decoder.logits()
    total_sec = decoder.total_compute_sec()
    n_bins = int(raw_features.shape[0])
    return {
        "decoder": decoder,
        "logits": logits,
        "decoded": decoder.decoded_sequence(),
        "n_bins": n_bins,
        "n_frames": int(logits.shape[0]),
        "decoded_len": int(len(decoder.collapsed_tokens)),
        "per_bin_sec": decoder.per_bin_compute_sec,
        "per_patch_sec": decoder.per_patch_compute_sec,
        "finish_sec": float(decoder.finish_compute_sec),
        "total_acoustic_sec": total_sec,
        "rtf": float(total_sec / (n_bins * BIN_SECONDS)) if n_bins > 0 else float("nan"),
    }


def build_metrics(
    checkpoint_dir: Path,
    args: Any,
    device: torch.device,
    trial_metrics: List[Dict[str, Any]],
    n_warmup_trials: int,
) -> Dict[str, Any]:
    all_per_bin = [
        value
        for trial in trial_metrics
        for value in trial["per_bin_sec"]
    ]
    all_per_patch = [
        value
        for trial in trial_metrics
        for value in trial["per_patch_sec"]
    ]
    compact_trials = []
    for trial in trial_metrics:
        compact = {
            key: value
            for key, value in trial.items()
            if key not in {"per_bin_sec", "per_patch_sec"}
        }
        compact["per_bin_sec"] = summarize(trial["per_bin_sec"])
        compact["per_emitted_patch_sec"] = summarize(trial["per_patch_sec"])
        compact_trials.append(compact)

    total_times = [trial["total_acoustic_sec"] for trial in trial_metrics]
    n_bins = [trial["n_bins"] for trial in trial_metrics]
    transforms = args["dataset"]["data_transforms"]

    return {
        "mode": "streaming",
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
        "timing": {
            "per_bin_sec": summarize(all_per_bin),
            "per_emitted_patch_sec": summarize(all_per_patch),
            "finish_sec": summarize([trial["finish_sec"] for trial in trial_metrics]),
            "total_acoustic_sec": summarize(total_times),
        },
        "rtf": rtf_stats(total_times, n_bins),
        "trials": compact_trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="One-bin-at-a-time streaming acoustic benchmark.")
    parser.add_argument(
        "--checkpoint_dir",
        default="model_training/trained_models/causal_la0/checkpoint",
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
        result = run_streaming_trial(
            model,
            args,
            device,
            raw_features,
            trial.day_idx,
            enable_timing=index >= n_warmup,
        )
        if index < n_warmup:
            continue

        result.pop("decoder")
        result.pop("logits")
        result.pop("decoded")
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
    metrics["saved_args"] = args_to_container(args)
    out_path = resolve_output_path(checkpoint_dir, parsed.out)
    write_json(out_path, metrics)

    print(f"Wrote {out_path}")
    if metrics["rtf"]["any_ge_1"]:
        print("WARNING: streaming acoustic RTF is >= 1.0 on at least one measured trial")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
