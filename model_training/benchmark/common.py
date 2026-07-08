import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d

BENCHMARK_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_DIR = BENCHMARK_DIR.parent
REPO_ROOT = MODEL_TRAINING_DIR.parent

if str(MODEL_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_DIR))

from data_augmentations import gauss_smooth  # noqa: E402
from rnn_model import GRUDecoder  # noqa: E402

BLANK_CLASS = 0
BIN_SECONDS = 0.020


@dataclass(frozen=True)
class TrialSpec:
    session: str
    day_idx: int
    hdf5_path: Path
    trial_key: str
    trial_index: int


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    repo_path = (REPO_ROOT / path).resolve()
    if cwd_path.exists() or Path.cwd().resolve() == REPO_ROOT:
        return cwd_path
    if repo_path.exists() or str(path).startswith(("model_training/", "data/", "results/")):
        return repo_path
    return cwd_path


def resolve_output_path(checkpoint_dir: Path, out: str | Path) -> Path:
    out_path = Path(out)
    if out_path.is_absolute():
        return out_path
    return checkpoint_dir / out_path


def load_args(checkpoint_dir: str | Path) -> Any:
    checkpoint_dir = resolve_path(checkpoint_dir)
    args_path = checkpoint_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing saved args: {args_path}")
    return OmegaConf.load(args_path)


def args_to_container(args: Any) -> Dict[str, Any]:
    return OmegaConf.to_container(args, resolve=True)


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        cleaned[new_key] = value
    return cleaned


def build_model(args: Any) -> GRUDecoder:
    return GRUDecoder(
        neural_dim=args["model"]["n_input_features"],
        n_units=args["model"]["n_units"],
        n_days=len(args["dataset"]["sessions"]),
        n_classes=args["dataset"]["n_classes"],
        rnn_dropout=args["model"]["rnn_dropout"],
        input_dropout=args["model"]["input_network"]["input_layer_dropout"],
        n_layers=args["model"]["n_layers"],
        patch_size=args["model"]["patch_size"],
        patch_stride=args["model"]["patch_stride"],
    )


def load_model(
    checkpoint_dir: str | Path,
    device: torch.device,
    args: Optional[Any] = None,
) -> Tuple[GRUDecoder, Any, Dict[str, Any]]:
    checkpoint_dir = resolve_path(checkpoint_dir)
    if args is None:
        args = load_args(checkpoint_dir)

    checkpoint_path = checkpoint_dir / "best_checkpoint"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict_keys(checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()
    return model, args, checkpoint


def select_device(device_name: str) -> torch.device:
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return requested


def configure_fp32_inference() -> None:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, fn, *args, **kwargs):
    sync_device(device)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    sync_device(device)
    return result, time.perf_counter() - start


def get_feature_subset(args: Any) -> Optional[Sequence[int]]:
    subset = args["dataset"].get("feature_subset", None)
    if subset is None:
        return None
    return list(subset)


def session_enabled_for_split(args: Any, day_idx: int, split: str) -> bool:
    if split != "val":
        return True
    probabilities = args["dataset"].get("dataset_probability_val", None)
    if probabilities is None or day_idx >= len(probabilities):
        return True
    return bool(probabilities[day_idx])


def discover_trials(
    args: Any,
    data_dir: str | Path,
    split: str = "val",
    n_trials: Optional[int] = None,
    include_disabled_days: bool = False,
) -> List[TrialSpec]:
    data_dir = resolve_path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    trials: List[TrialSpec] = []
    for day_idx, session in enumerate(args["dataset"]["sessions"]):
        if not include_disabled_days and not session_enabled_for_split(args, day_idx, split):
            continue

        hdf5_path = data_dir / session / f"data_{split}.hdf5"
        if not hdf5_path.exists():
            continue

        with h5py.File(hdf5_path, "r") as h5:
            keys = sorted(
                (key for key in h5.keys() if key.startswith("trial_")),
                key=lambda key: int(key.split("_")[-1]),
            )
        for key in keys:
            trials.append(
                TrialSpec(
                    session=session,
                    day_idx=day_idx,
                    hdf5_path=hdf5_path,
                    trial_key=key,
                    trial_index=int(key.split("_")[-1]),
                )
            )
            if n_trials is not None and len(trials) >= n_trials:
                return trials

    if not trials:
        raise RuntimeError(
            f"No data_{split}.hdf5 trials found under {data_dir} for saved sessions"
        )
    return trials


def load_trial_features(
    trial: TrialSpec,
    feature_subset: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    with h5py.File(trial.hdf5_path, "r") as h5:
        group = h5[trial.trial_key]
        features = group["input_features"][:].astype(np.float32, copy=False)
        if feature_subset is not None:
            features = features[:, feature_subset]
        attrs = {}
        for key in group.attrs.keys():
            value = group.attrs[key]
            attrs[key] = value.item() if hasattr(value, "item") else value
    return features, attrs


def smooth_features(
    features: torch.Tensor,
    args: Any,
    device: torch.device,
) -> torch.Tensor:
    transforms = args["dataset"]["data_transforms"]
    if not transforms.get("smooth_data", True):
        return features
    return gauss_smooth(
        inputs=features,
        device=device,
        smooth_kernel_std=transforms["smooth_kernel_std"],
        smooth_kernel_size=transforms["smooth_kernel_size"],
        lookahead=transforms.get("smooth_lookahead", None),
    )


def offline_logits(
    model: GRUDecoder,
    raw_features: torch.Tensor,
    day_idx: int,
    args: Any,
    device: torch.device,
) -> torch.Tensor:
    smoothed = smooth_features(raw_features, args, device)
    day_tensor = torch.tensor([day_idx], device=device, dtype=torch.long)
    return model(smoothed, day_tensor)


def greedy_ctc_decode_tensor(logits: torch.Tensor) -> np.ndarray:
    """Match rnn_trainer validation: argmax, unique_consecutive, strip blank 0."""
    decoded = torch.argmax(logits.clone().detach(), dim=-1)
    decoded = torch.unique_consecutive(decoded, dim=-1)
    decoded_np = decoded.cpu().numpy()
    return np.array([int(token) for token in decoded_np if int(token) != BLANK_CLASS])


def greedy_ctc_decode_numpy(logits: np.ndarray) -> np.ndarray:
    return greedy_ctc_decode_tensor(torch.as_tensor(logits))


def update_online_ctc(
    frame_logits: torch.Tensor,
    previous_argmax: Optional[int],
    tokens: List[int],
) -> int:
    argmax = int(torch.argmax(frame_logits.detach(), dim=-1).item())
    if argmax != previous_argmax and argmax != BLANK_CLASS:
        tokens.append(argmax)
    return argmax


def smoothing_kernel_from_args(args: Any, device: torch.device) -> Tuple[torch.Tensor, int, int]:
    transforms = args["dataset"]["data_transforms"]
    lookahead = transforms.get("smooth_lookahead", None)
    if lookahead is None:
        raise ValueError("Streaming requires saved dataset.data_transforms.smooth_lookahead")

    smooth_kernel_size = transforms["smooth_kernel_size"]
    smooth_kernel_std = transforms["smooth_kernel_std"]

    # Mirrors gauss_smooth exactly so the streaming FIR uses the same weights.
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gauss_kernel = gaussian_filter1d(inp, smooth_kernel_std)
    valid_idx = np.argwhere(gauss_kernel > 0.01)
    gauss_kernel = gauss_kernel[valid_idx]
    gauss_kernel = np.squeeze(gauss_kernel / np.sum(gauss_kernel))

    full_width = gauss_kernel.shape[0]
    past_taps = full_width // 2
    if not 0 <= lookahead <= past_taps:
        raise ValueError(f"lookahead must be in [0, {past_taps}], got {lookahead}")

    gauss_kernel = gauss_kernel[: past_taps + lookahead + 1]
    gauss_kernel = gauss_kernel / gauss_kernel.sum()
    return torch.tensor(gauss_kernel, dtype=torch.float32, device=device), past_taps, int(lookahead)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def rtf_stats(total_times: Sequence[float], n_bins: Sequence[int]) -> Dict[str, float]:
    rtfs = [
        float(total_time / (trial_bins * BIN_SECONDS))
        for total_time, trial_bins in zip(total_times, n_bins)
        if trial_bins > 0
    ]
    stats = summarize(rtfs)
    stats["max"] = float(np.max(rtfs)) if rtfs else float("nan")
    stats["any_ge_1"] = bool(any(value >= 1.0 for value in rtfs))
    return stats


def checkpoint_id(checkpoint_dir: Path) -> str:
    checkpoint_dir = checkpoint_dir.resolve()
    if checkpoint_dir.name == "checkpoint":
        return checkpoint_dir.parent.name
    return checkpoint_dir.name


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def warmup_count(total_trials: int, requested: int) -> int:
    if total_trials <= 1:
        return 0
    return min(requested, total_trials - 1)


def iter_limited(values: Iterable[Any], limit: Optional[int]) -> Iterable[Any]:
    for index, value in enumerate(values):
        if limit is not None and index >= limit:
            return
        yield value
