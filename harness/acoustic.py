"""Acoustic stage (torch): effective-args model loading and the batched-logits cache.

Logits are fp32 with TF32 disabled, computed per trial with common.offline_logits on the
effective preprocessing, and stored in the stream_lm cache format (WFST class order) so the
existing `stream_lm.py decode|sweep` tools read them unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from model_training.benchmark.common import (TrialSpec, configure_fp32_inference, get_feature_subset,
                                             greedy_ctc_decode_numpy, load_args, load_model,
                                             load_trial_features, offline_logits, select_device)
from model_training.benchmark.stream_lm import levenshtein

from . import REPO
from .scopes import read_labels
from .store import sha256_json, write_new_json

CACHE_VERSION = 1
SOURCE_FILES = ("model_training/rnn_model.py", "model_training/data_augmentations.py",
                "model_training/benchmark/common.py")


def effective_args(spec):
    """Saved training args with the pipeline's effective preprocessing applied."""
    from omegaconf import OmegaConf
    container = OmegaConf.to_container(load_args(spec["acoustic"]["checkpoint_dir"]), resolve=True)
    transforms = container["dataset"]["data_transforms"]
    for key, value in spec["preprocess"]["effective"].items():
        if value is None:
            transforms.pop(key, None)
        else:
            transforms[key] = value
    return OmegaConf.create(container)


class Runtime:
    """A loaded acoustic model on one device, with the pipeline's effective args."""

    def __init__(self, spec, device="cuda"):
        import torch
        configure_fp32_inference()
        self.torch = torch
        self.device = select_device(device)
        self.args = effective_args(spec)
        self.model, _, _ = load_model(spec["acoustic"]["checkpoint_dir"], self.device, self.args)
        self.sessions = list(self.args["dataset"]["sessions"])
        self.subset = get_feature_subset(self.args)
        self.patch_size = int(self.args["model"]["patch_size"])
        self.patch_stride = int(self.args["model"]["patch_stride"])

    def day_index(self, session):
        if session not in self.sessions:
            raise KeyError(f"Session {session} has no day layer in this model")
        return self.sessions.index(session)

    def features(self, trial):
        spec = TrialSpec(trial.session, self.day_index(trial.session), Path(trial.hdf5_path),
                         trial.trial_key, int(trial.trial_key.split("_")[-1]))
        features, attrs = load_trial_features(spec, self.subset)
        return features, attrs

    def offline(self, features, session):
        """Acoustic-order [frames, classes] fp32 logits for one whole trial."""
        with self.torch.inference_mode():
            raw = self.torch.as_tensor(features, device=self.device).unsqueeze(0)
            logits = offline_logits(self.model, raw, self.day_index(session), self.args, self.device)
            return logits[0].float().cpu().numpy()

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)


def to_wfst_order(logits):
    """[blank, phones..., sil] -> [blank, sil, phones...] (paced_replay.reorder_logits, vectorized)."""
    return np.concatenate((logits[:, :1], logits[:, -1:], logits[:, 1:-1]), axis=1).astype(np.float32)


def cache_key(identity, scope, hasher):
    import torch
    payload = dict(version=CACHE_VERSION, acoustic_key=identity["acoustic_key"], scope_hash=scope["hash"],
                   precision="fp32_tf32_disabled", torch=torch.__version__,
                   sources={name: hasher(REPO / name) for name in SOURCE_FILES})
    return sha256_json(payload)[:16], payload


def ensure_logits_cache(runtime, identity, scope, root, hasher, log=print):
    """Return (cache_dir, meta, hit). Built once per (model, preprocessing, scope, code)."""
    key, payload = cache_key(identity, scope, hasher)
    cache_dir = Path(root) / "logits" / key
    if cache_dir.exists():
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            raise RuntimeError(f"Incomplete logits cache {cache_dir}; inspect it (nothing is deleted)")
        meta = json.loads(meta_path.read_text())
        if meta["key_payload"] != payload:
            raise RuntimeError(f"Logits cache {cache_dir} key payload mismatch")
        return cache_dir, meta, True

    tmp = cache_dir.with_name(f".tmp-{key}-{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    flat, lengths, days, refs, trials = [], [], [], [], []
    phone_edits = phone_len = 0
    for index, trial in enumerate(scope["trials"]):
        features, attrs = runtime.features(trial)
        logits = runtime.offline(features, trial.session)
        reference, phones = read_labels(trial)
        if reference is None:
            raise ValueError(f"Accuracy scope trial {trial} has no labels")
        greedy = greedy_ctc_decode_numpy(logits)
        edits = levenshtein([int(p) for p in phones], [int(g) for g in greedy])
        phone_edits += edits
        phone_len += len(phones)
        flat.append(to_wfst_order(logits))
        lengths.append(logits.shape[0])
        days.append(runtime.day_index(trial.session))
        refs.append(reference.replace("\n", " "))
        trials.append(dict(session=trial.session, split=trial.split, trial_key=trial.trial_key,
                           n_bins=int(features.shape[0]), n_frames=int(logits.shape[0]),
                           block_num=attrs.get("block_num"), trial_num=attrs.get("trial_num"),
                           day_index=days[-1], phone_edits=int(edits), phone_len=int(len(phones))))
        if (index + 1) % 200 == 0:
            log(f"  acoustic logits {index + 1}/{scope['n_trials']}")
    np.savez(tmp / "logits.npz", n_trials=np.int64(len(flat)), lengths=np.asarray(lengths, dtype=np.int64),
             days=np.asarray(days, dtype=np.int64), flat=np.concatenate(flat, axis=0),
             avg_PER=np.float64(phone_edits / phone_len if phone_len else np.nan))
    (tmp / "logits.txt").write_text("\n".join(refs))
    meta = dict(key=key, key_payload=payload, created=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic() - start, n_trials=len(trials), trials=trials,
                files_sha256=dict(npz=hasher(tmp / "logits.npz"), txt=hasher(tmp / "logits.txt")),
                note="logits in WFST order [blank, sil, phones]; references are for scoring only")
    write_new_json(tmp / "meta.json", meta)
    os.rename(tmp, cache_dir)
    return cache_dir, meta, False
