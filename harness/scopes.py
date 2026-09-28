"""Named, deterministic trial scopes over the B2T HDF5 release (h5py; no torch).

val-dev is the public validation split minus the former val-test sessions
(model_training/splits.py). Stratified samples use the midpoint rule within each session's
trial list. A scope's hash covers the ordered trial identities and the HDF5 file hashes.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import functools
import json
from pathlib import Path

import numpy as np

from model_training.splits import N_VAL_DEV_TRIALS, N_VAL_TEST_TRIALS, VAL_TEST_SESSIONS

from . import REPO
from .store import flocked, sha256_json

DATA_DIR = REPO / "data" / "hdf5_data_final"
SCOPES = {
    # name: (split, partition, per-session k or None for all, expected count on the real release)
    "val-dev-full": ("val", "val-dev", None, N_VAL_DEV_TRIALS),
    "val-dev-sample": ("val", "val-dev", 2, 70),
    "former-val-test-full": ("val", "former-val-test", None, N_VAL_TEST_TRIALS),
}


@dataclass(frozen=True)
class Trial:
    session: str
    split: str
    trial_key: str
    hdf5_path: str


def partition_of(session, split):
    if split == "val":
        return "former-val-test" if session in VAL_TEST_SESSIONS else "val-dev"
    return split


def list_trials(data_dir, split, sessions=None):
    import h5py
    trials = []
    for session_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        if sessions is not None and session_dir.name not in sessions:
            continue
        path = session_dir / f"data_{split}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            keys = sorted((k for k in handle.keys() if k.startswith("trial_")),
                          key=lambda k: int(k.split("_")[-1]))
        trials.extend(Trial(session_dir.name, split, key, str(path)) for key in keys)
    return trials


def midpoint_picks(n, k):
    """k evenly spread positions in [0, n): floor((j + 0.5) * n / k); all when k >= n."""
    if k >= n:
        return list(range(n))
    return [int((j + 0.5) * n / k) for j in range(k)]


def build_scope(name, data_dir=DATA_DIR, limit=None, allow_exposed=None, hasher=None):
    split, partition, k, expected = SCOPES[name]
    if partition == "former-val-test" and not allow_exposed:
        raise PermissionError("former val-test influenced checkpoint selection; pass "
                              "--allow-exposed \"<reason>\" to use it as an explicitly exposed scope")
    trials = [t for t in list_trials(data_dir, split) if partition_of(t.session, split) == partition]
    if k is not None:
        by_session = {}
        for trial in trials:
            by_session.setdefault(trial.session, []).append(trial)
        trials = [group[i] for session in sorted(by_session)
                  for group in [by_session[session]] for i in midpoint_picks(len(group), k)]
    if Path(data_dir).resolve() == DATA_DIR.resolve() and len(trials) != expected:
        raise ValueError(f"{name}: found {len(trials)} trials, expected {expected}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        trials = trials[:limit]
    files = sorted({t.hdf5_path for t in trials})
    file_sha = {f: hasher(f) for f in files} if hasher else {}
    identity = [[t.session, t.split, t.trial_key] for t in trials]
    return dict(name=name, split=split, partition=partition, per_session=k, limit=limit,
                exposed_reason=allow_exposed if partition == "former-val-test" else None,
                n_trials=len(trials), n_sessions=len({t.session for t in trials}),
                n_participants=1, trials=trials,
                hash=sha256_json(dict(trials=identity, files=sorted(file_sha.values()))),
                files_sha256={str(Path(f).relative_to(REPO)) if Path(f).is_relative_to(REPO) else f: s
                              for f, s in file_sha.items()})


def scope_record(scope):
    """JSON-serializable scope description (manifests) with the ordered trial list."""
    record = {k: v for k, v in scope.items() if k != "trials"}
    record["trials"] = [[t.session, t.split, t.trial_key] for t in scope["trials"]]
    return record


def decode_text(encoded):
    encoded = np.asarray(encoded)
    return bytes(encoded[encoded > 0].astype(np.uint8)).decode("ascii").strip()


def read_labels(trial):
    """Reference text and phoneme labels (acoustic order); None for unlabeled partitions."""
    import h5py
    with h5py.File(trial.hdf5_path, "r") as handle:
        group = handle[trial.trial_key]
        if "transcription" not in group:
            return None, None
        phones = np.asarray(group["seq_class_ids"][:])
        seq_len = int(group.attrs.get("seq_len", np.count_nonzero(phones)))
        return decode_text(group["transcription"][:]), phones[:seq_len].astype(np.int64)


def corpus_csv(data_dir):
    """The release's block description sits beside the HDF5 directory (data/)."""
    return Path(data_dir).resolve().parent / "t15_copyTaskData_description.csv"


@functools.lru_cache(maxsize=8)
def block_corpora(csv_path):
    """(session, block) -> sentence source corpus; empty if the description file is absent."""
    path = Path(csv_path)
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {("t15." + row["Date"].replace("-", "."), int(row["Block number"])): row["Corpus"]
                for row in csv.DictReader(handle)}


def corpus_of(session, block_num, data_dir):
    if block_num is None:
        return None
    return block_corpora(str(corpus_csv(data_dir))).get((session, int(block_num)))


def trial_info(trial):
    import h5py
    with h5py.File(trial.hdf5_path, "r") as handle:
        group = handle[trial.trial_key]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in group.attrs.items()}
        return dict(n_bins=int(group["input_features"].shape[0]),
                    block_num=attrs.get("block_num"), trial_num=attrs.get("trial_num"),
                    labeled="transcription" in group,
                    corpus=corpus_of(trial.session, attrs.get("block_num"), Path(trial.hdf5_path).parents[1]))


def train_sentence_set(data_dir, cache_root):
    """Normalized TRAIN-split transcriptions, cached by file stats; defines 'seen' text."""
    from model_training.benchmark.stream_lm import norm_words
    files = sorted(Path(data_dir).glob("*/data_train.hdf5"))
    stats = [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in files]
    key = sha256_json(stats)[:16]
    path = Path(cache_root) / "cache" / f"train_sentences_{key}.json"
    with flocked(path.with_suffix(".lock")):
        if path.exists():
            return set(json.loads(path.read_text())["sentences"])
        import h5py
        sentences = set()
        for file in files:
            with h5py.File(file, "r") as handle:
                for trial_key in handle:
                    if "transcription" in handle[trial_key]:
                        sentences.add(" ".join(norm_words(decode_text(handle[trial_key]["transcription"][:]))))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            json.dump(dict(definition="normalized TRAIN-split transcriptions", files=stats,
                           sentences=sorted(sentences)), handle)
        return sentences
