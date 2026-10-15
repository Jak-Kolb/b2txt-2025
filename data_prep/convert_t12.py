"""Convert Willett et al. 2023 (participant T12) competitionData .mat files to the B2T '25 T15 HDF5 layout.

Each session becomes <out_dir>/t12.YYYY.MM.DD/data_train.hdf5 with one group per trial
(trial_0000, ...), holding the same datasets and attributes as data/hdf5_data_final:

  input_features (T, 512) float32   [tx2 (-4.5 x RMS threshold crossings, 256 electrodes),
                                     spikePow (256 electrodes)], the T15 "crossings then
                                     spike-band power" order; electrodes keep T12's column
                                     order (0-127 area 6v, 128-255 area 44)
  seq_class_ids  (500,)   int32     acoustic class ids (BLANK=0, AA..ZH=1..39, SIL=40), zero padded
  transcription  (500,)   int32     character codes of sentence_label, zero padded
  attrs: block_num, n_time_steps, seq_len, trial_num (int64); session, sentence_label (str);
         extra provenance: source_split ('train'/'test'), source_index (row in the .mat file)

Features are z-scored per (block, feature) over all labelled trials of that block (train and
test files pooled, before any exclusion; zero-variance features become 0), then clipped to
+/-10. Phonemes come from the repo's nejm_b2txt_utils.general_utils.sentence_to_phonemes (g2p_en).
Labelled train/ and test/ trials both go to data_train.hdf5; competitionHoldOut is unlabelled and
never read. trial_num is the trial's index within its block in source order, so it stays stable
when trials are excluded. T12 is training data only: no data_val.hdf5 is written.

Usage (repo root, acoustic environment):
    .venv/bin/python -m data_prep.convert_t12 \
        --input-dir data/external/willett2023/extracted/competitionData \
        --out-dir data/hdf5_data_t12 \
        [--exclude-sentences FILE] [--combined-dir data/hdf5_data_t15_t12 --t15-dir data/hdf5_data_final]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import h5py
import numpy as np
import scipy.io

MAX_SEQ_ELEMENTS = 500          # seq_class_ids / transcription length in the T15 files
CLIP = 10.0                     # T15 features rail at exactly +10.0
LABELLED_SPLITS = ("train", "test")
N_ELECTRODES = 256


def class_order():
    """Acoustic class order (model_training/evaluate_model_helpers.LOGIT_TO_PHONEME, 'SIL' for ' | ').

    Derived from general_utils.LOGIT_PHONE_DEF ([BLANK, SIL, AA..ZH]) with SIL moved to the end,
    which reproduces the stored T15 seq_class_ids.
    """
    from nejm_b2txt_utils.general_utils import LOGIT_PHONE_DEF
    return ["BLANK"] + [p for p in LOGIT_PHONE_DEF if p not in ("BLANK", "SIL")] + ["SIL"]


def repo_phonemizer():
    """sentence -> phoneme list, via the same g2p path used for the B2T '25 labels."""
    from g2p_en import G2p
    from nejm_b2txt_utils.general_utils import sentence_to_phonemes
    g2p = G2p()
    return lambda sentence: sentence_to_phonemes(sentence, g2p)[0]


def normalize_sentence(sentence):
    """Exclusion-matching key: lowercase, drop every character except a-z, 0-9 and whitespace, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9\s]", "", str(sentence).lower()).split())


def load_exclusions(path):
    if path is None:
        return set()
    lines = Path(path).read_text().splitlines()
    return {key for key in (normalize_sentence(line) for line in lines) if key}


def read_mat(path, split):
    """Labelled trials of one competitionData .mat file, in file order."""
    mat = scipy.io.loadmat(path)
    texts = [str(t).strip() for t in np.atleast_1d(mat["sentenceText"])]
    tx = mat["tx2"].ravel()
    sbp = mat["spikePow"].ravel()
    blocks = mat["blockIdx"].ravel()
    if not (len(texts) == len(tx) == len(sbp) == len(blocks)):
        raise ValueError(f"{path}: field lengths differ ({len(texts)}, {len(tx)}, {len(sbp)}, {len(blocks)})")
    trials = []
    for i in range(len(texts)):
        if tx[i].shape != sbp[i].shape or tx[i].ndim != 2 or tx[i].shape[1] != N_ELECTRODES:
            raise ValueError(f"{path} trial {i}: tx2 {tx[i].shape} vs spikePow {sbp[i].shape}")
        features = np.concatenate([tx[i].astype(np.float64), sbp[i].astype(np.float64)], axis=1)
        trials.append(dict(text=texts[i], features=features, block=int(blocks[i]), split=split, index=i))
    return trials


def zscore_by_block(trials, clip=CLIP):
    """In place: z-score each feature over all trials of its block, zero constant features, clip."""
    for block in sorted({t["block"] for t in trials}):
        members = [t for t in trials if t["block"] == block]
        stacked = np.concatenate([t["features"] for t in members], axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        live = std > 0
        for t in members:
            z = np.zeros_like(t["features"])
            z[:, live] = (t["features"][:, live] - mean[live]) / std[live]
            t["features"] = np.clip(z, -clip, clip).astype(np.float32)


def encode_transcription(text):
    codes = [ord(c) for c in text]
    if len(codes) > MAX_SEQ_ELEMENTS:
        raise ValueError(f"sentence has {len(codes)} characters (> {MAX_SEQ_ELEMENTS})")
    if any(c > 127 for c in codes):
        raise ValueError("sentence has non-ASCII characters")
    out = np.zeros(MAX_SEQ_ELEMENTS, dtype=np.int32)
    out[:len(codes)] = codes
    return out


def encode_phonemes(phonemes, class_index):
    unknown = sorted({p for p in phonemes if p not in class_index})
    if unknown:
        raise ValueError(f"phonemes outside the class set: {unknown}")
    if not [p for p in phonemes if p != "SIL"]:
        raise ValueError("no phonemes (empty after punctuation removal)")
    if len(phonemes) > MAX_SEQ_ELEMENTS:
        raise ValueError(f"{len(phonemes)} phonemes (> {MAX_SEQ_ELEMENTS})")
    out = np.zeros(MAX_SEQ_ELEMENTS, dtype=np.int32)
    out[:len(phonemes)] = [class_index[p] for p in phonemes]
    return out, len(phonemes)


AREA44_ELECTRODES = range(128, 256)   # README channel map: 0-127 area 6v, 128-255 area 44


def write_trials(out_path, session, trials, source_attrs=()):
    """Write labelled trials (features, seq_class_ids, seq_len, transcription, text, block, trial_num) in the T15 layout.

    source_attrs: extra per-trial keys stored as provenance attributes named source_<key>.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=False)
    with h5py.File(out_path, "x") as f:
        for i, t in enumerate(trials):
            g = f.create_group(f"trial_{i:04d}")
            g.create_dataset("input_features", data=t["features"], compression="lzf")
            g.create_dataset("seq_class_ids", data=t["seq_class_ids"], compression="lzf")
            g.create_dataset("transcription", data=t["transcription"], compression="lzf")
            g.attrs["n_time_steps"] = np.int64(t["features"].shape[0])
            g.attrs["seq_len"] = np.int64(t["seq_len"])
            g.attrs["block_num"] = np.int64(t["block"])
            g.attrs["trial_num"] = np.int64(t["trial_num"])
            g.attrs["session"] = session
            g.attrs["sentence_label"] = t["text"]
            for key in source_attrs:
                value = t[key]
                g.attrs[f"source_{key}"] = np.int64(value) if isinstance(value, (int, np.integer)) else value


def convert_session(session, mat_paths, out_path, phonemize, class_index, exclusions=frozenset(),
                    drop_blocks=frozenset(), zero_area44=False):
    """Write one session's data_train.hdf5; return its report.

    drop_blocks: block numbers removed before normalization (e.g. a block with a known timing error).
    zero_area44: set both features of electrodes 128-255 to 0 after normalization (the input stays 512 wide).
    """
    trials = []
    for split in LABELLED_SPLITS:
        if split in mat_paths:
            trials.extend(read_mat(mat_paths[split], split))
    dropped = [dict(split=t["split"], index=t["index"], block=t["block"]) for t in trials if t["block"] in drop_blocks]
    trials = [t for t in trials if t["block"] not in drop_blocks]
    blocks_by_split = {s: sorted({t["block"] for t in trials if t["split"] == s}) for s in LABELLED_SPLITS}
    shared_blocks = sorted(set(blocks_by_split["train"]) & set(blocks_by_split["test"]))
    zscore_by_block(trials)
    if zero_area44:
        columns = list(AREA44_ELECTRODES) + [N_ELECTRODES + e for e in AREA44_ELECTRODES]
        for t in trials:
            t["features"][:, columns] = 0

    per_block = {}
    for t in sorted(trials, key=lambda t: (t["block"], LABELLED_SPLITS.index(t["split"]), t["index"])):
        t["trial_num"] = per_block.get(t["block"], 0)
        per_block[t["block"]] = t["trial_num"] + 1
    trials.sort(key=lambda t: (t["block"], t["trial_num"]))

    failures, excluded, written = [], [], []
    for t in trials:
        where = dict(split=t["split"], index=t["index"], block=t["block"], sentence=t["text"])
        if normalize_sentence(t["text"]) in exclusions:
            excluded.append(where)
            continue
        try:
            t["transcription"] = encode_transcription(t["text"])
            t["seq_class_ids"], t["seq_len"] = encode_phonemes(phonemize(t["text"]), class_index)
        except Exception as error:     # report every labelling failure instead of aborting the session
            failures.append(dict(where, reason=str(error)))
            continue
        written.append(t)

    write_trials(out_path, session, written, source_attrs=("split", "index"))
    return dict(session=session, path=str(out_path),
                n_source={s: sum(t["split"] == s for t in trials) for s in LABELLED_SPLITS},
                n_written=len(written), n_excluded=len(excluded), n_failed=len(failures), n_dropped_blocks=len(dropped),
                blocks=blocks_by_split, blocks_in_both_splits=shared_blocks,
                max_seq_len=max((t["seq_len"] for t in written), default=0),
                excluded=excluded, failures=failures, dropped_blocks=dropped)


def find_sessions(input_dir):
    sessions = {}
    for split in LABELLED_SPLITS:
        for path in sorted((Path(input_dir) / split).glob("t12.*.mat")):
            sessions.setdefault(path.stem, {})[split] = path
    if not sessions:
        raise FileNotFoundError(f"no t12.*.mat files under {input_dir}/{{train,test}}")
    return dict(sorted(sessions.items()))


def link_combined(combined_dir, t15_dir, t12_dir):
    """A new directory of relative symlinks to every session directory of both datasets."""
    combined_dir = Path(combined_dir)
    sources = [p for d in (t15_dir, t12_dir) for p in sorted(Path(d).iterdir()) if p.is_dir()]
    names = [p.name for p in sources]
    if len(set(names)) != len(names):
        raise ValueError("duplicate session names across the T15 and T12 directories")
    combined_dir.mkdir(parents=True, exist_ok=False)
    for source in sources:
        os.symlink(os.path.relpath(source.resolve(), combined_dir.resolve()), combined_dir / source.name)
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", required=True, help="competitionData directory with train/ and test/")
    parser.add_argument("--out-dir", required=True, help="new output directory (must not exist)")
    parser.add_argument("--exclude-sentences", help="file of sentences to drop, one per line (matched after "
                        "lowercasing, removing punctuation and collapsing whitespace); default: drop nothing")
    parser.add_argument("--sessions", nargs="+", help="convert only these sessions (default: all)")
    parser.add_argument("--drop-blocks", nargs="+", default=[], metavar="SESSION:BLOCK",
                        help="remove whole blocks before normalization, e.g. t12.2022.06.28:5")
    parser.add_argument("--zero-area44", action="store_true",
                        help="set electrodes 128-255 (area 44) to 0 in both features; input stays 512 wide")
    parser.add_argument("--combined-dir", help="also create this new directory of symlinks to the T15 and T12 sessions")
    parser.add_argument("--t15-dir", default="data/hdf5_data_final", help="T15 session directory for --combined-dir")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; choose a fresh output directory")
    sessions = find_sessions(args.input_dir)
    if args.sessions:
        missing = sorted(set(args.sessions) - set(sessions))
        if missing:
            raise SystemExit(f"sessions not found: {missing}")
        sessions = {s: sessions[s] for s in args.sessions}
    exclusions = load_exclusions(args.exclude_sentences)
    drops = {}
    for item in args.drop_blocks:
        name, block = item.rsplit(":", 1)
        if name not in sessions:
            raise SystemExit(f"--drop-blocks session not found: {name}")
        drops.setdefault(name, set()).add(int(block))
    classes = class_order()
    class_index = {p: i for i, p in enumerate(classes)}
    phonemize = repo_phonemizer()

    out_dir.mkdir(parents=True)
    reports = []
    for session, paths in sessions.items():
        report = convert_session(session, paths, out_dir / session / "data_train.hdf5", phonemize, class_index,
                                 exclusions, frozenset(drops.get(session, ())), args.zero_area44)
        reports.append(report)
        print(f"{session}: {report['n_written']} written ({report['n_source']['train']} train + "
              f"{report['n_source']['test']} test source; {report['n_excluded']} excluded, "
              f"{report['n_failed']} failed)", flush=True)
    summary = dict(
        input_dir=str(Path(args.input_dir).resolve()), out_dir=str(out_dir.resolve()),
        exclude_sentences=args.exclude_sentences, n_exclusion_keys=len(exclusions),
        drop_blocks=args.drop_blocks, zero_area44=args.zero_area44,
        features="[tx2 (256), spikePow (256)]; per-(block, feature) z-score over labelled trials, "
                 f"constant features -> 0, clip +/-{CLIP}",
        classes=classes,
        n_written=sum(r["n_written"] for r in reports), n_excluded=sum(r["n_excluded"] for r in reports),
        n_failed=sum(r["n_failed"] for r in reports), sessions=reports)
    with (out_dir / "conversion_summary.json").open("x") as handle:
        json.dump(summary, handle, indent=1)
    print(f"total: {summary['n_written']} trials written in {len(reports)} sessions; "
          f"{summary['n_excluded']} excluded; {summary['n_failed']} failed -> {out_dir / 'conversion_summary.json'}")
    if args.combined_dir:
        names = link_combined(args.combined_dir, args.t15_dir, out_dir)
        print(f"linked {len(names)} session directories into {args.combined_dir}")
    return summary


if __name__ == "__main__":
    main()
