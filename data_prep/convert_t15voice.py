"""Convert the Wairagkar et al. 2025 T15 voice-synthesis release to the B2T '25 T15 HDF5 layout (training data only).

Only sessions whose date B2T '25 does not contain are converted: every trial on a B2T date is a B2T train, val or
test recording (learning/extra_data_audit), so those dates are skipped whole (rule R1). Per session:

  input_features (T, 512) float32   threshcross then spikepow (both 256 electrodes, order v6v, 4, 55b, d6v as in B2T),
                                    10 ms bins averaged in pairs to 20 ms, optionally mapped by a fitted per-electrode
                                    linear filter (--harmonizer, from learning/extra_data_audit/fit_harmonizer.py),
                                    z-scored per session (the release has no block numbers), clipped at +/-10
  seq_class_ids / transcription     as convert_t12 (repo g2p, class order of the acoustic model)

The release's features are already log-transformed, rolling-normalized and causally smoothed (1.5 s), so they are
not identical to B2T's even after this conversion. Sessions are named t15v.YYYY.MM.DD (date = implant date + day).

    .venv/bin/python -m data_prep.convert_t15voice --voice-dir data/external/wairagkar2025/extracted/Dryad_T15_all \\
        --b2t-dir data/hdf5_data_final --b2t-csv data/t15_copyTaskData_description.csv --out-dir data/hdf5_data_t15v \\
        [--harmonizer FILE.npz] [--exclude-sentences FILE] [--combined-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import scipy.io

from data_prep import convert_t12 as ct

N_FEATURES = 2 * ct.N_ELECTRODES
ASCII_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})   # some cues use curly quotes


def implant_date(csv_path):
    """The implant date implied by the B2T description CSV (date minus post-implant day, identical for all rows)."""
    with open(csv_path) as handle:
        dates = {dt.date.fromisoformat(r["Date"]) - dt.timedelta(days=int(r["Post-implant day"]))
                 for r in csv.DictReader(handle)}
    if len(dates) != 1:
        raise ValueError(f"inconsistent implant dates in {csv_path}: {sorted(dates)}")
    return dates.pop()


def rebin(v):
    """10 ms bins -> 20 ms by averaging consecutive pairs (a trailing odd bin is dropped)."""
    return v[: v.shape[0] // 2 * 2].reshape(-1, 2, v.shape[1]).mean(axis=1)


def harmonize(v, W, K):
    """Per electrode e: [tx_e, sbp_e] at frame t = W[e] . ([tx_e, sbp_e] at t-K..t+K, 1); zeros beyond the trial."""
    T, E = v.shape[0], ct.N_ELECTRODES
    per = v.reshape(T, 2, E).transpose(0, 2, 1)
    pad = np.pad(per, ((K, K), (0, 0), (0, 0)))
    lagged = np.stack([pad[k:k + T] for k in range(2 * K + 1)], axis=3).reshape(T, E, -1)
    design = np.concatenate([lagged, np.ones((T, E, 1), lagged.dtype)], axis=2)
    return np.einsum("tef,efo->teo", design, W).transpose(0, 2, 1).reshape(T, N_FEATURES)


def zscore_session(trials, clip=ct.CLIP):
    """In place: z-score each feature over all trials of the session, zero constant features, clip."""
    stacked = np.concatenate([t["features"] for t in trials], axis=0)
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    live = std > 0
    for t in trials:
        z = np.zeros_like(t["features"])
        z[:, live] = (t["features"][:, live] - mean[live]) / std[live]
        t["features"] = np.clip(z, -clip, clip).astype(np.float32)


def read_session(mat_path):
    m = scipy.io.loadmat(mat_path, variable_names=["cues", "threshcross_trials", "spikepow_trials"])
    cues = [str(c).strip().translate(ASCII_QUOTES) for c in np.ravel(m["cues"])]
    tx, sp = np.ravel(m["threshcross_trials"]), np.ravel(m["spikepow_trials"])
    if not len(cues) == len(tx) == len(sp):
        raise ValueError(f"{mat_path}: field lengths differ")
    trials = []
    for i, (cue, a, b) in enumerate(zip(cues, tx, sp)):
        if a.shape != b.shape or a.ndim != 2 or a.shape[1] != ct.N_ELECTRODES:
            raise ValueError(f"{mat_path} trial {i}: threshcross {a.shape} vs spikepow {b.shape}")
        trials.append(dict(text=cue, features=np.concatenate([a, b], axis=1).astype(np.float64), index=i))
    return trials


def convert_session(session, day, mat_path, out_path, phonemize, class_index, exclusions=frozenset(), harmonizer=None):
    trials = read_session(mat_path)
    for t in trials:
        v = rebin(t["features"])
        t["features"] = harmonize(v, *harmonizer) if harmonizer is not None else v
    zscore_session(trials)
    failures, excluded, written = [], [], []
    for t in trials:
        where = dict(index=t["index"], sentence=t["text"])
        if ct.normalize_sentence(t["text"]) in exclusions:
            excluded.append(where)
            continue
        try:
            t["transcription"] = ct.encode_transcription(t["text"])
            t["seq_class_ids"], t["seq_len"] = ct.encode_phonemes(phonemize(t["text"]), class_index)
        except Exception as error:
            failures.append(dict(where, reason=str(error)))
            continue
        t.update(block=0, trial_num=t["index"], day=day)
        written.append(t)
    ct.write_trials(out_path, session, written, source_attrs=("day", "index"))
    return dict(session=session, day=day, path=str(out_path), n_source=len(trials), n_written=len(written),
                n_excluded=len(excluded), n_failed=len(failures), excluded=excluded, failures=failures)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--voice-dir", required=True, help="extracted Dryad_T15_all directory (t15_dayXXX folders)")
    parser.add_argument("--b2t-dir", required=True, help="B2T '25 session directory; its dates are skipped")
    parser.add_argument("--b2t-csv", required=True, help="t15_copyTaskData_description.csv (gives the implant date)")
    parser.add_argument("--out-dir", required=True, help="new output directory (must not exist)")
    parser.add_argument("--harmonizer", help="npz with W [256, F, 2] and K from fit_harmonizer.py; default: re-bin only")
    parser.add_argument("--exclude-sentences", help="file of sentences to drop (same matching as convert_t12)")
    parser.add_argument("--combined-dir", help="also create this new directory of symlinks to B2T and converted sessions")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; choose a fresh output directory")
    implant = implant_date(args.b2t_csv)
    b2t_dates = {p.name[4:].replace(".", "-") for p in Path(args.b2t_dir).iterdir() if p.name.startswith("t15.")}
    harmonizer = None
    if args.harmonizer:
        with np.load(args.harmonizer) as h:
            harmonizer = (h["W"], int(h["K"]))
    exclusions = ct.load_exclusions(args.exclude_sentences)
    class_index = {p: i for i, p in enumerate(ct.class_order())}
    phonemize = ct.repo_phonemizer()

    out_dir.mkdir(parents=True)
    reports, skipped = [], []
    for folder in sorted(Path(args.voice_dir).glob("t15_day*")):
        day = int(folder.name[len("t15_day"):])
        date = implant + dt.timedelta(days=day)
        if str(date) in b2t_dates:
            skipped.append(dict(day=day, date=str(date)))
            continue
        session = f"t15v.{date:%Y.%m.%d}"
        report = convert_session(session, day, folder / f"{folder.name}_neural_data.mat",
                                 out_dir / session / "data_train.hdf5", phonemize, class_index, exclusions, harmonizer)
        reports.append(report)
        print(f"{session} (day {day}): {report['n_written']} written of {report['n_source']}; "
              f"{report['n_excluded']} excluded, {report['n_failed']} failed", flush=True)
    summary = dict(voice_dir=str(Path(args.voice_dir).resolve()), implant_date=str(implant),
                   harmonizer=args.harmonizer, exclude_sentences=args.exclude_sentences,
                   skipped_b2t_dates=skipped, n_written=sum(r["n_written"] for r in reports),
                   n_excluded=sum(r["n_excluded"] for r in reports), n_failed=sum(r["n_failed"] for r in reports),
                   sessions=reports)
    with (out_dir / "conversion_summary.json").open("x") as handle:
        json.dump(summary, handle, indent=1)
    print(f"total: {summary['n_written']} trials in {len(reports)} sessions; {len(skipped)} B2T dates skipped; "
          f"{summary['n_excluded']} excluded; {summary['n_failed']} failed")
    if args.combined_dir:
        names = ct.link_combined(args.combined_dir, args.b2t_dir, out_dir)
        print(f"linked {len(names)} session directories into {args.combined_dir}")
    return summary


if __name__ == "__main__":
    main()
