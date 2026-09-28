"""Where word errors come from: acoustic phonemes vs language-model words (numpy only).

For each reference word we align three things:
- the reference phonemes (the dataset's phoneme labels; silence tokens separate words),
- the acoustic model's greedy phonemes (per-frame argmax, CTC-collapsed, from cached logits),
- the LM decoder's word (word-level edit alignment) and its lexicon pronunciation.

Categories per reference word: both_correct, lm_fixed (phonemes wrong, word right),
lm_introduced (phonemes exactly right, word wrong), both_wrong. Hypothesis words aligned to no
reference word are lm_inserted. Greedy phonemes are a diagnostic proxy for the acoustic
evidence: the WFST search uses full frame posteriors, not argmax labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model_training.benchmark.stream_lm import norm_words

VERSION = 1
# Acoustic class order (= model_training/evaluate_model_helpers.LOGIT_TO_PHONEME, '|' for silence).
PHONES = ["BLANK", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
          "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
          "UH", "UW", "V", "W", "Y", "Z", "ZH", "|"]
SIL = 40
CATEGORIES = ("both_correct", "lm_fixed", "lm_introduced", "both_wrong")
LABEL = ("greedy phonemes are the per-frame argmax (CTC-collapsed): a proxy for the acoustic evidence, "
         "not what the WFST search saw; reference phonemes are the dataset labels for the cued sentence")


def wfst_to_acoustic(index):
    """WFST class order [blank, sil, phones...] -> acoustic order [blank, phones..., sil]."""
    return 0 if index == 0 else SIL if index == 1 else index - 1


def greedy_tokens(wfst_logits):
    """[(acoustic_class, frame)] for each CTC-collapsed, non-blank argmax token."""
    best = np.argmax(wfst_logits, axis=1)
    out, previous = [], None
    for frame, index in enumerate(best.tolist()):
        if index != previous and index != 0:
            out.append((wfst_to_acoustic(index), frame))
        previous = index
    return out


def align(ref, hyp, boundary=None):
    """Minimum-edit alignment: [(op, ref_index|None, hyp_index|None)], op in ok/sub/del/ins.

    With `boundary` (the silence class), substituting a phoneme for a word boundary costs as
    much as an insertion plus a deletion, so phonemes stay inside their words.
    """
    def sub(a, b):
        return 0 if a == b else 2 if boundary is not None and boundary in (a, b) else 1
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + sub(ref[i - 1], hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + sub(ref[i - 1], hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif j > 0 and d[i, j] == d[i, j - 1] + 1:
            ops.append(("ins", None, j - 1))
            j -= 1
        else:
            ops.append(("del", i - 1, None))
            i -= 1
    return ops[::-1]


def load_lexicon(graph_dir):
    """First pronunciation per word from the graph's lexicon_numbers.txt (+ units.txt); {} if absent."""
    graph_dir = Path(graph_dir)
    lexicon_path, units_path = graph_dir / "lexicon_numbers.txt", graph_dir / "units.txt"
    if not lexicon_path.exists() or not units_path.exists():
        return {}
    units = {}
    for line in units_path.read_text().splitlines():
        name, number = line.split()
        units[number] = name
    lexicon = {}
    for line in lexicon_path.read_text().splitlines():
        word, *numbers = line.split()
        lexicon.setdefault(word.lower(), [units.get(n, "?") for n in numbers])
    return lexicon


def attribute_trial(reference, ref_phone_ids, tokens, hypothesis, lexicon, frame_ms):
    """Per-reference-word alignment of reference phonemes, greedy phonemes, and the LM word."""
    ref_words, hyp_words = norm_words(reference), norm_words(hypothesis)
    ref_ids = [int(p) for p in ref_phone_ids]
    word_of = []  # reference position -> word index (None for silence)
    word = 0
    for p in ref_ids:
        if p == SIL:
            word_of.append(None)
            if word_of[:-1] and word_of[-2] is not None:
                word += 1
        else:
            word_of.append(word)
    n_ref_word_groups = len({w for w in word_of if w is not None})
    greedy_ids = [t[0] for t in tokens]
    result = dict(ref_words=ref_words, hyp_words=hyp_words,
                  ref_phones=[PHONES[p] for p in ref_ids],
                  greedy=[dict(p=PHONES[c], t_ms=round(frame_ms(f), 1)) for c, f in tokens])
    if n_ref_word_groups != len(ref_words):
        result.update(status="ref_word_phone_mismatch", words=[], insertions=[], counts={})
        return result

    words = [dict(ref=w, ref_phones=[], greedy=[], phone_edits=0) for w in ref_words]
    last_word = 0
    for op, i, j in align(ref_ids, greedy_ids, boundary=SIL):
        if i is not None and word_of[i] is not None:
            last_word = word_of[i]
            target = last_word
        elif i is not None:          # reference silence: a boundary, not part of any word
            target = None
        else:                        # insertion: attach to the preceding word
            target = last_word
        entry = dict(op=op, ref=PHONES[ref_ids[i]] if i is not None else None,
                     hyp=PHONES[greedy_ids[j]] if j is not None else None,
                     t_ms=round(frame_ms(tokens[j][1]), 1) if j is not None else None)
        if target is None:
            continue
        if entry["ref"] is not None:
            words[target]["ref_phones"].append(entry["ref"])
        if entry["hyp"] == "|" and entry["ref"] is None:
            continue                 # an extra word-boundary token is not a phoneme error
        words[target]["greedy"].append(entry)
        words[target]["phone_edits"] += op != "ok"

    insertions = []
    previous_ref = -1  # index of the last reference word passed (-1: before the first)
    for op, i, j in align(ref_words, hyp_words):
        if op == "ins":
            insertions.append(dict(word=hyp_words[j], after=previous_ref,
                                   lm_phones=lexicon.get(hyp_words[j]), category="lm_inserted"))
            continue
        previous_ref = i
        w = words[i]
        w["lm_op"] = op
        w["lm_word"] = hyp_words[j] if j is not None else None
        w["lm_phones"] = lexicon.get(w["lm_word"]) if w["lm_word"] else None
        acoustic_ok, lm_ok = w["phone_edits"] == 0, op == "ok"
        w["category"] = ("both_correct" if acoustic_ok and lm_ok else "lm_fixed" if lm_ok
                         else "lm_introduced" if acoustic_ok else "both_wrong")
        times = [g["t_ms"] for g in w["greedy"] if g["t_ms"] is not None]
        w["t_ms"] = min(times) if times else None
    counts = {c: sum(w["category"] == c for w in words) for c in CATEGORIES}
    counts["lm_inserted"] = len(insertions)
    result.update(status="ok", words=words, insertions=insertions, counts=counts)
    return result


def summarize(per_trial):
    counts = {c: 0 for c in CATEGORIES + ("lm_inserted",)}
    skipped = 0
    for trial in per_trial:
        if trial["status"] != "ok":
            skipped += 1
            continue
        for key, value in trial["counts"].items():
            counts[key] += value
    words = sum(counts[c] for c in CATEGORIES)
    acoustic_wrong = counts["lm_fixed"] + counts["both_wrong"]
    word_errors = counts["lm_introduced"] + counts["both_wrong"] + counts["lm_inserted"]
    share = lambda a, b: 100.0 * a / b if b else None
    return dict(version=VERSION, label=LABEL, counts=counts, reference_words=words, trials_skipped=skipped,
                pct_words_greedy_phonemes_exact=share(words - acoustic_wrong, words),
                pct_acoustic_word_errors_fixed_by_lm=share(counts["lm_fixed"], acoustic_wrong),
                pct_word_errors_with_exact_phonemes=share(counts["lm_introduced"], word_errors),
                pct_word_errors_inserted=share(counts["lm_inserted"], word_errors))


def analyze(rows, logits_npz, spec, data_dir):
    """Attribution for scored rows (trials.jsonl order) and the logits cache they were decoded from."""
    import h5py
    with np.load(logits_npz) as npz:
        lengths, flat = npz["lengths"], npz["flat"]
    if len(lengths) != len(rows):
        raise ValueError("Logits cache and trial rows differ in length")
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    model = spec["model"]
    lookahead = spec["preprocess"]["effective"].get("smooth_lookahead") or 0
    frame_ms = lambda f: (model["patch_size"] + f * model["patch_stride"] + lookahead) * 20.0
    lexicon = load_lexicon(spec["lm"]["graph_dir"])
    per_trial, handles = [], {}
    try:
        for index, row in enumerate(rows):
            path = Path(data_dir) / row["session"] / f"data_{row['split']}.hdf5"
            handle = handles.get(path) or handles.setdefault(path, h5py.File(path, "r"))
            group = handle[row["trial_key"]]
            phones = np.asarray(group["seq_class_ids"][:])[:int(group.attrs["seq_len"])]
            tokens = greedy_tokens(flat[offsets[index]:offsets[index + 1]])
            trial = attribute_trial(row["ref"], phones, tokens, row["hyp"], lexicon, frame_ms)
            trial.update(i=index, session=row["session"], trial_key=row["trial_key"])
            per_trial.append(trial)
    finally:
        for handle in handles.values():
            handle.close()
    summary = summarize(per_trial)
    summary["lexicon"] = str(Path(spec["lm"]["graph_dir"]) / "lexicon_numbers.txt") if lexicon else None
    return dict(summary=summary, trials=per_trial)


def analyze_run(run_dir, data_dir, cache_root):
    """Attribution for a completed standard run, cached under results/harness/cache/analysis/."""
    run_dir = Path(run_dir)
    cache = Path(cache_root) / "cache" / "analysis" / f"{run_dir.name}_errors_v{VERSION}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("tier") != "standard" or "logits_cache" not in manifest:
        raise ValueError("Error attribution needs a standard run (paced runs keep no logits)")
    with (run_dir / "trials.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    result = analyze(rows, Path(manifest["logits_cache"]["dir"]) / "logits.npz", manifest["pipeline"], data_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in ((summary_path(cache), result["summary"]), (cache, result)):
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    return result


def summary_path(cache):
    return cache.with_name(cache.name.replace(".json", ".summary.json"))


def cached_summary(run_dir, cache_root):
    """The run's attribution summary if it has already been computed (no computation here)."""
    path = summary_path(Path(cache_root) / "cache" / "analysis" / f"{Path(run_dir).name}_errors_v{VERSION}.json")
    return json.loads(path.read_text()) if path.exists() else None
