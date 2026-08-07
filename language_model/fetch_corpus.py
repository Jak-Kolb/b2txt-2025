"""C28 — pull an OpenWebText subset and format it as an LM corpus.

Why a subset and not the whole thing
------------------------------------
Full OpenWebText is ~38 GB / ~9B tokens. SRILM's `ngram-count` holds counts in RAM, so a 4-gram
over 9B tokens needs the make-batch-counts / merge-batch-counts / make-big-lm path and far more
than this box has. A few GB is enough to cover the vocabulary and give the 4-gram real sequential
statistics; the in-domain LM supplies domain fit through interpolation (see interpolate_lm.sh).

Streams from the hub so only the shards actually consumed are downloaded, and stops at a target
size rather than a document count.

Text handling matches what the decoder can emit, which is the part that actually matters:
  * lowercase, since the decode is scored lowercased
  * one sentence per line (nltk punkt), because the LM is sentence-level with <s>/</s>
  * drop anything containing a character outside the lexicon's alphabet -- URLs, code, markup
    and non-English lines contribute vocabulary the decoder can never emit and dilute the counts

    cd language_model && ../.venv/bin/python fetch_corpus.py --target_gb 8
"""
import argparse
import os
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "utils"))
from convert_number_to_words import number_to_words  # noqa: E402

# Keep letters, apostrophes and sentence-internal hyphens; everything else marks a line as junk.
_OK = re.compile(r"^[a-z' \-]+$")
_STRIP = re.compile(r"[^\w\s'\-]")
_DASHES = re.compile(r"-{2,}")
_JUNK = re.compile(r"https?://|www\.|\S+@\S+|[#@]\w+|\w+\.(com|org|net|edu|gov|io|co)\b", re.I)
_WS = re.compile(r"\s+")


def clean(sentence):
    # Drop URLs/emails/handles BEFORE punctuation is stripped. Afterwards "https://foo.com"
    # survives as the innocuous-looking token sequence "https foo com" and passes every later
    # check, quietly poisoning the counts.
    if _JUNK.search(sentence):
        return None
    # The lexicon contains exactly TWO entries with digits (3-D, 3D), so a numeral is an OOV token
    # the decoder can never emit. Dropping every sentence with a digit would throw away a large and
    # biased slice of web text, so spell them out instead -- the repo ships
    # utils/convert_number_to_words.py for precisely this.
    if any(ch.isdigit() for ch in sentence):
        try:
            sentence = number_to_words(sentence)
        except Exception:
            return None          # num2words chokes on some malformed numerics; drop those lines
    s = _STRIP.sub(" ", sentence.lower())
    # Em-dashes survive as "--" and would glue two words into one OOV token ("dog--the").
    # Single sentence-internal hyphens are kept: the lexicon has entries like "3-D".
    s = _DASHES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    s = " ".join(t.strip("-") for t in s.split() if t.strip("-"))
    if not s or not _OK.match(s):
        return None
    n = s.count(" ") + 1
    return s if 2 <= n <= 60 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Skylion007/openwebtext")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=str(REPO / "results/corpus/openwebtext_subset.txt"))
    ap.add_argument("--target_gb", type=float, default=8.0)
    ap.add_argument("--report_every", type=int, default=200000)
    a = ap.parse_args()

    import datasets
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    target = int(a.target_gb * 1e9)

    print(f"streaming {a.dataset}:{a.split} -> {out}  (target {a.target_gb} GB)")
    ds = datasets.load_dataset(a.dataset, split=a.split, streaming=True,
                               trust_remote_code=True)

    # Write to .partial and rename only on success. A fetch killed midway (a WSL restart, a
    # dropped connection) otherwise leaves a short file that looks like a finished corpus, and
    # you find out from a quietly undertrained LM hours later.
    partial = out.with_suffix(out.suffix + ".partial")
    written = kept = seen = docs = 0
    t0 = time.time()
    with open(partial, "w") as fh:
        for doc in ds:
            docs += 1
            for sent in nltk.sent_tokenize(doc["text"]):
                seen += 1
                c = clean(sent)
                if c is None:
                    continue
                fh.write(c + "\n")
                kept += 1
                written += len(c) + 1
            if written >= target:
                break
            if kept and kept % a.report_every < 50:
                el = time.time() - t0
                print(f"  {docs:>8} docs  {kept:>10} sentences  {written/1e9:.2f} GB  "
                      f"{el:.0f}s", flush=True)

    partial.rename(out)
    print(f"\ndone: {docs} docs, {kept}/{seen} sentences kept ({100*kept/max(seen,1):.1f} %), "
          f"{written/1e9:.2f} GB in {time.time()-t0:.0f}s")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # datasets' streaming reader crashes in PyGILState_Release during interpreter finalization
    # (a library teardown bug, after all our work is done and fsynced). Exiting hard turns a
    # spurious core dump into a clean exit code, so a real failure stays distinguishable.
    os._exit(code)
