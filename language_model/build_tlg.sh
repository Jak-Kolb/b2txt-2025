#!/bin/bash
# C28 — build an n-gram LM and its TLG decoding graph.
#
# Wires together the three toolchains the recipe needs, none of which setup_lm.sh builds:
#   SRILM             ngram-count / ngram      -> built by `make MAKE_PIC=yes World` in srilm-1.7.3
#   Kaldi FST tools   arpa2fst, fsttablecompose, fstdeterminizestar, fstminimizeencoded,
#                     fstisstochastic, fstaddselfloops
#                     -> cmake --build <build> --target <each>, in the ALREADY-CONFIGURED tree at
#                        runtime/server/x86/build/temp.linux-x86_64-cpython-39
#   OpenFST           fstcompile / fstarcsort  -> already present in fc_base
#
# path.sh points at runtime/server/x86/build, but setup.py configured the tree one level deeper,
# so the kaldi binaries are NOT where path.sh expects. This script prepends the real locations.
#
# Vocabulary comes from the DICT (133,854 CMUdict entries), not from the corpus: build_lm.sh
# passes `-limit-vocab -vocab lexicons.txt`. The corpus only supplies the n-gram statistics.
# make_tlg.sh then greps out <unk>, so a word the corpus never saw has no path in G.fst at all.
#
# Usage:
#   ./build_tlg.sh <output_dir> <corpus.txt> <order> [prune_threshold] [dict]
# e.g.
#   ./build_tlg.sh $PWD/../results/lm_indomain_4gram ../results/train_sentences.txt 4 0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S0="$HERE/examples/speech/s0"
BUILD="$HERE/runtime/server/x86/build/temp.linux-x86_64-cpython-39"
OPENFST="$HERE/runtime/server/x86/fc_base/openfst-subbuild/openfst-populate-prefix/bin"
LM_PY="${LM_PY:-$(conda info --base 2>/dev/null)/envs/b2txt25_lm/bin/python}"  # needs nltk + py3.9

OUT_DIR="${1:?usage: build_tlg.sh <output_dir> <corpus.txt> <order> [prune] [dict]}"
CORPUS="${2:?need a corpus text file, one sentence per line}"
ORDER="${3:?need an n-gram order}"
PRUNE="${4:-0}"
DICT="${5:-$S0/dict.txt}"
SIL_PROB=0.9

for tool in ngram-count ngram; do
    command -v "$tool" >/dev/null || export PATH="$PATH:$HERE/srilm-1.7.3/bin/i686-m64"
done
export PATH="$BUILD/kaldi:$OPENFST:$PATH"

for tool in ngram-count arpa2fst fsttablecompose fstdeterminizestar fstminimizeencoded \
            fstisstochastic fstaddselfloops fstcompile fstarcsort; do
    command -v "$tool" >/dev/null || { echo "MISSING: $tool"; exit 1; }
done

# run.sh calls bare `python` and needs nltk; expose the LM env as `python` without activating it.
SHIM="$(mktemp -d)"; trap 'rm -rf "$SHIM"' EXIT
ln -sf "$LM_PY" "$SHIM/python"
export PATH="$SHIM:$PATH"

CORPUS="$(cd "$(dirname "$CORPUS")" && pwd)/$(basename "$CORPUS")"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "=== C28 TLG build ==="
echo "  corpus : $CORPUS  ($(wc -l < "$CORPUS") lines)"
echo "  dict   : $DICT  ($(wc -l < "$DICT") entries)"
echo "  order  : $ORDER   prune: $PRUNE   sil_prob: $SIL_PROB"
echo "  output : $OUT_DIR"
echo

cd "$S0"
# run.sh takes 8 positional args under `set -u`; the notebook's 7-arg call fails on $8.
bash run.sh "$OUT_DIR" "$DICT" "$CORPUS" "$SIL_PROB" none "$PRUNE" "$ORDER" 0

echo
echo "=== built ==="
ls -la "$OUT_DIR/data/lang_test/TLG.fst" "$OUT_DIR/data/lang_test/words.txt"
echo "decode with:  stream_lm.py decode --lm $OUT_DIR/data/lang_test"
