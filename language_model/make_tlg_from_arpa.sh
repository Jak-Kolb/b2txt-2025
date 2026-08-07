#!/bin/bash
# C28 — compose TLG from an already-built ARPA, reusing L.fst/T.fst.
#
# L.fst (lexicon) and T.fst (CTC token topology) are functions of the DICT alone, not the corpus
# or the LM, so they are identical across every LM built against the same lexicon. Rebuilding them
# per LM costs minutes and buys nothing; this reuses a prepared lang_phn directory and runs only
# the composition, which is the stage that actually differs and the stage that dominates RAM.
#
# Composition peak RSS runs ~8x the resulting TLG (measured: 1.6 GB TLG at 13.0 GB peak), so this
# is the step that decides how large an LM this machine can ship.
#
# Usage:
#   ./make_tlg_from_arpa.sh <arpa> <out_dir> <src_lang_dir>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S0="$HERE/examples/speech/s0"
BUILD="$HERE/runtime/server/x86/build/temp.linux-x86_64-cpython-39"
OPENFST="$HERE/runtime/server/x86/fc_base/openfst-subbuild/openfst-populate-prefix/bin"
export PATH="$BUILD/kaldi:$OPENFST:$HERE/srilm-1.7.3/bin/i686-m64:$PATH"

ARPA="$(readlink -f "${1:?usage: make_tlg_from_arpa.sh <arpa> <out_dir> <src_lang_dir>}")"
OUT_DIR="${2:?need an output dir}"
SRC_LANG="$(readlink -f "${3:?need a prepared lang_phn dir (has L.fst, T.fst, words.txt)}")"

for f in L.fst T.fst words.txt; do
    [ -f "$SRC_LANG/$f" ] || { echo "MISSING $SRC_LANG/$f"; exit 1; }
done

mkdir -p "$OUT_DIR/data/local/lm"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
ln -sf "$ARPA" "$OUT_DIR/data/local/lm/lm.arpa"

echo "=== composing TLG ==="
echo "  arpa : $ARPA  ($(du -h "$ARPA" | cut -f1))"
grep -E "^ngram [0-9]=" "$ARPA" | sed 's/^/         /'
echo "  lang : $SRC_LANG"
echo "  out  : $OUT_DIR/data/lang_test"
echo

cd "$S0"
# Keep the FULL log. An earlier version grepped for a whitelist of interesting lines, which threw
# away the only diagnostics when a stage failed and left a 0-byte TLG with no explanation.
/usr/bin/time -v tools/fst/make_tlg.sh "$OUT_DIR/data/local/lm" "$SRC_LANG" \
    "$OUT_DIR/data/lang_test" 2>&1 | grep -viE "^I[0-9]{4}|WARNING: Logging"

# make_tlg.sh redirects into TLG.fst, so the file exists (empty) from the instant the last stage
# starts. Never treat its existence as completion -- check that it is non-empty.
TLG="$OUT_DIR/data/lang_test/TLG.fst"
[ -s "$TLG" ] || { echo "FAILED: $TLG is empty"; exit 1; }

echo
ls -la "$OUT_DIR/data/lang_test/TLG.fst"
