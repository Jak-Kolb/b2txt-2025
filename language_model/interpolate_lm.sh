#!/bin/bash
# C28 — interpolate the general (OpenWebText) 4-gram with the in-domain 4-gram, then build TLG.
#
# The two LMs answer different questions. The general LM supplies coverage: it removes the 6.78 %
# of val reference tokens that the in-domain LM has no path for at all. The in-domain LM supplies
# domain fit, which measured -19.1 points on its own. Interpolation keeps both.
#
#   p(w|h) = lambda * p_indomain(w|h) + (1-lambda) * p_general(w|h)
#
# lambda is chosen by perplexity on a held-out set. Use val-dev text -- that is what val-dev is
# for -- and NEVER val-test. Note this makes lambda a tuned hyperparameter of the deployable
# system, so it must be reported as such.
#
# Usage:
#   ./interpolate_lm.sh <general_arpa> <indomain_arpa> <heldout.txt> <out_dir> [order]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$PATH:$HERE/srilm-1.7.3/bin/i686-m64"

GENERAL="${1:?usage: interpolate_lm.sh <general_arpa> <indomain_arpa> <heldout.txt> <out_dir> [order]}"
INDOMAIN="${2:?need the in-domain arpa}"
HELDOUT="${3:?need a held-out text file for lambda selection}"
OUT_DIR="${4:?need an output dir}"
ORDER="${5:-4}"

command -v ngram >/dev/null || { echo "MISSING: ngram (build SRILM first)"; exit 1; }
mkdir -p "$OUT_DIR"

echo "=== selecting lambda by held-out perplexity ==="
BEST_LAMBDA=""; BEST_PPL=""
for LAMBDA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    PPL=$(ngram -order "$ORDER" -lm "$INDOMAIN" -mix-lm "$GENERAL" -lambda "$LAMBDA" \
              -ppl "$HELDOUT" 2>/dev/null | awk '/ppl=/ {for(i=1;i<=NF;i++) if($i=="ppl=") print $(i+1)}' | tail -1)
    [ -z "$PPL" ] && { echo "  lambda $LAMBDA -> ngram failed"; continue; }
    printf "  lambda %-4s ppl %s\n" "$LAMBDA" "$PPL"
    if [ -z "$BEST_PPL" ] || awk "BEGIN{exit !($PPL < $BEST_PPL)}"; then
        BEST_PPL=$PPL; BEST_LAMBDA=$LAMBDA
    fi
done
[ -z "$BEST_LAMBDA" ] && { echo "no lambda scored; aborting"; exit 1; }
echo "  -> best lambda=$BEST_LAMBDA (ppl $BEST_PPL).  lambda weights the IN-DOMAIN model."

echo
echo "=== writing mixed ARPA ==="
ngram -order "$ORDER" -lm "$INDOMAIN" -mix-lm "$GENERAL" -lambda "$BEST_LAMBDA" \
      -write-lm "$OUT_DIR/lm_mixed.arpa"
echo "$BEST_LAMBDA" > "$OUT_DIR/lambda.txt"
ls -la "$OUT_DIR/lm_mixed.arpa"
echo
echo "Next: prune to fit RAM, then rebuild TLG against this ARPA."
echo "  ngram -prune <thresh> -order $ORDER -lm $OUT_DIR/lm_mixed.arpa -write-lm $OUT_DIR/lm.arpa"
