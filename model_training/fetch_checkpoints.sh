#!/bin/bash
# Pull causal_la0 / causal_la4 checkpoints from the Mac and install them.
#
# WSL2 runs in NAT mode here, so the Mac CANNOT scp *into* this box without an admin-side
# netsh portproxy or networkingMode=mirrored (the latter needs `wsl --shutdown`, which would
# kill any training run in flight). Outbound from WSL works fine, so we pull instead.
#
# On the Mac, enable: System Settings -> General -> Sharing -> Remote Login
#
# Usage:
#   ./fetch_checkpoints.sh <mac-user>@<mac-ip>:<dir-holding-causal_la0/causal_la4>
# e.g.
#   ./fetch_checkpoints.sh <user>@<mac-ip>:~/path/to/nejm-brain-to-text/results
#
# Handles either layout: *.zip archives, or the unzipped causal_la0/ causal_la4/ directories.
# Authenticates ONCE via an ssh ControlMaster, so you type the password a single time.
set -uo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <mac-user>@<mac-ip>:<dir-holding-causal_la0/causal_la4>"
    echo "   eg: $0 <user>@<mac-ip>:~/path/to/nejm-brain-to-text/results"
    echo
    echo "Find the Mac's IP on the Mac with:  ipconfig getifaddr en0"
    echo "Find this box with: hostname -I"
    exit 1
fi

ARG="$1"
HOST="${ARG%%:*}"
DIR="${ARG#*:}"
[ "$HOST" = "$ARG" ] && { echo "error: need <user>@<host>:<dir>, with a colon"; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/trained_models/_incoming"
mkdir -p "$DEST"

CTL="/tmp/.fetch_ckpt_ctl_$$"
cleanup() { ssh -S "$CTL" -O exit "$HOST" 2>/dev/null; }
trap cleanup EXIT

echo "==> connecting to $HOST (one password prompt for the whole transfer)"
ssh -M -S "$CTL" -fN -o ControlPersist=300 "$HOST" || {
    echo "ssh failed. Check Remote Login is on, and that the IP is right (ipconfig getifaddr en0)."
    exit 1
}
SSH() { ssh -S "$CTL" "$HOST" "$@"; }

echo "==> looking in $DIR"
SSH "ls -la -- $DIR" 2>/dev/null | sed 's/^/    /' || { echo "cannot list $DIR"; exit 1; }

pulled=0
for tag in causal_la0 causal_la4; do
    # 1) any zip whose name contains the tag
    zips=$(SSH "ls -1 -- $DIR/*${tag}*.zip 2>/dev/null" | tr -d '\r')
    if [ -n "$zips" ]; then
        while IFS= read -r z; do
            [ -z "$z" ] && continue
            echo "==> $tag: pulling archive $(basename "$z")"
            scp -o ControlPath="$CTL" "$HOST:$z" "$DEST/" && pulled=$((pulled+1))
        done <<< "$zips"
        continue
    fi
    # 2) otherwise the unzipped directory
    if SSH "test -d $DIR/$tag" 2>/dev/null; then
        echo "==> $tag: no zip found, pulling the directory itself"
        rm -rf "${DEST:?}/$tag"
        scp -r -o ControlPath="$CTL" "$HOST:$DIR/$tag" "$DEST/" && pulled=$((pulled+1))
    else
        echo "==> $tag: NOT FOUND in $DIR (neither *${tag}*.zip nor $tag/)"
    fi
done

echo
if [ "$pulled" -eq 0 ]; then
    echo "nothing pulled. Check the listing above for the real names, then either re-run with the"
    echo "correct directory or copy the files by hand into:"
    echo "  $DEST"
    exit 1
fi

echo "==> pulled into _incoming:"
ls -lh "$DEST" | tail -n +2 | awk '{print "   ", $5, $9}'

echo
echo "==> installing"
"$HERE/../.venv/bin/python" "$HERE/install_reference_checkpoints.py"
