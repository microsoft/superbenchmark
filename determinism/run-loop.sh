#!/usr/bin/env bash
# Back-to-back SDC soak runner (Approach B, hash). Runs the hash config N times
# with NO cooldown between runs so the GPUs stay hot across the whole campaign.
# After each run it records the output dir; compare-hashes.py then checks whether
# the per-checkpoint / combined activation hashes ever diverge (self-consistency)
# and, if a gold baseline is given, against gold.
#
# Usage:
#   determinism/run-loop.sh [N] [CONFIG]
#     N       number of back-to-back runs (default 4)
#     CONFIG  sb config (default determinism/llama2-7b-det-6h-hash.yaml)
#
# Survives SSH disconnect (launch with nohup ... & as below). Does NOT survive the
# container being stopped.
#
#   nohup determinism/run-loop.sh 4 > temp/logs/soak-loop.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

N="${1:-4}"
CONFIG="${2:-determinism/llama2-7b-det-6h-hash.yaml}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="temp/logs/soak-${STAMP}-outputs.txt"
mkdir -p temp/logs
: > "$MANIFEST"

echo "[soak] $N runs of $CONFIG; manifest -> $MANIFEST"
for i in $(seq 1 "$N"); do
    echo "[soak] === run $i/$N started $(date -Is) ==="
    before="$(ls -d outputs/2026-* 2>/dev/null | sort | tail -1)"
    sb run --no-docker --host-list localhost -c "$CONFIG"
    rc=$?
    after="$(ls -dt outputs/2026-* 2>/dev/null | head -1)"
    echo "[soak] run $i return code $rc -> $after"
    if [[ -f "$after/results-summary.jsonl" ]]; then
        echo "$after/results-summary.jsonl" >> "$MANIFEST"
    else
        echo "[soak] WARNING: no results-summary.jsonl for run $i" >&2
    fi
    echo "[soak] === run $i/$N finished $(date -Is) ==="
done

echo "[soak] all runs done. Comparing hashes..."
python3 determinism/compare-hashes.py "$MANIFEST" \
    --gold new_tests_determinis_gold_data/baseline.json || true
echo "[soak] complete."
