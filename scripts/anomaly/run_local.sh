#!/bin/bash
# Run all training scripts locally (sequential, one GPU).
#
# Runs fwi_classical, fwi_quantum, and every variant under variants/.
# Each script logs to results/anomaly/logs/<name>.log. The orchestrator
# itself logs to results/anomaly/logs/run_all.log.
#
# Usage:
#   ./run_local.sh [extra args passed to every python invocation]

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$(cd "$SCRIPTS_DIR/../.." && pwd)/results/anomaly"
mkdir -p "$RESULTS_DIR/logs"

MAIN_SCRIPTS=(fwi_classical fwi_quantum)

VARIANTS=(
    alpha10x2 alpha10x3 alpha20x3 alpha40x3 alpha40x5 alpha60x3
    phi16x2 phi64x3
    sub3x2x3 sub8x4x8 overlap50
    lr5e4 lr5e5
    wpde01_wbc01 wpde1_wbc1
)

nohup bash -c '
cd "'"$SCRIPTS_DIR"'"
RESULTS_DIR="'"$RESULTS_DIR"'"

for m in '"${MAIN_SCRIPTS[@]}"'; do
    echo "=== [$(date)] Starting $m ==="
    conda run -n jaxcu12 python ${m}.py "$@" >> "$RESULTS_DIR/logs/${m}.log" 2>&1
    echo "=== [$(date)] Finished $m ==="
done

for m in '"${VARIANTS[@]}"'; do
    echo "=== [$(date)] Starting variant $m ==="
    conda run -n jaxcu12 python variants/${m}.py "$@" >> "$RESULTS_DIR/logs/${m}.log" 2>&1
    echo "=== [$(date)] Finished variant $m ==="
done

echo "All done."
' "$@" > "$RESULTS_DIR/logs/run_all.log" 2>&1 &

echo "PID: $!"
echo "Progress: tail -f $RESULTS_DIR/logs/run_all.log"
