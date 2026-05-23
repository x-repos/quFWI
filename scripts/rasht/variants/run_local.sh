#!/bin/bash
SCRIPTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$(cd "$SCRIPTS_DIR/../.." && pwd)/results/rasht"
cd "$SCRIPTS_DIR"
mkdir -p "$RESULTS_DIR/logs"

nohup bash -c '
cd "'"$SCRIPTS_DIR"'"
RESULTS_DIR="'"$RESULTS_DIR"'"
for m in lr5e4 lr5e5 wpde01_wbc01 wpde1_wbc1 overlap50; do
    echo "=== Starting $m ($(date)) ==="
    python variants/${m}.py >> "$RESULTS_DIR/logs/${m}.log" 2>&1
    echo "=== Finished $m ($(date)) ==="
done
echo "All done."
' > "$RESULTS_DIR/logs/run_all.log" 2>&1 &

echo "PID: $!"
echo "Progress: tail -f $RESULTS_DIR/logs/run_all.log"
