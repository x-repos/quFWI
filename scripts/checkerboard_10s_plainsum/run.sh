#!/bin/bash
RESULTS_DIR="$(cd "$(dirname "$0")/../.." && pwd)/results/checkerboard_10s_plainsum"
mkdir -p "$RESULTS_DIR/logs"
nohup conda run -n jaxcu12 python fwi_classical.py --resume 2000000 "$@" >> "$RESULTS_DIR/logs/fwi_classical.log" 2>&1 &
echo "PID: $!"
