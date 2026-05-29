#!/bin/bash
# Submit all training runs as separate SLURM jobs (one GPU each).
#
# Submits fwi_classical, fwi_quantum, and every variant under variants/.
# Each job logs to results/rasht/logs/slurm_<jobname>_<jobid>.log.

cd "$(dirname "$0")/../.."

mkdir -p results/rasht/logs

MAIN_SCRIPTS=(fwi_classical fwi_quantum)

VARIANTS=(
    alpha10x2 alpha10x3 alpha20x3 alpha40x3 alpha40x5 alpha60x3
    phi16x2 phi64x3
    sub3x2x3 sub8x4x8 overlap50
    lr5e4 lr5e5
    wpde01_wbc01 wpde1_wbc1
)

submit() {
    local name="$1"
    local script="$2"
    echo "Submitting $name"
    sbatch --job-name="$name" --output="results/rasht/logs/slurm_%x_%j.log" \
        -p gpu --account=2602090805 --gres=gpu:v100:1 --ntasks=1 --cpus-per-task=4 \
        --wrap="module load apps/python3 && conda activate jaxcu12 && cd $(pwd)/scripts/rasht && python ${script} $*"
}

for m in "${MAIN_SCRIPTS[@]}"; do
    submit "$m" "${m}.py"
done

for m in "${VARIANTS[@]}"; do
    submit "$m" "variants/${m}.py"
done

echo "Done. Check with: squeue -u \$USER"
