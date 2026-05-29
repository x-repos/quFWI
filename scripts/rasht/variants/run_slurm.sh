#!/bin/bash
cd "$(dirname "$0")/../../.."

for m in overlap50 wpde1_wbc1 wpde01_wbc01 lr5e5 lr5e4 alpha60x3 alpha40x5 alpha40x3 alpha20x3 alpha10x3 alpha10x2 sub8x4x8 sub3x2x3 phi64x3 phi16x2; do
    echo "Submitting $m"
    sbatch --job-name="$m" --output="results/rasht/logs/slurm_%x_%j.log" \
        -p gpu --account=2602090805 --gres=gpu:v100:1 --ntasks=1 --cpus-per-task=4 \
        --wrap="module load apps/python3 && conda activate jaxcu12 && cd $(pwd)/scripts/rasht && python variants/${m}.py"
done

echo "Done. Check with: squeue -u \$USER"
