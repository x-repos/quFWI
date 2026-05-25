#!/bin/bash
#SBATCH --job-name=fwi_2gpu
#SBATCH --partition=gpu
#SBATCH --account=2602090805
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# Go to the project root directory
cd ~/Workspace/quFWI

# Run the training script with 2 GPUs enabled!
# Note: SLURM automatically sets CUDA_VISIBLE_DEVICES to the 2 allocated GPUs,
# so JAX will only see and use those 2 GPUs.
export TF_GPU_ALLOCATOR=cuda_malloc_async
uv run python scripts/rasht/fwi_classical_multigpu.py
