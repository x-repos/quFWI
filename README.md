# `quFWI`: Accelerating physics-informed neural networks for full waveform inversion using a hybrid quantum-classical finite-basis architecture

Hybrid quantum-classical Full Waveform Inversion (FWI) using FBPINNs (Finite Basis Physics-Informed Neural Networks) with JAX.

## Overview

This repository implements the acoustic FWI problem using both:
- **Classical FBPINNs** (`scripts/fwi_classical.py`) — fully connected networks
- **Quantum-hybrid FBPINNs** (`scripts/fwi_quantum.py`) — classical layers + variational quantum circuit (PQC)

The quantum circuit implementation (`src/qufwi/pqcs/`) is a standalone JAX module with no fbpinns dependency, enabling future comparison work (e.g., PennyLane).

## Installation

```bash
uv sync
```

This installs all dependencies including JAX with CUDA 12 GPU support. To run scripts:

```bash
uv run python scripts/anomaly/fwi_classical.py
```

## Usage

```bash
# Classical FWI
uv run python scripts/anomaly/fwi_classical.py

# Quantum-hybrid FWI
uv run python scripts/anomaly/fwi_quantum.py

# Resume from checkpoint
uv run python scripts/anomaly/fwi_classical.py --resume 200000

# Monitor training
tensorboard --logdir results/anomaly/summaries/
```

## Results

<p align="center">
<img src="results/anomaly/plots/l1_and_velocity.png" width="600">
</p>

**(a)** True velocity model with ellipsoidal anomaly. **(b)** Initial (homogeneous) guess. **(c)** Inverted velocity from the classical FBPINN baseline. **(d)** L1 velocity error convergence across hyperparameter variants and the quantum-hybrid model.

## Repository Structure

- `src/qufwi/pqcs/` — Standalone PQC library
- `src/qufwi/fbpinns/` — Core FBPINNs framework (merged classical + quantum)
- `scripts/` — FWI training scripts for anomaly and checkerboard models
- `data/` — SPECFEM synthetic seismic data
- `pennylane/` — Tests and benchmarks comparing the custom PQC against PennyLane

## Dependencies

Managed via `pyproject.toml` and `uv.lock`. Core dependencies:

- JAX >= 0.8 (with CUDA 12)
- Optax >= 0.2
- NumPy, SciPy, Matplotlib, TensorboardX

## Citation
Please cite the following if you use quFWI in your research
```bash
@misc{nguyen2026acceleratingphysicsinformedneuralnetworks,
      title={Accelerating physics-informed neural networks for full waveform inversion using a hybrid quantum-classical finite-basis architecture}, 
      author={Hoang Anh Nguyen and Divakar Vashisth and Ali Tura},
      year={2026},
      eprint={2606.01110},
      archivePrefix={arXiv},
      primaryClass={physics.geo-ph},
      url={https://arxiv.org/abs/2606.01110}, 
}
```
