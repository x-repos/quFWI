#!/usr/bin/env python3
"""Plot PQC benchmark results from pqc_benchmark.npz.

Usage:
    python plot_pqc_benchmark.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "pqc_benchmark.npz")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "plots")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        print("Run: python compare_jax_vs_pennylane.py first")
        return

    d = dict(np.load(DATA_PATH, allow_pickle=True))
    os.makedirs(OUT_DIR, exist_ok=True)

    configs_q = d["configs_qubits"]
    configs_l = d["configs_layers"]

    # ── Plot 1: Gradient agreement ──
    fig, ax = plt.subplots(figsize=(5.3, 5.3))
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs_q)))
    all_lim = 0.0

    for i, (nq, nl) in enumerate(zip(configs_q, configs_l)):
        key = f"{nq}q_{nl}L"
        jax_g = d[f"{key}_jax_grad"]
        pl_g = d[f"{key}_pl_grad"]
        max_diff = float(d[f"{key}_max_diff"])
        all_lim = max(all_lim, np.max(np.abs(jax_g)), np.max(np.abs(pl_g)))

        n_params = jax_g.size
        label = f"{nq}q, {nl}L ($\\Delta$={max_diff:.1e})"
        ax.scatter(jax_g, pl_g, s=50, alpha=0.7, color=colors[i],
                   edgecolors='k', linewidths=0.3, label=label)

    lim = all_lim * 1.2
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='$y = x$', zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"JAX (ours) $\partial f / \partial \theta$", fontsize=12)
    ax.set_ylabel(r"PennyLane $\partial f / \partial \theta$", fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "pqc_gradient_agreement.pdf")
    fig.savefig(p1, bbox_inches="tight")
    print(f"Saved {p1}")
    plt.close(fig)

    print(f"\nPlot saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
