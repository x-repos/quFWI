#!/usr/bin/env python3
"""Benchmark JAX PQC (pqcs/) vs PennyLane: saves all data to .npz for replotting.

Usage:
    pip install pennylane
    python compare_jax_vs_pennylane.py
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jnp
from jax import grad, jit

import pennylane as qml

from pqcs.circuit import quantum_circuit

SEED = 42
N_TIMING_RUNS = 100
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def make_params(n_qubits, n_layers, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_qubits,)).astype(np.float32)
    basis = rng.normal(size=(n_qubits,)).astype(np.float32) * 0.5
    theta = rng.normal(size=(n_layers, n_qubits, 3)).astype(np.float32) * 0.1
    return jnp.array(x), jnp.array(basis), jnp.array(theta)


def build_jax_fns(n_qubits, n_layers):
    def fwd(x, basis, theta):
        return quantum_circuit(x, basis, theta, n_qubits, n_layers)
    return jit(fwd), jit(grad(fwd, argnums=2))


def build_pennylane_fns(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def circuit(x, basis, theta):
        for i in range(n_qubits):
            qml.RY(basis[i] * x[i], wires=i)
        for layer in range(n_layers):
            for wire in range(n_qubits):
                qml.RX(theta[layer, wire, 0], wires=wire)
                qml.RY(theta[layer, wire, 1], wires=wire)
                qml.RZ(theta[layer, wire, 2], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def fwd(x, basis, theta):
        return jnp.mean(jnp.array(circuit(x, basis, theta)))

    return jit(fwd), jit(grad(fwd, argnums=2))


def time_fn(fn, *args, n=N_TIMING_RUNS):
    for _ in range(5):
        fn(*args).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args).block_until_ready()
    return (time.perf_counter() - t0) / n * 1000  # ms


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Gradient agreement across circuit sizes ──
    grad_configs = [(2, 1), (2, 3), (4, 1), (4, 3), (4, 6), (6, 3), (6, 6), (8, 3)]
    grad_data = {}  # keyed by "2q_1L" etc.

    print("=== Gradient agreement ===")
    for nq, nl in grad_configs:
        key = f"{nq}q_{nl}L"
        x, basis, theta = make_params(nq, nl)
        jax_fwd, jax_grd = build_jax_fns(nq, nl)
        pl_fwd, pl_grd = build_pennylane_fns(nq, nl)

        jax_out = float(jax_fwd(x, basis, theta))
        pl_out = float(pl_fwd(x, basis, theta))
        jax_g = np.asarray(jax_grd(x, basis, theta)).ravel()
        pl_g = np.asarray(pl_grd(x, basis, theta)).ravel()
        max_diff = np.max(np.abs(jax_g - pl_g))

        grad_data[f"{key}_jax_grad"] = jax_g
        grad_data[f"{key}_pl_grad"] = pl_g
        grad_data[f"{key}_jax_fwd"] = np.array(jax_out)
        grad_data[f"{key}_pl_fwd"] = np.array(pl_out)
        grad_data[f"{key}_max_diff"] = np.array(max_diff)

        print(f"  {key}: fwd diff={abs(jax_out-pl_out):.2e}, "
              f"grad max|diff|={max_diff:.2e}, n_params={jax_g.size}")

    grad_data["configs_qubits"] = np.array([c[0] for c in grad_configs])
    grad_data["configs_layers"] = np.array([c[1] for c in grad_configs])

    # ── 2. Speed scaling vs qubits ──
    qubit_range = np.array([2, 3, 4, 5, 6, 7, 8])
    fixed_layers = 3
    jax_fwd_q, pl_fwd_q, jax_grad_q, pl_grad_q = [], [], [], []

    print("\n=== Scaling vs qubits ===")
    for nq in qubit_range:
        print(f"  {nq} qubits, {fixed_layers} layers ...", end=" ", flush=True)
        x, basis, theta = make_params(int(nq), fixed_layers)
        jf, jg = build_jax_fns(int(nq), fixed_layers)
        pf, pg = build_pennylane_fns(int(nq), fixed_layers)

        jax_fwd_q.append(time_fn(jf, x, basis, theta))
        pl_fwd_q.append(time_fn(pf, x, basis, theta))
        jax_grad_q.append(time_fn(jg, x, basis, theta))
        pl_grad_q.append(time_fn(pg, x, basis, theta))
        print(f"JAX fwd={jax_fwd_q[-1]:.2f}ms, PL fwd={pl_fwd_q[-1]:.2f}ms")

    # ── 3. Speed scaling vs layers ──
    layer_range = np.array([1, 2, 3, 5, 8, 12])
    fixed_qubits = 4
    jax_fwd_l, pl_fwd_l, jax_grad_l, pl_grad_l = [], [], [], []

    print("\n=== Scaling vs layers ===")
    for nl in layer_range:
        print(f"  {fixed_qubits} qubits, {nl} layers ...", end=" ", flush=True)
        x, basis, theta = make_params(fixed_qubits, int(nl))
        jf, jg = build_jax_fns(fixed_qubits, int(nl))
        pf, pg = build_pennylane_fns(fixed_qubits, int(nl))

        jax_fwd_l.append(time_fn(jf, x, basis, theta))
        pl_fwd_l.append(time_fn(pf, x, basis, theta))
        jax_grad_l.append(time_fn(jg, x, basis, theta))
        pl_grad_l.append(time_fn(pg, x, basis, theta))
        print(f"JAX grad={jax_grad_l[-1]:.2f}ms, PL grad={pl_grad_l[-1]:.2f}ms")

    # ── Save everything ──
    out_path = os.path.join(OUT_DIR, "pqc_benchmark.npz")
    np.savez(
        out_path,
        # Gradient agreement
        **grad_data,
        # Scaling vs qubits
        qubit_range=qubit_range,
        fixed_layers_for_qubits=np.array(fixed_layers),
        jax_fwd_time_qubits=np.array(jax_fwd_q),
        pl_fwd_time_qubits=np.array(pl_fwd_q),
        jax_grad_time_qubits=np.array(jax_grad_q),
        pl_grad_time_qubits=np.array(pl_grad_q),
        # Scaling vs layers
        layer_range=layer_range,
        fixed_qubits_for_layers=np.array(fixed_qubits),
        jax_fwd_time_layers=np.array(jax_fwd_l),
        pl_fwd_time_layers=np.array(pl_fwd_l),
        jax_grad_time_layers=np.array(jax_grad_l),
        pl_grad_time_layers=np.array(pl_grad_l),
    )
    print(f"\nAll data saved to {out_path}")
    print("Replot with: python plot_pqc_benchmark.py")


if __name__ == "__main__":
    main()
