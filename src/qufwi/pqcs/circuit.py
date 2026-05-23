"""Variational quantum circuit (PQC) pipeline: embedding, variational layers, measurement.

Pure JAX implementation — no fbpinns dependency.
"""

import jax.numpy as jnp

from qufwi.pqcs.gates import quantum_gates, apply_single_qubit_gate, apply_cnot


def expectation_pauli_z(state, qubit, n_qubits):
    """Compute <Z> expectation for a qubit"""
    state_tensor = state.reshape([2] * n_qubits)

    indices_0 = [slice(None)] * n_qubits
    indices_0[qubit] = 0
    prob_0 = jnp.sum(jnp.abs(state_tensor[tuple(indices_0)])**2)

    indices_1 = [slice(None)] * n_qubits
    indices_1[qubit] = 1
    prob_1 = jnp.sum(jnp.abs(state_tensor[tuple(indices_1)])**2)

    return prob_0 - prob_1


def quantum_circuit_vectorized(x_classical, basis, theta, n_qubits, n_quantum_layers):
    """Optimized Variational Quantum Circuit using vectorized operations

    Key optimizations:
    1. Fused rotation gates (3 gates -> 1 combined gate)
    2. Reduced gate applications from 3 to 1 per qubit per layer
    3. Single reshape at the end for all measurements

    Args:
        x_classical: Output from classical layers (should have n_qubits components)
        basis: Basis parameters for embedding [n_qubits]
        theta: Variational parameters [n_layers, n_qubits, 3]
        n_qubits: Number of qubits
        n_quantum_layers: Number of variational layers

    Returns:
        Expectation value (scalar)
    """
    # Get n_qubits and n_quantum_layers from actual array shapes (JAX-friendly)
    n_qubits_actual = basis.shape[0]
    n_quantum_layers_actual = theta.shape[0]

    # Initialize state |0...0> (using complex64 to save memory)
    state = jnp.zeros(2**n_qubits_actual, dtype=jnp.complex64)
    state = state.at[0].set(1.0)

    # Get gate functions
    RX, RY, RZ = quantum_gates(theta)

    # Embedding layer: encode classical data into quantum state
    for i in range(n_qubits_actual):
        angle = basis[i] * x_classical[i] if i < len(x_classical) else basis[i] * jnp.mean(x_classical)
        gate = RY(angle)
        state = apply_single_qubit_gate(state, gate, i, n_qubits_actual)

    # Variational layers
    for layer in range(n_quantum_layers_actual):
        # Apply rotations with FUSED gates (KEY OPTIMIZATION)
        # Instead of applying RX, RY, RZ separately, combine them into one gate
        for wire in range(n_qubits_actual):
            # Get rotation angles
            rx_angle = theta[layer, wire, 0]
            ry_angle = theta[layer, wire, 1]
            rz_angle = theta[layer, wire, 2]

            # OPTIMIZATION: Fuse 3 rotation gates into 1 combined gate
            # This reduces reshape/flatten operations from 3 to 1 per qubit
            combined_gate = RZ(rz_angle) @ RY(ry_angle) @ RX(rx_angle)
            state = apply_single_qubit_gate(state, combined_gate, wire, n_qubits_actual)

        # Entanglement (CNOT ladder)
        for wire in range(n_qubits_actual - 1):
            state = apply_cnot(state, wire, wire + 1, n_qubits_actual)

    # Measurement: Compute Z expectations
    # OPTIMIZATION: Reshape once at the end instead of in each expectation call
    state_tensor = state.reshape([2] * n_qubits_actual)

    # Compute all Z expectations
    expectation = 0.0
    for i in range(n_qubits_actual):
        indices_0 = [slice(None)] * n_qubits_actual
        indices_0[i] = 0
        prob_0 = jnp.sum(jnp.abs(state_tensor[tuple(indices_0)])**2)

        indices_1 = [slice(None)] * n_qubits_actual
        indices_1[i] = 1
        prob_1 = jnp.sum(jnp.abs(state_tensor[tuple(indices_1)])**2)

        expectation += prob_0 - prob_1

    # Normalize by number of qubits to keep output in [-1, 1] range
    return jnp.real(expectation) / n_qubits_actual


def quantum_circuit(x_classical, basis, theta, n_qubits, n_quantum_layers):
    """Variational Quantum Circuit (calls optimized version)

    Args:
        x_classical: Output from classical layers (should have n_qubits components)
        basis: Basis parameters for embedding [n_qubits]
        theta: Variational parameters [n_layers, n_qubits, 3]
        n_qubits: Number of qubits
        n_quantum_layers: Number of variational layers

    Returns:
        Expectation value (scalar)
    """
    return quantum_circuit_vectorized(
        x_classical, basis, theta, n_qubits, n_quantum_layers)
