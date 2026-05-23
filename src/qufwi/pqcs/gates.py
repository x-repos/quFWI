"""Quantum gate operations for variational quantum circuits.

Pure JAX implementation — no fbpinns dependency.
"""

import jax.numpy as jnp


def quantum_gates(theta):
    """Define quantum gate operations"""
    # Rotation gates (using complex64 to save memory)
    def RX(angle):
        return jnp.array([
            [jnp.cos(angle/2), -1j * jnp.sin(angle/2)],
            [-1j * jnp.sin(angle/2), jnp.cos(angle/2)]
        ], dtype=jnp.complex64)

    def RY(angle):
        return jnp.array([
            [jnp.cos(angle/2), -jnp.sin(angle/2)],
            [jnp.sin(angle/2), jnp.cos(angle/2)]
        ], dtype=jnp.complex64)

    def RZ(angle):
        return jnp.array([
            [jnp.exp(-1j * angle/2), 0],
            [0, jnp.exp(1j * angle/2)]
        ], dtype=jnp.complex64)

    return RX, RY, RZ


def apply_single_qubit_gate(state, gate, target_qubit, n_qubits):
    """Apply single-qubit gate to target qubit"""
    shape = [2] * n_qubits
    state_tensor = state.reshape(shape)

    state_tensor = jnp.tensordot(gate, state_tensor, axes=[[1], [target_qubit]])

    axes_order = list(range(1, target_qubit + 1)) + [0] + list(range(target_qubit + 1, n_qubits))
    state_tensor = jnp.transpose(state_tensor, axes_order)

    return state_tensor.flatten()


def apply_cnot(state, control, target, n_qubits):
    """Apply CNOT gate"""
    state_tensor = state.reshape([2] * n_qubits)

    indices_c1_t0 = [slice(None)] * n_qubits
    indices_c1_t0[control] = 1
    indices_c1_t0[target] = 0
    indices_c1_t0 = tuple(indices_c1_t0)

    indices_c1_t1 = [slice(None)] * n_qubits
    indices_c1_t1[control] = 1
    indices_c1_t1[target] = 1
    indices_c1_t1 = tuple(indices_c1_t1)

    temp = state_tensor[indices_c1_t0].copy()
    state_tensor = state_tensor.at[indices_c1_t0].set(state_tensor[indices_c1_t1])
    state_tensor = state_tensor.at[indices_c1_t1].set(temp)

    return state_tensor.flatten()
