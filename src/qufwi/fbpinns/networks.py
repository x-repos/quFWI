"""
Defines standard neural network models

Each network class must inherit from the Network base class.
Each network class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax.numpy as jnp
from jax import random

from qufwi.pqcs.circuit import quantum_circuit


class Network:
    """Base neural network class to be inherited by different neural network classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn't appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""
        raise NotImplementedError

    @staticmethod
    def network_fn(params, x):
        """Forward model, for a SINGLE point with shape (xd,)"""
        raise NotImplementedError




class FCN(Network):
    "Fully connected network"

    @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [FCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w,b

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x

class FourierFCN(FCN):
    "Fully connected network with Fourier features"

    @staticmethod
    def init_params(key, layer_sizes, mu, sd, n_features):

        # get Fourier feature parameters
        key, subkey = random.split(key)
        omega = 2*jnp.pi*(mu+sd*random.normal(subkey, (n_features, layer_sizes[0])))
        layer_sizes = [2*n_features]+list(layer_sizes)[1:]

        # get FCN parameters
        keys = random.split(key, len(layer_sizes)-1)
        params = [FCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {"omega":omega}, trainable_params

    @staticmethod
    def network_fn(params, x):
        omega = params["static"]["network"]["subdomain"]["omega"]
        params = params["trainable"]["network"]["subdomain"]["layers"]
        x = jnp.dot(omega, x)
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)])# (2*n_features)
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x


class HybridQuantumFCN(Network):
    """Hybrid classical-quantum network

    Architecture: Classical layers -> Quantum embedding + PQC -> Output
    """

    @staticmethod
    def init_params(key, classical_layer_sizes, n_qubits, n_quantum_layers):
        """Initialize hybrid quantum network parameters

        Args:
            key: JAX random key
            classical_layer_sizes: List of classical layer sizes [input_dim, hidden1, hidden2, ...]
            n_qubits: Number of qubits for quantum circuit (output dim of last classical layer)
            n_quantum_layers: Number of variational quantum layers

        Returns:
            static_params: Dict with quantum circuit structure info
            trainable_params: Dict with classical weights and quantum parameters
        """
        # Split key for different parameter initialization
        key, classical_key, basis_key, theta_key, output_key = random.split(key, 5)

        # Initialize classical layers
        keys = random.split(classical_key, len(classical_layer_sizes)-1)
        classical_params = [HybridQuantumFCN._random_layer_params(k, m, n)
                           for k, m, n in zip(keys, classical_layer_sizes[:-1],
                                             classical_layer_sizes[1:])]

        # Initialize quantum embedding basis parameters
        # These encode the classical output into quantum states
        basis = random.normal(basis_key, (n_qubits,)) * 0.5

        # Initialize variational quantum circuit parameters
        # Shape: [n_layers, n_qubits, 3] for RX, RY, RZ rotations
        # Use same scale as classical network for FAIR comparison
        theta = random.normal(theta_key, (n_quantum_layers, n_qubits, 3)) * 0.1

        # Initialize output layer: maps quantum output (scalar) to final output (scalar)
        #
        # FAIR COMPARISON STRATEGY:
        # We want initial loss similar to classical network (~20k-30k)
        # Classical outputs: mean ~0, std ~0.004
        # Quantum circuit outputs: mean ~1.0, std ~0.001 (with theta=0.1)
        #
        # To get similar initial loss, quantum network final outputs should have
        # similar distribution to classical. Use output layer to scale down:
        # target_std = 0.004, quantum_std = 1.0
        # weight_scale = 0.004 / 1.0 = 0.004
        #
        # But we also want reasonable gradient flow, so use slightly larger: 0.01-0.02
        w_out_key, b_out_key = random.split(output_key)
        v = 0.02  # Small enough for initial stability, large enough for gradients
        # v = jnp.sqrt(1/32)
        output_weight = random.uniform(w_out_key, (), minval=-v, maxval=v)
        output_bias = random.uniform(b_out_key, (), minval=-v, maxval=v)

        static_params = {
            "n_qubits": n_qubits,
            "n_quantum_layers": n_quantum_layers
        }

        trainable_params = {
            "classical_layers": classical_params,
            "basis": basis,
            "theta": theta,
            "output_weight": output_weight,
            "output_bias": output_bias
        }

        return static_params, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        """Create random layer parameters for classical layers"""
        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w, b

    @staticmethod
    def network_fn(params, x):
        """Forward pass through hybrid network

        Args:
            params: Full parameter dict with structure from trainer
            x: Input point with shape (xd,)

        Returns:
            Output scalar value
        """
        # Extract parameters
        static_params = params["static"]["network"]["subdomain"]
        trainable_params = params["trainable"]["network"]["subdomain"]

        n_qubits = static_params["n_qubits"]
        n_quantum_layers = static_params["n_quantum_layers"]

        classical_layers = trainable_params["classical_layers"]
        basis = trainable_params["basis"]
        theta = trainable_params["theta"]
        output_weight = trainable_params["output_weight"]
        output_bias = trainable_params["output_bias"]

        # Forward through classical layers
        for w, b in classical_layers:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)

        # Forward through quantum circuit
        quantum_output = quantum_circuit(
            x, basis, theta, n_qubits, n_quantum_layers)

        # Apply output layer: w * quantum_output + b
        # This allows the network to learn appropriate scaling/shifting
        final_output = output_weight * quantum_output + output_bias

        # Return as array for consistency with other networks
        return jnp.array([final_output])


def norm(mu, sd, x):
    return (x-mu)/sd

def unnorm(mu, sd, x):
    return x*sd + mu
