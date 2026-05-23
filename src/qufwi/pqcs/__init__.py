"""Standalone quantum circuit library for variational quantum circuits (VQC).

Pure JAX implementation — no fbpinns dependency.
"""

from qufwi.pqcs.gates import quantum_gates, apply_single_qubit_gate, apply_cnot
from qufwi.pqcs.circuit import expectation_pauli_z, quantum_circuit_vectorized, quantum_circuit
