"""
Quantum feature map implementations for classification experiments.

Provides four different data encoding strategies to compare their
impact on quantum classifier performance.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap


def build_angle_encoding(n_qubits=2):
    """
    Simple angle encoding — one Ry rotation per qubit.
    No entanglement, creates product states only.
    This is the simplest feature map but least expressive.
    """
    params = ParameterVector('x', n_qubits)
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.ry(params[i], i)

    return qc


def build_zz_feature_map(n_qubits=2, reps=2):
    """
    ZZFeatureMap from Qiskit — entangling feature map with pairwise
    ZZ interactions between qubits. Creates correlations based on
    products of input features.
    """
    fm = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement='linear'
    )
    return fm


def build_iqp_feature_map(n_qubits=2, reps=2):
    """
    IQP-style feature map — Hadamard layers interspersed with
    diagonal ZZ entangling gates.

    The IQP (Instantaneous Quantum Polynomial) structure is interesting
    because sampling from these circuits is believed to be classically
    hard, which suggests the feature space is genuinely quantum.
    """
    params = ParameterVector('x', n_qubits)
    qc = QuantumCircuit(n_qubits)

    for _ in range(reps):
        # hadamard layer
        for i in range(n_qubits):
            qc.h(i)

        # diagonal ZZ interactions using input features
        for i in range(n_qubits):
            qc.rz(params[i], i)

        # pairwise ZZ entangling
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(params[i] * params[i + 1], i + 1)
            qc.cx(i, i + 1)

    return qc


def build_amplitude_encoding(n_qubits=2):
    """
    Amplitude encoding — encode data into the amplitudes of the quantum state.
    More data-efficient (2^n amplitudes for n qubits) but requires a
    state preparation circuit.

    For 2 qubits, we encode 4 features. For classification with 2 features,
    we pad with zeros.

    Note: this uses a simplified version that applies rotations to create
    an approximate amplitude encoding.
    """
    params = ParameterVector('x', n_qubits)
    qc = QuantumCircuit(n_qubits)

    # initialize in superposition
    for i in range(n_qubits):
        qc.h(i)

    # encode using controlled rotations for entanglement
    for i in range(n_qubits):
        qc.ry(params[i], i)

    # add entanglement
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.ry(params[i] * 0.5, i + 1)
        qc.cx(i, i + 1)

    return qc


def get_all_feature_maps(n_qubits=2):
    """Return a dict of all feature maps for comparison."""
    return {
        'Angle Encoding': build_angle_encoding(n_qubits),
        'ZZFeatureMap': build_zz_feature_map(n_qubits),
        'IQP-style': build_iqp_feature_map(n_qubits),
        'Amplitude': build_amplitude_encoding(n_qubits),
    }
# IQP encoding uses commuting diagonal gates
