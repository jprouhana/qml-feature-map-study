"""
Quantum kernel computation and kernel target alignment analysis.
"""

import numpy as np
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def compute_quantum_kernel_matrix(feature_map, X):
    """
    Compute the quantum kernel matrix for dataset X using the given feature map.
    K[i,j] = |<phi(x_i)|phi(x_j)>|^2
    """
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    K = kernel.evaluate(X)
    return K


def compute_kernel_target_alignment(K, y):
    """
    Compute kernel target alignment (KTA).

    KTA measures how well the kernel matrix aligns with the ideal
    label kernel yy^T. Higher KTA means the feature map naturally
    groups same-class points together.

    KTA = <K, yy^T>_F / (||K||_F * ||yy^T||_F)
    """
    # convert labels to +1/-1
    y_signed = 2 * y.astype(float) - 1
    y_outer = np.outer(y_signed, y_signed)

    # frobenius inner product
    numerator = np.sum(K * y_outer)
    denominator = np.linalg.norm(K, 'fro') * np.linalg.norm(y_outer, 'fro')

    if denominator < 1e-10:
        return 0.0

    return numerator / denominator


def compare_kernel_matrices(feature_maps_dict, X, y):
    """
    Compute kernel matrices and target alignment for all feature maps.
    Returns dict mapping feature map name -> {'kernel': K, 'kta': alignment}.
    """
    results = {}
    for name, fm in feature_maps_dict.items():
        K = compute_quantum_kernel_matrix(fm, X)
        kta = compute_kernel_target_alignment(K, y)
        results[name] = {'kernel': K, 'kta': kta}
    return results
