"""
Expressibility metrics for parameterized quantum circuits.

Measures how well a feature map explores the Hilbert space by comparing
the fidelity distribution of random circuit instances against the
Haar-random distribution.
"""

import numpy as np
from qiskit.quantum_info import Statevector


def compute_expressibility(circuit, n_samples=500):
    """
    Compute expressibility of a parameterized circuit.

    Samples pairs of random parameter vectors, computes fidelity between
    the resulting states, and compares the distribution to Haar random.

    Returns dict with fidelities array and KL divergence from Haar.
    """
    n_params = circuit.num_parameters
    n_qubits = circuit.num_qubits
    dim = 2 ** n_qubits

    rng = np.random.default_rng(42)
    fidelities = []

    for _ in range(n_samples):
        # sample two random parameter vectors
        params1 = rng.uniform(0, 2 * np.pi, n_params)
        params2 = rng.uniform(0, 2 * np.pi, n_params)

        # bind and compute statevectors
        qc1 = circuit.assign_parameters(params1)
        qc2 = circuit.assign_parameters(params2)

        sv1 = Statevector(qc1)
        sv2 = Statevector(qc2)

        # fidelity = |<psi1|psi2>|^2
        fid = abs(sv1.inner(sv2)) ** 2
        fidelities.append(fid)

    fidelities = np.array(fidelities)

    # compute KL divergence from Haar random distribution
    # for Haar random on dim-dimensional space: P(F) = (dim-1)(1-F)^(dim-2)
    kl_div = _kl_from_haar(fidelities, dim)

    return {
        'fidelities': fidelities,
        'kl_divergence': kl_div,
        'n_qubits': n_qubits,
    }


def _kl_from_haar(fidelities, dim, n_bins=50):
    """
    Compute KL divergence between sampled fidelity distribution and
    Haar random distribution using histogram binning.
    """
    # bin the sampled fidelities
    hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # haar random PDF: P(F) = (dim-1)(1-F)^(dim-2)
    haar_pdf = (dim - 1) * (1 - bin_centers) ** (dim - 2)

    # KL divergence: sum p(x) * log(p(x)/q(x))
    kl = 0.0
    for p, q in zip(hist, haar_pdf):
        if p > 1e-10 and q > 1e-10:
            kl += p * np.log(p / q) * bin_width

    return max(kl, 0.0)  # clamp to non-negative
