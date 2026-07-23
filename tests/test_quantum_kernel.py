from __future__ import annotations

import numpy as np

from src.quantum.feature_maps import FeatureMapSpec, build_feature_map
from src.quantum.kernels import (
    assert_kernel_valid,
    compute_kernel,
    kernel_diagnostics,
    regularize_psd,
)


def test_all_feature_maps_produce_normalized_statevectors():
    vector = np.array([0.2, -0.5, 1.1])
    for name in ("zz", "pauli", "custom"):
        circuit = build_feature_map(
            vector, FeatureMapSpec(name=name, repetitions=1)
        )
        state = circuit.get_statevector()
        assert circuit.n_qubits == 3
        assert np.isclose(np.vdot(state, state).real, 1.0)


def test_fidelity_kernel_invariants_and_diagnostics():
    X = np.array(
        [
            [0.1, -0.2, 0.3],
            [0.2, 0.4, -0.1],
            [-0.3, 0.1, 0.5],
            [0.7, -0.8, 0.2],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    kernel = compute_kernel(X, FeatureMapSpec(name="zz", repetitions=1))
    assert_kernel_valid(kernel)
    regularized, report = regularize_psd(kernel)
    diagnostics = kernel_diagnostics(regularized, labels)
    assert report["minimum_eigenvalue_after"] >= -1e-9
    assert diagnostics["symmetry_max_abs_error"] < 1e-9
    assert diagnostics["diagonal_max_abs_error"] < 1e-9
    assert 0 <= diagnostics["effective_rank"] <= len(X)
