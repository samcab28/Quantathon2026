from __future__ import annotations

import numpy as np

from src.evaluation.metrics import classification_metrics, paired_bootstrap_difference


def test_safety_metric_counts_unsafe_water_declared_potable():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    metrics = classification_metrics(y_true, y_pred, y_pred)
    assert metrics["confusion_matrix"] == [[1, 2], [1, 1]]
    assert metrics["safety_interpretation"]["unsafe_declared_potable"] == 2
    assert np.isclose(metrics["false_positive_rate"], 2 / 3)


def test_paired_bootstrap_is_deterministic():
    y = np.array([0, 0, 1, 1, 0, 1])
    a = np.array([0, 1, 1, 0, 0, 1])
    b = np.array([0, 0, 1, 1, 0, 1])
    first = paired_bootstrap_difference(y, a, b, iterations=100, seed=7)
    second = paired_bootstrap_difference(y, a, b, iterations=100, seed=7)
    assert first == second
    assert first["mean_difference_b_minus_a"] > 0
