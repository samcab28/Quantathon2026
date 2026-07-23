"""Consistent metrics, predictions, uncertainty, and paired comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

REQUIRED_METRICS = ("accuracy", "precision", "recall", "f1")


def get_scores(model, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None = None,
) -> dict[str, object]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    false_positive_rate = fp / (tn + fp) if (tn + fp) else float("nan")
    negative_predictive_value = tn / (tn + fn) if (tn + fn) else float("nan")
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, zero_division=0)
        ),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_arr, y_pred_arr)
        ),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "negative_predictive_value": float(negative_predictive_value),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true_arr)),
        "safety_interpretation": {
            "unsafe_declared_potable": fp,
            "unsafe_total": tn + fp,
            "unsafe_declared_potable_rate": float(false_positive_rate),
        },
    }
    if y_score is not None and len(np.unique(y_true_arr)) == 2:
        score_arr = np.asarray(y_score, dtype=float)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, score_arr))
        metrics["average_precision"] = float(
            average_precision_score(y_true_arr, score_arr)
        )
    return metrics


def bootstrap_intervals(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None,
    iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    score_arr = None if y_score is None else np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "balanced_accuracy": [],
        "specificity": [],
        "false_positive_rate": [],
        "mcc": [],
    }
    if score_arr is not None:
        collected["roc_auc"] = []
        collected["average_precision"] = []

    for _ in range(iterations):
        indices = rng.integers(0, len(y_true_arr), size=len(y_true_arr))
        if len(np.unique(y_true_arr[indices])) < 2:
            continue
        sample_metrics = classification_metrics(
            y_true_arr[indices],
            y_pred_arr[indices],
            None if score_arr is None else score_arr[indices],
        )
        for name in collected:
            collected[name].append(float(sample_metrics[name]))

    intervals: dict[str, dict[str, float]] = {}
    for name, values in collected.items():
        values_arr = np.asarray(values, dtype=float)
        intervals[name] = {
            "lower": float(np.quantile(values_arr, alpha / 2)),
            "upper": float(np.quantile(values_arr, 1 - alpha / 2)),
            "bootstrap_std": float(values_arr.std(ddof=1)),
        }
    return intervals


def prediction_frame(
    sample_ids: Iterable[int],
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float],
    model_name: str,
    run_id: str,
    **metadata: object,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": list(sample_ids),
            "y_true": list(y_true),
            "y_pred": list(y_pred),
            "y_score": list(y_score),
            "model": model_name,
            "run_id": run_id,
        }
    )
    for key, value in metadata.items():
        frame[key] = value
    return frame


def evaluate_model(
    model,
    X,
    y,
    sample_ids: Iterable[int],
    model_name: str,
    run_id: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    **metadata: object,
) -> tuple[dict[str, object], pd.DataFrame]:
    y_pred = np.asarray(model.predict(X), dtype=int)
    y_score = get_scores(model, X)
    metrics = classification_metrics(y, y_pred, y_score)
    metrics["confidence_intervals_95"] = bootstrap_intervals(
        y,
        y_pred,
        y_score,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    predictions = prediction_frame(
        sample_ids,
        y,
        y_pred,
        y_score,
        model_name,
        run_id,
        **metadata,
    )
    return metrics, predictions


def summarize_values(values: Iterable[float]) -> dict[str, object]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": arr.tolist(),
        "n": int(len(arr)),
    }


def paired_bootstrap_difference(
    y_true: Iterable[int],
    pred_a: Iterable[int],
    pred_b: Iterable[int],
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    iterations: int = 2000,
    seed: int = 20260723,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    rng = np.random.default_rng(seed)
    if metric is None:
        metric = lambda truth, prediction: f1_score(
            truth, prediction, zero_division=0
        )
    diffs = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), size=len(y))
        diffs.append(float(metric(y[idx], b[idx]) - metric(y[idx], a[idx])))
    arr = np.asarray(diffs)
    return {
        "mean_difference_b_minus_a": float(arr.mean()),
        "lower_95": float(np.quantile(arr, 0.025)),
        "upper_95": float(np.quantile(arr, 0.975)),
        "probability_b_better": float((arr > 0).mean()),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
