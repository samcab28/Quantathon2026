"""Métricas usadas por la SVM clásica y la QSVM."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true,
    y_pred,
    scores=None,
) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["average_precision"] = float(
            average_precision_score(y_true, scores)
        )
    return result
