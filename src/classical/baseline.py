"""Classical RBF-SVM baseline (Challenge 2, Part 2).

Trains an SVM with an RBF kernel, tuned via 5-fold cross-validation over the
full grid C in {0.1, 1, 10} x gamma in {scale, auto, 0.01}, and reports
accuracy, precision, recall, F1 and the confusion matrix on the held-out test
set produced by src/data_prep/prepare_data.py.

Run as a script to regenerate results/metrics/classical_baseline.json and
results/figures/fig_confusion_matrix_classical.png:
    python -m src.classical.baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data/processed"
METRICS_DIR = ROOT / "results/metrics"
FIGURES_DIR = ROOT / "results/figures"

PARAM_GRID = {"C": [0.1, 1, 10], "gamma": ["scale", "auto", 0.01]}


def load_processed():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test):
    grid = GridSearchCV(
        SVC(kernel="rbf"), PARAM_GRID, cv=5, scoring="f1", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    metrics = {
        "best_params": grid.best_params_,
        "cv_best_f1": float(grid.best_score_),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return best_model, metrics


def plot_confusion_matrix(cm, out_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No potable", "Potable"],
        yticklabels=["No potable", "Potable"], ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("SVM-RBF clásica — Matriz de confusión")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed()
    _, metrics = train_and_evaluate(X_train, y_train, X_test, y_test)

    with open(METRICS_DIR / "classical_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        metrics["confusion_matrix"], FIGURES_DIR / "fig_confusion_matrix_classical.png"
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
