"""Publication-ready evaluation figures shared by all model families."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def plot_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    cm = np.asarray(matrix, dtype=int)
    row_totals = cm.sum(axis=1, keepdims=True)
    percentages = np.divide(
        cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0
    )
    annotations = np.empty_like(cm, dtype=object)
    for row in range(2):
        for column in range(2):
            annotations[row, column] = (
                f"{cm[row, column]}\n{percentages[row, column]:.1%}"
            )

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    sns.heatmap(
        percentages,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Proporción dentro de la clase real"},
        xticklabels=["No potable", "Potable"],
        yticklabels=["No potable", "Potable"],
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_roc_pr(
    predictions: pd.DataFrame,
    title_prefix: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(
        predictions["y_true"], predictions["y_score"], ax=axes[0]
    )
    axes[0].set_title(f"{title_prefix}: ROC")
    PrecisionRecallDisplay.from_predictions(
        predictions["y_true"], predictions["y_score"], ax=axes[1]
    )
    axes[1].set_title(f"{title_prefix}: Precision-Recall")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(
    summary: pd.DataFrame,
    out_path: Path,
    title: str = "Comparación de modelos",
) -> None:
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    plot_data = summary.loc[:, ["model", *metrics]].melt(
        id_vars="model", var_name="metric", value_name="value"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_data, x="metric", y="value", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Valor")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Modelo", loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
