"""Shared plotting utilities for classical and quantum classifiers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_2d_decision_boundary(model, X, y, title: str, out_path: Path, grid_steps: int = 200):
    """Visualize a classifier trained on high-dimensional X by projecting onto
    its top-2 PCA components.

    Honesty note: the contour shown is the REAL model's decision_function,
    evaluated on a grid in PC1-PC2 space and mapped back to the original
    feature space via PCA.inverse_transform (i.e. the other components are
    held at their PCA reconstruction, not re-optimized). It is the actual
    trained classifier's boundary restricted to the plane spanned by the two
    directions of largest variance in X — not a separate 2D-only model. When
    the first two components explain a small fraction of total variance,
    this is a rough approximation of the true 9D boundary and should be
    reported as such.
    """
    import matplotlib.pyplot as plt

    columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
    X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_arr)
    explained = pca.explained_variance_ratio_

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps), np.linspace(y_min, y_max, grid_steps)
    )
    grid_2d = np.column_stack([xx.ravel(), yy.ravel()])
    grid_nd = pca.inverse_transform(grid_2d)
    if columns is not None:
        grid_nd = pd.DataFrame(grid_nd, columns=columns)

    if hasattr(model, "decision_function"):
        Z = model.decision_function(grid_nd)
    else:
        Z = model.predict_proba(grid_nd)[:, 1] - 0.5
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 21), cmap="RdBu", alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], colors="black", linewidths=1.5, linestyles="--")

    for cls, label, color in [(0, "No potable", "#d62728"), (1, "Potable", "#1f77b4")]:
        mask = y_arr == cls
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1], s=18, c=color, label=label,
            edgecolors="black", linewidths=0.3, alpha=0.85,
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var.)")
    ax.set_title(title, fontsize=10, wrap=True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return explained
