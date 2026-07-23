"""Exact Pytket statevector kernels, validation, caching, and diagnostics."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.quantum.feature_maps import FeatureMapSpec, build_feature_map


def _cache_key(values: np.ndarray, spec: FeatureMapSpec) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    digest.update(json.dumps(asdict(spec), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def compute_statevectors(
    values: np.ndarray,
    spec: FeatureMapSpec,
    cache_dir: Path | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Statevector input must be a 2D matrix")
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"states_{spec.name}_{_cache_key(values, spec)}.npz"
        if cache_path.exists():
            return np.load(cache_path)["states"]

    states = np.vstack(
        [build_feature_map(row, spec).get_statevector() for row in values]
    )
    if cache_path is not None:
        np.savez_compressed(cache_path, states=states)
    return states


def kernel_from_states(
    left_states: np.ndarray, right_states: np.ndarray | None = None
) -> np.ndarray:
    if right_states is None:
        right_states = left_states
    overlaps = left_states.conj() @ right_states.T
    kernel = np.abs(overlaps) ** 2
    return np.asarray(kernel.real, dtype=float)


def compute_kernel(
    left: np.ndarray,
    spec: FeatureMapSpec,
    right: np.ndarray | None = None,
    cache_dir: Path | None = None,
) -> np.ndarray:
    left_states = compute_statevectors(left, spec, cache_dir)
    right_states = (
        left_states
        if right is None
        else compute_statevectors(right, spec, cache_dir)
    )
    return kernel_from_states(left_states, right_states)


def regularize_psd(
    kernel: np.ndarray,
    tolerance: float = 1e-10,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    symmetric = (np.asarray(kernel, dtype=float) + np.asarray(kernel).T) / 2
    eigenvalues = np.linalg.eigvalsh(symmetric)
    minimum = float(eigenvalues.min())
    shift = 0.0
    if minimum < -tolerance:
        shift = -minimum + epsilon
        symmetric = symmetric + shift * np.eye(len(symmetric))
    return symmetric, {
        "minimum_eigenvalue_before": minimum,
        "diagonal_shift": float(shift),
        "minimum_eigenvalue_after": float(np.linalg.eigvalsh(symmetric).min()),
    }


def kernel_target_alignment(kernel: np.ndarray, labels: np.ndarray) -> float:
    signed = 2 * np.asarray(labels, dtype=float) - 1
    target = np.outer(signed, signed)
    denominator = np.linalg.norm(kernel, "fro") * np.linalg.norm(target, "fro")
    return float(np.sum(kernel * target) / denominator) if denominator else 0.0


def kernel_diagnostics(
    kernel: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    kernel = np.asarray(kernel, dtype=float)
    labels = np.asarray(labels, dtype=int)
    eigenvalues = np.linalg.eigvalsh((kernel + kernel.T) / 2)
    positive = np.clip(eigenvalues, 0, None)
    effective_rank = (
        float(positive.sum() ** 2 / np.square(positive).sum())
        if np.square(positive).sum()
        else 0.0
    )
    same_mask = labels[:, None] == labels[None, :]
    different_mask = ~same_mask
    off_diagonal = ~np.eye(len(kernel), dtype=bool)
    same_values = kernel[same_mask & off_diagonal]
    different_values = kernel[different_mask]
    return {
        "symmetry_max_abs_error": float(np.max(np.abs(kernel - kernel.T))),
        "diagonal_max_abs_error": float(
            np.max(np.abs(np.diag(kernel) - 1.0))
        ),
        "value_min": float(kernel.min()),
        "value_max": float(kernel.max()),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "effective_rank": effective_rank,
        "kernel_target_alignment": kernel_target_alignment(kernel, labels),
        "within_class_similarity_mean": float(same_values.mean()),
        "between_class_similarity_mean": float(different_values.mean()),
        "eigenvalues": eigenvalues.tolist(),
    }


def assert_kernel_valid(
    kernel: np.ndarray,
    *,
    symmetry_tolerance: float = 1e-8,
    diagonal_tolerance: float = 1e-8,
    value_tolerance: float = 1e-8,
) -> None:
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Training kernel must be square")
    if np.max(np.abs(kernel - kernel.T)) > symmetry_tolerance:
        raise ValueError("Kernel is not symmetric")
    if np.max(np.abs(np.diag(kernel) - 1.0)) > diagonal_tolerance:
        raise ValueError("Kernel diagonal is not one")
    if kernel.min() < -value_tolerance or kernel.max() > 1 + value_tolerance:
        raise ValueError("Fidelity kernel values must be in [0, 1]")


def plot_kernel_heatmap(
    kernel: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    order = np.argsort(labels, kind="stable")
    ordered = kernel[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(ordered, vmin=0, vmax=1, cmap="viridis", ax=ax)
    boundary = int((np.asarray(labels)[order] == 0).sum())
    ax.axhline(boundary, color="white", linewidth=1)
    ax.axvline(boundary, color="white", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Muestras ordenadas por clase")
    ax.set_ylabel("Muestras ordenadas por clase")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_kernel_spectrum(
    diagnostics: dict[str, object],
    title: str,
    out_path: Path,
) -> None:
    eigenvalues = np.sort(
        np.asarray(diagnostics["eigenvalues"], dtype=float)
    )[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Índice")
    ax.set_ylabel("Valor propio")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
