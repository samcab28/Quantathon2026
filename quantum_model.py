"""Mapa de características, kernel cuántico y comparación con SVM-RBF."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault(
    "XDG_CONFIG_HOME",
    str(Path(__file__).resolve().parent / ".runtime_config"),
)

from pytket import Circuit
from pytket.circuit.display import render_circuit_as_html
from pytket.qasm import circuit_to_qasm_str
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from metrics import calculate_metrics


def build_feature_map(values, name="custom", repetitions=2):
    angles = np.clip(np.asarray(values, dtype=float), -3, 3) / 3 * np.pi
    circuit = Circuit(len(angles), name=f"{name}_feature_map")
    pairs = [(index, index + 1) for index in range(len(angles) - 1)]

    if name in {"zz", "pauli"}:
        for qubit in range(len(angles)):
            circuit.H(qubit)

    for repetition in range(repetitions):
        if name == "zz":
            for qubit, angle in enumerate(angles):
                circuit.Rz(float(angle / np.pi), qubit)
            for left, right in pairs:
                circuit.CX(left, right)
                circuit.Rz(
                    float(angles[left] * angles[right] / np.pi**2),
                    right,
                )
                circuit.CX(left, right)
        elif name == "pauli":
            for qubit, angle in enumerate(angles):
                circuit.Rz(float(angle / np.pi), qubit)
                circuit.Rx(float(angle / (2 * np.pi)), qubit)
            for left, right in pairs:
                circuit.CX(left, right)
                circuit.Rx(
                    float((angles[left] + angles[right]) / (2 * np.pi)),
                    right,
                )
                circuit.CX(left, right)
        elif name == "custom":
            for qubit, angle in enumerate(angles):
                circuit.Ry(float(angle / np.pi), qubit)
                circuit.Rz(float(angle**2 / np.pi**2), qubit)
            for left, right in pairs:
                circuit.CX(left, right)
        else:
            raise ValueError(f"Mapa no reconocido: {name}")

        if repetition + 1 < repetitions and name in {"zz", "pauli"}:
            for qubit in range(len(angles)):
                circuit.H(qubit)
    return circuit


def quantum_kernel(X_left, X_right=None, name="custom", repetitions=2):
    left_states = np.vstack(
        [
            build_feature_map(row, name, repetitions).get_statevector()
            for row in np.asarray(X_left)
        ]
    )
    if X_right is None:
        right_states = left_states
    else:
        right_states = np.vstack(
            [
                build_feature_map(row, name, repetitions).get_statevector()
                for row in np.asarray(X_right)
            ]
        )
    return np.abs(left_states.conj() @ right_states.T) ** 2


def balanced_subset(X, y, size, seed):
    if size % 2:
        raise ValueError("El tamaño del subset debe ser par")
    random = np.random.default_rng(seed)
    per_class = size // 2
    selected = np.concatenate(
        [
            random.choice(y.index[y == label], per_class, replace=False)
            for label in [0, 1]
        ]
    )
    random.shuffle(selected)
    return X.loc[selected], y.loc[selected]


def preprocess_subset(X_train, X_test):
    transform = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return transform.fit_transform(X_train), transform.transform(X_test)


def _quantum_search(kernel, y, cv_folds, seed):
    search = GridSearchCV(
        SVC(kernel="precomputed"),
        {"C": [0.1, 1, 10]},
        scoring="f1",
        cv=StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=seed,
        ),
    )
    search.fit(kernel, y)
    return search


def choose_feature_map(
    X_train,
    y_train,
    maps,
    subset_size,
    repeats,
    cv_folds,
    seed,
    circuit_repetitions,
):
    rows = []
    for name in maps:
        for repeat in range(repeats):
            X_subset, y_subset = balanced_subset(
                X_train,
                y_train,
                subset_size,
                seed + repeat,
            )
            X_scaled, _ = preprocess_subset(X_subset, X_subset)
            kernel = quantum_kernel(
                X_scaled,
                name=name,
                repetitions=circuit_repetitions,
            )
            search = _quantum_search(
                kernel,
                y_subset.to_numpy(),
                cv_folds,
                seed + repeat,
            )
            rows.append(
                {
                    "feature_map": name,
                    "repeat": repeat,
                    "cv_f1": float(search.best_score_),
                }
            )

    study = pd.DataFrame(rows)
    means = study.groupby("feature_map")["cv_f1"].mean()
    return means.idxmax(), study


def compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_map,
    subset_sizes,
    repeats,
    cv_folds,
    seed,
    circuit_repetitions,
):
    rows = []
    for size in subset_sizes:
        for repeat in range(repeats):
            current_seed = seed + 100 * size + repeat
            X_subset, y_subset = balanced_subset(
                X_train,
                y_train,
                size,
                current_seed,
            )
            X_small, X_eval = preprocess_subset(X_subset, X_test)

            train_kernel = quantum_kernel(
                X_small,
                name=feature_map,
                repetitions=circuit_repetitions,
            )
            test_kernel = quantum_kernel(
                X_eval,
                X_small,
                name=feature_map,
                repetitions=circuit_repetitions,
            )
            quantum_search = _quantum_search(
                train_kernel,
                y_subset.to_numpy(),
                cv_folds,
                current_seed,
            )
            quantum_predictions = quantum_search.predict(test_kernel)
            quantum_scores = quantum_search.decision_function(test_kernel)
            quantum_result = calculate_metrics(
                y_test,
                quantum_predictions,
                quantum_scores,
            )

            classical_search = GridSearchCV(
                SVC(kernel="rbf"),
                {
                    "C": [0.1, 1, 10],
                    "gamma": ["scale", "auto", 0.01],
                },
                scoring="f1",
                cv=StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=current_seed,
                ),
            )
            classical_search.fit(X_small, y_subset)
            classical_predictions = classical_search.predict(X_eval)
            classical_scores = classical_search.decision_function(X_eval)
            classical_result = calculate_metrics(
                y_test,
                classical_predictions,
                classical_scores,
            )

            for model_name, result in [
                ("QSVM", quantum_result),
                ("SVM-RBF", classical_result),
            ]:
                rows.append(
                    {
                        "model": model_name,
                        "subset_size": size,
                        "repeat": repeat,
                        **{
                            key: result[key]
                            for key in [
                                "accuracy",
                                "balanced_accuracy",
                                "precision",
                                "recall",
                                "f1",
                                "false_positive_rate",
                            ]
                        },
                    }
                )
    return pd.DataFrame(rows)


def kernel_diagnostics(kernel, labels):
    """Diagnósticos cuantitativos de la geometría del kernel (Parte 4).

    Un buen mapa deja las muestras de la misma clase con solapamiento alto y
    las de clases distintas con solapamiento bajo: `separation` > 0 y una
    alineación kernel-objetivo alta indican estructura útil para el SVM.
    """
    kernel = np.asarray(kernel, dtype=float)
    labels = np.asarray(labels)
    off_diagonal = ~np.eye(len(kernel), dtype=bool)
    same_class = (labels[:, None] == labels[None, :]) & off_diagonal
    other_class = (labels[:, None] != labels[None, :]) & off_diagonal

    target = np.where(labels[:, None] == labels[None, :], 1.0, -1.0)
    alignment = float(
        (kernel * target).sum()
        / (np.linalg.norm(kernel) * np.linalg.norm(target))
    )

    eigenvalues = np.linalg.eigvalsh((kernel + kernel.T) / 2)
    positive = eigenvalues[eigenvalues > 1e-12]
    if len(positive):
        weights = positive / positive.sum()
        effective_rank = float(np.exp(-(weights * np.log(weights)).sum()))
    else:
        effective_rank = 0.0

    within = float(kernel[same_class].mean()) if same_class.any() else 0.0
    between = float(kernel[other_class].mean()) if other_class.any() else 0.0
    return {
        "within_class_mean": within,
        "between_class_mean": between,
        "separation": within - between,
        "kernel_target_alignment": alignment,
        "off_diagonal_mean": float(kernel[off_diagonal].mean()),
        "min_eigenvalue": float(eigenvalues.min()),
        "effective_rank": effective_rank,
    }


def save_kernel_heatmap(kernel, labels, output_path, title):
    """Mapa de calor de la matriz de kernel (entregable de la Parte 3).

    Las muestras se ordenan por clase para que la estructura de bloques
    (clase 0 arriba-izquierda, clase 1 abajo-derecha) sea visible.
    """
    kernel = np.asarray(kernel, dtype=float)
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="stable")
    ordered = kernel[np.ix_(order, order)]
    boundary = int((labels[order] == 0).sum())

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(ordered, cmap="viridis", vmin=0, vmax=1)
    ax.axhline(boundary - 0.5, color="white", linewidth=1.2)
    ax.axvline(boundary - 0.5, color="white", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("muestra j (ordenadas por clase)")
    ax.set_ylabel("muestra i")
    fig.colorbar(image, ax=ax, label=r"$K_{ij}=|\langle\phi_i|\phi_j\rangle|^2$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_circuit_example(values, feature_map, repetitions, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit = build_feature_map(values, feature_map, repetitions)
    (output_dir / "feature_map.html").write_text(
        str(render_circuit_as_html(circuit, jupyter=False)),
        encoding="utf-8",
    )
    (output_dir / "feature_map.qasm").write_text(
        circuit_to_qasm_str(circuit),
        encoding="utf-8",
    )


def save_comparison_plot(results, output_path):
    summary = (
        results.groupby(["model", "subset_size"])["f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name, group in summary.groupby("model"):
        ax.errorbar(
            group["subset_size"],
            group["mean"],
            yerr=group["std"].fillna(0),
            marker="o",
            capsize=4,
            label=model_name,
        )
    ax.set_xlabel("Muestras de entrenamiento")
    ax.set_ylabel("F1")
    ax.set_title("SVM-RBF vs. QSVM")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
