"""Training-only feature-map study and paired RBF-SVM/QSVM comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.classical.pipelines import build_pipeline, feature_matrix
from src.config import ClassicalConfig, QuantumConfig
from src.data_prep.prepare_data import FEATURES, SAMPLE_ID, TARGET
from src.evaluation.metrics import (
    bootstrap_intervals,
    classification_metrics,
    paired_bootstrap_difference,
    prediction_frame,
    write_json,
)
from src.quantum.feature_maps import (
    FeatureMapSpec,
    build_feature_map,
    circuit_resources,
    save_representative_circuit,
)
from src.quantum.kernels import (
    assert_kernel_valid,
    compute_kernel,
    kernel_diagnostics,
    plot_kernel_heatmap,
    plot_kernel_spectrum,
    regularize_psd,
)


def subset_frame(
    train: pd.DataFrame,
    subset_manifest: pd.DataFrame,
    size: int,
    repeat: int,
) -> pd.DataFrame:
    selected = subset_manifest.loc[
        (subset_manifest["subset_size"] == size)
        & (subset_manifest["repeat"] == repeat)
    ].sort_values("order")
    subset = selected.merge(
        train,
        on=SAMPLE_ID,
        how="left",
        validate="one_to_one",
    )
    if len(subset) != size:
        raise AssertionError("Subset manifest did not resolve to the expected size")
    if subset[TARGET].value_counts().to_dict() != {0: size // 2, 1: size // 2}:
        raise AssertionError("Resolved subset is not balanced")
    return subset


def fit_quantum_preprocessor(
    train_features: pd.DataFrame,
) -> tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    scaler = StandardScaler()
    imputed = imputer.fit_transform(train_features)
    scaled = scaler.fit_transform(imputed)
    return imputer, scaler, scaled


def transform_quantum_preprocessor(
    features: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    return scaler.transform(imputer.transform(features))


def qsvm_training_cv(
    subset: pd.DataFrame,
    spec: FeatureMapSpec,
    config: QuantumConfig,
    seed: int,
    cache_dir: Path,
) -> tuple[float, pd.DataFrame]:
    X = subset[FEATURES].reset_index(drop=True)
    y = subset[TARGET].reset_index(drop=True)
    n_splits = min(config.cv_splits, int(y.value_counts().min()))
    if n_splits < 2:
        raise ValueError("At least two samples per class are required for QSVM CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, float | int]] = []
    for fold, (train_idx, validation_idx) in enumerate(cv.split(X, y)):
        imputer, scaler, X_train = fit_quantum_preprocessor(X.iloc[train_idx])
        X_validation = transform_quantum_preprocessor(
            X.iloc[validation_idx], imputer, scaler
        )
        fold_cache = cache_dir / f"fold_{fold}"
        K_train = compute_kernel(X_train, spec, cache_dir=fold_cache)
        assert_kernel_valid(K_train)
        K_train, psd = regularize_psd(
            K_train, config.psd_tolerance, config.psd_epsilon
        )
        K_validation = compute_kernel(
            X_validation,
            spec,
            right=X_train,
            cache_dir=fold_cache,
        )
        for C in config.c_grid:
            model = SVC(kernel="precomputed", C=float(C))
            model.fit(K_train, y.iloc[train_idx])
            prediction = model.predict(K_validation)
            rows.append(
                {
                    "fold": fold,
                    "C": float(C),
                    "f1": float(f1_score(y.iloc[validation_idx], prediction)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(
                            y.iloc[validation_idx], prediction
                        )
                    ),
                    "psd_shift": float(psd["diagonal_shift"]),
                }
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("C", as_index=False)
        .agg(
            cv_f1_mean=("f1", "mean"),
            cv_f1_std=("f1", "std"),
            cv_balanced_accuracy_mean=("balanced_accuracy", "mean"),
            cv_balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .sort_values(
            ["cv_f1_mean", "cv_balanced_accuracy_mean"],
            ascending=False,
        )
    )
    best_c = float(summary.iloc[0]["C"])
    return best_c, summary


def full_subset_kernel_diagnostics(
    subset: pd.DataFrame,
    spec: FeatureMapSpec,
    config: QuantumConfig,
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    _, _, scaled = fit_quantum_preprocessor(subset[FEATURES])
    kernel = compute_kernel(scaled, spec, cache_dir=cache_dir)
    assert_kernel_valid(kernel)
    diagnostics = kernel_diagnostics(kernel, subset[TARGET].to_numpy())
    regularized, psd = regularize_psd(
        kernel, config.psd_tolerance, config.psd_epsilon
    )
    diagnostics["psd_regularization"] = psd
    return regularized, diagnostics, scaled


def run_feature_map_study(
    train: pd.DataFrame,
    subset_manifest: pd.DataFrame,
    config: QuantumConfig,
    run_dir: Path,
) -> tuple[pd.DataFrame, str]:
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    circuits_dir = run_dir / "circuits"
    cache_dir = run_dir / "cache" / "quantum"
    for directory in (metrics_dir, figures_dir, circuits_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sizes = sorted(subset_manifest["subset_size"].unique())
    repeats = sorted(subset_manifest["repeat"].unique())
    rows: list[dict[str, object]] = []
    largest_size = int(max(sizes))

    for map_name in config.feature_maps:
        spec = FeatureMapSpec(
            name=map_name,
            repetitions=config.repetitions,
            entanglement=config.entanglement,
            feature_clip=config.feature_clip,
            angle_scale=config.angle_scale,
        )
        for repeat in repeats:
            for size in sizes:
                subset = subset_frame(train, subset_manifest, int(size), int(repeat))
                seed = int(
                    subset_manifest.loc[
                        (subset_manifest["subset_size"] == size)
                        & (subset_manifest["repeat"] == repeat),
                        "subset_seed",
                    ].iloc[0]
                )
                experiment_cache = (
                    cache_dir / map_name / f"size_{size}" / f"repeat_{repeat}"
                )
                best_c, cv_summary = qsvm_training_cv(
                    subset,
                    spec,
                    config,
                    seed,
                    experiment_cache / "cv",
                )
                selected = cv_summary.loc[cv_summary["C"] == best_c].iloc[0]
                kernel, diagnostics, scaled = full_subset_kernel_diagnostics(
                    subset,
                    spec,
                    config,
                    experiment_cache / "full",
                )
                representative = build_feature_map(scaled[0], spec)
                resources = circuit_resources(representative)
                rows.append(
                    {
                        "feature_map": map_name,
                        "subset_size": int(size),
                        "repeat": int(repeat),
                        "subset_seed": seed,
                        "best_C": best_c,
                        "cv_f1_mean": float(selected["cv_f1_mean"]),
                        "cv_f1_std": float(selected["cv_f1_std"]),
                        "cv_balanced_accuracy_mean": float(
                            selected["cv_balanced_accuracy_mean"]
                        ),
                        "cv_balanced_accuracy_std": float(
                            selected["cv_balanced_accuracy_std"]
                        ),
                        "kernel_target_alignment": float(
                            diagnostics["kernel_target_alignment"]
                        ),
                        "within_class_similarity_mean": float(
                            diagnostics["within_class_similarity_mean"]
                        ),
                        "between_class_similarity_mean": float(
                            diagnostics["between_class_similarity_mean"]
                        ),
                        "effective_rank": float(diagnostics["effective_rank"]),
                        "minimum_eigenvalue": float(
                            diagnostics["minimum_eigenvalue"]
                        ),
                        "psd_shift": float(
                            diagnostics["psd_regularization"]["diagonal_shift"]
                        ),
                        "n_qubits": int(resources["n_qubits"]),
                        "circuit_depth": int(resources["depth"]),
                        "n_gates": int(resources["n_gates"]),
                        "two_qubit_gates": int(resources["two_qubit_gates"]),
                    }
                )
                cv_summary.assign(
                    feature_map=map_name,
                    subset_size=size,
                    repeat=repeat,
                    subset_seed=seed,
                ).to_csv(
                    metrics_dir
                    / f"qsvm_cv_{map_name}_n{size}_r{repeat}.csv",
                    index=False,
                )
                if int(size) == largest_size and int(repeat) == 0:
                    plot_kernel_heatmap(
                        kernel,
                        subset[TARGET].to_numpy(),
                        f"Kernel {map_name} (n={size})",
                        figures_dir / f"kernel_heatmap_{map_name}.png",
                    )
                    plot_kernel_spectrum(
                        diagnostics,
                        f"Espectro del kernel {map_name} (n={size})",
                        figures_dir / f"kernel_spectrum_{map_name}.png",
                    )
                    save_representative_circuit(
                        representative,
                        spec,
                        circuits_dir,
                    )
                    write_json(
                        metrics_dir / f"kernel_diagnostics_{map_name}.json",
                        diagnostics,
                    )

    study = pd.DataFrame(rows)
    study.to_csv(metrics_dir / "feature_map_study.csv", index=False)
    map_ranking = (
        study.groupby("feature_map", as_index=False)
        .agg(
            cv_f1_mean=("cv_f1_mean", "mean"),
            cv_f1_std_across_subsets=("cv_f1_mean", "std"),
            balanced_accuracy_mean=("cv_balanced_accuracy_mean", "mean"),
            kernel_target_alignment_mean=("kernel_target_alignment", "mean"),
            effective_rank_mean=("effective_rank", "mean"),
        )
        .sort_values(
            ["cv_f1_mean", "balanced_accuracy_mean"],
            ascending=False,
        )
        .fillna(0.0)
    )
    map_ranking.to_csv(metrics_dir / "feature_map_ranking.csv", index=False)
    primary_map = str(map_ranking.iloc[0]["feature_map"])
    write_json(
        metrics_dir / "feature_map_selection.json",
        {
            "primary_map": primary_map,
            "selection_rule": (
                "Highest mean training-only CV F1 across all registered subset "
                "sizes and repeats; balanced accuracy is the tie-breaker. Test "
                "labels are not used for map selection."
            ),
            "ranking": map_ranking.to_dict(orient="records"),
        },
    )
    return study, primary_map


def _paired_classical_model(
    subset: pd.DataFrame,
    seed: int,
    cv_splits: int,
) -> GridSearchCV:
    folds = min(cv_splits, int(subset[TARGET].value_counts().min()))
    search = GridSearchCV(
        build_pipeline(balance_strategy="class_weight", seed=seed),
        {
            "svc__C": [0.1, 1.0, 10.0],
            "svc__gamma": ["scale", "auto", 0.01],
        },
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        scoring="f1",
        refit=True,
        n_jobs=1,
    )
    search.fit(feature_matrix(subset), subset[TARGET])
    return search


def run_paired_comparison(
    train: pd.DataFrame,
    test: pd.DataFrame,
    subset_manifest: pd.DataFrame,
    primary_map: str,
    quantum_config: QuantumConfig,
    classical_config: ClassicalConfig,
    run_dir: Path,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    figures_dir = run_dir / "figures"
    cache_dir = run_dir / "cache" / "paired"
    for directory in (metrics_dir, predictions_dir, figures_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    spec = FeatureMapSpec(
        name=primary_map,
        repetitions=quantum_config.repetitions,
        entanglement=quantum_config.entanglement,
        feature_clip=quantum_config.feature_clip,
        angle_scale=quantum_config.angle_scale,
    )
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    paired_differences: list[dict[str, object]] = []
    test_X = feature_matrix(test)

    for (repeat, size), registry in subset_manifest.groupby(
        ["repeat", "subset_size"], sort=True
    ):
        size = int(size)
        repeat = int(repeat)
        seed = int(registry["subset_seed"].iloc[0])
        subset = subset_frame(train, subset_manifest, size, repeat)

        classical = _paired_classical_model(
            subset, seed, classical_config.cv_splits
        )
        classical_prediction = classical.predict(test_X)
        classical_score = classical.decision_function(test_X)
        classical_metrics = classification_metrics(
            test[TARGET], classical_prediction, classical_score
        )
        rows.append(
            {
                "model": "rbf_svm_paired",
                "feature_map": "",
                "subset_size": size,
                "repeat": repeat,
                "subset_seed": seed,
                "best_C": float(classical.best_params_["svc__C"]),
                "best_gamma": str(classical.best_params_["svc__gamma"]),
                **{
                    name: classical_metrics[name]
                    for name in (
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "specificity",
                        "false_positive_rate",
                        "mcc",
                        "roc_auc",
                        "average_precision",
                    )
                },
            }
        )
        prediction_frames.append(
            prediction_frame(
                test[SAMPLE_ID],
                test[TARGET],
                classical_prediction,
                classical_score,
                "rbf_svm_paired",
                run_id,
                subset_size=size,
                repeat=repeat,
                subset_seed=seed,
                feature_map="",
            )
        )

        best_c, cv_summary = qsvm_training_cv(
            subset,
            spec,
            quantum_config,
            seed,
            cache_dir
            / primary_map
            / f"size_{size}"
            / f"repeat_{repeat}"
            / "cv",
        )
        imputer, scaler, subset_scaled = fit_quantum_preprocessor(
            subset[FEATURES]
        )
        test_scaled = transform_quantum_preprocessor(
            test[FEATURES], imputer, scaler
        )
        pair_cache = (
            cache_dir / primary_map / f"size_{size}" / f"repeat_{repeat}" / "final"
        )
        K_train = compute_kernel(subset_scaled, spec, cache_dir=pair_cache)
        assert_kernel_valid(K_train)
        K_train, psd = regularize_psd(
            K_train,
            quantum_config.psd_tolerance,
            quantum_config.psd_epsilon,
        )
        K_test = compute_kernel(
            test_scaled,
            spec,
            right=subset_scaled,
            cache_dir=pair_cache,
        )
        qsvm = SVC(kernel="precomputed", C=best_c)
        qsvm.fit(K_train, subset[TARGET])
        quantum_prediction = qsvm.predict(K_test)
        quantum_score = qsvm.decision_function(K_test)
        quantum_metrics = classification_metrics(
            test[TARGET], quantum_prediction, quantum_score
        )
        rows.append(
            {
                "model": "qsvm_paired",
                "feature_map": primary_map,
                "subset_size": size,
                "repeat": repeat,
                "subset_seed": seed,
                "best_C": float(best_c),
                "best_gamma": "",
                **{
                    name: quantum_metrics[name]
                    for name in (
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "specificity",
                        "false_positive_rate",
                        "mcc",
                        "roc_auc",
                        "average_precision",
                    )
                },
                "psd_shift": float(psd["diagonal_shift"]),
                "cv_f1_mean": float(
                    cv_summary.loc[cv_summary["C"] == best_c, "cv_f1_mean"].iloc[0]
                ),
            }
        )
        prediction_frames.append(
            prediction_frame(
                test[SAMPLE_ID],
                test[TARGET],
                quantum_prediction,
                quantum_score,
                "qsvm_paired",
                run_id,
                subset_size=size,
                repeat=repeat,
                subset_seed=seed,
                feature_map=primary_map,
            )
        )
        paired_differences.append(
            {
                "subset_size": size,
                "repeat": repeat,
                **paired_bootstrap_difference(
                    test[TARGET],
                    classical_prediction,
                    quantum_prediction,
                    iterations=classical_config.bootstrap_iterations,
                    seed=classical_config.cv_seed + size + repeat,
                ),
            }
        )

    results = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    results.to_csv(metrics_dir / "paired_classical_qsvm.csv", index=False)
    predictions.to_csv(
        predictions_dir / "paired_classical_qsvm_predictions.csv", index=False
    )
    pd.DataFrame(paired_differences).to_csv(
        metrics_dir / "paired_bootstrap_f1_difference.csv", index=False
    )

    summary = (
        results.groupby(["model", "feature_map", "subset_size"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            false_positive_rate_mean=("false_positive_rate", "mean"),
            false_positive_rate_std=("false_positive_rate", "std"),
        )
        .fillna(0.0)
    )
    summary.to_csv(metrics_dir / "paired_scaling_summary.csv", index=False)
    plot_scaling(summary, figures_dir / "paired_scaling_f1.png")
    return results, summary


def plot_scaling(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, group in summary.groupby("model"):
        group = group.sort_values("subset_size")
        ax.errorbar(
            group["subset_size"],
            group["f1_mean"],
            yerr=group["f1_std"],
            marker="o",
            capsize=4,
            label=model,
        )
    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel("F1 en el test bloqueado")
    ax.set_ylim(0, 1)
    ax.set_title("Escalado pareado: RBF-SVM vs QSVM")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
