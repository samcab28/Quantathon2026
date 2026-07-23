"""Classical controls beyond the required RBF-SVM."""
from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.classical.pipelines import SCORING, feature_matrix, make_cv
from src.config import ClassicalConfig
from src.data_prep.prepare_data import SAMPLE_ID, TARGET
from src.evaluation.metrics import (
    classification_metrics,
    evaluate_model,
    prediction_frame,
    write_json,
)
from src.evaluation.plotting import plot_confusion_matrix, plot_roc_pr


def resolve_xgboost_device(requested: str) -> tuple[str, str]:
    if requested != "cuda":
        return "cpu", "CPU requested by configuration"
    probe_X = np.asarray([[0.0], [1.0], [0.2], [0.8]], dtype=np.float32)
    probe_y = np.asarray([0, 1, 0, 1], dtype=int)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            probe = xgb.XGBClassifier(
                n_estimators=1,
                max_depth=1,
                tree_method="hist",
                device="cuda",
                verbosity=0,
            )
            probe.fit(probe_X, probe_y)
        config = probe.get_booster().save_config()
        warning_text = " ".join(str(item.message) for item in caught)
        if '"device":"cuda' in config and "No visible GPU" not in warning_text:
            return "cuda", "CUDA device detected by XGBoost"
        return "cpu", f"CUDA unavailable; XGBoost fallback: {warning_text or config}"
    except Exception as exc:  # pragma: no cover - hardware-dependent
        return "cpu", f"CUDA probe failed: {type(exc).__name__}: {exc}"


def candidate_models(seed: int, xgboost_device: str) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "linear_svm_calibrated": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                ("scaler", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        LinearSVC(C=1.0, class_weight="balanced", random_state=seed),
                        cv=3,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                (
                    "model",
                    xgb.XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=4,
                        min_child_weight=4,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        device=xgboost_device,
                        random_state=seed,
                        n_jobs=1,
                        verbosity=0,
                    ),
                ),
            ]
        ),
    }


def training_cv_benchmark(
    train: pd.DataFrame,
    config: ClassicalConfig,
) -> tuple[pd.DataFrame, str, Pipeline, dict[str, str]]:
    X = feature_matrix(train)
    y = train[TARGET]
    cv = make_cv(config.cv_splits, config.cv_seed + 4000)
    rows = []
    xgboost_device, accelerator_reason = resolve_xgboost_device(
        config.accelerator
    )
    candidates = candidate_models(config.cv_seed, xgboost_device)
    for name, model in candidates.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            return_train_score=False,
        )
        row: dict[str, object] = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    benchmark = pd.DataFrame(rows).sort_values(
        f"{config.primary_metric}_mean", ascending=False
    )
    winner_name = str(benchmark.iloc[0]["model"])
    return (
        benchmark,
        winner_name,
        candidates[winner_name],
        {
            "requested": config.accelerator,
            "xgboost_device": xgboost_device,
            "reason": accelerator_reason,
        },
    )


def _positive_scores(model: Pipeline, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def choose_safety_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    max_unsafe_declared_potable_rate: float = 0.10,
) -> float:
    candidates = np.unique(scores)
    feasible: list[tuple[float, float]] = []
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        metrics = classification_metrics(y_true, predicted, scores)
        if (
            float(metrics["false_positive_rate"])
            <= max_unsafe_declared_potable_rate
        ):
            feasible.append((float(metrics["recall"]), float(threshold)))
    if not feasible:
        return float(np.nextafter(scores.max(), np.inf))
    feasible.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return feasible[0][1]


def run_classical_controls(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ClassicalConfig,
    run_dir: Path,
    run_id: str,
) -> dict[str, object]:
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    figures_dir = run_dir / "figures"
    models_dir = run_dir / "models"
    for directory in (metrics_dir, predictions_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    benchmark, winner_name, winner, accelerator = training_cv_benchmark(
        train, config
    )
    benchmark.to_csv(metrics_dir / "classical_controls_cv.csv", index=False)
    X_train = feature_matrix(train)
    y_train = train[TARGET]
    cv = make_cv(config.cv_splits, config.cv_seed + 5000)
    score_method = (
        "decision_function" if hasattr(winner, "decision_function") else "predict_proba"
    )
    oof_raw = cross_val_predict(
        winner,
        X_train,
        y_train,
        cv=cv,
        method=score_method,
        n_jobs=1,
    )
    if score_method == "predict_proba":
        oof_scores = np.asarray(oof_raw)[:, 1]
    else:
        oof_scores = np.asarray(oof_raw)
    threshold = choose_safety_threshold(y_train, oof_scores)

    winner.fit(X_train, y_train)
    standard_metrics, standard_predictions = evaluate_model(
        winner,
        feature_matrix(test),
        test[TARGET],
        test[SAMPLE_ID],
        winner_name,
        run_id,
        config.bootstrap_iterations,
        config.cv_seed + 6,
    )
    standard_metrics["selected_from_training_cv"] = winner_name
    standard_metrics["training_cv_f1_mean"] = float(
        benchmark.iloc[0]["f1_mean"]
    )
    standard_metrics["accelerator"] = accelerator
    write_json(metrics_dir / "classical_control_winner.json", standard_metrics)
    standard_predictions.to_csv(
        predictions_dir / "classical_control_winner_predictions.csv", index=False
    )
    joblib.dump(winner, models_dir / "classical_control_winner.joblib")
    plot_confusion_matrix(
        standard_metrics["confusion_matrix"],
        f"Mejor control clásico ({winner_name})",
        figures_dir / "classical_control_winner_confusion.png",
    )
    plot_roc_pr(
        standard_predictions,
        f"Mejor control clásico ({winner_name})",
        figures_dir / "classical_control_winner_roc_pr.png",
    )

    test_scores = _positive_scores(winner, feature_matrix(test))
    safety_pred = (test_scores >= threshold).astype(int)
    safety_metrics = classification_metrics(test[TARGET], safety_pred, test_scores)
    safety_metrics.update(
        {
            "model": winner_name,
            "threshold": float(threshold),
            "threshold_policy": (
                "Chosen from out-of-fold training scores to keep the rate of "
                "unsafe water declared potable at or below 10%."
            ),
        }
    )
    write_json(metrics_dir / "classical_control_safety_threshold.json", safety_metrics)
    safety_predictions = prediction_frame(
        test[SAMPLE_ID],
        test[TARGET],
        safety_pred,
        test_scores,
        f"{winner_name}_safety_threshold",
        run_id,
        threshold=threshold,
    )
    safety_predictions.to_csv(
        predictions_dir / "classical_control_safety_predictions.csv", index=False
    )
    return {
        "cv_benchmark": benchmark,
        "winner_name": winner_name,
        "winner_metrics": standard_metrics,
        "safety_metrics": safety_metrics,
        "winner_model": winner,
        "winner_predictions": standard_predictions,
        "accelerator": accelerator,
    }
