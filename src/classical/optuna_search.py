"""Nested-CV Optuna search for a stronger, leakage-safe RBF-SVM."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.classical.pipelines import build_pipeline, feature_matrix, make_cv
from src.config import ClassicalConfig, ROOT
from src.data_prep.prepare_data import SAMPLE_ID, TARGET
from src.evaluation.metrics import (
    classification_metrics,
    evaluate_model,
    get_scores,
    summarize_values,
    write_json,
)
from src.evaluation.plotting import plot_confusion_matrix, plot_roc_pr


def _trial_parameters(trial: optuna.Trial) -> dict[str, object]:
    gamma_kind = trial.suggest_categorical(
        "gamma_kind", ["scale", "auto", "float"]
    )
    gamma: str | float
    if gamma_kind == "float":
        gamma = trial.suggest_float("gamma_float", 1e-4, 10.0, log=True)
    else:
        gamma = gamma_kind
    return {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "gamma": gamma,
        "balance_strategy": trial.suggest_categorical(
            "balance_strategy", ["undersample", "smote", "class_weight"]
        ),
        "scaler_strategy": trial.suggest_categorical(
            "scaler_strategy", ["standard", "robust"]
        ),
        "add_indicator": trial.suggest_categorical(
            "add_indicator", [False, True]
        ),
    }


def make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    config: ClassicalConfig,
    seed: int,
):
    cv = make_cv(config.cv_splits, seed)

    def objective(trial: optuna.Trial) -> float:
        params = _trial_parameters(trial)
        pipeline = build_pipeline(**params, seed=seed)
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=config.primary_metric,
            n_jobs=1,
            error_score="raise",
        )
        trial.set_user_attr("fold_scores", [float(value) for value in scores])
        trial.set_user_attr("fold_std", float(scores.std(ddof=1)))
        return float(scores.mean())

    return objective


def run_study(
    X: pd.DataFrame,
    y: pd.Series,
    config: ClassicalConfig,
    seed: int,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(X, y, config, seed),
        n_trials=config.optuna_trials,
        n_jobs=1,
        show_progress_bar=False,
    )
    return study


def best_pipeline(study: optuna.Study, seed: int):
    params = dict(study.best_trial.params)
    gamma_kind = params.pop("gamma_kind")
    gamma = params.pop("gamma_float") if gamma_kind == "float" else gamma_kind
    return build_pipeline(gamma=gamma, seed=seed, **params), {
        **params,
        "gamma": gamma,
    }


def nested_cv_estimate(
    train: pd.DataFrame,
    config: ClassicalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = feature_matrix(train)
    y = train[TARGET]
    outer = StratifiedKFold(
        n_splits=config.nested_outer_splits,
        shuffle=True,
        random_state=config.cv_seed + 100,
    )
    fold_rows: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []
    for fold, (train_idx, validation_idx) in enumerate(outer.split(X, y)):
        seed = config.cv_seed + 1000 + fold
        study = run_study(
            X.iloc[train_idx],
            y.iloc[train_idx],
            config,
            seed,
        )
        model, params = best_pipeline(study, seed)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_validation = y.iloc[validation_idx]
        predictions = model.predict(X.iloc[validation_idx])
        scores = get_scores(model, X.iloc[validation_idx])
        metrics = classification_metrics(y_validation, predictions, scores)
        fold_rows.append(
            {
                "outer_fold": fold,
                "inner_best_f1": float(study.best_value),
                **{
                    name: metrics[name]
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
                "best_params": json.dumps(params, sort_keys=True),
            }
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    SAMPLE_ID: train.iloc[validation_idx][SAMPLE_ID].to_numpy(),
                    "y_true": y_validation.to_numpy(),
                    "y_pred": predictions,
                    "y_score": scores,
                    "outer_fold": fold,
                }
            )
        )
    return pd.DataFrame(fold_rows), pd.concat(oof_frames, ignore_index=True)


def run_optuna_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ClassicalConfig,
    run_dir: Path,
    run_id: str,
) -> dict[str, object]:
    model_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    figures_dir = run_dir / "figures"
    for directory in (model_dir, metrics_dir, predictions_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    nested_folds, oof_predictions = nested_cv_estimate(train, config)
    nested_folds.to_csv(metrics_dir / "classical_optuna_nested_cv.csv", index=False)
    oof_predictions.to_csv(
        predictions_dir / "classical_optuna_oof_predictions.csv", index=False
    )

    X_train = feature_matrix(train)
    y_train = train[TARGET]
    final_study = run_study(X_train, y_train, config, config.cv_seed + 2000)
    model, params = best_pipeline(final_study, config.cv_seed)

    # These are scores of the winning configuration across folds, not the
    # standard deviation across unrelated Optuna trials.
    selected_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=make_cv(config.cv_splits, config.cv_seed + 3000),
        scoring=config.primary_metric,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    metrics, predictions = evaluate_model(
        model,
        feature_matrix(test),
        test[TARGET],
        test[SAMPLE_ID],
        "svm_rbf_optuna_nested",
        run_id,
        config.bootstrap_iterations,
        config.cv_seed + 4,
    )
    metrics.update(
        {
            "best_params": params,
            "cv_selected_f1": summarize_values(selected_scores),
            "nested_cv_f1": summarize_values(nested_folds["f1"]),
            "nested_cv_balanced_accuracy": summarize_values(
                nested_folds["balanced_accuracy"]
            ),
            "n_trials_final_search": len(final_study.trials),
            "selection_protocol": (
                "Nested CV estimates generalization. Final hyperparameters are "
                "selected on all training rows. Test is evaluated only after "
                "the search is complete."
            ),
        }
    )
    write_json(metrics_dir / "classical_optuna_nested.json", metrics)
    predictions.to_csv(
        predictions_dir / "classical_optuna_predictions.csv", index=False
    )
    final_study.trials_dataframe().to_csv(
        metrics_dir / "classical_optuna_trials.csv", index=False
    )
    joblib.dump(model, model_dir / "classical_optuna_nested.joblib")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        "SVM-RBF Optuna con CV anidada: matriz de confusión",
        figures_dir / "classical_optuna_confusion.png",
    )
    plot_roc_pr(
        predictions,
        "SVM-RBF Optuna",
        figures_dir / "classical_optuna_roc_pr.png",
    )
    return {
        "metrics": metrics,
        "model": model,
        "predictions": predictions,
        "nested_folds": nested_folds,
    }


def main() -> None:
    from src.config import ExperimentConfig
    from src.data_prep.prepare_data import load_prepared

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train, test, _ = load_prepared()
    run_dir = ROOT / "results/runs/manual-optuna"
    result = run_optuna_baseline(
        train,
        test,
        ExperimentConfig().classical,
        run_dir,
        "manual-optuna",
    )
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
