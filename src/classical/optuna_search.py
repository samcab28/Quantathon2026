"""Extended classical baseline: Optuna-tuned RBF-SVM (strongest classical model).

The rubric's required baseline (src/classical/baseline.py) uses a fixed
3x3 grid over pre-balanced training data. Two limitations of that setup,
addressed here:

  1. Balancing (undersampling) is applied once, upfront, before any
     cross-validation. For undersampling this discards ~1/3 of the majority
     class permanently; for a resampler like SMOTE it would leak information
     across CV folds (synthetic points generated from a minority sample
     that later lands in the validation fold). The fix: fold the resampler
     INSIDE an imblearn Pipeline together with the SVC, and cross-validate
     that whole pipeline so resampling is re-fit fresh on each training
     fold only.
  2. The grid (C in {0.1,1,10}, gamma in {scale,auto,0.01}) is coarse.
     Optuna's TPE sampler searches continuous log-scale ranges for C and
     gamma, and also treats the balancing strategy itself (undersample /
     SMOTE / none-with-class_weight) as a tunable hyperparameter.

General judging criteria explicitly reward benchmarking against "the
strongest available classical method" — this module is that stronger
reference, reported alongside (not instead of) the required grid baseline.

Run as a script to regenerate results/metrics/classical_optuna.json and
results/figures/fig_confusion_matrix_optuna.png:
    python -m src.classical.optuna_search
"""
from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC

from src.classical.baseline import plot_confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data/processed"
METRICS_DIR = ROOT / "results/metrics"
FIGURES_DIR = ROOT / "results/figures"
SEED = 42
N_TRIALS = 60
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)


def load_raw_train_and_test():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train_raw.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train_raw.csv").iloc[:, 0]
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    return X_train, y_train, X_test, y_test


def build_pipeline(balance_strategy: str, C: float, gamma):
    if balance_strategy == "undersample":
        sampler = RandomUnderSampler(random_state=SEED)
        svc = SVC(kernel="rbf", C=C, gamma=gamma)
    elif balance_strategy == "smote":
        sampler = SMOTE(random_state=SEED)
        svc = SVC(kernel="rbf", C=C, gamma=gamma)
    elif balance_strategy == "class_weight":
        sampler = "passthrough"
        svc = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")
    else:
        raise ValueError(f"Unknown balance_strategy: {balance_strategy}")
    return Pipeline([("sampler", sampler), ("svc", svc)])


def make_objective(X_train, y_train):
    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        gamma_kind = trial.suggest_categorical("gamma_kind", ["scale", "auto", "float"])
        gamma = trial.suggest_float("gamma_float", 1e-4, 10.0, log=True) if gamma_kind == "float" else gamma_kind
        balance_strategy = trial.suggest_categorical(
            "balance_strategy", ["undersample", "smote", "class_weight"]
        )
        pipe = build_pipeline(balance_strategy, C, gamma)
        # n_jobs=1: this dataset is small enough that a multiprocessing pool
        # (n_jobs=-1) costs more in spawn overhead than it saves, and on
        # Windows leaves noisy (harmless) joblib resource_tracker tracebacks.
        scores = cross_val_score(pipe, X_train, y_train, cv=CV, scoring="f1", n_jobs=1)
        return float(scores.mean())

    return objective


def run_study(X_train, y_train, n_trials: int = N_TRIALS) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(X_train, y_train), n_trials=n_trials, show_progress_bar=False)
    return study


def best_params_to_pipeline_kwargs(best_params: dict):
    gamma = best_params["gamma_float"] if best_params["gamma_kind"] == "float" else best_params["gamma_kind"]
    return best_params["balance_strategy"], best_params["C"], gamma


def evaluate_on_test(study: optuna.Study, X_train, y_train, X_test, y_test):
    balance_strategy, C, gamma = best_params_to_pipeline_kwargs(study.best_params)
    best_pipeline = build_pipeline(balance_strategy, C, gamma)
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    metrics = {
        "best_params": {"C": C, "gamma": gamma, "balance_strategy": balance_strategy},
        "cv_best_f1_mean": float(study.best_value),
        "cv_f1_std": float(
            pd.Series([t.value for t in study.trials if t.value is not None]).std()
        ),
        "n_trials": len(study.trials),
        "cv_scheme": "RepeatedStratifiedKFold(n_splits=5, n_repeats=3)",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return best_pipeline, metrics


def main(n_trials: int = N_TRIALS):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_raw_train_and_test()
    study = run_study(X_train, y_train, n_trials=n_trials)
    best_pipeline, metrics = evaluate_on_test(study, X_train, y_train, X_test, y_test)

    with open(METRICS_DIR / "classical_optuna.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        metrics["confusion_matrix"], FIGURES_DIR / "fig_confusion_matrix_optuna.png"
    )

    print(json.dumps(metrics, indent=2))
    return best_pipeline, metrics, study


if __name__ == "__main__":
    main()
