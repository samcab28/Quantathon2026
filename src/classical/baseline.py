"""Required, leakage-safe RBF-SVM baseline and trivial controls."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.classical.pipelines import SCORING, build_pipeline, feature_matrix, make_cv
from src.config import ClassicalConfig, ROOT
from src.data_prep.prepare_data import SAMPLE_ID, TARGET
from src.evaluation.metrics import evaluate_model, write_json
from src.evaluation.plotting import plot_confusion_matrix, plot_roc_pr

PARAM_GRID = {
    "svc__C": [0.1, 1.0, 10.0],
    "svc__gamma": ["scale", "auto", 0.01],
}


def train_required_baseline(
    train: pd.DataFrame,
    config: ClassicalConfig,
) -> GridSearchCV:
    pipeline = build_pipeline(
        balance_strategy="undersample",
        seed=config.cv_seed,
    )
    search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=make_cv(config.cv_splits, config.cv_seed),
        scoring=SCORING,
        refit=config.primary_metric,
        n_jobs=1,
        return_train_score=True,
    )
    search.fit(feature_matrix(train), train[TARGET])
    return search


def train_dummy(train: pd.DataFrame) -> SklearnPipeline:
    model = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        ]
    )
    model.fit(feature_matrix(train), train[TARGET])
    return model


def cv_results_frame(search: GridSearchCV) -> pd.DataFrame:
    columns = [
        "param_svc__C",
        "param_svc__gamma",
        "mean_test_accuracy",
        "mean_test_balanced_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
        "std_test_f1",
        "mean_test_roc_auc",
        "mean_test_average_precision",
        "rank_test_f1",
    ]
    return pd.DataFrame(search.cv_results_).loc[:, columns].sort_values(
        "rank_test_f1"
    )


def run_required_baseline(
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

    search = train_required_baseline(train, config)
    cv_frame = cv_results_frame(search)
    cv_frame.to_csv(metrics_dir / "classical_required_cv.csv", index=False)

    X_test = feature_matrix(test)
    metrics, predictions = evaluate_model(
        search.best_estimator_,
        X_test,
        test[TARGET],
        test[SAMPLE_ID],
        "svm_rbf_required",
        run_id,
        config.bootstrap_iterations,
        config.cv_seed + 1,
    )
    metrics["best_params"] = {
        key.replace("svc__", ""): value for key, value in search.best_params_.items()
    }
    metrics["cv_best_f1_mean"] = float(search.best_score_)
    best_index = int(search.best_index_)
    metrics["cv_best_f1_std"] = float(
        search.cv_results_["std_test_f1"][best_index]
    )
    metrics["protocol"] = {
        "preprocessing": "median imputer -> standard scaler -> undersampling -> SVC",
        "cv": (
            f"StratifiedKFold({config.cv_splits}, shuffle=True, "
            f"random_state={config.cv_seed})"
        ),
        "grid": {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto", 0.01],
        },
        "test_opened_after_selection": True,
    }
    write_json(metrics_dir / "classical_required.json", metrics)
    predictions.to_csv(
        predictions_dir / "classical_required_predictions.csv", index=False
    )
    joblib.dump(search.best_estimator_, model_dir / "classical_required.joblib")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        "SVM-RBF requerida: matriz de confusión",
        figures_dir / "classical_required_confusion.png",
    )
    plot_roc_pr(
        predictions,
        "SVM-RBF requerida",
        figures_dir / "classical_required_roc_pr.png",
    )

    dummy = train_dummy(train)
    dummy_metrics, dummy_predictions = evaluate_model(
        dummy,
        X_test,
        test[TARGET],
        test[SAMPLE_ID],
        "dummy_most_frequent",
        run_id,
        config.bootstrap_iterations,
        config.cv_seed + 2,
    )
    write_json(metrics_dir / "dummy_most_frequent.json", dummy_metrics)
    dummy_predictions.to_csv(
        predictions_dir / "dummy_most_frequent_predictions.csv", index=False
    )
    joblib.dump(dummy, model_dir / "dummy_most_frequent.joblib")
    return {
        "required": metrics,
        "dummy": dummy_metrics,
        "required_model": search.best_estimator_,
        "required_predictions": predictions,
        "dummy_predictions": dummy_predictions,
    }


def main() -> None:
    from src.config import ExperimentConfig
    from src.data_prep.prepare_data import load_prepared

    train, test, _ = load_prepared()
    run_dir = ROOT / "results/runs/manual-classical"
    result = run_required_baseline(
        train,
        test,
        ExperimentConfig().classical,
        run_dir,
        "manual-classical",
    )
    print(json.dumps(result["required"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
