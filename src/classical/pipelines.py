"""Leakage-safe model pipelines and cross-validation helpers."""
from __future__ import annotations

from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from src.data_prep.prepare_data import FEATURES

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def feature_matrix(frame):
    return frame.loc[:, FEATURES]


def make_cv(n_splits: int, seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def build_pipeline(
    *,
    balance_strategy: str = "undersample",
    C: float = 1.0,
    gamma: str | float = "scale",
    scaler_strategy: str = "standard",
    add_indicator: bool = False,
    seed: int = 20260723,
) -> Pipeline:
    if scaler_strategy == "standard":
        scaler: Any = StandardScaler()
    elif scaler_strategy == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler strategy: {scaler_strategy}")

    if balance_strategy == "undersample":
        sampler: Any = RandomUnderSampler(random_state=seed)
        class_weight = None
    elif balance_strategy == "smote":
        sampler = SMOTE(random_state=seed, k_neighbors=3)
        class_weight = None
    elif balance_strategy == "class_weight":
        sampler = "passthrough"
        class_weight = "balanced"
    elif balance_strategy == "none":
        sampler = "passthrough"
        class_weight = None
    else:
        raise ValueError(f"Unknown balance strategy: {balance_strategy}")

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=add_indicator),
            ),
            ("scaler", scaler),
            ("sampler", sampler),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    class_weight=class_weight,
                    probability=False,
                ),
            ),
        ]
    )
