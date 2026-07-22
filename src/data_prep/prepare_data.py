"""Data preparation pipeline for the Water Potability dataset (Challenge 2, Part 1).

Steps (in leakage-safe order):
  1. Load raw CSV.
  2. Stratified 80/20 train/test split.
  3. Median-by-class imputation (medians fit on train only, applied per-row by its own class).
  4. Standardization (scaler fit on train only).
  5. Class balancing on the training set only (undersampling or SMOTE).
  6. Selection of a balanced 16-64 sample subset from training data for the quantum experiment.

Two versions of the training set are persisted:
  - X_train_raw.csv / y_train_raw.csv: imputed + standardized, NOT balanced.
    Use this with a resampler folded INSIDE a cross-validation pipeline
    (e.g. imblearn.pipeline.Pipeline) so balancing never leaks information
    across CV folds. This is what src/classical/optuna_search.py uses.
  - X_train.csv / y_train.csv: the same data balanced once upfront
    (undersampling by default). Kept for the exact grid-search baseline
    required by the rubric (Part 2) and for quantum subset selection, both
    of which assume an already-balanced, fixed-size training set.

Run as a script to regenerate everything under data/processed/ and data/quantum_subset/:
    python -m src.data_prep.prepare_data
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/water_potability.csv"
PROCESSED_DIR = ROOT / "data/processed"
QUANTUM_DIR = ROOT / "data/quantum_subset"
TARGET = "Potability"
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
SEED = 42


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def impute_median_by_class(X_train, y_train, X_test, y_test):
    """Fit per-class medians on the training set; apply to train and test
    by each row's own class label. No test feature values are used to
    compute the medians."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    medians = {cls: X_train[y_train == cls].median() for cls in y_train.unique()}

    for cls, med in medians.items():
        train_mask = y_train == cls
        X_train.loc[train_mask] = X_train.loc[train_mask].fillna(med)
        test_mask = y_test == cls
        X_test.loc[test_mask] = X_test.loc[test_mask].fillna(med)

    return X_train, X_test, {str(k): v.to_dict() for k, v in medians.items()}


def standardize(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURES, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def balance(X_train, y_train, method: str = "undersample", seed: int = SEED):
    if method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=seed)
    elif method == "smote":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=seed)
    else:
        raise ValueError(f"Unknown balance method: {method}")
    X_bal, y_bal = sampler.fit_resample(X_train, y_train)
    return X_bal, y_bal


def select_quantum_subset(X_train, y_train, n: int = 32, seed: int = SEED):
    """Stratified random sample of size n from the (already balanced)
    training set, preserving class balance. Drawn only from training data."""
    if n % 2 != 0:
        raise ValueError("n must be even to keep exact class balance")
    per_class = n // 2
    rng = np.random.RandomState(seed)
    idx_parts = []
    for cls in sorted(y_train.unique()):
        cls_idx = y_train[y_train == cls].index.to_numpy()
        chosen = rng.choice(cls_idx, size=per_class, replace=False)
        idx_parts.append(chosen)
    subset_idx = np.concatenate(idx_parts)
    rng.shuffle(subset_idx)
    return X_train.loc[subset_idx], y_train.loc[subset_idx]


def main(balance_method: str = "undersample", quantum_sizes=(16, 32, 64)):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    QUANTUM_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    X_train, X_test, y_train, y_test = split(df)
    X_train, X_test, medians = impute_median_by_class(X_train, y_train, X_test, y_test)
    X_train, X_test, scaler = standardize(X_train, X_test)

    # Unbalanced version: imputed + scaled only. Balancing must happen inside
    # a per-fold CV pipeline for any model-selection procedure (see
    # src/classical/optuna_search.py) to avoid leaking resampled information
    # across folds.
    X_train.to_csv(PROCESSED_DIR / "X_train_raw.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train_raw.csv", index=False)

    X_train_bal, y_train_bal = balance(X_train, y_train, method=balance_method)

    X_train_bal.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train_bal.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    metadata = {
        "seed": SEED,
        "test_size": 0.2,
        "balance_method": balance_method,
        "train_size_before_balance": int(len(X_train)),
        "train_size_after_balance": int(len(X_train_bal)),
        "test_size_n": int(len(X_test)),
        "class_counts_train_before_balance": y_train.value_counts().to_dict(),
        "class_counts_train_after_balance": y_train_bal.value_counts().to_dict(),
        "class_counts_test": y_test.value_counts().to_dict(),
        "imputation_medians_by_class": medians,
        "note": (
            "X_train_raw/y_train_raw are imputed+scaled but NOT balanced; "
            "use them with in-pipeline resampling for model selection. "
            "X_train/y_train are pre-balanced (this run: "
            f"{balance_method}) for the fixed-grid rubric baseline."
        ),
    }
    with open(PROCESSED_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    for n in quantum_sizes:
        X_q, y_q = select_quantum_subset(X_train_bal, y_train_bal, n=n)
        X_q.assign(**{TARGET: y_q}).to_csv(QUANTUM_DIR / f"quantum_subset_{n}.csv", index=False)

    quantum_meta = {
        "sizes": list(quantum_sizes),
        "seed": SEED,
        "strategy": (
            "Stratified random sample without replacement, drawn only from the "
            "balanced training set (never from the test set), with an equal "
            "number of samples per class to preserve class balance."
        ),
        "source": "data/processed/X_train.csv (post-balancing) + y_train.csv",
    }
    with open(QUANTUM_DIR / "metadata.json", "w") as f:
        json.dump(quantum_meta, f, indent=2)

    print("Train (balanced):", X_train_bal.shape, "| Test:", X_test.shape)
    print("Train class counts (balanced):", y_train_bal.value_counts().to_dict())
    print("Test class counts:", y_test.value_counts().to_dict())
    print("Quantum subsets written for sizes:", quantum_sizes)


if __name__ == "__main__":
    main()
