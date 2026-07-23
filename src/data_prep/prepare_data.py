"""Leakage-safe data contracts for the Water Potability challenge.

This module deliberately persists *raw* train/test rows plus immutable sample
identifiers. Imputation, scaling, and balancing are model operations and are
therefore fitted inside cross-validation pipelines, never here.

The challenge document asks for "median by class" imputation. Applying that
rule to validation/test rows would require their true label and is target
leakage. We record class medians as an audit artifact, but the deployable
pipelines use a label-agnostic median fitted within each training fold.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DataConfig, ROOT

RAW_PATH = ROOT / "data/raw/water_potability.csv"
PROCESSED_DIR = ROOT / "data/processed"
QUANTUM_DIR = ROOT / "data/quantum_subset"
TARGET = "Potability"
SAMPLE_ID = "sample_id"
FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
EXPECTED_MISSING = {"ph": 491, "Sulfate": 781, "Trihalomethanes": 162}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_raw(frame)
    frame = frame.copy()
    frame.insert(0, SAMPLE_ID, np.arange(len(frame), dtype=int))
    return frame


def validate_raw(frame: pd.DataFrame) -> None:
    expected_columns = FEATURES + [TARGET]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns. Expected {expected_columns}, got {list(frame.columns)}"
        )
    if len(frame) != 3276:
        raise ValueError(f"Expected 3276 rows, got {len(frame)}")
    if not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0/1")
    if frame.duplicated().any():
        raise ValueError("Raw dataset contains duplicate rows")
    missing = frame[FEATURES].isna().sum().to_dict()
    for column, expected in EXPECTED_MISSING.items():
        if int(missing[column]) != expected:
            raise ValueError(
                f"Unexpected missing count for {column}: {missing[column]} != {expected}"
            )


def create_split_manifest(
    frame: pd.DataFrame, test_size: float, seed: int
) -> pd.DataFrame:
    train_ids, test_ids = train_test_split(
        frame[SAMPLE_ID],
        test_size=test_size,
        stratify=frame[TARGET],
        random_state=seed,
    )
    train_set = set(train_ids.astype(int))
    manifest = frame[[SAMPLE_ID, TARGET]].copy()
    manifest["split"] = np.where(
        manifest[SAMPLE_ID].isin(train_set), "train", "test"
    )
    manifest["split_seed"] = seed
    if set(manifest["split"]) != {"train", "test"}:
        raise AssertionError("Both train and test splits are required")
    if manifest[SAMPLE_ID].duplicated().any():
        raise AssertionError("sample_id must be unique")
    return manifest.sort_values(SAMPLE_ID).reset_index(drop=True)


def apply_manifest(
    frame: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = frame.merge(
        manifest[[SAMPLE_ID, "split"]], on=SAMPLE_ID, how="inner", validate="one_to_one"
    )
    train = merged.loc[merged["split"] == "train"].drop(columns="split")
    test = merged.loc[merged["split"] == "test"].drop(columns="split")
    if set(train[SAMPLE_ID]) & set(test[SAMPLE_ID]):
        raise AssertionError("Train/test sample overlap detected")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_subset_manifest(
    train: pd.DataFrame,
    sizes: Iterable[int],
    repeats: int,
    seed_start: int,
) -> pd.DataFrame:
    """Create balanced, nested training subsets for paired SVM/QSVM runs."""
    sizes = tuple(sorted(int(size) for size in sizes))
    if not sizes or any(size <= 0 or size % 2 for size in sizes):
        raise ValueError("Subset sizes must be positive even integers")
    min_class_count = int(train[TARGET].value_counts().min())
    if sizes[-1] // 2 > min_class_count:
        raise ValueError("Largest subset exceeds available samples per class")

    records: list[dict[str, int]] = []
    for repeat in range(repeats):
        seed = seed_start + repeat
        rng = np.random.default_rng(seed)
        ordered_by_class: dict[int, np.ndarray] = {}
        for label in (0, 1):
            ids = train.loc[train[TARGET] == label, SAMPLE_ID].to_numpy(copy=True)
            rng.shuffle(ids)
            ordered_by_class[label] = ids

        for size in sizes:
            per_class = size // 2
            selected = np.concatenate(
                [
                    ordered_by_class[0][:per_class],
                    ordered_by_class[1][:per_class],
                ]
            )
            rng.shuffle(selected)
            for order, sample_id in enumerate(selected):
                records.append(
                    {
                        SAMPLE_ID: int(sample_id),
                        "subset_size": size,
                        "repeat": repeat,
                        "subset_seed": seed,
                        "order": order,
                    }
                )
    manifest = pd.DataFrame.from_records(records)
    _validate_subset_manifest(train, manifest, sizes, repeats)
    return manifest


def _validate_subset_manifest(
    train: pd.DataFrame,
    manifest: pd.DataFrame,
    sizes: tuple[int, ...],
    repeats: int,
) -> None:
    train_ids = set(train[SAMPLE_ID])
    if not set(manifest[SAMPLE_ID]).issubset(train_ids):
        raise AssertionError("Quantum subset contains non-training samples")
    labels = train.set_index(SAMPLE_ID)[TARGET]
    for (repeat, size), group in manifest.groupby(["repeat", "subset_size"]):
        if len(group) != size or group[SAMPLE_ID].nunique() != size:
            raise AssertionError(f"Invalid subset size for repeat={repeat}, size={size}")
        counts = labels.loc[group[SAMPLE_ID]].value_counts().to_dict()
        if counts != {0: size // 2, 1: size // 2}:
            raise AssertionError(f"Subset is not balanced: {counts}")
    if manifest["repeat"].nunique() != repeats:
        raise AssertionError("Unexpected number of subset repeats")

    # Each larger subset must contain all IDs from the smaller subset for the
    # same repeat. This reduces composition noise in scaling comparisons.
    for repeat in range(repeats):
        previous: set[int] = set()
        for size in sizes:
            current = set(
                manifest.loc[
                    (manifest["repeat"] == repeat)
                    & (manifest["subset_size"] == size),
                    SAMPLE_ID,
                ]
            )
            if not previous.issubset(current):
                raise AssertionError("Subsets are not nested")
            previous = current


def class_median_audit(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    medians = train.groupby(TARGET)[FEATURES].median(numeric_only=True)
    return {
        str(int(label)): {
            feature: float(value) for feature, value in row.items()
        }
        for label, row in medians.iterrows()
    }


def dataset_profile(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "features": len(FEATURES),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in frame[TARGET].value_counts().sort_index().items()
        },
        "missing_counts": {
            feature: int(count)
            for feature, count in frame[FEATURES].isna().sum().items()
        },
        "duplicate_rows": int(frame.drop(columns=SAMPLE_ID).duplicated().sum()),
    }


def prepare_data(config: DataConfig) -> dict[str, object]:
    raw_path = Path(config.raw_path)
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    frame = load_raw(raw_path)
    manifest = create_split_manifest(frame, config.test_size, config.split_seed)
    train, test = apply_manifest(frame, manifest)
    subset_manifest = build_subset_manifest(
        train,
        config.subset_sizes,
        config.subset_repeats,
        config.subset_seed_start,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    QUANTUM_DIR.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(PROCESSED_DIR / "split_manifest.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train_raw.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test_raw.csv", index=False)
    subset_manifest.to_csv(QUANTUM_DIR / "subset_manifest.csv", index=False)

    audit = {
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha256_file(raw_path),
        "split_seed": config.split_seed,
        "test_size": config.test_size,
        "profiles": {
            "all": dataset_profile(frame),
            "train": dataset_profile(train),
            "test": dataset_profile(test),
        },
        "class_medians_train_only_audit": class_median_audit(train),
        "imputation_policy": (
            "Primary models use label-agnostic median imputation fitted inside "
            "each CV fold. Class medians are audit-only because applying them "
            "to unseen rows would require the unknown target."
        ),
        "subset_sizes": list(config.subset_sizes),
        "subset_repeats": config.subset_repeats,
        "subset_seed_start": config.subset_seed_start,
        "subset_policy": (
            "Balanced, nested samples drawn only from the training split. "
            "The same sample IDs are used for paired RBF-SVM/QSVM experiments."
        ),
    }
    with (PROCESSED_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    with (QUANTUM_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                key: audit[key]
                for key in (
                    "raw_sha256",
                    "split_seed",
                    "subset_sizes",
                    "subset_repeats",
                    "subset_seed_start",
                    "subset_policy",
                )
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return audit


def load_prepared() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED_DIR / "train_raw.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_raw.csv")
    subsets = pd.read_csv(QUANTUM_DIR / "subset_manifest.csv")
    return train, test, subsets


def main() -> None:
    audit = prepare_data(DataConfig())
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
