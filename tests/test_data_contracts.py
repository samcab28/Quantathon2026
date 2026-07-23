from __future__ import annotations

import pandas as pd

from src.config import DataConfig, ROOT
from src.data_prep.prepare_data import (
    SAMPLE_ID,
    TARGET,
    apply_manifest,
    build_subset_manifest,
    create_split_manifest,
    load_raw,
)


def test_split_is_disjoint_stratified_and_reproducible():
    frame = load_raw(ROOT / "data/raw/water_potability.csv")
    first = create_split_manifest(frame, test_size=0.2, seed=20260723)
    second = create_split_manifest(frame, test_size=0.2, seed=20260723)
    pd.testing.assert_frame_equal(first, second)
    train, test = apply_manifest(frame, first)
    assert set(train[SAMPLE_ID]).isdisjoint(set(test[SAMPLE_ID]))
    assert len(train) + len(test) == len(frame)
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.01


def test_quantum_subsets_are_train_only_balanced_and_nested():
    frame = load_raw(ROOT / "data/raw/water_potability.csv")
    split = create_split_manifest(frame, test_size=0.2, seed=20260723)
    train, test = apply_manifest(frame, split)
    subsets = build_subset_manifest(train, (16, 32, 64), 2, 20261000)
    assert set(subsets[SAMPLE_ID]).isdisjoint(set(test[SAMPLE_ID]))
    for repeat in (0, 1):
        previous: set[int] = set()
        for size in (16, 32, 64):
            group = subsets.loc[
                (subsets["repeat"] == repeat)
                & (subsets["subset_size"] == size)
            ]
            labels = train.set_index(SAMPLE_ID).loc[group[SAMPLE_ID], TARGET]
            assert labels.value_counts().to_dict() == {0: size // 2, 1: size // 2}
            current = set(group[SAMPLE_ID])
            assert previous.issubset(current)
            previous = current


def test_default_data_config_uses_unopened_final_lockbox_seed():
    assert DataConfig().split_seed == 20260802
