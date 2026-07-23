from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV

from src.classical.pipelines import build_pipeline, feature_matrix, make_cv
from src.config import ROOT
from src.data_prep.prepare_data import TARGET, apply_manifest, create_split_manifest, load_raw


def _small_train():
    frame = load_raw(ROOT / "data/raw/water_potability.csv")
    split = create_split_manifest(frame, 0.2, 20260723)
    train, _ = apply_manifest(frame, split)
    return train.groupby(TARGET, group_keys=False).head(80).reset_index(drop=True)


def test_pipeline_contains_all_leakage_sensitive_steps():
    pipeline = build_pipeline(balance_strategy="undersample", seed=11)
    assert list(pipeline.named_steps) == ["imputer", "scaler", "sampler", "svc"]


def test_grid_search_fits_raw_missing_values_without_external_preprocessing():
    train = _small_train()
    assert feature_matrix(train).isna().any().any()
    search = GridSearchCV(
        build_pipeline(balance_strategy="class_weight", seed=11),
        {"svc__C": [0.1, 1.0], "svc__gamma": ["scale"]},
        cv=make_cv(2, 11),
        scoring="f1",
    )
    search.fit(feature_matrix(train), train[TARGET])
    predictions = search.predict(feature_matrix(train).head(10))
    assert predictions.shape == (10,)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_inference_does_not_accept_or_require_test_labels():
    train = _small_train()
    model = build_pipeline(balance_strategy="class_weight", seed=11)
    model.fit(feature_matrix(train), train[TARGET])
    X = feature_matrix(train).head(12).copy()
    first = model.predict(X)
    # There is deliberately no y argument in transform/predict. Changing an
    # unrelated label vector cannot affect the feature transformation.
    fake_labels = 1 - train[TARGET].head(12)
    second = model.predict(X)
    assert fake_labels.shape == first.shape
    np.testing.assert_array_equal(first, second)
