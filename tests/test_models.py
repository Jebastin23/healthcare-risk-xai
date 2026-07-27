"""Tests for the RiskModel wrapper across all supported estimators."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from config import ModelConfig
from src.data_generator import FEATURE_NAMES, generate_clinical_data
from src.exception import HCException
from src.models import RiskModel, _build_estimator


def _training_data(n: int = 400):
    data = generate_clinical_data(n_samples=n, random_state=3)
    return data.X, data.y


@pytest.mark.parametrize(
    "name", ["logistic", "random_forest", "gradient_boosting", "mlp"]
)
def test_each_model_fits_and_predicts(name: str) -> None:
    x, y = _training_data()
    cfg = ModelConfig(name=name, n_estimators=40, max_iter=200)
    model = RiskModel(cfg).fit(x, y)

    proba = model.predict_proba(x)
    assert proba.shape == (len(y),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    preds = model.predict(x, threshold=0.5)
    assert set(np.unique(preds)).issubset({0, 1})
    # A fitted model should beat chance on its own training data.
    assert (preds == y).mean() > 0.6


def test_predict_before_fit_raises() -> None:
    cfg = ModelConfig(name="logistic")
    with pytest.raises(HCException):
        RiskModel(cfg).predict_proba(np.zeros((2, len(FEATURE_NAMES))))


def test_unknown_model_rejected() -> None:
    with pytest.raises(HCException):
        _build_estimator(ModelConfig(name="does_not_exist"))


def test_native_importance_shapes() -> None:
    x, y = _training_data()
    # Tree model exposes feature_importances_
    rf = RiskModel(ModelConfig(name="random_forest", n_estimators=30)).fit(x, y)
    imp = rf.native_importance()
    assert imp is not None and imp.shape == (len(FEATURE_NAMES),)

    # Linear model exposes coef_
    lr = RiskModel(ModelConfig(name="logistic")).fit(x, y)
    imp_lr = lr.native_importance()
    assert imp_lr is not None and imp_lr.shape == (len(FEATURE_NAMES),)


def test_save_and_load_round_trip(tmp_path) -> None:
    x, y = _training_data()
    cfg = ModelConfig(name="logistic")
    model = RiskModel(cfg).fit(x, y)
    before = model.predict_proba(x)

    path = tmp_path / "model.pkl"
    model.save(path)
    reloaded = RiskModel(cfg).load(path)
    after = reloaded.predict_proba(x)

    assert np.allclose(before, after)


def test_threshold_changes_positive_rate() -> None:
    x, y = _training_data()
    model = RiskModel(ModelConfig(name="gradient_boosting", n_estimators=40)).fit(x, y)
    low = model.predict(x, threshold=0.2).mean()
    high = model.predict(x, threshold=0.8).mean()
    # A stricter threshold flags fewer positives.
    assert low >= high
