"""Tests for the model wrapper, trainer, and FastAPI endpoints (offline)."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
from config import Config, ModelConfig
from src.data_loader import DataLoader
from src.models import RiskModel
from src.trainer import Trainer


@pytest.fixture(scope="module")
def small_split():
    cfg = Config()
    cfg.data.n_samples = 800
    return DataLoader(cfg.data).load()


def test_model_fit_predict_proba_range(small_split) -> None:
    model = RiskModel(ModelConfig(name="logistic")).fit(
        small_split.x_train, small_split.y_train
    )
    proba = model.predict_proba(small_split.x_test)
    assert proba.shape == (len(small_split.y_test),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_model_requires_fit() -> None:
    with pytest.raises(Exception):
        RiskModel(ModelConfig(name="logistic")).predict_proba(np.zeros((2, 10)))


def test_native_importance_available_for_trees(small_split) -> None:
    model = RiskModel(ModelConfig(name="random_forest", n_estimators=20)).fit(
        small_split.x_train, small_split.y_train
    )
    imp = model.native_importance()
    assert imp is not None
    assert len(imp) == small_split.x_train.shape[1]


def test_unknown_model_rejected() -> None:
    with pytest.raises(Exception):
        RiskModel(ModelConfig(name="does_not_exist"))


def test_cross_validation_runs(small_split) -> None:
    cfg = Config()
    cfg.model.name = "logistic"
    x = np.concatenate([small_split.x_train, small_split.x_val])
    y = np.concatenate([small_split.y_train, small_split.y_val])
    result = Trainer(cfg).cross_validate(x, y, n_folds=3)
    assert result.n_folds == 3
    assert "roc_auc" in result.mean
    assert 0.0 <= result.mean["roc_auc"] <= 1.0


@pytest.fixture(scope="module")
def client():
    cfg = Config()
    cfg.data.n_samples = 800
    cfg.model.name = "logistic"
    # Manually run the lifespan setup with a small, fast config.
    from src.explainer import Explainer

    split = DataLoader(cfg.data).load()
    model = Trainer(cfg).fit(split.x_train, split.y_train)
    api._state["model"] = model
    api._state["feature_names"] = split.feature_names
    api._state["explainer"] = Explainer(
        model.predict_proba, split.x_train[:50], split.feature_names, cfg.explain
    )
    with TestClient(api.app) as c:
        # override the lifespan-populated state after startup
        api._state["model"] = model
        api._state["feature_names"] = split.feature_names
        api._state["explainer"] = Explainer(
            model.predict_proba, split.x_train[:50], split.feature_names, cfg.explain
        )
        yield c, split


def test_health(client) -> None:
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_features_endpoint(client) -> None:
    c, split = client
    r = c.get("/features")
    assert r.status_code == 200
    assert len(r.json()["features"]) == split.x_train.shape[1]


def test_predict_endpoint(client) -> None:
    c, split = client
    row = split.x_test[0].tolist()
    r = c.post("/predict", json={"features": row})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["predicted_class"] in (0, 1)


def test_predict_wrong_length_rejected(client) -> None:
    c, _ = client
    r = c.post("/predict", json={"features": [0.1, 0.2]})
    assert r.status_code == 422


def test_explain_endpoint(client) -> None:
    c, split = client
    row = split.x_test[0].tolist()
    r = c.post("/explain", json={"features": row})
    assert r.status_code == 200
    body = r.json()
    assert "contributions" in body
    assert len(body["contributions"]) == split.x_train.shape[1]
