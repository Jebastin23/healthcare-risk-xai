"""Tests for the XAI explainer: Shapley axioms, permutation importance, PDP.

A linear prediction function is used as an oracle because a linear model has
closed-form Shapley values: for f(x) = w·x, the Shapley value of feature i for
instance x against a background mean m is ``w_i * (x_i - m_i)``. This lets us
check the Monte-Carlo estimator for both the efficiency (additivity) axiom and
per-feature correctness.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import ExplainConfig
from src.explainer import Explainer
from src.metrics import roc_auc


@pytest.fixture()
def linear_setup():
    """A deterministic linear model plus a background sample."""
    rng = np.random.default_rng(0)
    weights = np.array([2.0, -1.0, 0.5, 0.0])

    def predict(x: np.ndarray) -> np.ndarray:
        return x @ weights

    background = rng.normal(0, 1, size=(300, 4))
    cfg = ExplainConfig(n_shapley_samples=3000, random_state=1)
    return predict, weights, background, cfg


def test_shapley_efficiency_axiom(linear_setup) -> None:
    """Sum of Shapley values ≈ prediction − base value."""
    predict, weights, background, cfg = linear_setup
    explainer = Explainer(predict, background, cfg=cfg)
    instance = np.array([1.5, -2.0, 3.0, 1.0])
    exp = explainer.shapley_values(instance)
    gap = exp.prediction - exp.base_value
    assert exp.values.sum() == pytest.approx(gap, abs=0.1)


def test_shapley_matches_linear_closed_form(linear_setup) -> None:
    """Per-feature Shapley values ≈ w_i * (x_i - mean_background_i)."""
    predict, weights, background, cfg = linear_setup
    explainer = Explainer(predict, background, cfg=cfg)
    instance = np.array([1.5, -2.0, 3.0, 1.0])
    exp = explainer.shapley_values(instance)
    expected = weights * (instance - background.mean(axis=0))
    assert np.allclose(exp.values, expected, atol=0.1)


def test_zero_weight_feature_has_negligible_contribution(linear_setup) -> None:
    predict, weights, background, cfg = linear_setup
    explainer = Explainer(predict, background, cfg=cfg)
    instance = np.array([1.5, -2.0, 3.0, 5.0])  # feature 3 has weight 0
    exp = explainer.shapley_values(instance)
    assert abs(exp.values[3]) < 0.1


def test_ranking_is_sorted_by_absolute_value(linear_setup) -> None:
    predict, _, background, cfg = linear_setup
    explainer = Explainer(predict, background, ["a", "b", "c", "d"], cfg)
    exp = explainer.shapley_values(np.array([1.5, -2.0, 3.0, 1.0]))
    ranking = exp.as_ranking()
    mags = [abs(r["contribution"]) for r in ranking]
    assert mags == sorted(mags, reverse=True)


def test_permutation_importance_identifies_signal() -> None:
    """The feature the label depends on should have the highest importance."""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, size=(400, 3))
    # label depends only on feature 0
    y = (x[:, 0] > 0).astype(int)

    def predict(a: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-3 * a[:, 0]))

    explainer = Explainer(predict, x, ["s", "n1", "n2"], ExplainConfig(random_state=0))
    result = explainer.permutation_importance(x, y, roc_auc, n_repeats=5)
    assert int(np.argmax(result["importance_mean"])) == 0


def test_partial_dependence_shapes(linear_setup) -> None:
    predict, _, background, cfg = linear_setup
    explainer = Explainer(predict, background, cfg=cfg)
    pdp = explainer.partial_dependence(background, feature=0, grid_resolution=15)
    assert pdp["grid"].shape == (15,)
    assert pdp["pd"].shape == (15,)
    assert pdp["ice"].shape == (len(background), 15)
    # For a linear model with positive weight, PD should increase with the feature.
    assert pdp["pd"][-1] > pdp["pd"][0]


def test_instance_dimension_validation(linear_setup) -> None:
    predict, _, background, cfg = linear_setup
    explainer = Explainer(predict, background, cfg=cfg)
    with pytest.raises(Exception):
        explainer.shapley_values(np.array([1.0, 2.0]))  # wrong length
