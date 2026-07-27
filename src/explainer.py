"""Explainable-AI (XAI) module.

Three complementary, model-agnostic explanation techniques implemented from
scratch in NumPy:

* **Monte-Carlo Shapley values** (:meth:`Explainer.shapley_values`) — a local
  explanation attributing a single prediction to each feature, based on the
  game-theoretic Shapley value approximated by sampling random feature orderings
  (Štrumbelj & Kononenko, 2014). Contributions sum (in expectation) to the gap
  between the instance's prediction and the background-average prediction.
* **Permutation importance** (:meth:`Explainer.permutation_importance`) — a global
  measure of how much a metric degrades when each feature is shuffled.
* **Partial dependence & ICE** (:meth:`Explainer.partial_dependence`) — how the
  average (and per-instance) prediction varies as one feature is swept across a
  grid.

The module depends only on a prediction function ``f(X) -> proba`` and NumPy, so
it works with any of the project's models (or any external callable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from config import ExplainConfig
from src.exception import HCException
from src.logger import get_logger

logger = get_logger(__name__)

PredictFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class ShapleyExplanation:
    """Local attribution for a single instance."""

    values: np.ndarray  # per-feature Shapley contributions
    base_value: float  # mean prediction over the background
    prediction: float  # model prediction for the instance
    feature_names: List[str] = field(default_factory=list)

    def as_ranking(self) -> List[Dict[str, float]]:
        """Return features sorted by absolute contribution (descending)."""
        order = np.argsort(-np.abs(self.values))
        return [
            {
                "feature": self.feature_names[i]
                if self.feature_names
                else f"f{i}",
                "contribution": float(self.values[i]),
            }
            for i in order
        ]


class Explainer:
    """Model-agnostic explainer over a prediction function and background data."""

    def __init__(
        self,
        predict_fn: PredictFn,
        background: np.ndarray,
        feature_names: List[str] | None = None,
        cfg: ExplainConfig | None = None,
    ) -> None:
        if background.ndim != 2:
            raise HCException(ValueError("background must be 2-D"))
        self.predict_fn = predict_fn
        self.background = background
        self.n_features = background.shape[1]
        self.feature_names = feature_names or [
            f"f{i}" for i in range(self.n_features)
        ]
        self.cfg = cfg or ExplainConfig()
        self._rng = np.random.default_rng(self.cfg.random_state)
        self._base_value = float(np.mean(self.predict_fn(background)))

    # -- local: Monte-Carlo Shapley values ---------------------------------
    def shapley_values(
        self, instance: np.ndarray, n_samples: int | None = None
    ) -> ShapleyExplanation:
        """Approximate Shapley values for a single ``instance``.

        For each of ``n_samples`` random feature permutations, and for each
        feature, we compare the prediction of a coalition that *excludes* the
        feature with one that *includes* it, drawing the excluded features'
        values from a random background row. Averaging the marginal contribution
        over permutations yields the Shapley estimate.
        """
        instance = np.asarray(instance, dtype=np.float64).ravel()
        if instance.shape[0] != self.n_features:
            raise HCException(
                ValueError(
                    f"instance has {instance.shape[0]} features, "
                    f"expected {self.n_features}"
                )
            )
        n_samples = n_samples or self.cfg.n_shapley_samples
        phi = np.zeros(self.n_features, dtype=np.float64)

        for _ in range(n_samples):
            perm = self._rng.permutation(self.n_features)
            reference = self.background[
                self._rng.integers(0, len(self.background))
            ].copy()
            # Start from the reference (all features "off").
            synth = reference.copy()
            pred_without = float(self.predict_fn(synth[None, :])[0])
            # Turn features on one at a time, in permutation order.
            for feature in perm:
                synth[feature] = instance[feature]
                pred_with = float(self.predict_fn(synth[None, :])[0])
                phi[feature] += pred_with - pred_without
                pred_without = pred_with

        phi /= n_samples
        prediction = float(self.predict_fn(instance[None, :])[0])
        logger.debug(
            "Shapley: base=%.4f pred=%.4f sum_phi=%.4f",
            self._base_value,
            prediction,
            float(phi.sum()),
        )
        return ShapleyExplanation(
            values=phi,
            base_value=self._base_value,
            prediction=prediction,
            feature_names=self.feature_names,
        )

    # -- global: permutation importance ------------------------------------
    def permutation_importance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metric: Callable[[np.ndarray, np.ndarray], float],
        n_repeats: int | None = None,
    ) -> Dict[str, np.ndarray]:
        """Global feature importance by metric degradation under shuffling.

        Returns the mean and std importance per feature, where importance is the
        drop in ``metric`` when a feature column is randomly permuted.
        """
        n_repeats = n_repeats or self.cfg.n_permutation_repeats
        baseline = metric(y, self.predict_fn(x))
        means = np.zeros(self.n_features)
        stds = np.zeros(self.n_features)
        for feature in range(self.n_features):
            scores = np.empty(n_repeats)
            for r in range(n_repeats):
                permuted = x.copy()
                self._rng.shuffle(permuted[:, feature])
                scores[r] = baseline - metric(y, self.predict_fn(permuted))
            means[feature] = float(scores.mean())
            stds[feature] = float(scores.std())
        return {"importance_mean": means, "importance_std": stds}

    # -- global: partial dependence & ICE ----------------------------------
    def partial_dependence(
        self, x: np.ndarray, feature: int, grid_resolution: int | None = None
    ) -> Dict[str, np.ndarray]:
        """Compute the partial-dependence curve and ICE lines for a feature.

        The grid spans the observed range of the feature. For each grid value the
        feature column is set to that value for every row and the mean prediction
        (PD) and per-row predictions (ICE) are recorded.
        """
        if not 0 <= feature < self.n_features:
            raise HCException(ValueError("feature index out of range"))
        grid_resolution = grid_resolution or self.cfg.pdp_grid_resolution
        col = x[:, feature]
        grid = np.linspace(col.min(), col.max(), grid_resolution)
        ice = np.empty((len(x), grid_resolution), dtype=np.float64)
        for j, value in enumerate(grid):
            probe = x.copy()
            probe[:, feature] = value
            ice[:, j] = self.predict_fn(probe)
        return {"grid": grid, "pd": ice.mean(axis=0), "ice": ice}

    @property
    def base_value(self) -> float:
        """Mean prediction over the background dataset."""
        return self._base_value
