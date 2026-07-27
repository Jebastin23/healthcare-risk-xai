"""Model registry: a uniform wrapper around scikit-learn classifiers.

:class:`RiskModel` gives the rest of the codebase one interface —
``fit`` / ``predict`` / ``predict_proba`` / ``save`` / ``load`` — regardless of
whether the underlying estimator is a logistic regression, a random forest,
gradient boosting, or a multi-layer perceptron. All four expose calibrated
probabilities via ``predict_proba``, which the explainability module relies on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from config import ModelConfig
from src.exception import HCException
from src.logger import get_logger
from src.utils import load_pickle, save_pickle

logger = get_logger(__name__)


def _build_estimator(cfg: ModelConfig) -> Any:
    """Instantiate a scikit-learn estimator from the model configuration."""
    name = cfg.name.lower()
    class_weight = "balanced" if cfg.class_weight_balanced else None
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=cfg.C, class_weight=class_weight, max_iter=1000
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth if cfg.max_depth > 0 else None,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            random_state=42,
        )
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=tuple(cfg.hidden_layer_sizes),
            max_iter=cfg.max_iter,
            random_state=42,
            early_stopping=True,
        )
    raise HCException(ValueError(f"Unknown model '{cfg.name}'"))


class RiskModel:
    """Uniform wrapper over a scikit-learn binary classifier."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.estimator = _build_estimator(cfg)
        self._fitted = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RiskModel":
        """Fit the underlying estimator."""
        self.estimator.fit(x, y)
        self._fitted = True
        logger.info("Fitted model '%s'", self.cfg.name)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return the probability of the positive class for each row."""
        if not self._fitted:
            raise HCException(RuntimeError("Model must be fit before predict"))
        proba = self.estimator.predict_proba(x)
        return proba[:, 1]

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at the given decision threshold."""
        return (self.predict_proba(x) >= threshold).astype(np.int64)

    def native_importance(self) -> np.ndarray | None:
        """Return model-native feature importances if available, else None."""
        if hasattr(self.estimator, "feature_importances_"):
            return np.asarray(self.estimator.feature_importances_)
        if hasattr(self.estimator, "coef_"):
            return np.abs(np.ravel(self.estimator.coef_))
        return None

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk."""
        save_pickle(self.estimator, path)

    def load(self, path: str | Path) -> "RiskModel":
        """Load a previously saved estimator."""
        self.estimator = load_pickle(path)
        self._fitted = True
        return self
