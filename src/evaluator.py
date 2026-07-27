"""Evaluation orchestration: metrics plus a global-importance summary.

:class:`Evaluator` produces a JSON-serialisable report combining threshold and
ranking metrics with model-native and permutation-based feature importances, so a
single call yields both "how good is it?" and "why does it decide?" summaries.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from config import Config
from src.explainer import Explainer
from src.logger import get_logger
from src.metrics import compute_all, roc_auc
from src.models import RiskModel

logger = get_logger(__name__)


class Evaluator:
    """Compute a full evaluation-and-explanation report for a fitted model."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        model: RiskModel,
        x_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        background: np.ndarray,
    ) -> Dict[str, object]:
        """Return metrics, native importances and permutation importances."""
        proba = model.predict_proba(x_test)
        metrics = compute_all(y_test, proba)
        logger.info("Test AUC=%.3f F1=%.3f", metrics["roc_auc"], metrics["f1"])

        importances: Dict[str, object] = {}
        native = model.native_importance()
        if native is not None:
            importances["native"] = self._rank(feature_names, native)

        explainer = Explainer(
            model.predict_proba, background, feature_names, self.cfg.explain
        )
        perm = explainer.permutation_importance(x_test, y_test, roc_auc)
        importances["permutation"] = self._rank(
            feature_names, perm["importance_mean"]
        )

        return {
            "model": self.cfg.model.name,
            "metrics": metrics,
            "importances": importances,
        }

    @staticmethod
    def _rank(
        names: List[str], values: np.ndarray
    ) -> List[Dict[str, float]]:
        """Return (feature, importance) pairs sorted by descending importance."""
        order = np.argsort(-np.asarray(values))
        return [
            {"feature": names[i], "importance": float(values[i])} for i in order
        ]
