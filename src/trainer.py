"""Model training and cross-validation orchestration.

:class:`Trainer` fits a :class:`~src.models.RiskModel` on the training split and
optionally runs stratified k-fold cross-validation to estimate generalisation
before committing to a final fit. It is deliberately thin — the heavy lifting is
in the estimator — but it centralises seeding, logging and CV so the CLIs and API
share one code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from config import Config
from src.exception import HCException
from src.logger import get_logger
from src.metrics import compute_all
from src.models import RiskModel

logger = get_logger(__name__)


@dataclass
class CVResult:
    """Cross-validation metrics aggregated across folds."""

    per_fold: List[Dict[str, float]] = field(default_factory=list)
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    n_folds: int = 0

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable summary."""
        return {
            "n_folds": self.n_folds,
            "mean": self.mean,
            "std": self.std,
            "per_fold": self.per_fold,
        }


class Trainer:
    """Fit a risk model and optionally cross-validate it."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def fit(self, x: np.ndarray, y: np.ndarray) -> RiskModel:
        """Fit a fresh model on the full training data."""
        model = RiskModel(self.cfg.model).fit(x, y)
        return model

    def cross_validate(
        self, x: np.ndarray, y: np.ndarray, n_folds: int = 5
    ) -> CVResult:
        """Run stratified k-fold cross-validation and aggregate metrics."""
        if n_folds < 2:
            raise HCException(ValueError("n_folds must be >= 2"))
        folds = self._stratified_folds(y, n_folds)
        per_fold: List[Dict[str, float]] = []
        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            model = RiskModel(self.cfg.model).fit(x[train_idx], y[train_idx])
            proba = model.predict_proba(x[val_idx])
            metrics = compute_all(y[val_idx], proba)
            per_fold.append(metrics)
            logger.info(
                "Fold %d/%d: AUC=%.3f F1=%.3f",
                k + 1,
                n_folds,
                metrics["roc_auc"],
                metrics["f1"],
            )
        return self._aggregate(per_fold)

    def _stratified_folds(self, y: np.ndarray, n_folds: int) -> List[np.ndarray]:
        """Build stratified fold index arrays preserving class balance."""
        rng = np.random.default_rng(self.cfg.data.random_state)
        folds: List[List[int]] = [[] for _ in range(n_folds)]
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            for i, idx in enumerate(cls_idx):
                folds[i % n_folds].append(int(idx))
        return [np.array(sorted(f)) for f in folds]

    @staticmethod
    def _aggregate(per_fold: List[Dict[str, float]]) -> CVResult:
        if not per_fold:
            raise HCException(RuntimeError("No folds evaluated"))
        keys = per_fold[0].keys()
        mean = {k: float(np.mean([f[k] for f in per_fold])) for k in keys}
        std = {k: float(np.std([f[k] for f in per_fold])) for k in keys}
        return CVResult(
            per_fold=per_fold, mean=mean, std=std, n_folds=len(per_fold)
        )
