"""Classification metrics for binary clinical risk prediction.

Implemented in pure NumPy so the evaluation path has no scikit-learn dependency
and is trivially testable offline. Includes threshold metrics (accuracy,
precision, recall, specificity, F1), the confusion matrix, ROC-AUC via the
rank-based (Mann-Whitney) formulation, and a simple calibration summary. In a
clinical setting recall/sensitivity and calibration often matter more than raw
accuracy, so all are reported.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.exception import HCException


def _check(y_true: np.ndarray, y_score: np.ndarray) -> None:
    if y_true.shape != y_score.shape:
        raise HCException(
            ValueError(f"Shape mismatch: {y_true.shape} vs {y_score.shape}")
        )
    if y_true.size == 0:
        raise HCException(ValueError("Empty arrays"))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return TP/FP/TN/FN counts for binary predictions."""
    _check(y_true, y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions."""
    _check(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Positive predictive value TP / (TP + FP)."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["tp"] + cm["fp"]
    return float(cm["tp"] / denom) if denom else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sensitivity / true-positive rate TP / (TP + FN)."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["tp"] + cm["fn"]
    return float(cm["tp"] / denom) if denom else 0.0


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True-negative rate TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["tn"] + cm["fp"]
    return float(cm["tn"] / denom) if denom else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Harmonic mean of precision and recall."""
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    denom = prec + rec
    return float(2 * prec * rec / denom) if denom else 0.0


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via the rank-based Mann-Whitney U statistic.

    Equivalent to the probability that a random positive is ranked above a random
    negative. Robust to score scale and requires no threshold sweep.
    """
    _check(y_true, y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    _assign_tie_ranks(y_score, ranks)
    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    n_pos = len(pos)
    n_neg = len(neg)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    """Average ranks across tied score groups (in place)."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = np.mean(ranks[order[i : j + 1]])
            ranks[order[i : j + 1]] = avg
        i = j + 1


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and outcomes."""
    _check(y_true, y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_all(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Return a dictionary of all metrics at the given decision threshold."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
    }
    return metrics
