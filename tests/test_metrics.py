"""Unit tests for the classification metrics module."""
from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    accuracy,
    compute_all,
    confusion_matrix,
    f1_score,
    precision,
    recall,
    roc_auc,
    specificity,
)


def test_perfect_predictions() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])
    m = compute_all(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0


def test_confusion_matrix_counts() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    cm = confusion_matrix(y_true, y_pred)
    assert cm == {"tp": 1, "fn": 1, "tn": 1, "fp": 1}


def test_precision_recall_specificity() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    assert precision(y_true, y_pred) == pytest.approx(0.5)
    assert recall(y_true, y_pred) == pytest.approx(0.5)
    assert specificity(y_true, y_pred) == pytest.approx(0.5)
    assert f1_score(y_true, y_pred) == pytest.approx(0.5)


def test_roc_auc_ranking() -> None:
    # Positives all score higher than negatives -> AUC = 1.
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y_true, y_score) == pytest.approx(1.0)
    # Reversed -> AUC = 0.
    assert roc_auc(y_true, 1 - y_score) == pytest.approx(0.0)


def test_roc_auc_handles_ties() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])  # all tied -> AUC 0.5
    assert roc_auc(y_true, y_score) == pytest.approx(0.5)


def test_accuracy_half() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(Exception):
        accuracy(np.array([1, 0]), np.array([1]))
