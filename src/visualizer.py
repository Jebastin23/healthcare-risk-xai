"""Plotting utilities (matplotlib, non-interactive Agg backend).

Saves PNG figures for the ROC curve, confusion matrix, global feature importance,
a local Shapley explanation (a horizontal contribution bar chart), and
partial-dependence curves. Agg is selected explicitly so the module works
headlessly on servers and in CI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from src.exception import HCException  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.metrics import confusion_matrix  # noqa: E402

logger = get_logger(__name__)


def plot_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray, out_path: str | Path
) -> Path:
    """Plot the ROC curve by sweeping the decision threshold."""
    try:
        thresholds = np.linspace(0, 1, 101)
        tpr, fpr = [], []
        pos = np.sum(y_true == 1)
        neg = np.sum(y_true == 0)
        for thr in thresholds:
            pred = (y_score >= thr).astype(int)
            tp = np.sum((y_true == 1) & (pred == 1))
            fp = np.sum((y_true == 0) & (pred == 1))
            tpr.append(tp / pos if pos else 0.0)
            fpr.append(fp / neg if neg else 0.0)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color="#1f77b4", label="model")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        return _save(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        raise HCException(exc) from exc


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path
) -> Path:
    """Plot a 2x2 confusion matrix heatmap."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(matrix[i, j]), ha="center", va="center",
                    color="black", fontsize=14,
                )
        ax.set_title("Confusion matrix")
        fig.colorbar(im, ax=ax, fraction=0.046)
        return _save(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        raise HCException(exc) from exc


def plot_importance(
    names: Sequence[str], values: Sequence[float], out_path: str | Path,
    title: str = "Feature importance",
) -> Path:
    """Plot a horizontal bar chart of feature importances (descending)."""
    try:
        order = np.argsort(values)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(names))))
        ax.barh(
            [names[i] for i in order],
            [values[i] for i in order],
            color="#2ca02c",
        )
        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x", alpha=0.3)
        return _save(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        raise HCException(exc) from exc


def plot_shapley(
    names: Sequence[str], values: Sequence[float], out_path: str | Path,
    title: str = "Local explanation (Shapley)",
) -> Path:
    """Plot signed per-feature Shapley contributions for one prediction."""
    try:
        vals = np.asarray(values)
        order = np.argsort(np.abs(vals))
        colors = ["#d62728" if vals[i] > 0 else "#1f77b4" for i in order]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(names))))
        ax.barh([names[i] for i in order], [vals[i] for i in order], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Contribution to predicted risk (+ increases, − decreases)")
        return _save(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        raise HCException(exc) from exc


def plot_partial_dependence(
    grid: np.ndarray, pd_values: np.ndarray, feature_name: str,
    out_path: str | Path,
) -> Path:
    """Plot a partial-dependence curve for one feature."""
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(grid, pd_values, color="#9467bd", linewidth=2)
        ax.set_title(f"Partial dependence: {feature_name}")
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Average predicted risk")
        ax.grid(True, alpha=0.3)
        return _save(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        raise HCException(exc) from exc


def _save(fig: "plt.Figure", out_path: str | Path) -> Path:
    """Persist a figure to disk and close it to free memory."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=120)
    plt.close(fig)
    logger.info("Saved plot to %s", target)
    return target
