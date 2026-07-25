"""Synthetic clinical dataset generation.

Produces a tabular dataset of patient records with clinically plausible, mutually
correlated features (age, BMI, blood pressure, glucose, cholesterol, smoking,
etc.) and a binary ``risk`` label generated from a known logistic ground-truth.
Because the data-generating process is explicit, we know which features *truly*
drive risk — invaluable for validating that the explainability module recovers
the right signal. The generator is fully offline and seeded, so the whole
pipeline (training, evaluation, explanation) runs with no downloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.logger import get_logger
from src.utils import sigmoid

logger = get_logger(__name__)

FEATURE_NAMES: List[str] = [
    "age",
    "bmi",
    "systolic_bp",
    "glucose",
    "cholesterol",
    "hdl",
    "smoker",
    "family_history",
    "physical_activity",
    "resting_heart_rate",
]


@dataclass
class Dataset:
    """A generated/loaded tabular dataset."""

    X: np.ndarray  # shape (n_samples, n_features)
    y: np.ndarray  # shape (n_samples,), binary
    feature_names: List[str]

    @property
    def n_features(self) -> int:  # noqa: D102
        return self.X.shape[1]

    def __len__(self) -> int:  # noqa: D105
        return len(self.y)


def generate_clinical_data(
    n_samples: int = 4000,
    positive_rate: float = 0.35,
    random_state: int = 42,
) -> Dataset:
    """Generate a synthetic clinical risk dataset.

    Parameters
    ----------
    n_samples:
        Number of patient records to generate.
    positive_rate:
        Approximate target prevalence of the positive (high-risk) class; the
        logistic intercept is calibrated to hit roughly this rate.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    Dataset
        Features, binary labels and feature names.
    """
    rng = np.random.default_rng(random_state)
    n = n_samples

    # -- generate correlated, plausible features ---------------------------
    age = np.clip(rng.normal(50, 15, n), 18, 90)
    # BMI rises slightly with age plus noise
    bmi = np.clip(rng.normal(27, 5, n) + (age - 50) * 0.03, 15, 55)
    # Systolic BP correlates with age and BMI
    systolic_bp = np.clip(
        90 + 0.5 * age + 0.4 * bmi + rng.normal(0, 10, n), 90, 200
    )
    # Glucose correlates with BMI
    glucose = np.clip(80 + 0.8 * (bmi - 27) + rng.normal(0, 15, n), 60, 300)
    cholesterol = np.clip(rng.normal(200, 35, n) + 0.3 * age, 120, 400)
    hdl = np.clip(rng.normal(55, 12, n) - 0.1 * bmi, 20, 100)
    smoker = (rng.random(n) < 0.25).astype(np.float64)
    family_history = (rng.random(n) < 0.30).astype(np.float64)
    physical_activity = np.clip(rng.normal(3, 1.5, n), 0, 7)  # days/week
    resting_heart_rate = np.clip(
        rng.normal(70, 10, n) + 4 * smoker - 1.0 * physical_activity, 40, 120
    )

    features = np.column_stack(
        [
            age,
            bmi,
            systolic_bp,
            glucose,
            cholesterol,
            hdl,
            smoker,
            family_history,
            physical_activity,
            resting_heart_rate,
        ]
    )

    # -- known logistic ground truth (standardised contributions) ----------
    # Coefficients encode clinical priors: glucose, BP, BMI, age, smoking and
    # family history increase risk; HDL and activity decrease it.
    z = _standardise(features)
    coeffs = np.array(
        [
            0.60,  # age
            0.55,  # bmi
            0.70,  # systolic_bp
            0.90,  # glucose (dominant driver)
            0.40,  # cholesterol
            -0.45,  # hdl (protective)
            0.65,  # smoker
            0.50,  # family_history
            -0.55,  # physical_activity (protective)
            0.25,  # resting_heart_rate
        ]
    )
    logits = z @ coeffs
    # Add mild interaction: smoker & high glucose compound risk.
    logits += 0.4 * (features[:, 6] * (z[:, 3] > 0.5))
    logits += rng.normal(0, 0.5, n)  # irreducible noise

    intercept = _calibrate_intercept(logits, positive_rate)
    probs = sigmoid(logits + intercept)
    y = (rng.random(n) < probs).astype(np.int64)

    logger.info(
        "Generated clinical data: n=%d, prevalence=%.3f", n, float(y.mean())
    )
    return Dataset(X=features, y=y, feature_names=list(FEATURE_NAMES))


def _standardise(x: np.ndarray) -> np.ndarray:
    """Z-score standardise columns (guarding zero variance)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def _calibrate_intercept(
    logits: np.ndarray, target_rate: float, tol: float = 0.01
) -> float:
    """Bisection search for an intercept giving ~``target_rate`` positives."""
    low, high = -10.0, 10.0
    for _ in range(60):
        mid = (low + high) / 2
        rate = float(sigmoid(logits + mid).mean())
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def true_coefficients() -> Tuple[List[str], np.ndarray]:
    """Return the ground-truth standardised coefficients used by the generator.

    Useful for validating that an explainer recovers the right feature ranking.
    """
    coeffs = np.array(
        [0.60, 0.55, 0.70, 0.90, 0.40, -0.45, 0.65, 0.50, -0.55, 0.25]
    )
    return list(FEATURE_NAMES), coeffs
