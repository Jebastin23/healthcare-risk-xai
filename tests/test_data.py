"""Tests for synthetic data generation and the loader/splitter."""
from __future__ import annotations

import numpy as np
import pytest

from config import DataConfig
from src.data_generator import FEATURE_NAMES, generate_clinical_data, true_coefficients
from src.data_loader import DataLoader


def test_generate_shapes_and_labels() -> None:
    data = generate_clinical_data(n_samples=500, random_state=0)
    assert data.X.shape == (500, len(FEATURE_NAMES))
    assert set(np.unique(data.y)).issubset({0, 1})
    assert data.n_features == len(FEATURE_NAMES)


def test_generate_prevalence_is_calibrated() -> None:
    data = generate_clinical_data(n_samples=3000, positive_rate=0.3, random_state=1)
    assert 0.24 <= data.y.mean() <= 0.36  # near the requested 0.3


def test_generate_is_reproducible() -> None:
    a = generate_clinical_data(n_samples=200, random_state=7)
    b = generate_clinical_data(n_samples=200, random_state=7)
    assert np.allclose(a.X, b.X)
    assert np.array_equal(a.y, b.y)


def test_true_coefficients_align_with_features() -> None:
    names, coeffs = true_coefficients()
    assert names == FEATURE_NAMES
    assert len(coeffs) == len(FEATURE_NAMES)


def test_loader_stratified_split_preserves_balance() -> None:
    cfg = DataConfig(n_samples=2000, test_size=0.2, val_size=0.1, random_state=3)
    split = DataLoader(cfg).load()
    # no overlap and correct sizes
    assert len(split.y_train) + len(split.y_val) + len(split.y_test) == 2000
    # class balance is similar across splits (stratified)
    assert abs(split.y_train.mean() - split.y_test.mean()) < 0.05


def test_loader_scales_training_to_zero_mean() -> None:
    cfg = DataConfig(n_samples=1000, random_state=5)
    split = DataLoader(cfg).load()
    # training features are standardised
    assert np.allclose(split.x_train.mean(axis=0), 0.0, atol=1e-6)


def test_loader_rejects_unknown_source() -> None:
    cfg = DataConfig(source="nope")
    with pytest.raises(Exception):
        DataLoader(cfg).load()
