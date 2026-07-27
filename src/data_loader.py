"""Load a clinical dataset (synthetic or CSV) and split it for modelling.

:class:`DataLoader` returns train/validation/test splits together with the fitted
preprocessing objects. Splitting is stratified on the label so class balance is
preserved across splits, and the scaler is fit on the training split only to
avoid leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import DataConfig
from src.data_generator import Dataset, generate_clinical_data
from src.exception import HCException
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SplitData:
    """Container for train/val/test splits and metadata."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray

    def summary(self) -> dict:
        """Return a small dictionary describing the splits."""
        return {
            "n_train": int(len(self.y_train)),
            "n_val": int(len(self.y_val)),
            "n_test": int(len(self.y_test)),
            "n_features": int(self.x_train.shape[1]),
            "train_prevalence": float(self.y_train.mean()),
            "test_prevalence": float(self.y_test.mean()),
        }


class DataLoader:
    """Produce split, scaled data from synthetic generation or a CSV file."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg

    def load(self) -> SplitData:
        """Load, split (stratified) and scale the dataset."""
        dataset = self._load_raw()
        return self._split_and_scale(dataset)

    # -- raw loading --------------------------------------------------------
    def _load_raw(self) -> Dataset:
        if self.cfg.source == "synthetic":
            return generate_clinical_data(
                n_samples=self.cfg.n_samples,
                positive_rate=self.cfg.positive_rate,
                random_state=self.cfg.random_state,
            )
        if self.cfg.source == "csv":
            return self._load_csv()
        raise HCException(ValueError(f"Unknown data source: {self.cfg.source}"))

    def _load_csv(self) -> Dataset:
        path = Path(self.cfg.csv_path)
        if not path.is_file():
            raise HCException(FileNotFoundError(f"CSV not found: {path}"))
        import pandas as pd  # lazy import

        frame = pd.read_csv(path)
        if self.cfg.target_column not in frame.columns:
            raise HCException(
                ValueError(f"Target column '{self.cfg.target_column}' not found")
            )
        y = frame[self.cfg.target_column].to_numpy(dtype=np.int64)
        feature_frame = frame.drop(columns=[self.cfg.target_column])
        # keep numeric columns only; simple mean-imputation of NaNs
        feature_frame = feature_frame.select_dtypes(include=[np.number])
        x = feature_frame.to_numpy(dtype=np.float64)
        x = self._impute(x)
        logger.info("Loaded %d rows, %d features from %s", len(y), x.shape[1], path.name)
        return Dataset(X=x, y=y, feature_names=list(feature_frame.columns))

    @staticmethod
    def _impute(x: np.ndarray) -> np.ndarray:
        """Mean-impute NaNs column-wise."""
        col_mean = np.nanmean(x, axis=0)
        inds = np.where(np.isnan(x))
        x[inds] = np.take(col_mean, inds[1])
        return x

    # -- splitting & scaling ------------------------------------------------
    def _split_and_scale(self, dataset: Dataset) -> SplitData:
        x_train, x_val, x_test, y_train, y_val, y_test = self._stratified_split(
            dataset.X, dataset.y
        )
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std == 0] = 1.0

        def scale(a: np.ndarray) -> np.ndarray:
            return (a - mean) / std

        logger.info(
            "Split: train=%d val=%d test=%d", len(y_train), len(y_val), len(y_test)
        )
        return SplitData(
            x_train=scale(x_train),
            y_train=y_train,
            x_val=scale(x_val),
            y_val=y_val,
            x_test=scale(x_test),
            y_test=y_test,
            feature_names=dataset.feature_names,
            scaler_mean=mean,
            scaler_std=std,
        )

    def _stratified_split(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Stratified train/val/test split preserving class balance."""
        rng = np.random.default_rng(self.cfg.random_state)
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            n_cls = len(cls_idx)
            n_test = int(n_cls * self.cfg.test_size)
            n_val = int(n_cls * self.cfg.val_size)
            test_idx.extend(cls_idx[:n_test].tolist())
            val_idx.extend(cls_idx[n_test : n_test + n_val].tolist())
            train_idx.extend(cls_idx[n_test + n_val :].tolist())
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
        return (
            x[train_idx],
            x[val_idx],
            x[test_idx],
            y[train_idx],
            y[val_idx],
            y[test_idx],
        )
