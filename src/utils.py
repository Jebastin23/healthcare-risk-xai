"""General-purpose helpers shared across the healthcare ML pipeline.

Kept dependency-light (standard library plus NumPy) so they can be imported
anywhere without pulling in scikit-learn or other heavy frameworks.
"""
from __future__ import annotations

import json
import pickle
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.exception import HCException
from src.logger import get_logger

logger = get_logger(__name__)


def read_json(path: str | Path) -> Any:
    """Load and return the JSON content at ``path``."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise HCException(exc) from exc


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Serialise ``obj`` to JSON at ``path``, creating parent dirs as needed."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        def _default(value: Any) -> Any:
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            return str(value)

        with target.open("w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=indent, ensure_ascii=False, default=_default)
    except Exception as exc:  # noqa: BLE001
        raise HCException(exc) from exc


def save_pickle(obj: Any, path: str | Path) -> None:
    """Persist a Python object (e.g. a fitted model) to ``path`` via pickle."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(obj, handle)
        logger.info("Saved object to %s", target)
    except Exception as exc:  # noqa: BLE001
        raise HCException(exc) from exc


def load_pickle(path: str | Path) -> Any:
    """Load a pickled Python object from ``path``."""
    try:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise HCException(exc) from exc


@contextmanager
def timer(label: str = "operation") -> Iterator[None]:
    """Context manager that logs the wall-clock duration of a code block."""
    start = time.perf_counter()
    logger.debug("Starting %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s finished in %.3fs", label, elapsed)


def set_seed(seed: int) -> None:
    """Seed NumPy (and Python's ``random``) for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out
