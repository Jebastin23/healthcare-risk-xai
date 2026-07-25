"""Typed, dataclass-based configuration for the healthcare risk project.

Every setting has an explicit type and a default. A YAML file may override any
subset of values via :func:`load_config`; anything absent falls back to the
defaults here. A module-level :data:`config` singleton is provided for
convenient importing across the codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

_ROOT = Path(__file__).resolve().parent


@dataclass
class PathConfig:
    """Filesystem locations used throughout the project."""

    root: Path = _ROOT
    data_dir: Path = _ROOT / "data"
    artifacts_dir: Path = _ROOT / "artifacts"
    model_dir: Path = _ROOT / "artifacts" / "models"
    report_dir: Path = _ROOT / "artifacts" / "reports"
    plot_dir: Path = _ROOT / "artifacts" / "plots"

    def ensure(self) -> None:
        """Create the writable directories if they do not yet exist."""
        for directory in (
            self.data_dir,
            self.model_dir,
            self.report_dir,
            self.plot_dir,
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Controls the dataset: synthetic generation or CSV loading."""

    source: str = "synthetic"  # "synthetic" | "csv"
    csv_path: str = ""
    target_column: str = "risk"
    n_samples: int = 4000
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    positive_rate: float = 0.35  # approximate prevalence of the positive class


@dataclass
class ModelConfig:
    """Model selection and hyper-parameters."""

    # "logistic" | "random_forest" | "gradient_boosting" | "mlp"
    name: str = "gradient_boosting"
    # Common
    class_weight_balanced: bool = True
    # Random forest / gradient boosting
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    # MLP
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [64, 32])
    max_iter: int = 300
    # Logistic
    C: float = 1.0


@dataclass
class ExplainConfig:
    """Explainability (XAI) settings."""

    n_shapley_samples: int = 200  # Monte-Carlo permutations for Shapley values
    n_permutation_repeats: int = 10  # permutation-importance repeats
    pdp_grid_resolution: int = 20  # partial-dependence grid points
    background_size: int = 100  # reference rows for the Shapley baseline
    random_state: int = 42


@dataclass
class ServerConfig:
    """FastAPI runtime settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "Healthcare Risk Prediction API"
    version: str = "1.0.0"


@dataclass
class Config:
    """Root configuration aggregating every sub-section."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def _merge(target: Any, overrides: Dict[str, Any]) -> None:
    """Recursively apply ``overrides`` onto a dataclass instance in place."""
    valid = {f.name: f for f in fields(target)}
    for key, value in overrides.items():
        if key not in valid:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, Path):
            setattr(target, key, Path(value))
        else:
            setattr(target, key, value)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Build a :class:`Config`, optionally overlaying values from a YAML file."""
    cfg = Config()
    if path is not None:
        yaml_path = Path(path)
        if yaml_path.is_file():
            with yaml_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            _merge(cfg, data)
    cfg.paths.ensure()
    return cfg


# Convenient module-level singleton -------------------------------------------
config = load_config(os.environ.get("HC_CONFIG", _ROOT / "config.yaml"))
