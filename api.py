"""FastAPI service exposing risk prediction and per-patient explanation.

Endpoints
---------
* ``GET  /health``   – liveness probe.
* ``GET  /features`` – ordered feature names the model expects.
* ``POST /predict``  – predicted risk probability for a patient record.
* ``POST /explain``  – Shapley attribution explaining a single prediction.

A model is trained once at startup (on the configured synthetic data) via the
modern ``lifespan`` context manager, together with a background sample and an
:class:`~src.explainer.Explainer`. In production the startup hook would instead
load a pre-trained model artifact.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import config
from src.exception import HCException
from src.logger import get_logger

logger = get_logger(__name__)

_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Train the model and build the explainer once at startup."""
    from src.data_loader import DataLoader
    from src.explainer import Explainer
    from src.trainer import Trainer

    logger.info("API startup: preparing model and explainer.")
    split = DataLoader(config.data).load()
    model = Trainer(config).fit(split.x_train, split.y_train)
    background = split.x_train[: config.explain.background_size]
    _state["model"] = model
    _state["feature_names"] = split.feature_names
    _state["explainer"] = Explainer(
        model.predict_proba, background, split.feature_names, config.explain
    )
    _state["scaler_mean"] = split.scaler_mean
    _state["scaler_std"] = split.scaler_std
    yield
    logger.info("API shutdown.")
    _state.clear()


app = FastAPI(
    title=config.server.title, version=config.server.version, lifespan=lifespan
)


class PatientRecord(BaseModel):
    """A single patient's (already-scaled) feature vector."""

    features: List[float] = Field(
        ..., description="Feature values in the model's expected order."
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Prediction response."""

    risk_probability: float
    predicted_class: int
    threshold: float


class ExplainResponse(BaseModel):
    """Explanation response."""

    risk_probability: float
    base_value: float
    contributions: List[Dict[str, Union[str, float]]]


def _validate_features(features: List[float]) -> np.ndarray:
    expected = len(_state["feature_names"])
    if len(features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, got {len(features)}",
        )
    return np.asarray(features, dtype=np.float64)[None, :]


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness probe."""
    return {"status": "ok", "model": config.model.name}


@app.get("/features")
def features() -> Dict[str, List[str]]:
    """Return the ordered feature names the model expects."""
    return {"features": _state["feature_names"]}


@app.post("/predict", response_model=PredictResponse)
def predict(record: PatientRecord) -> PredictResponse:
    """Predict risk probability for a patient record."""
    x = _validate_features(record.features)
    proba = float(_state["model"].predict_proba(x)[0])
    return PredictResponse(
        risk_probability=proba,
        predicted_class=int(proba >= record.threshold),
        threshold=record.threshold,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(record: PatientRecord) -> ExplainResponse:
    """Return a Shapley explanation for a patient's predicted risk."""
    x = _validate_features(record.features)
    try:
        explanation = _state["explainer"].shapley_values(x[0])
    except HCException as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExplainResponse(
        risk_probability=explanation.prediction,
        base_value=explanation.base_value,
        contributions=explanation.as_ranking(),
    )
