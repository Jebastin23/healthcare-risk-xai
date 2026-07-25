"""Streamlit front-end for the healthcare risk project.

Run with::

    streamlit run app_streamlit.py

Trains a model on the synthetic data, lets the user adjust a patient's features
with sliders, shows the predicted risk, and displays a per-patient Shapley
explanation as a signed contribution bar chart.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from config import load_config
from src.data_loader import DataLoader
from src.explainer import Explainer
from src.trainer import Trainer


@st.cache_resource
def _prepare(model_name: str):
    """Train once and cache the model, explainer and split statistics."""
    cfg = load_config()
    cfg.model.name = model_name
    split = DataLoader(cfg.data).load()
    model = Trainer(cfg).fit(split.x_train, split.y_train)
    background = split.x_train[: cfg.explain.background_size]
    explainer = Explainer(
        model.predict_proba, background, split.feature_names, cfg.explain
    )
    return cfg, split, model, explainer


def main() -> None:
    """Render the Streamlit UI."""
    st.set_page_config(page_title="Clinical Risk + XAI", page_icon="🩺")
    st.title("🩺 Clinical Risk Prediction with Explainable AI")
    st.caption(
        "Adjust a patient's (standardised) features and see the predicted risk "
        "plus a Shapley explanation of the drivers."
    )

    model_name = st.sidebar.selectbox(
        "Model",
        ["gradient_boosting", "random_forest", "logistic", "mlp"],
    )
    cfg, split, model, explainer = _prepare(model_name)

    st.sidebar.markdown("### Patient features (z-scores)")
    values = []
    for i, name in enumerate(split.feature_names):
        col = split.x_test[:, i]
        default = float(np.median(col))
        values.append(
            st.sidebar.slider(
                name, float(col.min()), float(col.max()), default, step=0.1
            )
        )
    instance = np.asarray(values, dtype=np.float64)

    proba = float(model.predict_proba(instance[None, :])[0])
    st.metric("Predicted risk", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))

    if st.button("Explain this prediction"):
        with st.spinner("Computing Shapley values…"):
            explanation = explainer.shapley_values(instance)
        st.write(
            f"Baseline (population average): **{explanation.base_value:.1%}** — "
            f"this patient: **{explanation.prediction:.1%}**"
        )
        ranking = explanation.as_ranking()
        import pandas as pd

        frame = pd.DataFrame(ranking).set_index("feature")
        st.bar_chart(frame)
        st.caption(
            "Positive contributions raise predicted risk; negative ones lower it."
        )


if __name__ == "__main__":
    main()
