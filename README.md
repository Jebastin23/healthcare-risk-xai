# 🩺 Clinical Risk Prediction with Explainable AI (XAI)

Predict a patient's risk from tabular clinical features **and explain every
prediction** — both globally (which factors matter across the population) and
locally (why *this* patient was flagged). Built offline-first on scikit-learn with
a from-scratch explainability layer: Monte-Carlo Shapley values, permutation
importance, and partial-dependence / ICE curves.

> Internship project • Domain: Healthcare AI + Explainability • Python 3.11+

---

## Why explainability matters here

In healthcare, an accurate black box is not enough — clinicians need to know
*why*. A model that flags a patient as high-risk must be able to say "because of
elevated glucose and blood pressure," and a deployment team must be able to audit
which factors drive the model overall. This project treats explanation as a
first-class output alongside the prediction.

> ⚠️ **Educational project.** It uses synthetic data and is **not** a medical
> device. Nothing here should be used for real clinical decisions.

## What it does

- Generates a realistic **synthetic clinical dataset** (10 features: age, BMI,
  systolic BP, glucose, cholesterol, HDL, smoker, family history, physical
  activity, resting heart rate) with a known risk function — so the whole
  pipeline runs offline and the "true" drivers are known for validating the XAI.
- Trains one of four models: `logistic`, `random_forest`, `gradient_boosting`,
  `mlp` (all scikit-learn, uniform interface).
- Evaluates with a full metric suite and **stratified k-fold cross-validation**.
- Explains predictions with **global** (permutation importance, partial
  dependence) and **local** (Monte-Carlo Shapley) methods.
- Serves everything via a CLI, a FastAPI service and a Streamlit app.

## Architecture

```
Clinical data (synthetic│CSV)
        │
        ▼
   DataLoader ─► preprocessing ─► stratified split
        │
        ▼
   RiskModel  (logistic│random_forest│gradient_boosting│mlp)
        │  predict_proba
        ├───────────────► Evaluator  (accuracy, precision, recall,
        │                             specificity, F1, ROC-AUC, Brier)
        └───────────────► Explainer
                            ├ global: permutation importance, PDP/ICE
                            └ local:  Monte-Carlo Shapley contributions
        exposed via CLI · FastAPI · Streamlit
```

## Quick start (offline, no GPU, no data needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Evaluate a model on synthetic data (prints metrics + top features)
python main.py evaluate --model gradient_boosting

# Stratified 5-fold cross-validation
python main.py crossval --model logistic --folds 5

# Explain a single patient's prediction (local Shapley contributions)
python main.py explain --model gradient_boosting --index 0

# Generate a dataset to CSV
python main.py generate --out data/patients.csv
```

## Models

| Name | Estimator |
|---|---|
| `logistic` | Logistic Regression (interpretable linear baseline) |
| `random_forest` | Random Forest |
| `gradient_boosting` | Gradient Boosting (default) |
| `mlp` | Multi-Layer Perceptron |

## Explainability methods

- **Permutation importance** (global): shuffle each feature and measure the drop
  in performance — model-agnostic and honest about what the model *uses*.
- **Partial dependence & ICE** (global/local): how the predicted risk changes as
  one feature is swept across its range, averaged (PDP) or per-instance (ICE).
- **Monte-Carlo Shapley values** (local): attribute a single prediction to each
  feature via averaged marginal contributions over random feature orderings —
  a from-scratch approximation of the game-theoretic Shapley value.

## Interfaces

| Interface | Command |
|---|---|
| Unified CLI | `python main.py {generate,train,evaluate,crossval,explain,serve}` |
| Dedicated CLIs | `python train.py` · `python evaluate.py` · `python explain.py` |
| REST API | `python main.py serve` → http://localhost:8000/docs |
| Streamlit | `streamlit run app_streamlit.py` |
| Docker | `docker compose up --build` (API :8000, Streamlit :8501) |

## API

```bash
# Predict risk for a patient (feature dict or ordered vector)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 61, "bmi": 32.5, "systolic_bp": 148, "glucose": 165, "cholesterol": 240, "hdl": 38, "smoker": 1, "family_history": 1, "physical_activity": 1, "resting_heart_rate": 88}}'

# Explain a prediction
curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d '{"features": {...}}'

curl http://localhost:8000/health
```

## Using your own data

Set in `config.yaml`:

```yaml
data:
  source: csv
  csv_path: data/your_patients.csv
  target_column: risk
```

The CSV should contain the ten feature columns plus a binary target column.

## Evaluation metrics

`accuracy`, `precision`, `recall` (sensitivity), `specificity`, `F1`, **`ROC-AUC`**
(threshold-independent ranking quality) and the **Brier score** (probability
calibration — lower is better). In a clinical setting recall/sensitivity and
calibration usually matter more than raw accuracy.

## Testing

```bash
pytest -q     # 40 offline tests: data, metrics, models, explainer, API
```

No GPU, network, or downloads required.

## Docs

`docs/PROJECT_DOCUMENTATION.md` is the full 50-section write-up; report templates
live alongside it.

## License

MIT — see `LICENSE`.
