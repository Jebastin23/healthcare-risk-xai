# Explainable Clinical Risk Prediction with Machine Learning — Complete Project Documentation

> Internship Project • Student 4 • Domain: Healthcare AI / Explainable AI (XAI)
> Level: Intermediate → Advanced • Language: Python 3.11+

---

## 1. Project Title

**Explainable Clinical Risk Prediction with Machine Learning and Shapley-Value
Attribution.**

A tabular machine-learning system that predicts a patient's binary health-risk
from clinical features (age, BMI, blood pressure, glucose, cholesterol, smoking
status, etc.) and — crucially — *explains* each prediction. It combines several
classifiers (logistic regression, random forest, gradient boosting, MLP) with a
from-scratch, model-agnostic explainability layer: Monte-Carlo Shapley values for
per-patient attribution, permutation importance for global drivers, and
partial-dependence/ICE curves for feature-effect shapes.

## 2. Problem Statement

Risk-prediction models are increasingly used to flag patients for follow-up, yet
a bare probability is not clinically actionable or trustworthy on its own. A
clinician needs to know *why* a patient was flagged, whether the reasoning is
medically plausible, and how confident and calibrated the model is. The problem
is therefore two-fold: (1) build an accurate, well-calibrated binary risk
classifier over tabular clinical data, and (2) produce faithful, per-patient and
global explanations of its decisions so the model can be audited, trusted and
acted upon.

## 3. Background

Clinical risk scoring has a long history (e.g. Framingham for cardiovascular
risk). Modern ML can improve discrimination but at the cost of interpretability —
the "black-box" problem. **Explainable AI (XAI)** addresses this. Two families
exist: *intrinsically interpretable* models (logistic regression, shallow trees)
and *post-hoc* explainers applied to any model. The most principled post-hoc
method is based on **Shapley values** from cooperative game theory: treating each
feature as a "player", the Shapley value fairly distributes the difference between
a prediction and a baseline among the features, satisfying axioms of efficiency,
symmetry, dummy and additivity. Computing exact Shapley values is exponential in
the number of features, so practical methods approximate them — here via
Monte-Carlo sampling of feature orderings. Complementary global tools include
permutation importance (how much accuracy depends on each feature) and
partial-dependence plots (the marginal effect of a feature on predictions).

## 4. Real-world Applications

Explainable risk models appear across healthcare: cardiovascular and diabetes
risk stratification; hospital readmission and length-of-stay prediction; ICU
deterioration early-warning; sepsis and adverse-event prediction; and screening
prioritisation. Beyond medicine, the same predict-and-explain pattern is
mandatory in credit scoring, insurance underwriting and fraud detection, where
regulations increasingly require a "right to explanation". The techniques here
transfer directly to any high-stakes tabular decision system.

## 5. Objectives

Build a configurable data layer (synthetic generator with a *known* ground truth,
plus CSV loading); implement leakage-safe, stratified splitting and scaling;
provide four interchangeable classifiers behind one interface; implement clinical
-appropriate metrics (including ROC-AUC, sensitivity/specificity and calibration);
implement a **from-scratch Monte-Carlo Shapley explainer**, permutation importance
and partial-dependence curves in pure NumPy; validate that the explainer recovers
the known ground-truth drivers; and expose predict-and-explain through a CLI, REST
API and web UI with an offline test suite.

## 6. Expected Outcomes

A runnable system that trains a calibrated risk classifier on tabular data,
reports discrimination and calibration metrics, and produces both global feature
rankings and per-patient Shapley explanations that agree with the data-generating
ground truth. It serves `/predict` and `/explain` over HTTP, ships a Streamlit
"what-if" explorer, and passes an automated test suite that verifies the Shapley
efficiency axiom and closed-form correctness on a linear oracle.

## 7. Learning Outcomes

The student will learn: tabular ML workflow (splitting, scaling, class imbalance,
calibration); the strengths and trade-offs of logistic regression, random forests,
gradient boosting and MLPs; clinical evaluation (why AUC, sensitivity and
calibration matter more than accuracy); the theory of Shapley values and their
axioms; how to *implement* a Monte-Carlo Shapley approximation from scratch;
permutation importance and partial-dependence analysis; and how to validate that
an explanation method is faithful using data with a known ground truth.

## 8. Technology Stack

Python 3.11+; NumPy and pandas for data; scikit-learn for the classifiers; a
custom NumPy XAI module (no dependency on the `shap` library, though it can be
enabled optionally); matplotlib (headless Agg) for plots; FastAPI, Uvicorn and
Pydantic v2 for the API; Streamlit for the UI; pytest and httpx for testing;
flake8, black and mypy for quality; Docker and docker-compose for deployment.

## 9. Libraries Required

Core: `numpy`, `PyYAML`, `pandas`, `scikit-learn`. Visualisation: `matplotlib`.
API/UI: `fastapi`, `uvicorn`, `pydantic`, `streamlit`. Testing/quality: `pytest`,
`httpx`, `flake8`, `black`, `mypy`. Optional: `shap` (a reference implementation
to compare against; the project ships its own Shapley code so this is not
required).

## 10. Folder Structure

```
healthcare-risk-xai/
├── config.py                 # Typed dataclass config + YAML loader
├── config.yaml               # Example overrides
├── main.py                   # Unified CLI (generate/train/evaluate/crossval/explain/serve)
├── train.py · evaluate.py · explain.py   # Thin dedicated CLIs
├── api.py                    # FastAPI service (/predict, /explain; lifespan)
├── app_streamlit.py          # Streamlit "what-if" explorer
├── src/
│   ├── logger.py · exception.py · utils.py
│   ├── data_generator.py     # Synthetic clinical data with known ground truth
│   ├── data_loader.py        # Stratified split + train-only scaling; CSV support
│   ├── metrics.py            # Accuracy, precision/recall, specificity, F1, ROC-AUC, Brier
│   ├── models.py             # Uniform wrapper over 4 sklearn classifiers
│   ├── trainer.py            # Fit + stratified k-fold cross-validation
│   ├── explainer.py          # ⭐ Monte-Carlo Shapley + permutation imp. + PDP/ICE
│   ├── evaluator.py          # Metrics + global-importance report
│   └── visualizer.py         # ROC, confusion, importance, Shapley, PDP plots
├── tests/                    # Offline pytest suite (31 tests)
├── data/ · docs/
├── requirements.txt · Dockerfile · docker-compose.yml
├── pytest.ini · setup.cfg · .gitignore · LICENSE · README.md
```

## 11. Complete Architecture Diagram (ASCII)

```
   Clinical data (synthetic│CSV)
            │
            ▼
   ┌──────────────────┐  stratified split + train-only scaling
   │    DataLoader    │────────────► train / val / test
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐   logistic · random forest · gradient boosting · MLP
   │     RiskModel    │   (uniform fit / predict_proba interface)
   └────────┬─────────┘
            │ predict_proba
     ┌──────┴───────────────────────────┐
     ▼                                   ▼
┌──────────┐                    ┌──────────────────────────┐
│ metrics  │                    │        Explainer         │  model-agnostic
│ AUC/F1/  │                    │  ┌────────────────────┐  │
│ Brier…   │                    │  │ Monte-Carlo Shapley│  │  local, per-patient
└──────────┘                    │  ├────────────────────┤  │
                                │  │ permutation import.│  │  global drivers
                                │  ├────────────────────┤  │
                                │  │ partial dependence │  │  feature-effect shape
                                │  └────────────────────┘  │
                                └──────────────────────────┘
              exposed via CLI · FastAPI (/predict, /explain) · Streamlit
```

## 12. Workflow Diagram

```
TRAIN:    load ─► stratified split ─► scale(train-only) ─► fit ─► save
EVALUATE: fit ─► predict_proba(test) ─► metrics + permutation importance ─► JSON
CROSSVAL: stratified k-fold ─► per-fold metrics ─► mean ± std
EXPLAIN:  fit ─► pick patient ─► Monte-Carlo Shapley ─► signed contributions
SERVE:    POST /predict {features} ─► risk;  POST /explain ─► attribution
```

## 13. Dataset

The default dataset is **synthetic**, produced by `data_generator.py`: ten
clinically plausible, mutually-correlated features and a binary risk label drawn
from a *known* logistic ground truth (glucose, blood pressure, BMI, age, smoking
and family history raise risk; HDL and physical activity lower it). Because the
true drivers are known, we can *validate* that the explainer recovers them — a
rare and valuable property. The prevalence is calibrated to a configurable rate
(default 35%). To use **real data**, set `data.source: csv` and point
`data.csv_path` at a file with a target column; public options for
experimentation include the **UCI Heart Disease**, **Pima Indians Diabetes**, and
**UCI Breast Cancer Wisconsin** datasets (all small, tabular, binary).

## 14. Data Preprocessing

Preprocessing enforces two disciplines. Splitting is **stratified** on the label
so class balance is preserved across train/validation/test. Standardisation is
**fit on the training split only** and then applied to validation/test, avoiding
leakage of test statistics. CSV inputs are reduced to numeric columns and
mean-imputed for missing values. Class imbalance is handled at the model level via
balanced class weights where supported.

## 15. Feature Engineering

The synthetic features are already clinically meaningful, so the emphasis is on
correct scaling and on *interpretation* rather than transformation. For real data,
sensible engineering includes deriving BMI from height/weight, pulse pressure from
systolic/diastolic BP, ratios such as total/HDL cholesterol, and binned age
groups. The XAI layer then operates on whatever feature set is used, attributing
predictions to those engineered features — which is exactly where interpretability
becomes clinically useful.

## 16. Model Selection

Four models span the interpretability/accuracy spectrum: **logistic regression**
(intrinsically interpretable linear baseline, good calibration), **random forest**
(non-linear, robust, provides native importances), **gradient boosting** (often
the strongest on tabular data), and an **MLP** (a small neural network). All are
wrapped in a uniform `RiskModel` interface exposing `predict_proba`, so the
explainer and API treat them identically. The guiding principle: pick the simplest
model that meets the accuracy/calibration bar, since interpretability and
deployment are easier the simpler the model.

## 17. Training Pipeline

`Trainer` fits the chosen estimator on the scaled training split and can run
**stratified k-fold cross-validation** to estimate generalisation with mean ± std
across folds before a final fit. Seeding is centralised for reproducibility. Model
artifacts are pickled to `artifacts/models/`. The pipeline is intentionally light
because the estimators handle optimisation internally; the engineering value is in
consistent splitting, evaluation and explanation.

## 18. Evaluation Metrics

Clinical evaluation goes beyond accuracy. Implemented in pure NumPy:
**accuracy**, **precision** (PPV), **recall/sensitivity**, **specificity**, **F1**,
**ROC-AUC** (rank-based, threshold-free discrimination) and the **Brier score**
(a calibration/accuracy measure on probabilities). In screening, sensitivity
(catching true positives) and calibration are often prioritised over raw accuracy,
and the decision threshold is chosen to reflect the clinical cost of false
negatives vs false positives.

## 19. Testing Methodology

The 31-test suite runs fully offline. `test_metrics.py` checks each metric on
known inputs (perfect predictions, tie handling in AUC, confusion counts).
`test_data.py` checks the generator (shapes, reproducibility, calibrated
prevalence) and the loader (stratification, train-only scaling). `test_explainer.py`
is the centrepiece: it verifies the **Shapley efficiency axiom** (contributions
sum to prediction − baseline) and **closed-form correctness** against a linear
oracle where Shapley values are known exactly, plus permutation-importance signal
detection and PDP monotonicity. `test_api.py` uses FastAPI's TestClient to exercise
`/health`, `/features`, `/predict` and `/explain` with validation.

## 20. Deployment Method

Three surfaces: a **FastAPI** service (`/health`, `/features`, `/predict`,
`/explain`) that trains a model once at startup via the modern `lifespan` context
manager (in production it would load a pre-trained artifact); a **Streamlit**
"what-if" explorer where a clinician adjusts feature sliders and sees the risk and
its Shapley explanation update; and **Docker** + docker-compose running the API on
8000 and Streamlit on 8501 with a shared artifacts volume.

## 21. Future Enhancements

Add probability **calibration** (Platt scaling, isotonic regression) and
reliability diagrams; **threshold optimisation** by clinical cost; **counterfactual
explanations** ("what change would flip this prediction?"); **global surrogate**
models; fairness/subgroup auditing across sex, age and ethnicity; SHAP-library
cross-validation of the from-scratch values; and support for missing-data-aware
models.

## 22. Research Extensions

Compare the from-scratch Monte-Carlo Shapley against KernelSHAP and TreeSHAP for
accuracy and cost; study explanation stability under resampling; evaluate
faithfulness with feature-ablation curves; investigate calibration under dataset
shift; and quantify fairness–accuracy trade-offs. Any of these can support a
workshop or journal paper (see Section 44).

## 23. GitHub README

The repository ships a `README.md` with the problem framing, an architecture
diagram, an offline quick-start (evaluate → explain), the model list, API examples
for `/predict` and `/explain`, and testing instructions — designed to get a
newcomer to a first explained prediction in minutes without any dataset or GPU.

## 24. Installation Guide

```bash
git clone <your-repo-url> healthcare-risk-xai
cd healthcare-risk-xai
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pytest -q                     # 31 offline tests
```

Everything runs on CPU with no downloads: the synthetic generator provides data
and scikit-learn provides the models.

## 25. Requirements.txt

Core: `numpy`, `PyYAML`, `pandas`, `scikit-learn`. Visualisation: `matplotlib`.
API/UI: `fastapi`, `uvicorn`, `pydantic`, `streamlit`. Testing/quality: `pytest`,
`httpx`, `flake8`, `black`, `mypy`. Optional: `shap`. See the file for exact
version constraints.

## 26. Environment Setup

Python 3.11+ in a virtual environment is recommended. No API keys or GPU are
needed. For real data set `data.source`, `data.csv_path` and `data.target_column`
in `config.yaml`. Artifacts (pickled models, JSON reports, plots) are written under
`artifacts/`, which is git-ignored and Docker-volume-mounted.

## 27. Virtual Environment Commands

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
deactivate
pip freeze > requirements.lock.txt
```

## 28. How to Run

```bash
# Generate a synthetic dataset to CSV
python main.py generate --out data/patients.csv

# Evaluate a model (metrics + permutation importance), optionally with plots
python main.py evaluate --model gradient_boosting --plot

# Stratified k-fold cross-validation
python main.py crossval --model logistic --folds 5

# Explain a single patient's prediction with Shapley values
python main.py explain --index 0 --plot

# Serve the REST API
python main.py serve --port 8000
#   POST /predict  {"features": [...]}
#   POST /explain  {"features": [...]}

# Web UI
streamlit run app_streamlit.py

# Docker
docker compose up --build          # API :8000, Streamlit :8501
```

## 29. Screenshots Required

Capture: (1) CLI `evaluate` output with all metrics and the permutation-importance
ranking; (2) the ROC curve and confusion-matrix plots; (3) CLI `explain` output
with signed contributions; (4) the Shapley contribution bar chart; (5) a
partial-dependence plot; (6) the Swagger UI at `/docs`; (7) an `/explain` response
in Swagger; (8) the Streamlit what-if explorer; (9) the passing pytest run.

## 30. Presentation Outline

Title & problem (accuracy is not enough in healthcare); the black-box problem and
why XAI matters; data and the known ground truth; models and clinical metrics;
Shapley values from game theory (axioms, intuition); the Monte-Carlo
approximation; global vs local explanations; validating the explainer against
ground truth; live demo; results; deployment and ethics; future work.

## 31. Demo Script

"Here's a gradient-boosting risk model — AUC around 0.85. But accuracy alone isn't
enough in medicine, so let's explain a specific patient. This patient's predicted
risk is 22%, below the 35% population baseline. The Shapley chart shows *why*: low
glucose and high physical activity pull the risk down, while a slightly elevated
BMI pushes it up. Notice the explanation is faithful — the global importance
ranking recovers exactly the features I baked into the data generator, which is how
I know the explainer is trustworthy, not just plausible-looking."

## 32. Viva Questions (20)

1. Why isn't accuracy sufficient for a clinical model? 2. What is ROC-AUC and why
is it threshold-free? 3. Sensitivity vs specificity — which matters more for
screening and why? 4. What is calibration and the Brier score? 5. Why stratify the
train/test split? 6. Why fit the scaler on training data only? 7. What is a Shapley
value? 8. State the axioms Shapley values satisfy. 9. Why is exact Shapley
computation intractable? 10. How does the Monte-Carlo approximation work? 11. What
is the efficiency axiom and how did you test it? 12. Local vs global explanations —
give an example of each here. 13. What is permutation importance? 14. What is a
partial-dependence plot, and its main limitation? 15. How did you validate the
explainer is faithful? 16. Why compare against a linear oracle in tests? 17. How do
you handle class imbalance? 18. What are the risks of trusting explanations
blindly? 19. How would you add counterfactual explanations? 20. What ethical issues
arise in clinical risk models?

## 33. Interview Questions (20)

1. Compare SHAP, LIME and permutation importance. 2. What is KernelSHAP vs
TreeSHAP? 3. Why can correlated features distort importance? 4. What is the
difference between interpretability and explainability? 5. How does gradient
boosting work? 6. Random forest vs gradient boosting trade-offs. 7. How do you
calibrate a classifier? 8. What is isotonic vs Platt scaling? 9. How do you choose
a decision threshold from cost? 10. What is data leakage and how do you prevent it?
11. How do you evaluate under class imbalance (PR-AUC, etc.)? 12. What is
concept/dataset shift? 13. How do you audit a model for fairness? 14. What are
counterfactual explanations? 15. Why might PDP mislead with interactions (and what
is ALE)? 16. How do you explain a deep model? 17. What regulations mandate
explanations? 18. How would you productionise this? 19. How do you monitor a
deployed risk model? 20. What are the failure modes of Shapley approximations?

## 34. Possible Errors and Solutions

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: config` | Script dir on `sys.path` | Run from project root or `PYTHONPATH=.` |
| Shapley values don't sum to pred − base | Too few Monte-Carlo samples | Increase `explain.n_shapley_samples` |
| Explanation looks implausible | Correlated features / weak model | Improve the model; interpret with care |
| Slow `/explain` | Large background × many samples | Reduce `background_size` or samples |
| `422` on `/predict` | Wrong number of features sent | Send exactly `len(/features)` values |
| Poor AUC | Under-tuned model / hard data | Try gradient boosting; tune depth/estimators |
| Over-optimistic metrics | Scaling fit on full data | Fit scaler on train only (already enforced) |
| Native importance is None | Model has no `feature_importances_`/`coef_` | Use permutation importance instead |

## 35. Project Timeline (12 Weeks)

Week 1: XAI and clinical-ML theory, setup. Week 2: data generator + loader/split.
Week 3: metrics + tests. Week 4: model wrapper + training/CV. Weeks 5–6: implement
Monte-Carlo Shapley + validate against ground truth. Week 7: permutation importance
+ PDP/ICE. Week 8: evaluator + visualisations. Week 9: API + Streamlit. Week 10:
calibration/threshold extension. Week 11: Docker, docs, polish. Week 12: report,
presentation, research extension.

## 36. Mentor Evaluation Rubric

| Criterion | Weight | Excellent (9–10) |
|---|---|---|
| ML correctness & leakage-safety | 20% | Stratified split, train-only scaling, calibrated |
| XAI implementation & validity | 25% | Correct Shapley (axioms tested), recovers ground truth |
| Clinical evaluation | 15% | AUC/sensitivity/calibration reported and interpreted |
| Code quality (typing, tests) | 20% | 30+ passing tests, typed, modular |
| Deployment & docs | 10% | API + Docker + clear README |
| Communication & ethics | 10% | Clear report; discusses limitations and fairness |

## 37. Weekly Progress Report Template

See `docs/WEEKLY_REPORT_TEMPLATE.md`.

## 38. Final Report Template

See `docs/FINAL_REPORT_TEMPLATE.md`.

## 39. IEEE Paper Writing Guidance

Target 6–8 pages, two-column IEEE. Abstract stating the predict-and-explain
problem, method and headline AUC plus an explanation-faithfulness result.
Introduction on the black-box problem in healthcare. Related work on XAI (SHAP,
LIME), Shapley values and clinical risk models. Methodology formalising the
Monte-Carlo Shapley estimator and the metrics. Experiments on a public dataset with
model comparison and an explanation-faithfulness study (recovery of known drivers,
ablation curves). Results with importance rankings and calibration diagrams.
Cite Lundberg & Lee (SHAP, 2017), Ribeiro et al. (LIME, 2016), Štrumbelj &
Kononenko (2014) and Shapley (1953).

## 40. Resume Description

"Built an explainable clinical risk-prediction system in Python: stratified,
leakage-safe pipeline with four scikit-learn classifiers behind one interface,
clinical metrics (AUC, sensitivity, Brier) and a **from-scratch Monte-Carlo
Shapley** explainer plus permutation importance and partial-dependence analysis;
validated explanation faithfulness against a known ground truth; served via FastAPI
(/predict, /explain) and Streamlit, 31 tests, Dockerised, runs fully offline."

## 41. LinkedIn Project Description

"I built a clinical risk model that doesn't just predict — it explains. Every
prediction comes with a Shapley-value breakdown of which factors pushed a patient's
risk up or down, and I validated that the explanations recover the true drivers in
the data. It implements Shapley values from scratch (no black-box library), reports
clinical metrics like sensitivity and calibration, and ships a REST API and a
what-if explorer. A hands-on tour of Explainable AI done rigorously."

## 42. GitHub Description

"Explainable clinical risk prediction — 4 classifiers, clinical metrics
(AUC/sensitivity/Brier), and a from-scratch Monte-Carlo Shapley explainer +
permutation importance + partial dependence. FastAPI /predict & /explain,
Streamlit what-if UI, Docker, offline-runnable, fully tested."

## 43. Portfolio Description

Lead with a Shapley explanation chart for a single patient and the global
importance ranking, then the architecture diagram and the key insight — that the
explainer recovers the ground-truth drivers, proving faithfulness. Emphasise both
the XAI depth (Shapley from scratch, axioms tested) and the engineering (typed,
tested, containerised, served). This predict-and-explain pattern is exactly what
regulated industries demand.

## 44. Publication Possibility

Moderate. A clean implementation is portfolio-grade; novelty comes from a rigorous
faithfulness study. Benchmarking the from-scratch Shapley against KernelSHAP/
TreeSHAP with explanation-stability and ablation-faithfulness metrics on a public
clinical dataset, plus a fairness audit, can support a workshop paper; a novel
approximation or a clinical validation study is competitive at applied venues.

## 45. Innovation Score

**7 / 10.** Risk prediction is standard, but implementing Shapley values from
scratch, validating faithfulness against a known ground truth, and integrating
local + global explanation into a served system is a substantive, distinctive
piece of work. Adding calibration, counterfactuals and a fairness audit would push
toward 8–9.

## 46. Industry Relevance

**Very high.** Explainability is now a hard requirement in healthcare, finance and
insurance. Fluency with Shapley values, permutation importance, calibration and
the predict-and-explain deployment pattern maps directly to responsible-AI and
applied-ML roles.

## 47. Estimated Difficulty

**Intermediate → Advanced.** The modelling is approachable, but implementing a
correct Monte-Carlo Shapley estimator, testing its axioms, and reasoning about
faithfulness and calibration push firmly into advanced territory.

## 48. Estimated Completion Time

**8–12 weeks** part-time including documentation, an XAI faithfulness study and one
extension. Offline defaults keep development cost-free.

## 49. Hardware Requirements

No GPU required. All models train in seconds to a minute on any laptop; the
Monte-Carlo Shapley explainer is the main compute cost and is easily tuned via the
sample count. Minimum 8 GB RAM; 16 GB comfortable.

## 50. Software Requirements

Python 3.11+ on Linux, macOS or Windows. Core packages: numpy, PyYAML, pandas,
scikit-learn. Visualisation adds matplotlib. API/UI adds fastapi, uvicorn,
pydantic, streamlit. Testing adds pytest, httpx. Docker (with the Compose plugin)
is needed only for containerised deployment. Quality tooling (flake8, black, mypy)
is optional but recommended.

---

*End of documentation. See `README.md` for the quick start and the report
templates in `docs/` for reporting.*
