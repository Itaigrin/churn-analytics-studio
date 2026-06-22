# ChurnApp Pro

Professional Customer Churn Prediction — built with Streamlit & scikit-learn.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

```
app.py                       Main Streamlit UI
src/
  config.py                  Constants, mode definitions
  data_loader.py             CSV/Excel loading with auto delimiter detection
  profiler.py                Step 2: automatic data profiling
  pipeline_builder.py        Step 3+7: raw cleaning + ColumnTransformer builder
  feature_engineering.py     Step 4: algebraic feature creation (zero leakage)
  model_registry.py          All classifiers + hyperparameter search spaces
  trainer.py                 Steps 8-9: training + RandomizedSearch / Optuna
  evaluator.py               Step 10: 7 metrics, correct best-model selection
  explainer.py               Step 11: built-in importance, permutation, SHAP
  predictor.py               Predict new customers with risk levels
  viz.py                     All Plotly charts
  utils.py                   File I/O, package auto-install, report building
```

## Key Design Decisions

| Concern | Decision |
|---|---|
| Data leakage | Train/test split **before** any imputation or encoding |
| All preprocessing | Inside sklearn `Pipeline` (fit only on train) |
| Tuning metric | **ROC-AUC** — better than accuracy for imbalanced data |
| Best-model selection | ROC-AUC + F1 for imbalanced; ROC-AUC + accuracy for balanced |
| Class imbalance | `class_weight='balanced'` on all models that support it |
| Missing values | Numeric → median; Categorical → most_frequent; inside pipeline |
| Scaling | Applied **only** for Logistic Regression and KNN (not tree-based models) |
| Boolean columns | Auto-detected Yes/No, True/False → converted to 0/1 |

## Execution Modes

| Mode | Runtime | CV | n_iter | Models |
|---|---|---|---|---|
| Fast | 2–5 min | 3 | 10 | 4 core models |
| Balanced (default) | 5–10 min | 5 | 20 | Up to 10 models |
| Best Accuracy | 20–40 min | 5 | 50 | Up to 10 models + Optuna |

## Optional Libraries

Automatically installed on first run if missing:
- `xgboost` — XGBoost classifier
- `lightgbm` — LightGBM classifier
- `catboost` — CatBoost classifier
- `optuna` — Bayesian hyperparameter optimisation
- `shap` — SHAP explainability values
