---
title: Churn Analytics Studio
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# Churn Analytics Studio

AI & Machine Learning Powered Customer Churn Prediction Platform.

Upload your historical customer data, train 5 production-grade ML models, and get instant churn predictions — no coding required.

## Features

- Automatic data profiling and predictive signal assessment
- 5 production models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Best model selected by: `0.45 × PR-AUC + 0.35 × Recall + 0.20 × ROC-AUC`
- Zero data leakage — all preprocessing inside sklearn Pipelines
- Predict new customers with a single CSV upload

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```
