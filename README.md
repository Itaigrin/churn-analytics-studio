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

## Quick Start (Online)

The app is live at: **[HuggingFace Spaces](https://huggingface.co/spaces/Itaigrin/churn-analytics-studio)**

> ⚠️ **For companies with sensitive customer data** — use the local installation below. Data never leaves your machine.

---

## Run Locally (Recommended for Companies)

Running locally means your customer data stays entirely on your own computer.

### Requirements

- Python 3.10 or higher ([download](https://www.python.org/downloads/))
- Git ([download](https://git-scm.com/downloads)) — or download the ZIP directly from GitHub

### Step 1 — Get the code

**Option A: Git clone**
```bash
git clone https://github.com/Itaigrin/churn-analytics-studio.git
cd churn-analytics-studio
```

**Option B: Download ZIP**

Click the green **Code** button on this GitHub page → **Download ZIP** → extract the folder.

### Step 2 — Create a virtual environment (recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> First-time installation takes 2–5 minutes depending on your internet speed.

### Step 4 — Run the app

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

### Stopping the app

Press `Ctrl+C` in the terminal window.

---

## Data Privacy

When running locally:
- ✅ Your data never leaves your computer
- ✅ No internet connection required after installation
- ✅ No accounts, API keys, or external services needed
