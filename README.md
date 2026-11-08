---
title: Churn Analytics Studio
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.60.0
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
- Best model selected by PR-AUC, with every model scored on one common basis. Results are reported as ranking (PR-AUC, top-10% lift, a 0–100 quality score), probability accuracy (calibration, log loss) and decision (recall and precision at the chosen threshold), with 95% ranges
- What drives churn across all customers, plus risk levels and a 0–100 risk score per customer
- Flag customers by balance, by profit or by how many you can contact, and weigh risk by each customer's value
- Results reported on customers held back from training, tuning, model choice and threshold choice
- Calibrated probabilities: among customers given 30% risk, about 30% really leave
- Screening for columns that leak the answer (single-column leaks; a person still has to review the data)
- Predict new customers with a single CSV upload. Each customer's score depends only on their own row, and files that don't match the training data are stopped
- Save a trained model and score new customers later without retraining. Files are signed, so only this installation's own files are opened
- The main reasons behind each flagged customer's risk, in your own columns
- Dates become days before a reference date (the day the data was taken, a chosen date, or each row's own snapshot date), and the model can be tested on the most recent customers. A customer with several rows is never in both training and testing
- Monitoring: every scored file is compared with the training customers, column by column and on the scores (PSI), with a clear call when the model should be retrained
- Checking past predictions against what really happened, with a clear call when the model ranks worse or its probabilities drift from the real churn rate
- The money side: enter what a customer is worth, what a contact costs and how many churners it keeps, and get the most profitable threshold, the expected net result, and how many churners the riskiest 10% hold

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

`python tools/benchmark.py <data_folder>` trains on one file and scores a separate one, to measure real performance.

Model files are signed with a private key. A private installation creates one in `~/.churn_studio/model_key` on first use, and files saved under one key can only be opened with that key. A hosted app keeps saving turned off until `CHURN_MODEL_KEY` is set in its secrets.

## Quick Start (Online)

The app is live at: **[Streamlit Community Cloud](https://churn-analytics-studio.streamlit.app)**

> ⚠️ **For companies with sensitive customer data** — use the local installation below. Data never leaves your machine.

---

## Run Locally (Recommended for Companies)

Running locally means your customer data stays entirely on your own computer.

### Requirements

- Python 3.12 or higher ([download](https://www.python.org/downloads/))
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
