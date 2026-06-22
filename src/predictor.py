"""
Predict new customers using the best fitted pipeline + optimal threshold.

Returns a DataFrame with:
  customer_id, churn_prediction, churn_probability, risk_level
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

RISK_HIGH   = 0.70
RISK_MEDIUM = 0.40


def predict_new_customers(
    pipe: Pipeline,
    X_new: pd.DataFrame,
    ids: pd.Series,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Apply the fitted pipeline + optimal threshold to new customers.
    The pipeline handles all preprocessing internally — zero leakage risk.
    """
    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_new)[:, 1]
    else:
        y_prob = pipe.predict(X_new).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    return pd.DataFrame({
        "customer_id":       ids.values,
        "churn_prediction":  ["Yes" if p == 1 else "No" for p in y_pred],
        "churn_probability": np.round(y_prob, 4),
        "risk_level":        [_risk(p) for p in y_prob],
    })


def _risk(prob: float) -> str:
    if prob >= RISK_HIGH:
        return "🔴 High"
    if prob >= RISK_MEDIUM:
        return "🟡 Medium"
    return "🟢 Low"
