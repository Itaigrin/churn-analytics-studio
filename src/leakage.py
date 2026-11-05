"""
Data-leak detection, run before training.

A leak is a column that is only known once the customer has already left, such
as a cancellation reason. It produces a brilliant score in testing and a useless
model in production, so this is the one mistake good metrics cannot reveal.

Two checks, either of which flags a column:
  • its name suggests something recorded after churn
  • on its own, it predicts churn almost perfectly on customers it was not fitted on

This catches single-column leaks. Leaks spread over several columns need a person
who knows how the data was collected, which is why the UI asks for confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .config import LEAK_FEATURE_AUC, LEAK_HINTS, RANDOM_STATE

_MAX_ROWS = 20_000
_FOLDS = 3


@dataclass
class LeakFinding:
    column: str
    reason: str
    auc: float | None = None


def scan_for_leakage(X: pd.DataFrame, y: pd.Series, columns: list[str]) -> list[LeakFinding]:
    """Check the columns that would otherwise be model inputs."""
    if len(X) > _MAX_ROWS:
        y = y.groupby(y, group_keys=False).sample(frac=_MAX_ROWS / len(X), random_state=RANDOM_STATE)
        X = X.loc[y.index]

    findings = []
    for col in columns:
        auc = single_column_auc(X[col], y)
        if auc >= LEAK_FEATURE_AUC:
            findings.append(LeakFinding(
                col, f"on its own it predicts churn almost perfectly (ROC-AUC {auc:.3f})", auc))
        elif any(hint in col.lower() for hint in LEAK_HINTS):
            findings.append(LeakFinding(col, "its name suggests it is recorded after the customer left", auc))
    return findings


def single_column_auc(values: pd.Series, y: pd.Series) -> float:
    """ROC-AUC of a shallow tree on this column alone, measured out of fold."""
    if pd.api.types.is_numeric_dtype(values):
        frame = values.astype(float).to_frame()
        # Missingness itself can be the leak, so it is kept as information
        model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE)
    else:
        frame = values.astype(object).where(values.notna(), "__missing__").to_frame()
        model = Pipeline([
            ("encode", ColumnTransformer([(
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False)),
                ]),
                [values.name],
            )])),
            ("tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE)),
        ])
    folds = StratifiedKFold(n_splits=_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_predict(model, frame, np.asarray(y), cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, scores))
