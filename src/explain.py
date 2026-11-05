"""
Why a customer is at risk, in terms of the columns the business knows.

For each customer and each input column, the column is replaced by the range of
values seen in training, and the customer is scored again each time. The effect of
that column is the customer's risk minus the average of those scores. It answers
a question a retention team can act on: "how much of this risk comes from this
value, compared with what other customers have?"

Comparing with the whole range matters. Compared with a single typical value, a
month-to-month contract could never be a reason when most customers have one, even
when it is the strongest driver of churn in the data.

The same method works for all five models, stays in the original columns rather
than encoded or engineered ones, and depends only on the customer's own row.
It looks at one column at a time, so effects that exist only in combination are
not separated out, and effects are not additive across columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .feature_engineering import FeatureSpec, apply_features

TOP_REASONS = 3
MIN_EFFECT = 0.01       # below one percentage point it is not worth a line
MAX_CUSTOMERS = 5_000   # the riskiest customers, to keep scoring time bounded
_NUMERIC_POINTS = 7     # evenly spaced percentiles standing in for a numeric column
_TOP_CATEGORIES = 10


@dataclass
class ColumnReference:
    """The values a column takes in training, and how common each one is."""
    values: list
    weights: np.ndarray
    median: float | None = None


@dataclass
class Reason:
    column: str
    value: object
    median: float | None
    effect: float        # churn probability this value adds, compared with the training range


def build_reference(X_train: pd.DataFrame, numeric_cols: list[str]) -> dict[str, ColumnReference]:
    """Describe each column's training values, without keeping any customer's row."""
    reference = {}
    for col in X_train.columns:
        values = X_train[col].dropna()
        if values.empty:
            continue
        if col in numeric_cols:
            percentiles = (np.arange(_NUMERIC_POINTS) + 0.5) / _NUMERIC_POINTS * 100
            points = np.percentile(values.astype(float), percentiles)
            reference[col] = ColumnReference(
                list(points), np.full(_NUMERIC_POINTS, 1 / _NUMERIC_POINTS), float(values.median()))
        else:
            shares = values.value_counts(normalize=True).head(_TOP_CATEGORIES)
            reference[col] = ColumnReference(list(shares.index), (shares / shares.sum()).to_numpy())
    return reference


def column_effects(
    X: pd.DataFrame,
    reference: dict[str, ColumnReference],
    spec: FeatureSpec,
    model_columns: list[str],
    predict,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """One row per customer, one column per input: how much risk that value adds."""
    effects = {}
    for col, ref in reference.items():
        expected = np.zeros(len(X))
        for value, weight in zip(ref.values, ref.weights):
            changed = X.copy()
            changed[col] = value
            expected += weight * predict(apply_features(changed, spec)[model_columns])
        effects[col] = probabilities - expected
    return pd.DataFrame(effects, index=X.index)


def top_reasons(
    effects: pd.DataFrame, X: pd.DataFrame, reference: dict[str, ColumnReference],
) -> list[list[Reason]]:
    """The columns raising each customer's risk the most, strongest first."""
    columns = np.array(effects.columns)
    values = effects.to_numpy()
    reasons = []
    for position, row in enumerate(values):
        order = np.argsort(-row)[:TOP_REASONS]
        reasons.append([
            Reason(columns[i], X.iloc[position][columns[i]], reference[columns[i]].median, float(row[i]))
            for i in order if row[i] >= MIN_EFFECT
        ])
    return reasons


def format_reason(reason: Reason) -> str:
    compared = f" (median {_format_value(reason.median)})" if reason.median is not None else ""
    return f"{reason.column.replace('_', ' ')}: {_format_value(reason.value)}{compared}, +{reason.effect * 100:.0f} pts"


def _format_value(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):,.2f}".rstrip("0").rstrip(".")
    return str(value)
