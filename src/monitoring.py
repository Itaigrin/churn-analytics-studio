"""
Has the data moved away from what the model learned from?

Each column of a new file is compared with the training customers using the
Population Stability Index (PSI): the training values are cut into bins, and the
share of new customers in each bin is compared with the share of training
customers. The churn scores themselves are compared the same way.

Common reading of PSI: below 0.1 stable, 0.1 to 0.25 worth watching, above 0.25
a real shift. A shift does not prove the model is wrong, but it is the earliest
warning available before the actual outcomes are known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WATCH_PSI = 0.10
SHIFT_PSI = 0.25
MIN_CUSTOMERS = 100        # fewer than this and the shares are mostly noise
_BINS = 10
_TOP_CATEGORIES = 20
_EPSILON = 1e-4
_MISSING = "(missing)"
_OTHER = "(other)"


@dataclass
class ColumnBins:
    """How a training column is binned, and the share of training customers in each bin."""
    edges: list | None           # numeric: inner cut points
    categories: list | None      # categorical: the categories kept by name
    expected: np.ndarray

    def shares(self, values: pd.Series) -> np.ndarray:
        if self.edges is not None:
            numbers = pd.to_numeric(values, errors="coerce")
            bins = np.digitize(numbers.fillna(0), self.edges)
            bins = np.where(numbers.isna(), len(self.edges) + 1, bins)
            counts = np.bincount(bins, minlength=len(self.edges) + 2)
        else:
            labels = values.map(lambda v: _MISSING if pd.isna(v) else (v if v in self.categories else _OTHER))
            counts = labels.value_counts().reindex(self.categories + [_OTHER, _MISSING], fill_value=0).to_numpy()
        return counts / max(counts.sum(), 1)


@dataclass
class DriftReference:
    columns: dict[str, ColumnBins] = field(default_factory=dict)
    scores: ColumnBins | None = None


def build_drift_reference(X_train: pd.DataFrame, numeric_cols: list[str], train_scores: np.ndarray) -> DriftReference:
    reference = DriftReference()
    for col in X_train.columns:
        reference.columns[col] = _bins_for(X_train[col], numeric=col in numeric_cols)
    reference.scores = _bins_for(pd.Series(train_scores), numeric=True)
    return reference


def _bins_for(values: pd.Series, numeric: bool) -> ColumnBins:
    if numeric:
        filled = pd.to_numeric(values, errors="coerce").dropna()
        edges = list(np.unique(np.quantile(filled, np.linspace(0, 1, _BINS + 1)[1:-1]))) if len(filled) else []
        bins = ColumnBins(edges, None, np.array([]))
    else:
        top = list(values.dropna().value_counts().head(_TOP_CATEGORIES).index)
        bins = ColumnBins(None, top, np.array([]))
    bins.expected = bins.shares(values)
    return bins


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, _EPSILON, None)
    a = np.clip(actual, _EPSILON, None)
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass
class DriftReport:
    n_customers: int
    columns: pd.DataFrame          # column, psi, status
    score_psi: float

    @property
    def too_few(self) -> bool:
        return self.n_customers < MIN_CUSTOMERS

    @property
    def shifted(self) -> list[str]:
        return self.columns.loc[self.columns["PSI"] > SHIFT_PSI, "Column"].tolist()

    @property
    def retrain_recommended(self) -> bool:
        """The scores moved, or several inputs did: the model is out of its depth."""
        return not self.too_few and (self.score_psi > SHIFT_PSI or len(self.shifted) >= 2)


def drift_report(reference: DriftReference, X_new: pd.DataFrame, scores: np.ndarray) -> DriftReport:
    rows = []
    for col, bins in reference.columns.items():
        if col not in X_new.columns:
            continue
        value = psi(bins.expected, bins.shares(X_new[col]))
        rows.append({"Column": col, "PSI": value, "Status": _status(value)})
    table = pd.DataFrame(rows, columns=["Column", "PSI", "Status"]).sort_values("PSI", ascending=False)
    score_value = psi(reference.scores.expected, reference.scores.shares(pd.Series(scores)))
    return DriftReport(len(X_new), table.reset_index(drop=True), score_value)


def _status(value: float) -> str:
    if value > SHIFT_PSI:
        return "🔴 Shifted"
    if value > WATCH_PSI:
        return "🟡 Watch"
    return "🟢 Stable"
