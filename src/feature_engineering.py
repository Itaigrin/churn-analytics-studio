"""
Step 4 — Feature Engineering.

The set of engineered features is decided once, on the training rows, and
stored in a FeatureSpec. After that, apply_features() is a pure per-row
calculation: a customer gets the same values whether scored alone or in a
file of a million, because no statistic of the current batch is involved.

  plan_features()               candidates, from X_train only
  select_engineered_features()  keeps candidates with signal, from X_train / y_train only
  fit_feature_spec()            both of the above, returning the frozen spec
  apply_features()              the calculation itself

Candidates are dataset-agnostic:
  • Generic pairwise ratios for top positive-numeric columns (ranked by variance)
  • Log-transforms for highly-skewed positive numerics
  • Boolean service count (feat_n_active_services) — sum of all binary flag columns
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── Generic feature generation limits ────────────────────────────────────────
_MAX_RATIO_COLS  = 5    # top-N positive numeric cols used for generic pairwise ratios
_MAX_RATIO_PAIRS = 10   # hard cap on generated ratio features
_MIN_SKEW_FOR_LOG = 1.5
_BOOL_COUNT_NAME = "feat_n_active_services"


@dataclass
class FeatureSpec:
    ratios: list[tuple[str, str, str]] = field(default_factory=list)  # (name, numerator, denominator)
    logs: list[tuple[str, str]] = field(default_factory=list)         # (name, source column)
    bool_count_cols: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        names = [r[0] for r in self.ratios] + [l[0] for l in self.logs]
        return names + ([_BOOL_COUNT_NAME] if self.bool_count_cols else [])

    def restricted_to(self, keep: list[str]) -> "FeatureSpec":
        kept = set(keep)
        return FeatureSpec(
            ratios=[r for r in self.ratios if r[0] in kept],
            logs=[l for l in self.logs if l[0] in kept],
            bool_count_cols=self.bool_count_cols if _BOOL_COUNT_NAME in kept else [],
        )


def apply_features(X: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Add the engineered columns. Each value depends only on that customer's own row."""
    new_cols: dict[str, pd.Series] = {}
    for name, numerator, denominator in spec.ratios:
        ratio = X[numerator] / X[denominator].replace(0, np.nan)
        new_cols[name] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for name, col in spec.logs:
        new_cols[name] = np.log1p(X[col].fillna(0.0).clip(lower=0.0))
    if spec.bool_count_cols:
        new_cols[_BOOL_COUNT_NAME] = X[spec.bool_count_cols].fillna(0.0).sum(axis=1)
    if not new_cols:
        return X
    return pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)


def plan_features(
    X_train: pd.DataFrame,
    numeric_cols: list[str],
    bool_cols: list[str] | None = None,
) -> FeatureSpec:
    """Decide the candidate features from the training rows. No target is used."""
    bool_cols = [c for c in (bool_cols or []) if c in X_train.columns]
    valid_num = [c for c in numeric_cols if c in X_train.columns]
    spec = FeatureSpec()

    pos_num = [c for c in valid_num
               if (X_train[c].dropna() >= 0).all() and X_train[c].dropna().std() > 0]
    pos_num = sorted(pos_num, key=lambda c: float(X_train[c].var()), reverse=True)[:_MAX_RATIO_COLS]
    seen: set[str] = set()
    for i, c1 in enumerate(pos_num):
        for c2 in pos_num[i + 1:]:
            name = f"feat_ratio_{_short(c1)}_{_short(c2)}"
            if len(spec.ratios) < _MAX_RATIO_PAIRS and name not in seen:
                spec.ratios.append((name, c1, c2))
                seen.add(name)

    for col in valid_num:
        values = X_train[col].dropna()
        if abs(float(values.skew())) > _MIN_SKEW_FOR_LOG and (values >= 0).all():
            spec.logs.append((f"feat_{col}_log", col))

    if len(bool_cols) >= 2:
        spec.bool_count_cols = bool_cols
    return spec


def fit_feature_spec(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    bool_cols: list[str] | None = None,
) -> FeatureSpec:
    """Plan the candidates, then keep those that vary and carry signal on the training rows."""
    spec = plan_features(X_train, numeric_cols, bool_cols)
    augmented = apply_features(X_train, spec)
    varying = [c for c in spec.names
               if augmented[c].nunique() >= 2 and float(augmented[c].std()) > 0]
    selected = select_engineered_features(augmented, y_train, varying)
    return spec.restricted_to(selected)


def select_engineered_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_names: list[str],
    min_mi: float = 0.003,
) -> list[str]:
    """
    Keep candidates that pass two filters, computed on X_train / y_train only:

    1. Correlation deduplication — if two candidates correlate > 0.95 with each
       other, drop the one with lower variance.
    2. Mutual-information filter — keep only candidates with MI >= min_mi.

    Returns the selected subset of candidate_names.
    """
    from sklearn.feature_selection import mutual_info_classif

    valid = [c for c in candidate_names if c in X_train.columns]
    if not valid:
        return []

    X_cand = X_train[valid].fillna(0)

    if len(valid) > 1:
        corr    = X_cand.corr().abs()
        upper   = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop: set[str] = set()
        for col in upper.columns:
            for partner in upper.index[upper[col] > 0.95]:
                to_drop.add(col if X_cand[col].var() < X_cand[partner].var() else partner)
        valid  = [c for c in valid if c not in to_drop]
        X_cand = X_train[valid].fillna(0)

    if not valid:
        return []

    mi_scores = mutual_info_classif(
        X_cand, y_train,
        discrete_features=False,
        random_state=42,
    )
    return [c for c, mi in zip(valid, mi_scores) if mi >= min_mi]


# ── Helper ────────────────────────────────────────────────────────────────────

def _short(col: str, max_len: int = 10) -> str:
    """Compact column name for use in generated feature names."""
    return col.lower().replace(" ", "").replace("-", "").replace("_", "")[:max_len]
