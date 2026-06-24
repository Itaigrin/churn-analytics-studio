"""
Step 4 — Feature Engineering.

Two-phase approach:
  Phase 1  engineer_features()          — create candidate features (no target, no leakage)
  Phase 2  select_engineered_features() — keep only candidates with MI signal on training data

Phase 1 generates truly dataset-agnostic features only:
  • Generic pairwise ratios for top positive-numeric columns (ranked by variance)
  • Log-transforms for highly-skewed positive numerics
  • Boolean service count (feat_n_active_services) — sum of all binary flag columns

Phase 2 (called after train/test split in the pipeline):
  • Removes near-duplicate features (pairwise correlation > 0.95)
  • Keeps only features with mutual information > threshold on training data
  • Returns the selected subset — applied to both train and test sets

Zero leakage: Phase 1 uses only X (no target).  Phase 2 uses only X_train / y_train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Generic feature generation limits ────────────────────────────────────────
_MAX_RATIO_COLS  = 5    # top-N positive numeric cols used for generic pairwise ratios
_MAX_RATIO_PAIRS = 10   # hard cap on generated ratio features


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Candidate generation (no target, no leakage)
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str] | None = None,
    bool_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create candidate engineered features using only dataset-agnostic logic.
    No target information is used.

    Parameters
    ----------
    df           : DataFrame after raw_clean (booleans already 0/1)
    numeric_cols : numeric column names present in df
    cat_cols     : unused, kept for interface compatibility
    bool_cols    : binary/boolean column names — values already 0/1 (optional)

    Returns
    -------
    (augmented_df, candidate_feature_names)
    """
    df = df.copy()
    new_cols: list[str] = []

    bool_cols = [c for c in (bool_cols or []) if c in df.columns]
    valid_num  = [c for c in numeric_cols if c in df.columns]

    # ── Generic pairwise ratios for positive numeric columns ──────────────────
    # Selects top-N columns by variance; creates ratio features between every pair.
    # Works on any dataset regardless of column names.
    pos_num = [c for c in valid_num
               if (df[c].dropna() >= 0).all() and df[c].dropna().std() > 0]
    if pos_num:
        pos_num = sorted(pos_num, key=lambda c: float(df[c].var()), reverse=True)
        pos_num = pos_num[:_MAX_RATIO_COLS]
        pairs_created = 0
        for i, c1 in enumerate(pos_num):
            for c2 in pos_num[i + 1:]:
                if pairs_created >= _MAX_RATIO_PAIRS:
                    break
                name = f"feat_ratio_{_short(c1)}_{_short(c2)}"
                if name in new_cols:
                    continue
                denom = df[c2].replace(0, np.nan)
                df[name] = df[c1] / denom
                new_cols.append(name)
                pairs_created += 1

    # ── Log-transform highly-skewed positive numerics ─────────────────────────
    for col in valid_num:
        if col in new_cols:
            continue
        try:
            skew = float(df[col].dropna().skew())
        except Exception:
            continue
        if abs(skew) > 1.5 and (df[col].dropna() >= 0).all():
            name = f"feat_{col}_log"
            df[name] = np.log1p(df[col].fillna(0))
            new_cols.append(name)

    # ── Boolean service count ─────────────────────────────────────────────────
    # Number of active binary-flag features (e.g. subscriptions, add-ons).
    if len(bool_cols) >= 2:
        bool_df = df[bool_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        name = "feat_n_active_services"
        df[name] = bool_df.sum(axis=1)
        new_cols.append(name)

    # ── Fill NaNs and drop near-constant candidates ───────────────────────────
    for c in new_cols:
        df[c] = df[c].fillna(0)

    keep: list[str] = []
    for c in new_cols:
        try:
            if df[c].nunique() >= 2 and float(df[c].std()) > 0:
                keep.append(c)
            else:
                df.drop(columns=[c], inplace=True, errors="ignore")
        except Exception:
            keep.append(c)

    return df, keep


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Statistical selection (uses X_train + y_train only)
# ══════════════════════════════════════════════════════════════════════════════

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

    # ── Step 1: Correlation deduplication ────────────────────────────────────
    if len(valid) > 1:
        try:
            corr    = X_cand.corr().abs()
            upper   = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            to_drop: set[str] = set()
            for col in upper.columns:
                partners = upper.index[upper[col] > 0.95].tolist()
                for p in partners:
                    loser = col if X_cand[col].var() < X_cand[p].var() else p
                    to_drop.add(loser)
            valid  = [c for c in valid if c not in to_drop]
            X_cand = X_train[valid].fillna(0)
        except Exception:
            pass

    if not valid:
        return []

    # ── Step 2: Mutual-information filter ────────────────────────────────────
    try:
        mi_scores = mutual_info_classif(
            X_cand, y_train,
            discrete_features=False,
            random_state=42,
        )
        selected = [c for c, mi in zip(valid, mi_scores) if mi >= min_mi]
    except Exception:
        selected = valid

    return selected


# ── Helper ────────────────────────────────────────────────────────────────────

def _short(col: str, max_len: int = 10) -> str:
    """Compact column name for use in generated feature names."""
    return col.lower().replace(" ", "").replace("-", "").replace("_", "")[:max_len]
