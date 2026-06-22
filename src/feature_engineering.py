"""
Step 4 — Lightweight Feature Engineering.

Creates algebraic features (ratios, products) based solely on column-name patterns.
No statistics are computed → zero data leakage, safe to apply before the train/test split.

Returns an augmented DataFrame and the list of new column names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Column-name keyword groups for pattern matching
_TENURE   = {"tenure", "months_active", "months", "age", "seniority", "duration"}
_MONTHLY  = {"monthlycharges", "monthly_charges", "monthly_fee", "monthly_payment",
              "monthly_spend", "monthly_cost", "monthly_revenue"}
_TOTAL    = {"totalcharges", "total_charges", "total_payment", "total_spend",
              "total_cost", "total_revenue", "lifetimevalue", "lifetime_value", "clv", "ltv"}
_CONTRACT = {"contract", "plan_type", "subscription_type"}


def engineer_features(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Add algebraic features where applicable.
    Returns (augmented_df, new_feature_names).
    """
    df = df.copy()
    new_cols: list[str] = []

    # Normalise column names for matching
    col_map = {c: c.lower().replace(" ", "_").replace("-", "_") for c in numeric_cols}

    tenure_col   = _find_col(col_map, _TENURE)
    monthly_col  = _find_col(col_map, _MONTHLY)
    total_col    = _find_col(col_map, _TOTAL)

    # ── Telco / subscription domain features ─────────────────────────────────

    if monthly_col and tenure_col:
        name = "feat_monthly_x_tenure"
        df[name] = df[monthly_col] * df[tenure_col]
        new_cols.append(name)

    if total_col and monthly_col:
        denom = df[monthly_col].replace(0, np.nan)
        name  = "feat_total_per_monthly"
        df[name] = df[total_col] / denom
        new_cols.append(name)

    if total_col and tenure_col:
        denom = (df[tenure_col] + 1)          # +1 avoids div-by-zero for new customers
        name  = "feat_avg_monthly_spend"
        df[name] = df[total_col] / denom
        new_cols.append(name)

    if monthly_col and total_col and tenure_col:
        denom = df[total_col].replace(0, np.nan)
        name  = "feat_monthly_charge_ratio"
        df[name] = df[monthly_col] / denom
        new_cols.append(name)

    # ── Generic: log-transform highly-skewed numerics ─────────────────────────
    # (|skew| > 1 → apply log1p; creates a companion _log column)
    for col in numeric_cols:
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

    # Fill any NaNs in engineered columns with 0
    for c in new_cols:
        df[c] = df[c].fillna(0)

    return df, new_cols


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_col(col_map: dict[str, str], keywords: set[str]) -> str | None:
    """Return the first original column whose normalised name contains a keyword."""
    for orig, norm in col_map.items():
        norm_stripped = norm.replace("_", "")
        for kw in keywords:
            if kw in norm or kw == norm_stripped:
                return orig
    return None
