"""
Step 4 — Lightweight Feature Engineering.

Creates algebraic features (ratios, products) based solely on column-name patterns.
No statistics are computed → zero data leakage, safe to apply before the train/test split.

Returns an augmented DataFrame and the list of new column names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Keyword groups for numeric column matching ─────────────────────────────────
_TENURE   = {"tenure", "months_active", "months", "age", "seniority", "duration"}
_MONTHLY  = {"monthlycharges", "monthly_charges", "monthly_fee", "monthly_payment",
              "monthly_spend", "monthly_cost", "monthly_revenue"}
_TOTAL    = {"totalcharges", "total_charges", "total_payment", "total_spend",
              "total_cost", "total_revenue", "lifetimevalue", "lifetime_value", "clv", "ltv"}

# ── Keyword groups for boolean column matching ─────────────────────────────────
_PROTECTION_KW = {"protection", "techsupport", "tech_support", "backup",
                  "security", "onlinesecurity", "onlinebackup", "deviceprotection"}
_STREAMING_KW  = {"streaming", "streamingtv", "streamingmovies", "stream"}

# ── Keyword groups for categorical column matching ────────────────────────────
_PAYMENT_KW  = {"payment", "paymentmethod", "pay_method", "payment_type"}
_CONTRACT_KW = {"contract", "contracttype", "contract_type", "plan_type", "subscription_type"}


def engineer_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str] | None = None,
    bool_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add algebraic and indicator features where applicable.

    Parameters
    ----------
    df           : DataFrame (after raw_clean; booleans already 0/1)
    numeric_cols : list of numeric column names
    cat_cols     : list of categorical column names (optional)
    bool_cols    : list of boolean/binary column names (optional, values already 0/1)

    Returns
    -------
    (augmented_df, new_feature_names)
    """
    df = df.copy()
    new_cols: list[str] = []

    # Filter to columns that actually exist in df
    cat_cols  = [c for c in (cat_cols  or []) if c in df.columns]
    bool_cols = [c for c in (bool_cols or []) if c in df.columns]

    # Normalise numeric column names for keyword matching
    col_map = {c: c.lower().replace(" ", "_").replace("-", "_") for c in numeric_cols}

    tenure_col  = _find_col(col_map, _TENURE)
    monthly_col = _find_col(col_map, _MONTHLY)
    total_col   = _find_col(col_map, _TOTAL)

    # ── Numeric ratio / product features ─────────────────────────────────────

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
        denom = (df[tenure_col] + 1)
        name  = "feat_avg_monthly_spend"
        df[name] = df[total_col] / denom
        new_cols.append(name)

    if monthly_col and total_col and tenure_col:
        denom = df[total_col].replace(0, np.nan)
        name  = "feat_monthly_charge_ratio"
        df[name] = df[monthly_col] / denom
        new_cols.append(name)

    # ── Boolean / binary features (services, indicators) ─────────────────────

    if bool_cols:
        bool_df = df[bool_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        # Count of active services / yes-features
        if len(bool_cols) >= 2:
            name = "feat_n_active_services"
            df[name] = bool_df.sum(axis=1)
            new_cols.append(name)

        # Protection-type services
        prot_cols = [c for c in bool_cols
                     if any(kw in c.lower().replace("_", "") for kw in _PROTECTION_KW)]
        if prot_cols:
            name = "feat_has_protection"
            df[name] = bool_df[prot_cols].max(axis=1).clip(0, 1).astype(int)
            new_cols.append(name)

        # Streaming services
        stream_cols = [c for c in bool_cols
                       if any(kw in c.lower().replace("_", "") for kw in _STREAMING_KW)]
        if stream_cols:
            name = "feat_has_streaming"
            df[name] = bool_df[stream_cols].max(axis=1).clip(0, 1).astype(int)
            new_cols.append(name)

    # ── Categorical indicator features ────────────────────────────────────────

    if cat_cols:
        cat_norm = {c: c.lower().replace(" ", "_").replace("-", "_") for c in cat_cols}

        # AutoPay indicator  (payment method contains "automatic")
        pay_col = next(
            (orig for orig, norm in cat_norm.items()
             if any(kw in norm for kw in _PAYMENT_KW)),
            None,
        )
        if pay_col:
            name = "feat_autopay"
            df[name] = (
                df[pay_col].astype(str).str.lower()
                .str.contains(r"automatic|auto.?pay|bank.transfer.auto", regex=True, na=False)
                .astype(int)
            )
            new_cols.append(name)

        # Month-to-month contract indicator
        contract_col = next(
            (orig for orig, norm in cat_norm.items()
             if any(kw in norm for kw in _CONTRACT_KW)),
            None,
        )
        if contract_col:
            name = "feat_month_to_month"
            df[name] = (
                df[contract_col].astype(str).str.lower()
                .str.replace(r"[\s\-]", "", regex=True)
                .str.startswith("monthtomonth", na=False)
                .astype(int)
            )
            new_cols.append(name)

    # ── Log-transform for highly-skewed numerics ──────────────────────────────

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

    # Fill any remaining NaNs in engineered columns with 0
    for c in new_cols:
        df[c] = df[c].fillna(0)

    # ── Filter near-constant engineered features ──────────────────────────────
    keep: list[str] = []
    for c in new_cols:
        try:
            if df[c].nunique() >= 2 and df[c].std() > 0:
                keep.append(c)
            else:
                df.drop(columns=[c], inplace=True, errors="ignore")
        except Exception:
            keep.append(c)
    new_cols = keep

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
