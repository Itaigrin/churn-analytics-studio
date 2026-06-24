"""
Step 4 — Feature Engineering.

Two-phase approach:
  Phase 1  engineer_features()          — create candidate features (no target, no leakage)
  Phase 2  select_engineered_features() — keep only candidates with MI signal on training data

Phase 1 generates:
  • Keyword-based ratio / product features (tenure, charges, spend — broad keyword matching)
  • Generic pairwise ratios for top positive-numeric columns
  • Boolean count / indicator features
  • Categorical indicator features (autopay, contract type)
  • Log-transforms for skewed numerics
  • Near-constant candidates are removed immediately

Phase 2 (called after train/test split in the pipeline):
  • Removes near-duplicate features (pairwise correlation > 0.95)
  • Keeps only features with mutual information > threshold on training data
  • Returns the selected subset — applied to both train and test sets

Zero leakage: Phase 1 uses only X (no target).  Phase 2 uses only X_train / y_train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Keyword groups for numeric column detection ───────────────────────────────
_TENURE  = {"tenure", "months_active", "months", "age", "seniority", "duration"}
_MONTHLY = {"monthlycharges", "monthly_charges", "monthly_fee", "monthly_payment",
             "monthly_spend", "monthly_cost", "monthly_revenue"}
_TOTAL   = {"totalcharges", "total_charges", "total_payment", "total_spend",
             "total_cost", "total_revenue", "lifetimevalue", "lifetime_value", "clv", "ltv"}

# ── Keyword groups for boolean column detection ───────────────────────────────
_PROTECTION_KW = {"protection", "techsupport", "tech_support", "backup",
                  "security", "onlinesecurity", "onlinebackup", "deviceprotection"}
_STREAMING_KW  = {"streaming", "streamingtv", "streamingmovies", "stream"}

# ── Keyword groups for categorical column detection ───────────────────────────
_PAYMENT_KW  = {"payment", "paymentmethod", "pay_method", "payment_type"}
_CONTRACT_KW = {"contract", "contracttype", "contract_type", "plan_type", "subscription_type"}

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
    Create candidate engineered features.  No target information is used.

    Parameters
    ----------
    df           : DataFrame after raw_clean (booleans already 0/1)
    numeric_cols : numeric column names present in df
    cat_cols     : categorical column names (optional)
    bool_cols    : binary/boolean column names — values already 0/1 (optional)

    Returns
    -------
    (augmented_df, candidate_feature_names)
    """
    df = df.copy()
    new_cols: list[str] = []

    cat_cols  = [c for c in (cat_cols  or []) if c in df.columns]
    bool_cols = [c for c in (bool_cols or []) if c in df.columns]

    col_map = {c: c.lower().replace(" ", "_").replace("-", "_") for c in numeric_cols
               if c in df.columns}

    tenure_col  = _find_col(col_map, _TENURE)
    monthly_col = _find_col(col_map, _MONTHLY)
    total_col   = _find_col(col_map, _TOTAL)

    # ── Keyword-based ratio / product features ────────────────────────────────

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
        name = "feat_avg_monthly_spend"
        df[name] = df[total_col] / (df[tenure_col] + 1)
        new_cols.append(name)

    if monthly_col and total_col and tenure_col:
        denom = df[total_col].replace(0, np.nan)
        name  = "feat_monthly_charge_ratio"
        df[name] = df[monthly_col] / denom
        new_cols.append(name)

    # ── Generic pairwise ratios for positive numeric columns ──────────────────
    # Picks top-N numeric cols by variance; creates pairwise ratios.
    # These adapt to any dataset regardless of column names.
    pos_num = [c for c in col_map           # original name present in df
               if (df[c].dropna() >= 0).all() and df[c].dropna().std() > 0]
    if pos_num:
        # Sort by variance descending, take top N
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

    # ── Boolean count / indicator features ───────────────────────────────────

    if bool_cols:
        bool_df = df[bool_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        if len(bool_cols) >= 2:
            name = "feat_n_active_services"
            df[name] = bool_df.sum(axis=1)
            new_cols.append(name)

        prot_cols = [c for c in bool_cols
                     if any(kw in c.lower().replace("_", "") for kw in _PROTECTION_KW)]
        if prot_cols:
            name = "feat_has_protection"
            df[name] = bool_df[prot_cols].max(axis=1).clip(0, 1).astype(int)
            new_cols.append(name)

        stream_cols = [c for c in bool_cols
                       if any(kw in c.lower().replace("_", "") for kw in _STREAMING_KW)]
        if stream_cols:
            name = "feat_has_streaming"
            df[name] = bool_df[stream_cols].max(axis=1).clip(0, 1).astype(int)
            new_cols.append(name)

    # ── Categorical indicator features ────────────────────────────────────────

    if cat_cols:
        cat_norm = {c: c.lower().replace(" ", "_").replace("-", "_") for c in cat_cols}

        pay_col = next(
            (orig for orig, norm in cat_norm.items()
             if any(kw in norm for kw in _PAYMENT_KW)),
            None,
        )
        if pay_col:
            name = "feat_autopay"
            df[name] = (
                df[pay_col].astype(str).str.lower()
                .str.contains(r"automatic|auto.?pay", regex=True, na=False)
                .astype(int)
            )
            new_cols.append(name)

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

    # ── Log-transform highly-skewed positive numerics ─────────────────────────

    for col in list(col_map):
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

    # Fill NaNs in all candidate features
    for c in new_cols:
        df[c] = df[c].fillna(0)

    # Remove near-constant candidates immediately (no target needed)
    keep: list[str] = []
    for c in new_cols:
        try:
            if df[c].nunique() >= 2 and float(df[c].std()) > 0:
                keep.append(c)
            else:
                df.drop(columns=[c], inplace=True, errors="ignore")
        except Exception:
            keep.append(c)
    new_cols = keep

    return df, new_cols


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
       other, drop the one with lower variance (retains the more informative one).
    2. Mutual-information filter — keep only candidates with MI ≥ min_mi
       (mutual_info_classif from sklearn).

    Returns the selected subset of candidate_names.
    """
    from sklearn.feature_selection import mutual_info_classif

    valid = [c for c in candidate_names if c in X_train.columns]
    if not valid:
        return []

    X_cand = X_train[valid].fillna(0)

    # ── Step 1: Remove near-duplicate features by correlation ─────────────────
    if len(valid) > 1:
        try:
            corr = X_cand.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            # For each correlated pair, drop the lower-variance one
            to_drop: set[str] = set()
            for col in upper.columns:
                partners = upper.index[upper[col] > 0.95].tolist()
                for p in partners:
                    loser = col if X_cand[col].var() < X_cand[p].var() else p
                    to_drop.add(loser)
            valid = [c for c in valid if c not in to_drop]
            X_cand = X_train[valid].fillna(0)
        except Exception:
            pass  # if correlation check fails, proceed with all valid

    if not valid:
        return []

    # ── Step 2: Mutual-information filter ─────────────────────────────────────
    try:
        mi_scores = mutual_info_classif(
            X_cand, y_train,
            discrete_features=False,
            random_state=42,
        )
        selected = [c for c, mi in zip(valid, mi_scores) if mi >= min_mi]
    except Exception:
        selected = valid  # if MI fails, keep all valid features

    return selected


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_col(col_map: dict[str, str], keywords: set[str]) -> str | None:
    """Return the first original column whose normalised name contains a keyword."""
    for orig, norm in col_map.items():
        norm_stripped = norm.replace("_", "")
        for kw in keywords:
            if kw in norm or kw == norm_stripped:
                return orig
    return None


def _short(col: str, max_len: int = 10) -> str:
    """Compact column name for use in generated feature names."""
    return col.lower().replace(" ", "").replace("-", "").replace("_", "")[:max_len]
