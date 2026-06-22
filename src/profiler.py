"""
Step 2 — Automatic Data Profiling.

Classifies every column as: numeric / categorical / boolean / datetime / identifier.
Detects: ID columns, email, phone, UUID, hash, constant, near-constant,
         high-cardinality, and potential data-leakage columns.
No statistics are computed that could later cause leakage — only column-level properties.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    NULL_THRESHOLD, NEAR_CONSTANT_THRESHOLD, HIGH_CARDINALITY_LIMIT,
    ID_HINTS, CHURN_HINTS, LEAK_HINTS,
)

# ── Regex patterns ────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"^[\w.+\-]+@[\w\-]+\.[a-z]{2,}$", re.I)
_UUID_RE  = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
_HASH_RE  = re.compile(r"^[0-9a-f]{32,64}$", re.I)
_PHONE_RE = re.compile(r"^[\d\s\-\+\(\)\.]{7,20}$")
_BOOL_STR = {"yes", "no", "true", "false", "1", "0", "y", "n", "t", "f"}


# ── Public helpers (used by app.py) ──────────────────────────────────────────

def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristically detect the customer ID column."""
    norm = {c.lower().replace(" ", "_"): c for c in df.columns}
    for hint in ID_HINTS:
        if hint in norm:
            return norm[hint]
    # Fallback: all-unique object column
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() == len(df):
            if any(kw in col.lower() for kw in ["id", "no", "num", "code", "key"]):
                return col
    return None


def detect_target_column(df: pd.DataFrame, exclude: list[str] | None = None) -> Optional[str]:
    """Heuristically detect the churn / target column."""
    exclude = set(exclude or [])
    norm = {c.lower().replace(" ", "_"): c for c in df.columns if c not in exclude}

    # Name-based match
    for hint in CHURN_HINTS:
        if hint in norm:
            return norm[hint]

    # Value-based: binary column whose name contains a churn hint
    for col in df.columns:
        if col in exclude:
            continue
        vals = {str(v).strip().lower() for v in df[col].dropna().unique()}
        if vals <= _BOOL_STR and any(h in col.lower() for h in ["churn", "attrition", "left", "exit"]):
            return col

    # Last resort: single binary column
    candidates = [
        col for col in df.columns
        if col not in exclude
        and {str(v).strip().lower() for v in df[col].dropna().unique()} <= _BOOL_STR
    ]
    return candidates[0] if len(candidates) == 1 else None


# ── Main profiling function ───────────────────────────────────────────────────

def profile_dataset(
    df: pd.DataFrame,
    target_col: str,
    id_col: Optional[str] = None,
) -> dict:
    """
    Full dataset profile.  Returns a report dict with:
      - per-column info  (type, nulls, cardinality, flags)
      - drop_cols        {col: reason}
      - numeric_cols, categorical_cols, boolean_cols, datetime_cols
      - leakage_warnings []
      - summary stats
    """
    exclude = {c for c in [target_col, id_col] if c}
    X = df.drop(columns=list(exclude), errors="ignore")

    report = {
        "n_rows":           len(df),
        "n_cols_raw":       X.shape[1],
        "target_col":       target_col,
        "id_col":           id_col,
        "columns":          {},
        "drop_cols":        {},
        "numeric_cols":     [],
        "categorical_cols": [],
        "boolean_cols":     [],
        "datetime_cols":    [],
        "leakage_warnings": [],
        "general_warnings": [],
    }

    for col in X.columns:
        info = _profile_column(X[col])
        report["columns"][col] = info

        reason = _drop_reason(col, info)
        if reason:
            report["drop_cols"][col] = reason
            continue

        # Leakage hints
        col_low = col.lower()
        if any(h in col_low for h in LEAK_HINTS):
            report["leakage_warnings"].append(
                f"🚨 **{col}** — name suggests a post-churn event (data leakage risk)"
            )

        # Classify
        t = info["detected_type"]
        if t == "datetime":
            report["datetime_cols"].append(col)
        elif t == "boolean":
            report["boolean_cols"].append(col)
        elif t == "numeric":
            report["numeric_cols"].append(col)
        else:
            report["categorical_cols"].append(col)

        # Missing value warning
        if 0.05 < info["null_pct"] <= NULL_THRESHOLD:
            report["general_warnings"].append(
                f"⚠️ **{col}** has {info['null_pct']:.1%} missing values — will be imputed"
            )

    return report


# ── Per-column profiling ──────────────────────────────────────────────────────

def _profile_column(s: pd.Series) -> dict:
    n = len(s)
    non_null = s.dropna()
    null_count = int(s.isna().sum())

    info = {
        "dtype":            str(s.dtype),
        "null_count":       null_count,
        "null_pct":         null_count / max(n, 1),
        "n_unique":         int(s.nunique()),
        "detected_type":    None,
        "top_value":        None,
        "top_freq":         None,
        "is_constant":      False,
        "is_near_constant": False,
        "is_high_card":     False,
        "is_id_like":       False,
        "is_email":         False,
        "is_phone":         False,
        "is_uuid":          False,
        "is_hash":          False,
        "skewness":         None,
        "mean":             None,
        "std":              None,
    }

    if len(non_null) == 0:
        info["detected_type"] = "empty"
        return info

    # Top frequency
    vc = s.value_counts(normalize=True)
    if len(vc) > 0:
        info["top_value"] = str(vc.index[0])
        info["top_freq"]  = float(vc.iloc[0])

    info["is_constant"]      = info["n_unique"] <= 1
    info["is_near_constant"] = (info["top_freq"] or 0.0) >= NEAR_CONSTANT_THRESHOLD

    # ── Datetime ──────────────────────────────────────────────────────────────
    if "datetime" in str(s.dtype):
        info["detected_type"] = "datetime"
        return info

    if s.dtype == object:
        try:
            pd.to_datetime(non_null.head(100), infer_datetime_format=True, errors="raise")
            info["detected_type"] = "datetime"
            return info
        except Exception:
            pass

    # ── Numeric ───────────────────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(s):
        info["mean"] = float(non_null.mean())
        info["std"]  = float(non_null.std())
        try:
            info["skewness"] = float(non_null.skew())
        except Exception:
            pass
        unique_vals = set(non_null.unique())
        if unique_vals <= {0, 1, 0.0, 1.0, True, False}:
            info["detected_type"] = "boolean"
        else:
            info["detected_type"] = "numeric"
        return info

    # ── String checks ─────────────────────────────────────────────────────────
    sample = non_null.astype(str).head(200)
    lower_vals = {str(v).strip().lower() for v in non_null.unique()}

    # Boolean-like strings
    if lower_vals <= _BOOL_STR:
        info["detected_type"] = "boolean"
        return info

    # Email
    if sample.apply(lambda x: bool(_EMAIL_RE.match(x.strip()))).mean() > 0.80:
        info["is_email"] = True
        info["detected_type"] = "identifier"
        return info

    # UUID
    if sample.apply(lambda x: bool(_UUID_RE.match(x.strip()))).mean() > 0.80:
        info["is_uuid"] = True
        info["detected_type"] = "identifier"
        return info

    # Hash
    if sample.apply(lambda x: bool(_HASH_RE.match(x.strip()))).mean() > 0.80:
        info["is_hash"] = True
        info["detected_type"] = "identifier"
        return info

    # Phone
    if sample.apply(lambda x: bool(_PHONE_RE.match(x.strip()))).mean() > 0.80:
        info["is_phone"] = True
        info["detected_type"] = "identifier"
        return info

    # All-unique → likely an ID
    if info["n_unique"] == n:
        info["is_id_like"] = True
        info["detected_type"] = "identifier"
        return info

    # High cardinality
    if info["n_unique"] > HIGH_CARDINALITY_LIMIT or (info["n_unique"] / max(n, 1)) > 0.50:
        info["is_high_card"] = True

    info["detected_type"] = "categorical"
    return info


def _drop_reason(col: str, info: dict) -> Optional[str]:
    if info["null_pct"] > NULL_THRESHOLD:
        return f"Too many missing values ({info['null_pct']:.0%} null)"
    if info["is_constant"]:
        return "Constant column — carries no signal"
    if info["is_near_constant"]:
        return f"Near-constant ({info['top_freq']:.1%} same value — carries no signal)"
    if info["detected_type"] == "identifier":
        flags = [k for k in ("is_email","is_uuid","is_hash","is_phone","is_id_like") if info[k]]
        return f"Identifier column ({', '.join(flags) or 'all-unique values'})"
    if info["is_high_card"] and info["detected_type"] == "categorical":
        return f"High cardinality ({info['n_unique']} unique values — likely free-text or ID)"
    if info["detected_type"] == "empty":
        return "Empty column"
    return None
