"""
Step 7 — Preprocessing Pipeline Builder.

Builds an unfitted sklearn ColumnTransformer that goes inside every model pipeline.
All fitting happens on TRAINING data only → zero data leakage.

Two variants:
  build_preprocessor(scaled=False)  — for tree-based models (RF, XGBoost, etc.)
  build_preprocessor(scaled=True)   — for distance/linear models (LR, KNN, SVM)

Boolean columns are treated as numeric (0/1) after imputation.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    boolean_cols: list[str],
    scale: bool = False,
) -> ColumnTransformer:
    """
    Return an unfitted ColumnTransformer.

    numeric + boolean → median imputation (+ optional StandardScaler)
    categorical       → most-frequent imputation → OneHotEncoder(handle_unknown='ignore')
    all other cols    → dropped (remainder='drop')
    """
    transformers = []

    # Numeric (+ boolean treated as 0/1 numeric)
    num_cols = numeric_cols + boolean_cols
    if num_cols:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", skPipeline(steps), num_cols))

    # Categorical
    if categorical_cols:
        cat_pipe = skPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=0.01,   # collapse very rare categories to 'infrequent'
                    max_categories=30,    # hard cap to avoid explosion
                ),
            ),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


# ── Target conversion ─────────────────────────────────────────────────────────

def convert_target(series: pd.Series) -> pd.Series:
    """
    Convert churn column to binary int (0/1).
    Handles: Yes/No, True/False, 1/0, strings "1"/"0".
    """
    _MAP = {
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "1": 1, "0": 0,
        1: 1, 0: 0,
        True: 1, False: 0,
    }
    converted = series.map(lambda v: _MAP.get(str(v).strip().lower(), np.nan))
    n_bad = converted.isna().sum()
    if n_bad > 0:
        raise ValueError(
            f"Target column has {n_bad} unrecognised values. "
            "Expected: Yes/No, True/False, 1/0."
        )
    return converted.astype(int)


# ── Raw text normalisation (pure — no statistics) ─────────────────────────────

_BOOL_MAP = {
    "yes": 1, "no": 0, "true": 1, "false": 0,
    "y": 1, "n": 0, "t": 1, "f": 0, "1": 1, "0": 0,
}


def raw_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names and fix common data quirks.
    No statistics are computed — safe to call before the train/test split.
    """
    df = df.copy()

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Collapse "No phone service" / "No internet service" → "No"
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vals = df[col].dropna().unique()
        has_yes        = any(str(v).strip().lower() == "yes" for v in vals)
        has_no_variant = any(
            str(v).strip().lower().startswith("no ") and str(v).strip().lower() != "no"
            for v in vals
        )
        if has_yes and has_no_variant:
            df[col] = df[col].apply(
                lambda v: "No"
                if isinstance(v, str) and v.strip().lower().startswith("no ")
                else v
            )

    # Convert object columns that are really numeric (e.g. TotalCharges stored as " ")
    for col in df.select_dtypes(include=["object"]).columns:
        cleaned   = df[col].astype(str).str.strip().str.replace(",", "", regex=False)
        converted = pd.to_numeric(cleaned, errors="coerce")
        valid_ratio = converted.notna().sum() / max(len(converted), 1)
        if valid_ratio > 0.80:
            df[col] = converted

    # Convert boolean-string columns (Yes/No, True/False) → 0/1 numeric
    # This must happen AFTER numeric-string conversion to avoid conflicts.
    for col in df.select_dtypes(include=["object", "category"]).columns:
        lower_vals = {str(v).strip().lower() for v in df[col].dropna().unique()}
        if lower_vals and lower_vals <= set(_BOOL_MAP.keys()):
            df[col] = df[col].map(
                lambda v: _BOOL_MAP.get(str(v).strip().lower(), np.nan)
                if pd.notna(v) else np.nan
            )

    return df
