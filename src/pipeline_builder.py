"""
Step 7 — Preprocessing Pipeline Builder, plus the text parsing shared by
training and scoring.

Builds an unfitted sklearn ColumnTransformer that goes inside every model
pipeline, so imputation and encoding are fitted on training folds only.

Two variants:
  build_preprocessor(scaled=False)  — for tree-based models (RF, XGBoost, etc.)
  build_preprocessor(scaled=True)   — for distance/linear models (LR, KNN, SVM)

Boolean columns are treated as numeric (0/1) after imputation.
"""

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .errors import DataValidationError


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


# ── Column names ──────────────────────────────────────────────────────────────

def normalize_column_name(name) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalised names. Names that collide stop the run."""
    df = df.copy()
    names = [normalize_column_name(c) for c in df.columns]
    seen: dict[str, str] = {}
    for original, name in zip(df.columns, names):
        if name in seen:
            raise DataValidationError(
                f"Columns '{seen[name]}' and '{original}' have the same name once "
                "spaces and capital letters are ignored. Rename one of them and upload again."
            )
        seen[name] = str(original)
    df.columns = names
    return df


# ── Target conversion ─────────────────────────────────────────────────────────

_POSITIVE_LABELS = {
    "1", "yes", "y", "true", "t", "churn", "churned", "left", "exited", "attrited",
    "attrited customer", "cancelled", "canceled", "lost", "inactive",
    "כן", "נטש", "נטשו", "עזב",
}
_NEGATIVE_LABELS = {
    "0", "no", "n", "false", "f", "stayed", "retained", "existing", "existing customer",
    "active", "not churned", "no churn", "current",
    "לא", "נשאר", "לקוח פעיל", "פעיל",
}
_RETENTION_NAME_HINTS = ("retain", "retention", "stay", "active", "loyal", "renew")
_TRUE_LIKE = {"1", "yes", "y", "true", "t", "כן"}


def normalize_label(value) -> str:
    """One spelling for each label: 1, 1.0 and ' 1 ' are the same value."""
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):g}"
    text = str(value).strip().lower()
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def target_labels(series: pd.Series) -> list:
    """The distinct values of a target column, with spelling variants merged."""
    labels: dict[str, object] = {}
    for value in series.dropna().unique():
        labels.setdefault(normalize_label(value), value)
    return list(labels.values())


def suggest_positive_label(series: pd.Series, target_name: str = ""):
    """
    Guess which of two target values means "churned". The UI shows the guess for
    confirmation, because a wrong guess silently inverts every prediction.
    Returns (label, confident).
    """
    labels = target_labels(series)
    if len(labels) != 2:
        return None, False
    normalized = {normalize_label(v): v for v in labels}

    name = normalize_column_name(target_name)
    is_retention_name = any(hint in name for hint in _RETENTION_NAME_HINTS)
    known_pos = [v for k, v in normalized.items() if k in _POSITIVE_LABELS]
    known_neg = [v for k, v in normalized.items() if k in _NEGATIVE_LABELS]
    if len(known_pos) == 1 and len(known_neg) == 1:
        # In a column called "retained", 1 means the customer stayed
        if is_retention_name and normalize_label(known_pos[0]) in _TRUE_LIKE:
            return known_neg[0], True
        return known_pos[0], True
    # One recognised word is enough ("Churned" / "Gone"), a lone 1 or 0 is not (1/2 codes)
    if len(known_pos) == 1 and normalize_label(known_pos[0]) != "1":
        return known_pos[0], True
    if len(known_neg) == 1 and normalize_label(known_neg[0]) != "0":
        return next(v for v in labels if v is not known_neg[0]), True

    # Nothing recognisable: the rarer value is only a starting point
    counts = series.dropna().map(normalize_label).value_counts()
    return normalized[counts.index[-1]], False


def convert_target(series: pd.Series, positive_label=None) -> pd.Series:
    """
    Convert a churn column to 1/0, leaving missing labels as NaN.

    positive_label is the value that means churned. When it is left out it is
    suggested from the values, and only a confident suggestion is accepted.
    """
    labels = target_labels(series)
    if len(labels) > 2:
        shown = ", ".join(str(v) for v in labels[:6])
        raise DataValidationError(
            f"The target column has {len(labels)} different values ({shown}…). "
            "It needs exactly two: one for customers who left and one for customers who stayed."
        )
    if positive_label is None:
        positive_label, confident = suggest_positive_label(series, str(series.name or ""))
        if len(labels) == 2 and not confident:
            raise DataValidationError(
                f"Could not tell which of {labels[0]} / {labels[1]} means the customer left. "
                "Choose it explicitly."
            )
    positive = normalize_label(positive_label) if positive_label is not None else None
    return series.map(
        lambda v: np.nan if pd.isna(v) else float(normalize_label(v) == positive)
    ).astype(float)


# ── Parsing numbers written as text ───────────────────────────────────────────

_BLANKS = {"", "nan", "na", "n/a", "null", "none", "-", "--", "?"}
_SYMBOLS = re.compile(r"[\s $€£₪¥%]")
_PARENTHESES = re.compile(r"^\((.*)\)$")
_DOT_DECIMAL = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)?(\.\d+)?([eE][+-]?\d+)?$")
_COMMA_DECIMAL = re.compile(r"^[+-]?(\d{1,3}(\.\d{3})+|\d+)?(,\d+)?$")
_COMMA_FRACTION = re.compile(r"^[+-]?\d+,(\d{1,2}|\d{4,})$")
_COMMA_LAST = re.compile(r"\.\d+,\d+$")
_MIN_NUMERIC_SHARE = 0.90


def _strip_symbols(value: str) -> str:
    return _PARENTHESES.sub(r"-\1", _SYMBOLS.sub("", value))


def _is_blank(value) -> bool:
    return pd.isna(value) or str(value).strip().lower() in _BLANKS


def detect_decimal(series: pd.Series) -> str:
    """
    Which separator marks the fraction in this column of number-like text.

    It has to be decided for the column as a whole, because "1,234" alone could
    mean either. "12,5" or "1.234,50" anywhere in the column settles it.
    """
    for value in series.dropna().astype(str):
        text = _strip_symbols(value)
        if _COMMA_LAST.search(text) or _COMMA_FRACTION.match(text):
            return ","
    return "."


def parse_number(value, decimal: str = ".") -> float:
    """Parse one value. Anything that is not clearly a number becomes NaN."""
    if _is_blank(value):
        return np.nan
    if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
        return float(value)
    text = _strip_symbols(str(value))
    pattern, thousands = (_COMMA_DECIMAL, ".") if decimal == "," else (_DOT_DECIMAL, ",")
    if not re.search(r"\d", text) or not pattern.match(text):
        return np.nan
    return float(text.replace(thousands, "").replace(",", "."))


def parse_numeric_column(series: pd.Series, decimal: str = ".") -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.map(lambda v: parse_number(v, decimal)).astype(float)


def is_text_column(series: pd.Series) -> bool:
    return (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype))


# ── Raw text normalisation (pure — no statistics) ─────────────────────────────

BOOL_MAP = {
    "yes": 1, "no": 0, "true": 1, "false": 0,
    "y": 1, "n": 0, "t": 1, "f": 0, "1": 1, "0": 0,
    "כן": 1, "לא": 0,
}


def parse_boolean(value) -> float:
    if pd.isna(value):
        return np.nan
    return float(BOOL_MAP.get(normalize_label(value), np.nan))


def raw_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names and decide what each text column really holds.
    No statistics are computed — safe to call before the train/test split.
    """
    df = normalize_columns(df)

    for col in df.columns:
        if not is_text_column(df[col]):
            continue
        text = df[col].map(lambda v: np.nan if pd.isna(v) else str(v).strip()).astype(object)
        filled = text[text.map(lambda v: not _is_blank(v))]

        # Numbers stored as text: "1,200", "12,5", "$300", "7%", blank TotalCharges
        if len(filled) > 0:
            decimal = detect_decimal(filled)
            parsed = filled.map(lambda v: parse_number(v, decimal))
            if parsed.notna().mean() >= _MIN_NUMERIC_SHARE:
                df[col] = text.map(lambda v: parse_number(v, decimal)).astype(float)
                continue

        # Yes/No, True/False → 0/1
        lower_vals = {normalize_label(v) for v in filled}
        if lower_vals and lower_vals <= set(BOOL_MAP):
            df[col] = text.map(parse_boolean).astype(float)
            continue

        df[col] = text.where(text != "", np.nan)

    return df
