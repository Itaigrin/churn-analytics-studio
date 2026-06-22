"""
Predictive Signal Assessment — runs on raw data before any ML.

Uses only statistical tests to measure how well each feature separates
churned vs non-churned customers. No trained models, no ROC-AUC.

Numerical features  : Cohen's d, point-biserial r, mutual information
Categorical features: Cramér's V, chi-square p-value, information gain

Output
------
{
  "overall_score"      : float 0–100,
  "potential_label"    : "Low" | "Medium" | "High",
  "n_strong"           : int,
  "n_medium"           : int,
  "n_weak"             : int,
  "feature_rows"       : list[dict],   # one per feature, sorted by signal desc
  "top_features"       : list[dict],   # top 10
  "weak_features"      : list[str],    # names only
  "explanation"        : str,
}
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Thresholds for Strong / Medium / Weak ─────────────────────────────────────
# Based on normalised signal score 0–1
_STRONG = 0.30
_MEDIUM = 0.10


# ── Per-feature statistics ────────────────────────────────────────────────────

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size between two groups."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    pooled_var = ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return 0.0
    return abs(np.mean(a) - np.mean(b)) / np.sqrt(pooled_var)


def _point_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import pointbiserialr
        r, _ = pointbiserialr(y, x)
        return abs(float(r))
    except Exception:
        return 0.0


def _mutual_info_single(x: np.ndarray, y: np.ndarray) -> float:
    """Fast mutual information for a single numerical feature via binning."""
    try:
        from sklearn.feature_selection import mutual_info_classif
        xi = x.reshape(-1, 1)
        mi = mutual_info_classif(xi, y, discrete_features=False, random_state=42, n_neighbors=3)
        return float(mi[0])
    except Exception:
        return 0.0


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    try:
        ct = pd.crosstab(x, y)
        from scipy.stats import chi2_contingency
        chi2, p, dof, _ = chi2_contingency(ct)
        n = ct.sum().sum()
        r, k = ct.shape
        phi2 = chi2 / n
        phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r_corr = r - (r - 1) ** 2 / (n - 1)
        k_corr = k - (k - 1) ** 2 / (n - 1)
        denom = min(k_corr - 1, r_corr - 1)
        return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0
    except Exception:
        return 0.0


def _chi2_p(x: pd.Series, y: pd.Series) -> float:
    try:
        from scipy.stats import chi2_contingency
        ct = pd.crosstab(x, y)
        _, p, _, _ = chi2_contingency(ct)
        return float(p)
    except Exception:
        return 1.0


def _information_gain(x: pd.Series, y: pd.Series) -> float:
    """Normalised information gain (0–1)."""
    try:
        from sklearn.feature_selection import mutual_info_classif
        # encode categories to integers
        codes = x.astype("category").cat.codes.values.reshape(-1, 1)
        mi = mutual_info_classif(codes, y.values, discrete_features=True, random_state=42)
        # normalise by entropy of target
        p = y.mean()
        target_entropy = -p * np.log2(p + 1e-9) - (1 - p) * np.log2(1 - p + 1e-9)
        return float(mi[0] / (target_entropy + 1e-9))
    except Exception:
        return 0.0


# ── Convert raw target to binary ──────────────────────────────────────────────

def _binarise_target(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if pd.api.types.is_numeric_dtype(s):
        return (s != 0).astype(int)
    low = s.astype(str).str.strip().str.lower()
    pos = {"1", "yes", "true", "y", "t", "churn", "churned", "left", "exited", "cancelled", "canceled"}
    return low.isin(pos).astype(int)


# ── Normalise raw scores to 0–1 ───────────────────────────────────────────────

def _norm_numeric(d: float, r: float, mi: float) -> float:
    """Combine Cohen's d, |r|, and MI into a single 0–1 signal score."""
    d_norm  = min(d / 2.0, 1.0)            # Cohen's d saturates at 2
    r_norm  = min(r, 1.0)
    mi_norm = min(mi / 1.0, 1.0)           # MI in nats, rarely exceeds 1 for a single feature
    return float(np.clip(0.40 * d_norm + 0.35 * r_norm + 0.25 * mi_norm, 0, 1))


def _norm_categorical(v: float, ig: float, p: float) -> float:
    """Combine Cramér's V, Information Gain, and chi2 significance."""
    v_norm  = min(v, 1.0)
    ig_norm = min(ig, 1.0)
    p_bonus = 1.0 if p < 0.001 else (0.7 if p < 0.01 else (0.4 if p < 0.05 else 0.0))
    return float(np.clip(0.45 * v_norm + 0.40 * ig_norm + 0.15 * p_bonus, 0, 1))


# ── Natural-language explanation generator ────────────────────────────────────

def _feature_label(col: str) -> str:
    """Turn a snake_case column name into a readable label."""
    return col.replace("_", " ").strip().title()


def _explain_feature(
    col: str,
    tier: str,
    ftype: str,
    mean_diff: float = 0.0,       # churn_mean − nonchurn_mean  (numerical only)
    n_categories: int = 0,        # unique values (categorical only)
) -> str:
    label = _feature_label(col)

    if tier == "Weak":
        return f"{label} has little influence on whether a customer churns."

    # ── Numerical ─────────────────────────────────────────────────────────────
    if ftype == "Numerical":
        direction = mean_diff > 0   # True = higher value → more churn

        if tier == "Strong":
            if direction:
                return (f"Customers with higher {label.lower()} are significantly "
                        f"more likely to churn.")
            else:
                return (f"Customers with higher {label.lower()} are significantly "
                        f"less likely to churn.")
        else:  # Medium
            if direction:
                return (f"Customers with higher {label.lower()} tend to churn "
                        f"somewhat more often.")
            else:
                return (f"Customers with higher {label.lower()} tend to be "
                        f"somewhat more loyal.")

    # ── Categorical ───────────────────────────────────────────────────────────
    if tier == "Strong":
        if n_categories == 2:
            return (f"The two groups in {label.lower()} show a strong difference "
                    f"in churn behavior.")
        return (f"Customers with different {label.lower()} values show a strong "
                f"difference in churn behavior.")
    else:  # Medium
        if n_categories == 2:
            return (f"The two groups in {label.lower()} show a moderate difference "
                    f"in churn behavior.")
        return (f"Customers with different {label.lower()} values show a moderate "
                f"difference in churn behavior.")


# ── Main assessment ───────────────────────────────────────────────────────────

def assess_predictive_signal(
    df: pd.DataFrame,
    target_col: str,
    id_col: str | None = None,
    max_rows: int = 20_000,
) -> dict:
    """
    Runs entirely on raw data. Returns the signal assessment dict.
    Designed to finish in < 5 seconds on typical datasets.
    """
    # Work on a sample for speed; stratify on target if possible
    if len(df) > max_rows:
        try:
            df = df.groupby(target_col, group_keys=False).apply(
                lambda g: g.sample(min(len(g), max_rows // 2), random_state=42)
            ).reset_index(drop=True)
        except Exception:
            df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    # Standardise column names (same logic as raw_clean)
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    target_std = target_col.strip().lower().replace(" ", "_").replace("-", "_")
    id_std     = id_col.strip().lower().replace(" ", "_").replace("-", "_") if id_col else None

    if target_std not in df.columns:
        return _empty_result("Target column not found.")

    y_raw = df[target_std]
    y     = _binarise_target(y_raw)
    if y.nunique() < 2:
        return _empty_result("Target column has only one class — cannot assess signal.")

    # Align index
    df = df.loc[y.index]
    y  = y.loc[df.index]

    # Drop target and ID
    drop_cols = [c for c in [target_std, id_std] if c and c in df.columns]
    features  = df.drop(columns=drop_cols, errors="ignore")

    feature_rows: list[dict] = []

    for col in features.columns:
        s     = features[col]
        valid = s.dropna()
        if len(valid) < 30:
            continue
        y_aligned = y.loc[valid.index]

        # Detect type
        is_numeric = pd.api.types.is_numeric_dtype(valid)

        # Treat boolean-like strings as categorical
        if is_numeric:
            n_unique = valid.nunique()
            if n_unique <= 2:
                is_numeric = False   # binary → categorical path

        if is_numeric:
            churn_vals    = valid[y_aligned == 1].values
            nonchurn_vals = valid[y_aligned == 0].values
            if len(churn_vals) < 5 or len(nonchurn_vals) < 5:
                continue

            d  = _cohens_d(churn_vals, nonchurn_vals)
            r  = _point_biserial_r(valid.values, y_aligned.values)
            mi = _mutual_info_single(valid.values.astype(float), y_aligned.values)

            signal    = _norm_numeric(d, r, mi)
            mean_diff = float(np.mean(churn_vals) - np.mean(nonchurn_vals))
            n_cats    = 0

        else:
            # Limit cardinality for speed
            n_cats = int(valid.nunique())
            if n_cats > 50:
                top = valid.value_counts().head(30).index
                valid     = valid[valid.isin(top)]
                y_aligned = y.loc[valid.index]
                n_cats    = 30

            v  = _cramers_v(valid, y_aligned)
            ig = _information_gain(valid, y_aligned)
            p  = _chi2_p(valid, y_aligned)

            signal    = _norm_categorical(v, ig, p)
            mean_diff = 0.0

        # Classify
        if signal >= _STRONG:
            tier = "Strong"
        elif signal >= _MEDIUM:
            tier = "Medium"
        else:
            tier = "Weak"

        ftype = "Numerical" if is_numeric else "Categorical"
        feature_rows.append({
            "feature":     col,
            "signal":      signal,
            "tier":        tier,
            "type":        ftype,
            "explanation": _explain_feature(col, tier, ftype, mean_diff, n_cats),
        })

    if not feature_rows:
        return _empty_result("No features with sufficient data for signal assessment.")

    feature_rows.sort(key=lambda r: r["signal"], reverse=True)

    n_strong = sum(1 for r in feature_rows if r["tier"] == "Strong")
    n_medium = sum(1 for r in feature_rows if r["tier"] == "Medium")
    n_weak   = sum(1 for r in feature_rows if r["tier"] == "Weak")
    n_total  = len(feature_rows)

    # Overall score: weighted mean of top-60% features to avoid drag from noise columns
    top_k = max(1, int(n_total * 0.60))
    top_signals = [r["signal"] for r in feature_rows[:top_k]]
    raw_score = float(np.mean(top_signals)) if top_signals else 0.0
    # Bonus for having many strong features
    strong_bonus = min(n_strong / max(n_total, 1) * 0.3, 0.3)
    overall_score = int(np.clip((raw_score + strong_bonus) * 100, 0, 100))

    if overall_score >= 60:
        potential = "High"
    elif overall_score >= 35:
        potential = "Medium"
    else:
        potential = "Low"

    top_features = feature_rows[:10]
    weak_features = [r["feature"] for r in feature_rows if r["tier"] == "Weak"]

    explanation = _build_explanation(
        overall_score, potential, n_strong, n_medium, n_weak, n_total, feature_rows
    )

    return {
        "overall_score": overall_score,
        "potential_label": potential,
        "n_strong": n_strong,
        "n_medium": n_medium,
        "n_weak": n_weak,
        "n_total": n_total,
        "feature_rows": feature_rows,
        "top_features": top_features,
        "weak_features": weak_features,
        "explanation": explanation,
        "error": None,
    }


def _empty_result(msg: str) -> dict:
    return {
        "overall_score": 0, "potential_label": "Unknown",
        "n_strong": 0, "n_medium": 0, "n_weak": 0, "n_total": 0,
        "feature_rows": [], "top_features": [], "weak_features": [],
        "explanation": msg, "error": msg,
    }


def _build_explanation(
    score: int, potential: str,
    n_strong: int, n_medium: int, n_weak: int, n_total: int,
    rows: list[dict],
) -> str:
    top_name = rows[0]["feature"] if rows else "N/A"
    top_tier = rows[0]["tier"]    if rows else "Weak"
    strong_pct = n_strong / max(n_total, 1) * 100

    if potential == "High":
        opening = (
            f"The dataset contains strong statistical signals for churn prediction. "
            f"{n_strong} out of {n_total} features ({strong_pct:.0f}%) show a clear difference "
            f"between customers who churned and those who stayed."
        )
        outlook = (
            "A well-tuned model should be able to learn these patterns reliably. "
            f"The most informative feature is **{top_name}**."
        )
    elif potential == "Medium":
        opening = (
            f"The dataset has moderate predictive signal. "
            f"{n_strong} features show strong separation, {n_medium} show moderate separation, "
            f"and {n_weak} contribute little to no signal."
        )
        outlook = (
            "A model can learn useful patterns here, but results will depend heavily on "
            f"feature engineering and model choice. **{top_name}** is the most informative feature."
        )
    else:
        opening = (
            f"The dataset has weak overall predictive signal. "
            f"Only {n_strong} out of {n_total} features show meaningful separation "
            "between churners and loyal customers."
        )
        outlook = (
            "This suggests that the available features may not fully capture the reasons customers leave. "
            "Consider adding behavioural, engagement, or usage features if possible. "
            f"Currently the most informative feature is **{top_name}**."
        )

    weak_note = ""
    if n_weak > n_total * 0.5:
        weak_note = (
            f" Note: {n_weak} features appear to carry almost no predictive value — "
            "they may be noise or irrelevant to churn in this dataset."
        )

    return opening + " " + outlook + weak_note
