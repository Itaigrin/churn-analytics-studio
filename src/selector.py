"""
Fast dataset-aware model pre-selection.

Analyzes dataset characteristics and returns the Top 5 most suitable models
with a short explanation for each choice.  No model is trained here — this
is purely heuristic scoring so that the training step wastes no time on
poorly-suited models.
"""

from __future__ import annotations


# ── Scoring weights ───────────────────────────────────────────────────────────

def select_models(
    registry: dict,
    n_samples: int,
    n_features: int,
    numeric_ratio: float,
    categorical_ratio: float,
    missing_rate: float,
    is_imbalanced: bool,
    mode_key: str,
    top_n: int = 5,
) -> list[tuple[str, str]]:
    """
    Score every model in the registry against dataset characteristics.
    Returns a list of (model_name, reason) sorted best-first, length ≤ top_n.

    Parameters
    ----------
    registry        : dict from model_registry.get_registry()
    n_samples       : training set row count
    n_features      : total feature count (after engineering)
    numeric_ratio   : fraction of features that are numeric
    categorical_ratio: fraction of features that are categorical
    missing_rate    : fraction of cells that are NaN (pre-imputation)
    is_imbalanced   : churn rate < 20% or > 80%
    mode_key        : "fast" | "balanced" | "best"
    top_n           : maximum models to return
    """
    large   = n_samples > 50_000
    medium  = 5_000 < n_samples <= 50_000
    small   = n_samples <= 5_000
    many_cat = categorical_ratio > 0.35
    many_num = numeric_ratio > 0.60
    high_dim = n_features > 50

    scores: dict[str, tuple[float, str]] = {}

    # ── Logistic Regression — always a fast, interpretable baseline ────────────
    if "Logistic Regression" in registry:
        score = 3.0
        reason = "Fast linear baseline; always useful as a reference point."
        if high_dim:
            score += 0.5
            reason += " Handles high-dimensional spaces well."
        scores["Logistic Regression"] = (score, reason)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if "XGBoost" in registry:
        score = 5.0
        reason = "Strong gradient boosting; excellent on numeric-heavy tabular data."
        if many_num:
            score += 1.5
            reason += " Your data is mostly numeric — XGBoost's sweet spot."
        if large:
            score += 0.5
            reason += " Scales well to large datasets with n_jobs=-1."
        if is_imbalanced:
            score += 1.0
            reason += " scale_pos_weight handles class imbalance automatically."
        scores["XGBoost"] = (score, reason)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if "LightGBM" in registry:
        score = 5.0
        reason = "Fastest gradient boosting; great all-rounder."
        if large:
            score += 2.0
            reason += " Histogram-based splits make it very fast on large datasets."
        if many_cat:
            score += 1.5
            reason += " Handles categorical features efficiently."
        if is_imbalanced:
            score += 0.5
            reason += " class_weight='balanced' applied automatically."
        scores["LightGBM"] = (score, reason)

    # ── CatBoost ──────────────────────────────────────────────────────────────
    if "CatBoost" in registry:
        score = 4.0
        reason = "Robust to categorical features and missing values."
        if many_cat:
            score += 2.5
            reason += " Your data has many categorical columns — CatBoost's strongest advantage."
        if is_imbalanced:
            score += 1.0
            reason += " auto_class_weights='Balanced' applied automatically."
        if large:
            score -= 0.5   # slightly slower than LGB on very large data
        scores["CatBoost"] = (score, reason)

    # ── HistGradientBoosting ──────────────────────────────────────────────────
    if "HistGradientBoosting" in registry:
        score = 4.0
        reason = "sklearn's native fast gradient boosting; no extra libraries needed."
        if large:
            score += 1.5
            reason += " Histogram approximation keeps it fast on large datasets."
        if missing_rate > 0.05:
            score += 1.0
            reason += " Handles missing values natively — no imputation overhead."
        if is_imbalanced:
            score += 0.5
            reason += " class_weight='balanced' applied automatically."
        scores["HistGradientBoosting"] = (score, reason)

    # ── Random Forest ─────────────────────────────────────────────────────────
    if "Random Forest" in registry:
        score = 3.5
        reason = "Reliable ensemble; low overfitting risk, good feature importances."
        if small or medium:
            score += 1.0
            reason += " Works well on small-to-medium datasets."
        if is_imbalanced:
            score += 0.5
            reason += " class_weight='balanced' applied automatically."
        scores["Random Forest"] = (score, reason)

    # ── ExtraTrees ────────────────────────────────────────────────────────────
    if "ExtraTrees" in registry:
        score = 2.5
        reason = "Faster than Random Forest; good with high-dimensional noisy data."
        if high_dim:
            score += 1.0
            reason += " Extra randomisation reduces overfitting on high-dimensional data."
        if mode_key == "best":
            score += 0.5
        scores["ExtraTrees"] = (score, reason)

    # ── KNN ───────────────────────────────────────────────────────────────────
    if "KNN" in registry:
        score = 1.0
        reason = "Instance-based learner; useful on small clean datasets."
        if large:
            score = -99.0   # effectively exclude
            reason = "Excluded: too slow on large datasets."
        elif small:
            score += 2.0
            reason += " Small dataset makes KNN practical."
        elif medium:
            score += 0.5
        if many_cat:
            score -= 1.0   # bad fit for many categoricals
        scores["KNN"] = (score, reason)

    # ── GradientBoosting (sklearn, slow) ──────────────────────────────────────
    if "GradientBoosting" in registry:
        score = 1.5
        reason = "Classic gradient boosting; accurate but slow."
        if mode_key == "best":
            score += 2.0
            reason += " Included in Best Accuracy mode for thorough search."
        elif large:
            score = -99.0
            reason = "Excluded: too slow on large datasets."
        scores["GradientBoosting"] = (score, reason)

    # ── Decision Tree ─────────────────────────────────────────────────────────
    if "Decision Tree" in registry:
        score = 1.0
        reason = "Simple single tree; rarely competitive but very fast."
        if mode_key == "fast":
            score += 0.5
            reason += " Included in Fast mode as a quick baseline."
        else:
            score = -99.0   # skip in balanced/best — RF is strictly better
        scores["Decision Tree"] = (score, reason)

    # ── Sort and take top_n ───────────────────────────────────────────────────
    ranked = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    selected = [(name, reason) for name, (s, reason) in ranked if s > 0]
    return selected[:top_n]
