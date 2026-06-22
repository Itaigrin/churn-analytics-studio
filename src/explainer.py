"""
Step 11 — Model Explainability.

1. Built-in feature importances (tree-based models)
2. Permutation importance (model-agnostic, computed on TEST data)
3. SHAP values (on a small sample for speed)

All computations use the FITTED pipeline — no risk of data leakage.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, SHAP_SAMPLE_SIZE, PERMUTATION_SAMPLE_SIZE

warnings.filterwarnings("ignore")


# ── Feature-name extraction ───────────────────────────────────────────────────

def get_feature_names(pipe: Pipeline) -> list[str]:
    """Extract human-readable feature names from the fitted ColumnTransformer."""
    try:
        raw = pipe.named_steps["pre"].get_feature_names_out()
        names = []
        for n in raw:
            # Strip transformer prefixes: "num__tenure" → "tenure"
            n = str(n)
            for prefix in ("num__", "cat__", "bool__"):
                if n.startswith(prefix):
                    n = n[len(prefix):]
                    break
            names.append(n)
        return names
    except Exception:
        return []


# ── Built-in importances ──────────────────────────────────────────────────────

def get_builtin_importance(pipe: Pipeline) -> pd.Series | None:
    clf = pipe.named_steps.get("clf")
    if clf is None:
        return None

    feature_names = get_feature_names(pipe)

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        if len(imp) == len(feature_names):
            return pd.Series(imp, index=feature_names).sort_values(ascending=False)

    if hasattr(clf, "coef_"):
        coef = np.abs(clf.coef_[0]) if clf.coef_.ndim > 1 else np.abs(clf.coef_)
        if len(coef) == len(feature_names):
            return pd.Series(coef, index=feature_names).sort_values(ascending=False)

    return None


# ── Permutation importance ────────────────────────────────────────────────────

def get_permutation_importance(
    pipe: Pipeline,
    X_test,
    y_test,
    n_repeats: int = 5,
) -> pd.Series | None:
    try:
        # Subsample for speed
        n = min(len(X_test), PERMUTATION_SAMPLE_SIZE)
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=n, replace=False)
        X_s = X_test.iloc[idx] if hasattr(X_test, "iloc") else X_test[idx]
        y_s = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]

        result = permutation_importance(
            pipe, X_s, y_s,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=-1,
        )
        # Feature names are the INPUT features (before the preprocessor)
        if hasattr(X_s, "columns"):
            names = list(X_s.columns)
        else:
            names = [f"feature_{i}" for i in range(X_s.shape[1])]

        return pd.Series(
            result.importances_mean, index=names
        ).sort_values(ascending=False)

    except Exception:
        return None


# ── SHAP ─────────────────────────────────────────────────────────────────────

def get_shap_values(
    pipe: Pipeline,
    X_test,
    sample_size: int = SHAP_SAMPLE_SIZE,
) -> tuple[np.ndarray | None, list[str] | None, pd.DataFrame | None]:
    """
    Compute SHAP values on a small sample of the test set.
    Returns (shap_values, feature_names, X_sample_transformed).
    Returns (None, None, None) if SHAP is unavailable or fails.
    """
    try:
        import shap
    except ImportError:
        return None, None, None

    try:
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
        feature_names = get_feature_names(pipe)

        # Sample rows
        n = min(len(X_test), sample_size)
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=n, replace=False)
        X_s = X_test.iloc[idx] if hasattr(X_test, "iloc") else X_test[idx]

        # Transform only the sample (preprocessor already fitted)
        X_transformed = pre.transform(X_s)

        # Choose explainer
        clf_name = type(clf).__name__.lower()

        if any(k in clf_name for k in ["forest", "tree", "boosting", "xgb", "lgbm", "catboost", "gradient", "extra"]):
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_transformed)
            # For binary: shap_values may be a list [neg_class, pos_class]
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_vals = shap_vals[1]
        else:
            background = shap.sample(X_transformed, min(50, X_transformed.shape[0]))
            explainer  = shap.KernelExplainer(clf.predict_proba, background)
            shap_vals  = explainer.shap_values(X_transformed, nsamples=100)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

        X_sample_df = pd.DataFrame(X_transformed, columns=feature_names)
        return shap_vals, feature_names, X_sample_df

    except Exception:
        return None, None, None
