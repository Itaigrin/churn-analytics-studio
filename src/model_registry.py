"""
Model registry — classifiers + hyperparameter search spaces.

Imbalance is handled at the classifier level (never via SMOTE):
  - sklearn models: class_weight="balanced"
  - XGBoost:        scale_pos_weight = neg_count / pos_count  (exact ratio)
  - LightGBM:       class_weight="balanced"
  - CatBoost:       auto_class_weights="Balanced"

Search spaces are kept intentionally tight (n_iter ≤ 20) so the
Balanced mode finishes in under 10 minutes.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import os as _os
from .config import RANDOM_STATE

_ON_CLOUD = _os.path.exists("/mount/src")
_N_JOBS   = 1 if _ON_CLOUD else -1


# ── Optional library imports ──────────────────────────────────────────────────

def _try_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None

def _try_lightgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        return None

def _try_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        return None


# ── Registry ──────────────────────────────────────────────────────────────────

def get_registry(n_samples: int, is_imbalanced: bool,
                 pos_count: int = 0, neg_count: int = 0) -> dict:
    """
    Build and return the full model registry.

    Parameters
    ----------
    n_samples    : training set size
    is_imbalanced: True when churn rate < 20% or > 80%
    pos_count    : number of positive (churn=1) training samples
    neg_count    : number of negative (churn=0) training samples
    """
    cw = "balanced" if is_imbalanced else None

    # Exact imbalance ratio for XGBoost (better than guessing)
    if pos_count > 0 and neg_count > 0:
        spw = round(neg_count / pos_count, 2)
    elif is_imbalanced:
        spw = 4.0   # conservative fallback (~20% churn)
    else:
        spw = 1.0

    registry: dict = {

        "Logistic Regression": {
            "needs_scale": True,
            "clf": lambda: LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE,
                class_weight=cw, solver="lbfgs", n_jobs=_N_JOBS,
            ),
            "param_dist": {
                "clf__C":       [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                "clf__penalty": ["l2"],
            },
        },

        "Decision Tree": {
            "needs_scale": False,
            "clf": lambda: DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight=cw,
            ),
            "param_dist": {
                "clf__max_depth":        list(range(2, 16)),
                "clf__min_samples_leaf": [1, 2, 4, 8, 16],
                "clf__criterion":        ["gini", "entropy"],
            },
        },

        "Random Forest": {
            "needs_scale": False,
            "clf": lambda: RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight=cw, n_jobs=_N_JOBS,
            ),
            "param_dist": {
                "clf__n_estimators":     [100, 200, 300],
                "clf__max_depth":        [4, 6, 8, 12, None],
                "clf__min_samples_leaf": [1, 2, 4, 8],
                "clf__max_features":     ["sqrt", "log2"],
            },
        },

        "ExtraTrees": {
            "needs_scale": False,
            "clf": lambda: ExtraTreesClassifier(
                random_state=RANDOM_STATE, class_weight=cw, n_jobs=_N_JOBS,
            ),
            "param_dist": {
                "clf__n_estimators":     [100, 200, 300],
                "clf__max_depth":        [4, 6, 8, 12, None],
                "clf__min_samples_leaf": [1, 2, 4, 8],
                "clf__max_features":     ["sqrt", "log2"],
            },
        },

        "GradientBoosting": {
            "needs_scale": False,
            "clf": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_dist": {
                "clf__n_estimators":  [100, 200],
                "clf__max_depth":     [2, 3, 4],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__subsample":     [0.7, 0.8, 1.0],
            },
        },

        "HistGradientBoosting": {
            "needs_scale": False,
            "clf": lambda: HistGradientBoostingClassifier(
                random_state=RANDOM_STATE, class_weight=cw,
            ),
            "param_dist": {
                "clf__max_iter":          [100, 200],
                "clf__max_depth":         [3, 4, 5, None],
                "clf__learning_rate":     [0.05, 0.1, 0.2],
                "clf__min_samples_leaf":  [20, 40, 80],
                "clf__l2_regularization": [0.0, 0.1, 1.0],
            },
        },

        "KNN": {
            "needs_scale": True,
            "clf": lambda: KNeighborsClassifier(n_jobs=-1),
            "param_dist": {
                "clf__n_neighbors": list(range(5, 31, 2)),
                "clf__metric":      ["euclidean", "manhattan"],
                "clf__weights":     ["uniform", "distance"],
            },
        },
    }

    # ── XGBoost ───────────────────────────────────────────────────────────────
    XGBClassifier = _try_xgboost()
    if XGBClassifier is not None:
        _spw = spw  # capture in closure
        registry["XGBoost"] = {
            "needs_scale": False,
            "clf": lambda: XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=_N_JOBS,
                eval_metric="logloss",
                verbosity=0,
                scale_pos_weight=_spw,
            ),
            "param_dist": {
                "clf__n_estimators":     [100, 200],
                "clf__max_depth":        [3, 4, 5, 6],
                "clf__learning_rate":    [0.05, 0.1, 0.2],
                "clf__subsample":        [0.7, 0.8, 1.0],
                "clf__colsample_bytree": [0.7, 0.8, 1.0],
                "clf__reg_alpha":        [0, 0.1, 1.0],
                "clf__reg_lambda":       [1.0, 5.0],
            },
        }

    # ── LightGBM ──────────────────────────────────────────────────────────────
    LGBMClassifier = _try_lightgbm()
    if LGBMClassifier is not None:
        registry["LightGBM"] = {
            "needs_scale": False,
            "clf": lambda: LGBMClassifier(
                random_state=RANDOM_STATE,
                n_jobs=_N_JOBS,
                verbose=-1,
                class_weight=cw,
                n_estimators=200,
            ),
            "param_dist": {
                "clf__n_estimators":     [100, 200],
                "clf__max_depth":        [3, 4, 5, 6, -1],
                "clf__learning_rate":    [0.05, 0.1, 0.2],
                "clf__num_leaves":       [31, 63, 127],
                "clf__subsample":        [0.7, 0.8, 1.0],
                "clf__colsample_bytree": [0.7, 0.8, 1.0],
                "clf__reg_alpha":        [0, 0.1, 1.0],
                "clf__reg_lambda":       [1.0, 5.0],
            },
        }

    # ── CatBoost ──────────────────────────────────────────────────────────────
    CatBoostClassifier = _try_catboost()
    if CatBoostClassifier is not None:
        _acw = "Balanced" if is_imbalanced else None
        registry["CatBoost"] = {
            "needs_scale": False,
            "clf": lambda: CatBoostClassifier(
                random_seed=RANDOM_STATE,
                verbose=0,
                auto_class_weights=_acw,
                allow_writing_files=False,
                iterations=200,
            ),
            "param_dist": {
                "clf__iterations":    [100, 200],
                "clf__depth":         [4, 5, 6, 7],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__l2_leaf_reg":   [1, 3, 5, 9],
            },
        }

    return registry
