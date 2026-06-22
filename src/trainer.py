"""
Steps 8 & 9 — Model Training + Hyperparameter Optimisation.

Flow:
  1. Build model registry (with exact imbalance weights)
  2. Train each selected model with RandomizedSearchCV (cv=5, n_iter=20)
  3. Return results dict ready for evaluate_all()

Tuning metric: ROC-AUC (best for imbalanced churn data).
All fitting is on TRAINING data only — zero data leakage.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, TUNING_METRIC, PIPELINE_CONFIG
from .model_registry import get_registry
from .pipeline_builder import build_preprocessor

warnings.filterwarnings("ignore")


def train_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    profile: dict,
    selected_models: list[str],
    n_samples: int,
    is_imbalanced: bool,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> dict:
    """
    Train the selected models with HPO.

    Returns
    -------
    results : {model_name: result_dict}
    """
    numeric_cols     = profile["numeric_cols"]
    categorical_cols = profile["categorical_cols"]
    boolean_cols     = profile["boolean_cols"]

    cv_folds = PIPELINE_CONFIG["cv_folds"]
    n_iter   = PIPELINE_CONFIG["n_iter"]

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)

    registry = get_registry(
        n_samples=n_samples,
        is_imbalanced=is_imbalanced,
        pos_count=pos_count,
        neg_count=neg_count,
    )

    available = [m for m in selected_models if m in registry]

    cv    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    total = len(available)
    results: dict = {}

    for i, name in enumerate(available):
        entry = registry[name]

        if progress_callback:
            progress_callback(name, i, total, f"cv={cv_folds}, n_iter={n_iter}")

        try:
            t0 = time.time()

            pre = build_preprocessor(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                boolean_cols=boolean_cols,
                scale=entry["needs_scale"],
            )
            clf  = entry["clf"]()
            pipe = Pipeline([("pre", pre), ("clf", clf)])

            param_dist    = entry["param_dist"]
            actual_n_iter = min(n_iter, _count_combos(param_dist))

            search = RandomizedSearchCV(
                pipe, param_dist,
                n_iter=actual_n_iter,
                cv=cv,
                scoring=TUNING_METRIC,
                n_jobs=1,
                refit=True,
                random_state=RANDOM_STATE,
                error_score=0.0,
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            cv_score  = search.best_score_

            results[name] = {
                "model":       best_pipe,
                "cv_score":    float(cv_score),
                "train_time":  time.time() - t0,
                "best_params": _extract_params(best_pipe),
            }

        except Exception as exc:
            results[name] = {"error": str(exc), "model": None}

    return results


def _count_combos(param_dist: dict) -> int:
    total = 1
    for v in param_dist.values():
        if isinstance(v, list):
            total *= len(v)
    return max(total, 1)


def _extract_params(pipe: Pipeline) -> dict:
    clf = pipe.named_steps["clf"]
    return {k: v for k, v in clf.get_params().items() if not callable(v) and v is not None}
