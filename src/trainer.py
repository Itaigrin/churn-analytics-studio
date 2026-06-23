"""
Steps 8 & 9 — Optuna HPO with 5-fold Stratified CV + Early Stopping.

Per-model strategy:
  Logistic Regression : Optuna  →  cross_val_score (Pipeline)
  Random Forest       : Optuna  →  cross_val_score (Pipeline)
  XGBoost             : Optuna  →  manual 5-fold, early_stopping_rounds=30
  LightGBM            : Optuna  →  manual 5-fold, early_stopping_rounds=30
  CatBoost            : Optuna  →  manual 5-fold, early_stopping_rounds=30

All preprocessing is fitted INSIDE each fold → zero data leakage.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Optional

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from .config import RANDOM_STATE, PIPELINE_CONFIG
from .pipeline_builder import build_preprocessor

warnings.filterwarnings("ignore")

_CV_FOLDS       = 5
_EARLY_STOP_RDS = 30
_MAX_ITERS      = 500
_BOOSTING       = {"XGBoost", "LightGBM", "CatBoost"}


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

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
    n_trials = PIPELINE_CONFIG.get("optuna_trials", 30)
    if n_trials <= 0:
        n_trials = 30
    if n_samples > 100_000:
        n_trials = max(10, n_trials // 3)
    elif n_samples > 50_000:
        n_trials = max(15, n_trials // 2)

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    spw = round(neg_count / pos_count, 2) if pos_count > 0 else 1.0
    acw = "Balanced" if is_imbalanced else None
    cw  = "balanced" if is_imbalanced else None

    num_cols  = profile["numeric_cols"]
    cat_cols  = profile["categorical_cols"]
    bool_cols = profile["boolean_cols"]

    available = [m for m in selected_models
                 if m in {"Logistic Regression", "Random Forest",
                          "XGBoost", "LightGBM", "CatBoost"}]
    total   = len(available)
    results: dict = {}

    for i, name in enumerate(available):
        if progress_callback:
            progress_callback(name, i, total,
                              f"Optuna {n_trials} trials · 5-fold CV")

        # Optuna-level callback: fires after every trial → keeps WebSocket alive
        optuna_cb = _make_optuna_cb(progress_callback, name, i, total, n_trials)

        t0 = time.time()
        try:
            if name in _BOOSTING:
                res = _train_boosting(name, X_train, y_train,
                                      num_cols, cat_cols, bool_cols,
                                      n_trials, spw, cw, acw,
                                      optuna_callbacks=optuna_cb)
            else:
                res = _train_standard(name, X_train, y_train,
                                      num_cols, cat_cols, bool_cols,
                                      n_trials, cw, is_imbalanced,
                                      optuna_callbacks=optuna_cb)
            res["train_time"] = time.time() - t0
            results[name] = res
        except Exception as exc:
            import traceback as _tb
            results[name] = {"error": str(exc), "model": None,
                             "_tb": _tb.format_exc()}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Standard models  (LR, RF)
# ══════════════════════════════════════════════════════════════════════════════

def _train_standard(name, X_train, y_train, num_cols, cat_cols, bool_cols,
                    n_trials, cw, is_imbalanced, optuna_callbacks=None):
    needs_scale = (name == "Logistic Regression")

    def objective(trial):
        params = _suggest(trial, name, cw, is_imbalanced)
        pre  = build_preprocessor(num_cols, cat_cols, bool_cols, scale=needs_scale)
        clf  = _make_clf(name, params)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        cv   = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
        return float(cross_val_score(pipe, X_train, y_train,
                                     cv=cv, scoring="roc_auc",
                                     n_jobs=-1).mean())

    study = _new_study()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=optuna_callbacks or [])

    best  = study.best_params
    pre   = build_preprocessor(num_cols, cat_cols, bool_cols, scale=needs_scale)
    clf   = _make_clf(name, best)
    pipe  = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    return {"model": pipe,
            "cv_score": study.best_value,
            "optuna_val_roc_auc": study.best_value,
            "best_params": best}


# ══════════════════════════════════════════════════════════════════════════════
# Boosting models  (XGBoost, LightGBM, CatBoost)  —  with early stopping
# ══════════════════════════════════════════════════════════════════════════════

def _train_boosting(name, X_train, y_train, num_cols, cat_cols, bool_cols,
                    n_trials, spw, cw, acw, optuna_callbacks=None):
    cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params      = _suggest(trial, name, cw, False, spw=spw)
        fold_scores = []
        fold_n_est  = []

        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[tr_idx]; X_va = X_train.iloc[va_idx]
            y_tr = y_train.iloc[tr_idx]; y_va = y_train.iloc[va_idx]

            pre    = build_preprocessor(num_cols, cat_cols, bool_cols, scale=False)
            X_tr_t = pre.fit_transform(X_tr)
            X_va_t = pre.transform(X_va)

            clf = _make_clf_es(name, params, spw, acw)
            _fit_es(clf, name, X_tr_t, y_tr, X_va_t, y_va)

            y_prob = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(roc_auc_score(y_va, y_prob))
            fold_n_est.append(_best_n(clf, name))

        best_n = int(np.clip(np.median(fold_n_est), 50, _MAX_ITERS))
        trial.set_user_attr("best_n_estimators", best_n)
        return float(np.mean(fold_scores))

    study = _new_study()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=optuna_callbacks or [])

    best_params = study.best_params
    best_n_est  = study.best_trial.user_attrs.get("best_n_estimators", 200)

    pre  = build_preprocessor(num_cols, cat_cols, bool_cols, scale=False)
    clf  = _make_clf_final(name, best_params, best_n_est, spw, acw)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    return {"model": pipe,
            "cv_score": study.best_value,
            "optuna_val_roc_auc": study.best_value,
            "best_params": {**best_params, "n_estimators": best_n_est}}


# ══════════════════════════════════════════════════════════════════════════════
# Optuna parameter suggestions
# ══════════════════════════════════════════════════════════════════════════════

def _suggest(trial, name, cw, is_imbalanced, spw=1.0) -> dict:
    if name == "Logistic Regression":
        return {
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "class_weight": (
                "balanced" if is_imbalanced
                else trial.suggest_categorical("class_weight", ["balanced", None])
            ),
        }

    if name == "Random Forest":
        return {
            "max_depth":        trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
            "min_samples_split":trial.suggest_int("min_samples_split", 2, 20),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight":     "balanced" if is_imbalanced else None,
        }

    if name == "XGBoost":
        return {
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

    if name == "LightGBM":
        return {
            "num_leaves":      trial.suggest_int("num_leaves", 15, 63),
            "max_depth":       trial.suggest_int("max_depth", 3, 8),
            "feature_fraction":trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1":       trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2":       trial.suggest_float("lambda_l2", 1.0, 10.0),
        }

    if name == "CatBoost":
        return {
            "depth":               trial.suggest_int("depth", 3, 7),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":     trial.suggest_float("random_strength", 0.0, 2.0),
        }

    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Classifier factories
# ══════════════════════════════════════════════════════════════════════════════

def _make_clf(name, params):
    if name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE,
            solver="lbfgs", penalty="l2", n_jobs=-1, **params)

    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, **params)

    return None


def _make_clf_es(name, params, spw, acw):
    """Classifier with early stopping for Optuna fold loop."""
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=_MAX_ITERS,
            early_stopping_rounds=_EARLY_STOP_RDS,
            eval_metric="auc",
            scale_pos_weight=spw,
            verbosity=0, random_state=RANDOM_STATE, n_jobs=-1,
            **params)

    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        bf = params.get("bagging_fraction", 1.0)
        return LGBMClassifier(
            n_estimators=_MAX_ITERS,
            class_weight=acw,
            verbose=-1, random_state=RANDOM_STATE, n_jobs=-1,
            bagging_freq=5 if bf < 1.0 else 0,
            **params)

    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=_MAX_ITERS,
            early_stopping_rounds=_EARLY_STOP_RDS,
            eval_metric="AUC",
            use_best_model=True,
            auto_class_weights=acw,
            allow_writing_files=False,
            random_seed=RANDOM_STATE, verbose=0,
            **params)

    return None


def _fit_es(clf, name, X_tr, y_tr, X_va, y_va):
    if name == "XGBoost":
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    elif name == "LightGBM":
        import lightgbm as lgb
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(_EARLY_STOP_RDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ])

    elif name == "CatBoost":
        clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)


def _best_n(clf, name) -> int:
    if name == "XGBoost":
        bi = getattr(clf, "best_iteration", None)
        return int(bi) + 1 if bi is not None else 200

    if name == "LightGBM":
        bi = getattr(clf, "best_iteration_", None)
        return int(bi) if (bi and bi > 0) else 200

    if name == "CatBoost":
        bi = getattr(clf, "best_iteration_", None)
        if bi is None:
            try:
                bi = clf.get_best_iteration()
            except Exception:
                bi = None
        return int(bi) + 1 if bi is not None else 200

    return 200


def _make_clf_final(name, params, n_estimators, spw, acw):
    """Final Pipeline classifier — no early stopping."""
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=n_estimators,
            scale_pos_weight=spw,
            eval_metric="logloss",
            verbosity=0, random_state=RANDOM_STATE, n_jobs=-1,
            **params)

    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        bf = params.get("bagging_fraction", 1.0)
        return LGBMClassifier(
            n_estimators=n_estimators,
            class_weight=acw,
            verbose=-1, random_state=RANDOM_STATE, n_jobs=-1,
            bagging_freq=5 if bf < 1.0 else 0,
            **params)

    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=n_estimators,
            auto_class_weights=acw,
            allow_writing_files=False,
            random_seed=RANDOM_STATE, verbose=0,
            **params)

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _new_study() -> optuna.Study:
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )


def _make_optuna_cb(streamlit_cb, name, model_idx, total_models, n_trials):
    """
    Returns an Optuna study callback list that fires after every trial.
    Calls the Streamlit progress_callback → keeps the WebSocket alive during
    long training runs (prevents 'SessionInfo not initialized' timeout errors).
    """
    if streamlit_cb is None:
        return []

    def _inner(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        completed = trial.number + 1
        try:
            best_val = study.best_value
            detail = f"trial {completed}/{n_trials} · best ROC-AUC {best_val:.4f}"
        except Exception:
            detail = f"trial {completed}/{n_trials}"
        try:
            streamlit_cb(name, model_idx, total_models, detail)
        except Exception:
            pass  # never crash the Optuna loop over a UI error

    return [_inner]
