"""
Utility helpers: file I/O, serialisation, package installation.
"""

from __future__ import annotations

import io
import json
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd


# ── Download helpers ──────────────────────────────────────────────────────────

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=2, default=_json_default).encode("utf-8")


def to_pickle_bytes(obj) -> bytes:
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return buf.getvalue()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ── Package auto-install ──────────────────────────────────────────────────────

def try_install(package: str) -> bool:
    """Attempt to pip-install a package. Returns True if successful."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def ensure_optional_libraries() -> dict[str, bool]:
    """
    Try to import optional ML libraries; install if missing.
    Returns {library_name: available}.
    """
    libs = {
        "xgboost":   "xgboost",
        "lightgbm":  "lightgbm",
        "catboost":  "catboost",
        "optuna":    "optuna",
        "shap":      "shap",
    }
    status = {}
    for name, pkg in libs.items():
        try:
            __import__(name)
            status[name] = True
        except ImportError:
            installed = try_install(pkg)
            try:
                __import__(name)
                status[name] = True
            except ImportError:
                status[name] = False
    return status


# ── Report building ───────────────────────────────────────────────────────────

def build_final_report(
    results: dict,
    best_name: str,
    profile: dict,
    mode_key: str,
    new_feature_names: list[str],
) -> dict:
    """
    Build the Step 12 final report as a serialisable dict.
    """
    best = results.get(best_name, {})

    report = {
        "best_model":        best_name,
        "best_params":       best.get("best_params", {}),
        "best_roc_auc":      best.get("roc_auc"),
        "best_f1":           best.get("f1"),
        "best_accuracy":     best.get("accuracy"),
        "best_cv_score":     best.get("cv_score"),
        "mode":              mode_key,
        "dataset_rows":      profile["n_rows"],
        "dataset_cols":      profile["n_cols_raw"],
        "numeric_features":  profile["numeric_cols"],
        "categorical_features": profile["categorical_cols"],
        "boolean_features":  profile["boolean_cols"],
        "dropped_features":  profile["drop_cols"],
        "engineered_features": new_feature_names,
        "all_models": {
            name: {
                "roc_auc":          r.get("roc_auc"),
                "f1":               r.get("f1"),
                "accuracy":         r.get("accuracy"),
                "balanced_accuracy": r.get("balanced_accuracy"),
                "precision":        r.get("precision"),
                "recall":           r.get("recall"),
                "pr_auc":           r.get("pr_auc"),
                "cv_score":         r.get("cv_score"),
                "train_time_s":     r.get("train_time"),
                "error":            r.get("error"),
            }
            for name, r in results.items()
        },
    }

    return report
