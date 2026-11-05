"""
Step 10 — Model Evaluation, Selection and Threshold Choice.

Two separate data sets answer two separate questions:

  Out-of-fold predictions on the training rows choose. Every model is scored on
  customers it did not see in that fold, on one common basis, and this decides
  the calibration, the decision threshold and the winner.

  The holdout reports. It is touched once, by the winner, with the threshold
  already fixed, so the numbers shown are not influenced by any of those choices.

Best Overall Score = 0.45 * PR-AUC + 0.35 * Recall + 0.20 * ROC-AUC
Recall is measured at each model's own threshold on calibrated probabilities.
At a fixed 0.5 it would mostly reward whichever model has the most inflated scores.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .calibration import CalibratedModel, PlattCalibrator
from .config import (
    OOF_FOLDS, RANDOM_STATE, SCORE_WEIGHT_PR_AUC, SCORE_WEIGHT_RECALL, SCORE_WEIGHT_ROC,
)
from .errors import ChurnAppError

logger = logging.getLogger(__name__)

# Calibrated probabilities can sit anywhere: with rare churn the best cut is often below 0.2
_THRESHOLDS = np.round(np.arange(0.02, 0.91, 0.01), 2)
_CALIBRATION_BINS = 10


# ── Threshold optimisation ────────────────────────────────────────────────────

def threshold_curve(y_true, y_prob) -> pd.DataFrame:
    """Recall, precision and F1 at every candidate threshold."""
    y_true = np.asarray(y_true)
    positives = max(int(y_true.sum()), 1)
    rows = []
    for t in _THRESHOLDS:
        flagged = y_prob >= t
        tp = int(y_true[flagged].sum())
        n_flagged = int(flagged.sum())
        recall = tp / positives
        precision = tp / n_flagged if n_flagged else 0.0
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        rows.append({"threshold": t, "f1": f1, "recall": recall, "precision": precision})
    return pd.DataFrame(rows)


def optimize_threshold(y_true, y_prob) -> float:
    curve = threshold_curve(y_true, y_prob)
    return float(curve.loc[curve["f1"].idxmax(), "threshold"])


def overall_score(pr_auc: float, recall: float, roc_auc: float) -> float:
    return (SCORE_WEIGHT_PR_AUC * pr_auc
            + SCORE_WEIGHT_RECALL * recall
            + SCORE_WEIGHT_ROC * roc_auc)


def _metrics_at(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    flagged = y_prob >= threshold
    tp = int(y_true[flagged].sum())
    recall = tp / max(int(y_true.sum()), 1)
    precision = tp / max(int(flagged.sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "best_overall_score": overall_score(pr_auc, recall, roc_auc),
    }


# ── Choosing: out-of-fold on the training rows ────────────────────────────────

def out_of_fold_scores(pipeline, X, y) -> np.ndarray:
    folds = StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(clone(pipeline), X, y, cv=folds, method="predict_proba")[:, 1]


def evaluate_candidate(pipeline, X, y) -> dict:
    """Calibrator, threshold and metrics for one model, from its out-of-fold scores."""
    scores = out_of_fold_scores(pipeline, X, y)
    calibrator = PlattCalibrator().fit(scores, y)
    calibrated = calibrator.transform(scores)
    threshold = optimize_threshold(y, calibrated)
    return {"calibrator": calibrator, "threshold": threshold,
            **_metrics_at(y, calibrated, threshold)}


def evaluate_all(results: dict, X_train, y_train, progress_callback=None) -> dict:
    """Evaluate every successfully trained model on the same basis."""
    usable = [name for name, r in results.items() if r.get("model") is not None]
    for i, name in enumerate(usable):
        if progress_callback:
            progress_callback(name, i, len(usable))
        try:
            results[name].update(evaluate_candidate(results[name]["model"], X_train, y_train))
        except Exception as exc:
            # A model that cannot be evaluated cannot win; its error is shown in the table
            logger.exception("Evaluating %s failed", name)
            results[name].update({"error": str(exc), "model": None})
    return results


def pick_best_model(results: dict) -> str:
    """The model with the highest Best Overall Score on out-of-fold predictions."""
    valid = {name: r for name, r in results.items()
             if r.get("model") is not None and "best_overall_score" in r}
    if not valid:
        details = "; ".join(f"{name}: {r.get('error', 'unknown error')[:200]}"
                            for name, r in results.items())
        raise ChurnAppError(f"No model could be trained on this data. {details}")
    return max(valid, key=lambda name: valid[name]["best_overall_score"])


# ── Reporting: the untouched holdout ──────────────────────────────────────────

def calibration_error(y_true, y_prob) -> float:
    """Average gap between predicted probability and what actually happened."""
    y_true = np.asarray(y_true)
    edges = np.linspace(0, 1, _CALIBRATION_BINS + 1)
    bins = np.clip(np.digitize(y_prob, edges) - 1, 0, _CALIBRATION_BINS - 1)
    error = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        error += in_bin.mean() * abs(y_true[in_bin].mean() - y_prob[in_bin].mean())
    return float(error)


def evaluate_holdout(model: CalibratedModel, threshold: float, X_test, y_test) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        **_metrics_at(y_test, y_prob, threshold),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "calibration_error": calibration_error(y_test, y_prob),
        "mean_probability": float(y_prob.mean()),
        "actual_rate": float(np.mean(y_test)),
        "n_customers": int(len(y_test)),
        "confusion_matrix": confusion_matrix(y_test, y_prob >= threshold),
        "threshold_curve": threshold_curve(y_test, y_prob),
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(results: dict) -> pd.DataFrame:
    """All models on one basis (out-of-fold), best first. Failed models go last."""
    rows = []
    for name, r in results.items():
        if r.get("error") or "best_overall_score" not in r:
            rows.append({
                "Model": name, "Status": f"❌ {str(r.get('error', 'not evaluated'))[:50]}",
                "PR-AUC": "—", "Recall": "—", "ROC-AUC": "—", "Best Overall Score": "—",
            })
            continue
        rows.append({
            "Model":              name,
            "Status":             "✅",
            "PR-AUC":             f"{r['pr_auc']:.4f}",
            "Recall":             f"{r['recall']:.4f}",
            "ROC-AUC":            f"{r['roc_auc']:.4f}",
            "Best Overall Score": f"{r['best_overall_score']:.4f}",
        })

    table = pd.DataFrame(rows)
    order = pd.to_numeric(table["Best Overall Score"], errors="coerce")
    return table.loc[order.sort_values(ascending=False, na_position="last").index].reset_index(drop=True)
