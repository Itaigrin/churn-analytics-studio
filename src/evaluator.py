"""
Step 10 — Model Evaluation, Selection and Threshold Choice.

Two separate data sets answer two separate questions:

  Out-of-fold predictions on the training rows choose. Every model is scored on
  customers it did not see in that fold, on one common basis, and this decides
  the calibration, the decision threshold and the winner.

  The holdout reports. It is touched once, by the winner, with the threshold
  already fixed, so the numbers shown are not influenced by any of those choices.

The winner is the model with the highest PR-AUC. Three questions are reported
separately, because each needs its own measure:
  ranking      PR-AUC, ROC-AUC and the lift in the riskiest 10%
  probability  log loss, Brier score and calibration error
  decision     recall and precision, which only exist at a chosen threshold
A blend of them would mix a model's quality with a decision about how many to contact.
On the public datasets, choosing by PR-AUC found the best model on unseen customers
more often than the weighted score it replaced.
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
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from .business import lift_table
from .calibration import CalibratedModel, PlattCalibrator
from .config import INTERVAL_ROUNDS, OOF_FOLDS, RANDOM_STATE, TOP_SHARE
from .errors import ChurnAppError
from .trainer import make_folds

logger = logging.getLogger(__name__)

# Calibrated probabilities can sit anywhere: with rare churn the best cut is often below 0.2
_THRESHOLDS = np.round(np.arange(0.02, 0.91, 0.01), 2)
_CALIBRATION_BINS = 10


# ── Threshold optimisation ────────────────────────────────────────────────────

def threshold_curve(y_true, y_prob) -> pd.DataFrame:
    """Recall, precision, F1 and the share of customers flagged, at every candidate threshold."""
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
        rows.append({"threshold": t, "f1": f1, "recall": recall, "precision": precision,
                     "flagged": n_flagged / max(len(y_true), 1)})
    return pd.DataFrame(rows)


def optimize_threshold(y_true, y_prob) -> float:
    curve = threshold_curve(y_true, y_prob)
    return float(curve.loc[curve["f1"].idxmax(), "threshold"])


def metrics_at(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    flagged = y_prob >= threshold
    tp = int(y_true[flagged].sum())
    recall = tp / max(int(y_true.sum()), 1)
    precision = tp / max(int(flagged.sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def lift_at_top(y_true, y_prob, share: float = TOP_SHARE) -> float:
    """Churn rate among the riskiest share of customers, divided by the overall churn rate."""
    y_true = np.asarray(y_true)
    top = np.argsort(-np.asarray(y_prob), kind="stable")[: max(1, int(round(len(y_true) * share)))]
    return float(y_true[top].mean() / max(y_true.mean(), 1e-12))


def model_quality(y_true, y_prob) -> dict:
    """How well a model ranks customers and how true its probabilities are, at no threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "lift_top10": lift_at_top(y_true, y_prob),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_error": calibration_error(y_true, y_prob),
    }


def quality_score(roc_auc: float) -> float:
    """
    0 for a coin toss, 100 for a perfect ranking: the accuracy ratio, 2 × ROC-AUC − 1.
    It is the same idea as the 0–100 quality scores of campaign tools: lift over random.
    """
    return float(np.clip(200 * roc_auc - 100, 0, 100))


# Churn depends partly on things no file records, so a near-perfect score usually means a
# column gives the answer away (it was recorded after the customer left)
TOO_GOOD_SCORE = 95


def quality_grade(score: float) -> str:
    if score >= TOO_GOOD_SCORE:
        return "Check for leaks"
    if score >= 60:
        return "Excellent"
    if score >= 40:
        return "Good"
    if score >= 20:
        return "Fair"
    return "Weak"


def metric_intervals(y_true, y_prob) -> dict:
    """95% ranges for the headline metrics, by resampling the customers."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(RANDOM_STATE)
    draws = {"pr_auc": [], "roc_auc": [], "lift_top10": []}
    for _ in range(INTERVAL_ROUNDS):
        i = rng.integers(0, len(y_true), len(y_true))
        if 0 < y_true[i].sum() < len(i):
            draws["pr_auc"].append(average_precision_score(y_true[i], y_prob[i]))
            draws["roc_auc"].append(roc_auc_score(y_true[i], y_prob[i]))
            draws["lift_top10"].append(lift_at_top(y_true[i], y_prob[i]))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items() if v}


# ── Choosing: out-of-fold on the training rows ────────────────────────────────

def out_of_fold_scores(pipeline, X, y, groups=None) -> np.ndarray:
    folds = make_folds(OOF_FOLDS, groups)
    return cross_val_predict(clone(pipeline), X, y, groups=groups, cv=folds, method="predict_proba")[:, 1]


def evaluate_candidate(pipeline, X, y, groups=None) -> dict:
    """Calibrator, threshold and metrics for one model, from its out-of-fold scores."""
    scores = out_of_fold_scores(pipeline, X, y, groups)
    calibrator = PlattCalibrator().fit(scores, y)
    calibrated = calibrator.transform(scores)
    threshold = optimize_threshold(y, calibrated)
    return {"calibrator": calibrator, "threshold": threshold,
            **metrics_at(y, calibrated, threshold), **model_quality(y, calibrated)}


def evaluate_all(results: dict, X_train, y_train, progress_callback=None, groups=None) -> dict:
    """Evaluate every successfully trained model on the same basis."""
    usable = [name for name, r in results.items() if r.get("model") is not None]
    for i, name in enumerate(usable):
        if progress_callback:
            progress_callback(name, i, len(usable))
        try:
            results[name].update(evaluate_candidate(results[name]["model"], X_train, y_train, groups))
        except Exception as exc:
            # A model that cannot be evaluated cannot win; its error is shown in the table
            logger.exception("Evaluating %s failed", name)
            results[name].update({"error": str(exc), "model": None})
    return results


def pick_best_model(results: dict) -> str:
    """The model with the highest PR-AUC on out-of-fold predictions."""
    valid = {name: r for name, r in results.items()
             if r.get("model") is not None and "pr_auc" in r}
    if not valid:
        details = "; ".join(f"{name}: {r.get('error', 'unknown error')[:200]}"
                            for name, r in results.items())
        raise ChurnAppError(f"No model could be trained on this data. {details}")
    return max(valid, key=lambda name: valid[name]["pr_auc"])


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
    quality = model_quality(y_test, y_prob)
    score = quality_score(quality["roc_auc"])
    return {
        **metrics_at(y_test, y_prob, threshold),
        **quality,
        "quality_score": score,
        "quality_grade": quality_grade(score),
        "intervals": metric_intervals(y_test, y_prob),
        "mean_probability": float(y_prob.mean()),
        "actual_rate": float(np.mean(y_test)),
        "n_customers": int(len(y_test)),
        "confusion_matrix": confusion_matrix(y_test, y_prob >= threshold),
        "threshold_curve": threshold_curve(y_test, y_prob),
        "lift_table": lift_table(y_test, y_prob),
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(results: dict) -> pd.DataFrame:
    """All models on one basis (out-of-fold), highest PR-AUC first. Failed models go last."""
    rows = []
    for name, r in results.items():
        if r.get("error") or "pr_auc" not in r:
            rows.append({
                "Model": name, "Status": f"❌ {str(r.get('error', 'not evaluated'))[:50]}",
                "PR-AUC": "—", "Top 10% Lift": "—", "Log Loss": "—", "ROC-AUC": "—",
            })
            continue
        rows.append({
            "Model":        name,
            "Status":       "✅",
            "PR-AUC":       f"{r['pr_auc']:.4f}",
            "Top 10% Lift": f"{r['lift_top10']:.2f}×",
            "Log Loss":     f"{r['log_loss']:.4f}",
            "ROC-AUC":      f"{r['roc_auc']:.4f}",
        })

    table = pd.DataFrame(rows)
    order = pd.to_numeric(table["PR-AUC"], errors="coerce")
    return table.loc[order.sort_values(ascending=False, na_position="last").index].reset_index(drop=True)
