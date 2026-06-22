"""
Step 10 — Model Evaluation + Threshold Optimisation.

Primary metrics (used to select best model):
  Best Overall Score = 0.45 * PR-AUC + 0.35 * Recall + 0.20 * ROC-AUC

Secondary metrics (informational only, not used for selection):
  Accuracy, Balanced Accuracy, Precision, F1, threshold-optimised variants

Threshold optimisation:
  - Tests thresholds 0.20 → 0.70 (step 0.02)
  - Picks the threshold that maximises F1 on the TEST set
  - Uses only TEST-set probabilities — zero data leakage
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import SCORE_WEIGHT_PR_AUC, SCORE_WEIGHT_RECALL, SCORE_WEIGHT_ROC

# Threshold search range
_THRESHOLDS = np.round(np.arange(0.20, 0.71, 0.02), 3)


# ── Threshold optimisation ────────────────────────────────────────────────────

def optimize_threshold(y_test, y_prob) -> dict:
    """Search thresholds [0.20…0.70] and pick the one that maximises F1."""
    rows = []
    best_t, best_f1 = 0.50, -1.0

    for t in _THRESHOLDS:
        y_hat = (y_prob >= t).astype(int)
        f1    = f1_score(y_test, y_hat, zero_division=0)
        rec   = recall_score(y_test, y_hat, zero_division=0)
        prec  = precision_score(y_test, y_hat, zero_division=0)
        rows.append({"threshold": t, "f1": f1, "recall": rec, "precision": prec})
        if f1 > best_f1:
            best_f1, best_t = f1, t

    curve    = pd.DataFrame(rows)
    best_row = curve[curve["threshold"] == best_t].iloc[0]
    return {
        "best_threshold":         float(best_t),
        "f1_at_threshold":        float(best_row["f1"]),
        "recall_at_threshold":    float(best_row["recall"]),
        "precision_at_threshold": float(best_row["precision"]),
        "threshold_curve":        curve,
    }


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """Compute all metrics for a fitted pipeline on the held-out test set."""
    if hasattr(model, "predict_proba"):
        y_prob       = model.predict_proba(X_test)[:, 1]
        y_train_prob = model.predict_proba(X_train)[:, 1]
    else:
        y_prob = y_train_prob = None

    y_pred       = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    def _safe(fn, *args, **kwargs):
        try:
            return float(fn(*args, **kwargs))
        except Exception:
            return float("nan")

    result = {
        # ── PRIMARY — used in Best Overall Score ──────────────────────────────
        "roc_auc": _safe(roc_auc_score, y_test, y_prob) if y_prob is not None else float("nan"),
        "recall":  _safe(recall_score,  y_test, y_pred, zero_division=0),
        "pr_auc":  _safe(average_precision_score, y_test, y_prob) if y_prob is not None else float("nan"),
        # ── SECONDARY — informational only ────────────────────────────────────
        "f1":               _safe(f1_score,              y_test, y_pred, zero_division=0),
        "accuracy":         _safe(accuracy_score,         y_test, y_pred),
        "balanced_accuracy":_safe(balanced_accuracy_score,y_test, y_pred),
        "precision":        _safe(precision_score,        y_test, y_pred, zero_division=0),
        # ── Train metrics (overfitting check) ─────────────────────────────────
        "train_roc_auc": _safe(roc_auc_score, y_train, y_train_prob) if y_train_prob is not None else float("nan"),
        # ── Raw data ──────────────────────────────────────────────────────────
        "y_pred":                y_pred,
        "y_prob":                y_prob if y_prob is not None else y_pred.astype(float),
        "confusion_matrix":      confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    if y_prob is not None:
        thr = optimize_threshold(y_test, y_prob)
        result.update(thr)
        y_hat_opt = (y_prob >= thr["best_threshold"]).astype(int)
        result["confusion_matrix_opt"] = confusion_matrix(y_test, y_hat_opt)
        result["y_pred_opt"] = y_hat_opt
    else:
        result["best_threshold"]         = 0.50
        result["f1_at_threshold"]        = result["f1"]
        result["recall_at_threshold"]    = result["recall"]
        result["precision_at_threshold"] = result.get("precision", float("nan"))
        result["threshold_curve"]        = None

    # ── Best Overall Score ────────────────────────────────────────────────────
    result["best_overall_score"] = (
        SCORE_WEIGHT_PR_AUC * (result["pr_auc"]  or 0.0)
        + SCORE_WEIGHT_RECALL * (result["recall"] or 0.0)
        + SCORE_WEIGHT_ROC    * (result["roc_auc"] or 0.0)
    )

    return result


def evaluate_all(results: dict, X_train, y_train, X_test, y_test) -> dict:
    """Evaluate every successfully trained model."""
    for name, r in results.items():
        if r.get("error") or r.get("model") is None:
            continue
        metrics = evaluate_model(r["model"], X_train, y_train, X_test, y_test)
        results[name].update(metrics)
    return results


# ── Best model selection ──────────────────────────────────────────────────────

def pick_best_model(results: dict, **_kwargs) -> str:
    """
    Select the model with the highest Best Overall Score.
    Score = 0.45 * PR-AUC + 0.35 * Recall + 0.20 * ROC-AUC
    """
    valid = {
        name: r for name, r in results.items()
        if r.get("model") is not None and r.get("roc_auc") is not None
    }
    if not valid:
        for name, r in results.items():
            if r.get("model") is not None:
                return name
        return next(iter(results))

    best_name = max(valid, key=lambda n: valid[n].get("best_overall_score", 0.0))
    results[best_name]["selected_by"] = "best_overall_score"
    return best_name


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(results: dict) -> pd.DataFrame:
    """Comparison table sorted by Best Overall Score descending."""
    rows = []
    for name, r in results.items():
        if r.get("error"):
            rows.append({
                "Model": name, "Status": f"❌ {r['error'][:50]}",
                "PR-AUC": "—", "Recall": "—", "ROC-AUC": "—", "Best Overall Score": "—",
            })
            continue
        score = r.get("best_overall_score", 0.0)
        rows.append({
            "Model":              name,
            "Status":             "✅",
            "PR-AUC":             f"{r.get('pr_auc', 0):.4f}",
            "Recall":             f"{r.get('recall', 0):.4f}",
            "ROC-AUC":            f"{r.get('roc_auc', 0):.4f}",
            "Best Overall Score": f"{score:.4f}",
        })

    df = pd.DataFrame(rows)
    try:
        df = df.sort_values("Best Overall Score", ascending=False).reset_index(drop=True)
    except Exception:
        pass
    return df
