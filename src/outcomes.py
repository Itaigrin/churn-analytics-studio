"""
Is the model still right? Checked against what really happened.

Drift monitoring sees that customers changed. Only real outcomes show whether the
model got worse. Once the outcome period is over, the customers scored earlier get
a column saying who really left, and the model's scores for them are compared with
the numbers it achieved when it was trained.

Two things can go wrong, and they call for different fixes:
  ranking    the model no longer puts leavers above stayers (ROC-AUC fell)
  level      the ranking holds but the probabilities are too high or too low,
             because the churn rate itself moved
Both mean the model should be trained again on recent history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from .evaluator import metrics_at, calibration_error

MIN_CUSTOMERS = 100
MIN_EACH_OUTCOME = 20
RANKING_DROP = 0.05       # ROC-AUC points that count as a real loss
LEVEL_GAP = 0.05          # gap between predicted and actual churn rate that matters
_BOOTSTRAP_ROUNDS = 300


@dataclass
class OutcomeReport:
    n_customers: int
    n_churned: int
    now: dict
    then: dict
    roc_interval: tuple[float, float]
    mean_predicted: float
    actual_rate: float

    @property
    def too_few(self) -> bool:
        return self.n_customers < MIN_CUSTOMERS or min(self.n_churned, self.n_customers - self.n_churned) < MIN_EACH_OUTCOME

    @property
    def ranking_worse(self) -> bool:
        """ROC-AUC fell by more than the threshold, and not just by chance."""
        return (self.then["roc_auc"] - self.now["roc_auc"] > RANKING_DROP
                and self.roc_interval[1] < self.then["roc_auc"])

    @property
    def level_off(self) -> bool:
        """Predicted and actual churn rates differ by more than chance allows."""
        gap = abs(self.mean_predicted - self.actual_rate)
        noise = 2 * np.sqrt(self.actual_rate * (1 - self.actual_rate) / max(self.n_customers, 1))
        return gap > max(LEVEL_GAP, noise)

    @property
    def retrain_recommended(self) -> bool:
        return not self.too_few and (self.ranking_worse or self.level_off)


def outcome_report(probabilities: np.ndarray, y_true, threshold: float, then: dict) -> OutcomeReport:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities)
    now = {}
    if 0 < y.sum() < len(y):
        now = metrics_at(y, p, threshold)
        now["calibration_error"] = calibration_error(y, p)
    return OutcomeReport(
        n_customers=len(y),
        n_churned=int(y.sum()),
        now=now,
        then=then,
        roc_interval=_roc_interval(y, p) if now else (float("nan"), float("nan")),
        mean_predicted=float(p.mean()) if len(p) else float("nan"),
        actual_rate=float(y.mean()) if len(y) else float("nan"),
    )


def _roc_interval(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    values = []
    for _ in range(_BOOTSTRAP_ROUNDS):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            values.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
