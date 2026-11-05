"""
Probability calibration.

Class weighting improves how a model ranks customers but inflates its
probabilities, so "40% risk" stops meaning 40 of 100 such customers leave.
A logistic curve fitted on out-of-fold predictions maps the scores back.
Out-of-fold matters: a model's scores on its own training rows are overconfident.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPSILON = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPSILON, 1 - _EPSILON)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Maps raw scores to probabilities. Monotonic, so it never changes the ranking."""

    def fit(self, scores: np.ndarray, y) -> "PlattCalibrator":
        self._curve = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self._curve.fit(_logit(scores), np.asarray(y))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self._curve.predict_proba(_logit(scores))[:, 1]


class CalibratedModel:
    """A fitted pipeline plus the calibrator for its scores."""

    def __init__(self, pipeline, calibrator: PlattCalibrator):
        self.pipeline = pipeline
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:
        calibrated = self.calibrator.transform(self.pipeline.predict_proba(X)[:, 1])
        return np.column_stack([1 - calibrated, calibrated])
