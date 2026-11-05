"""Checking the model against what really happened."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_customers
from src.outcomes import outcome_report

THEN = {"roc_auc": 0.85, "pr_auc": 0.6, "recall": 0.7}


def _scored(n=3000, seed=0, rate_shift=0.0, noise=0.0):
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 6, n)
    y = (rng.random(n) < np.clip(p + rate_shift, 0, 1)).astype(int)
    flip = rng.random(n) < noise
    y[flip] = rng.integers(0, 2, flip.sum())
    return p, y


def test_a_model_that_still_works_is_left_alone():
    p, y = _scored()
    report = outcome_report(p, y, 0.3, {**THEN, "roc_auc": 0.73})
    assert not report.ranking_worse and not report.level_off and not report.retrain_recommended


def test_a_model_that_ranks_worse_is_flagged():
    p, y = _scored(noise=0.8)
    report = outcome_report(p, y, 0.3, THEN)
    assert report.ranking_worse and report.retrain_recommended


def test_a_moved_churn_rate_is_flagged_even_if_the_ranking_holds():
    p, y = _scored(rate_shift=0.12)
    report = outcome_report(p, y, 0.3, {**THEN, "roc_auc": report_roc(p, y)})
    assert not report.ranking_worse
    assert report.level_off and report.retrain_recommended


def report_roc(p, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def test_a_small_drop_by_chance_is_not_called_a_loss():
    p, y = _scored(n=300)
    report = outcome_report(p, y, 0.3, {**THEN, "roc_auc": report_roc(p, y) + 0.04})
    assert not report.ranking_worse


def test_too_few_outcomes_are_not_judged():
    p, y = _scored(n=80)
    assert outcome_report(p, y, 0.3, THEN).too_few
    p = np.full(500, 0.2)
    y = np.zeros(500, dtype=int)
    y[:5] = 1
    report = outcome_report(p, y, 0.3, THEN)
    assert report.too_few and not report.retrain_recommended


def test_the_trained_model_checks_real_outcomes(training_run):
    artifact = training_run.artifact
    later = make_customers(1500, seed=21)
    report = artifact.check_outcomes(later, "Churn", "Yes", training_run.holdout)
    assert not report.too_few and not report.retrain_recommended
    assert abs(report.mean_predicted - report.actual_rate) < 0.05

    # A year later nobody's churn depends on these columns any more
    shuffled = later.assign(Churn=later["Churn"].sample(frac=1, random_state=0).to_numpy())
    assert artifact.check_outcomes(shuffled, "Churn", "Yes", training_run.holdout).ranking_worse


def test_rows_without_an_outcome_are_left_out(training_run):
    later = make_customers(400, seed=22)
    later.loc[later.index[:50], "Churn"] = None
    report = training_run.artifact.check_outcomes(later, "Churn", "Yes", training_run.holdout)
    assert report.n_customers == 350
