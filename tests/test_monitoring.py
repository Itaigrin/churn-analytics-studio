"""Telling when new customers no longer look like the training customers."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_customers
from src.monitoring import MIN_CUSTOMERS, build_drift_reference, drift_report, psi


def _reference(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"age": rng.normal(40, 10, n), "plan": rng.choice(["a", "b", "c"], n, p=[.6, .3, .1])})
    return X, build_drift_reference(X, ["age"], rng.beta(2, 6, n))


def test_psi_is_zero_for_the_same_shares_and_grows_with_change():
    e = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(e, e) == pytest.approx(0)
    assert psi(e, np.array([.3, .2, .25, .25])) < psi(e, np.array([.7, .1, .1, .1]))


def test_the_same_population_is_stable():
    X, ref = _reference()
    new, _ = _reference(n=1000, seed=1)
    report = drift_report(ref, new, np.random.default_rng(1).beta(2, 6, 1000))
    assert report.columns["PSI"].max() < 0.1 and report.score_psi < 0.1
    assert not report.retrain_recommended


def test_an_older_more_premium_population_is_flagged():
    X, ref = _reference()
    rng = np.random.default_rng(2)
    new = pd.DataFrame({"age": rng.normal(60, 10, 1000), "plan": rng.choice(["a", "b", "c"], 1000, p=[.1, .3, .6])})
    report = drift_report(ref, new, rng.beta(5, 3, 1000))
    assert set(report.shifted) == {"age", "plan"}
    assert report.retrain_recommended


def test_new_categories_and_missing_values_count_as_change():
    X, ref = _reference()
    new = pd.DataFrame({"age": [np.nan] * 500, "plan": ["z"] * 500})
    report = drift_report(ref, new, np.full(500, 0.2))
    assert report.columns.set_index("Column")["PSI"].min() > 0.25


def test_a_handful_of_customers_is_not_judged():
    X, ref = _reference()
    report = drift_report(ref, X.head(MIN_CUSTOMERS - 1), np.full(MIN_CUSTOMERS - 1, 0.9))
    assert report.too_few and not report.retrain_recommended


def test_the_trained_model_monitors_new_files(training_run, customers):
    artifact = training_run.artifact
    same = make_customers(800, seed=11).drop(columns=["Churn"])
    report = artifact.monitor(same, artifact.predict_proba(same))
    assert not report.retrain_recommended

    shifted = same.copy()
    shifted["Tenure"] = 70
    shifted["Support Calls"] = shifted["Support Calls"] + 6
    report = artifact.monitor(shifted, artifact.predict_proba(shifted))
    assert {"tenure", "support_calls"} <= set(report.shifted)
    assert report.retrain_recommended
