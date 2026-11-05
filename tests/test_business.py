"""Money: break-even, campaign estimates and lift."""
import numpy as np
import pandas as pd
import pytest

from src.business import Economics, campaign_estimate, contact_everyone, file_estimate, lift_table
from src.evaluator import threshold_curve


def test_a_contact_pays_exactly_above_the_breakeven_probability():
    economics = Economics(value=1000, cost=50, success=0.25)
    assert economics.breakeven == pytest.approx(0.2)
    gain = economics.gain(np.array([0.1, 0.2, 0.4]))
    assert gain[0] < 0 and gain[1] == pytest.approx(0) and gain[2] > 0


def test_the_campaign_estimate_matches_a_hand_count():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .3, .8, .1, .1, .1, .1, .1, .1, .1])
    curve = threshold_curve(y, p)
    estimate = campaign_estimate(curve, churn_rate=0.2, economics=Economics(1000, 50, 0.5), per=10)
    row = estimate[estimate["threshold"] == 0.5].iloc[0]
    # Two of ten flagged, one of the two churners among them, half of whom are kept
    assert row["contacted"] == pytest.approx(2)
    assert row["churners_reached"] == pytest.approx(1)
    assert row["net"] == pytest.approx(0.5 * 1000 - 2 * 50)


def test_contacting_everyone_is_the_baseline():
    assert contact_everyone(0.2, Economics(1000, 50, 0.25), per=1000) == pytest.approx(200 * 0.25 * 1000 - 1000 * 50)


def test_the_file_estimate_adds_up_each_customer():
    economics = Economics(1000, 50, 0.25)
    p = np.array([0.9, 0.5, 0.1])
    estimate = file_estimate(p, 0.3, economics)
    assert estimate["contacted"] == 2
    assert estimate["expected_churners_reached"] == pytest.approx(1.4)
    assert estimate["net"] == pytest.approx(1.4 * 0.25 * 1000 - 2 * 50)


def test_with_calibrated_probabilities_the_breakeven_is_the_most_profitable_cut():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 4, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    economics = Economics(1000, 60, 0.3)
    estimate = campaign_estimate(threshold_curve(y, p), y.mean(), economics)
    best = estimate.loc[estimate["net"].idxmax(), "threshold"]
    assert best == pytest.approx(economics.breakeven, abs=0.03)


def test_the_lift_table_ranks_customers_by_risk():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0] * 10)
    p = np.where(y == 1, 0.9, 0.1) + np.linspace(0, 0.01, 100)
    table = lift_table(y, p, groups=10)
    assert table["Customers"].sum() == 100
    assert table.iloc[0]["Churn Rate"] == 1.0
    assert table.iloc[0]["Lift"] == pytest.approx(1 / 0.3)
    assert table.iloc[2]["Churners Caught So Far"] == pytest.approx(1.0)
    assert table.iloc[-1]["Churners Caught So Far"] == pytest.approx(1.0)


def test_the_holdout_carries_the_lift_table(training_run):
    table = training_run.holdout["lift_table"]
    assert len(table) == 10
    assert table.iloc[0]["Lift"] > 1.5
    assert table["Churners Caught So Far"].is_monotonic_increasing
