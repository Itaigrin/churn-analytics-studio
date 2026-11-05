"""Dates as model inputs, and testing on later or unseen customers."""
import numpy as np
import pandas as pd
import pytest

from conftest import FAST_BUDGET, make_customers
from src.errors import DataValidationError
from src.pipeline_builder import detect_date_order, parse_dates
from src.workflow import date_columns, run_training, screen_columns

LR = ["Logistic Regression"]


# ── Reading dates ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("values,order,first", [
    (["2024-01-05", "2024-12-31"], "yearfirst", "2024-01-05"),
    (["05/01/2024", "31/12/2024"], "dayfirst", "2024-01-05"),
    (["01/05/2024", "12/31/2024"], "monthfirst", "2024-01-05"),
    (["05.01.2024", "07.02.2024"], "dayfirst", "2024-01-05"),
])
def test_each_way_of_writing_dates_is_read_correctly(values, order, first):
    s = pd.Series(values)
    assert detect_date_order(s) == order
    assert parse_dates(s, order)[0] == pd.Timestamp(first)


def test_iso_dates_are_never_read_day_first():
    # pandas reads "2024-01-05" as 1 May when told the day comes first
    assert parse_dates(pd.Series(["2024-01-05"]), detect_date_order(pd.Series(["2024-01-05"])))[0].month == 1


def test_excel_dates_and_unreadable_text():
    s = pd.Series(pd.to_datetime(["2024-03-01", None]))
    assert detect_date_order(s) == "native"
    assert parse_dates(pd.Series(["31/12/2024", "soon"]), "dayfirst").isna().tolist() == [False, True]


# ── Dates in the model ────────────────────────────────────────────────────────

def dated_customers(n=600, seed=0):
    """Customers who stopped logging in long ago are the ones who leave."""
    df = make_customers(n, seed=seed)
    rng = np.random.default_rng(seed)
    churned = df["Churn"] == "Yes"
    days = np.where(churned, rng.integers(40, 200, n), rng.integers(0, 60, n))
    df["Last Login"] = (pd.Timestamp("2025-06-30") - pd.to_timedelta(days, "D")).strftime("%d/%m/%Y")
    df["Signup"] = (pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), "D")).strftime("%Y-%m-%d")
    return df


@pytest.fixture(scope="module")
def dated_run():
    return run_training(dated_customers(), "Churn", id_col="Customer ID", selected_models=LR, budget=FAST_BUDGET)


def test_dates_become_days_before_the_latest_date(dated_run):
    schema = dated_run.artifact.schema
    assert {"last_login", "signup"} <= set(schema.columns)
    assert "last_login_days_ago" in dated_run.artifact.model_columns
    assert any("2025-06-30" in n and "last_login" in n for n in dated_run.notes)


def test_recency_carries_the_signal_it_should(dated_run):
    assert dated_run.holdout["roc_auc"] > 0.9


def test_a_customer_scores_the_same_alone_given_the_date(dated_run):
    artifact = dated_run.artifact
    new = dated_customers(40, seed=3).drop(columns=["Churn"]).reset_index(drop=True)
    as_of = pd.Timestamp("2025-06-30")
    in_file = artifact.predict_proba(new, as_of=as_of)
    alone = np.array([artifact.predict_proba(new.iloc[[i]], as_of=as_of)[0] for i in range(len(new))])
    np.testing.assert_allclose(in_file, alone, atol=1e-9)


def test_a_later_reference_date_makes_the_same_customer_riskier(dated_run):
    new = dated_customers(40, seed=3).drop(columns=["Churn"])
    now = dated_run.artifact.predict_proba(new, as_of="2025-06-30")
    later = dated_run.artifact.predict_proba(new, as_of="2025-12-31")
    assert later.mean() > now.mean()


def test_a_reason_can_be_how_long_ago_something_happened(dated_run):
    new = dated_customers(60, seed=4).drop(columns=["Churn"])
    p = dated_run.artifact.predict_proba(new, as_of="2025-06-30")
    reasons = dated_run.artifact.explain(new, p, as_of="2025-06-30")
    assert any(r and r[0].column == "last_login_days_ago" for r in reasons)


def test_each_row_can_carry_its_own_reference_date():
    df = dated_customers()
    df["Snapshot"] = "30/06/2025"
    run = run_training(df, "Churn", id_col="Customer ID", selected_models=LR, budget=FAST_BUDGET, as_of="Snapshot")
    schema = run.artifact.schema
    assert schema.as_of_column == "snapshot"
    assert "snapshot_days_ago" not in run.artifact.model_columns
    new = dated_customers(10, seed=5).drop(columns=["Churn"])
    new["Snapshot"] = "30/06/2025"
    fixed = run.artifact.predict_proba(new, as_of="2025-06-30")
    np.testing.assert_allclose(run.artifact.predict_proba(new), fixed, atol=1e-9)
    assert "snapshot" in run.artifact.check(new.drop(columns=["Snapshot"])).missing


def test_date_columns_are_listed_by_their_names_in_the_file():
    assert set(date_columns(dated_customers(), "Churn", "Customer ID")) == {"Last Login", "Signup"}


def test_a_date_recorded_after_leaving_is_flagged_as_a_leak():
    df = dated_customers()
    # Leavers' status changed after the data was taken, everyone else's long before
    df["Status Date"] = np.where(df["Churn"] == "Yes", "15/07/2025", "01/03/2024")
    assert "status_date" in [f.column for f in screen_columns(df, "Churn", "Customer ID")]


def test_a_non_date_column_cannot_be_the_reference():
    with pytest.raises(DataValidationError, match="not a date column"):
        run_training(dated_customers(), "Churn", id_col="Customer ID", selected_models=LR,
                     budget=FAST_BUDGET, split_by="Plan")


# ── Testing on later customers ────────────────────────────────────────────────

def test_the_time_split_tests_on_the_most_recent_customers():
    df = dated_customers()
    df["Joined"] = (pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(len(df)), "D")).strftime("%Y-%m-%d")
    run = run_training(df, "Churn", id_col="Customer ID", selected_models=LR, budget=FAST_BUDGET, split_by="Joined")
    assert run.n_holdout == 120
    assert "most recent" in run.holdout_description and "joined" in run.holdout_description
    assert "(2024-04-25 to 2024-08-22)" in run.holdout_description


def test_a_time_split_needs_both_outcomes_in_the_recent_customers():
    df = dated_customers()
    df["Joined"] = (pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(len(df)), "D")).strftime("%Y-%m-%d")
    df.loc[df.index[-150:], "Churn"] = "No"
    with pytest.raises(DataValidationError, match="most recent customers"):
        run_training(df, "Churn", id_col="Customer ID", selected_models=LR, budget=FAST_BUDGET, split_by="Joined")


# ── Several rows per customer ─────────────────────────────────────────────────

def test_a_customer_with_several_rows_is_never_on_both_sides():
    monthly = pd.concat([make_customers(300).assign(Month=m) for m in range(3)], ignore_index=True)
    run = run_training(monthly, "Churn", id_col="Customer ID", selected_models=LR, budget=FAST_BUDGET)
    assert "kept whole" in run.holdout_description
    assert any("kept on the same side" in w for w in run.warnings)


def test_grouped_holdout_has_no_shared_customers():
    from src.workflow import _split
    y = pd.Series([0, 1] * 300)
    groups = pd.Series(np.repeat(np.arange(200), 3).astype(str))
    train_idx, test_idx, _ = _split(pd.DataFrame(index=y.index), y, groups, None)
    assert not set(groups.loc[train_idx]) & set(groups.loc[test_idx])
    assert 0.15 < len(test_idx) / len(y) < 0.25


def test_hyperparameter_search_and_selection_keep_customers_together(monkeypatch):
    from sklearn.model_selection import StratifiedGroupKFold
    from src import trainer
    seen = []
    original = trainer.make_folds
    monkeypatch.setattr(trainer, "make_folds", lambda n, groups=None: seen.append(groups is not None) or original(n, groups))
    import src.evaluator as evaluator
    monkeypatch.setattr(evaluator, "make_folds", trainer.make_folds)
    monthly = pd.concat([make_customers(300).assign(Month=m) for m in range(3)], ignore_index=True)
    run_training(monthly, "Churn", id_col="Customer ID", selected_models=LR, budget=FAST_BUDGET)
    assert seen and all(seen)
    assert isinstance(original(5, pd.Series(["a"])), StratifiedGroupKFold)
