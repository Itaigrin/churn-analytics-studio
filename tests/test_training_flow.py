"""The whole flow on data with a known answer."""
import numpy as np
import pandas as pd
import pytest

from conftest import FAST_BUDGET, make_customers
from src.calibration import PlattCalibrator
from src.errors import ChurnAppError, DataValidationError
from src.evaluator import (
    build_comparison_table, calibration_error, optimize_threshold, pick_best_model, threshold_curve,
)
from src.trainer import train_all_models
from src.workflow import run_training, screen_columns


# ── Calibration ───────────────────────────────────────────────────────────────

def test_calibrated_probabilities_match_what_happened(training_run):
    holdout = training_run.holdout
    assert abs(holdout["mean_probability"] - holdout["actual_rate"]) < 0.05
    assert holdout["calibration_error"] < 0.10


def test_calibration_repairs_inflated_scores_without_changing_the_ranking():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.02, 0.5, 5000)
    y = (rng.random(5000) < true_p).astype(int)
    inflated = np.clip(true_p * 1.8, 0, 0.99)
    calibrated = PlattCalibrator().fit(inflated, y).transform(inflated)
    assert calibration_error(y, inflated) > 0.1
    assert calibration_error(y, calibrated) < 0.03
    assert (np.argsort(calibrated) == np.argsort(inflated)).all()


def test_threshold_is_chosen_on_the_calibrated_scale(training_run):
    # The flagged rate on unseen customers would collapse if it were not
    holdout = training_run.holdout
    assert holdout["recall"] > 0.4
    assert 0.02 <= training_run.artifact.threshold <= 0.90


def test_threshold_can_go_low_when_churn_is_rare():
    y = np.array([1] * 5 + [0] * 95)
    p = np.array([0.10] * 5 + [0.01] * 95)
    assert optimize_threshold(y, p) <= 0.10


def test_threshold_curve_matches_a_hand_count():
    row = threshold_curve([1, 1, 0, 0], np.array([0.9, 0.2, 0.8, 0.1]))
    row = row[row["threshold"] == 0.50].iloc[0]
    assert row["recall"] == 0.5 and row["precision"] == 0.5


# ── Class weights ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["Logistic Regression", "Random Forest"])
def test_class_weight_reaches_the_final_model_when_churn_is_rare(name):
    df = make_customers(600, churn_share=0.10)
    X = pd.DataFrame({"tenure": df["Tenure"].astype(float), "calls": df["Support Calls"].astype(float)})
    y = (df["Churn"] == "Yes").astype(int)
    profile = {"numeric_cols": ["tenure", "calls"], "categorical_cols": [], "boolean_cols": []}
    res = train_all_models(X, y, profile, [name], len(X), True, None, FAST_BUDGET)
    assert res[name]["model"].named_steps["clf"].class_weight == "balanced"


@pytest.mark.parametrize("is_imbalanced,expected", [(False, 1.0), (True, 3.0)])
def test_xgboost_follows_the_same_weighting_policy(is_imbalanced, expected):
    df = make_customers(400)
    X = pd.DataFrame({"tenure": df["Tenure"].astype(float)})
    y = (df["Churn"] == "Yes").astype(int)
    profile = {"numeric_cols": ["tenure"], "categorical_cols": [], "boolean_cols": []}
    res = train_all_models(X, y, profile, ["XGBoost"], len(X), is_imbalanced, None, FAST_BUDGET)
    assert res["XGBoost"]["model"].named_steps["clf"].scale_pos_weight == expected


# ── Model choice ──────────────────────────────────────────────────────────────

def test_all_five_models_train_without_error(training_run):
    assert len(training_run.results) == 5
    assert not [r["error"] for r in training_run.results.values() if r.get("error")]


def test_the_winner_tops_a_table_built_on_one_basis(training_run):
    table = build_comparison_table(training_run.results)
    assert table.iloc[0]["Model"] == training_run.best_model_name
    scores = table["Best Overall Score"].astype(float).tolist()
    assert scores == sorted(scores, reverse=True)


def test_a_failed_model_goes_last_and_cannot_win():
    results = {
        "A": {"error": "boom", "model": None},
        "B": {"model": object(), "pr_auc": .5, "recall": .5, "roc_auc": .7, "best_overall_score": .54},
    }
    assert pick_best_model(results) == "B"
    assert build_comparison_table(results).iloc[-1]["Model"] == "A"


def test_when_every_model_fails_the_message_says_why():
    with pytest.raises(ChurnAppError, match="boom"):
        pick_best_model({"A": {"error": "boom", "model": None}})


def test_the_real_signal_is_found(training_run):
    assert training_run.holdout["roc_auc"] > 0.8


# ── Scoring the same way every time ───────────────────────────────────────────

def test_a_customer_scores_the_same_alone_or_in_any_file(training_run, customers):
    artifact = training_run.artifact
    new = customers.drop(columns=["Churn"]).head(60).reset_index(drop=True)
    in_file = artifact.predict_proba(new)
    alone = np.array([artifact.predict_proba(new.iloc[[i]])[0] for i in range(len(new))])
    in_small_file = artifact.predict_proba(new.iloc[20:30])
    # Equal apart from floating-point rounding in the last digits
    np.testing.assert_allclose(in_file, alone, rtol=0, atol=1e-9)
    np.testing.assert_allclose(in_file[20:30], in_small_file, rtol=0, atol=1e-9)


def test_scoring_reads_a_file_the_way_training_did(training_run, customers):
    artifact = training_run.artifact
    new = customers.drop(columns=["Churn"]).head(5).copy()
    as_text = new.astype(str)
    np.testing.assert_allclose(artifact.predict_proba(new), artifact.predict_proba(as_text), rtol=0, atol=1e-9)


def test_extra_columns_and_their_order_do_not_matter(training_run, customers):
    artifact = training_run.artifact
    new = customers.head(20)
    expected = artifact.predict_proba(new.drop(columns=["Churn"]))
    np.testing.assert_allclose(expected, artifact.predict_proba(new[new.columns[::-1]]), rtol=0, atol=1e-9)


# ── Refusing what cannot be scored ────────────────────────────────────────────

def test_a_file_from_another_business_is_refused(training_run):
    other = pd.DataFrame({"Balance": [1, 2], "Geography": ["FR", "DE"], "Tenure": [3, 4]})
    report = training_run.artifact.check(other)
    assert report.is_refused
    with pytest.raises(DataValidationError, match="does not look like"):
        training_run.artifact.predict_proba(other)


def test_a_missing_column_needs_explicit_consent(training_run, customers):
    new = customers.drop(columns=["Churn", "Plan"]).head(10)
    report = training_run.artifact.check(new)
    assert report.missing == ["plan"] and not report.is_refused
    with pytest.raises(DataValidationError, match="missing"):
        training_run.artifact.predict_proba(new)
    assert len(training_run.artifact.predict_proba(new, allow_missing=True)) == 10


def test_unreadable_new_values_are_reported_and_do_not_crash(training_run, customers):
    new = customers.drop(columns=["Churn"]).head(20).copy()
    new["Monthly Charges"] = new["Monthly Charges"].astype(object)
    new["Auto Pay"] = new["Auto Pay"].astype(object)
    new.loc[new.index[:10], "Monthly Charges"] = "unknown"
    new.loc[new.index[:10], "Auto Pay"] = "maybe"
    report = training_run.artifact.check(new)
    assert report.unparseable["monthly_charges"] == pytest.approx(0.5)
    assert report.unparseable["auto_pay"] == pytest.approx(0.5)
    assert len(training_run.artifact.predict_proba(new)) == 20


def test_values_far_outside_training_are_reported(training_run, customers):
    new = customers.drop(columns=["Churn"]).head(50).copy()
    new["Monthly Charges"] = new["Monthly Charges"] * 1000
    assert "monthly_charges" in training_run.artifact.check(new).out_of_range


def test_new_categories_are_reported_and_scored(training_run, customers):
    new = customers.drop(columns=["Churn"]).head(10).copy()
    new["Plan"] = "enterprise"
    assert training_run.artifact.check(new).unseen_categories["plan"] == 1.0
    assert len(training_run.artifact.predict_proba(new)) == 10


# ── Refusing what cannot be trained on ────────────────────────────────────────

def test_too_few_rows_is_refused():
    with pytest.raises(DataValidationError, match="at least 100"):
        run_training(make_customers(60), "Churn", id_col="Customer ID", budget=FAST_BUDGET)


@pytest.mark.parametrize("n_churners", [0, 3])
def test_too_few_churners_is_refused_with_a_clear_message(n_churners):
    df = make_customers(400)
    df["Churn"] = "No"
    df.loc[df.index[:n_churners], "Churn"] = "Yes"
    with pytest.raises(DataValidationError, match="same target value|rarer outcome"):
        run_training(df, "Churn", id_col="Customer ID", positive_label="Yes", budget=FAST_BUDGET)


def test_no_usable_columns_is_refused():
    df = pd.DataFrame({"id": range(200), "same": [1] * 200, "churn": [1, 0] * 100})
    with pytest.raises(DataValidationError, match="No usable columns"):
        run_training(df, "churn", id_col="id", budget=FAST_BUDGET)


def test_unlabelled_rows_are_left_out_and_reported():
    df = make_customers(300)
    df.loc[df.index[:5], "Churn"] = None
    run = run_training(df, "Churn", id_col="Customer ID", selected_models=["Logistic Regression"], budget=FAST_BUDGET)
    assert any("5 rows" in w for w in run.warnings)
    assert run.n_train + run.n_holdout == 295


def test_repeated_customers_are_reported():
    df = make_customers(300)
    df.loc[df.index[:30], "Customer ID"] = "C99999"
    run = run_training(df, "Churn", id_col="Customer ID", selected_models=["Logistic Regression"], budget=FAST_BUDGET)
    assert any("repeat a customer ID" in w for w in run.warnings)


# ── Leaks ─────────────────────────────────────────────────────────────────────

def test_a_leak_is_left_out_of_the_model_by_default():
    df = make_customers(500)
    df["status_code"] = (df["Churn"] == "Yes").astype(int)
    run = run_training(df, "Churn", id_col="Customer ID", selected_models=["Logistic Regression"], budget=FAST_BUDGET)
    assert "status_code" not in run.artifact.schema.columns
    assert run.holdout["roc_auc"] < 0.995
    assert any("status_code" in w for w in run.warnings)


def test_the_user_can_keep_a_flagged_column():
    df = make_customers(500)
    df["status_code"] = (df["Churn"] == "Yes").astype(int)
    assert [f.column for f in screen_columns(df, "Churn", "Customer ID")] == ["status_code"]
    run = run_training(df, "Churn", id_col="Customer ID", selected_models=["Logistic Regression"],
                       budget=FAST_BUDGET, exclude_cols=[])
    assert "status_code" in run.artifact.schema.columns
    assert any("Results this good" in w for w in run.warnings)


def test_the_real_columns_of_a_clean_file_are_not_flagged(customers):
    assert screen_columns(customers, "Churn", "Customer ID") == []
