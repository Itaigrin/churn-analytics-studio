"""Drives the real app.py through a real file upload, as a user would."""
import os
import time

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT, make_customers

APP = os.path.join(ROOT, "app.py")
TIMEOUT = 600


def start_app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    return at


def upload_file(at, index, data: bytes, name="customers.csv"):
    at.file_uploader[index].upload(name, data, "text/csv")
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def run_pipeline(at):
    for box in at.multiselect:
        if box.label == "Models to train":
            box.set_value(["Logistic Regression"])
    for box in at.selectbox:
        if box.label == "Search depth":
            box.select("Quick")
    at.run()
    next(b for b in at.button if "Run Pipeline" in b.label).click()
    at.run()
    wait_for_training(at)
    assert not at.exception, [e.value for e in at.exception]


def training_is_running(at) -> bool:
    return any("Running Pipeline" in m.value for m in at.markdown)


def wait_for_training(at, timeout=TIMEOUT):
    """Training runs as a server job, so the page has to be asked again until it is done."""
    deadline = time.time() + timeout
    while training_is_running(at):
        assert time.time() < deadline, "training did not finish"
        time.sleep(0.5)
        at.run()


def text_of(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [w.value for w in at.warning]
    return "\n".join(parts)


@pytest.fixture(scope="module")
def trained_app():
    at = start_app()
    upload_file(at, 0, make_customers(500).to_csv(index=False).encode())
    run_pipeline(at)
    assert not at.error, [e.value for e in at.error]
    return at


def _uploader(at, label):
    return next(u for u in at.file_uploader if u.label.startswith(label))


def _downloads(at):
    return [d.proto.label for d in at.get("download_button")]


def test_flagged_customers_show_their_reasons(trained_app):
    new = make_customers(50, seed=5).drop(columns=["Churn"])
    new["Tenure"] = 2
    new["Support Calls"] = 6
    upload_file(trained_app, 1, new.to_csv(index=False).encode(), "risky.csv")
    next(b for b in trained_app.button if "Predict Churn" in b.label).click()
    trained_app.run()
    table = next(d.value for d in trained_app.dataframe if "Churn Probability" in d.value.columns)
    flagged = table[table["Flagged At Risk"] == "Yes"]
    assert len(flagged) > 0
    assert flagged["Reason 1"].str.contains("pts").any()
    assert (table.loc[table["Flagged At Risk"] == "No", "Reason 1"] == "").all()


def test_a_model_saved_in_the_app_works_after_a_restart(trained_app, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_KEY", "app-test-key")
    trained_app.run()
    next(b for b in trained_app.button if b.label == "Prepare Model File").click()
    trained_app.run()
    assert not trained_app.exception, [e.value for e in trained_app.exception]
    assert "⬇️ Download Model File" in _downloads(trained_app)
    model_file = trained_app.session_state.model_file

    fresh = start_app()
    _uploader(fresh, "Model file").upload("churn.churnmodel", model_file, "application/octet-stream")
    fresh.run()
    assert not fresh.exception, [e.value for e in fresh.exception]
    assert any("Loaded **Logistic Regression**" in s.value for s in fresh.success)

    new = make_customers(50, seed=5).drop(columns=["Churn"])
    _uploader(fresh, "Upload new customers").upload("new.csv", new.to_csv(index=False).encode(), "text/csv")
    fresh.run()
    next(b for b in fresh.button if "Predict Churn" in b.label).click()
    fresh.run()
    assert not fresh.exception, [e.value for e in fresh.exception]
    assert {m.label: m.value for m in fresh.metric}["Total Customers"] == "50"
    table = next(d.value for d in fresh.dataframe if "Churn Probability" in d.value.columns)
    assert len(table) == 50


def test_a_forged_model_file_is_refused_in_the_page():
    at = start_app()
    _uploader(at, "Model file").upload("evil.churnmodel", b"not a model at all", "application/octet-stream")
    at.run()
    assert not at.exception
    assert any("not a model file" in e.value for e in at.error)


def test_without_a_key_on_a_hosted_app_saving_is_off(monkeypatch):
    from src import model_store
    monkeypatch.delenv("CHURN_MODEL_KEY", raising=False)
    monkeypatch.setattr(model_store, "ON_CLOUD", True)
    at = start_app()
    assert any("Saving models is turned off" in i.value for i in at.info)
    assert len(at.file_uploader) == 1


def test_results_show_who_to_contact_first(trained_app):
    assert "Who To Contact First" in text_of(trained_app)
    lift = next(d.value for d in trained_app.dataframe if "Lift" in d.value.columns)
    assert len(lift) == 10


def test_the_profit_goal_sets_the_breakeven_threshold_and_shows_the_money(trained_app):
    new = make_customers(60, seed=4).drop(columns=["Churn"])
    upload_file(trained_app, 1, new.to_csv(index=False).encode(), "money.csv")
    goal = next(r for r in trained_app.radio if r.label == "Choose who gets flagged by")
    goal.set_value("Most profit (your numbers above)")
    trained_app.run()
    # Defaults: cost 50 / (25% kept x value 1,000) = 0.20
    assert any(s.label == "Decision Threshold" and s.value == pytest.approx(0.20) for s in trained_app.slider)
    assert "per 1,000 customers like the unseen ones" in text_of(trained_app)
    next(b for b in trained_app.button if "Predict Churn" in b.label).click()
    trained_app.run()
    assert not trained_app.exception, [e.value for e in trained_app.exception]
    labels = {m.label for m in trained_app.metric}
    assert {"Expected Customers Kept", "Expected Net Result"} <= labels
    table = next(d.value for d in trained_app.dataframe if "Churn Probability" in d.value.columns)
    expected_gain = table["Churn Probability"] * 0.25 * 1000 - 50
    # The probability shown is rounded to 4 decimals, so allow for that
    assert (table["Expected Gain From Contact"] - expected_gain).abs().max() < 0.02


def test_dates_can_be_used_and_the_model_tested_on_recent_customers():
    from test_dates_and_splits import dated_customers
    df = dated_customers(400)
    df["Joined"] = (pd.Timestamp("2023-01-01") + pd.to_timedelta(range(len(df)), "D")).strftime("%Y-%m-%d")
    at = start_app()
    upload_file(at, 0, df.to_csv(index=False).encode())
    assert "Date columns found" in text_of(at)
    next(r for r in at.radio if r.label == "Test the model on").set_value("The most recent customers")
    at.run()
    next(b for b in at.selectbox if b.label == "Date that orders the customers").select("Joined")
    run_pipeline(at)
    assert not at.error, [e.value for e in at.error]
    assert "most recent customers by" in text_of(at)

    new = dated_customers(30, seed=7).drop(columns=["Churn"])
    new["Joined"] = "2024-06-01"
    upload_file(at, 1, new.to_csv(index=False).encode(), "new.csv")
    assert any(d.label == "📅 Count days back from" for d in at.date_input)
    next(b for b in at.button if "Predict Churn" in b.label).click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert {m.label: m.value for m in at.metric}["Total Customers"] == "30"


def test_scoring_shows_whether_the_data_has_changed(trained_app):
    new = make_customers(300, seed=12).drop(columns=["Churn"])
    upload_file(trained_app, 1, new.to_csv(index=False).encode(), "same.csv")
    next(b for b in trained_app.button if "Predict Churn" in b.label).click()
    trained_app.run()
    assert "Has The Data Changed Since Training?" in text_of(trained_app)
    assert not any("train the model again" in w.value for w in trained_app.warning)

    shifted = new.assign(Tenure=70)
    shifted["Support Calls"] = shifted["Support Calls"] + 6
    upload_file(trained_app, 1, shifted.to_csv(index=False).encode(), "shifted.csv")
    next(b for b in trained_app.button if "Predict Churn" in b.label).click()
    trained_app.run()
    assert any("train the model again" in w.value for w in trained_app.warning)


def test_past_predictions_are_checked_against_what_happened(trained_app):
    later = make_customers(600, seed=31)
    _uploader(trained_app, "Customers with their real outcome").upload(
        "outcomes.csv", later.to_csv(index=False).encode(), "text/csv")
    trained_app.run()
    # Auto Pay also has two values and comes first; the column trained on must win
    assert next(b for b in trained_app.selectbox if b.label == "Column with what really happened").value == "Churn"
    next(b for b in trained_app.button if b.label == "📊 Check The Model").click()
    trained_app.run()
    assert not trained_app.exception, [e.value for e in trained_app.exception]
    assert "Predicted vs Actual Churn" in {m.label for m in trained_app.metric}
    assert any("holding up" in s.value for s in trained_app.success)


def test_results_show_quality_lift_and_drivers(trained_app):
    labels = {m.label for m in trained_app.metric}
    assert {"PR-AUC", "Top 10% Lift", "Calibration Error"} <= labels
    assert any(label.startswith("Prediction Quality · ") for label in labels)
    assert "Best Overall Score" not in labels
    assert "What Drives Churn" in text_of(trained_app)


def test_flagging_by_how_many_can_be_contacted(trained_app):
    new = make_customers(200, seed=13).drop(columns=["Churn"])
    upload_file(trained_app, 1, new.to_csv(index=False).encode(), "capacity.csv")
    next(r for r in trained_app.radio if r.label == "Choose who gets flagged by").set_value(
        "How many customers I can contact")
    trained_app.run()
    next(n for n in trained_app.number_input if n.label == "Customers you can contact").set_value(25)
    trained_app.run()
    next(b for b in trained_app.button if "Predict Churn" in b.label).click()
    trained_app.run()
    assert not trained_app.exception, [e.value for e in trained_app.exception]
    assert {m.label: m.value for m in trained_app.metric}["Flagged At Risk"] == "25"
    table = next(d.value for d in trained_app.dataframe if "Churn Probability" in d.value.columns)
    flagged = table[table["Flagged At Risk"] == "Yes"]
    assert flagged["Churn Probability"].min() >= table.loc[table["Flagged At Risk"] == "No", "Churn Probability"].max()
    assert set(table["Risk Level"]) <= {"High", "Medium", "Low"} and (flagged["Risk Level"] == "High").all()
    assert table["Risk Score"].between(0, 100).all()


def test_customer_value_weighs_risk_in_money(trained_app):
    new = make_customers(200, seed=14).drop(columns=["Churn"])
    upload_file(trained_app, 1, new.to_csv(index=False).encode(), "value.csv")
    next(r for r in trained_app.radio if r.label == "Choose who gets flagged by").set_value(
        "Most profit (your numbers above)")
    trained_app.run()
    next(b for b in trained_app.selectbox if b.label == "Column with each customer's value (optional)").select(
        "Total Charges")
    trained_app.run()
    next(b for b in trained_app.button if "Predict Churn" in b.label).click()
    trained_app.run()
    assert not trained_app.exception, [e.value for e in trained_app.exception]
    table = next(d.value for d in trained_app.dataframe if "Churn Probability" in d.value.columns)
    assert "Value At Risk" in table.columns
    gain_positive = table["Expected Gain From Contact"] > 0
    assert (table.loc[gain_positive, "Flagged At Risk"] == "Yes").all()
    assert "Value At Risk In This File" in {m.label for m in trained_app.metric}


def test_training_survives_a_lost_connection():
    at = start_app()
    upload_file(at, 0, make_customers(400).to_csv(index=False).encode())
    for box in at.multiselect:
        if box.label == "Models to train":
            box.set_value(["Logistic Regression"])
    for box in at.selectbox:
        if box.label == "Search depth":
            box.select("Quick")
    at.run()
    next(b for b in at.button if "Run Pipeline" in b.label).click()
    at.run()
    job_id = at.query_params["job"]
    assert job_id and training_is_running(at)

    # The browser reconnects: a brand new session, without the uploaded file, same address
    reconnected = AppTest.from_file(APP, default_timeout=TIMEOUT)
    reconnected.query_params["job"] = job_id
    reconnected.run()
    assert not reconnected.exception, [e.value for e in reconnected.exception]
    wait_for_training(reconnected)
    assert "Best model: Logistic Regression" in text_of(reconnected)
    assert any("not loaded in this session any more" in i.value for i in reconnected.info)
    # The model can still score customers in the new session
    new = make_customers(30, seed=5).drop(columns=["Churn"])
    reconnected.file_uploader[1].upload("new.csv", new.to_csv(index=False).encode(), "text/csv")
    reconnected.run()
    next(b for b in reconnected.button if "Predict Churn" in b.label).click()
    reconnected.run()
    assert not reconnected.exception, [e.value for e in reconnected.exception]
    assert {m.label: m.value for m in reconnected.metric}["Total Customers"] == "30"


def test_an_unknown_job_address_just_shows_the_start_page():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["job"] = "no-such-job"
    at.run()
    assert not at.exception
    assert any("Upload a dataset to begin" in i.value for i in at.info)


def test_a_training_error_is_shown_after_the_job_ends():
    at = start_app()
    upload_file(at, 0, make_customers(60).to_csv(index=False).encode())
    run_pipeline(at)
    assert any("at least 100" in e.value for e in at.error)


def test_the_two_sets_of_numbers_are_explained_side_by_side(trained_app):
    note = next(i.value for i in trained_app.info if "different numbers" in i.value)
    run = trained_app.session_state.run
    oof = run.results[run.best_model_name]
    assert f"{oof['pr_auc']:.4f}" in note and f"{run.holdout['pr_auc']:.4f}" in note
    assert f"{run.n_holdout:,} new customers" in text_of(trained_app)
    assert "on the training customers, 5-fold" in text_of(trained_app)


def test_home_page_renders():
    at = start_app()
    assert "Churn Analytics Studio" in at.title[0].value


def test_upload_then_training_shows_results_on_unseen_customers(trained_app):
    metrics = {m.label: m.value for m in trained_app.metric}
    assert "Best model: Logistic Regression" in text_of(trained_app)
    assert float(metrics["ROC-AUC"]) > 0.7
    assert "held back" in text_of(trained_app)


def test_the_app_no_longer_promises_zero_leakage(trained_app):
    assert "zero data leakage" not in text_of(trained_app).lower()


def test_predicting_new_customers_end_to_end(trained_app):
    new = make_customers(50, seed=5).drop(columns=["Churn"])
    upload_file(trained_app, 1, new.to_csv(index=False).encode(), "new.csv")
    next(b for b in trained_app.button if "Predict Churn" in b.label).click()
    trained_app.run()
    assert not trained_app.exception
    metrics = {m.label: m.value for m in trained_app.metric}
    assert metrics["Total Customers"] == "50"
    assert "Flagged At Risk" in metrics and "Will Churn" not in metrics
    table = next(d.value for d in trained_app.dataframe if "Churn Probability" in d.value.columns)
    assert len(table) == 50


def test_a_tiny_file_is_not_compared_with_history(trained_app):
    new = make_customers(5, seed=6).drop(columns=["Churn"])
    upload_file(trained_app, 1, new.to_csv(index=False).encode(), "tiny.csv")
    next(b for b in trained_app.button if "Predict Churn" in b.label).click()
    trained_app.run()
    assert any("too few to compare" in i.value for i in trained_app.info)


def test_a_file_from_another_business_is_refused_in_the_page(trained_app):
    wrong = b"Balance,Geography,Age\n1,FR,30\n2,DE,40\n"
    upload_file(trained_app, 1, wrong, "wrong.csv")
    assert any("does not look like" in e.value for e in trained_app.error)
    assert not [b for b in trained_app.button if "Predict Churn" in b.label]


def test_a_new_training_file_clears_the_old_results():
    at = start_app()
    upload_file(at, 0, make_customers(300).to_csv(index=False).encode())
    run_pipeline(at)
    assert at.session_state.run is not None
    upload_file(at, 0, make_customers(300, seed=9).to_csv(index=False).encode(), "other.csv")
    assert at.session_state.run is None


def test_too_small_a_file_gets_a_clear_message():
    at = start_app()
    upload_file(at, 0, make_customers(60).to_csv(index=False).encode())
    run_pipeline(at)
    assert any("at least 100" in e.value for e in at.error)


def test_broken_file_gets_a_clear_message():
    at = start_app()
    upload_file(at, 0, b"a,b\n1,2\n4,5,6,7,8\n")
    assert any("not a well-formed table" in e.value for e in at.error)


def test_colliding_column_names_get_a_clear_message():
    at = start_app()
    upload_file(at, 0, b"Total Charges,total_charges,churn\n" + b"1.5,2.5,1\n3.5,4.5,0\n" * 60)
    assert any("same name" in e.value for e in at.error)


def test_a_flagged_leak_is_shown_and_left_out_by_default():
    df = make_customers(400)
    df["status_code"] = (df["Churn"] == "Yes").astype(int)
    at = start_app()
    upload_file(at, 0, df.to_csv(index=False).encode())
    assert "Possible data leaks" in text_of(at)
    box = next(b for b in at.multiselect if b.label == "Columns to leave out of the model")
    assert box.value == ["status_code"]


def test_undetected_target_says_so_instead_of_claiming_a_detection():
    df = make_customers(300).rename(columns={"Churn": "Outcome"})
    at = start_app()
    upload_file(at, 0, df.to_csv(index=False).encode())
    box = next(b for b in at.selectbox if b.label == "Churn / Target column ★")
    assert box.value == "Auto Pay"
    assert "Could not detect it" in text_of(at)


def test_unrecognised_target_codes_ask_the_user():
    df = make_customers(400)
    df["Churn"] = (df["Churn"] == "Yes").map({True: 2, False: 1})
    at = start_app()
    upload_file(at, 0, df.to_csv(index=False).encode())
    box = next(b for b in at.selectbox if b.label == "Which value means the customer left?")
    assert box.value == "2"
    assert "only a guess" in text_of(at)
