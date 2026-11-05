"""Drives the real app.py through a real file upload, as a user would."""
import os

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
    assert not at.exception, [e.value for e in at.exception]


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


def test_home_page_renders():
    at = start_app()
    assert "Churn Analytics Studio" in at.title[0].value


def test_upload_then_training_shows_results_on_unseen_customers(trained_app):
    metrics = {m.label: m.value for m in trained_app.metric}
    assert metrics["🏆 Best Model"] == "Logistic Regression"
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
