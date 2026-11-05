"""Which columns are kept, dropped or flagged, and why."""
import numpy as np
import pandas as pd

from src.leakage import scan_for_leakage
from src.pipeline_builder import raw_clean
from src.profiler import detect_id_column, detect_target_column, profile_dataset


def profile(df):
    return profile_dataset(raw_clean(df), target_col=None)


def test_continuous_all_unique_column_is_a_feature_not_an_id():
    rng = np.random.default_rng(0)
    p = profile(pd.DataFrame({"salary": rng.uniform(20_000, 90_000, 500)}))
    assert "salary" in p["numeric_cols"]


def test_row_numbers_are_ids():
    p = profile(pd.DataFrame({"rownumber": range(1, 501), "x": [1.5, 2.5] * 250}))
    assert "rownumber" in p["drop_cols"]


def test_fixed_length_codes_are_ids():
    rng = np.random.default_rng(0)
    codes = rng.choice(np.arange(15_000_000, 16_000_000), 500, replace=False)
    assert "code" in profile(pd.DataFrame({"code": codes}))["drop_cols"]


def test_uneven_whole_numbers_are_features():
    rng = np.random.default_rng(0)
    amounts = rng.choice(np.arange(1_000, 500_000), 500, replace=False)
    assert "amount" in profile(pd.DataFrame({"amount": amounts}))["numeric_cols"]


def test_personal_data_never_becomes_an_input():
    p = profile(pd.DataFrame({"surname": ["Cohen", "Levi", "Mizrahi"] * 100, "x": [1.5, 2.5, 3.5] * 100}))
    assert "surname" in p["drop_cols"]
    assert "surname" not in p["categorical_cols"]


def test_iso_dates_are_dates_not_phone_numbers():
    days = pd.date_range("2024-01-01", periods=300).strftime("%Y-%m-%d").tolist()
    p = profile(pd.DataFrame({"last_login": days}))
    assert p["columns"]["last_login"]["detected_type"] == "datetime"
    assert p["columns"]["last_login"]["is_phone"] is False
    assert any("last_login" in w for w in p["general_warnings"])


def test_phone_numbers_are_still_identifiers():
    phones = [f"050-{i:07d}" for i in range(300)]
    assert profile(pd.DataFrame({"contact": phones}))["columns"]["contact"]["is_phone"] is True


def test_rare_flag_with_enough_rows_is_kept():
    flag = np.zeros(2000)
    flag[:30] = 1
    assert "complaint" in profile(pd.DataFrame({"complaint": flag}))["boolean_cols"]


def test_flag_seen_only_a_few_times_is_dropped():
    flag = np.zeros(2000)
    flag[:3] = 1
    assert "complaint" in profile(pd.DataFrame({"complaint": flag}))["drop_cols"]


def test_text_id_column_is_detected_under_current_pandas():
    df = pd.DataFrame({"customer_code": [f"X{i}" for i in range(50)], "a": range(50)})
    assert detect_id_column(df) == "customer_code"


def test_hebrew_churn_column_is_detected():
    df = pd.DataFrame({"customer_id": range(6), "עיר": list("אבגדהו"), "נטש": ["כן", "לא"] * 3})
    assert detect_target_column(df, exclude=["customer_id"]) == "נטש"


def test_single_hebrew_yes_no_column_is_detected_whatever_its_name():
    df = pd.DataFrame({"a": [1.5, 2.5, 3.5] * 2, "סטטוס": ["כן", "לא"] * 3})
    assert detect_target_column(df) == "סטטוס"


def _leak_data():
    rng = np.random.default_rng(0)
    y = pd.Series((rng.random(1000) < 0.25).astype(int))
    return pd.DataFrame({
        "honest": rng.normal(size=1000) + y * 0.5,
        "account_status_code": np.where(rng.random(1000) < 0.99, y, 1 - y),
        "exit_note": np.where(y == 1, rng.choice(["price", "moved"], 1000), None),
        "churn_score": rng.normal(size=1000),
    }), y


def test_innocently_named_copy_of_the_answer_is_flagged():
    X, y = _leak_data()
    found = {f.column: f for f in scan_for_leakage(X, y, ["honest", "account_status_code"])}
    assert set(found) == {"account_status_code"}
    assert found["account_status_code"].auc > 0.95


def test_column_filled_only_for_leavers_is_flagged():
    X, y = _leak_data()
    assert [f.column for f in scan_for_leakage(X, y, ["exit_note"])] == ["exit_note"]


def test_suspicious_name_is_flagged_even_without_signal():
    X, y = _leak_data()
    assert [f.column for f in scan_for_leakage(X, y, ["churn_score"])] == ["churn_score"]
