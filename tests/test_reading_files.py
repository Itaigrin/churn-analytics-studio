"""Real customer files: encodings, number formats, broken tables, target spellings."""
import numpy as np
import pandas as pd
import pytest

from conftest import upload
from src.data_loader import load_uploaded_file
from src.errors import DataFileError, DataValidationError
from src.pipeline_builder import convert_target, raw_clean, suggest_positive_label
from src.utils import to_csv_bytes


def test_hebrew_csv_saved_by_excel_keeps_its_letters():
    df = pd.DataFrame({"עיר": ["תל אביב", "חיפה"] * 30, "נטש": [1, 0] * 30})
    loaded = load_uploaded_file(upload(df.to_csv(index=False).encode("cp1255")))
    assert list(loaded.columns) == ["עיר", "נטש"]
    assert loaded.iloc[0, 0] == "תל אביב"


def test_western_accents_are_not_mistaken_for_hebrew():
    data = "ville,churn\nMontréal,1\nZürich,0\nCafé,1\n".encode("cp1252")
    assert load_uploaded_file(upload(data))["ville"].tolist() == ["Montréal", "Zürich", "Café"]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_unicode_files_load(encoding):
    data = "city,churn\nתל אביב,1\nחיפה,0\n".encode(encoding)
    loaded = load_uploaded_file(upload(data))
    assert list(loaded.columns) == ["city", "churn"]
    assert loaded.iloc[0, 0] == "תל אביב"


def test_malformed_csv_gets_a_readable_error():
    with pytest.raises(DataFileError, match="not a well-formed table"):
        load_uploaded_file(upload(b"a,b\n1,2\n4,5,6,7,8\n"))


def test_empty_file_gets_a_readable_error():
    with pytest.raises(DataFileError, match="empty"):
        load_uploaded_file(upload(b"   "))


def test_semicolon_delimiter_is_detected():
    loaded = load_uploaded_file(upload(b"id;amount;churn\n1;12,5;1\n2;3,5;0\n"))
    assert list(loaded.columns) == ["id", "amount", "churn"]


@pytest.mark.parametrize("raw,expected", [
    (["1.234,50", "12,5", "7,25"], [1234.5, 12.5, 7.25]),
    (["1,234.50", "12.5", "7.25"], [1234.5, 12.5, 7.25]),
    (["1,200", "3,400", "50"], [1200.0, 3400.0, 50.0]),
    (["$1,200", "$300", "$45"], [1200.0, 300.0, 45.0]),
    (["₪1,200", "₪300", "₪45"], [1200.0, 300.0, 45.0]),
    (["12%", "5%", "7%"], [12.0, 5.0, 7.0]),
    (["(100)", "50", "20"], [-100.0, 50.0, 20.0]),
])
def test_number_formats_are_read_correctly(raw, expected):
    assert raw_clean(pd.DataFrame({"x": raw}))["x"].tolist() == expected


def test_blank_values_in_a_numeric_column_are_missing_not_text():
    cleaned = raw_clean(pd.DataFrame({"total": [" ", "10.5", "11", "12"]}))["total"]
    assert np.isnan(cleaned[0]) and cleaned[1] == 10.5


def test_mostly_text_column_stays_text():
    cleaned = raw_clean(pd.DataFrame({"plan": ["a", "b", "1", "c"]}))["plan"]
    assert cleaned.tolist() == ["a", "b", "1", "c"]


def test_yes_no_columns_become_flags():
    assert raw_clean(pd.DataFrame({"x": ["Yes", "no", "YES"]}))["x"].tolist() == [1.0, 0.0, 1.0]


def test_colliding_column_names_get_a_readable_error():
    df = pd.DataFrame({"Total Charges": [1.0], "total_charges": [2.0]})
    with pytest.raises(DataValidationError, match="same name"):
        raw_clean(df)


@pytest.mark.parametrize("values,churned_value", [
    (["Yes", "No"], "Yes"),
    ([1.0, 0.0], 1.0),
    (["Attrited Customer", "Existing Customer"], "Attrited Customer"),
    (["Churned", "Stayed"], "Churned"),
    (["כן", "לא"], "כן"),
])
def test_common_target_spellings_are_understood(values, churned_value):
    s = pd.Series(values * 5, name="target")
    assert suggest_positive_label(s, "target") == (churned_value, True)
    assert convert_target(s).sum() == 5


def test_codes_without_meaning_require_a_choice():
    s = pd.Series([1, 1, 1, 2], name="status")
    label, confident = suggest_positive_label(s, "status")
    assert label == 2 and confident is False
    with pytest.raises(DataValidationError, match="Choose it explicitly"):
        convert_target(s)
    assert convert_target(s, positive_label=1).tolist() == [1.0, 1.0, 1.0, 0.0]


def test_retained_column_means_the_opposite():
    assert suggest_positive_label(pd.Series([1, 0, 1], name="retained"), "retained")[0] == 0


def test_missing_target_values_stay_missing():
    assert np.isnan(convert_target(pd.Series([1.0, 0.0, None]))[2])


def test_target_with_many_values_is_refused():
    with pytest.raises(DataValidationError, match="different values"):
        convert_target(pd.Series(["a", "b", "c"]))


def test_download_opens_in_excel_with_hebrew_and_cannot_run_formulas():
    data = to_csv_bytes(pd.DataFrame({"name": ["תל אביב", "=HYPERLINK(1)"], "n": [-1, 2]}))
    assert data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8-sig")
    assert "תל אביב" in text and "'=HYPERLINK(1)" in text and "\n=" not in text
    assert ",-1" in text
