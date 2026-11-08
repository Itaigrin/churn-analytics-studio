"""The reasons shown for each customer."""
import numpy as np
import pandas as pd
import pytest

from conftest import FAST_BUDGET, make_customers
from src.explain import MAX_CUSTOMERS, ColumnReference, Reason, build_reference, column_effects, format_reason, top_reasons
from src.feature_engineering import FeatureSpec
from src.workflow import run_training


def _risky_new_customers():
    new = make_customers(40, seed=3).drop(columns=["Churn"])
    new["Tenure"] = 2
    new["Support Calls"] = 6
    return new


def test_the_cause_built_into_the_data_is_named_first(training_run):
    new = _risky_new_customers()
    p = training_run.artifact.predict_proba(new)
    reasons = training_run.artifact.explain(new, p)
    leading = [r[0].column for r in reasons if r]
    assert leading
    share = sum(col in {"tenure", "support_calls"} for col in leading) / len(leading)
    assert share >= 0.8


def test_a_reason_really_moves_the_risk_by_what_it_claims(training_run):
    artifact = training_run.artifact
    new = _risky_new_customers().reset_index(drop=True)
    p = artifact.predict_proba(new)
    reasons = artifact.explain(new, p)
    customer = next(i for i, r in enumerate(reasons) if r)
    first = reasons[customer][0]
    original_name = {"tenure": "Tenure", "support_calls": "Support Calls", "monthly_charges": "Monthly Charges",
                     "total_charges": "Total Charges", "plan": "Plan", "auto_pay": "Auto Pay"}[first.column]
    ref = artifact.reference[first.column]
    expected = 0.0
    for value, weight in zip(ref.values, ref.weights):
        changed = new.iloc[[customer]].copy()
        changed[original_name] = value
        expected += weight * artifact.predict_proba(changed)[0]
    assert p[customer] - expected == pytest.approx(first.effect, abs=1e-9)


def test_the_most_common_value_can_still_be_the_reason():
    # Nearly everyone is on "monthly", and it is what drives the risk. Compared with a single
    # typical value this reason would be invisible.
    X = pd.DataFrame({"contract": ["monthly"]})
    reference = {"contract": ColumnReference(["monthly", "yearly"], np.array([0.8, 0.2]))}
    risk = {"monthly": 0.5, "yearly": 0.1}
    predict = lambda features: features["contract"].map(risk).to_numpy()
    effects = column_effects(X, reference, FeatureSpec(), ["contract"], predict, np.array([0.5]))
    assert effects["contract"].iloc[0] == pytest.approx(0.08)
    assert top_reasons(effects, X, reference)[0][0].column == "contract"


def test_reasons_only_raise_risk_and_are_sorted(training_run):
    new = _risky_new_customers()
    p = training_run.artifact.predict_proba(new)
    for customer in training_run.artifact.explain(new, p):
        effects = [r.effect for r in customer]
        assert effects == sorted(effects, reverse=True)
        assert all(e >= 0.01 for e in effects) and len(customer) <= 3


def test_a_customer_gets_the_same_reasons_alone_or_in_a_file(training_run):
    artifact = training_run.artifact
    new = _risky_new_customers().reset_index(drop=True)
    in_file = artifact.explain(new, artifact.predict_proba(new))[7]
    single = new.iloc[[7]]
    alone = artifact.explain(single, artifact.predict_proba(single))[0]
    assert [r.column for r in in_file] == [r.column for r in alone]
    np.testing.assert_allclose([r.effect for r in in_file], [r.effect for r in alone], atol=1e-9)


def test_a_column_the_model_ignores_is_never_a_reason():
    X = pd.DataFrame({"a": [1.0, 5.0], "b": ["x", "y"]})
    reference = {"a": ColumnReference([1.0, 5.0], np.array([0.5, 0.5]), 3.0),
                 "b": ColumnReference(["x", "y"], np.array([0.5, 0.5]))}
    effects = column_effects(X, reference, FeatureSpec(), ["a", "b"], lambda f: np.full(len(f), 0.3), np.full(2, 0.3))
    assert top_reasons(effects, X, reference) == [[], []]


def test_only_the_riskiest_customers_are_explained_in_a_huge_file(training_run):
    artifact = training_run.artifact
    new = make_customers(MAX_CUSTOMERS + 50, seed=8).drop(columns=["Churn"])
    p = artifact.predict_proba(new)
    reasons = artifact.explain(new, p)
    assert len(reasons) == len(new)
    assert all(not reasons[i] for i in np.argsort(-p)[MAX_CUSTOMERS:])


def test_reason_text_is_readable():
    assert format_reason(Reason("support_calls", 5.0, 1.0, 0.18)) == "support calls: 5 (median 1), +18 pts"
    assert format_reason(Reason("plan", "basic", None, 0.052)) == "plan: basic, +5 pts"
    assert format_reason(Reason("fee", float("nan"), 12.5, 0.1)) == "fee: missing (median 12.5), +10 pts"


def test_the_reference_describes_training_without_keeping_rows():
    X = pd.DataFrame({"n": [float(i) for i in range(100)], "c": ["a"] * 75 + ["b"] * 25, "f": [1.0] * 100})
    reference = build_reference(X, ["n"])
    assert reference["n"].median == 49.5
    assert len(reference["n"].values) == 7 and sum(reference["n"].weights) == pytest.approx(1.0)
    assert reference["c"].values == ["a", "b"] and reference["c"].weights.tolist() == [0.75, 0.25]
    assert reference["f"].median is None


def test_reasons_work_with_a_missing_column(customers):
    run = run_training(customers.head(400), "Churn", id_col="Customer ID",
                       selected_models=["Logistic Regression"], budget=FAST_BUDGET)
    new = customers.drop(columns=["Churn", "Plan"]).head(10)
    p = run.artifact.predict_proba(new, allow_missing=True)
    assert len(run.artifact.explain(new, p)) == 10


def test_a_count_risky_at_both_ends_names_its_riskiest_value():
    from src.explain import overall_drivers
    X = pd.DataFrame({"products": [1.0, 2.0, 3.0, 4.0] * 50})
    effects = pd.DataFrame({"products": X["products"].map({1.0: 0.05, 2.0: -0.10, 3.0: 0.20, 4.0: 0.30})})
    drivers = overall_drivers(effects, X, ["products"], [])
    assert drivers.loc[0, "Raises Risk"] == "4"
