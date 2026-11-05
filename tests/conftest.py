import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402

import src  # noqa: E402
from src.runtime_env import limit_threads  # noqa: E402

limit_threads()

# app.py reloads src when its files differ from what was imported. The tests import src
# themselves, so mark it current, or the app would swap the classes under the tests.
src._source_stamp = hashlib.sha256(
    b"".join(p.read_bytes() for p in sorted(Path(ROOT, "src").glob("*.py")))).digest()

# Small enough for a test run, while still exercising the real search and refit
FAST_BUDGET = {"optuna_trials": 3, "hpo_cv_folds": 3, "hpo_max_rows": 0, "max_iters": 40, "rf_trees": 30}


def make_customers(n=1200, churn_share=0.25, seed=0) -> pd.DataFrame:
    """Customers whose churn really depends on their columns, with a known signal."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n).clip(10).round(2)
    support = rng.poisson(1.2, n)
    plan = rng.choice(["basic", "plus", "premium"], n, p=[0.5, 0.3, 0.2])
    auto_pay = rng.choice(["Yes", "No"], n)
    logit = -0.05 * tenure + 0.02 * monthly + 0.5 * support + (plan == "basic") * 0.6 + (auto_pay == "No") * 0.5
    logit = logit + rng.normal(0, 0.5, n)
    churned = logit >= np.quantile(logit, 1 - churn_share)
    return pd.DataFrame({
        "Customer ID": [f"C{i:05d}" for i in range(n)],
        "Tenure": tenure,
        "Monthly Charges": monthly,
        "Total Charges": (tenure * monthly).round(2),
        "Support Calls": support,
        "Plan": plan,
        "Auto Pay": auto_pay,
        "Churn": np.where(churned, "Yes", "No"),
    })


def upload(data: bytes, name: str = "file.csv") -> io.BytesIO:
    buffer = io.BytesIO(data)
    buffer.name = name
    return buffer


@pytest.fixture(scope="session")
def customers() -> pd.DataFrame:
    return make_customers()


@pytest.fixture(scope="session")
def training_run(customers):
    """One real run through the whole flow, shared because it takes a while."""
    from src.workflow import run_training
    return run_training(customers, "Churn", id_col="Customer ID", budget=FAST_BUDGET)
