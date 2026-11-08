"""
Turning churn probabilities into money.

Three numbers from the business are enough:
  value    what a customer who stays is worth (for example a year of margin)
  cost     what one retention contact costs (a call, a discount, staff time)
  success  the share of contacted churners the contact actually keeps

Contacting a customer with churn probability p is then worth p · success · value − cost,
so it pays exactly when p is above cost / (success · value). That rule needs
probabilities that mean what they say, which is why the models are calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PER_CUSTOMERS = 1_000


@dataclass
class Economics:
    value: float
    cost: float
    success: float

    @property
    def breakeven(self) -> float:
        """The churn probability above which a contact pays for itself."""
        return self.cost / (self.success * self.value)

    def gain(self, probabilities: np.ndarray, values=None) -> np.ndarray:
        """
        Expected profit of contacting each customer. values, when given, is each
        customer's own worth; without it every customer is worth the same.
        """
        worth = self.value if values is None else np.nan_to_num(np.asarray(values, dtype=float))
        return probabilities * self.success * worth - self.cost


def campaign_estimate(curve: pd.DataFrame, churn_rate: float, economics: Economics,
                      per: int = PER_CUSTOMERS) -> pd.DataFrame:
    """
    What a campaign at each threshold earned on customers the model had not seen,
    per `per` customers. Built from the holdout's recall and share flagged.
    """
    reached = curve["recall"] * churn_rate * per
    contacted = curve["flagged"] * per
    saved = reached * economics.success
    return pd.DataFrame({
        "threshold": curve["threshold"],
        "contacted": contacted,
        "churners_reached": reached,
        "customers_saved": saved,
        "net": saved * economics.value - contacted * economics.cost,
    })


def contact_everyone(churn_rate: float, economics: Economics, per: int = PER_CUSTOMERS) -> float:
    return churn_rate * per * economics.success * economics.value - per * economics.cost


def file_estimate(probabilities: np.ndarray, contacted: np.ndarray, economics: Economics, values=None) -> dict:
    """Expected outcome of contacting the customers marked in `contacted`."""
    expected_churners = float(probabilities[contacted].sum())
    gains = economics.gain(probabilities, values)
    return {
        "contacted": int(contacted.sum()),
        "expected_churners_reached": expected_churners,
        "share_of_expected_churners": expected_churners / max(float(probabilities.sum()), 1e-12),
        "expected_saved": expected_churners * economics.success,
        "cost": float(contacted.sum() * economics.cost),
        "net": float(gains[contacted].sum()),
    }


def risk_levels(probabilities: np.ndarray, flagged: np.ndarray, churn_rate: float) -> np.ndarray:
    """
    High: flagged for contact. Medium: riskier than the average customer, but not flagged.
    Low: no riskier than the average customer.
    """
    return np.where(flagged, "High", np.where(probabilities >= churn_rate, "Medium", "Low"))


def lift_table(y_true, y_prob, groups: int = 10) -> pd.DataFrame:
    """
    Customers ranked by risk and cut into equal groups. Shows how many of the churners
    the riskiest few percent contain, which is what a campaign with a fixed budget gets.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_prob), kind="stable")
    sorted_y = y_true[order]
    edges = np.linspace(0, len(sorted_y), groups + 1).round().astype(int)
    total_churners = max(int(y_true.sum()), 1)
    base_rate = y_true.mean()
    rows = []
    for i in range(groups):
        group = sorted_y[edges[i]:edges[i + 1]]
        caught = sorted_y[:edges[i + 1]].sum()
        rows.append({
            "Riskiest": f"{(i + 1) * 100 // groups}%",
            "Customers": len(group),
            "Churn Rate": group.mean() if len(group) else 0.0,
            "Lift": (group.mean() / base_rate) if len(group) and base_rate else 0.0,
            "Churners Caught So Far": caught / total_churners,
        })
    return pd.DataFrame(rows)
