"""
The frozen description of the data a model was trained on.

conform() turns a raw file into the exact columns and types the model expects.
Training and scoring both go through it, so parsing cannot drift between them,
and no decision depends on what else happens to be in the current file. It
also reports what did not fit, so a wrong file is caught instead of scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .errors import DataValidationError
from .pipeline_builder import (
    detect_decimal, is_text_column, normalize_columns, parse_boolean,
    parse_numeric_column,
)

# Above this share of absent columns a file is refused outright
MAX_MISSING_COLUMN_SHARE = 0.30
_WARN_UNPARSEABLE_SHARE = 0.05
_WARN_UNSEEN_SHARE = 0.20
_WARN_OUT_OF_RANGE_SHARE = 0.20


@dataclass
class ColumnRule:
    name: str
    kind: str                     # "numeric" | "boolean" | "categorical"
    decimal: str = "."
    low: float | None = None      # numeric range seen in training
    high: float | None = None
    categories: set = field(default_factory=set)


@dataclass
class DataSchema:
    rules: dict[str, ColumnRule]

    @property
    def columns(self) -> list[str]:
        return list(self.rules)

    def of_kind(self, kind: str) -> list[str]:
        return [name for name, rule in self.rules.items() if rule.kind == kind]


@dataclass
class ConformReport:
    n_rows: int
    n_columns: int
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    unparseable: dict[str, float] = field(default_factory=dict)
    unseen_categories: dict[str, float] = field(default_factory=dict)
    out_of_range: dict[str, float] = field(default_factory=dict)

    @property
    def missing_share(self) -> float:
        return len(self.missing) / max(self.n_columns, 1)

    @property
    def is_refused(self) -> bool:
        return self.missing_share > MAX_MISSING_COLUMN_SHARE

    def warnings(self) -> list[str]:
        notes = []
        for col, share in self.unparseable.items():
            notes.append(f"**{col}**: {share:.0%} of values could not be read and are treated as unknown.")
        for col, share in self.unseen_categories.items():
            notes.append(f"**{col}**: {share:.0%} of values never appeared in the training data.")
        for col, share in self.out_of_range.items():
            notes.append(f"**{col}**: {share:.0%} of values are outside the range seen in training.")
        return notes


def build_schema(df_raw: pd.DataFrame, profile: dict) -> DataSchema:
    """Freeze the column decisions made while profiling the training file."""
    df = normalize_columns(df_raw)
    rules: dict[str, ColumnRule] = {}
    for col in profile["numeric_cols"]:
        decimal = detect_decimal(df[col]) if is_text_column(df[col]) else "."
        rules[col] = ColumnRule(col, "numeric", decimal=decimal)
    for col in profile["boolean_cols"]:
        rules[col] = ColumnRule(col, "boolean")
    for col in profile["categorical_cols"]:
        rules[col] = ColumnRule(col, "categorical")

    schema = DataSchema(rules)
    X, _ = conform(df_raw, schema)
    for col, rule in rules.items():
        if rule.kind == "numeric":
            rule.low, rule.high = float(X[col].min()), float(X[col].max())
        elif rule.kind == "categorical":
            rule.categories = set(X[col].dropna().unique())
    return schema


def conform(df_raw: pd.DataFrame, schema: DataSchema) -> tuple[pd.DataFrame, ConformReport]:
    """Return the model's input columns, in order, plus a report of what did not fit."""
    df = normalize_columns(df_raw)
    report = ConformReport(n_rows=len(df), n_columns=len(schema.rules))
    report.extra = [c for c in df.columns if c not in schema.rules]

    columns: dict[str, pd.Series] = {}
    for name, rule in schema.rules.items():
        if name not in df.columns:
            report.missing.append(name)
            is_categorical = rule.kind == "categorical"
            columns[name] = pd.Series(
                np.nan, index=df.index, dtype=object if is_categorical else float)
            continue
        columns[name] = _conform_column(df[name], rule, report)

    return pd.DataFrame(columns, index=df.index), report


def _conform_column(raw: pd.Series, rule: ColumnRule, report: ConformReport) -> pd.Series:
    if rule.kind == "numeric":
        values = parse_numeric_column(raw, rule.decimal)
        _note_unparseable(raw, values, rule.name, report)
        if rule.low is not None and values.notna().any():
            outside = ((values < rule.low) | (values > rule.high))[values.notna()].mean()
            if outside > _WARN_OUT_OF_RANGE_SHARE:
                report.out_of_range[rule.name] = float(outside)
        return values

    if rule.kind == "boolean":
        values = raw.map(parse_boolean).astype(float)
        _note_unparseable(raw, values, rule.name, report)
        return values

    values = raw.map(_clean_category).astype(object)
    if rule.categories and values.notna().any():
        unseen = (~values.dropna().isin(rule.categories)).mean()
        if unseen > _WARN_UNSEEN_SHARE:
            report.unseen_categories[rule.name] = float(unseen)
    return values


def _clean_category(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text if text else np.nan


def _note_unparseable(raw: pd.Series, parsed: pd.Series, name: str, report: ConformReport) -> None:
    filled = raw.map(lambda v: not pd.isna(v) and str(v).strip() != "")
    if not filled.any():
        return
    share = float(parsed[filled].isna().mean())
    if share > _WARN_UNPARSEABLE_SHARE:
        report.unparseable[name] = share


def require_usable(report: ConformReport) -> None:
    if report.n_rows == 0:
        raise DataValidationError("The file has no customer rows.")
    if report.is_refused:
        raise DataValidationError(
            f"This does not look like the data the model was trained on: "
            f"{len(report.missing)} of its {report.n_columns} columns are missing "
            f"({', '.join(report.missing[:8])}{'…' if len(report.missing) > 8 else ''}). "
            "Nothing was scored."
        )
