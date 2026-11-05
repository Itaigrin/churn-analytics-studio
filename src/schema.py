"""
The frozen description of the data a model was trained on.

conform() turns a raw file into the exact columns and types the model expects.
Training and scoring both go through it, so parsing cannot drift between them,
and no decision depends on what else happens to be in the current file. It
also reports what did not fit, so a wrong file is caught instead of scored.

Dates become a number of days before a reference date: the day the data was
taken, a date the user picks, or a date column in each row. That date has to be
known to score a customer, so it is an explicit input rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .errors import DataValidationError
from .pipeline_builder import (
    days_before, detect_date_order, detect_decimal, is_text_column, normalize_columns,
    parse_boolean, parse_dates, parse_numeric_column,
)

# Above this share of absent columns a file is refused outright
MAX_MISSING_COLUMN_SHARE = 0.30
_WARN_UNPARSEABLE_SHARE = 0.05
_WARN_UNSEEN_SHARE = 0.20
_WARN_OUT_OF_RANGE_SHARE = 0.20
DAYS_SUFFIX = "_days_ago"


@dataclass
class ColumnRule:
    name: str
    kind: str                     # "numeric" | "boolean" | "categorical" | "date"
    decimal: str = "."
    low: float | None = None      # numeric range seen in training
    high: float | None = None
    categories: set = field(default_factory=set)
    date_order: str = "dayfirst"

    @property
    def output(self) -> str:
        """The column name the model sees."""
        return self.name + DAYS_SUFFIX if self.kind == "date" else self.name


@dataclass
class DataSchema:
    rules: dict[str, ColumnRule]
    as_of_column: str | None = None       # when each row carries its own reference date
    as_of_order: str = "dayfirst"

    @property
    def columns(self) -> list[str]:
        """The columns a file has to provide."""
        return list(self.rules) + ([self.as_of_column] if self.as_of_column else [])

    @property
    def has_dates(self) -> bool:
        return any(rule.kind == "date" for rule in self.rules.values())

    def of_kind(self, kind: str) -> list[str]:
        """Model columns of one kind. Dates count as numbers once they are days."""
        kinds = {"numeric", "date"} if kind == "numeric" else {kind}
        return [rule.output for rule in self.rules.values() if rule.kind in kinds]

    def latest_date(self, df_raw: pd.DataFrame) -> pd.Timestamp | None:
        """The most recent date in a file, no later than today: the usual reference date."""
        df = normalize_columns(df_raw)
        latest = [parse_dates(df[name], rule.date_order).max()
                  for name, rule in self.rules.items() if rule.kind == "date" and name in df.columns]
        latest = [d for d in latest if pd.notna(d)]
        if not latest:
            return None
        return min(max(latest), pd.Timestamp.today().normalize())


@dataclass
class ConformReport:
    n_rows: int
    n_columns: int
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    unparseable: dict[str, float] = field(default_factory=dict)
    unseen_categories: dict[str, float] = field(default_factory=dict)
    out_of_range: dict[str, float] = field(default_factory=dict)
    as_of: object = None

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


def build_schema(df_raw: pd.DataFrame, profile: dict, as_of=None) -> DataSchema:
    """
    Freeze the column decisions made while profiling the training file.
    as_of is a date, the name of a date column holding each row's own date, or None
    for the latest date in the file.
    """
    df = normalize_columns(df_raw)
    as_of_column = as_of if isinstance(as_of, str) else None
    rules: dict[str, ColumnRule] = {}
    for col in profile["numeric_cols"]:
        decimal = detect_decimal(df[col]) if is_text_column(df[col]) else "."
        rules[col] = ColumnRule(col, "numeric", decimal=decimal)
    for col in profile["boolean_cols"]:
        rules[col] = ColumnRule(col, "boolean")
    for col in profile["categorical_cols"]:
        rules[col] = ColumnRule(col, "categorical")
    for col in profile.get("datetime_cols", []):
        if col != as_of_column:
            rules[col] = ColumnRule(col, "date", date_order=detect_date_order(df[col]))

    schema = DataSchema(rules)
    if as_of_column:
        if as_of_column not in df.columns:
            raise DataValidationError(f"The reference date column '{as_of_column}' is not in the file.")
        schema.as_of_column = as_of_column
        schema.as_of_order = detect_date_order(df[as_of_column])

    X, _ = conform(df_raw, schema, as_of=None if as_of_column else as_of)
    for rule in rules.values():
        if rule.kind in ("numeric", "date"):
            rule.low, rule.high = float(X[rule.output].min()), float(X[rule.output].max())
        elif rule.kind == "categorical":
            rule.categories = set(X[rule.name].dropna().unique())
    return schema


def conform(df_raw: pd.DataFrame, schema: DataSchema, as_of=None) -> tuple[pd.DataFrame, ConformReport]:
    """
    Return the model's input columns, in order, plus a report of what did not fit.
    as_of is the date days are counted back from. Without one, each row's date column
    is used if the model was trained that way, and otherwise the latest date in the file.
    """
    df = normalize_columns(df_raw)
    report = ConformReport(n_rows=len(df), n_columns=len(schema.columns))
    report.extra = [c for c in df.columns if c not in schema.columns]
    reference = _reference_dates(df, schema, as_of, report)

    columns: dict[str, pd.Series] = {}
    for name, rule in schema.rules.items():
        if name not in df.columns:
            report.missing.append(name)
            is_categorical = rule.kind == "categorical"
            columns[rule.output] = pd.Series(
                np.nan, index=df.index, dtype=object if is_categorical else float)
            continue
        columns[rule.output] = _conform_column(df[name], rule, report, reference)

    return pd.DataFrame(columns, index=df.index), report


def _reference_dates(df: pd.DataFrame, schema: DataSchema, as_of, report: ConformReport):
    if not schema.has_dates:
        return None
    if as_of is not None:
        report.as_of = pd.Timestamp(as_of).normalize()
        return report.as_of
    if schema.as_of_column:
        report.as_of = schema.as_of_column
        if schema.as_of_column not in df.columns:
            report.missing.append(schema.as_of_column)
            return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        return parse_dates(df[schema.as_of_column], schema.as_of_order)
    report.as_of = schema.latest_date(df)
    return report.as_of if report.as_of is not None else pd.NaT


def _conform_column(raw: pd.Series, rule: ColumnRule, report: ConformReport, reference) -> pd.Series:
    if rule.kind == "numeric":
        values = parse_numeric_column(raw, rule.decimal)
        _note_unparseable(raw, values, rule.name, report)
        _note_out_of_range(values, rule, report)
        return values

    if rule.kind == "date":
        dates = parse_dates(raw, rule.date_order)
        _note_unparseable(raw, dates, rule.name, report)
        values = days_before(dates, reference)
        _note_out_of_range(values, rule, report)
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


def _note_out_of_range(values: pd.Series, rule: ColumnRule, report: ConformReport) -> None:
    if rule.low is not None and values.notna().any():
        outside = ((values < rule.low) | (values > rule.high))[values.notna()].mean()
        if outside > _WARN_OUT_OF_RANGE_SHARE:
            report.out_of_range[rule.output] = float(outside)


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
