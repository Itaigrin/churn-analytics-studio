"""
The whole train-then-score flow, free of any UI.

app.py, the tests and tools/benchmark.py all run this same code, so what is
measured is what users get.

Order of events:
  1. Read the target, clean the file, profile it, screen for leaks, validate.
  2. Freeze the schema and hold back 20% of customers, untouched: at random, keeping
     each customer's rows together, or the most recent customers by a date column.
  3. Plan features and tune every model on the remaining 80% only.
  4. Choose calibration, threshold and winner on out-of-fold predictions.
  5. Score the holdout once. These are the numbers reported to the user.
  6. Refit the winner on all rows, keeping the calibration and threshold from step 4.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .calibration import CalibratedModel
from .config import (
    IMBALANCE_HIGH, IMBALANCE_LOW, MIN_ROWS_PER_CLASS, MIN_TRAINING_ROWS, RANDOM_STATE,
    SUSPICIOUS_HOLDOUT_AUC, TEST_SIZE, VERY_LARGE_DATASET_ROWS, WARN_ROWS_PER_CLASS,
)
from .errors import DataValidationError
from .evaluator import evaluate_all, evaluate_holdout, pick_best_model
from .explain import MAX_CUSTOMERS, ColumnReference, Reason, build_reference, column_effects, top_reasons
from .feature_engineering import FeatureSpec, apply_features, fit_feature_spec
from .leakage import LeakFinding, scan_for_leakage
from .monitoring import DriftReference, DriftReport, build_drift_reference, drift_report
from .outcomes import OutcomeReport, outcome_report
from .pipeline_builder import (
    convert_target, days_before, detect_date_order, normalize_column_name, parse_dates, raw_clean,
)
from .profiler import looks_like_dates, profile_dataset
from .schema import ConformReport, DataSchema, build_schema, conform, require_usable
from .trainer import train_all_models

logger = logging.getLogger(__name__)


@dataclass
class ChurnArtifact:
    """Everything needed to score new customers, frozen at training time."""
    schema: DataSchema
    feature_spec: FeatureSpec
    model: CalibratedModel
    threshold: float
    model_columns: list[str]
    reference: dict[str, ColumnReference] = field(default_factory=dict)
    drift_reference: DriftReference = field(default_factory=DriftReference)

    def check(self, df_new: pd.DataFrame, as_of=None) -> ConformReport:
        """What does not fit in this file, without scoring it."""
        return conform(df_new, self.schema, as_of)[1]

    def predict_proba(self, df_new: pd.DataFrame, allow_missing: bool = False, as_of=None) -> np.ndarray:
        """
        Calibrated churn probability per customer. Given the reference date, the result
        for a customer depends only on that customer's own row, never on the rest of the file.
        """
        X, report = conform(df_new, self.schema, as_of)
        require_usable(report)
        if report.missing and not allow_missing:
            raise DataValidationError(
                f"These columns are missing: {', '.join(report.missing)}. "
                "Without them predictions are less accurate."
            )
        features = apply_features(X, self.feature_spec)[self.model_columns]
        return self.model.predict_proba(features)[:, 1]

    def monitor(self, df_new: pd.DataFrame, probabilities: np.ndarray, as_of=None) -> DriftReport:
        """How far this file's columns and scores are from the training customers."""
        X, _ = conform(df_new, self.schema, as_of)
        return drift_report(self.drift_reference, X, probabilities)

    def check_outcomes(self, df_scored: pd.DataFrame, outcome_col: str, positive_label, then: dict,
                       as_of=None) -> OutcomeReport:
        """
        Score customers whose real outcome is now known, and compare with how the model
        did when it was trained. Rows without an outcome are left out.
        """
        y = convert_target(df_scored[outcome_col], positive_label)
        known = y.notna()
        customers = df_scored.loc[known].drop(columns=[outcome_col])
        probabilities = self.predict_proba(customers, allow_missing=True, as_of=as_of)
        return outcome_report(probabilities, y[known].astype(int), self.threshold, then)

    def explain(self, df_new: pd.DataFrame, probabilities: np.ndarray, as_of=None) -> list[list[Reason]]:
        """The main reasons behind each customer's risk. Only the riskiest are explained."""
        X, _ = conform(df_new, self.schema, as_of)
        riskiest = np.argsort(-probabilities)[:MAX_CUSTOMERS]
        subset = X.iloc[riskiest]
        effects = column_effects(
            subset, self.reference, self.feature_spec, self.model_columns,
            lambda features: self.model.predict_proba(features)[:, 1],
            probabilities[riskiest],
        )
        reasons: list[list[Reason]] = [[] for _ in range(len(X))]
        for position, customer_reasons in zip(riskiest, top_reasons(effects, subset, self.reference)):
            reasons[position] = customer_reasons
        return reasons


@dataclass
class TrainingRun:
    artifact: ChurnArtifact
    results: dict
    best_model_name: str
    holdout: dict
    profile: dict
    churn_rate: float
    is_imbalanced: bool
    n_train: int
    n_holdout: int
    runtime_seconds: float
    leak_findings: list[LeakFinding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    holdout_description: str = ""


def screen_columns(df: pd.DataFrame, target_col: str, id_col: Optional[str],
                   positive_label=None) -> list[LeakFinding]:
    """Leak findings for the columns that would be model inputs. Used before training."""
    y = convert_target(df[target_col], positive_label).dropna().astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 3:
        return []
    clean = raw_clean(df.loc[y.index].drop(columns=[target_col]))
    profile = profile_dataset(clean, target_col=None, id_col=_std(id_col))
    return scan_for_leakage(_dates_as_days(clean, profile["datetime_cols"]), y, _candidates(profile))


def date_columns(df: pd.DataFrame, target_col: str, id_col: Optional[str]) -> list[str]:
    """
    The date columns, by their names in the file. A date that is the same in every
    row, such as the day of the snapshot, is listed too: it can be the reference date.
    """
    names = {normalize_column_name(c): c for c in df.columns}
    clean = raw_clean(df.drop(columns=[target_col]))
    profile = profile_dataset(clean, target_col=None, id_col=_std(id_col))
    constant_dates = [c for c in profile["drop_cols"] if _is_date_column(clean, c)]
    return [names[c] for c in profile["datetime_cols"] + constant_dates]


def run_training(
    df: pd.DataFrame,
    target_col: str,
    id_col: Optional[str] = None,
    selected_models: Optional[list[str]] = None,
    budget: Optional[dict] = None,
    positive_label=None,
    exclude_cols: Optional[list[str]] = None,
    progress: Optional[Callable[[int, str], None]] = None,
    as_of=None,
    split_by: Optional[str] = None,
) -> TrainingRun:
    """
    Train, choose and report. exclude_cols is the user's decision on which columns
    to leave out; when it is None, every column the leak screen flags is left out.

    as_of is what dates are counted back from: a date, the name of a date column
    holding each row's own date, or None for the latest date in the file.
    split_by names a date column; the most recent customers by it become the test set.
    """
    from .config import PRODUCTION_MODELS

    started = time.time()
    selected_models = selected_models or PRODUCTION_MODELS
    report = progress or (lambda pct, msg: None)
    warnings: list[str] = []
    notes: list[str] = []

    report(5, "Reading the target column and cleaning the data…")
    if target_col not in df.columns:
        raise DataValidationError(f"Target column '{target_col}' was not found in the file.")
    y_all = convert_target(df[target_col], positive_label)
    n_unlabelled = int(y_all.isna().sum())
    if n_unlabelled:
        warnings.append(f"{n_unlabelled:,} rows have no value in the target column and were left out.")
    y = y_all.dropna().astype(int)
    _validate_labels(y, warnings)
    df_features = df.loc[y.index].drop(columns=[target_col])
    clean = raw_clean(df_features)

    report(10, "Profiling the data and screening for leaks…")
    id_std = _std(id_col)
    profile = profile_dataset(clean, target_col=None, id_col=id_std)
    as_of = normalize_column_name(as_of) if isinstance(as_of, str) else as_of
    split_by = _std(split_by)
    for col in (as_of, split_by):
        # A snapshot date is often the same in every row, so profiling drops it as constant
        if isinstance(col, str) and not (col in profile["datetime_cols"] or _is_date_column(clean, col)):
            raise DataValidationError(f"'{col}' is not a date column, so it cannot be used as one.")

    if exclude_cols is None:
        findings = scan_for_leakage(_dates_as_days(clean, profile["datetime_cols"]), y, _candidates(profile))
        excluded = [f.column for f in findings]
        for finding in findings:
            warnings.append(f"**{finding.column}** was left out: {finding.reason}.")
    else:
        findings = []
        excluded = [normalize_column_name(c) for c in exclude_cols]
    _remove_columns(profile, excluded)

    _validate_columns(profile)
    _check_duplicates(clean, id_std, warnings)
    warnings.extend(profile["general_warnings"][:5])
    if len(y) > VERY_LARGE_DATASET_ROWS:
        warnings.append(f"Very large dataset ({len(y):,} rows). Consider stratified sampling.")

    schema = build_schema(df_features, profile, as_of=as_of)
    X, conformed = conform(df_features, schema, None if isinstance(as_of, str) else as_of)
    if schema.has_dates:
        dated = [rule.name for rule in schema.rules.values() if rule.kind == "date"]
        counted_from = (f"each row's **{conformed.as_of}**" if isinstance(conformed.as_of, str)
                        else f"**{conformed.as_of:%Y-%m-%d}**")
        notes.append(
            f"📅 Dates are used as the number of days before {counted_from}: {', '.join(dated)}. "
            "Check that none of them is recorded after the customer left, such as a cancellation date."
        )

    churn_rate = float(y.mean())
    is_imbalanced = churn_rate < IMBALANCE_LOW or churn_rate > IMBALANCE_HIGH
    if is_imbalanced:
        notes.append(
            f"Only {churn_rate:.1%} of customers in this dataset churned. Models are told to "
            "pay extra attention to the rarer group, and probabilities are then calibrated back."
        )

    report(15, "Holding back customers for the final test…")
    groups = _customer_groups(clean, id_std)
    train_idx, test_idx, holdout_description = _split(df_features, y, groups, split_by)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    groups_train = groups.loc[train_idx] if groups is not None else None

    report(20, "Creating features from the training customers only…")
    spec = fit_feature_spec(X_train, y_train, schema.of_kind("numeric"), schema.of_kind("boolean"))
    X_train_f = apply_features(X_train, spec)
    X_test_f = apply_features(X_test, spec)
    model_profile = {
        "numeric_cols": schema.of_kind("numeric") + spec.names,
        "categorical_cols": schema.of_kind("categorical"),
        "boolean_cols": schema.of_kind("boolean"),
    }
    if spec.names:
        notes.append(f"Added engineered features: {', '.join(spec.names)}")

    def on_model_progress(name, i, total, detail):
        report(25 + int(i / max(total, 1) * 50), f"Training **{name}** ({i + 1}/{total}) · {detail}")

    results = train_all_models(
        X_train=X_train_f, y_train=y_train,
        profile=model_profile,
        selected_models=selected_models,
        n_samples=len(X_train_f),
        is_imbalanced=is_imbalanced,
        progress_callback=on_model_progress,
        budget=budget,
        groups=groups_train,
    )

    def on_evaluation_progress(name, i, total):
        report(75 + int(i / max(total, 1) * 15), f"Comparing models fairly: **{name}** ({i + 1}/{total})…")

    results = evaluate_all(results, X_train_f, y_train, on_evaluation_progress, groups_train)
    best_name = pick_best_model(results)
    best = results[best_name]

    report(92, "Testing the winner on customers it has never seen…")
    trained = CalibratedModel(best["model"], best["calibrator"])
    holdout = evaluate_holdout(trained, best["threshold"], X_test_f, y_test)
    if holdout["roc_auc"] >= SUSPICIOUS_HOLDOUT_AUC:
        warnings.append(
            f"The model scores ROC-AUC {holdout['roc_auc']:.3f} on unseen customers. Results this good "
            "usually mean a column contains the answer. Check that every column is known "
            "before the customer leaves."
        )

    report(96, "Refitting the winner on all customers…")
    X_all = apply_features(X, spec)
    final_pipeline = clone(best["model"]).fit(X_all, y)
    artifact = ChurnArtifact(
        schema=schema,
        feature_spec=spec,
        model=CalibratedModel(final_pipeline, best["calibrator"]),
        threshold=best["threshold"],
        model_columns=list(X_all.columns),
        reference=build_reference(X_train, schema.of_kind("numeric")),
        drift_reference=build_drift_reference(
            X_train, schema.of_kind("numeric"), trained.predict_proba(X_test_f)[:, 1]),
    )

    report(100, "Done")
    return TrainingRun(
        artifact=artifact,
        results=results,
        best_model_name=best_name,
        holdout=holdout,
        profile=profile,
        churn_rate=churn_rate,
        is_imbalanced=is_imbalanced,
        n_train=len(X_train),
        n_holdout=len(X_test),
        runtime_seconds=time.time() - started,
        leak_findings=findings,
        warnings=warnings,
        notes=notes,
        holdout_description=holdout_description,
    )


def _std(col: Optional[str]) -> Optional[str]:
    return normalize_column_name(col) if col else None


def _is_date_column(clean: pd.DataFrame, col: str) -> bool:
    if col not in clean.columns:
        return False
    values = clean[col].dropna()
    if pd.api.types.is_datetime64_any_dtype(values):
        return len(values) > 0
    return len(values) > 0 and looks_like_dates(values)


def _candidates(profile: dict) -> list[str]:
    return (profile["numeric_cols"] + profile["boolean_cols"]
            + profile["categorical_cols"] + profile["datetime_cols"])


def _dates_as_days(clean: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """Dates as days before the latest one, so the leak screen can read them as numbers."""
    if not date_cols:
        return clean
    clean = clean.copy()
    dates = {col: parse_dates(clean[col], detect_date_order(clean[col])) for col in date_cols}
    latest = max(d.max() for d in dates.values())
    for col, values in dates.items():
        clean[col] = days_before(values, latest)
    return clean


def _customer_groups(clean: pd.DataFrame, id_col: Optional[str]) -> Optional[pd.Series]:
    """Customer IDs, when some customers have several rows. Those rows must stay together."""
    if not id_col or id_col not in clean.columns or not clean[id_col].duplicated().any():
        return None
    return clean[id_col].astype(str)


def _split(df_features: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series],
           split_by: Optional[str]) -> tuple[pd.Index, pd.Index, str]:
    """Which customers train the model and which are held back to test it."""
    if split_by:
        raw = df_features[[c for c in df_features.columns if normalize_column_name(c) == split_by][0]]
        dates = parse_dates(raw, detect_date_order(raw))
        # Undated rows can only be used for training
        order = dates.fillna(pd.Timestamp.min).sort_values(kind="stable").index
        n_test = int(round(len(order) * TEST_SIZE))
        train_idx, test_idx = order[:-n_test], order[-n_test:]
        _require_both_outcomes(y.loc[test_idx], "most recent customers")
        start, end = dates.loc[test_idx].min(), dates.loc[test_idx].max()
        return train_idx, test_idx, (
            f"the {n_test:,} most recent customers by **{split_by}** ({start:%Y-%m-%d} to {end:%Y-%m-%d}). "
            "They are later than every customer the model learned from, as new customers will be"
        )
    if groups is not None:
        folds = StratifiedGroupKFold(n_splits=round(1 / TEST_SIZE), shuffle=True, random_state=RANDOM_STATE)
        train_pos, test_pos = next(folds.split(y, y, groups))
        return y.index[train_pos], y.index[test_pos], (
            f"{len(test_pos):,} rows of customers the model never saw. Customers with several rows "
            "were kept whole, so none of them is in both training and testing"
        )
    train_idx, test_idx = train_test_split(
        y.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return train_idx, test_idx, f"{len(test_idx):,} customers held back from the start"


def _require_both_outcomes(y_test: pd.Series, who: str) -> None:
    if y_test.nunique() < 2 or y_test.value_counts().min() < 5:
        raise DataValidationError(
            f"The {who} include fewer than 5 customers of one outcome, so they cannot test the "
            "model. Test on random customers instead, or add more recent history."
        )


def _remove_columns(profile: dict, columns: list[str]) -> None:
    for col in columns:
        for kind in ("numeric_cols", "boolean_cols", "categorical_cols", "datetime_cols"):
            if col in profile[kind]:
                profile[kind].remove(col)
                profile["drop_cols"][col] = "Left out by the leak check"


def _validate_labels(y: pd.Series, warnings: list[str]) -> None:
    if len(y) < MIN_TRAINING_ROWS:
        raise DataValidationError(
            f"The file has {len(y)} labelled customers and at least {MIN_TRAINING_ROWS} are needed."
        )
    counts = y.value_counts()
    if y.nunique() < 2:
        raise DataValidationError(
            "All customers in the file have the same target value, so there is nothing to learn from."
        )
    fewest = int(counts.min())
    if fewest < MIN_ROWS_PER_CLASS:
        raise DataValidationError(
            f"Only {fewest} customers belong to the rarer outcome and at least "
            f"{MIN_ROWS_PER_CLASS} are needed to train and test a model."
        )
    if fewest < WARN_ROWS_PER_CLASS:
        warnings.append(
            f"Only {fewest} customers belong to the rarer outcome, so the results below are "
            "a rough estimate. More history would make them more reliable."
        )


def _validate_columns(profile: dict) -> None:
    if not _candidates(profile):
        raise DataValidationError(
            "No usable columns are left after removing IDs and empty or constant columns."
        )


def _check_duplicates(clean: pd.DataFrame, id_col: Optional[str], warnings: list[str]) -> None:
    if not id_col or id_col not in clean.columns:
        return
    n_repeated = int(clean[id_col].duplicated().sum())
    if n_repeated:
        warnings.append(
            f"{n_repeated:,} rows repeat a customer ID that appears earlier in the file. All rows of "
            "a customer are kept on the same side when testing, so the results are not inflated by it."
        )
