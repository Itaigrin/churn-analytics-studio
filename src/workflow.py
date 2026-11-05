"""
The whole train-then-score flow, free of any UI.

app.py, the tests and tools/benchmark.py all run this same code, so what is
measured is what users get.

Order of events:
  1. Read the target, clean the file, profile it, screen for leaks, validate.
  2. Freeze the schema and hold back 20% of customers, untouched.
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
from sklearn.model_selection import train_test_split

from .calibration import CalibratedModel
from .config import (
    IMBALANCE_HIGH, IMBALANCE_LOW, MIN_ROWS_PER_CLASS, MIN_TRAINING_ROWS, RANDOM_STATE,
    SUSPICIOUS_HOLDOUT_AUC, TEST_SIZE, VERY_LARGE_DATASET_ROWS, WARN_ROWS_PER_CLASS,
)
from .errors import DataValidationError
from .evaluator import evaluate_all, evaluate_holdout, pick_best_model
from .feature_engineering import FeatureSpec, apply_features, fit_feature_spec
from .leakage import LeakFinding, scan_for_leakage
from .pipeline_builder import convert_target, normalize_column_name, raw_clean
from .profiler import profile_dataset
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

    def check(self, df_new: pd.DataFrame) -> ConformReport:
        """What does not fit in this file, without scoring it."""
        return conform(df_new, self.schema)[1]

    def predict_proba(self, df_new: pd.DataFrame, allow_missing: bool = False) -> np.ndarray:
        """
        Calibrated churn probability per customer. The result for a customer
        depends only on that customer's own row, never on the rest of the file.
        """
        X, report = conform(df_new, self.schema)
        require_usable(report)
        if report.missing and not allow_missing:
            raise DataValidationError(
                f"These columns are missing: {', '.join(report.missing)}. "
                "Without them predictions are less accurate."
            )
        features = apply_features(X, self.feature_spec)[self.model_columns]
        return self.model.predict_proba(features)[:, 1]


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


def screen_columns(df: pd.DataFrame, target_col: str, id_col: Optional[str],
                   positive_label=None) -> list[LeakFinding]:
    """Leak findings for the columns that would be model inputs. Used before training."""
    y = convert_target(df[target_col], positive_label).dropna().astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 3:
        return []
    clean = raw_clean(df.loc[y.index].drop(columns=[target_col]))
    profile = profile_dataset(clean, target_col=None, id_col=_std(id_col))
    columns = profile["numeric_cols"] + profile["boolean_cols"] + profile["categorical_cols"]
    return scan_for_leakage(clean, y, columns)


def run_training(
    df: pd.DataFrame,
    target_col: str,
    id_col: Optional[str] = None,
    selected_models: Optional[list[str]] = None,
    budget: Optional[dict] = None,
    positive_label=None,
    exclude_cols: Optional[list[str]] = None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> TrainingRun:
    """
    Train, choose and report. exclude_cols is the user's decision on which columns
    to leave out; when it is None, every column the leak screen flags is left out.
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
    candidates = profile["numeric_cols"] + profile["boolean_cols"] + profile["categorical_cols"]

    if exclude_cols is None:
        findings = scan_for_leakage(clean, y, candidates)
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

    schema = build_schema(df_features, profile)
    X, _ = conform(df_features, schema)

    churn_rate = float(y.mean())
    is_imbalanced = churn_rate < IMBALANCE_LOW or churn_rate > IMBALANCE_HIGH
    if is_imbalanced:
        notes.append(
            f"Only {churn_rate:.1%} of customers in this dataset churned. Models are told to "
            "pay extra attention to the rarer group, and probabilities are then calibrated back."
        )

    report(15, "Holding back customers for the final test…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

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
    )

    def on_evaluation_progress(name, i, total):
        report(75 + int(i / max(total, 1) * 15), f"Comparing models fairly: **{name}** ({i + 1}/{total})…")

    results = evaluate_all(results, X_train_f, y_train, on_evaluation_progress)
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
    )


def _std(col: Optional[str]) -> Optional[str]:
    return normalize_column_name(col) if col else None


def _remove_columns(profile: dict, columns: list[str]) -> None:
    for col in columns:
        for kind in ("numeric_cols", "boolean_cols", "categorical_cols"):
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
    if not (profile["numeric_cols"] or profile["boolean_cols"] or profile["categorical_cols"]):
        raise DataValidationError(
            "No usable columns are left after removing IDs and empty or constant columns."
        )


def _check_duplicates(clean: pd.DataFrame, id_col: Optional[str], warnings: list[str]) -> None:
    if not id_col or id_col not in clean.columns:
        return
    n_repeated = int(clean[id_col].duplicated().sum())
    if n_repeated:
        warnings.append(
            f"{n_repeated:,} rows repeat a customer ID that appears earlier in the file. A customer "
            "can then land in both training and testing, which makes the results look better than "
            "they really are. Use one row per customer."
        )
