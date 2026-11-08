"""
Honest benchmark: train on one file, score a separate file the app never saw,
and compare with the true outcomes. Runs the same code the app runs.

    python tools/benchmark.py <data_folder> [--depth Thorough] [--only telco,waze] [--out results.json]

The folder holds the public churn datasets named in DATASETS.
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
)

from src.config import SEARCH_PRESETS
from src.data_loader import load_uploaded_file
from src.errors import DataValidationError
from src.evaluator import calibration_error
from src.feature_engineering import apply_features
from src.profiler import detect_id_column, detect_target_column
from src.schema import conform
from src.workflow import run_training

DATASETS = {
    "bank_churn": dict(train="Churn_Modelling_TRAIN.csv", test="Churn_Modelling_TEST.csv", full="Churn_Modelling.csv",
                       key="CustomerId", label="Exited", churned=("1",)),
    "telco": dict(train="WA_Fn-UseC_-Telco-Customer-Churn_TRAIN.csv", test="WA_Fn-UseC_-Telco-Customer-Churn_TEST.csv",
                  full="WA_Fn-UseC_-Telco-Customer-Churn.csv", key="customerID", label="Churn", churned=("Yes",)),
    "credit_cards": dict(train="BankChurners_train.csv", test="BankChurners_test.csv", full="BankChurners.csv",
                         key="CLIENTNUM", label="Attrition_Flag", churned=("1", "1.0", "Attrited Customer")),
    "waze": dict(train="waze_dataset1.csv", test="waze_dataset_Test.csv", full="waze_dataset.csv",
                 key="ID", label="label", churned=("churned",)),
}


def load(folder, name):
    with open(os.path.join(folder, name), "rb") as fh:
        import io
        buffer = io.BytesIO(fh.read())
    buffer.name = name
    return load_uploaded_file(buffer)


def pr_gap_interval(y, better, worse, rounds=1000):
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(rounds):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            gaps.append(average_precision_score(y[i], better[i]) - average_precision_score(y[i], worse[i]))
    return float(np.mean(gaps)), *[float(x) for x in np.percentile(gaps, [2.5, 97.5])]


def run_one(folder, name, cfg, depth):
    print(f"\n{'=' * 70}\n{name} ({depth})\n{'=' * 70}", flush=True)
    train, test, full = load(folder, cfg["train"]), load(folder, cfg["test"]), load(folder, cfg["full"])
    id_col = detect_id_column(train)
    target = detect_target_column(train, exclude=[id_col] if id_col else [])
    print(f"train {train.shape}, test {test.shape}, id={id_col}, target={target}")

    started = time.time()
    run = run_training(train, target, id_col=id_col, budget=SEARCH_PRESETS[depth])
    artifact, best = run.artifact, run.best_model_name
    errors = {k: v["error"][:100] for k, v in run.results.items() if v.get("error")}
    print(f"winner={best}, errors={errors or 'none'}, threshold={artifact.threshold:.2f}, "
          f"runtime={run.runtime_seconds:.0f}s")
    print("left out:", run.profile["drop_cols"])
    print("leak findings:", [(f.column, f.auc) for f in run.leak_findings])
    print("warnings:", run.warnings)

    truth = full[[cfg["key"], cfg["label"]]].dropna(subset=[cfg["label"]]).copy()
    truth["_y"] = truth[cfg["label"]].astype(str).isin(cfg["churned"]).astype(int)
    labelled = test.merge(truth[[cfg["key"], "_y"]], on=cfg["key"], how="inner")
    y = labelled["_y"].values
    new = labelled.drop(columns=["_y"])
    print(f"unseen: {len(y)} customers, true churn rate {y.mean():.3f}")

    report = artifact.check(new)
    print("check on unseen file: missing", report.missing, "| warnings", report.warnings())
    p = artifact.predict_proba(new)
    flagged = p >= artifact.threshold

    # Every model as fitted on the training split, for a same-basis comparison
    X_new, _ = conform(new, artifact.schema)
    X_new = apply_features(X_new, artifact.feature_spec)[artifact.model_columns]
    per_model = {k: r["model"].predict_proba(X_new)[:, 1] for k, r in run.results.items() if r.get("model") is not None}
    table = pd.DataFrame([
        dict(model=k, selection_pr_auc=run.results[k]["pr_auc"], unseen_pr_auc=average_precision_score(y, v),
             unseen_roc_auc=roc_auc_score(y, v))
        for k, v in per_model.items()
    ]).sort_values("unseen_pr_auc", ascending=False)
    print(table.round(4).to_string(index=False))
    top = table.iloc[0]["model"]
    gap = pr_gap_interval(y, per_model[top], per_model[best])
    print(f"PR-AUC gap {top} minus winner {best}: {gap[0]:.4f}, 95% CI [{gap[1]:.4f}, {gap[2]:.4f}]")

    # Same customers alone, then in small files
    rng = np.random.default_rng(0)
    sample = rng.choice(len(new), size=min(300, len(new)), replace=False)
    alone = np.array([artifact.predict_proba(new.iloc[[i]])[0] for i in sample])
    delta = np.abs(alone - p[sample])
    flips = float(((alone >= artifact.threshold) != flagged[sample]).mean())
    print(f"alone vs in file: max |dp|={delta.max():.2e}, decision flips={flips:.2%}")

    half = [c for c in test.columns if c != cfg["key"]][::2]
    try:
        artifact.predict_proba(new.drop(columns=half))
        half_columns = "scored without consent"
    except DataValidationError as exc:
        half_columns = f"stopped: {str(exc)[:80]}"
    print("half the columns removed:", half_columns)

    h = run.holdout
    expected_gap = abs(p.mean() - run.churn_rate) * 100
    result = dict(
        winner=best, best_on_unseen=top, gap=gap, threshold=artifact.threshold,
        reported=dict(pr_auc=h["pr_auc"], recall=h["recall"], precision=h["precision"], roc=h["roc_auc"],
                      n=h["n_customers"], mean_prob=h["mean_probability"], actual=h["actual_rate"]),
        unseen=dict(pr_auc=float(average_precision_score(y, p)), recall=float(recall_score(y, flagged)),
                    precision=float(precision_score(y, flagged, zero_division=0)), f1=float(f1_score(y, flagged)),
                    roc=float(roc_auc_score(y, p)), flagged=float(flagged.mean()), mean_prob=float(p.mean()),
                    actual=float(y.mean()), brier=float(brier_score_loss(y, p)), ece=calibration_error(y, p)),
        alone_max_delta=float(delta.max()), alone_flips=flips, half_columns=half_columns,
        same_population_expected_gap_pp=float(expected_gap),
        dropped=run.profile["drop_cols"], engineered=artifact.feature_spec.names,
        runtime=time.time() - started,
    )
    print("REPORTED (holdout):", {k: round(v, 4) for k, v in result["reported"].items()})
    print("UNSEEN   (file)   :", {k: round(v, 4) for k, v in result["unseen"].items()})
    return result, artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--depth", default="Thorough", choices=list(SEARCH_PRESETS))
    parser.add_argument("--only", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    out, artifacts = {}, {}
    for name, cfg in DATASETS.items():
        if args.only and name not in args.only.split(","):
            continue
        out[name], artifacts[name] = run_one(args.folder, name, cfg, args.depth)

    if "bank_churn" in artifacts:
        wrong = load(args.folder, DATASETS["telco"]["test"])
        try:
            artifacts["bank_churn"].predict_proba(wrong)
            print("\nWRONG FILE: scored, nothing stopped it")
        except DataValidationError as exc:
            print(f"\nWRONG FILE: refused: {exc}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
