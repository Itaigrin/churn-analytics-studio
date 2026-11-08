"""
ChurnApp Pro - Professional Customer Churn Prediction
=====================================================
Streamlit application implementing a full ML pipeline:

  Step 1   Upload & Validate Dataset
  Step 2   Column Detection, Leak Screening
  Step 3   Model Training, with hyperparameter search
  Step 4   Results, measured on customers held back from all of the above
           Predict New Customers

The training flow itself lives in src/workflow.py, which has no UI code.
"""

import sys
import os
import hashlib
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))


def _drop_stale_modules():
    """
    Streamlit Cloud updates the files of a running app without restarting Python, so
    app.py is new while the src modules already imported stay old, and an import of
    anything added since fails. When the src files change, forget the old copies.
    """
    stamp = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("src/*.py"))))
    loaded = sys.modules.get("src")
    # Modules loaded before this check existed carry no stamp, so they count as stale too
    if loaded is not None and getattr(loaded, "_source_stamp", None) != stamp.digest():
        for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
            del sys.modules[name]
    import src
    src._source_stamp = stamp.digest()


_drop_stale_modules()

# Prevent SIGSEGV (exit code 139) on cloud: OpenMP oversubscription + fork
# corrupts the heap. Must run BEFORE numpy / lightgbm / xgboost are imported.
from src.runtime_env import ON_CLOUD, limit_threads
limit_threads()

import warnings
warnings.filterwarnings("ignore")

import time
import threading
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.section-header {
    font-size: 1.25rem; font-weight: 700; color: #1e40af;
    border-bottom: 2px solid #2563eb; padding-bottom: 6px;
    margin-bottom: 14px; margin-top: 4px;
}
.stProgress > div > div { background-color: #2563eb !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (defined before any Streamlit widget calls)
# ══════════════════════════════════════════════════════════════════════════════

_MIN_CUSTOMERS_TO_COMPARE = 30


def _sec(icon: str, title: str):
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def _show_signal_assessment(df: pd.DataFrame, target_col: str, id_col, positive_label,
                            exclude_cols: list):
    """Displays the Predictive Signal Assessment results."""
    from src.pipeline_builder import raw_clean, normalize_column_name
    from src.profiler import profile_dataset
    from src.signal_assessor import assess_predictive_signal

    _data_hash = hash(df.to_csv(index=False)[:5000])
    cache_key  = f"signal_{len(df)}_{target_col}_{id_col}_{positive_label}_{exclude_cols}_{_data_hash}"
    if st.session_state.get("_signal_cache_key") != cache_key:
        with st.spinner("Analysing predictive signal…"):
            _profile = profile_dataset(
                raw_clean(df.drop(columns=[target_col])), target_col=None,
                id_col=normalize_column_name(id_col) if id_col else None,
            )
            _used = (_profile["numeric_cols"] + _profile["boolean_cols"]
                     + _profile["categorical_cols"])
            _left_out = {normalize_column_name(c) for c in exclude_cols}
            result = assess_predictive_signal(
                df, target_col, id_col, positive_label=positive_label,
                only_columns=[c for c in _used if c not in _left_out],
            )
        st.session_state["_signal_result"]    = result
        st.session_state["_signal_cache_key"] = cache_key
    else:
        result = st.session_state["_signal_result"]

    if result.get("error"):
        st.warning(f"Signal assessment skipped: {result['error']}")
        return

    _sec("📡", "Top Features")

    tier_icon = {"Strong": "🟢", "Medium": "🟡", "Weak": "🔴"}
    top_rows = []
    for row in result["top_features"]:
        top_rows.append({
            "Feature":                  row["feature"],
            "Tier":                     f"{tier_icon.get(row['tier'], '')} {row['tier']}",
            "Signal":                   f"{row['signal']:.3f}",
            "Why This Feature Matters": row["explanation"],
        })
    st.dataframe(pd.DataFrame(top_rows), width="stretch", hide_index=True)

    st.divider()


def _estimate_runtime(df: pd.DataFrame, selected_models: list,
                      budget: dict | None = None) -> tuple[str, str]:
    """Estimate pipeline runtime based on data size, models and search depth."""
    _on_cloud = ON_CLOUD

    from src.config import PIPELINE_CONFIG
    cfg = {**PIPELINE_CONFIG, **(budget or {})}
    n_trials  = max(cfg.get("optuna_trials", 30), 1)
    cv_folds  = cfg.get("hpo_cv_folds", cfg["cv_folds"])
    n_rows    = len(df)

    # Effective trials after dataset-size scaling in trainer.py
    if n_rows > 100_000:
        n_trials = min(n_trials, max(10, n_trials // 3))
    elif n_rows > 50_000:
        n_trials = min(n_trials, max(15, n_trials // 2))

    # The HPO search may run on a capped subsample (see trainer._hpo_sample)
    _cap = cfg.get("hpo_max_rows", 0)
    if _cap:
        n_rows = min(n_rows, _cap)

    n_cols = df.shape[1]

    # Base seconds per (fold × trial) at 5,000 rows / 15 cols
    _model_base = {
        "Logistic Regression": 0.06,
        "Random Forest":       0.35,
        "XGBoost":             0.20,
        "LightGBM":            0.15,
        "CatBoost":            0.45,
    }

    size_scale = (n_rows / 5_000) * ((n_cols / 15) ** 0.5)
    size_scale = max(size_scale, 0.1)

    total_base = sum(_model_base.get(m, 0.25) for m in selected_models)
    est_sec    = total_base * n_trials * cv_folds * size_scale

    if _on_cloud:
        est_sec *= 2.5

    lo = est_sec * 0.6
    hi = est_sec * 1.8

    def _fmt(s: float) -> str:
        if s < 60:
            return f"{max(1, int(s))}s"
        m = int(s // 60)
        s2 = int(s % 60)
        return f"{m}m {s2}s" if s2 else f"{m}m"

    return _fmt(lo), _fmt(hi)


@st.cache_resource
def _job_registry():
    """One registry per server process, shared by every session."""
    from src.jobs import JobRegistry
    return JobRegistry()


_PRED_STATE = ("model_file", "_pred_file_key", "_pred_df_new", "_pred_report", "pred_raw_probs", "pred_reasons")


def _run_pipeline(df: pd.DataFrame, id_col, target_col: str, selected_models: list,
                  budget: dict | None, positive_label, exclude_cols: list,
                  as_of=None, split_by=None):
    """Start training as a server job. The page follows it, and finds it again after a reconnect."""
    from src.errors import ChurnAppError
    from src.workflow import run_training

    try:
        job = _job_registry().start(
            run_training,
            df=df.copy(), target_col=target_col, id_col=id_col,
            selected_models=selected_models, budget=budget,
            positive_label=positive_label, exclude_cols=exclude_cols,
            as_of=as_of, split_by=split_by,
        )
    except ChurnAppError as exc:
        st.error(str(exc))
        return
    st.session_state.job_id = job.id
    st.query_params["job"] = job.id
    st.rerun()


def _current_job():
    """The training job this page follows, from the session or, after a reconnect, from the address."""
    job_id = st.session_state.get("job_id") or st.query_params.get("job")
    return _job_registry().get(job_id)


def _forget_job():
    st.session_state.job_id = None
    st.query_params.pop("job", None)


def _attach_finished_job(job):
    """Show a finished job's model, once, and start prediction afresh for it."""
    if job.result is None or st.session_state.get("run") is job.result:
        return
    st.session_state.update(run=job.result, training_done=True, job_id=job.id,
                            **{name: None for name in _PRED_STATE})


def _render_progress(pct: int, msg: str):
    st.markdown(
        f"""
<div style="background:#1e2130;border:1px solid #3a3f5c;border-radius:10px;padding:16px 20px;margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <span style="color:#a0aec0;font-size:13px">Running Pipeline…</span>
    <span style="color:#63b3ed;font-weight:700;font-size:18px">{pct}%</span>
  </div>
  <div style="background:#2d3748;border-radius:6px;height:10px;overflow:hidden">
    <div style="background:linear-gradient(90deg,#4299e1,#63b3ed);width:{pct}%;height:100%;border-radius:6px;transition:width 0.3s ease"></div>
  </div>
  <div style="margin-top:10px;color:#e2e8f0;font-size:13px">⏳ {msg.replace("**", "")}</div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.fragment(run_every=2)
def _training_progress(job_id: str):
    """Redrawn every two seconds while the job runs; reloads the page when it is done."""
    job = _job_registry().get(job_id)
    if job is None:
        return
    if job.done:
        st.rerun()
    _render_progress(job.pct, job.message)
    st.caption(
        "Training runs on the server. If the page reloads or the connection drops, it carries on, "
        "and this page picks it up again. Keep this address to come back to it."
    )


def _show_job_error(job):
    st.error(job.error)
    if job.error_details:
        with st.expander("Full traceback"):
            st.code(job.error_details)


def _show_trained_sections():
    """Results, prediction and outcome checks for the model in this session."""
    from src.model_store import SavedModel

    _show_results()
    st.divider()
    _sec("🔮", "Predict New Customers")
    saved = SavedModel.from_run(st.session_state.run)
    _predict_section(saved)
    st.divider()
    _sec("📊", "Check Past Predictions Against What Happened")
    _outcomes_section(saved)


def _show_results():
    """Renders the results of the last training run from session_state."""
    from src.evaluator import build_comparison_table

    run       = st.session_state.run
    results   = run.results
    best_name = run.best_model_name
    holdout   = run.holdout

    st.divider()

    for _msg in run.warnings:
        st.warning(_msg)
    for _msg in run.notes:
        st.info(_msg)

    _sec("📈", "Step 4 - Evaluation Results")

    mins = int(run.runtime_seconds // 60)
    secs = int(run.runtime_seconds % 60)
    st.caption(f"⏱️ Total pipeline runtime: **{mins}m {secs}s**")
    st.markdown(
        f"Measured on {run.holdout_description}. They played no part in training, tuning, choosing "
        f"the model or setting the threshold, so these numbers are the fair estimate for new customers."
    )

    _unseen_help = f"Measured on {run.n_holdout:,} customers the model had never seen."
    ranges = holdout.get("intervals", {})

    def _range(key, fmt="{:.3f}"):
        low_high = ranges.get(key)
        return f" 95% range: {fmt.format(low_high[0])} to {fmt.format(low_high[1])}." if low_high else ""

    st.markdown(f"#### 🏆 Best model: {best_name}")
    st.caption("Chosen by the highest PR-AUC in the comparison below.")
    st.markdown(f"##### How well does it rank customers?  *(on {run.n_holdout:,} new customers)*")
    k2, k3, k4, k5 = st.columns(4)
    k2.metric(f"Prediction Quality · {holdout['quality_grade']}", f"{holdout['quality_score']:.0f}/100",
              help="0 is a coin toss and 100 a perfect ranking (2 × ROC-AUC − 1), the same idea as the "
                   "0–100 quality scores of campaign tools. 60+ is excellent. " + _unseen_help)
    k3.metric("PR-AUC", f"{holdout['pr_auc']:.4f}",
              help="How well real leavers are ranked above stayers. The model is chosen by this. "
                   + _unseen_help + _range("pr_auc"))
    k4.metric("Top 10% Lift", f"{holdout['lift_top10']:.1f}×",
              help="The riskiest 10% of customers leave this many times more often than average. "
                   + _unseen_help + _range("lift_top10", "{:.2f}"))
    k5.metric("ROC-AUC", f"{holdout['roc_auc']:.4f}", help=_unseen_help + _range("roc_auc"))

    st.markdown(f"##### How true are its probabilities?  *(on {run.n_holdout:,} new customers)*")
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted vs Actual Churn", f"{holdout['mean_probability']:.1%} vs {holdout['actual_rate']:.1%}",
              help="Average predicted risk against the share who really left. " + _unseen_help)
    p2.metric("Calibration Error", f"{holdout['calibration_error']:.3f}",
              help="Average gap between predicted and real churn rates across risk groups. Below 0.05 is "
                   "good. The money estimates depend on this.")
    p3.metric("Log Loss", f"{holdout['log_loss']:.4f}",
              help="Penalises confident wrong predictions. Lower is better.")

    chosen = results[best_name]
    st.info(
        f"**Why the table below shows different numbers:** it measures every model on the "
        f"{run.n_train:,} training customers, each scored by a model that did not learn from them (5-fold), "
        f"and that is how the winner is chosen. There {best_name} scored PR-AUC **{chosen['pr_auc']:.4f}** "
        f"and ROC-AUC **{chosen['roc_auc']:.4f}**. The numbers above are its final test on "
        f"{run.n_holdout:,} other customers, kept out of every choice: PR-AUC **{holdout['pr_auc']:.4f}**, "
        f"ROC-AUC **{holdout['roc_auc']:.4f}**. Two different groups of customers give slightly "
        "different results, which is normal. Expect the numbers above on new customers."
    )

    st.markdown("#### Model Comparison  *(on the training customers, 5-fold · sorted by PR-AUC)*")
    st.caption(
        "All models on one basis: each customer is scored by a model trained without them. "
        "This is how the winner is chosen."
    )
    cmp_df = build_comparison_table(results)
    st.dataframe(
        cmp_df,
        width="stretch",
        hide_index=True,
        column_config={
            "PR-AUC": st.column_config.TextColumn(
                "PR-AUC", help="How well real leavers are ranked above stayers. The winner has the highest."),
            "Top 10% Lift": st.column_config.TextColumn(
                "Top 10% Lift", help="How many times more often the riskiest 10% leave than the average customer."),
            "Log Loss": st.column_config.TextColumn(
                "Log Loss", help="How true the probabilities are. Lower is better."),
            "ROC-AUC": st.column_config.TextColumn(
                "ROC-AUC", help="How often a random leaver is ranked above a random stayer. 0.5 is a coin toss."),
        },
    )

    # ── Step 11: Professional Model Analysis ─────────────────────────────────
    _show_professional_analysis(best_name, holdout)

    _show_drivers(run.drivers)

    _show_lift(holdout, run.n_holdout)

    _save_model_section(run)


def _explain_roc(roc: float) -> str:
    if roc >= 0.90:
        return (f"**{roc:.4f}** - Excellent. The model is very strong at distinguishing between customers who will churn "
                "and those who will stay, ranking high-risk customers well ahead of low-risk ones.")
    elif roc >= 0.80:
        return (f"**{roc:.4f}** - Good. The model does a solid job of separating churners from loyal customers "
                "across all decision thresholds. A reliable result for real-world business data.")
    elif roc >= 0.70:
        return (f"**{roc:.4f}** - Moderate. The model is meaningfully better than random guessing "
                "at ranking high-risk customers ahead of low-risk ones.")
    else:
        return (f"**{roc:.4f}** - Weak. The model struggles to reliably tell churners apart from loyal customers. "
                "Consider adding more features or collecting more historical data.")


def _explain_recall(rec: float) -> str:
    caught = int(round(rec * 100))
    missed = 100 - caught
    if rec >= 0.80:
        return (f"**{rec*100:.1f}%** - Out of every 100 customers who actually churned, "
                f"the model correctly identified {caught} of them. Only {missed} were missed. "
                "Very few at-risk customers slip through undetected.")
    elif rec >= 0.60:
        return (f"**{rec*100:.1f}%** - Out of every 100 customers who actually churned, "
                f"the model caught {caught} of them. {missed} were missed - "
                "they left without being flagged in time.")
    else:
        return (f"**{rec*100:.1f}%** - Out of every 100 customers who actually churned, "
                f"only {caught} were correctly identified. {missed} left without the model raising a warning. "
                "Many at-risk customers are being missed.")


def _explain_lift(lift: float, churn_rate: float) -> str:
    top_rate = min(lift * churn_rate, 1.0)
    return (f"**{lift:.1f}×** - Among the riskiest 10% of customers, about **{top_rate:.0%}** leave, against "
            f"{churn_rate:.0%} of all customers. A campaign that can only reach a few customers gets "
            f"{lift:.1f} times more leavers per contact than picking customers at random.")


def _explain_pr_auc(pr_auc: float) -> str:
    if pr_auc >= 0.75:
        return (f"**{pr_auc:.4f}** - Excellent. The model is very effective at identifying customers likely to churn "
                "while keeping false alarms low. It finds real churn customers without incorrectly flagging too many loyal ones.")
    elif pr_auc >= 0.55:
        return (f"**{pr_auc:.4f}** - Good. The model does a reasonable job of identifying churn customers "
                "while limiting unnecessary alerts. Some loyal customers may still be flagged by mistake, "
                "but the overall balance is solid.")
    else:
        return (f"**{pr_auc:.4f}** - Moderate. The model has difficulty finding churn customers "
                "without also flagging many loyal customers incorrectly. "
                "Consider reviewing the data quality or adding more predictive features.")


def _explain_f1(f1: float) -> str:
    if f1 >= 0.80:
        return (f"**{f1:.4f}** - Excellent balance between catching churners and avoiding false alarms. "
                "The model is both sensitive enough to flag most at-risk customers and precise enough to avoid "
                "flooding your team with false positives.")
    elif f1 >= 0.65:
        return (f"**{f1:.4f}** - Good overall balance. The model catches a solid portion of churners "
                "while keeping unnecessary outreach at a reasonable level. "
                "Small trade-offs between Recall and Precision exist but are manageable.")
    else:
        return (f"**{f1:.4f}** - Moderate balance. Either the model is missing some churners (low Recall) "
                "or flagging too many loyal customers by mistake (low Precision), or both. "
                "Consider reviewing the decision threshold or trying a different execution mode.")


def _show_professional_analysis(best_name: str, holdout: dict):
    """Plain-language explanation of the winner's results on unseen customers."""
    _sec("🧠", "What Do the Results Mean?")

    st.markdown(
        "Below is a plain-language explanation of what these numbers actually mean - "
        "no technical background needed."
    )

    st.markdown(f"### 🏆 {best_name}")
    st.markdown(f"**PR-AUC:** {_explain_pr_auc(holdout['pr_auc'])}")
    st.markdown(f"**Top 10% Lift:** {_explain_lift(holdout['lift_top10'], holdout['actual_rate'])}")
    st.markdown(f"**ROC-AUC:** {_explain_roc(holdout['roc_auc'])}")
    st.markdown(
        f"**At the recommended threshold:** {_explain_recall(holdout['recall'])} Of the customers it "
        f"flags, **{holdout['precision']:.0%}** really leave. The threshold can be changed below, so these two "
        "numbers describe a choice, not the model itself."
    )


def _show_guide():
    """Plain-language guide to every metric and model used in the app."""

    st.markdown("## 📖 What does this app do?")
    st.markdown(
        "This app looks at your historical customer data and learns the pattern of "
        "customers who eventually left (churned). It then uses that pattern to flag "
        "which of your current customers are most likely to leave next - so you can "
        "reach out to them before they do."
    )

    st.divider()

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("## 📏 The Numbers - What Do They Mean?")
    st.caption("Think of the model as a detector that reads each customer's profile and raises a flag if it thinks they're about to leave.")

    metrics = [
        (
            "PR-AUC",
            "How well the model finds churners without flooding you with false alarms.",
            "Measures how well the model identifies customers who are likely to churn while keeping false alarms as low as possible. "
            "A higher PR-AUC means the model is better at finding real churn customers without incorrectly flagging too many loyal customers. "
            "**This is the metric the app uses to choose the best model.**",
            "📈",
        ),
        (
            "Recall",
            "Out of all customers who actually churned - how many did the model catch?",
            "Measures how many customers who actually churned were correctly identified by the model. "
            "A higher Recall means fewer at-risk customers are missed, helping businesses take action before customers leave. "
            "If 100 customers were going to churn and the model caught 80, Recall = 80%. "
            "**It depends on the threshold you choose, so it is shown next to the threshold, not used to choose the model.**",
            "🔍",
        ),
        (
            "ROC-AUC",
            "The model's overall ability to tell churners from loyal customers.",
            "Measures the model's overall ability to distinguish between customers who will churn and those who will stay. "
            "A higher ROC-AUC means the model is better at ranking high-risk customers ahead of low-risk customers across all decision thresholds. "
            "**1.0 = perfect. 0.5 = random guessing.** The Prediction Quality score is 2 × ROC-AUC − 1, on a 0–100 scale.",
            "🎯",
        ),
        (
            "Top 10% Lift",
            "How much better than random the riskiest customers are.",
            "Sort customers by risk and take the top 10%. Lift is how many times more often they leave than the "
            "average customer. Campaign tools such as Braze and Pecan judge churn models this way, because a "
            "retention team can usually reach only part of the customer base.",
            "🎯",
        ),
        (
            "Calibration and Log Loss",
            "Whether a 30% risk really means 30 out of 100 leave.",
            "Calibration error compares predicted and real churn rates across risk groups. Log loss penalises "
            "confident wrong predictions. Both matter because the money estimates multiply probabilities by "
            "customer value.",
            "⚖️",
        ),
    ]

    for name, headline, explanation, _ in metrics:
        with st.expander(f"**{name}** - {headline}", expanded=False):
            st.markdown(explanation)

    st.divider()

    # ── Models ────────────────────────────────────────────────────────────────
    st.markdown("## 🤖 The Models - What Are They?")
    st.caption(
        "Each model is a different mathematical recipe for spotting churners. "
        "The app trains all selected models and automatically picks the best one for your data."
    )

    models = [
        (
            "Logistic Regression",
            "The simple, transparent baseline.",
            "Imagine drawing a straight line through your data to separate churners from loyal customers. "
            "It's fast, easy to explain, and works surprisingly well when the signals in your data are clear. "
            "Best for smaller datasets or when you need a simple, explainable result.",
        ),
        (
            "Random Forest",
            "A crowd of decision trees voting together.",
            "Instead of one decision path, it builds hundreds of slightly different ones and takes a majority vote. "
            "Much more reliable than a single tree - one tree can be swayed by noise; "
            "a forest is harder to fool. A solid all-rounder that works well on most churn datasets.",
        ),
        (
            "XGBoost",
            "The competition winner - highly tuned gradient boosting.",
            "Builds trees that learn from each other's mistakes - each new tree specifically fixes "
            "what the previous ones got wrong. XGBoost is an extremely optimised version of this approach "
            "that has won more machine learning competitions than any other algorithm. "
            "Works well on medium-to-large datasets with many features.",
        ),
        (
            "LightGBM",
            "XGBoost's faster sibling - built for scale.",
            "Developed by Microsoft. Works similarly to XGBoost but is significantly faster "
            "on large datasets (50K+ rows). Often matches or beats XGBoost in accuracy while training "
            "in a fraction of the time.",
        ),
        (
            "CatBoost",
            "Gradient boosting that handles categories natively.",
            "Developed by Yandex. Excels when your data has many categorical columns "
            "(like product category, region, plan type). "
            "It handles them natively without needing manual encoding - it figures that out itself. "
            "Often the strongest performer on real-world business datasets.",
        ),
    ]

    for name, headline, explanation in models:
        with st.expander(f"**{name}** - {headline}", expanded=False):
            st.markdown(explanation)

    st.divider()

    # ── How the app picks the best model ─────────────────────────────────────
    st.markdown("## 🏆 How Does the App Pick the Best Model?")
    st.markdown(
        "Every model is scored on customers it did not see while being fitted, using 5-fold predictions on the "
        "training customers. The model with the **highest PR-AUC** wins: the best at ranking real leavers "
        "above stayers.\n\n"
        "The results are then reported as three separate questions, each with its own measure:\n\n"
        "- **Ranking:** PR-AUC, Top 10% Lift and ROC-AUC.\n"
        "- **Probabilities:** calibration error and log loss.\n"
        "- **Decision:** recall and precision, which depend on the threshold you choose.\n\n"
        "On public churn datasets, choosing by PR-AUC found the model that was best on new customers more "
        "often than a weighted mix of metrics. The results at the top are measured on a separate group of "
        "customers, held back until the winner has been chosen."
    )


def _date_options(df: pd.DataFrame, target_col: str, id_col):
    """How dates are read, and whether the model is tested on later customers."""
    from src.workflow import date_columns

    cache_key = (st.session_state.get("_df_hash"), target_col, id_col)
    if st.session_state.get("_dates_key") != cache_key:
        st.session_state["_date_cols"] = date_columns(df, target_col, id_col)
        st.session_state["_dates_key"] = cache_key
    dates = st.session_state["_date_cols"]
    if not dates:
        return None, None

    key = f"{st.session_state.get('_df_hash')}_{target_col}"
    with st.expander("📅 Dates and testing", expanded=True):
        st.markdown(
            f"Date columns found: **{', '.join(dates)}**. Each one is used as the number of days "
            "before a reference date, such as how long ago a customer last logged in."
        )
        reference = st.radio(
            "Count days back from",
            ["The latest date in the file", "A date I choose", "Each row's own date, from a column"],
            key=f"as_of_mode_{key}",
            help="Use the day the data was taken. If the file holds several monthly snapshots, "
                 "pick the column with each row's snapshot date.",
        )
        as_of = None
        if reference == "A date I choose":
            as_of = pd.Timestamp(st.date_input("Reference date", key=f"as_of_date_{key}"))
        elif reference.startswith("Each row"):
            as_of = st.selectbox("Column with each row's date", dates, key=f"as_of_col_{key}")

        test_on = st.radio(
            "Test the model on",
            ["Random customers", "The most recent customers"],
            key=f"split_mode_{key}",
            help="Testing on the most recent customers shows how the model does on the future, "
                 "which is how it will be used. It is usually a little lower than a random test.",
        )
        split_by = None
        if test_on == "The most recent customers":
            split_by = st.selectbox("Date that orders the customers", dates, key=f"split_col_{key}")
    return as_of, split_by


def _show_drift(report):
    """Whether these customers still look like the ones the model learned from."""
    from src.monitoring import SHIFT_PSI, WATCH_PSI

    if report is None:
        return
    st.markdown("#### Has The Data Changed Since Training?")
    if report.too_few:
        st.info(f"{report.n_customers} customers are too few to compare with the training data reliably.")
        return
    if report.retrain_recommended:
        moved = ", ".join(f"**{c}**" for c in report.shifted) or "none of the single columns"
        st.warning(
            f"🔁 **These customers differ clearly from the training data** (score shift "
            f"{report.score_psi:.2f}; columns that moved: {moved}). If this file is a segment you chose, "
            "such as one region, that is expected and the model still applies. If it is your whole "
            "customer base, the customers have changed: **train the model again on recent history**."
        )
    elif report.score_psi > WATCH_PSI or not report.columns.empty and report.columns["PSI"].max() > WATCH_PSI:
        st.info(
            f"🟡 Some change since training (score shift {report.score_psi:.2f}). "
            "Nothing that requires retraining yet, but keep an eye on the columns marked below."
        )
    else:
        st.success(f"✅ These customers look like the training customers (score shift {report.score_psi:.2f}).")
    with st.expander("Change per column (PSI)"):
        st.caption(
            f"PSI compares how this file's values spread across the training ranges. Below {WATCH_PSI} is "
            f"stable, above {SHIFT_PSI} is a real shift."
        )
        table = report.columns.assign(PSI=report.columns["PSI"].map("{:.3f}".format))
        st.dataframe(table, width="stretch", hide_index=True)


def _outcomes_section(model, prefix: str = ""):
    """Compare the model's scores for customers whose outcome is now known with what really happened."""
    from src.data_loader import load_uploaded_file
    from src.errors import ChurnAppError
    from src.outcomes import LEVEL_GAP, MIN_CUSTOMERS, MIN_EACH_OUTCOME, RANKING_DROP
    from src.pipeline_builder import suggest_positive_label, target_labels

    st.markdown(
        "Once the outcome period is over, upload the customers you scored earlier with a column "
        "saying who really left. This is the only way to know the model is still right, so do it "
        "every month or quarter."
    )
    upload = st.file_uploader(
        "Customers with their real outcome (CSV or Excel)", type=["csv", "xlsx", "xls"],
        key=f"{prefix}uploader_outcomes",
    )
    if upload is None:
        return
    try:
        df = load_uploaded_file(upload)
    except ChurnAppError as e:
        st.error(str(e))
        return

    columns = list(df.columns)
    two_valued = [c for c in columns if len(target_labels(df[c])) == 2]
    if not two_valued:
        st.error("No column with two values was found. Add a column saying who left, such as Yes/No.")
        return
    from src.profiler import detect_target_column
    # The column the model was trained on, or failing that the one that looks like churn
    trained_on = getattr(model, "target_col", "")
    likely = trained_on if trained_on in two_valued else detect_target_column(df)
    c1, c2, c3 = st.columns([2, 2, 2])
    outcome_col = c1.selectbox("Column with what really happened", two_valued, key=f"{prefix}outcome_col",
                               index=two_valued.index(likely) if likely in two_valued else 0)
    labels = [str(v) for v in target_labels(df[outcome_col])]
    suggested, _ = suggest_positive_label(df[outcome_col], outcome_col)
    left_label = c2.selectbox("Value that means the customer left", labels,
                              index=labels.index(str(suggested)), key=f"{prefix}outcome_label")
    as_of = None
    if model.artifact.schema.has_dates and not model.artifact.schema.as_of_column:
        latest = model.artifact.schema.latest_date(df) or pd.Timestamp.today().normalize()
        as_of = pd.Timestamp(c3.date_input(
            "Date the customers were scored", value=latest.date(), key=f"{prefix}outcome_as_of",
            help="Days are counted back from this date, as they were when the customers were scored."))

    if not st.button("📊 Check The Model", key=f"{prefix}check_outcomes"):
        return
    positive = target_labels(df[outcome_col])[labels.index(left_label)]
    try:
        with st.spinner("Scoring these customers and comparing with what happened…"):
            report = model.artifact.check_outcomes(df, outcome_col, positive, model.holdout, as_of=as_of)
    except ChurnAppError as e:
        st.error(str(e))
        return

    if report.too_few:
        st.info(
            f"{report.n_customers:,} customers with {report.n_churned:,} leavers are too few to judge. "
            f"At least {MIN_CUSTOMERS} customers and {MIN_EACH_OUTCOME} of each outcome are needed."
        )
        return

    then, now = report.then, report.now
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{now['roc_auc']:.3f}", delta=f"{now['roc_auc'] - then['roc_auc']:+.3f} vs training",
              help=f"95% range from resampling: {report.roc_interval[0]:.3f} to {report.roc_interval[1]:.3f}.")
    m2.metric("PR-AUC", f"{now['pr_auc']:.3f}", delta=f"{now['pr_auc'] - then['pr_auc']:+.3f} vs training",
              help="Depends on the churn rate, so it moves when the churn rate moves.")
    m3.metric("Recall", f"{now['recall']:.1%}", delta=f"{(now['recall'] - then['recall']) * 100:+.1f} pp vs training")
    m4.metric("Predicted vs Actual Churn", f"{report.mean_predicted:.1%} vs {report.actual_rate:.1%}")

    if report.ranking_worse:
        st.warning(
            f"🔁 **Retrain the model.** It separates leavers from stayers less well than when it was "
            f"trained (ROC-AUC {now['roc_auc']:.3f} against {then['roc_auc']:.3f}, a drop of more than "
            f"{RANKING_DROP}). What drives churn has probably changed."
        )
    elif report.level_off:
        st.warning(
            f"🔁 **Retrain the model.** It still ranks customers well, but it expected "
            f"{report.mean_predicted:.1%} to leave and {report.actual_rate:.1%} did. The churn rate has "
            f"moved by more than {LEVEL_GAP:.0%}, so the probabilities and the money estimates are off."
        )
    else:
        st.success(
            f"✅ **The model is holding up.** On {report.n_customers:,} customers whose outcome is known "
            f"it ranks as well as when it was trained, and its probabilities match what happened."
        )


def _show_drivers(drivers):
    """What moves churn risk across all customers, measured on customers the model did not train on."""
    if drivers is None or len(drivers) == 0:
        return
    _sec("🧭", "What Drives Churn")
    top = drivers.iloc[0]
    st.markdown(
        f"The column that moves churn risk the most is **{top['Column']}**: on average it shifts a customer's "
        f"risk by **{top['Average Effect']:.1f} percentage points**. The table lists every column the model "
        "uses, strongest first, with the values that raise risk."
    )
    st.dataframe(
        drivers.head(15),
        width="stretch",
        hide_index=True,
        column_config={
            "Average Effect": st.column_config.ProgressColumn(
                "Average Effect (pts)", format="%.1f", min_value=0.0,
                max_value=float(max(drivers["Average Effect"].max(), 0.1)),
                help="How much this column moves a customer's churn risk on average, in percentage points, "
                     "compared with the range of values other customers have."),
            "Raises Risk": st.column_config.TextColumn(
                "Raises Risk", help="For numbers, whether higher or lower values raise risk. "
                                    "For categories, the value that raises it most."),
        },
    )
    st.caption("This describes what the model learned, not cause and effect: a column can stand in for "
               "something else that really drives churn.")


def _show_lift(holdout: dict, n_holdout: int):
    """How much of the churn a campaign reaches when it can only contact the riskiest few."""
    _sec("🎯", "Who To Contact First")
    lift = holdout["lift_table"]
    top = lift.iloc[0]
    top_two = lift.iloc[1]
    st.markdown(
        f"On {n_holdout:,} customers the model had not seen, the riskiest **{top['Riskiest']}** included "
        f"**{top['Churners Caught So Far']:.0%}** of everyone who left, a churn rate "
        f"**{top['Lift']:.1f}×** the average. The riskiest **{top_two['Riskiest']}** included "
        f"**{top_two['Churners Caught So Far']:.0%}**. With a limited budget, start at the top of the list."
    )
    table = lift.assign(**{
        "Churn Rate": lift["Churn Rate"].map("{:.1%}".format),
        "Lift": lift["Lift"].map("{:.1f}×".format),
        "Churners Caught So Far": lift["Churners Caught So Far"].map("{:.0%}".format),
    })
    st.dataframe(table, width="stretch", hide_index=True)


def _economics_inputs(prefix: str, df_new: pd.DataFrame | None = None):
    """The business numbers that turn probabilities into money, and optionally each customer's value."""
    from src.business import Economics
    from src.pipeline_builder import detect_decimal, is_text_column, parse_numeric_column

    with st.expander("💰 Campaign economics — set these to see the money", expanded=False):
        st.caption("Use your own currency. Rough numbers are fine; the result shows how much they matter.")
        e1, e2, e3 = st.columns(3)
        value = e1.number_input(
            "Value of a customer who stays", min_value=1.0, value=1000.0, step=100.0,
            key=f"{prefix}econ_value",
            help="What you lose when a customer leaves, for example a year of profit from them.")
        cost = e2.number_input(
            "Cost of one retention contact", min_value=0.0, value=50.0, step=10.0,
            key=f"{prefix}econ_cost",
            help="A call, a discount or an offer, including staff time.")
        success = e3.slider(
            "Churners a contact keeps", min_value=0.05, max_value=1.0, value=0.25, step=0.05,
            key=f"{prefix}econ_success",
            help="Of the customers who would have left and were contacted, the share who stay. "
                 "Measure it with a test campaign if you can.")

        values = None
        if df_new is not None:
            numeric = {}
            for col in df_new.columns:
                raw = df_new[col]
                parsed = parse_numeric_column(raw, detect_decimal(raw) if is_text_column(raw) else ".")
                if parsed.notna().mean() >= 0.9 and parsed.nunique() > 2:
                    numeric[col] = parsed
            choice = st.selectbox(
                "Column with each customer's value (optional)", ["(same value for everyone)"] + list(numeric),
                key=f"{prefix}econ_value_col",
                help="For example yearly revenue or margin per customer. Risk is then weighed in money, "
                     "so a valuable customer at 30% risk can come before a small one at 50%.")
            if choice in numeric:
                values = numeric[choice].to_numpy()
    return Economics(value, cost, success), values


_SAVING_OFF = (
    "Saving models is turned off here. It needs a private signing key: set `CHURN_MODEL_KEY` "
    "in the app's secrets. Private installations create their own key automatically."
)


def _save_model_section(run):
    """Lets the user keep the trained model and use it again without retraining."""
    from src.model_store import SavedModel, save_model, signing_key

    _sec("💾", "Save This Model")
    key = signing_key()
    if key is None:
        st.info(_SAVING_OFF)
        return

    st.caption(
        "Keep this file to score new customers later without training again. It holds the model and "
        "the category values of your columns, but no customer rows. Treat it as confidential."
    )
    if st.session_state.get("model_file") is None:
        if st.button("Prepare Model File", key="prepare_model"):
            with st.spinner("Packing the model…"):
                st.session_state.model_file = save_model(SavedModel.from_run(run), key)
            st.rerun()
    else:
        st.download_button(
            "⬇️ Download Model File",
            data=st.session_state.model_file,
            file_name=f"churn_model_{run.best_model_name.lower().replace(' ', '_')}.churnmodel",
            mime="application/octet-stream",
            on_click="ignore",
        )


def _saved_model_tab():
    """Score new customers with a model saved earlier."""
    from src.errors import ChurnAppError
    from src.model_store import load_model, signing_key

    st.markdown("## 💾 Use a Saved Model")
    st.markdown("Upload a model file saved from this app to score new customers without training again.")
    key = signing_key()
    if key is None:
        st.info(_SAVING_OFF)
        return

    model_file = st.file_uploader("Model file", type=["churnmodel"], key="uploader_model")
    if model_file is not None:
        file_key = hash(model_file.getvalue())
        if st.session_state.get("_saved_model_key") != file_key:
            for name in ("_pred_file_key", "_pred_df_new", "_pred_report", "pred_raw_probs", "pred_reasons"):
                st.session_state[f"saved_{name}"] = None
            st.session_state.saved_model = None
            try:
                st.session_state.saved_model = load_model(model_file.getvalue(), key)
            except ChurnAppError as e:
                st.error(str(e))
                return
            st.session_state._saved_model_key = file_key

    saved = st.session_state.get("saved_model")
    if saved is None:
        return

    holdout = saved.holdout
    st.success(f"✅ Loaded **{saved.model_name}**, trained {saved.trained_at[:10]} on {saved.n_train:,} customers")
    _then = f"Measured on {saved.n_holdout:,} unseen customers when it was trained."
    k1, k2, k3 = st.columns(3)
    if "quality_score" in holdout:
        k1.metric(f"Prediction Quality · {holdout['quality_grade']}", f"{holdout['quality_score']:.0f}/100", help=_then)
    else:
        k1.metric("ROC-AUC", f"{holdout['roc_auc']:.4f}", help=_then)
    k2.metric("PR-AUC", f"{holdout['pr_auc']:.4f}", help=_then)
    if "lift_top10" in holdout:
        k3.metric("Top 10% Lift", f"{holdout['lift_top10']:.1f}×", help=_then)
    else:
        k3.metric("Recall", f"{holdout['recall']:.4f}", help=_then)
    st.caption(
        "These numbers are from when the model was trained. Customers change over time, "
        "so retrain regularly on recent history."
    )
    _show_drivers(getattr(saved, "drivers", None))
    st.divider()
    _sec("🔮", "Predict New Customers")
    _predict_section(saved, prefix="saved_")
    st.divider()
    _sec("📊", "Check Past Predictions Against What Happened")
    _outcomes_section(saved, prefix="saved_")


def _predict_section(model, prefix: str = ""):
    """New customer prediction UI, for a model just trained or loaded from a file."""
    from src.data_loader import load_uploaded_file
    from src.errors import ChurnAppError
    from src.explain import MAX_CUSTOMERS, TOP_REASONS, format_reason
    from src.schema import require_usable
    from src.utils import to_csv_bytes

    artifact = model.artifact
    state = st.session_state
    file_key_name = f"{prefix}_pred_file_key"
    df_name       = f"{prefix}_pred_df_new"
    report_name   = f"{prefix}_pred_report"
    probs_name    = f"{prefix}pred_raw_probs"
    reasons_name  = f"{prefix}pred_reasons"
    drift_name    = f"{prefix}pred_drift"

    # ── File upload ───────────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        new_file = st.file_uploader(
            "Upload new customers file (CSV or Excel) - WITHOUT the churn column",
            type=["csv", "xlsx", "xls"],
            key=f"{prefix}uploader_new",
        )
    with c2:
        st.info(
            "The file is checked against the training data before anything is scored.\n"
            "Upload raw customer data - same columns as training, minus the target."
        )

    if new_file is None:
        # HuggingFace resets the file uploader on every slider rerun.
        # If we already loaded the file into session_state, keep using it.
        if state.get(df_name) is None:
            state[probs_name] = None
            return
    else:
        file_key = hash(new_file.getvalue())
        if state.get(file_key_name) != file_key:
            state[probs_name] = None   # clear old results on new file
            state[df_name] = None
            try:
                df_new = load_uploaded_file(new_file)
                report = artifact.check(df_new)
                require_usable(report)
            except ChurnAppError as e:
                st.error(str(e))
                return
            state[file_key_name] = file_key
            state[df_name]       = df_new.reset_index(drop=True)
            state[report_name]   = report

    df_new = state[df_name]
    report = state[report_name]
    st.success(f"✅ Loaded **{len(df_new):,}** new customers")

    as_of = None
    if artifact.schema.has_dates:
        if artifact.schema.as_of_column:
            st.caption(f"📅 Days are counted back from each row's **{artifact.schema.as_of_column}**.")
        else:
            latest = artifact.schema.latest_date(df_new) or pd.Timestamp.today().normalize()
            as_of = pd.Timestamp(st.date_input(
                "📅 Count days back from", value=latest.date(), key=f"{prefix}as_of_{state[file_key_name]}",
                help="The date this file describes, usually the day it was exported. "
                     "It starts at the latest date in the file.",
            ))

    allow_missing = True
    if report.missing:
        st.error(
            f"**{len(report.missing)} of {report.n_columns} columns the model uses are missing:** "
            f"{', '.join(report.missing)}. Predictions without them are less accurate."
        )
        allow_missing = st.checkbox(
            "Score anyway, treating the missing columns as unknown", key=f"{prefix}pred_allow_missing")
    for _note in report.warnings():
        st.warning(_note)

    # ── Who gets flagged ──────────────────────────────────────────────────────
    from src.business import campaign_estimate, contact_everyone, file_estimate, risk_levels

    thr_curve = model.holdout["threshold_curve"]
    economics, values = _economics_inputs(prefix, df_new)
    goal = st.radio(
        "Choose who gets flagged by",
        ["Best balance (F1)", "Most profit (your numbers above)", "How many customers I can contact"],
        horizontal=True, key=f"{prefix}threshold_goal",
    )
    by_profit = goal.startswith("Most profit")
    by_capacity = goal.startswith("How many")
    per_customer_value = by_profit and values is not None
    breakeven = economics.breakeven
    capacity = None
    selected_thr = None

    if by_capacity:
        capacity = int(st.number_input(
            "Customers you can contact", min_value=1, max_value=len(df_new),
            value=min(100, len(df_new)), step=10, key=f"{prefix}capacity",
            help="The riskiest customers in this file are flagged, up to this number."))
        st.caption(f"The **{capacity:,} riskiest** customers in this file will be flagged.")
    elif per_customer_value:
        st.caption(
            "Each customer is flagged when contacting them is expected to pay off: churn probability × share "
            "kept × that customer's value is more than the cost of the contact. Valuable customers are "
            "contacted at lower risk than small ones."
        )
    else:
        auto_thr = (float(np.clip(round(breakeven, 2), 0.02, 0.90)) if by_profit
                    else round(float(artifact.threshold), 2))
        st.markdown("#### Sensitivity Setting")
        sl_col, hint_col = st.columns([3, 2])
        with sl_col:
            selected_thr = st.slider(
                "Decision Threshold",
                min_value=0.02, max_value=0.90,
                value=auto_thr, step=0.01,
                key=f"{prefix}threshold_slider_{goal}",
                help=(
                    "A customer is flagged when their churn probability reaches this value.\n\n"
                    "Lower → more customers flagged (higher Recall, more false alarms).\n\n"
                    "Higher → fewer customers flagged (fewer false alarms, may miss some churners)."
                ),
            )
            row = thr_curve.iloc[(thr_curve["threshold"] - selected_thr).abs().argsort()[:1]]
            st.caption(
                f"At threshold **{selected_thr:.2f}** — measured on customers the model had not seen: "
                f"Recall **{float(row['recall'].values[0]):.0%}** · "
                f"Precision **{float(row['precision'].values[0]):.0%}**"
            )

            with st.expander(f"Why {auto_thr:.2f}?"):
                if by_profit:
                    clipped = "" if 0.02 <= breakeven <= 0.90 else (
                        f" The exact value, {breakeven:.3f}, is outside the 0.02–0.90 range, so the nearest end is used.")
                    st.markdown(
                        f"A contact pays for itself when the customer's churn probability is above "
                        f"cost ÷ (share kept × value) = {economics.cost:,.0f} ÷ ({economics.success:.0%} × "
                        f"{economics.value:,.0f}) = **{breakeven:.3f}**. Below that, the expected saving is smaller "
                        f"than the cost of the contact. This works because the probabilities are calibrated."
                        + clipped
                    )
                else:
                    st.markdown(
                        f"**{auto_thr:.2f}** gave the highest **F1 score**, the balance between catching "
                        f"churners (Recall) and avoiding false alarms (Precision). Every threshold from 0.02 "
                        f"to 0.90 was tested on calibrated predictions for customers each fold had not seen."
                    )

        with hint_col:
            if abs(selected_thr - auto_thr) <= 0.02:
                st.info("💰 **Recommended setting** — the most profitable cut for your numbers." if by_profit else
                        "⚖️ **Recommended setting** — the best balance between catching churners and avoiding false alarms.")
            elif selected_thr < auto_thr:
                st.info("🔍 **More sensitive than recommended** — catches more churners, but flags more loyal customers too.")
            else:
                st.info("🎯 **Stricter than recommended** — fewer false alarms, but more real churners will be missed.")

        estimate = campaign_estimate(thr_curve, model.holdout["actual_rate"], economics)
        at_threshold = estimate.iloc[(estimate["threshold"] - selected_thr).abs().argsort()[:1]].iloc[0]
        everyone = contact_everyone(model.holdout["actual_rate"], economics)
        st.markdown(
            f"**At this threshold, per 1,000 customers like the unseen ones:** contact "
            f"**{at_threshold['contacted']:,.0f}**, reach **{at_threshold['churners_reached']:,.0f}** of the "
            f"{model.holdout['actual_rate'] * 1000:,.0f} who would leave, keep about "
            f"**{at_threshold['customers_saved']:,.0f}**. Net result **{at_threshold['net']:,.0f}**, against "
            f"{everyone:,.0f} for contacting everyone and 0 for doing nothing."
        )

    # ── Predict button ────────────────────────────────────────────────────────
    if st.button("🔮 Predict Churn", type="primary", key=f"{prefix}predict_btn", disabled=not allow_missing):
        with st.spinner("Scoring customers and working out the reasons…"):
            try:
                probabilities = artifact.predict_proba(df_new, allow_missing=True, as_of=as_of)
                state[probs_name] = probabilities
                state[reasons_name] = artifact.explain(df_new, probabilities, as_of=as_of)
                state[drift_name] = artifact.monitor(df_new, probabilities, as_of=as_of)
            except ChurnAppError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                return

    # ── Results (apply the current choice to stored probabilities) ────────────
    raw_probs = state.get(probs_name)
    if raw_probs is None:
        return

    gains = economics.gain(raw_probs, values)
    if by_capacity:
        flagged_mask = np.zeros(len(raw_probs), dtype=bool)
        flagged_mask[np.argsort(-raw_probs, kind="stable")[:capacity]] = True
    elif per_customer_value:
        flagged_mask = gains > 0
    else:
        flagged_mask = raw_probs >= selected_thr
    total     = len(raw_probs)
    n_flagged = int(flagged_mask.sum())
    levels    = risk_levels(raw_probs, flagged_mask, model.churn_rate)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers",  f"{total:,}")
    m2.metric("Flagged At Risk",  f"{n_flagged:,}", help="High risk: the customers to contact.")
    m3.metric("Medium Risk",      f"{int((levels == 'Medium').sum()):,}",
              help="Riskier than the average customer, but not flagged. Worth watching.")
    m4.metric("Low Risk",         f"{int((levels == 'Low').sum()):,}",
              help="No riskier than the average customer.")

    campaign = file_estimate(raw_probs, flagged_mask, economics, values)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Expected Churners Among Flagged", f"{campaign['expected_churners_reached']:,.0f}",
              help="The sum of the flagged customers' churn probabilities.")
    b2.metric("Share Of Expected Churners Reached", f"{campaign['share_of_expected_churners']:.0%}",
              help="Of all the customers in this file expected to leave, the share among the flagged ones.")
    b3.metric("Expected Customers Kept", f"{campaign['expected_saved']:,.0f}",
              help="Expected churners among the flagged, times the share a contact keeps.")
    b4.metric("Expected Net Result", f"{campaign['net']:,.0f}",
              help="Value of the customers kept minus the cost of contacting every flagged customer. "
                   "Uses your numbers in Campaign economics.")
    if values is not None:
        st.metric("Value At Risk In This File", f"{float(np.nansum(raw_probs * np.nan_to_num(values))):,.0f}",
                  help="Each customer's value times their churn probability, added up: the value expected to leave.")

    # ── Risk compared with history ────────────────────────────────────────────
    expected = float(raw_probs.mean())
    historic = model.churn_rate
    diff_pp  = (expected - historic) * 100
    # A small file wanders by chance, so the alert scales with its size
    tolerance_pp = max(5.0, 300 * float(raw_probs.std()) / np.sqrt(max(total, 1)))

    st.markdown("#### Risk Compared With History")
    cr1, cr2, cr3 = st.columns(3)
    cr1.metric("Historical Churn Rate", f"{historic:.1%}",
               help="Churn rate observed in the training dataset.")
    cr2.metric("Average Predicted Risk", f"{expected:.1%}",
               help="The average churn probability across this file: the share of these customers the model expects to leave.")
    cr3.metric("Difference", f"{diff_pp:+.1f} pp", delta=f"{diff_pp:+.1f} pp", delta_color="inverse",
               help="Difference in percentage points between average predicted risk and the historical churn rate.")

    if total < _MIN_CUSTOMERS_TO_COMPARE:
        st.info(f"{total} customers are too few to compare with the historical churn rate.")
    elif abs(diff_pp) > tolerance_pp:
        st.warning(
            f"⚠️ This file's average predicted risk ({expected:.1%}) is {abs(diff_pp):.1f} points away "
            f"from the historical churn rate ({historic:.1%}). These customers may really be riskier or "
            "safer, or they may differ from the customers the model learned from."
        )
    else:
        st.success(
            f"✅ This file's average predicted risk ({expected:.1%}) is in line with the historical "
            f"churn rate ({historic:.1%})."
        )

    _show_drift(state.get(drift_name))

    st.divider()

    display_df = df_new.copy()
    display_df["Risk Score"]        = np.round(raw_probs * 100).astype(int)
    display_df["Risk Level"]        = levels
    display_df["Flagged At Risk"]   = np.where(flagged_mask, "Yes", "No")
    display_df["Churn Probability"] = np.round(raw_probs, 4)
    if values is not None:
        display_df["Value At Risk"] = np.round(raw_probs * np.nan_to_num(values), 2)
    display_df["Expected Gain From Contact"] = np.round(gains, 2)
    for i in range(TOP_REASONS):
        display_df[f"Reason {i + 1}"] = [
            format_reason(reasons[i]) if flagged and i < len(reasons) else ""
            for flagged, reasons in zip(flagged_mask, state[reasons_name])
        ]

    st.caption(
        "Risk Score is the churn probability on a 0–100 scale. Risk Level: **High** is flagged, **Medium** is "
        "riskier than the average customer but not flagged, **Low** is the rest. Reasons are shown for flagged "
        "customers: **+18 pts** means this value adds about 18 percentage points of risk, compared with the "
        "range of values other customers have. Each column is checked on its own, so the points don't add up "
        "to the total risk."
        + (f" Only the {MAX_CUSTOMERS:,} riskiest customers are explained." if total > MAX_CUSTOMERS else "")
    )
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.download_button(
        "⬇️ Download Predictions CSV",
        data=to_csv_bytes(display_df),
        file_name="churn_predictions.csv",
        mime="text/csv",
        key=f"{prefix}download_predictions",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
for _k in ["df", "run", "training_done"]:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("📊 Churn Analytics Studio")
st.caption("AI & Machine Learning Powered Customer Churn Prediction Platform")
st.divider()

_tab_pipeline, _tab_saved, _tab_guide = st.tabs(["🚀 Pipeline", "💾 Saved Model", "📖 How It Works"])

with _tab_guide:
    _show_guide()

with _tab_saved:
    _saved_model_tab()

with _tab_pipeline:
    # ── Step 1: Upload ────────────────────────────────────────────────────────
    _sec("📂", "Step 1 - Upload Training Dataset")

    up_col, hint_col = st.columns([2, 1])
    with up_col:
        hist_file = st.file_uploader(
            "Historical data with churn labels (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="uploader_hist",
        )
    with hint_col:
        st.info(
            "**Requirements:**\n"
            "- At least 100 rows\n"
            "- A churn column with two values, such as Yes/No or 1/0\n"
            "- CSV or Excel format\n"
            "- Auto-detects delimiter & encoding"
        )

    if hist_file is not None:
        try:
            # UploadedFile keeps its read position across reruns — without this
            # seek every rerun after the first reads b"" and the file is lost.
            hist_file.seek(0)
            _file_bytes = hist_file.read()
            _file_hash  = hash(_file_bytes)
        except Exception:
            _file_bytes = None
            _file_hash  = None

        # Reload on a genuinely new file, or whenever we somehow lost the
        # DataFrame (e.g. the session was reset by a dropped connection)
        _needs_load = (
            _file_hash != st.session_state.get("_df_hash")
            or st.session_state.df is None
        )
        if _file_bytes and _needs_load:
            _prog = st.progress(0, text="Loading…")
            try:
                import io as _io
                from src.data_loader import load_uploaded_file
                _prog.progress(5, text="Detecting delimiter & encoding…")
                _bio = _io.BytesIO(_file_bytes)
                _bio.name = hist_file.name
                _df = load_uploaded_file(_bio)
                _prog.progress(10, text="Done ✅")
                st.session_state.run = None
                st.session_state.training_done = None
                _forget_job()
                st.session_state.df      = _df
                st.session_state._df_hash = _file_hash
                time.sleep(0.2)
                _prog.empty()
            except Exception as _e:
                _prog.empty()
                st.error(f"Could not load file: {_e}")
                # Do NOT clear st.session_state.df — keep the last valid DataFrame

    df = st.session_state.df
    job = _current_job()
    if job is not None and job.done:
        _attach_finished_job(job)

    if df is None:
        # A reconnect starts a new session without the uploaded file, but the job carries on
        if job is not None and not job.done:
            st.info("Training is running on the server. Its results will appear here when it finishes.")
            _training_progress(job.id)
            st.stop()
        if st.session_state.training_done:
            st.info(
                "The training file is not loaded in this session any more, but the model trained on it is. "
                "Its results are below, and you can score new customers with it. Upload the file again "
                "to train a new model."
            )
            _show_trained_sections()
            st.stop()
        st.info("👆 Upload a dataset to begin.")
        st.stop()

    st.success(f"✅ Loaded **{len(df):,} rows × {df.shape[1]} columns**")
    with st.expander("Preview first rows", expanded=False):
        st.dataframe(df.head(5), width="stretch")

    st.divider()

    # ── Step 2: Column Detection ──────────────────────────────────────────────
    _sec("🔍", "Step 2 - Column Detection")

    from src.profiler import detect_id_column, detect_target_column

    auto_id     = detect_id_column(df)
    auto_target = detect_target_column(df, exclude=[auto_id] if auto_id else [])
    cols_list   = df.columns.tolist()

    ca, cb, cc1, cc2, cc3 = st.columns([3, 3, 1.2, 1.2, 2])
    with ca:
        id_col = st.selectbox(
            "Customer ID column",
            options=["(none)"] + cols_list,
            index=(cols_list.index(auto_id) + 1) if auto_id in cols_list else 0,
        )
        id_col = None if id_col == "(none)" else id_col
        if auto_id:
            st.caption(f"Auto-detected: **{auto_id}**")

    with cb:
        _first_guess = auto_target or next(
            (c for c in cols_list if c != auto_id and df[c].nunique() == 2), cols_list[0])
        target_col = st.selectbox(
            "Churn / Target column ★",
            options=cols_list,
            index=cols_list.index(_first_guess),
        )
        if auto_target:
            st.caption(f"Auto-detected: **{auto_target}**")
        else:
            st.caption("Could not detect it. Please choose the churn column.")

    with cc1:
        st.metric("Rows", f"{len(df):,}")

    with cc2:
        st.metric("Columns", df.shape[1])

    from src.errors import ChurnAppError
    from src.pipeline_builder import convert_target, suggest_positive_label, target_labels
    from src.workflow import screen_columns

    _labels = target_labels(df[target_col])
    if len(_labels) != 2:
        with cc3:
            st.metric("Values in target", len(_labels))
        st.error(
            f"**{target_col}** has {len(_labels)} different values. The churn column needs exactly two: "
            "one for customers who left and one for customers who stayed. Choose the right column above."
        )
        st.stop()

    _suggested, _confident = suggest_positive_label(df[target_col], target_col)
    _label_strs = [str(v) for v in _labels]
    positive_label_str = st.selectbox(
        "Which value means the customer left?",
        options=_label_strs,
        index=_label_strs.index(str(_suggested)),
        key=f"positive_label_{st.session_state.get('_df_hash')}_{target_col}",
        help="Every prediction depends on this. Choosing the wrong value would invert the model.",
    )
    positive_label = _labels[_label_strs.index(positive_label_str)]
    if not _confident:
        st.warning(
            f"The values in **{target_col}** don't say which one means the customer left. "
            f"**{positive_label_str}** is only a guess, because it is the rarer value. Please check it."
        )

    with cc3:
        _y_preview = convert_target(df[target_col], positive_label).dropna()
        st.metric(
            "Churned Customers",
            f"{int(_y_preview.sum()):,} ({_y_preview.mean():.1%})",
            help="Number and percentage of customers who churned in the historical data.",
        )

    # ── Leak screening ────────────────────────────────────────────────────────
    _leak_key = (st.session_state.get("_df_hash"), target_col, id_col, positive_label_str)
    if st.session_state.get("_leak_key") != _leak_key:
        with st.spinner("Checking columns for data leaks…"):
            try:
                st.session_state["_leak_findings"] = screen_columns(df, target_col, id_col, positive_label)
            except ChurnAppError as _e:
                st.error(str(_e))
                st.stop()
        st.session_state["_leak_key"] = _leak_key
    _findings = st.session_state["_leak_findings"]

    exclude_cols = []
    if _findings:
        st.warning(
            "⚠️ **Possible data leaks.** A column that is only known after a customer leaves makes a model "
            "look excellent in testing and useless in real life. These columns are left out unless you "
            "know they are available before the customer leaves:\n\n"
            + "\n".join(f"- **{f.column}**: {f.reason}" for f in _findings)
        )
        _names = [f.column for f in _findings]
        exclude_cols = st.multiselect(
            "Columns to leave out of the model", options=_names, default=_names,
            key=f"exclude_cols_{st.session_state.get('_df_hash')}_{target_col}",
        )

    st.divider()

    # ── Predictive Signal Assessment ──────────────────────────────────────────
    _show_signal_assessment(df, target_col, id_col, positive_label, exclude_cols)

    # ── Steps 3–11: Run Pipeline ──────────────────────────────────────────────
    from src.config import PRODUCTION_MODELS
    _sec("🚀", "Step 3 - Run Full Machine Learning Pipeline")

    from src.config import SEARCH_PRESETS, DEFAULT_SEARCH_DEPTH

    model_col, btn_col = st.columns([3, 1])
    with model_col:
        all_option   = "All Models"
        model_choices = [all_option] + PRODUCTION_MODELS
        sel_col, depth_col = st.columns([3, 1])
        with sel_col:
            selected_raw = st.multiselect(
                "Models to train",
                options=model_choices,
                default=[all_option],
                help="Select one or more models. Choose 'All Models' to train all 5 production models.",
            )
        with depth_col:
            _depths = list(SEARCH_PRESETS.keys())
            search_depth = st.selectbox(
                "Search depth",
                options=_depths,
                index=_depths.index(DEFAULT_SEARCH_DEPTH),
                help=(
                    "How many hyperparameter combinations to try per model. "
                    "Deeper searches can find a slightly better model but take "
                    "considerably longer. On a small shared CPU a very long run "
                    "can outlast the browser connection, so start shallower there."
                ),
            )
        budget = SEARCH_PRESETS[search_depth]

        if all_option in selected_raw or not selected_raw:
            selected_models = PRODUCTION_MODELS.copy()
        else:
            selected_models = [m for m in PRODUCTION_MODELS if m in selected_raw]

        st.caption(
            f"Training: **{', '.join(selected_models)}**  ·  "
            f"Optuna trials: **{budget['optuna_trials']}**  ·  "
            f"CV folds: **{budget['hpo_cv_folds']}**  ·  Tuning: **ROC-AUC**"
        )
        with st.expander("⚙️ Runtime details", expanded=False):
            from src.runtime_env import ON_CLOUD as _oc, N_JOBS as _nj, CPU_SOURCE as _src
            import sklearn as _sk
            st.markdown(
                f"- Environment: **{'hosted' if _oc else 'local'}**\n"
                f"- Parallel workers (`n_jobs`): **{_nj}** — {_src}\n"
                f"- Threads per worker (`OMP_NUM_THREADS`): **{os.environ.get('OMP_NUM_THREADS', 'default')}**\n"
                f"- numpy **{np.__version__}** · pandas **{pd.__version__}** · "
                f"scikit-learn **{_sk.__version__}**"
            )

        with st.expander("ℹ️ When to use each model", expanded=False):
            st.markdown(
                "| Model | Best when… | Watch out for |\n"
                "|---|---|---|\n"
                "| **Logistic Regression** | Dataset is small (<5K rows), features are mostly numeric, you need a fast baseline or an explainable result | Struggles with complex non-linear patterns |\n"
                "| **Random Forest** | General-purpose - works well on most churn datasets; handles missing values and mixed feature types gracefully | Can be slow on very large datasets (>200K rows) |\n"
                "| **XGBoost** | Medium-to-large datasets with many features; strong on tabular data with feature interactions | Needs more tuning; sensitive to imbalanced data without proper class weights |\n"
                "| **LightGBM** | Large datasets (>50K rows); fastest training time of the boosting models | May overfit on small datasets (<1K rows) |\n"
                "| **CatBoost** | Dataset contains many categorical features (e.g. product category, region, plan type); handles them natively without encoding | Slowest to train; less benefit if data is mostly numeric |\n"
                "\n💡 **Not sure?** Leave 'All Models' selected - the app will train all five and automatically pick the best one."
            )
        as_of, split_by = _date_options(df, target_col, id_col)
    with btn_col:
        run = st.button("🚀 Run Pipeline", type="primary", width="stretch")
        _eta_low, _eta_high = _estimate_runtime(df, selected_models, budget)
        st.caption(f"⏱️ Est. runtime: **{_eta_low} - {_eta_high}**")

    if run:
        st.session_state.run = None
        st.session_state.training_done = None
        _run_pipeline(df, id_col, target_col, selected_models, budget, positive_label, exclude_cols,
                      as_of, split_by)

    if job is not None and not job.done:
        _training_progress(job.id)
    elif job is not None and job.error:
        _show_job_error(job)

    if st.session_state.training_done:
        _show_trained_sections()

    st.divider()
    st.caption(
        "Churn Analytics Studio  ·  Built with Streamlit & scikit-learn  ·  "
        "Best model selected by PR-AUC  ·  "
        "Models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost"
    )
