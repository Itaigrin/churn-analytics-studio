"""
ChurnApp Pro - Professional Customer Churn Prediction
=====================================================
Streamlit application implementing a full ML pipeline:

  Step 1   Upload & Validate Dataset
  Step 2   Automatic Data Profiling
  Step 3   Data Cleaning (inside sklearn Pipelines - zero data leakage)
  Step 4   Lightweight Feature Engineering
  Step 6   Train / Test Split (stratified 80/20, random_state=42)
  Step 7   Preprocessing (ColumnTransformer, conditiohnal scaling per model)
  Step 8   Model Training (Fast / Balanced / Best Accuracy modes)
  Step 9   Hyperparameter Optimisation (RandomizedSearchCV / Optuna)
  Step 10  Evaluation (6 metrics, correct best-model selection for imbalanced data)
  Step 11  Professional Model Analysis (B2C business framing)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

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

def _sec(icon: str, title: str):
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def _show_signal_assessment(df: pd.DataFrame, target_col: str, id_col):
    """Displays the Predictive Signal Assessment results."""
    from src.signal_assessor import assess_predictive_signal

    _data_hash = hash(df.to_csv(index=False)[:5000])
    cache_key  = f"signal_{len(df)}_{target_col}_{id_col}_{_data_hash}"
    if st.session_state.get("_signal_cache_key") != cache_key:
        with st.spinner("Analysing predictive signal…"):
            result = assess_predictive_signal(df, target_col, id_col)
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
    st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

    st.divider()


def _estimate_runtime(df: pd.DataFrame, selected_models: list) -> tuple[str, str]:
    """Estimate pipeline runtime based on data size and selected models."""
    import os as _os
    _on_cloud = _os.path.exists("/mount/src")

    from src.config import PIPELINE_CONFIG
    n_trials  = max(PIPELINE_CONFIG.get("optuna_trials", 30), 1)
    cv_folds  = PIPELINE_CONFIG["cv_folds"]
    n_rows    = len(df)

    # Effective trials after dataset-size scaling in trainer.py
    if n_rows > 100_000:
        n_trials = max(10, n_trials // 3)
    elif n_rows > 50_000:
        n_trials = max(15, n_trials // 2)

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


def _run_pipeline(df: pd.DataFrame, id_col, target_col: str, selected_models: list):
    """Executes Steps 3–10 and stores results in st.session_state."""
    from sklearn.model_selection import train_test_split
    from src.config import (
        RANDOM_STATE, TEST_SIZE, IMBALANCE_LOW, IMBALANCE_HIGH, VERY_LARGE_DATASET_ROWS,
    )
    from src.pipeline_builder import raw_clean, convert_target
    from src.profiler import profile_dataset
    from src.feature_engineering import engineer_features
    from src.trainer import train_all_models
    from src.evaluator import evaluate_all, pick_best_model
    from src.utils import ensure_optional_libraries

    prog_container = st.empty()
    t_start        = time.time()

    _current_pct = [0]
    _current_msg = ["Starting…"]

    def _render(pct, msg):
        prog_container.markdown(
            f"""
<div style="background:#1e2130;border:1px solid #3a3f5c;border-radius:10px;padding:16px 20px;margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <span style="color:#a0aec0;font-size:13px">Running Pipeline…</span>
    <span style="color:#63b3ed;font-weight:700;font-size:18px">{pct}%</span>
  </div>
  <div style="background:#2d3748;border-radius:6px;height:10px;overflow:hidden">
    <div style="background:linear-gradient(90deg,#4299e1,#63b3ed);width:{pct}%;height:100%;border-radius:6px;transition:width 0.3s ease"></div>
  </div>
  <div style="margin-top:10px;color:#e2e8f0;font-size:13px">⏳ {msg}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    def upd(pct: int, msg: str):
        _current_pct[0] = pct
        _current_msg[0] = msg
        _render(pct, msg)

    try:
        upd(2, "Checking optional libraries (XGBoost, LightGBM, CatBoost, SHAP, Optuna)…")
        lib_status = ensure_optional_libraries()
        st.session_state.lib_status = lib_status

        # Step 3: Raw cleaning
        upd(5, "Step 3 - Raw cleaning (text normalisation, boolean→0/1, numeric strings)…")
        df_clean = raw_clean(df)

        def _std(c):
            return c.strip().lower().replace(" ", "_").replace("-", "_") if c else None

        id_col_std     = _std(id_col)
        target_col_std = _std(target_col)

        if target_col_std not in df_clean.columns:
            st.error(f"Target column '{target_col}' not found. Available: {list(df_clean.columns)[:10]}")
            prog_container.empty(); return

        try:
            y = convert_target(df_clean[target_col_std])
        except Exception as e:
            st.error(f"Target column error: {e}"); prog_container.empty(); return  # noqa

        ids = (df_clean[id_col_std].copy()
               if id_col_std and id_col_std in df_clean.columns
               else pd.Series(range(len(df_clean)), name="row_id"))

        drop_initial = [c for c in [id_col_std, target_col_std] if c and c in df_clean.columns]
        X_raw = df_clean.drop(columns=drop_initial, errors="ignore")

        # Step 2: Data Profiling
        upd(10, "Step 2 - Automatic data profiling (types, quality, leakage check)…")
        profile = profile_dataset(df_clean, target_col=target_col_std, id_col=id_col_std)
        X_raw   = X_raw.drop(columns=list(profile["drop_cols"].keys()), errors="ignore")

        for w in profile["leakage_warnings"]:
            st.warning(w)
        for w in profile["general_warnings"][:3]:
            st.info(w)

        n             = len(X_raw)
        churn_rate    = float(y.mean())
        is_imbalanced = churn_rate < IMBALANCE_LOW or churn_rate > IMBALANCE_HIGH

        if n > VERY_LARGE_DATASET_ROWS:
            st.warning(f"⚠️ Very large dataset ({n:,} rows). Consider stratified sampling.")
        if is_imbalanced:
            st.warning(
                f"⚠️ Only **{churn_rate:.1%}** of customers in this dataset churned. "
                "When churners are rare, models tend to ignore them and predict 'will stay' for everyone. "
                "The app automatically adjusts for this so the model pays extra attention to churners and doesn't miss them."
            )

        upd(15, f"✅ {n:,} rows · {X_raw.shape[1]} features · churn rate {churn_rate:.1%}")

        # Step 4: Feature Engineering
        upd(20, "Step 4 - Feature engineering (ratios, products, MI-selected features)…")
        X_raw, new_feat_names = engineer_features(
            X_raw, profile["numeric_cols"],
            cat_cols=profile["categorical_cols"],
            bool_cols=profile["boolean_cols"],
        )
        profile["numeric_cols"] = profile["numeric_cols"] + new_feat_names

        # Step 6: Train/Test Split
        upd(28, "Step 6 - Stratified 80/20 split (random_state=42)…")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        st.success(
            f"✅ Train: **{len(X_train):,}** · Test: **{len(X_test):,}** "
            "(split BEFORE imputation/encoding → zero data leakage)"
        )

        # Step 4b: MI-based feature selection (uses only X_train/y_train — no leakage)
        if new_feat_names:
            upd(30, "Step 4b - Selecting engineered features by mutual information (train set only)…")
            from src.feature_engineering import select_engineered_features
            selected_feat_names = select_engineered_features(X_train, y_train, new_feat_names)
            discarded = [c for c in new_feat_names if c not in set(selected_feat_names)]
            if discarded:
                X_train = X_train.drop(columns=discarded, errors="ignore")
                X_test  = X_test.drop(columns=discarded, errors="ignore")
                X_raw   = X_raw.drop(columns=discarded, errors="ignore")
                profile["numeric_cols"] = [c for c in profile["numeric_cols"]
                                           if c not in set(discarded)]
            new_feat_names = selected_feat_names
            if selected_feat_names:
                st.info(f"✅ Added engineered features: {', '.join(selected_feat_names)}")


        # Steps 8 & 9: Train + HPO
        model_list_str = ", ".join(selected_models)
        upd(33, f"Steps 8–9 - Training {len(selected_models)} models: {model_list_str}…")

        def _cb(name, i, total, detail):
            pct = 33 + int(i / max(total, 1) * 57)
            upd(pct, f"Training **{name}** ({i + 1}/{total}) · {detail}")

        results = train_all_models(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            profile=profile,
            selected_models=selected_models,
            n_samples=len(X_train),
            is_imbalanced=is_imbalanced,
            progress_callback=_cb,
        )

        # Step 10: Evaluate + pick best model
        upd(88, "Step 10 - Evaluating on test set…")
        results   = evaluate_all(results, X_train, y_train, X_test, y_test)
        best_name = pick_best_model(results, actual_positives=int(y_test.sum()))

        # Step 11: Cross-validation evaluation — uses the same threshold as production predictions
        upd(93, "Step 11 - Cross-validation evaluation (5 folds) for reliable metrics…")
        from src.evaluator import evaluate_model_cv
        X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        y_full = pd.concat([
            pd.Series(y_train.values, name=y_train.name),
            pd.Series(y_test.values,  name=y_test.name),
        ], axis=0).reset_index(drop=True)
        # Pass the optimized threshold so CV Recall matches what predictions will produce
        opt_thr_for_cv = results[best_name].get("best_threshold", 0.50)
        cv_metrics = evaluate_model_cv(
            results[best_name]["model"], X_full, y_full,
            cv_folds=5, threshold=opt_thr_for_cv,
        )
        results[best_name].update(cv_metrics)

        # Optional: probability calibration (only if it improves Brier Score)
        if st.session_state.get("calibrate_enabled", False):
            upd(96, "Calibrating probabilities (comparing Brier Score before/after)…")
            from src.evaluator import calibrate_model
            cal_model, was_calibrated, brier_before, brier_after = calibrate_model(
                results[best_name]["model"], X_train, y_train, X_test, y_test,
            )
            results[best_name]["calibrated"]    = was_calibrated
            results[best_name]["brier_before"]  = brier_before
            results[best_name]["brier_after"]   = brier_after
            if was_calibrated:
                results[best_name]["model"]  = cal_model
                results[best_name]["y_prob"] = cal_model.predict_proba(X_test)[:, 1]
                st.success(
                    f"✅ Calibration improved Brier Score: "
                    f"{brier_before:.4f} → {brier_after:.4f} (Δ {brier_after - brier_before:+.4f})"
                )
            else:
                note = ""
                if brier_before is not None and brier_after is not None:
                    note = (f" — calibrated Brier ({brier_after:.4f}) did not improve "
                            f"over original ({brier_before:.4f})")
                st.info(f"ℹ️ Calibration not applied{note}. Original model kept.")

        total_runtime = time.time() - t_start

        st.session_state.update(dict(
            profile           = profile,
            X_raw             = X_raw,
            y                 = y,
            ids               = ids,
            X_train           = X_train,
            X_test            = X_test,
            y_train           = y_train,
            y_test            = y_test,
            new_feature_names = new_feat_names,
            results           = results,
            best_model_name   = best_name,
            selected_info     = [],
            total_runtime     = total_runtime,
            churn_rate        = churn_rate,
            is_imbalanced     = is_imbalanced,
            training_done     = True,
        ))

        upd(100, f"✅ Done in {total_runtime:.0f}s")
        time.sleep(0.8)
        prog_container.empty()
        st.rerun()

    except Exception as exc:
        prog_container.empty()
        st.error(f"Pipeline error: {exc}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())


def _show_results():
    """Renders Steps 10–11 from session_state."""
    from src.evaluator import build_comparison_table
    from src.explainer import get_builtin_importance

    results       = st.session_state.results
    best_name     = st.session_state.best_model_name
    best          = results[best_name]
    total_runtime = st.session_state.get("total_runtime", 0)
    is_imbalanced = st.session_state.get("is_imbalanced", False)

    st.divider()

    # ── Step 10: Primary Metrics ──────────────────────────────────────────────
    _sec("📈", "Step 4 - Evaluation Results")

    # Use CV metrics if available (more reliable), fallback to single-split
    use_cv   = "cv_pr_auc" in best
    roc      = best.get("cv_roc_auc",           best.get("roc_auc",            0))
    recall   = best.get("cv_recall",            best.get("recall",             0))
    pr_auc   = best.get("cv_pr_auc",            best.get("pr_auc",             0))
    score    = best.get("cv_best_overall_score", best.get("best_overall_score", 0))
    cv_label = " (5-fold CV avg)" if use_cv else ""

    mins = int(total_runtime // 60)
    secs = int(total_runtime % 60)
    st.caption(f"⏱️ Total pipeline runtime: **{mins}m {secs}s**")
    if use_cv:
        _thr_display = best.get("best_threshold", 0.50)
        st.info(
            f"📊 Metrics below are **averaged over 5-fold cross-validation** on the full dataset, "
            f"computed at the same decision threshold ({_thr_display:.2f}) used for predictions "
            "— the closest estimate of how the model will perform on new unseen customers.",
            icon=None,
        )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🏆 Best Model",        best_name,
              help="Selected by the highest Best Overall Score.")
    k2.metric("Best Overall Score",   f"{score:.4f}",
              help=f"0.45 × PR-AUC + 0.35 × Recall + 0.20 × ROC-AUC{cv_label}")
    k3.metric("PR-AUC",               f"{pr_auc:.4f}",
              help=f"Measures how well the model identifies customers who are likely to churn while keeping false alarms as low as possible. A higher PR-AUC means the model is better at finding real churn customers without incorrectly flagging too many loyal customers.{cv_label}")
    k4.metric("Recall",               f"{recall:.4f}",
              help=f"Measures how many customers who actually churned were correctly identified by the model. A higher Recall means fewer at-risk customers are missed, helping businesses take action before customers leave. Higher is better (range: 0-1).{cv_label}")
    k5.metric("ROC-AUC",              f"{roc:.4f}",
              help=f"Measures the model's overall ability to distinguish between customers who will churn and those who will stay. A higher ROC-AUC means the model is better at ranking high-risk customers ahead of low-risk customers across all decision thresholds.{cv_label}")

    # Quick overfitting alert (uses test ROC-AUC, not CV)
    train_roc = best.get("train_roc_auc", 0)
    test_roc  = best.get("roc_auc", 0)
    gap = train_roc - test_roc
    if gap > 0.10:
        st.warning(
            f"⚠️ Possible overfitting detected for **{best_name}**: "
            f"Train ROC-AUC = {train_roc:.4f} → Test ROC-AUC = {test_roc:.4f} "
            f"(gap = {gap:.4f}). See the Overfitting Report below."
        )

    st.markdown("#### Model Comparison  *(sorted by Best Overall Score)*")
    cmp_df = build_comparison_table(results)
    st.dataframe(
        cmp_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PR-AUC": st.column_config.TextColumn(
                "PR-AUC",
                help="Measures how well the model identifies customers who are likely to churn while keeping false alarms as low as possible. A higher PR-AUC means the model is better at finding real churn customers without incorrectly flagging too many loyal customers. Weight: 45% in Best Overall Score.",
            ),
            "Recall": st.column_config.TextColumn(
                "Recall",
                help="Measures how many customers who actually churned were correctly identified by the model. A higher Recall means fewer at-risk customers are missed, helping businesses take action before customers leave. Higher is better (range: 0-1). Weight: 35% in Best Overall Score.",
            ),
            "ROC-AUC": st.column_config.TextColumn(
                "ROC-AUC",
                help="Measures the model's overall ability to distinguish between customers who will churn and those who will stay. A higher ROC-AUC means the model is better at ranking high-risk customers ahead of low-risk customers across all decision thresholds. Weight: 20% in Best Overall Score.",
            ),
            "Best Overall Score": st.column_config.TextColumn(
                "Best Overall Score",
                help="0.45 × PR-AUC + 0.35 × Recall + 0.20 × ROC-AUC. The model with the highest score is selected.",
            ),
        },
    )

    # ── Overfitting Report ────────────────────────────────────────────────────
    _sec("🔬", "Overfitting Report")
    st.caption(
        "Train ROC-AUC is measured on the training set (data the model memorised). "
        "Val ROC-AUC is the average of 5 hold-out folds from Optuna tuning (unseen data). "
        "Test ROC-AUC is scored on the final held-out test set. "
        "Gap = Train − Test  ·  🟢 < 0.02  ·  🟡 0.02–0.05  ·  🔴 > 0.05"
    )
    from src.evaluator import build_overfitting_report
    of_df = build_overfitting_report(results)
    if not of_df.empty:
        st.dataframe(of_df, use_container_width=True, hide_index=True)
    st.divider()

    # ── Probability Distribution ───────────────────────────────────────────────
    _show_prob_distribution(best, st.session_state.get("y_test"))

    # ── Step 11: Professional Model Analysis ─────────────────────────────────
    _show_professional_analysis(results, best_name, cmp_df)


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


def _show_professional_analysis(results: dict, best_name: str, cmp_df):
    """Plain-language explanation of each model's actual metric results."""
    _sec("🧠", "What Do the Results Mean?")

    valid = {
        n: r for n, r in results.items()
        if r.get("model") is not None and r.get("roc_auc") is not None
    }
    if not valid:
        st.info("No valid models to analyse.")
        return

    ranked = sorted(valid, key=lambda n: valid[n].get("roc_auc", 0), reverse=True)

    st.markdown(
        "Below is a plain-language explanation of what each model's numbers actually mean - "
        "no technical background needed."
    )

    best   = valid[best_name]
    roc    = best.get("roc_auc", 0)
    rec    = best.get("recall",  0)
    pr_auc = best.get("pr_auc",  0)

    st.markdown(f"### 🏆 {best_name}")
    st.markdown(f"**PR-AUC:** {_explain_pr_auc(pr_auc)}")
    st.markdown(f"**Recall:** {_explain_recall(rec)}")
    st.markdown(f"**ROC-AUC:** {_explain_roc(roc)}")


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
            "**This is the most important metric in this app - it carries 45% of the Best Overall Score.**",
            "📈",
        ),
        (
            "Recall",
            "Out of all customers who actually churned - how many did the model catch?",
            "Measures how many customers who actually churned were correctly identified by the model. "
            "A higher Recall means fewer at-risk customers are missed, helping businesses take action before customers leave. "
            "If 100 customers were going to churn and the model caught 80, Recall = 80%. "
            "**Higher is better (range: 0-1). Carries 35% of the Best Overall Score.**",
            "🔍",
        ),
        (
            "ROC-AUC",
            "The model's overall ability to tell churners from loyal customers.",
            "Measures the model's overall ability to distinguish between customers who will churn and those who will stay. "
            "A higher ROC-AUC means the model is better at ranking high-risk customers ahead of low-risk customers across all decision thresholds. "
            "**1.0 = perfect. 0.5 = random guessing. Carries 20% of the Best Overall Score.**",
            "🎯",
        ),
        (
            "Best Overall Score",
            "The single score used to decide which model wins.",
            "A weighted combination of the three metrics above:\n\n"
            "> `Best Overall Score = 45% × PR-AUC + 35% × Recall + 20% × ROC-AUC`\n\n"
            "The model with the highest Best Overall Score is automatically selected as the winner.",
            "🏆",
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
        "Every model is trained and evaluated on a held-out test set (data the model never saw during training). "
        "The app then calculates a **Best Overall Score** for each model:\n\n"
        "> `Best Overall Score = 45% × PR-AUC + 35% × Recall + 20% × ROC-AUC`\n\n"
        "**Why these weights?**\n\n"
        "- **PR-AUC (45%)** - The most important metric. It rewards models that catch churners accurately "
        "without flooding you with false alarms. A high Recall achieved by flagging everyone as a churner "
        "will score poorly here.\n\n"
        "- **Recall (35%)** - Missing a churner is costly. This ensures models that catch more at-risk "
        "customers are rewarded.\n\n"
        "- **ROC-AUC (20%)** - Measures the overall quality of the model's rankings across all thresholds. "
        "Used as a stability check.\n\n"
        "The model with the **highest Best Overall Score** is automatically selected as the winner."
    )


def _show_prob_distribution(best: dict, y_test):
    """Histogram of predicted churn probabilities split by actual label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_prob = best.get("y_prob")
    if y_prob is None or y_test is None:
        return

    threshold = float(best.get("best_threshold", 0.50))
    y_arr     = np.asarray(y_test)

    _sec("📊", "Probability Distribution")
    st.caption(
        "Predicted churn probabilities for the held-out test set. "
        "Well-separated peaks indicate a model that is confident and accurate."
    )

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="none")
    ax.set_facecolor("none")

    ax.hist(y_prob[y_arr == 0], bins=40, alpha=0.55,
            color="#2196F3", label="Retained (actual)")
    ax.hist(y_prob[y_arr == 1], bins=40, alpha=0.60,
            color="#F44336", label="Churned (actual)")
    ax.axvline(threshold, color="#FFC107", linewidth=2.5, linestyle="--",
               label=f"Decision threshold ({threshold:.2f})")
    ax.set_xlabel("Predicted churn probability", fontsize=10)
    ax.set_ylabel("Customers", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Calibration info if available
    if best.get("calibrated") is not None:
        b_before = best.get("brier_before")
        b_after  = best.get("brier_after")
        if best["calibrated"] and b_before is not None and b_after is not None:
            st.caption(
                f"🎯 **Probability calibration applied** — "
                f"Brier Score: {b_before:.4f} → {b_after:.4f}"
            )
        elif b_before is not None and b_after is not None:
            st.caption(
                f"ℹ️ Calibration was tested but not applied "
                f"(Brier: {b_before:.4f} original vs {b_after:.4f} calibrated)"
            )

    # Separation quality hint
    if len(y_arr) > 0 and y_arr.sum() > 0:
        mean_retained = float(y_prob[y_arr == 0].mean()) if (y_arr == 0).any() else 0.5
        mean_churned  = float(y_prob[y_arr == 1].mean()) if (y_arr == 1).any() else 0.5
        sep = mean_churned - mean_retained
        if sep >= 0.25:
            st.success(
                f"✅ **Good separation** — churners average {mean_churned:.2f} vs "
                f"retained {mean_retained:.2f} (Δ {sep:.2f}). "
                "The model assigns meaningfully higher probabilities to actual churners."
            )
        elif sep >= 0.10:
            st.info(
                f"📊 **Moderate separation** — churners average {mean_churned:.2f} vs "
                f"retained {mean_retained:.2f} (Δ {sep:.2f}). "
                "Some overlap between the two groups."
            )
        else:
            st.warning(
                f"⚠️ **Low separation** — churners average {mean_churned:.2f} vs "
                f"retained {mean_retained:.2f} (Δ {sep:.2f}). "
                "The model struggles to assign clearly different probabilities to the two groups."
            )

    st.divider()


def _predict_section():
    """New customer prediction UI."""
    from src.data_loader import load_uploaded_file
    from src.pipeline_builder import raw_clean
    from src.predictor import predict_new_customers
    from src.utils import to_csv_bytes

    profile   = st.session_state.profile
    results   = st.session_state.results
    best_name = st.session_state.best_model_name
    best_pipe = results[best_name]["model"]
    best      = results[best_name]

    # ── File upload ───────────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        new_file = st.file_uploader(
            "Upload new customers file (CSV or Excel) - WITHOUT the churn column",
            type=["csv", "xlsx", "xls"],
            key="uploader_new",
        )
    with c2:
        st.info(
            "The trained pipeline handles all preprocessing.\n"
            "Upload raw customer data - same columns as training, minus the target."
        )

    if new_file is None:
        # HuggingFace resets the file uploader on every slider rerun.
        # If we already loaded the file into session_state, keep using it.
        if st.session_state.get("_pred_X_new") is None:
            st.session_state.pred_raw_probs = None
            return
        # else: fall through and use stored data below
    else:
        # ── Load & preprocess (cached by file identity) ───────────────────────
        file_key = f"{new_file.name}_{new_file.size}"
        if st.session_state.get("_pred_file_key") != file_key:
            try:
                df_new = load_uploaded_file(new_file)
                df_new_original = df_new.copy()
                df_new = raw_clean(df_new)

                from src.feature_engineering import engineer_features as _eng
                _new_feat_names = st.session_state.get("new_feature_names") or []
                if _new_feat_names:
                    _orig_numeric = [c for c in profile["numeric_cols"]
                                     if c not in set(_new_feat_names)]
                    df_new, _ = _eng(
                        df_new, _orig_numeric,
                        cat_cols=profile.get("categorical_cols"),
                        bool_cols=profile.get("boolean_cols"),
                    )

                from src.profiler import detect_id_column
                auto_id_new = detect_id_column(df_new)
                known_cols = (profile["numeric_cols"]
                              + profile["categorical_cols"]
                              + profile["boolean_cols"])
                for col in known_cols:
                    if col not in df_new.columns:
                        df_new[col] = np.nan
                X_new   = df_new[[c for c in known_cols if c in df_new.columns]].copy()
                ids_new = (df_new[auto_id_new] if auto_id_new and auto_id_new in df_new.columns
                           else pd.Series(range(len(df_new)), name="row_id"))

                st.session_state._pred_file_key  = file_key
                st.session_state._pred_X_new     = X_new
                st.session_state._pred_ids_new   = ids_new
                st.session_state._pred_df_orig   = df_new_original.reset_index(drop=True)
                st.session_state.pred_raw_probs  = None   # clear old results on new file
            except Exception as e:
                st.error(f"Could not load file: {e}")
                return

    X_new        = st.session_state._pred_X_new
    ids_new      = st.session_state._pred_ids_new
    df_new_orig  = st.session_state._pred_df_orig
    st.success(f"✅ Loaded **{len(X_new):,}** new customers")

    # ── Threshold selector ────────────────────────────────────────────────────
    auto_thr   = float(best.get("best_threshold", 0.50))
    thr_curve  = best.get("threshold_curve")

    st.markdown("#### Sensitivity Setting")
    sl_col, hint_col = st.columns([3, 2])
    with sl_col:
        selected_thr = st.slider(
            "Decision Threshold",
            min_value=0.20, max_value=0.80,
            value=auto_thr, step=0.05,
            help=(
                "Controls the trade-off between catching churners and avoiding false alarms.\n\n"
                "Lower → more customers flagged as churners (higher Recall, more false alarms).\n\n"
                "Higher → only high-confidence predictions flagged (fewer false alarms, may miss some churners)."
            ),
        )
        if thr_curve is not None:
            row = thr_curve.iloc[(thr_curve["threshold"] - selected_thr).abs().argsort()[:1]]
            est_rec  = float(row["recall"].values[0])
            est_prec = float(row["precision"].values[0])
            st.caption(
                f"At threshold **{selected_thr:.2f}** — "
                f"estimated Recall ≈ **{est_rec:.0%}** · "
                f"estimated Precision ≈ **{est_prec:.0%}**"
            )

    with hint_col:
        if selected_thr <= 0.40:
            st.info("🔍 **High Recall mode** — catches the most churners, but also flags some loyal customers as at-risk.")
        elif selected_thr <= 0.55:
            st.info("⚖️ **Balanced mode** — best F1 balance between catching churners and avoiding false alarms.")
        else:
            st.info("🎯 **High Precision mode** — only flags customers the model is very confident about. Fewer false alarms, but may miss some real churners.")

    # ── Predict button ────────────────────────────────────────────────────────
    if st.button("🔮 Predict Churn", type="primary", key="predict_btn"):
        with st.spinner("Running predictions…"):
            try:
                raw_probs = best_pipe.predict_proba(X_new)[:, 1]
                st.session_state.pred_raw_probs   = raw_probs
                st.session_state.pred_ids         = ids_new
                st.session_state.pred_df_orig     = df_new_orig
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                return

    # ── Results (apply current threshold to stored probabilities) ─────────────
    raw_probs = st.session_state.get("pred_raw_probs")
    if raw_probs is None:
        return

    ids_display  = st.session_state.pred_ids
    df_orig_disp = st.session_state.pred_df_orig

    churn_mask     = raw_probs >= selected_thr
    churn_pred_col = np.where(churn_mask, "Yes", "No")
    total          = len(raw_probs)
    churning       = int(churn_mask.sum())
    not_churning   = total - churning
    pct_churn      = churning / max(total, 1)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers",  f"{total:,}")
    m2.metric("Will Churn",       f"{churning:,}")
    m3.metric("Will Not Churn",   f"{not_churning:,}")
    m4.metric("% Churn",          f"{pct_churn:.1%}")

    # ── Churn Rate Monitoring ─────────────────────────────────────────────────
    training_churn_rate = st.session_state.get("churn_rate")
    if training_churn_rate is not None:
        diff_pp  = (pct_churn - training_churn_rate) * 100
        abs_diff = abs(diff_pp)

        st.markdown("#### Churn Rate Monitor")
        cr1, cr2, cr3 = st.columns(3)
        cr1.metric(
            "Historical Churn Rate", f"{training_churn_rate:.1%}",
            help="Churn rate observed in the training dataset.",
        )
        cr2.metric(
            "Predicted Churn Rate", f"{pct_churn:.1%}",
            help="Churn rate predicted for this uploaded batch.",
        )
        cr3.metric(
            "Difference",
            f"{diff_pp:+.1f} pp",
            delta=f"{diff_pp:+.1f} pp",
            delta_color="inverse",
            help="Difference in percentage points between predicted and historical churn rates.",
        )

        if abs_diff > 10:
            st.warning(
                f"⚠️ **Distribution shift detected**: the predicted churn rate ({pct_churn:.1%}) "
                f"differs from the historical training rate ({training_churn_rate:.1%}) by "
                f"**{abs_diff:.1f} percentage points**. This customer batch may differ "
                "significantly from the training population, or the decision threshold may need adjustment."
            )
        else:
            st.success(
                f"✅ Predicted churn rate ({pct_churn:.1%}) is consistent with the "
                f"historical training rate ({training_churn_rate:.1%}) — "
                f"difference: {abs_diff:.1f} pp."
            )

    st.divider()

    if df_orig_disp is not None:
        display_df = df_orig_disp.copy()
        display_df["Churn Prediction"]  = churn_pred_col
        display_df["Churn Probability"] = np.round(raw_probs, 4)
    else:
        display_df = pd.DataFrame({
            "customer_id":       ids_display.values,
            "churn_prediction":  churn_pred_col,
            "churn_probability": np.round(raw_probs, 4),
        })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Predictions CSV",
        data=to_csv_bytes(display_df),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
for _k in [
    "df", "profile", "X_raw", "y", "ids",
    "X_train", "X_test", "y_train", "y_test",
    "new_feature_names", "results", "best_model_name",
    "lib_status", "predictions", "training_done",
    "selected_info", "total_runtime", "churn_rate", "is_imbalanced",
    "df_new_original",
]:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("📊 Churn Analytics Studio")
st.caption("AI & Machine Learning Powered Customer Churn Prediction Platform")
st.divider()

_tab_pipeline, _tab_guide = st.tabs(["🚀 Pipeline", "📖 How It Works"])

with _tab_guide:
    _show_guide()

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
            "- A binary churn column (Yes/No, 1/0, True/False)\n"
            "- CSV or Excel format\n"
            "- Auto-detects delimiter & encoding"
        )

    if hist_file is not None:
        _prog = st.progress(0, text="Loading…")
        try:
            from src.data_loader import load_uploaded_file
            _prog.progress(5, text="Detecting delimiter & encoding…")
            _df = load_uploaded_file(hist_file)
            _prog.progress(10, text="Done ✅")
            if st.session_state.df is None or len(st.session_state.df) != len(_df):
                for _k in ["profile","X_raw","y","ids","X_train","X_test","y_train","y_test",
                           "new_feature_names","results","best_model_name","predictions","training_done"]:
                    st.session_state[_k] = None
            st.session_state.df = _df
            time.sleep(0.2)
            _prog.empty()
        except Exception as _e:
            _prog.empty()
            st.error(f"Could not load file: {_e}")
            st.session_state.df = None

    df = st.session_state.df

    if df is None:
        st.info("👆 Upload a dataset to begin.")
        st.stop()

    st.success(f"✅ Loaded **{len(df):,} rows × {df.shape[1]} columns**")
    with st.expander("Preview first rows", expanded=False):
        st.dataframe(df.head(5), use_container_width=True)

    st.divider()

    # ── Step 2: Column Detection ──────────────────────────────────────────────
    _sec("🔍", "Step 2 - Column Detection")

    from src.profiler import detect_id_column, detect_target_column

    auto_id     = detect_id_column(df)
    auto_target = detect_target_column(df, exclude=[auto_id] if auto_id else [])
    cols_list   = df.columns.tolist()

    ca, cb, cc = st.columns(3)
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
        if auto_target not in cols_list:
            auto_target = cols_list[0] if cols_list else None
        target_col = st.selectbox(
            "Churn / Target column ★",
            options=cols_list,
            index=cols_list.index(auto_target) if auto_target in cols_list else 0,
        )
        if auto_target:
            st.caption(f"Auto-detected: **{auto_target}**")

    with cc:
        st.metric("Rows",    f"{len(df):,}")
        st.metric("Columns", df.shape[1])
        try:
            _target_series = df[target_col].astype(str).str.strip().str.lower()
            _churn_mask    = _target_series.isin(["1", "yes", "true", "churn", "1.0"])
            _n_churn       = int(_churn_mask.sum())
            _pct_churn     = _n_churn / len(df) if len(df) > 0 else 0
            st.metric(
                "Churned Customers",
                f"{_n_churn:,}  ({_pct_churn:.1%})",
                help="Number and percentage of customers who churned in the historical data.",
            )
        except Exception:
            pass

    st.divider()

    # ── Predictive Signal Assessment ──────────────────────────────────────────
    _show_signal_assessment(df, target_col, id_col)

    # ── Steps 3–11: Run Pipeline ──────────────────────────────────────────────
    from src.config import PRODUCTION_MODELS
    _sec("🚀", "Step 3 - Run Full Machine Learning Pipeline")

    model_col, btn_col = st.columns([3, 1])
    with model_col:
        all_option   = "All Models"
        model_choices = [all_option] + PRODUCTION_MODELS
        selected_raw = st.multiselect(
            "Models to train",
            options=model_choices,
            default=[all_option],
            help="Select one or more models. Choose 'All Models' to train all 5 production models.",
        )
        if all_option in selected_raw or not selected_raw:
            selected_models = PRODUCTION_MODELS.copy()
        else:
            selected_models = [m for m in PRODUCTION_MODELS if m in selected_raw]

        from src.config import PIPELINE_CONFIG
        st.caption(
            f"Training: **{', '.join(selected_models)}**  ·  "
            f"Optuna trials: **{PIPELINE_CONFIG['optuna_trials']}**  ·  "
            f"CV folds: **{PIPELINE_CONFIG['cv_folds']}**  ·  Tuning: **ROC-AUC**"
        )

        calibrate_enabled = st.checkbox(
            "Enable probability calibration",
            value=st.session_state.get("calibrate_enabled", False),
            help=(
                "After training, attempts to improve the raw probability outputs using "
                "CalibratedClassifierCV. Only applies if it measurably improves the Brier Score "
                "on the test set. Adds a few seconds to training time."
            ),
        )
        st.session_state.calibrate_enabled = calibrate_enabled

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
    with btn_col:
        run = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
        _eta_low, _eta_high = _estimate_runtime(df, selected_models)
        st.caption(f"⏱️ Est. runtime: **{_eta_low} - {_eta_high}**")

    if run:
        for _k in ["results", "best_model_name", "training_done", "predictions",
                   "selected_info", "total_runtime", "churn_rate", "is_imbalanced"]:
            st.session_state[_k] = None
        _run_pipeline(df, id_col, target_col, selected_models)

    if st.session_state.training_done:
        _show_results()

    st.divider()

    # ── Predict New Customers ─────────────────────────────────────────────────
    if st.session_state.training_done:
        _sec("🔮", "Predict New Customers")
        _predict_section()

    st.divider()
    st.caption(
        "Churn Analytics Studio  ·  Built with Streamlit & scikit-learn  ·  "
        "Zero data leakage  ·  Best model selected by PR-AUC · Recall · ROC-AUC  ·  "
        "Models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost"
    )
