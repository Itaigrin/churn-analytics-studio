"""
Central configuration for ChurnApp Pro.
All constants, thresholds, and pipeline definitions live here.
"""

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ── Column-drop thresholds ────────────────────────────────────────────────────
NULL_THRESHOLD           = 0.50
NEAR_CONSTANT_THRESHOLD  = 0.98
HIGH_CARDINALITY_LIMIT   = 50

# ── Dataset size buckets ──────────────────────────────────────────────────────
LARGE_DATASET_ROWS      = 50_000
VERY_LARGE_DATASET_ROWS = 200_000
SHAP_SAMPLE_SIZE        = 300
PERMUTATION_SAMPLE_SIZE = 2_000

# ── Class-imbalance thresholds ────────────────────────────────────────────────
IMBALANCE_LOW  = 0.20
IMBALANCE_HIGH = 0.80

# ── Tuning metric ─────────────────────────────────────────────────────────────
TUNING_METRIC         = "roc_auc"
TUNING_METRIC_DISPLAY = "ROC-AUC"

# ── Production model list ─────────────────────────────────────────────────────
PRODUCTION_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

# ── Fixed pipeline config (replaces execution modes) ─────────────────────────

# How hard to search for hyperparameters. The user picks this in Step 3.
#
# It exists because wall-clock time is a hard constraint on small shared CPUs:
# a search that runs for many minutes outlives the browser's connection to the
# app, and the session is lost before results are written. Rather than guess on
# the user's behalf, expose the trade-off and let them choose.
SEARCH_PRESETS = {
    "Quick": {
        "optuna_trials": 8,
        "hpo_cv_folds":  3,
        "hpo_max_rows":  4000,   # 0 = search on the full training set
        "max_iters":     200,    # boosting rounds ceiling
        "rf_trees":      150,
    },
    "Balanced": {
        "optuna_trials": 15,
        "hpo_cv_folds":  5,
        "hpo_max_rows":  6000,
        "max_iters":     300,
        "rf_trees":      200,
    },
    "Thorough": {
        "optuna_trials": 30,
        "hpo_cv_folds":  5,
        "hpo_max_rows":  0,
        "max_iters":     500,
        "rf_trees":      300,
    },
}

# Thorough is the original behaviour and stays the default everywhere.
# Quick / Balanced are there for when a faster answer is worth a slightly
# less tuned model — that is the user's call to make, not a silent default.
DEFAULT_SEARCH_DEPTH = "Thorough"

PIPELINE_CONFIG = {
    "cv_folds":      5,    # folds for the final CV evaluation
    "n_iter":        30,
    **SEARCH_PRESETS["Thorough"],
}

# ── Best Overall Score weights ────────────────────────────────────────────────
SCORE_WEIGHT_PR_AUC = 0.45
SCORE_WEIGHT_RECALL = 0.35
SCORE_WEIGHT_ROC    = 0.20

# ── Column-name hints ─────────────────────────────────────────────────────────
ID_HINTS = [
    "customerid", "customer_id", "userid", "user_id", "client_id", "clientid",
    "account_id", "accountid", "cust_id", "member_id", "employeeid",
    "employee_id", "empid", "custno", "custnum", "clientnum", "client_num",
    "rownum", "row_num", "rowno", "row_no", "rownumber", "row_number",
    "id", "idx", "index",
]
CHURN_HINTS = [
    "churn", "is_churned", "churned", "left_company", "attrition",
    "is_churn", "target", "label", "churned_flag", "left", "exited",
    "cancelled", "canceled", "subscription_status",
]
LEAK_HINTS = [
    "date_churned", "churn_date", "exit_date", "cancellation_date",
    "last_activity_after_churn", "days_since_churn", "churn_reason",
]
