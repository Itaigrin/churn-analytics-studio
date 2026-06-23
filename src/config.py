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
import os as _os
_ON_CLOUD = _os.path.exists("/mount/src")

PIPELINE_CONFIG = {
    "cv_folds":      3,
    "n_iter":        15,
    "optuna_trials": 0,
}

# ── Best Overall Score weights ────────────────────────────────────────────────
SCORE_WEIGHT_PR_AUC = 0.45
SCORE_WEIGHT_RECALL = 0.35
SCORE_WEIGHT_ROC    = 0.20

# ── Column-name hints ─────────────────────────────────────────────────────────
ID_HINTS = [
    "customerid", "customer_id", "userid", "user_id", "client_id", "clientid",
    "account_id", "accountid", "cust_id", "member_id", "employeeid",
    "employee_id", "empid", "custno", "custnum",
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
