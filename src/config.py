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
from .runtime_env import ON_CLOUD as _ON_CLOUD

# Free-tier cloud CPUs (2 vCPU) run single-threaded to avoid the OpenMP
# segfault, so the full 30-trial x 5-fold search would take 7-20 min and
# starve Streamlit's websocket until the browser session drops. A smaller
# search budget keeps the hosted demo responsive; local runs are untouched.
PIPELINE_CONFIG = {
    "cv_folds":      5,                       # folds for final CV evaluation
    "n_iter":        30,
    "optuna_trials": 8 if _ON_CLOUD else 30,  # HPO trials per model
    "hpo_cv_folds":  3 if _ON_CLOUD else 5,   # folds inside the HPO search
    "hpo_max_rows":  4000 if _ON_CLOUD else 0,  # 0 = no subsampling
    "max_iters":     200 if _ON_CLOUD else 500,  # boosting rounds ceiling
    "rf_trees":      150 if _ON_CLOUD else 300,
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
