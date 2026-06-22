"""
All Plotly visualisation helpers.
All functions return a plotly Figure — never render directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#16A34A",
    "danger":    "#DC2626",
    "warning":   "#D97706",
    "muted":     "#6B7280",
    "bg":        "#FFFFFF",
}


# ── Churn Distribution ────────────────────────────────────────────────────────

def plot_churn_distribution(y: pd.Series) -> go.Figure:
    counts = y.value_counts()
    labels = ["Not Churn" if k == 0 else "Churn" for k in counts.index]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.42,
        marker_colors=[_PALETTE["secondary"], _PALETTE["danger"]],
        textinfo="label+percent",
        hovertemplate="%{label}<br>Count: %{value:,}<extra></extra>",
    ))
    fig.update_layout(
        title_text="Churn Distribution — Training Data",
        height=340,
        paper_bgcolor=_PALETTE["bg"],
        showlegend=True,
    )
    return fig


# ── Model Comparison ──────────────────────────────────────────────────────────

def plot_model_comparison(results: dict) -> go.Figure:
    valid = {n: r for n, r in results.items() if not r.get("error") and "roc_auc" in r}
    names = sorted(valid, key=lambda n: valid[n]["roc_auc"], reverse=True)

    metrics = ["roc_auc", "f1", "accuracy", "balanced_accuracy"]
    colours = [_PALETTE["primary"], _PALETTE["secondary"], _PALETTE["warning"], _PALETTE["danger"]]
    labels  = ["ROC-AUC", "F1", "Accuracy", "Balanced Accuracy"]

    fig = go.Figure()
    for metric, colour, label in zip(metrics, colours, labels):
        fig.add_trace(go.Bar(
            name=label,
            x=names,
            y=[valid[n].get(metric, 0) for n in names],
            marker_color=colour,
            text=[f"{valid[n].get(metric, 0):.3f}" for n in names],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title="Model Comparison — Key Metrics (Test Set)",
        yaxis=dict(title="Score", range=[0, 1.12]),
        height=450,
        paper_bgcolor=_PALETTE["bg"],
        plot_bgcolor=_PALETTE["bg"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── ROC Curve ─────────────────────────────────────────────────────────────────

def plot_roc_curves(results: dict, y_test) -> go.Figure:
    from sklearn.metrics import roc_curve, auc

    fig = go.Figure()
    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color=_PALETTE["muted"]),
        showlegend=False,
    ))

    colours = px.colors.qualitative.Plotly
    for idx, (name, r) in enumerate(results.items()):
        if r.get("error") or "y_prob" not in r:
            continue
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        auc_score   = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name}  (AUC={auc_score:.3f})",
            line=dict(color=colours[idx % len(colours)], width=2),
        ))

    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        paper_bgcolor=_PALETTE["bg"],
        plot_bgcolor=_PALETTE["bg"],
    )
    return fig


# ── Precision-Recall Curve ────────────────────────────────────────────────────

def plot_pr_curves(results: dict, y_test) -> go.Figure:
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig = go.Figure()
    colours = px.colors.qualitative.Plotly

    for idx, (name, r) in enumerate(results.items()):
        if r.get("error") or "y_prob" not in r:
            continue
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        ap = average_precision_score(y_test, r["y_prob"])
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{name}  (AP={ap:.3f})",
            line=dict(color=colours[idx % len(colours)], width=2),
        ))

    fig.update_layout(
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=450,
        paper_bgcolor=_PALETTE["bg"],
        plot_bgcolor=_PALETTE["bg"],
    )
    return fig


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> go.Figure:
    labels = ["Not Churn", "Churn"]
    pct = cm.astype(float) / cm.sum() * 100

    text = [[f"{cm[i][j]}<br>({pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels, y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=380,
        paper_bgcolor=_PALETTE["bg"],
        font=dict(size=14),
    )
    return fig


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(
    importances: pd.Series,
    title: str = "Feature Importance",
    top_n: int = 20,
) -> go.Figure:
    top = importances.head(top_n).sort_values()
    colours = [_PALETTE["danger"] if v < 0 else _PALETTE["primary"] for v in top.values]

    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index,
        orientation="h",
        marker_color=colours,
        text=[f"{v:+.4f}" for v in top.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(350, top_n * 28),
        paper_bgcolor=_PALETTE["bg"],
        plot_bgcolor=_PALETTE["bg"],
        font=dict(size=12),
        margin=dict(l=200),
    )
    return fig


# ── SHAP Summary ──────────────────────────────────────────────────────────────

def plot_shap_bar(shap_values: np.ndarray, feature_names: list[str], top_n: int = 20) -> go.Figure:
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx      = np.argsort(mean_abs)[-top_n:]
    names    = [feature_names[i] for i in idx]
    values   = mean_abs[idx]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=_PALETTE["primary"],
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"SHAP Feature Importance (mean |SHAP value|) — Top {top_n}",
        xaxis_title="Mean |SHAP value|",
        height=max(350, top_n * 28),
        paper_bgcolor=_PALETTE["bg"],
        plot_bgcolor=_PALETTE["bg"],
        margin=dict(l=200),
    )
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """SHAP beeswarm-style scatter rendered with Plotly."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-top_n:][::-1]
    names    = X_sample.columns[top_idx].tolist()

    fig = go.Figure()
    colours = px.colors.diverging.RdBu

    for rank, feat_idx in enumerate(top_idx):
        feat_name  = X_sample.columns[feat_idx]
        sv         = shap_values[:, feat_idx]
        fv         = X_sample.iloc[:, feat_idx]

        # Normalise feature values for colouring
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        marker_colours = [
            f"rgb({int(255*(1-v))},{int(100)},{int(255*v)})" for v in fv_norm
        ]

        fig.add_trace(go.Scatter(
            x=sv,
            y=[rank + np.random.uniform(-0.2, 0.2) for _ in sv],
            mode="markers",
            name=feat_name,
            marker=dict(
                size=4,
                color=fv_norm,
                colorscale="RdBu",
                opacity=0.7,
            ),
            hovertemplate=f"<b>{feat_name}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title="SHAP Beeswarm — Top Features (positive = increases churn probability)",
        xaxis_title="SHAP value",
        yaxis=dict(
            tickvals=list(range(top_n)),
            ticktext=names,
            autorange="reversed",
        ),
        height=max(400, top_n * 30),
        showlegend=False,
        paper_bgcolor=_PALETTE["bg"],
        plot_bgcolor=_PALETTE["bg"],
        margin=dict(l=200),
    )
    return fig


# ── Threshold Curve ──────────────────────────────────────────────────────────

def plot_threshold_curve(curve_df, best_threshold: float, model_name: str) -> go.Figure:
    """
    Line chart: F1, Recall, Precision vs decision threshold.
    Highlights the chosen optimal threshold.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["f1"],
                             mode="lines", name="F1", line=dict(color=_PALETTE["primary"], width=2)))
    fig.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["recall"],
                             mode="lines", name="Recall", line=dict(color=_PALETTE["secondary"], width=2)))
    fig.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["precision"],
                             mode="lines", name="Precision", line=dict(color=_PALETTE["warning"], width=2)))
    # Mark optimal threshold
    fig.add_vline(x=best_threshold, line_dash="dash", line_color=_PALETTE["danger"],
                  annotation_text=f"Best thr = {best_threshold:.2f}",
                  annotation_position="top right")
    fig.update_layout(
        title=f"Threshold Optimisation — {model_name}",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        height=380,
        paper_bgcolor=_PALETTE["bg"],
        plot_bgcolor=_PALETTE["bg"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Prediction Distribution ───────────────────────────────────────────────────

def plot_prediction_distribution(preds_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, colour in [("Yes", _PALETTE["danger"]), ("No", _PALETTE["secondary"])]:
        subset = preds_df[preds_df["churn_prediction"] == label]["churn_probability"]
        fig.add_trace(go.Histogram(
            x=subset,
            name=f"Churn = {label}",
            marker_color=colour,
            opacity=0.7,
            nbinsx=30,
        ))
    fig.update_layout(
        barmode="overlay",
        title="Churn Probability Distribution (New Customers)",
        xaxis_title="Churn Probability",
        yaxis_title="Count",
        height=380,
        paper_bgcolor=_PALETTE["bg"],
        plot_bgcolor=_PALETTE["bg"],
    )
    return fig


# ── Risk Breakdown ────────────────────────────────────────────────────────────

def plot_risk_breakdown(preds_df: pd.DataFrame) -> go.Figure:
    counts = preds_df["risk_level"].value_counts()
    colour_map = {
        "🔴 High":   _PALETTE["danger"],
        "🟡 Medium": _PALETTE["warning"],
        "🟢 Low":    _PALETTE["secondary"],
    }
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=[colour_map.get(k, _PALETTE["muted"]) for k in counts.index],
        text=counts.values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Customer Risk Breakdown",
        xaxis_title="Risk Level",
        yaxis_title="Count",
        height=360,
        paper_bgcolor=_PALETTE["bg"],
        plot_bgcolor=_PALETTE["bg"],
    )
    return fig
