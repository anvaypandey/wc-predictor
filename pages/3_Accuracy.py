"""
Model Accuracy — how well the predictor performs on historical data.
Shows overall CV accuracy, WC back-test by year/stage, calibration, and feature importance.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

from src.flags import with_flag
from src.prepare import FEATURE_COLS

ARTIFACTS = Path("artifacts")

# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not (ARTIFACTS / "model.pkl").exists():
        return None
    return joblib.load(ARTIFACTS / "model.pkl")


@st.cache_data
def load_backtest() -> pd.DataFrame | None:
    p = ARTIFACTS / "backtest.pkl"
    if not p.exists():
        return None
    df = joblib.load(p)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


@st.cache_data
def load_metrics() -> dict | None:
    p = ARTIFACTS / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _confusion_matrix_fig(df: pd.DataFrame, classes: list[str]) -> go.Figure:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df["actual"], df["predicted"], labels=classes)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    text = [[f"{cm[i,j]}<br>{cm_pct[i,j]:.0f}%" for j in range(len(classes))]
            for i in range(len(classes))]
    fig = go.Figure(go.Heatmap(
        z=cm_pct, x=classes, y=classes,
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
        zmin=0, zmax=100,
    ))
    fig.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _calibration_fig(df: pd.DataFrame, classes: list[str]) -> go.Figure:
    fig = go.Figure()
    colors = {"Win": "#27ae60", "Draw": "#e67e22", "Loss": "#c0392b"}
    bins = np.linspace(0, 1, 11)

    for cls in classes:
        prob_col = f"prob_{cls}"
        if prob_col not in df.columns:
            continue
        actual_binary = (df["actual"] == cls).astype(int)
        bin_centers, actual_rates = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (df[prob_col] >= lo) & (df[prob_col] < hi)
            if mask.sum() >= 5:
                bin_centers.append((lo + hi) / 2)
                actual_rates.append(actual_binary[mask].mean())
        if bin_centers:
            fig.add_trace(go.Scatter(
                x=bin_centers, y=actual_rates, mode="lines+markers",
                name=cls, line=dict(color=colors.get(cls, "#888"), width=2),
                marker=dict(size=6),
            ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Perfect", line=dict(dash="dash", color="gray", width=1),
        showlegend=True,
    ))
    fig.update_layout(
        xaxis=dict(title="Predicted probability", range=[0, 1]),
        yaxis=dict(title="Actual frequency",      range=[0, 1]),
        height=320, legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _wc_by_year_fig(wc_df: pd.DataFrame) -> go.Figure:
    by_year = (wc_df.groupby("year")["correct"]
               .agg(accuracy=lambda x: x.mean() * 100, matches="count")
               .reset_index())
    fig = go.Figure(go.Bar(
        x=by_year["year"].astype(str),
        y=by_year["accuracy"],
        text=[f"{a:.0f}%<br>({n} matches)" for a, n in zip(by_year["accuracy"], by_year["matches"])],
        textposition="outside",
        marker_color="#f39c12",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis_title="World Cup year",
        yaxis=dict(title="Accuracy (%)", range=[0, 100]),
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _feature_importance_fig(model) -> go.Figure:
    pairs = sorted(zip(FEATURE_COLS, model.feature_importances_),
                   key=lambda x: x[1])
    names, imps = zip(*pairs)
    fig = go.Figure(go.Bar(
        x=list(imps), y=list(names), orientation="h",
        marker_color="#3498db",
        text=[f"{v:.3f}" for v in imps],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        xaxis_title="Importance", height=520,
        margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("Model Accuracy")
st.caption("How well the predictor performs on held-out data")

model = load_model()
if model is None:
    st.warning("No trained model found. Run `python train.py` first.")
    st.stop()

metrics = load_metrics()
backtest = load_backtest()

needs_retrain = backtest is None or metrics is None

# ── Summary cards ─────────────────────────────────────────────────────────────

st.markdown("## Overall Performance")

if metrics:
    cv_acc  = metrics["cv_accuracy_mean"] * 100
    cv_std  = metrics["cv_accuracy_std"] * 100
    n_rows  = metrics.get("n_training_rows", "—")
    classes = metrics.get("classes", ["Draw", "Loss", "Win"])
else:
    cv_acc, cv_std, n_rows = None, None, "—"
    classes = ["Draw", "Loss", "Win"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    if cv_acc:
        st.metric("5-fold CV Accuracy", f"{cv_acc:.1f}%", f"±{cv_std:.1f}%")
    else:
        st.metric("5-fold CV Accuracy", "—")
with c2:
    if backtest is not None:
        wc_df = backtest[backtest["tournament"] == "FIFA World Cup"]
        wc_acc = wc_df["correct"].mean() * 100 if len(wc_df) else 0
        st.metric("World Cup Accuracy", f"{wc_acc:.1f}%", f"{len(wc_df)} matches")
    else:
        st.metric("World Cup Accuracy", "—")
with c3:
    st.metric("Training rows", f"{n_rows:,}" if isinstance(n_rows, int) else n_rows)
with c4:
    st.metric("Outcome classes", "Win · Draw · Loss")

if needs_retrain:
    st.info(
        "Detailed accuracy data not found. Run `python train.py` to generate "
        "back-test data (adds ~2 min for 5-fold CV). The page will auto-populate after retraining.",
        icon="ℹ️",
    )

# ── Feature importance ────────────────────────────────────────────────────────

st.markdown("## Feature Importance")
st.caption("Which inputs the model relies on most when making predictions")
st.plotly_chart(_feature_importance_fig(model), use_container_width=True)

if backtest is None:
    st.stop()

# ── World Cup back-test ───────────────────────────────────────────────────────

st.markdown("## World Cup Back-Test")
st.caption(
    "Out-of-fold predictions on FIFA World Cup matches — each match was predicted "
    "by a model that had never seen it during training (5-fold CV holdout)."
)

wc_df = backtest[backtest["tournament"] == "FIFA World Cup"].copy()

if wc_df.empty:
    st.info("No World Cup matches found in the training dataset.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Accuracy by World Cup")
        st.plotly_chart(_wc_by_year_fig(wc_df), use_container_width=True)
    with col_b:
        st.markdown("### Confusion Matrix (WC matches)")
        st.caption("Rows = actual outcome, Columns = predicted outcome")
        st.plotly_chart(_confusion_matrix_fig(wc_df, classes), use_container_width=True)

    # Per-match table (most recent WC first)
    st.markdown("### Match-by-Match Results")
    recent_wc_year = wc_df["year"].max()
    show_year = st.selectbox(
        "Filter by World Cup",
        options=sorted(wc_df["year"].unique(), reverse=True),
        index=0,
        format_func=lambda y: f"{y} FIFA World Cup",
    )
    sub = wc_df[wc_df["year"] == show_year].copy()
    sub["home_flag"] = sub["home_team"].apply(with_flag)
    sub["away_flag"] = sub["away_team"].apply(with_flag)
    sub["Score"] = sub["home_score"].astype(str) + " – " + sub["away_score"].astype(str)
    sub["Result"] = sub.apply(
        lambda r: "✅" if r["correct"] else "❌", axis=1
    )
    display_cols = {
        "date": "Date",
        "home_flag": "Home Team",
        "Score": "Score",
        "away_flag": "Away Team",
        "actual": "Actual",
        "predicted": "Predicted",
        "Result": "Result",
    }
    st.dataframe(
        sub[list(display_cols)].rename(columns=display_cols).set_index("Date"),
        use_container_width=True,
        hide_index=False,
    )

# ── Calibration ───────────────────────────────────────────────────────────────

st.markdown("## Calibration")
st.caption(
    "When the model says 60% chance of a Win, does it actually win 60% of the time? "
    "A well-calibrated model hugs the diagonal."
)
st.plotly_chart(_calibration_fig(backtest, classes), use_container_width=True)

# ── All-tournament accuracy breakdown ─────────────────────────────────────────

st.markdown("## Accuracy by Tournament Type")
tier_labels = {1: "Friendly", 2: "Regional competitive", 3: "Qualification",
               4: "Continental", 5: "World Cup"}

if "tournament_tier" not in backtest.columns:
    st.info("Tournament tier data not available in back-test output.")
else:
    tier_acc = (backtest.groupby("tournament")["correct"]
                .agg(accuracy=lambda x: x.mean() * 100, matches="count")
                .reset_index()
                .sort_values("accuracy", ascending=False)
                .head(20))
    fig = go.Figure(go.Bar(
        x=tier_acc["accuracy"],
        y=tier_acc["tournament"],
        orientation="h",
        text=[f"{a:.0f}% ({n})" for a, n in zip(tier_acc["accuracy"], tier_acc["matches"])],
        textposition="outside",
        marker_color="#9b59b6",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(title="Accuracy (%)", range=[0, 100]),
        height=max(300, len(tier_acc) * 22),
        margin=dict(l=10, r=120, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
