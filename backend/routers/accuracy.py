import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException
from sklearn.metrics import confusion_matrix

from backend.schemas import AccuracyResponse, WCMatch
from backend.state import get_state
from backend.routers._charts import _DARK_LAYOUT
from src.prepare import FEATURE_COLS

router = APIRouter()
log = logging.getLogger(__name__)
ARTIFACTS = Path("artifacts")

_cache: AccuracyResponse | None = None
_cache_ts: float = 0.0
_CACHE_TTL = 3600.0


def _feature_chart(model) -> str:
    pairs = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: x[1])
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
        **_DARK_LAYOUT,
    )
    return fig.to_json()


def _confusion_chart(df: pd.DataFrame, classes: list[str]) -> str:
    cm = confusion_matrix(df["actual"], df["predicted"], labels=classes)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    text = [[f"{cm[i,j]}<br>{cm_pct[i,j]:.0f}%" for j in range(len(classes))]
            for i in range(len(classes))]
    fig = go.Figure(go.Heatmap(
        z=cm_pct, x=classes, y=classes,
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=False, zmin=0, zmax=100,
    ))
    fig.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        **_DARK_LAYOUT,
    )
    return fig.to_json()


def _wc_year_chart(wc_df: pd.DataFrame) -> str:
    by_year = (wc_df.groupby("year")["correct"]
               .agg(accuracy=lambda x: x.mean() * 100, matches="count")
               .reset_index())
    fig = go.Figure(go.Bar(
        x=by_year["year"].astype(str), y=by_year["accuracy"],
        text=[f"{a:.0f}%<br>({n})" for a, n in zip(by_year["accuracy"], by_year["matches"])],
        textposition="outside", marker_color="#f39c12", cliponaxis=False,
    ))
    fig.update_layout(
        xaxis_title="World Cup year",
        yaxis=dict(title="Accuracy (%)", range=[0, 100]),
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        **_DARK_LAYOUT,
    )
    return fig.to_json()


def _calibration_chart(df: pd.DataFrame, classes: list[str]) -> str:
    fig = go.Figure()
    colors = {"Win": "#27ae60", "Draw": "#e67e22", "Loss": "#c0392b"}
    bins = np.linspace(0, 1, 11)
    for cls in classes:
        prob_col = f"prob_{cls}"
        if prob_col not in df.columns:
            continue
        actual_binary = (df["actual"] == cls).astype(int)
        centers, rates = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (df[prob_col] >= lo) & (df[prob_col] < hi)
            if mask.sum() >= 5:
                centers.append((lo + hi) / 2)
                rates.append(actual_binary[mask].mean())
        if centers:
            fig.add_trace(go.Scatter(
                x=centers, y=rates, mode="lines+markers", name=cls,
                line=dict(color=colors.get(cls, "#888"), width=2),
                marker=dict(size=6),
            ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
        line=dict(dash="dash", color="gray", width=1),
    ))
    fig.update_layout(
        xaxis=dict(title="Predicted probability", range=[0, 1]),
        yaxis=dict(title="Actual frequency", range=[0, 1]),
        height=320, legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=10, b=10),
        **_DARK_LAYOUT,
    )
    return fig.to_json()


def _tournament_chart(df: pd.DataFrame) -> str:
    acc = (df.groupby("tournament")["correct"]
           .agg(accuracy=lambda x: x.mean() * 100, matches="count")
           .reset_index()
           .sort_values("accuracy", ascending=False)
           .head(20))
    fig = go.Figure(go.Bar(
        x=acc["accuracy"], y=acc["tournament"], orientation="h",
        text=[f"{a:.0f}% ({n})" for a, n in zip(acc["accuracy"], acc["matches"])],
        textposition="outside", marker_color="#9b59b6", cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(title="Accuracy (%)", range=[0, 100]),
        height=max(300, len(acc) * 22),
        margin=dict(l=10, r=120, t=10, b=10),
        **_DARK_LAYOUT,
    )
    return fig.to_json()


@router.get("/accuracy", response_model=AccuracyResponse)
def accuracy():
    global _cache, _cache_ts
    t0 = time.perf_counter()
    log.info("GET /api/accuracy")

    if _cache is not None and (time.perf_counter() - _cache_ts) < _CACHE_TTL:
        log.info("GET /api/accuracy served from cache in %.3fs", time.perf_counter() - t0)
        return _cache
    metrics_path = ARTIFACTS / "metrics.json"
    backtest_path = ARTIFACTS / "backtest.pkl"
    if not metrics_path.exists() or not backtest_path.exists():
        log.error("Accuracy artifacts missing: metrics=%s backtest=%s",
                  metrics_path.exists(), backtest_path.exists())
        raise HTTPException(status_code=412, detail="Run python train.py first to generate accuracy data")

    metrics = json.loads(metrics_path.read_text())
    df = joblib.load(backtest_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    classes = metrics.get("classes", ["Draw", "Loss", "Win"])
    wc_df = df[df["tournament"] == "FIFA World Cup"].copy()
    wc_acc = float(wc_df["correct"].mean() * 100) if len(wc_df) else 0.0

    # Per-year backtest records
    wc_years = sorted(wc_df["year"].unique().tolist(), reverse=True)
    backtest_by_year: dict[int, list[WCMatch]] = {}
    for year in wc_years:
        sub = wc_df[wc_df["year"] == year]
        backtest_by_year[year] = [
            WCMatch(
                date=str(r["date"].date()),
                home_team=r["home_team"],
                away_team=r["away_team"],
                score=f"{r['home_score']} – {r['away_score']}",
                actual=r["actual"],
                predicted=r["predicted"],
                correct=bool(r["correct"]),
            )
            for _, r in sub.iterrows()
        ]

    s = get_state()
    log.info("Accuracy: cv=%.2f%%  wc=%.2f%%  rows=%d  model=%s  wc_matches=%d",
             metrics["cv_accuracy_mean"] * 100, wc_acc,
             metrics.get("n_training_rows", 0), metrics.get("model_type", "?"), len(wc_df))
    resp = AccuracyResponse(
        cv_accuracy=round(metrics["cv_accuracy_mean"] * 100, 2),
        cv_std=round(metrics["cv_accuracy_std"] * 100, 2),
        n_training_rows=metrics.get("n_training_rows", 0),
        model_type=metrics.get("model_type", "—"),
        wc_accuracy=round(wc_acc, 2),
        wc_matches=len(wc_df),
        wc_years=wc_years,
        backtest_by_year=backtest_by_year,
        feature_chart=_feature_chart(s.model),
        confusion_chart=_confusion_chart(wc_df, classes),
        wc_year_chart=_wc_year_chart(wc_df),
        calibration_chart=_calibration_chart(df, classes),
        tournament_chart=_tournament_chart(df),
    )
    _cache = resp
    _cache_ts = time.perf_counter()
    log.info("GET /api/accuracy done in %.3fs (cached)", time.perf_counter() - t0)
    return resp
