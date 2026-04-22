import shap
import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException

from backend.schemas import PredictRequest, PredictResponse, TeamStats, H2HRecord, TeamsResponse
from backend.state import get_state
from src.prepare import build_feature_vector, FEATURE_COLS
from src.flags import with_flag

router = APIRouter()

_FEATURE_LABELS = {
    "home_win_rate": "Home win rate", "home_draw_rate": "Home draw rate",
    "home_loss_rate": "Home loss rate", "home_avg_scored": "Home avg scored",
    "home_avg_conceded": "Home avg conceded", "home_recent_form": "Home recent form",
    "away_win_rate": "Away win rate", "away_draw_rate": "Away draw rate",
    "away_loss_rate": "Away loss rate", "away_avg_scored": "Away avg scored",
    "away_avg_conceded": "Away avg conceded", "away_recent_form": "Away recent form",
    "home_comp_win_rate": "Home competitive win rate", "home_comp_draw_rate": "Home competitive draw rate",
    "away_comp_win_rate": "Away competitive win rate", "away_comp_draw_rate": "Away competitive draw rate",
    "home_recent_form_5": "Home form (last 5)", "home_recent_gd": "Home avg goal diff",
    "away_recent_form_5": "Away form (last 5)", "away_recent_gd": "Away avg goal diff",
    "h2h_home_win_rate": "H2H home win rate", "h2h_draw_rate": "H2H draw rate",
    "h2h_away_win_rate": "H2H away win rate", "h2h_total_games": "H2H total games",
    "is_neutral": "Neutral venue", "tournament_tier": "Tournament tier",
    "home_rank": "Home FIFA rank", "away_rank": "Away FIFA rank", "rank_diff": "Rank difference",
    "home_elo": "Home ELO", "away_elo": "Away ELO", "elo_diff": "ELO difference",
    "abs_elo_diff": "Abs ELO difference",
}


def _team_stats(name: str, team_stats: dict, rankings: dict) -> TeamStats:
    s = team_stats.get(name, {})
    g = max(s.get("games", 0), 1)
    r = s.get("recent", [])
    rank = rankings.get(name, {}).get("rank", "—")
    return TeamStats(
        fifa_rank=f"#{rank}" if isinstance(rank, int) else rank,
        games=s.get("games", 0),
        win_rate=f"{s.get('wins', 0) / g:.1%}",
        draw_rate=f"{s.get('draws', 0) / g:.1%}",
        avg_scored=f"{s.get('goals_scored', 0) / g:.2f}",
        avg_conceded=f"{s.get('goals_conceded', 0) / g:.2f}",
        recent_form=f"{sum(r) / (len(r) * 3):.1%}" if r else "—",
    )


def _prob_chart(home: str, away: str, win_p: float, draw_p: float, loss_p: float) -> str:
    fig = go.Figure(go.Bar(
        x=[win_p * 100, draw_p * 100, loss_p * 100],
        y=[f"{with_flag(home)} Win", "Draw", f"{with_flag(away)} Win"],
        orientation="h",
        marker_color=["#27ae60", "#e67e22", "#c0392b"],
        text=[f"{win_p*100:.1f}%", f"{draw_p*100:.1f}%", f"{loss_p*100:.1f}%"],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 105], title="Probability (%)", showgrid=False),
        yaxis=dict(autorange="reversed"),
        height=200, margin=dict(l=0, r=50, t=10, b=30),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    return fig.to_json()


def _shap_chart(model, X: pd.DataFrame, predicted_class: str) -> str:
    explainer = shap.TreeExplainer(model._clf)
    shv = explainer(X)
    class_idx = list(model.classes_).index(predicted_class)
    contributions = shv.values[0, :, class_idx]
    names = [_FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
    pairs = sorted(zip(contributions, names), key=lambda x: abs(x[0]))
    vals, lbls = zip(*pairs)
    colors = ["#27ae60" if v > 0 else "#c0392b" for v in vals]
    fig = go.Figure(go.Bar(
        x=list(vals), y=list(lbls), orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside", cliponaxis=False,
    ))
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        xaxis_title=f"SHAP contribution toward '{predicted_class}'",
        height=460, margin=dict(l=10, r=70, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    return fig.to_json()


@router.get("/teams", response_model=TeamsResponse)
def teams():
    s = get_state()
    return TeamsResponse(teams=s.teams, rankings=s.rankings, groups=s.groups)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    s = get_state()
    if req.home not in s.team_stats or req.away not in s.team_stats:
        raise HTTPException(status_code=404, detail="One or both teams not found")
    if req.home == req.away:
        raise HTTPException(status_code=400, detail="Teams must be different")

    hr = s.rankings.get(req.home, {}).get("rank", 0)
    ar = s.rankings.get(req.away, {}).get("rank", 0)
    feat = build_feature_vector(
        req.home, req.away, req.neutral, s.team_stats, s.h2h,
        tournament_tier=5, home_rank=hr, away_rank=ar, elo_ratings=s.elo_ratings,
    )
    X = pd.DataFrame([feat])[FEATURE_COLS]
    proba = s.model.predict_proba(X)[0]
    proba_dict = dict(zip(s.model.classes_, proba))
    win_p  = float(proba_dict.get("Win",  0))
    draw_p = float(proba_dict.get("Draw", 0))
    loss_p = float(proba_dict.get("Loss", 0))
    predicted = max(proba_dict, key=proba_dict.get)

    key = tuple(sorted([req.home, req.away]))
    rec = s.h2h.get(key, {"a_wins": 0, "draws": 0, "b_wins": 0})
    h_is_first = key[0] == req.home
    h2h = H2HRecord(
        home_wins=rec["a_wins"] if h_is_first else rec["b_wins"],
        draws=rec["draws"],
        away_wins=rec["b_wins"] if h_is_first else rec["a_wins"],
        total=rec["a_wins"] + rec["draws"] + rec["b_wins"],
    )

    return PredictResponse(
        home=req.home, away=req.away,
        predicted=predicted,
        win_prob=win_p, draw_prob=draw_p, loss_prob=loss_p,
        home_stats=_team_stats(req.home, s.team_stats, s.rankings),
        away_stats=_team_stats(req.away, s.team_stats, s.rankings),
        h2h=h2h,
        prob_chart=_prob_chart(req.home, req.away, win_p, draw_p, loss_p),
        shap_chart=_shap_chart(s.model, X, predicted),
    )
