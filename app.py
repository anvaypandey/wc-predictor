"""
World Cup Match Predictor — Streamlit app.

Run after training:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib

from src.prepare import build_feature_vector, FEATURE_COLS

ARTIFACTS = Path("artifacts")

st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="centered",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    if not (ARTIFACTS / "model.pkl").exists():
        return None
    model        = joblib.load(ARTIFACTS / "model.pkl")
    team_stats   = joblib.load(ARTIFACTS / "team_stats.pkl")
    h2h          = joblib.load(ARTIFACTS / "h2h.pkl")
    teams        = joblib.load(ARTIFACTS / "teams.pkl")
    return model, team_stats, h2h, teams


def run_prediction(home: str, away: str, neutral: bool,
                   model, team_stats: dict, h2h: dict) -> dict[str, float]:
    feat = build_feature_vector(home, away, neutral, team_stats, h2h)
    X = pd.DataFrame([feat])[FEATURE_COLS]
    proba = model.predict_proba(X)[0]
    return dict(zip(model.classes_, proba))


def team_summary(name: str, stats: dict) -> dict:
    s = stats.get(name, {})
    g = max(s.get("games", 0), 1)
    r = s.get("recent", [])
    return {
        "Games":              s.get("games", 0),
        "Win rate":           f"{s.get('wins',  0)/g:.1%}",
        "Draw rate":          f"{s.get('draws', 0)/g:.1%}",
        "Avg goals scored":   f"{s.get('goals_scored',   0)/g:.2f}",
        "Avg goals conceded": f"{s.get('goals_conceded', 0)/g:.2f}",
        "Recent form":        f"{sum(r)/30:.1%}" if r else "—",
    }


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("⚽ World Cup Match Predictor")
st.caption("Win / Draw / Loss predictions powered by historical FIFA international match data")

artifacts = load_artifacts()

if artifacts is None:
    st.warning("No trained model found. Run the training pipeline first.")
    st.code("pip install -r requirements.txt\npython train.py", language="bash")
    st.info("See `data/README.md` for instructions on downloading the dataset.")
    st.stop()

model, team_stats, h2h, teams = artifacts

# ── Team selection ────────────────────────────────────────────────────────────

col_home, col_vs, col_away = st.columns([10, 1, 10])

with col_home:
    default_home = teams.index("Brazil") if "Brazil" in teams else 0
    home_team = st.selectbox("Team 1", teams, index=default_home)

with col_vs:
    st.markdown("<div style='text-align:center;padding-top:32px;font-size:1.2rem'>vs</div>",
                unsafe_allow_html=True)

with col_away:
    default_away = teams.index("Argentina") if "Argentina" in teams else min(1, len(teams) - 1)
    away_team = st.selectbox("Team 2", teams, index=default_away)

is_neutral = st.checkbox(
    "Neutral venue",
    value=True,
    help="World Cup matches are played at neutral venues — tick this for a realistic prediction.",
)

if home_team == away_team:
    st.error("Select two different teams.")
    st.stop()

predict_clicked = st.button("Predict Match Outcome", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────

if predict_clicked:
    proba = run_prediction(home_team, away_team, is_neutral, model, team_stats, h2h)

    win_p  = proba.get("Win",  0.0)
    draw_p = proba.get("Draw", 0.0)
    loss_p = proba.get("Loss", 0.0)

    if win_p >= draw_p and win_p >= loss_p:
        headline, badge_color = f"{home_team} wins", "#27ae60"
    elif draw_p >= win_p and draw_p >= loss_p:
        headline, badge_color = "Draw", "#e67e22"
    else:
        headline, badge_color = f"{away_team} wins", "#c0392b"

    st.markdown("---")

    # Prediction badge
    st.markdown(
        f'<div style="padding:18px 24px;background:{badge_color}18;'
        f'border-left:5px solid {badge_color};border-radius:6px;margin-bottom:8px">'
        f'<span style="font-size:1.05rem;color:#888">Predicted outcome</span><br>'
        f'<span style="font-size:1.9rem;font-weight:700;color:{badge_color}">{headline}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Probability chart
    st.markdown("#### Probability breakdown")
    fig = go.Figure(go.Bar(
        x=[win_p * 100, draw_p * 100, loss_p * 100],
        y=[f"{home_team} Win", "Draw", f"{away_team} Win"],
        orientation="h",
        marker_color=["#27ae60", "#e67e22", "#c0392b"],
        text=[f"{win_p*100:.1f}%", f"{draw_p*100:.1f}%", f"{loss_p*100:.1f}%"],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 105], title="Probability (%)", showgrid=False),
        yaxis=dict(autorange="reversed"),
        height=200,
        margin=dict(l=0, r=50, t=10, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Team stats comparison
    st.markdown("#### Team stats")
    hs = team_summary(home_team, team_stats)
    as_ = team_summary(away_team, team_stats)
    rows = [{"Stat": k, home_team: hs[k], away_team: as_[k]} for k in hs]
    st.dataframe(
        pd.DataFrame(rows).set_index("Stat"),
        use_container_width=True,
    )

    # Head-to-head
    st.markdown("#### Head-to-head record")
    key = tuple(sorted([home_team, away_team]))
    rec = h2h.get(key, {"a_wins": 0, "draws": 0, "b_wins": 0})
    h_is_first = key[0] == home_team
    h2h_total = rec["a_wins"] + rec["draws"] + rec["b_wins"]

    if h2h_total == 0:
        st.info(f"No recorded head-to-head matches between {home_team} and {away_team}.")
    else:
        h_wins = rec["a_wins"] if h_is_first else rec["b_wins"]
        a_wins = rec["b_wins"] if h_is_first else rec["a_wins"]
        draws  = rec["draws"]

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home_team} wins", h_wins)
        c2.metric("Draws", draws)
        c3.metric(f"{away_team} wins", a_wins)
        st.caption(f"Based on {h2h_total} all-time meetings in the dataset.")
