"""
2026 FIFA World Cup Bracket Simulator.
Simulates the full tournament using Monte Carlo (500 runs) and the trained model.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

from src.prepare import build_feature_vector, FEATURE_COLS
from src.bracket import (
    simulate_groups,
    most_likely_group_standings,
    simulate_knockout,
    build_r32_bracket,
)
from src.scraper import fetch_fifa_rankings
from src.flags import flag, with_flag

ARTIFACTS = Path("artifacts")
GROUPS_FILE = Path("data/wc2026_groups.json")


# ── Load resources ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not (ARTIFACTS / "model.pkl").exists():
        return None
    elo_path = ARTIFACTS / "elo_ratings.pkl"
    return (
        joblib.load(ARTIFACTS / "model.pkl"),
        joblib.load(ARTIFACTS / "team_stats.pkl"),
        joblib.load(ARTIFACTS / "h2h.pkl"),
        joblib.load(elo_path) if elo_path.exists() else {},
    )


@st.cache_data(ttl=86400)
def load_rankings():
    return fetch_fifa_rankings()


@st.cache_data
def load_groups():
    with open(GROUPS_FILE) as f:
        data = json.load(f)
    return data["groups"]


def medal(pos: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, "  ")


# ── Bracket tree ──────────────────────────────────────────────────────────────

def _expected_winners(bracket: list[str], ko_results: dict):
    """
    Trace most-likely bracket path using round probabilities.
    Returns (qf8, sf4, final2, champion).
    """
    def likely(t1, t2, advance_round):
        return t1 if ko_results.get(t1, {}).get(advance_round, 0) >= ko_results.get(t2, {}).get(advance_round, 0) else t2

    r16 = [likely(bracket[i], bracket[i+1], "R16") for i in range(0, len(bracket), 2)]
    qf  = [likely(r16[i],    r16[i+1],    "QF")  for i in range(0, len(r16), 2)]
    sf  = [likely(qf[i],     qf[i+1],     "SF")  for i in range(0, len(qf), 2)]
    fin = [likely(sf[i],     sf[i+1],     "Final") for i in range(0, len(sf), 2)]
    champ = likely(fin[0], fin[1], "Winner")
    return qf[:8], sf[:4], fin[:2], champ


def _bracket_figure(ko_results: dict, bracket: list[str]) -> go.Figure:
    """Draw a QF→SF→Final→Champion bracket tree using Plotly shapes."""
    qf8, sf4, final2, champ = _expected_winners(bracket, ko_results)

    # Coordinate system
    x_qf, x_sf, x_fin, x_champ = 0, 3, 6, 9
    mx1, mx2, mx3 = 1.5, 4.5, 7.5  # vertical connector x positions

    # Y positions: 8 QF slots, pairs (0,1),(2,3),(4,5),(6,7)
    y_qf   = [14, 12,  8,  6,  2,  0, -4, -6]
    y_sf   = [13,  7,  1, -5]   # midpoints of QF pairs
    y_fin  = [10, -2]            # midpoints of SF pairs
    y_chmp = 4                   # midpoint of Final pair

    LINE = dict(color="#4a4a6a", width=2)
    WIN_LINE = dict(color="#f39c12", width=3)

    shapes = []
    annotations = []

    def hline(x0, x1, y, style=LINE):
        shapes.append(dict(type="line", x0=x0, y0=y, x1=x1, y1=y, line=style))

    def vline(x, y0, y1, style=LINE):
        shapes.append(dict(type="line", x0=x, y0=y0, x1=x, y1=y1, line=style))

    def label(x, y, text, anchor="right", size=11, color="#e0e0e0", bold=False):
        font = dict(size=size, color=color,
                    family="'Segoe UI', sans-serif")
        annotations.append(dict(
            x=x, y=y, text=f"<b>{text}</b>" if bold else text,
            showarrow=False, xanchor=anchor, yanchor="middle", font=font,
        ))

    # ── QF round ──
    for i, (team, y) in enumerate(zip(qf8, y_qf)):
        pct = ko_results.get(team, {}).get("QF", 0)
        f = flag(team)
        label(x_qf - 0.15, y, f"{f} {team}  {pct:.0f}%", anchor="right")
        hline(x_qf, mx1, y)

    # QF vertical connectors + SF horizontal
    for i in range(0, 8, 2):
        vline(mx1, y_qf[i], y_qf[i+1])
        sf_y = y_sf[i // 2]
        hline(mx1, x_sf, sf_y)

    # ── SF round ──
    for i, (team, y) in enumerate(zip(sf4, y_sf)):
        pct = ko_results.get(team, {}).get("SF", 0)
        f = flag(team)
        label(x_sf + 0.15, y, f"{f} {team}  {pct:.0f}%", anchor="left")
        hline(x_sf, mx2, y)

    # SF vertical connectors + Final horizontal
    for i in range(0, 4, 2):
        vline(mx2, y_sf[i], y_sf[i+1])
        fin_y = y_fin[i // 2]
        hline(mx2, x_fin, fin_y)

    # ── Final round ──
    for i, (team, y) in enumerate(zip(final2, y_fin)):
        pct = ko_results.get(team, {}).get("Final", 0)
        f = flag(team)
        label(x_fin + 0.15, y, f"{f} {team}  {pct:.0f}%", anchor="left")
        hline(x_fin, mx3, y)

    # Final vertical connector + Champion horizontal
    vline(mx3, y_fin[0], y_fin[1], style=WIN_LINE)
    hline(mx3, x_champ, y_chmp, style=WIN_LINE)

    # ── Champion ──
    champ_pct = ko_results.get(champ, {}).get("Winner", 0)
    champ_flag = flag(champ)
    label(x_champ + 0.2, y_chmp,
          f"🏆 {champ_flag} {champ}  {champ_pct:.1f}%",
          anchor="left", size=14, color="#f39c12", bold=True)

    # ── Round headers ──
    for x, title in [(x_qf, "Quarter-Finals"), (x_sf, "Semi-Finals"),
                     (x_fin, "Final"), (x_champ, "Champion")]:
        annotations.append(dict(
            x=x, y=16.5, text=f"<b>{title}</b>",
            showarrow=False, xanchor="center", yanchor="bottom",
            font=dict(size=12, color="#aaaaaa"),
        ))

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(range=[-5.5, 14], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-8, 18],  showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("2026 FIFA World Cup Bracket Simulator")
st.caption("Monte Carlo simulation (500 runs) using the trained match predictor")

artifacts = load_model()
if artifacts is None:
    st.warning("No trained model found. Run `python train.py` first.")
    st.stop()

model, team_stats, h2h, elo_ratings = artifacts
rankings = load_rankings()
groups   = load_groups()

# ── Groups display ────────────────────────────────────────────────────────────

st.markdown("## 🗂️ 2026 World Cup Groups")
st.caption("Groups A–L · 48 teams · 12 groups of 4")

cols = st.columns(4)
group_items = list(groups.items())
for i, (group_name, teams) in enumerate(group_items):
    with cols[i % 4]:
        st.markdown(f"**Group {group_name}**")
        for t in teams:
            rank_str = f"#{rankings.get(t, {}).get('rank', '?')}" if rankings else ""
            f = flag(t)
            st.markdown(
                f"&nbsp;&nbsp;{f} {t} <small style='color:gray'>{rank_str}</small>",
                unsafe_allow_html=True,
            )
        st.markdown("---")

# ── Simulation controls ───────────────────────────────────────────────────────

st.markdown("## 🎲 Simulate Tournament")

c1, c2 = st.columns([2, 1])
with c1:
    n_sims = st.slider("Number of simulations", 100, 1000, 500, step=100,
                       help="More simulations = more accurate probabilities but slower")
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_sim = st.button("▶ Run Full Simulation", type="primary", use_container_width=True)

if run_sim or "group_results" in st.session_state:

    if run_sim:
        with st.spinner(f"Running {n_sims} simulations of the group stage..."):
            group_results = simulate_groups(
                groups, model, team_stats, h2h,
                n_sims=n_sims, rankings=rankings, elo_ratings=elo_ratings,
            )
            likely_standings = most_likely_group_standings(
                groups, model, team_stats, h2h,
                rankings=rankings, elo_ratings=elo_ratings,
            )
            bracket = build_r32_bracket(likely_standings, group_results)
            with st.spinner("Simulating knockout rounds..."):
                ko_results = simulate_knockout(
                    bracket, model, team_stats, h2h,
                    n_sims=n_sims, rankings=rankings, elo_ratings=elo_ratings,
                )
        st.session_state["group_results"]    = group_results
        st.session_state["likely_standings"] = likely_standings
        st.session_state["n_sims"]           = n_sims
        st.session_state["bracket"]          = bracket
        st.session_state["ko_results"]       = ko_results

    group_results    = st.session_state["group_results"]
    likely_standings = st.session_state["likely_standings"]
    bracket          = st.session_state["bracket"]
    ko_results       = st.session_state["ko_results"]

    # ── Group stage results ───────────────────────────────────────────────────

    st.markdown("## 📊 Group Stage Results")
    st.caption("Most likely standings + advance probability from Monte Carlo simulation")

    cols = st.columns(4)
    for i, (group_name, teams) in enumerate(group_items):
        with cols[i % 4]:
            st.markdown(f"**Group {group_name}**")
            order = likely_standings[group_name]
            rows = []
            for pos, team in enumerate(order, 1):
                r = group_results[team]
                adv = r["advance_pct"]
                f = flag(team)
                rows.append({
                    "Pos":     f"{medal(pos)} {pos}",
                    "Team":    f"{f} {team}",
                    "Avg Pts": f"{r['avg_pts']:.1f}",
                    "Advance": f"{adv:.0f}%",
                })
            st.dataframe(
                pd.DataFrame(rows).set_index("Pos"),
                use_container_width=True,
                hide_index=False,
            )

    # ── Knockout bracket ──────────────────────────────────────────────────────

    st.markdown("## 🏟️ Knockout Stage")

    # ── Visual bracket tree ───────────────────────────────────────────────────

    st.markdown("### 🌳 Expected Bracket Path")
    st.caption("Most likely path based on round probabilities — QF through Champion")
    st.plotly_chart(_bracket_figure(ko_results, bracket), use_container_width=True)

    # ── Tournament winner chart ───────────────────────────────────────────────

    st.markdown("### 🥇 Tournament Win Probability")
    winner_df = pd.DataFrame([
        {"Team": with_flag(t), "Win %": ko_results[t]["Winner"]}
        for t in bracket
    ]).sort_values("Win %", ascending=False).head(16)

    fig = go.Figure(go.Bar(
        x=winner_df["Win %"],
        y=winner_df["Team"],
        orientation="h",
        marker_color="#f39c12",
        text=[f"{v:.1f}%" for v in winner_df["Win %"]],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(title="Win probability (%)", range=[0, winner_df["Win %"].max() * 1.25]),
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(l=0, r=60, t=10, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Round-by-round probability table ─────────────────────────────────────

    st.markdown("### 📋 Round-by-Round Probabilities")
    st.caption("Probability (%) of reaching each knockout round")

    ko_df = pd.DataFrame([
        {
            "Team":    with_flag(t),
            "R16 %":   ko_results[t]["R16"],
            "QF %":    ko_results[t]["QF"],
            "SF %":    ko_results[t]["SF"],
            "Final %": ko_results[t]["Final"],
            "Win %":   ko_results[t]["Winner"],
        }
        for t in bracket
    ]).sort_values("Win %", ascending=False)

    st.dataframe(
        ko_df.set_index("Team"),
        use_container_width=True,
        column_config={
            "R16 %":   st.column_config.ProgressColumn("R16",   min_value=0, max_value=100, format="%.1f%%"),
            "QF %":    st.column_config.ProgressColumn("QF",    min_value=0, max_value=100, format="%.1f%%"),
            "SF %":    st.column_config.ProgressColumn("SF",    min_value=0, max_value=100, format="%.1f%%"),
            "Final %": st.column_config.ProgressColumn("Final", min_value=0, max_value=100, format="%.1f%%"),
            "Win %":   st.column_config.ProgressColumn("🏆 Win", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    # Predicted finalist callout
    top2 = ko_df.head(2)["Team"].tolist()
    predicted_winner = ko_df.iloc[0]["Team"]
    win_pct = ko_df.iloc[0]["Win %"]
    st.markdown(
        f'<div style="padding:16px 24px;background:#f39c1218;border-left:5px solid #f39c12;'
        f'border-radius:6px;margin-top:12px">'
        f'<b>Predicted Final:</b> {top2[0]} vs {top2[1]}<br>'
        f'<b>Predicted Champion:</b> {predicted_winner} &nbsp;'
        f'<span style="color:#f39c12">({win_pct:.1f}% chance)</span></div>',
        unsafe_allow_html=True,
    )
