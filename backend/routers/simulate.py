import asyncio
import json
import logging
import time

import plotly.graph_objects as go
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from backend.state import get_state
from backend.routers._charts import _DARK_LAYOUT
from src.bracket import (
    simulate_groups,
    most_likely_group_standings,
    build_r32_bracket,
    simulate_knockout,
)
from src.flags import flag

router = APIRouter()
log = logging.getLogger(__name__)


def _bracket_figure(ko_results: dict, bracket: list[str]) -> str:
    def likely(t1, t2, rnd):
        return t1 if ko_results.get(t1, {}).get(rnd, 0) >= ko_results.get(t2, {}).get(rnd, 0) else t2

    r16 = [likely(bracket[i], bracket[i+1], "R16") for i in range(0, len(bracket), 2)]
    qf  = [likely(r16[i],    r16[i+1],    "QF")  for i in range(0, len(r16), 2)]
    sf  = [likely(qf[i],     qf[i+1],     "SF")  for i in range(0, len(qf), 2)]
    fin = [likely(sf[i],     sf[i+1],     "Final") for i in range(0, len(sf), 2)]
    champ = likely(fin[0], fin[1], "Winner")

    x_qf, x_sf, x_fin, x_champ = 0, 3, 6, 9
    mx1, mx2, mx3 = 1.5, 4.5, 7.5
    y_qf  = [14, 12, 8, 6, 2, 0, -4, -6]
    y_sf  = [13, 7, 1, -5]
    y_fin = [10, -2]
    y_chmp = 4

    LINE     = dict(color="#4a4a6a", width=2)
    WIN_LINE = dict(color="#f39c12", width=3)
    shapes, annotations = [], []

    def hline(x0, x1, y, style=LINE):
        shapes.append(dict(type="line", x0=x0, y0=y, x1=x1, y1=y, line=style))

    def vline(x, y0, y1, style=LINE):
        shapes.append(dict(type="line", x0=x, y0=y0, x1=x, y1=y1, line=style))

    def label(x, y, text, anchor="right", size=11, color="#e0e0e0", bold=False):
        annotations.append(dict(
            x=x, y=y, text=f"<b>{text}</b>" if bold else text,
            showarrow=False, xanchor=anchor, yanchor="middle",
            font=dict(size=size, color=color, family="'Segoe UI', sans-serif"),
        ))

    for i, (team, y) in enumerate(zip(qf[:8], y_qf)):
        pct = ko_results.get(team, {}).get("QF", 0)
        label(x_qf - 0.15, y, f"{flag(team)} {team}  {pct:.0f}%", anchor="right")
        hline(x_qf, mx1, y)
    for i in range(0, 8, 2):
        vline(mx1, y_qf[i], y_qf[i+1])
        hline(mx1, x_sf, y_sf[i // 2])
    for i, (team, y) in enumerate(zip(sf, y_sf)):
        pct = ko_results.get(team, {}).get("SF", 0)
        label(x_sf + 0.15, y, f"{flag(team)} {team}  {pct:.0f}%", anchor="left")
        hline(x_sf, mx2, y)
    for i in range(0, 4, 2):
        vline(mx2, y_sf[i], y_sf[i+1])
        hline(mx2, x_fin, y_fin[i // 2])
    for i, (team, y) in enumerate(zip(fin, y_fin)):
        pct = ko_results.get(team, {}).get("Final", 0)
        label(x_fin + 0.15, y, f"{flag(team)} {team}  {pct:.0f}%", anchor="left")
        hline(x_fin, mx3, y)
    vline(mx3, y_fin[0], y_fin[1], style=WIN_LINE)
    hline(mx3, x_champ, y_chmp, style=WIN_LINE)
    champ_pct = ko_results.get(champ, {}).get("Winner", 0)
    label(x_champ + 0.2, y_chmp,
          f"🏆 {flag(champ)} {champ}  {champ_pct:.1f}%",
          anchor="left", size=14, color="#f39c12", bold=True)
    for x, title in [(x_qf, "Quarter-Finals"), (x_sf, "Semi-Finals"),
                     (x_fin, "Final"), (x_champ, "Champion")]:
        annotations.append(dict(
            x=x, y=16.5, text=f"<b>{title}</b>",
            showarrow=False, xanchor="center", yanchor="bottom",
            font=dict(size=12, color="#aaaaaa"),
        ))

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(range=[-5.5, 14], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-8, 18],   showgrid=False, zeroline=False, showticklabels=False),
        height=600, margin=dict(l=10, r=10, t=30, b=10),
        **_DARK_LAYOUT,
    )
    return fig.to_json()


def _win_chart(bracket: list[str], ko_results: dict) -> str:
    from src.flags import with_flag
    df_rows = [{"team": with_flag(t), "pct": ko_results[t]["Winner"]} for t in bracket]
    df_rows.sort(key=lambda x: x["pct"], reverse=True)
    top = df_rows[:16]
    fig = go.Figure(go.Bar(
        x=[r["pct"] for r in top],
        y=[r["team"] for r in top],
        orientation="h",
        marker_color="#f39c12",
        text=[f"{r['pct']:.1f}%" for r in top],
        textposition="outside", cliponaxis=False,
    ))
    max_pct = max(r["pct"] for r in top) if top else 10
    fig.update_layout(
        xaxis=dict(title="Win probability (%)", range=[0, max_pct * 1.25]),
        yaxis=dict(autorange="reversed"),
        height=420, margin=dict(l=0, r=60, t=10, b=30),
        **_DARK_LAYOUT,
    )
    return fig.to_json()


def _event(data: dict) -> str:
    return json.dumps(data)


async def _stream(n_sims: int):
    t0 = time.perf_counter()
    log.info("Simulation started: n_sims=%d", n_sims)
    s = get_state()

    yield _event({"type": "progress", "stage": "groups", "pct": 0, "label": "Simulating group stage…"})

    t1 = time.perf_counter()
    group_results = await asyncio.to_thread(
        simulate_groups,
        s.groups, s.model, s.team_stats, s.h2h,
        n_sims, s.rankings, s.elo_ratings,
    )
    log.info("Group stage done in %.2fs", time.perf_counter() - t1)

    yield _event({"type": "progress", "stage": "groups", "pct": 100, "label": "Group stage done"})
    yield _event({"type": "progress", "stage": "knockout", "pct": 0, "label": "Simulating knockout rounds…"})

    t2 = time.perf_counter()
    likely_standings = await asyncio.to_thread(
        most_likely_group_standings,
        s.groups, s.model, s.team_stats, s.h2h, s.rankings, s.elo_ratings,
    )
    bracket = build_r32_bracket(likely_standings, group_results)
    ko_results = await asyncio.to_thread(
        simulate_knockout,
        bracket, s.model, s.team_stats, s.h2h,
        n_sims, s.rankings, s.elo_ratings,
    )
    log.info("Knockout stage done in %.2fs", time.perf_counter() - t2)

    yield _event({"type": "progress", "stage": "knockout", "pct": 100, "label": "Knockout stage done"})

    bracket_chart = await asyncio.to_thread(_bracket_figure, ko_results, bracket)
    win_chart     = await asyncio.to_thread(_win_chart, bracket, ko_results)

    # Log top 3 predicted winners
    top3 = sorted(ko_results.items(), key=lambda x: x[1].get("Winner", 0), reverse=True)[:3]
    log.info("Top 3 predicted winners: %s",
             ", ".join(f"{t}={r['Winner']:.1f}%" for t, r in top3))
    log.info("Simulation complete in %.2fs total", time.perf_counter() - t0)

    yield _event({
        "type": "result",
        "data": {
            "group_results":     group_results,
            "likely_standings":  likely_standings,
            "bracket":           bracket,
            "ko_results":        ko_results,
            "bracket_chart":     bracket_chart,
            "win_chart":         win_chart,
        },
    })


@router.get("/simulate/stream")
async def simulate_stream(n_sims: int = 500):
    log.info("GET /api/simulate/stream  n_sims=%d", n_sims)
    return EventSourceResponse(_stream(n_sims))
