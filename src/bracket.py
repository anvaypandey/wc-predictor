"""
Monte Carlo tournament simulator for the 2026 FIFA World Cup.
Runs N simulations of the group stage and knockout rounds using
match outcome probabilities from the trained model.
"""

import random
from collections import defaultdict
from itertools import combinations

import pandas as pd

from src.prepare import build_feature_vector, FEATURE_COLS


# ── Match helpers ─────────────────────────────────────────────────────────────

def _match_proba(home: str, away: str, model, team_stats: dict, h2h: dict,
                 rankings: dict | None = None,
                 elo_ratings: dict | None = None) -> dict[str, float]:
    """Returns {Win, Draw, Loss} probabilities from the home team's perspective."""
    hr = rankings.get(home, {}).get("rank", 0) if rankings else 0
    ar = rankings.get(away, {}).get("rank", 0) if rankings else 0
    feat = build_feature_vector(
        home, away, is_neutral=True, team_stats=team_stats, h2h=h2h,
        tournament_tier=5, home_rank=hr, away_rank=ar,
        elo_ratings=elo_ratings,
    )
    X = pd.DataFrame([feat])[FEATURE_COLS]
    proba = model.predict_proba(X)[0]
    return dict(zip(model.classes_, proba))


def _sample_outcome(p: dict[str, float]) -> str:
    return random.choices(
        ["Win", "Draw", "Loss"],
        weights=[p["Win"], p["Draw"], p["Loss"]],
    )[0]


def _knockout_winner(t1: str, t2: str, model, team_stats: dict, h2h: dict,
                     rankings: dict | None,
                     elo_ratings: dict | None = None) -> str:
    """Simulate a knockout match. Draws go to 50/50 penalties."""
    p = _match_proba(t1, t2, model, team_stats, h2h, rankings, elo_ratings)
    win_p = p["Win"] + p["Draw"] * 0.5
    return t1 if random.random() < win_p else t2


# ── Group stage ───────────────────────────────────────────────────────────────

def _simulate_group_once(teams: list[str], model, team_stats: dict,
                          h2h: dict, rankings: dict | None,
                          elo_ratings: dict | None = None) -> dict[str, dict]:
    pts = {t: 0 for t in teams}
    gd  = {t: 0 for t in teams}

    for home, away in combinations(teams, 2):
        p = _match_proba(home, away, model, team_stats, h2h, rankings, elo_ratings)
        r = _sample_outcome(p)
        if r == "Win":
            pts[home] += 3; gd[home] += 1; gd[away] -= 1
        elif r == "Draw":
            pts[home] += 1; pts[away] += 1
        else:
            pts[away] += 3; gd[away] += 1; gd[home] -= 1

    order = sorted(teams, key=lambda t: (pts[t], gd[t], random.random()), reverse=True)
    return {t: {"pts": pts[t], "gd": gd[t], "pos": order.index(t) + 1} for t in teams}


def simulate_groups(
    groups: dict[str, list[str]],
    model,
    team_stats: dict,
    h2h: dict,
    n_sims: int = 500,
    rankings: dict | None = None,
    elo_ratings: dict | None = None,
) -> dict[str, dict]:
    """
    Monte Carlo group stage simulation.

    Returns per-team stats:
      avg_pts, advance_pct, 1st_pct, 2nd_pct, 3rd_pct, 4th_pct
    """
    pos_count    = defaultdict(lambda: defaultdict(int))
    pts_sum      = defaultdict(float)
    advance_count = defaultdict(int)

    for _ in range(n_sims):
        third_place = []

        for group_teams in groups.values():
            standings = _simulate_group_once(
                group_teams, model, team_stats, h2h, rankings, elo_ratings
            )
            order = sorted(
                group_teams,
                key=lambda t: (standings[t]["pts"], standings[t]["gd"], random.random()),
                reverse=True,
            )
            for i, t in enumerate(order):
                pos_count[t][i + 1] += 1
                pts_sum[t] += standings[t]["pts"]

            advance_count[order[0]] += 1
            advance_count[order[1]] += 1
            third_place.append((standings[order[2]]["pts"], standings[order[2]]["gd"], order[2]))

        # Best 8 third-place teams advance to R32
        third_place.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for _, _, t in third_place[:8]:
            advance_count[t] += 1

    all_teams = [t for teams in groups.values() for t in teams]
    return {
        team: {
            "avg_pts":      round(pts_sum[team] / n_sims, 2),
            "advance_pct":  round(advance_count[team] / n_sims * 100, 1),
            "1st_pct":      round(pos_count[team][1] / n_sims * 100, 1),
            "2nd_pct":      round(pos_count[team][2] / n_sims * 100, 1),
            "3rd_pct":      round(pos_count[team][3] / n_sims * 100, 1),
            "4th_pct":      round(pos_count[team][4] / n_sims * 100, 1),
        }
        for team in all_teams
    }


def most_likely_group_standings(
    groups: dict[str, list[str]],
    model,
    team_stats: dict,
    h2h: dict,
    rankings: dict | None = None,
    elo_ratings: dict | None = None,
) -> dict[str, list[str]]:
    """
    Run a single deterministic-ish simulation using expected points
    (no sampling — use highest-probability outcome for each match).
    Returns {group_name: [1st, 2nd, 3rd, 4th]} team order.
    """
    result = {}
    for group_name, teams in groups.items():
        pts = {t: 0 for t in teams}
        gd  = {t: 0 for t in teams}
        for home, away in combinations(teams, 2):
            p = _match_proba(home, away, model, team_stats, h2h, rankings, elo_ratings)
            pts[home] += 3 * p["Win"] + 1 * p["Draw"]
            pts[away] += 3 * p["Loss"] + 1 * p["Draw"]
            gd[home]  += p["Win"] - p["Loss"]
            gd[away]  += p["Loss"] - p["Win"]
        order = sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True)
        result[group_name] = order
    return result


# ── Knockout stage ────────────────────────────────────────────────────────────

def simulate_knockout(
    qualifiers: list[str],
    model,
    team_stats: dict,
    h2h: dict,
    n_sims: int = 500,
    rankings: dict | None = None,
    elo_ratings: dict | None = None,
) -> dict[str, dict]:
    """
    Monte Carlo knockout simulation from R32 onwards (32 teams).

    qualifiers: ordered list of 32 teams (bracket seeding preserved).
    Returns per-team probabilities for reaching each round.
    """
    round_reach = defaultdict(lambda: defaultdict(int))
    rounds = ["R16", "QF", "SF", "Final", "Winner"]

    for _ in range(n_sims):
        bracket = list(qualifiers)

        for round_name in rounds:
            next_round = []
            for i in range(0, len(bracket), 2):
                w = _knockout_winner(
                    bracket[i], bracket[i + 1],
                    model, team_stats, h2h, rankings, elo_ratings,
                )
                round_reach[w][round_name] += 1
                next_round.append(w)
            bracket = next_round

    return {
        team: {r: round(round_reach[team][r] / n_sims * 100, 1) for r in rounds}
        for team in qualifiers
    }


def build_r32_bracket(group_standings: dict[str, list[str]],
                      group_sim_results: dict[str, dict]) -> list[str]:
    """
    Construct the 32-team R32 bracket from group standings.
    Top 2 from each of 12 groups (24 teams) + best 8 third-place teams (8 teams).
    Returns an ordered list of 32 teams paired for R32 (index 0 vs 1, 2 vs 3, …).

    Pairings (16 matches):
      - 8 matches: firsts[0..7]  vs best_thirds[0..7]
      - 4 matches: firsts[8..11] vs seconds[8..11]
      - 4 matches: seconds[0..3] vs seconds[4..7]
    """
    firsts, seconds, thirds = [], [], []

    for group_name in sorted(group_standings.keys()):
        order = group_standings[group_name]
        firsts.append(order[0])
        seconds.append(order[1])
        thirds.append((
            group_sim_results[order[2]]["avg_pts"],
            group_sim_results[order[2]]["advance_pct"],
            order[2],
        ))

    thirds.sort(reverse=True)
    best_thirds = [t for _, _, t in thirds[:8]]

    bracket: list[str] = []
    for i in range(8):
        bracket.append(firsts[i])
        bracket.append(best_thirds[i])
    for i in range(8, 12):
        bracket.append(firsts[i])
        bracket.append(seconds[i])
    for i in range(4):
        bracket.append(seconds[i])
        bracket.append(seconds[i + 4])

    assert len(bracket) == 32, f"Bracket has {len(bracket)} teams, expected 32"
    assert len(set(bracket)) == 32, f"Bracket has duplicate teams ({len(set(bracket))} unique)"

    return bracket
