import pandas as pd
from collections import defaultdict, deque

# Names in rankings dataset that differ from results dataset
RANKINGS_NAME_MAP = {
    "IR Iran":            "Iran",
    "Korea Republic":     "South Korea",
    "USA":                "United States",
    "Côte d'Ivoire":      "Ivory Coast",
    "Türkiye":            "Turkey",
    "Congo DR":           "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}

TOURNAMENT_TIERS: dict[str, int] = {
    "FIFA World Cup":                        5,
    "UEFA Euro":                             4,
    "Copa América":                          4,
    "African Cup of Nations":                4,
    "AFC Asian Cup":                         4,
    "Gold Cup":                              4,
    "CONCACAF Championship":                 4,
    "OFC Nations Cup":                       4,
    "FIFA World Cup qualification":          3,
    "UEFA Euro qualification":               3,
    "African Cup of Nations qualification":  3,
    "AFC Asian Cup qualification":           3,
    "Gold Cup qualification":                3,
    "CONCACAF Nations League":               3,
    "UEFA Nations League":                   3,
    "AFC Challenge Cup":                     3,
    "Friendly":                              1,
}

# ELO K factor per tier — WC result moves ratings more than a friendly
_ELO_K: dict[int, float] = {1: 16.0, 2: 24.0, 3: 30.0, 4: 40.0, 5: 50.0}
_ELO_INIT = 1500.0
_COMPETITIVE_TIER = 2   # tier >= this counts as competitive for comp-only stats


def _get_tier(tournament: str) -> int:
    return TOURNAMENT_TIERS.get(tournament, 1)


FEATURE_COLS = [
    # Overall rolling stats
    "home_win_rate", "home_draw_rate", "home_loss_rate",
    "home_avg_scored", "home_avg_conceded", "home_recent_form",
    "away_win_rate", "away_draw_rate", "away_loss_rate",
    "away_avg_scored", "away_avg_conceded", "away_recent_form",
    # Competitive-only stats (more WC-relevant)
    "home_comp_win_rate", "home_comp_draw_rate",
    "away_comp_win_rate", "away_comp_draw_rate",
    # Short-window form + goal-difference momentum
    "home_recent_form_5", "home_recent_gd",
    "away_recent_form_5", "away_recent_gd",
    # Head-to-head
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate", "h2h_total_games",
    # Context
    "is_neutral", "tournament_tier",
    # FIFA rankings
    "home_rank", "away_rank", "rank_diff",
    # ELO
    "home_elo", "away_elo", "elo_diff", "abs_elo_diff",
]


def load_rankings(path: str) -> pd.DataFrame:
    rk = pd.read_csv(path, parse_dates=["rank_date"])
    rk["country_full"] = rk["country_full"].replace(RANKINGS_NAME_MAP)
    rk = rk.sort_values("rank_date")
    return rk[["rank_date", "country_full", "rank", "total_points"]]


def _lookup_rank(team: str, date, rank_index: dict) -> tuple[int, float]:
    entries = rank_index.get(team)
    if not entries:
        return 200, 0.0
    lo, hi = 0, len(entries) - 1
    result = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if entries[mid][0] <= date:
            result = entries[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return (result[1], result[2]) if result else (200, 0.0)


def build_rank_index(rankings_df: pd.DataFrame) -> dict:
    index: dict = {}
    for team, grp in rankings_df.groupby("country_full"):
        index[team] = list(zip(
            grp["rank_date"], grp["rank"].astype(int), grp["total_points"].astype(float)
        ))
    return index


def load_data(path: str = "data/results.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    return df.sort_values("date").reset_index(drop=True)


def _new_team():
    return {
        # Overall stats
        "games": 0, "wins": 0, "draws": 0, "losses": 0,
        "goals_scored": 0, "goals_conceded": 0,
        "recent": deque(maxlen=10),   # points last 10 matches
        "recent_5": deque(maxlen=5),  # points last 5 matches
        "recent_gd": deque(maxlen=10),  # goal difference last 10 matches
        # Competitive-only stats
        "comp_games": 0, "comp_wins": 0, "comp_draws": 0, "comp_losses": 0,
    }


def _new_h2h():
    return {"a_wins": 0, "draws": 0, "b_wins": 0}


def _form_pct(seq) -> float:
    return sum(seq) / (len(seq) * 3) if seq else 0.0


def _h2h_rates(rec: dict, h_is_first: bool, total: int) -> tuple[float, float]:
    denom = max(total, 1)
    if h_is_first:
        return rec["a_wins"] / denom, rec["b_wins"] / denom
    return rec["b_wins"] / denom, rec["a_wins"] / denom


def build_features(df: pd.DataFrame, rank_index: dict | None = None):
    """
    Build rolling feature matrix with no data leakage.
    Features for each match are computed from all *prior* matches only.
    Returns (X, y, team_stats, h2h, elo_ratings).
    """
    team_stats: dict = defaultdict(_new_team)
    h2h: dict = defaultdict(_new_h2h)
    elos: dict = defaultdict(lambda: _ELO_INIT)

    feature_rows = []
    labels = []

    for row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        hs, as_ = int(row.home_score), int(row.away_score)
        ht, at = team_stats[h], team_stats[a]
        hg = max(ht["games"], 1)
        ag = max(at["games"], 1)
        hcg = max(ht["comp_games"], 1)
        acg = max(at["comp_games"], 1)

        # H2H (keyed in sorted order)
        key = tuple(sorted([h, a]))
        rec = h2h[key]
        h_is_first = key[0] == h
        h2h_total = rec["a_wins"] + rec["draws"] + rec["b_wins"]
        h2h_hw, h2h_aw = _h2h_rates(rec, h_is_first, h2h_total)

        # FIFA rankings at match date
        match_date = row.date
        if rank_index:
            h_rank, _ = _lookup_rank(h, match_date, rank_index)
            a_rank, _ = _lookup_rank(a, match_date, rank_index)
        else:
            h_rank, a_rank = 0, 0

        # ELO before this match
        elo_h, elo_a = elos[h], elos[a]
        elo_diff = elo_h - elo_a

        tier = _get_tier(row.tournament)

        feature_rows.append({
            "home_win_rate":       ht["wins"]           / hg,
            "home_draw_rate":      ht["draws"]          / hg,
            "home_loss_rate":      ht["losses"]         / hg,
            "home_avg_scored":     ht["goals_scored"]   / hg,
            "home_avg_conceded":   ht["goals_conceded"] / hg,
            "home_recent_form":    _form_pct(ht["recent"]),
            "away_win_rate":       at["wins"]           / ag,
            "away_draw_rate":      at["draws"]          / ag,
            "away_loss_rate":      at["losses"]         / ag,
            "away_avg_scored":     at["goals_scored"]   / ag,
            "away_avg_conceded":   at["goals_conceded"] / ag,
            "away_recent_form":    _form_pct(at["recent"]),
            "home_comp_win_rate":  ht["comp_wins"]      / hcg,
            "home_comp_draw_rate": ht["comp_draws"]     / hcg,
            "away_comp_win_rate":  at["comp_wins"]      / acg,
            "away_comp_draw_rate": at["comp_draws"]     / acg,
            "home_recent_form_5":  _form_pct(ht["recent_5"]),
            "home_recent_gd":      sum(ht["recent_gd"]) / max(len(ht["recent_gd"]), 1) if ht["recent_gd"] else 0,
            "away_recent_form_5":  _form_pct(at["recent_5"]),
            "away_recent_gd":      sum(at["recent_gd"]) / max(len(at["recent_gd"]), 1) if at["recent_gd"] else 0,
            "h2h_home_win_rate":   h2h_hw,
            "h2h_draw_rate":       rec["draws"] / max(h2h_total, 1),
            "h2h_away_win_rate":   h2h_aw,
            "h2h_total_games":     h2h_total,
            "is_neutral":          int(row.neutral),
            "tournament_tier":     tier,
            "home_rank":           h_rank,
            "away_rank":           a_rank,
            "rank_diff":           a_rank - h_rank,
            "home_elo":            elo_h,
            "away_elo":            elo_a,
            "elo_diff":            elo_diff,
            "abs_elo_diff":        abs(elo_diff),
        })

        # Label from home team perspective
        if hs > as_:
            result, h_pts, a_pts = "Win",  3, 0
        elif hs == as_:
            result, h_pts, a_pts = "Draw", 1, 1
        else:
            result, h_pts, a_pts = "Loss", 0, 3
        labels.append(result)

        # ── Update rolling stats (after features — no leakage) ──
        gd_h = hs - as_
        ht["games"] += 1; ht["goals_scored"] += hs; ht["goals_conceded"] += as_
        at["games"] += 1; at["goals_scored"] += as_; at["goals_conceded"] += hs

        if result == "Win":
            ht["wins"] += 1; at["losses"] += 1
        elif result == "Draw":
            ht["draws"] += 1; at["draws"] += 1
        else:
            ht["losses"] += 1; at["wins"] += 1

        ht["recent"].append(h_pts);    at["recent"].append(a_pts)
        ht["recent_5"].append(h_pts);  at["recent_5"].append(a_pts)
        ht["recent_gd"].append(gd_h);  at["recent_gd"].append(-gd_h)

        if tier >= _COMPETITIVE_TIER:
            ht["comp_games"] += 1; at["comp_games"] += 1
            if result == "Win":
                ht["comp_wins"] += 1; at["comp_losses"] += 1
            elif result == "Draw":
                ht["comp_draws"] += 1; at["comp_draws"] += 1
            else:
                at["comp_wins"] += 1; ht["comp_losses"] += 1

        # Update H2H
        if h_is_first:
            if hs > as_:    rec["a_wins"] += 1
            elif hs == as_: rec["draws"]  += 1
            else:           rec["b_wins"] += 1
        else:
            if hs > as_:    rec["b_wins"] += 1
            elif hs == as_: rec["draws"]  += 1
            else:           rec["a_wins"] += 1

        # Update ELO — margin-adjusted K (dominant wins count more)
        K = _ELO_K.get(tier, 24.0)
        gd_factor = min(1.0 + abs(gd_h) * 0.5, 2.5)  # 1-goal win = 1.5x, 3+ goal win = 2.5x
        K_eff = K * gd_factor
        exp_h = 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h) / 400.0))
        s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        elos[h] += K_eff * (s_h - exp_h)
        elos[a] += K_eff * ((1.0 - s_h) - (1.0 - exp_h))

    X = pd.DataFrame(feature_rows)[FEATURE_COLS]
    y = pd.Series(labels, name="result")
    return X, y, team_stats, h2h, dict(elos)


def serialize_stats(team_stats: dict, h2h: dict) -> tuple[dict, dict]:
    deque_fields = {"recent", "recent_5", "recent_gd"}
    ts = {
        team: {k: (list(v) if k in deque_fields else v) for k, v in s.items()}
        for team, s in team_stats.items()
    }
    return ts, dict(h2h)


def build_feature_vector(
    home_team: str, away_team: str, is_neutral: bool,
    team_stats: dict, h2h: dict,
    tournament_tier: int = 5,
    home_rank: int = 0, away_rank: int = 0,
    elo_ratings: dict | None = None,
) -> dict:
    """Build one feature dict for inference (uses saved end-of-history stats)."""
    ht = team_stats.get(home_team, {})
    at = team_stats.get(away_team, {})
    hg  = max(ht.get("games",      0), 1)
    ag  = max(at.get("games",      0), 1)
    hcg = max(ht.get("comp_games", 0), 1)
    acg = max(at.get("comp_games", 0), 1)

    key = tuple(sorted([home_team, away_team]))
    rec = h2h.get(key, {"a_wins": 0, "draws": 0, "b_wins": 0})
    h_is_first = key[0] == home_team
    h2h_total = rec["a_wins"] + rec["draws"] + rec["b_wins"]
    h2h_hw, h2h_aw = _h2h_rates(rec, h_is_first, h2h_total)

    rh    = ht.get("recent",    [])
    ra    = at.get("recent",    [])
    rh5   = ht.get("recent_5",  [])
    ra5   = at.get("recent_5",  [])
    rh_gd = ht.get("recent_gd", [])
    ra_gd = at.get("recent_gd", [])

    elo_h = elo_ratings.get(home_team, _ELO_INIT) if elo_ratings else _ELO_INIT
    elo_a = elo_ratings.get(away_team, _ELO_INIT) if elo_ratings else _ELO_INIT
    elo_diff = elo_h - elo_a

    return {
        "home_win_rate":       ht.get("wins",           0) / hg,
        "home_draw_rate":      ht.get("draws",          0) / hg,
        "home_loss_rate":      ht.get("losses",         0) / hg,
        "home_avg_scored":     ht.get("goals_scored",   0) / hg,
        "home_avg_conceded":   ht.get("goals_conceded", 0) / hg,
        "home_recent_form":    _form_pct(rh),
        "away_win_rate":       at.get("wins",           0) / ag,
        "away_draw_rate":      at.get("draws",          0) / ag,
        "away_loss_rate":      at.get("losses",         0) / ag,
        "away_avg_scored":     at.get("goals_scored",   0) / ag,
        "away_avg_conceded":   at.get("goals_conceded", 0) / ag,
        "away_recent_form":    _form_pct(ra),
        "home_comp_win_rate":  ht.get("comp_wins",      0) / hcg,
        "home_comp_draw_rate": ht.get("comp_draws",     0) / hcg,
        "away_comp_win_rate":  at.get("comp_wins",      0) / acg,
        "away_comp_draw_rate": at.get("comp_draws",     0) / acg,
        "home_recent_form_5":  _form_pct(rh5),
        "home_recent_gd":      sum(rh_gd) / max(len(rh_gd), 1) if rh_gd else 0,
        "away_recent_form_5":  _form_pct(ra5),
        "away_recent_gd":      sum(ra_gd) / max(len(ra_gd), 1) if ra_gd else 0,
        "h2h_home_win_rate":   h2h_hw,
        "h2h_draw_rate":       rec["draws"] / max(h2h_total, 1),
        "h2h_away_win_rate":   h2h_aw,
        "h2h_total_games":     h2h_total,
        "is_neutral":          int(is_neutral),
        "tournament_tier":     tournament_tier,
        "home_rank":           home_rank,
        "away_rank":           away_rank,
        "rank_diff":           away_rank - home_rank,
        "home_elo":            elo_h,
        "away_elo":            elo_a,
        "elo_diff":            elo_diff,
        "abs_elo_diff":        abs(elo_diff),
    }
