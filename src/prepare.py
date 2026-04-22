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

# Tournament importance tier — used as a feature and to filter training data.
# Friendlies (tier 1) are kept for rolling stats but excluded from training.
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

def _get_tier(tournament: str) -> int:
    return TOURNAMENT_TIERS.get(tournament, 2)  # default: regional competitive


FEATURE_COLS = [
    "home_win_rate", "home_draw_rate", "home_loss_rate",
    "home_avg_scored", "home_avg_conceded", "home_recent_form",
    "away_win_rate", "away_draw_rate", "away_loss_rate",
    "away_avg_scored", "away_avg_conceded", "away_recent_form",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
    "h2h_total_games", "is_neutral", "tournament_tier",
    "home_rank", "away_rank", "rank_diff",
]


def load_rankings(path: str) -> pd.DataFrame:
    """
    Load FIFA rankings CSV (tadhgfitzgerald dataset).
    Returns a DataFrame sorted by rank_date with normalised country names.
    """
    rk = pd.read_csv(path, parse_dates=["rank_date"])
    rk["country_full"] = rk["country_full"].replace(RANKINGS_NAME_MAP)
    rk = rk.sort_values("rank_date")
    return rk[["rank_date", "country_full", "rank", "total_points"]]


def _lookup_rank(team: str, date, rank_index: dict) -> tuple[int, float]:
    """
    Find the most recent ranking for `team` on or before `date`.
    rank_index: {team: sorted list of (date, rank, points)}
    Returns (rank, points) or (200, 0.0) if no data.
    """
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
    """Pre-build lookup structure: {team: [(date, rank, points), ...]}"""
    index: dict = {}
    for team, grp in rankings_df.groupby("country_full"):
        index[team] = list(zip(grp["rank_date"], grp["rank"].astype(int), grp["total_points"].astype(float)))
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
        "games": 0, "wins": 0, "draws": 0, "losses": 0,
        "goals_scored": 0, "goals_conceded": 0,
        "recent": deque(maxlen=10),
    }


def _new_h2h():
    return {"a_wins": 0, "draws": 0, "b_wins": 0}


def build_features(df: pd.DataFrame, rank_index: dict | None = None):
    """
    Build rolling feature matrix with no data leakage.
    Features for each match are computed from all *prior* matches only.
    rank_index: optional pre-built ranking lookup (from build_rank_index).
    Returns (X, y, team_stats, h2h).
    """
    team_stats: dict = defaultdict(_new_team)
    h2h: dict = defaultdict(_new_h2h)

    feature_rows = []
    labels = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        ht, at = team_stats[h], team_stats[a]
        hg, ag = max(ht["games"], 1), max(at["games"], 1)

        # H2H — always keyed in sorted order so (A,B) == (B,A)
        key = tuple(sorted([h, a]))
        rec = h2h[key]
        h_is_first = key[0] == h
        h2h_total = rec["a_wins"] + rec["draws"] + rec["b_wins"]

        if h_is_first:
            h2h_hw = rec["a_wins"] / max(h2h_total, 1)
            h2h_aw = rec["b_wins"] / max(h2h_total, 1)
        else:
            h2h_hw = rec["b_wins"] / max(h2h_total, 1)
            h2h_aw = rec["a_wins"] / max(h2h_total, 1)

        # FIFA rankings at match date (0 = no data available)
        match_date = row["date"]
        if rank_index:
            h_rank, _ = _lookup_rank(h, match_date, rank_index)
            a_rank, _ = _lookup_rank(a, match_date, rank_index)
        else:
            h_rank, a_rank = 0, 0

        feature_rows.append({
            "home_win_rate":      ht["wins"]          / hg,
            "home_draw_rate":     ht["draws"]         / hg,
            "home_loss_rate":     ht["losses"]        / hg,
            "home_avg_scored":    ht["goals_scored"]  / hg,
            "home_avg_conceded":  ht["goals_conceded"]/ hg,
            "home_recent_form":   sum(ht["recent"]) / 30 if ht["recent"] else 0,
            "away_win_rate":      at["wins"]          / ag,
            "away_draw_rate":     at["draws"]         / ag,
            "away_loss_rate":     at["losses"]        / ag,
            "away_avg_scored":    at["goals_scored"]  / ag,
            "away_avg_conceded":  at["goals_conceded"]/ ag,
            "away_recent_form":   sum(at["recent"]) / 30 if at["recent"] else 0,
            "h2h_home_win_rate":  h2h_hw,
            "h2h_draw_rate":      rec["draws"] / max(h2h_total, 1),
            "h2h_away_win_rate":  h2h_aw,
            "h2h_total_games":    h2h_total,
            "is_neutral":         int(row["neutral"]),
            "tournament_tier":    _get_tier(row["tournament"]),
            "home_rank":          h_rank,
            "away_rank":          a_rank,
            "rank_diff":          a_rank - h_rank,  # positive = home is better ranked
        })

        # Label from home team perspective
        if hs > as_:
            result, h_pts, a_pts = "Win", 3, 0
        elif hs == as_:
            result, h_pts, a_pts = "Draw", 1, 1
        else:
            result, h_pts, a_pts = "Loss", 0, 3
        labels.append(result)

        # Update running stats (after recording features to avoid leakage)
        ht["games"] += 1
        ht["goals_scored"] += hs
        ht["goals_conceded"] += as_
        at["games"] += 1
        at["goals_scored"] += as_
        at["goals_conceded"] += hs

        if result == "Win":
            ht["wins"] += 1
            at["losses"] += 1
        elif result == "Draw":
            ht["draws"] += 1
            at["draws"] += 1
        else:
            ht["losses"] += 1
            at["wins"] += 1

        ht["recent"].append(h_pts)
        at["recent"].append(a_pts)

        # Update H2H
        if h_is_first:
            if hs > as_:   rec["a_wins"] += 1
            elif hs == as_: rec["draws"] += 1
            else:           rec["b_wins"] += 1
        else:
            if hs > as_:   rec["b_wins"] += 1
            elif hs == as_: rec["draws"] += 1
            else:           rec["a_wins"] += 1

    X = pd.DataFrame(feature_rows)[FEATURE_COLS]
    y = pd.Series(labels, name="result")
    return X, y, team_stats, h2h


def serialize_stats(team_stats: dict, h2h: dict) -> tuple[dict, dict]:
    """Convert defaultdicts/deques to plain dicts/lists for joblib serialisation."""
    ts = {
        team: {k: (list(v) if k == "recent" else v) for k, v in s.items()}
        for team, s in team_stats.items()
    }
    return ts, dict(h2h)


def build_feature_vector(
    home_team: str, away_team: str, is_neutral: bool,
    team_stats: dict, h2h: dict,
    tournament_tier: int = 5,
    home_rank: int = 0, away_rank: int = 0,
) -> dict:
    """Build one feature dict for inference (uses saved end-of-history stats)."""
    ht = team_stats.get(home_team, {})
    at = team_stats.get(away_team, {})
    hg = max(ht.get("games", 0), 1)
    ag = max(at.get("games", 0), 1)

    key = tuple(sorted([home_team, away_team]))
    rec = h2h.get(key, {"a_wins": 0, "draws": 0, "b_wins": 0})
    h_is_first = key[0] == home_team
    h2h_total = rec["a_wins"] + rec["draws"] + rec["b_wins"]

    if h_is_first:
        h2h_hw = rec["a_wins"] / max(h2h_total, 1)
        h2h_aw = rec["b_wins"] / max(h2h_total, 1)
    else:
        h2h_hw = rec["b_wins"] / max(h2h_total, 1)
        h2h_aw = rec["a_wins"] / max(h2h_total, 1)

    rh = ht.get("recent", [])
    ra = at.get("recent", [])

    return {
        "home_win_rate":      ht.get("wins",           0) / hg,
        "home_draw_rate":     ht.get("draws",          0) / hg,
        "home_loss_rate":     ht.get("losses",         0) / hg,
        "home_avg_scored":    ht.get("goals_scored",   0) / hg,
        "home_avg_conceded":  ht.get("goals_conceded", 0) / hg,
        "home_recent_form":   sum(rh) / 30 if rh else 0,
        "away_win_rate":      at.get("wins",           0) / ag,
        "away_draw_rate":     at.get("draws",          0) / ag,
        "away_loss_rate":     at.get("losses",         0) / ag,
        "away_avg_scored":    at.get("goals_scored",   0) / ag,
        "away_avg_conceded":  at.get("goals_conceded", 0) / ag,
        "away_recent_form":   sum(ra) / 30 if ra else 0,
        "h2h_home_win_rate":  h2h_hw,
        "h2h_draw_rate":      rec["draws"] / max(h2h_total, 1),
        "h2h_away_win_rate":  h2h_aw,
        "h2h_total_games":    h2h_total,
        "is_neutral":         int(is_neutral),
        "tournament_tier":    tournament_tier,
        "home_rank":          home_rank,
        "away_rank":          away_rank,
        "rank_diff":          away_rank - home_rank,
    }
