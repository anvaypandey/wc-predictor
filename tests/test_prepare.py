"""
Tests for src/prepare.py — feature engineering, ELO, stats, serialization.
These cover the data plumbing that is easy to break silently.
"""

import sys
from collections import deque
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prepare import (
    FEATURE_COLS,
    _get_tier,
    _lookup_rank,
    _new_team,
    build_feature_vector,
    build_features,
    build_rank_index,
    serialize_stats,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(*matches):
    """Build a minimal results DataFrame from (home, away, hs, as_, neutral, tournament) tuples."""
    rows = []
    for i, (h, a, hs, a_s, neutral, tournament) in enumerate(matches):
        rows.append({
            "date":        pd.Timestamp(f"2020-{i+1:02d}-01"),
            "home_team":   h,
            "away_team":   a,
            "home_score":  hs,
            "away_score":  a_s,
            "neutral":     neutral,
            "tournament":  tournament,
        })
    return pd.DataFrame(rows)


def _make_rank_df(entries):
    """Build a rankings DataFrame from [(date_str, team, rank, points), ...] tuples."""
    rows = [{"rank_date": pd.Timestamp(d), "country_full": t, "rank": r, "total_points": p}
            for d, t, r, p in entries]
    return pd.DataFrame(rows).sort_values("rank_date")


# ── _new_team ─────────────────────────────────────────────────────────────────

def test_new_team_has_all_expected_fields():
    t = _new_team()
    required = {
        "games", "wins", "draws", "losses",
        "goals_scored", "goals_conceded",
        "recent", "recent_5", "recent_gd",
        "comp_games", "comp_wins", "comp_draws", "comp_losses",
    }
    assert required == set(t.keys()), f"Missing or extra fields: {set(t.keys()) ^ required}"


def test_new_team_deques_have_correct_maxlen():
    t = _new_team()
    assert t["recent"].maxlen == 10
    assert t["recent_5"].maxlen == 5
    assert t["recent_gd"].maxlen == 10


def test_new_team_numeric_fields_start_at_zero():
    t = _new_team()
    for field in ("games", "wins", "draws", "losses", "goals_scored", "goals_conceded",
                  "comp_games", "comp_wins", "comp_draws", "comp_losses"):
        assert t[field] == 0, f"{field} should start at 0"


# ── _get_tier ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tournament,expected_tier", [
    ("FIFA World Cup",               5),
    ("UEFA Euro",                    4),
    ("Copa América",                 4),
    ("African Cup of Nations",       4),
    ("FIFA World Cup qualification", 3),
    ("UEFA Nations League",          3),
    ("Friendly",                     1),
])
def test_get_tier_known_tournaments(tournament, expected_tier):
    assert _get_tier(tournament) == expected_tier


def test_get_tier_unknown_defaults_to_friendly():
    assert _get_tier("Some Random Cup") == 1
    assert _get_tier("") == 1
    assert _get_tier("Club Friendly") == 1


# ── _lookup_rank ──────────────────────────────────────────────────────────────

def test_lookup_rank_returns_correct_rank_at_exact_date():
    rk_df = _make_rank_df([
        ("2020-01-01", "Brazil", 1, 1800.0),
        ("2020-07-01", "Brazil", 2, 1750.0),
    ])
    index = build_rank_index(rk_df)
    rank, pts = _lookup_rank("Brazil", pd.Timestamp("2020-07-01"), index)
    assert rank == 2
    assert pts == 1750.0


def test_lookup_rank_uses_most_recent_entry_before_date():
    rk_df = _make_rank_df([
        ("2020-01-01", "Brazil", 1, 1800.0),
        ("2020-07-01", "Brazil", 2, 1750.0),
    ])
    index = build_rank_index(rk_df)
    rank, pts = _lookup_rank("Brazil", pd.Timestamp("2020-03-15"), index)
    assert rank == 1  # Jan entry, March date is between Jan and Jul


def test_lookup_rank_unknown_team_returns_default():
    index = {}
    rank, pts = _lookup_rank("Atlantis FC", pd.Timestamp("2020-01-01"), index)
    assert rank == 200
    assert pts == 0.0


def test_lookup_rank_before_first_entry_returns_default():
    rk_df = _make_rank_df([("2020-06-01", "Brazil", 3, 1700.0)])
    index = build_rank_index(rk_df)
    rank, _ = _lookup_rank("Brazil", pd.Timestamp("2019-01-01"), index)
    assert rank == 200  # no entry before this date


# ── build_features ────────────────────────────────────────────────────────────

def test_build_features_first_match_has_zero_stats():
    df = _make_df(("A", "B", 2, 1, False, "FIFA World Cup"))
    X, y, _, _, _ = build_features(df)
    row = X.iloc[0]
    # Before any match, both teams have no history
    assert row["home_win_rate"] == 0.0
    assert row["away_win_rate"] == 0.0
    assert row["home_recent_form"] == 0.0
    assert row["h2h_total_games"] == 0


def test_build_features_labels_are_correct():
    df = _make_df(
        ("A", "B", 2, 1, False, "FIFA World Cup"),   # A wins → Win
        ("A", "B", 0, 0, False, "FIFA World Cup"),   # draw → Draw
        ("A", "B", 0, 1, False, "FIFA World Cup"),   # A loses → Loss
    )
    _, y, _, _, _ = build_features(df)
    assert list(y) == ["Win", "Draw", "Loss"]


def test_build_features_has_all_feature_cols():
    df = _make_df(("A", "B", 1, 0, True, "FIFA World Cup"))
    X, _, _, _, _ = build_features(df)
    assert list(X.columns) == FEATURE_COLS


def test_build_features_no_leakage_stats_update_after_row():
    df = _make_df(
        ("A", "B", 3, 0, False, "FIFA World Cup"),  # A wins big
        ("A", "B", 1, 0, False, "FIFA World Cup"),  # second match
    )
    X, _, _, _, _ = build_features(df)
    # Row 0: A has no prior wins
    assert X.iloc[0]["home_win_rate"] == 0.0
    # Row 1: A should have 1 win from row 0
    assert X.iloc[1]["home_win_rate"] == 1.0


def test_build_features_h2h_updates_after_match():
    df = _make_df(
        ("A", "B", 2, 0, False, "FIFA World Cup"),
        ("A", "B", 0, 1, False, "FIFA World Cup"),
    )
    X, _, _, _, _ = build_features(df)
    assert X.iloc[0]["h2h_total_games"] == 0    # no prior H2H
    assert X.iloc[1]["h2h_total_games"] == 1    # 1 prior meeting


def test_build_features_elo_winner_gains_rating():
    df = _make_df(("A", "B", 1, 0, False, "FIFA World Cup"))
    _, _, _, _, elos = build_features(df)
    assert elos["A"] > 1500.0, "Winner should gain ELO"
    assert elos["B"] < 1500.0, "Loser should lose ELO"
    assert abs((elos["A"] - 1500.0) + (elos["B"] - 1500.0)) < 1e-9, "ELO is zero-sum"


def test_build_features_elo_draw_moves_toward_expected():
    # Equal teams draw: expect ELO should barely change (both at 1500, E=0.5, S=0.5)
    df = _make_df(("A", "B", 1, 1, False, "FIFA World Cup"))
    _, _, _, _, elos = build_features(df)
    # Both were at 1500 (equal), draw is expected — change should be near zero
    assert abs(elos["A"] - 1500.0) < 5.0
    assert abs(elos["B"] - 1500.0) < 5.0


def test_build_features_recent_form_normalized_by_actual_games():
    # A wins 5 matches in a row — after those 5, recent_5 deque is full with [3,3,3,3,3]
    # form should be 15/15 = 1.0, not 15/15 = 1.0 (same here since maxlen=5)
    # But for the FIRST match, form should be 0 (empty deque)
    df = _make_df(
        ("A", "B", 1, 0, False, "Friendly"),
        ("A", "B", 1, 0, False, "Friendly"),
        ("A", "B", 1, 0, False, "Friendly"),
        ("A", "B", 1, 0, False, "Friendly"),
        ("A", "B", 1, 0, False, "Friendly"),
        ("A", "B", 1, 0, False, "Friendly"),  # 6th match — after 5 wins recorded
    )
    X, _, _, _, _ = build_features(df)
    # Row 0: empty deque → 0
    assert X.iloc[0]["home_recent_form_5"] == 0.0
    # Row 1: 1 win in deque [3] → 3/(1*3) = 1.0
    assert X.iloc[1]["home_recent_form_5"] == pytest.approx(1.0)
    # Row 5: deque full [3,3,3,3,3] → 15/(5*3) = 1.0
    assert X.iloc[5]["home_recent_form_5"] == pytest.approx(1.0)


def test_build_features_comp_losses_tracked_correctly():
    # A loses a competitive match — comp_losses should increment
    df = _make_df(
        ("A", "B", 0, 2, False, "FIFA World Cup"),  # A loses (competitive)
        ("A", "B", 0, 0, False, "FIFA World Cup"),  # draw, so we can inspect prior stats
    )
    X, _, team_stats, _, _ = build_features(df)
    assert team_stats["A"]["comp_losses"] == 1
    assert team_stats["B"]["comp_losses"] == 0


def test_build_features_friendlies_dont_count_as_comp():
    df = _make_df(("A", "B", 0, 2, False, "Friendly"))
    _, _, team_stats, _, _ = build_features(df)
    assert team_stats["A"]["comp_games"] == 0
    assert team_stats["A"]["comp_losses"] == 0


# ── serialize_stats ───────────────────────────────────────────────────────────

def test_serialize_stats_converts_deques_to_lists():
    t = _new_team()
    t["recent"].extend([3, 1, 3])
    t["recent_5"].extend([3, 0])
    t["recent_gd"].extend([2, -1])
    stats = {"TeamA": t}
    ts, _ = serialize_stats(stats, {})
    assert isinstance(ts["TeamA"]["recent"],    list)
    assert isinstance(ts["TeamA"]["recent_5"],  list)
    assert isinstance(ts["TeamA"]["recent_gd"], list)
    assert ts["TeamA"]["recent"]   == [3, 1, 3]
    assert ts["TeamA"]["recent_5"] == [3, 0]


def test_serialize_stats_preserves_numeric_fields():
    t = _new_team()
    t["wins"] = 5; t["comp_wins"] = 3; t["comp_losses"] = 1
    ts, _ = serialize_stats({"X": t}, {})
    assert ts["X"]["wins"] == 5
    assert ts["X"]["comp_wins"] == 3
    assert ts["X"]["comp_losses"] == 1


def test_serialize_stats_h2h_roundtrips():
    from collections import defaultdict
    h2h = defaultdict(lambda: {"a_wins": 0, "draws": 0, "b_wins": 0})
    h2h[("A", "B")]["a_wins"] = 3
    h2h[("A", "B")]["draws"] = 1
    _, h2h_out = serialize_stats({}, h2h)
    assert h2h_out[("A", "B")]["a_wins"] == 3
    assert h2h_out[("A", "B")]["draws"] == 1


# ── build_feature_vector ──────────────────────────────────────────────────────

def test_build_feature_vector_returns_all_feature_cols():
    fv = build_feature_vector("A", "B", True, {}, {})
    assert set(fv.keys()) == set(FEATURE_COLS)


def test_build_feature_vector_unknown_teams_produce_zeros():
    fv = build_feature_vector("Unknown1", "Unknown2", True, {}, {})
    assert fv["home_win_rate"] == 0.0
    assert fv["away_win_rate"] == 0.0
    assert fv["h2h_total_games"] == 0
    assert fv["home_elo"] == 1500.0
    assert fv["away_elo"] == 1500.0
    assert fv["elo_diff"] == 0.0
    assert fv["abs_elo_diff"] == 0.0


def test_build_feature_vector_elo_diff_and_abs_elo_diff_consistent():
    elos = {"Brazil": 1700.0, "Peru": 1400.0}
    fv = build_feature_vector("Brazil", "Peru", True, {}, {}, elo_ratings=elos)
    assert fv["elo_diff"] == pytest.approx(300.0)
    assert fv["abs_elo_diff"] == pytest.approx(300.0)

    fv2 = build_feature_vector("Peru", "Brazil", True, {}, {}, elo_ratings=elos)
    assert fv2["elo_diff"] == pytest.approx(-300.0)
    assert fv2["abs_elo_diff"] == pytest.approx(300.0)


def test_build_feature_vector_rank_diff_is_away_minus_home():
    fv = build_feature_vector("A", "B", True, {}, {}, home_rank=5, away_rank=20)
    assert fv["rank_diff"] == 15  # away_rank - home_rank


def test_build_feature_vector_neutral_venue_flag():
    fv_neutral  = build_feature_vector("A", "B", True,  {}, {})
    fv_home     = build_feature_vector("A", "B", False, {}, {})
    assert fv_neutral["is_neutral"] == 1
    assert fv_home["is_neutral"] == 0
