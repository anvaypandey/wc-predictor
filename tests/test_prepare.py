"""
Tests for src/prepare.py — feature engineering, ELO, stats, serialization.
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
    TOURNAMENT_TIERS,
    RANKINGS_NAME_MAP,
    _ELO_INIT,
    _ELO_K,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(*matches):
    rows = []
    for i, (h, a, hs, a_s, neutral, tournament) in enumerate(matches):
        rows.append({
            "date":       pd.Timestamp(f"2020-{i+1:02d}-01"),
            "home_team":  h,
            "away_team":  a,
            "home_score": hs,
            "away_score": a_s,
            "neutral":    neutral,
            "tournament": tournament,
        })
    return pd.DataFrame(rows)


def _make_rank_df(entries):
    rows = [
        {"rank_date": pd.Timestamp(d), "country_full": t, "rank": r, "total_points": float(p)}
        for d, t, r, p in entries
    ]
    return pd.DataFrame(rows).sort_values("rank_date")


# ── Constants ─────────────────────────────────────────────────────────────────

def test_feature_cols_count():
    assert len(FEATURE_COLS) == 33

def test_feature_cols_are_unique():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

def test_elo_k_covers_all_tiers():
    for tier in (1, 2, 3, 4, 5):
        assert tier in _ELO_K, f"Tier {tier} missing from _ELO_K"

def test_elo_k_increases_with_tier():
    for t in range(1, 5):
        assert _ELO_K[t] < _ELO_K[t + 1], f"K[{t}] should be < K[{t+1}]"

def test_rankings_name_map_values_dont_overlap_keys():
    assert not (set(RANKINGS_NAME_MAP.keys()) & set(RANKINGS_NAME_MAP.values()))


# ── _new_team ─────────────────────────────────────────────────────────────────

def test_new_team_has_all_expected_fields():
    t = _new_team()
    required = {
        "games", "wins", "draws", "losses",
        "goals_scored", "goals_conceded",
        "recent", "recent_5", "recent_gd",
        "comp_games", "comp_wins", "comp_draws", "comp_losses",
    }
    assert set(t.keys()) == required

def test_new_team_deques_have_correct_maxlen():
    t = _new_team()
    assert t["recent"].maxlen == 10
    assert t["recent_5"].maxlen == 5
    assert t["recent_gd"].maxlen == 10

def test_new_team_numeric_fields_start_at_zero():
    t = _new_team()
    for field in ("games", "wins", "draws", "losses", "goals_scored", "goals_conceded",
                  "comp_games", "comp_wins", "comp_draws", "comp_losses"):
        assert t[field] == 0

def test_new_team_deques_are_empty():
    t = _new_team()
    assert len(t["recent"]) == 0
    assert len(t["recent_5"]) == 0
    assert len(t["recent_gd"]) == 0


# ── _get_tier ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tournament,expected", [
    ("FIFA World Cup",               5),
    ("UEFA Euro",                    4),
    ("Copa América",                 4),
    ("African Cup of Nations",       4),
    ("AFC Asian Cup",                4),
    ("Gold Cup",                     4),
    ("FIFA World Cup qualification", 3),
    ("UEFA Nations League",          3),
    ("CONCACAF Nations League",      3),
    ("Friendly",                     1),
])
def test_get_tier_known_tournaments(tournament, expected):
    assert _get_tier(tournament) == expected

def test_get_tier_unknown_defaults_to_friendly():
    assert _get_tier("Some Random Cup") == 1
    assert _get_tier("") == 1
    assert _get_tier("Club Friendly") == 1

def test_get_tier_all_defined_tournaments_are_reachable():
    for name, tier in TOURNAMENT_TIERS.items():
        assert _get_tier(name) == tier


# ── build_rank_index / _lookup_rank ───────────────────────────────────────────

def test_lookup_rank_exact_date():
    rk = _make_rank_df([
        ("2020-01-01", "Brazil", 1, 1800.0),
        ("2020-07-01", "Brazil", 2, 1750.0),
    ])
    idx = build_rank_index(rk)
    rank, pts = _lookup_rank("Brazil", pd.Timestamp("2020-07-01"), idx)
    assert rank == 2
    assert pts == 1750.0

def test_lookup_rank_uses_most_recent_prior_entry():
    rk = _make_rank_df([
        ("2020-01-01", "Brazil", 1, 1800.0),
        ("2020-07-01", "Brazil", 2, 1750.0),
    ])
    idx = build_rank_index(rk)
    rank, pts = _lookup_rank("Brazil", pd.Timestamp("2020-03-15"), idx)
    assert rank == 1

def test_lookup_rank_unknown_team_returns_default():
    rank, pts = _lookup_rank("Atlantis FC", pd.Timestamp("2020-01-01"), {})
    assert rank == 200
    assert pts == 0.0

def test_lookup_rank_before_first_entry_returns_default():
    rk = _make_rank_df([("2020-06-01", "Brazil", 3, 1700.0)])
    idx = build_rank_index(rk)
    rank, _ = _lookup_rank("Brazil", pd.Timestamp("2019-01-01"), idx)
    assert rank == 200

def test_lookup_rank_multiple_teams_independent():
    rk = _make_rank_df([
        ("2020-01-01", "Brazil",    1, 1800.0),
        ("2020-01-01", "Argentina", 3, 1750.0),
    ])
    idx = build_rank_index(rk)
    r_br, _ = _lookup_rank("Brazil",    pd.Timestamp("2020-06-01"), idx)
    r_ar, _ = _lookup_rank("Argentina", pd.Timestamp("2020-06-01"), idx)
    assert r_br == 1
    assert r_ar == 3

def test_build_rank_index_applies_name_map():
    rk = pd.DataFrame([{
        "rank_date": pd.Timestamp("2020-01-01"),
        "country_full": "Korea Republic",
        "rank": 30,
        "total_points": 1400.0,
    }])
    # load_rankings applies the name map, but build_rank_index does not —
    # so we test load_rankings maps the name before indexing
    from src.prepare import load_rankings
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        rk.to_csv(f, index=False)
        fname = f.name
    try:
        loaded = load_rankings(fname)
        assert "South Korea" in loaded["country_full"].values
        assert "Korea Republic" not in loaded["country_full"].values
    finally:
        os.unlink(fname)


# ── build_features ────────────────────────────────────────────────────────────

def test_build_features_first_match_has_zero_stats():
    df = _make_df(("A", "B", 2, 1, False, "FIFA World Cup"))
    X, y, _, _, _ = build_features(df)
    row = X.iloc[0]
    assert row["home_win_rate"] == 0.0
    assert row["away_win_rate"] == 0.0
    assert row["home_recent_form"] == 0.0
    assert row["h2h_total_games"] == 0

def test_build_features_labels_correct():
    df = _make_df(
        ("A", "B", 2, 1, False, "FIFA World Cup"),
        ("A", "B", 0, 0, False, "FIFA World Cup"),
        ("A", "B", 0, 1, False, "FIFA World Cup"),
    )
    _, y, _, _, _ = build_features(df)
    assert list(y) == ["Win", "Draw", "Loss"]

def test_build_features_has_all_feature_cols():
    df = _make_df(("A", "B", 1, 0, True, "FIFA World Cup"))
    X, _, _, _, _ = build_features(df)
    assert list(X.columns) == FEATURE_COLS

def test_build_features_no_leakage():
    df = _make_df(
        ("A", "B", 3, 0, False, "FIFA World Cup"),
        ("A", "B", 1, 0, False, "FIFA World Cup"),
    )
    X, _, _, _, _ = build_features(df)
    assert X.iloc[0]["home_win_rate"] == 0.0   # no prior games at row 0
    assert X.iloc[1]["home_win_rate"] == 1.0   # 1 win recorded after row 0

def test_build_features_h2h_updates_after_match():
    df = _make_df(
        ("A", "B", 2, 0, False, "FIFA World Cup"),
        ("A", "B", 0, 1, False, "FIFA World Cup"),
    )
    X, _, _, _, _ = build_features(df)
    assert X.iloc[0]["h2h_total_games"] == 0
    assert X.iloc[1]["h2h_total_games"] == 1

def test_build_features_h2h_win_rates_sum_to_one():
    df = _make_df(
        ("A", "B", 1, 0, False, "FIFA World Cup"),
        ("A", "B", 0, 1, False, "FIFA World Cup"),
        ("A", "B", 0, 0, False, "FIFA World Cup"),  # third match to inspect
    )
    X, _, _, _, _ = build_features(df)
    row = X.iloc[2]
    total = row["h2h_home_win_rate"] + row["h2h_draw_rate"] + row["h2h_away_win_rate"]
    assert total == pytest.approx(1.0)

def test_build_features_goals_tracked():
    df = _make_df(("A", "B", 3, 1, False, "FIFA World Cup"))
    _, _, team_stats, _, _ = build_features(df)
    assert team_stats["A"]["goals_scored"]   == 3
    assert team_stats["A"]["goals_conceded"] == 1
    assert team_stats["B"]["goals_scored"]   == 1
    assert team_stats["B"]["goals_conceded"] == 3

def test_build_features_win_loss_symmetry():
    df = _make_df(("A", "B", 2, 0, False, "FIFA World Cup"))
    _, _, team_stats, _, _ = build_features(df)
    assert team_stats["A"]["wins"]   == 1
    assert team_stats["A"]["losses"] == 0
    assert team_stats["B"]["wins"]   == 0
    assert team_stats["B"]["losses"] == 1

def test_build_features_draw_increments_both():
    df = _make_df(("A", "B", 1, 1, False, "FIFA World Cup"))
    _, _, team_stats, _, _ = build_features(df)
    assert team_stats["A"]["draws"] == 1
    assert team_stats["B"]["draws"] == 1

def test_build_features_elo_winner_gains_loser_drops():
    df = _make_df(("A", "B", 1, 0, False, "FIFA World Cup"))
    _, _, _, _, elos = build_features(df)
    assert elos["A"] > _ELO_INIT
    assert elos["B"] < _ELO_INIT

def test_build_features_elo_is_zero_sum():
    df = _make_df(("A", "B", 1, 0, False, "FIFA World Cup"))
    _, _, _, _, elos = build_features(df)
    delta = (elos["A"] - _ELO_INIT) + (elos["B"] - _ELO_INIT)
    assert abs(delta) < 1e-9

def test_build_features_elo_draw_equal_teams_barely_changes():
    df = _make_df(("A", "B", 1, 1, False, "FIFA World Cup"))
    _, _, _, _, elos = build_features(df)
    assert abs(elos["A"] - _ELO_INIT) < 5.0
    assert abs(elos["B"] - _ELO_INIT) < 5.0

def test_build_features_elo_bigger_margin_bigger_change():
    df1 = _make_df(("A", "B", 1, 0, False, "FIFA World Cup"))
    df3 = _make_df(("A", "B", 3, 0, False, "FIFA World Cup"))
    _, _, _, _, elos1 = build_features(df1)
    _, _, _, _, elos3 = build_features(df3)
    assert elos3["A"] > elos1["A"], "Bigger margin should give larger ELO gain"

def test_build_features_recent_form_5_empty_then_full():
    df = _make_df(
        ("A", "B", 1, 0, False, "Friendly"),
        ("A", "B", 1, 0, False, "Friendly"),
        ("A", "B", 1, 0, False, "Friendly"),
        ("A", "B", 1, 0, False, "Friendly"),
        ("A", "B", 1, 0, False, "Friendly"),
        ("A", "B", 1, 0, False, "Friendly"),
    )
    X, _, _, _, _ = build_features(df)
    assert X.iloc[0]["home_recent_form_5"] == 0.0
    assert X.iloc[1]["home_recent_form_5"] == pytest.approx(1.0)
    assert X.iloc[5]["home_recent_form_5"] == pytest.approx(1.0)

def test_build_features_recent_form_bounded_zero_to_one():
    df = _make_df(
        ("A", "B", 5, 0, False, "Friendly"),
        ("A", "B", 0, 5, False, "Friendly"),
        ("A", "B", 5, 0, False, "Friendly"),  # inspect row 2
    )
    X, _, _, _, _ = build_features(df)
    for col in ("home_recent_form", "home_recent_form_5",
                "away_recent_form", "away_recent_form_5"):
        for v in X[col]:
            assert 0.0 <= v <= 1.0, f"{col}={v} out of [0,1]"

def test_build_features_comp_games_only_competitive():
    df = _make_df(
        ("A", "B", 1, 0, False, "Friendly"),          # tier 1 — not competitive
        ("A", "B", 1, 0, False, "FIFA World Cup"),    # tier 5 — competitive
        ("A", "B", 1, 0, False, "UEFA Nations League"),  # tier 3 — competitive
    )
    _, _, team_stats, _, _ = build_features(df)
    assert team_stats["A"]["comp_games"] == 2
    assert team_stats["A"]["games"]      == 3

def test_build_features_comp_losses_tracked():
    df = _make_df(
        ("A", "B", 0, 2, False, "FIFA World Cup"),
        ("A", "B", 0, 0, False, "FIFA World Cup"),
    )
    _, _, team_stats, _, _ = build_features(df)
    assert team_stats["A"]["comp_losses"] == 1
    assert team_stats["B"]["comp_losses"] == 0

def test_build_features_friendlies_not_counted_as_comp():
    df = _make_df(("A", "B", 0, 2, False, "Friendly"))
    _, _, team_stats, _, _ = build_features(df)
    assert team_stats["A"]["comp_games"]  == 0
    assert team_stats["A"]["comp_losses"] == 0

def test_build_features_neutral_venue_flag():
    df = _make_df(
        ("A", "B", 1, 0, True,  "FIFA World Cup"),
        ("A", "B", 1, 0, False, "FIFA World Cup"),
    )
    X, _, _, _, _ = build_features(df)
    assert X.iloc[0]["is_neutral"] == 1
    assert X.iloc[1]["is_neutral"] == 0

def test_build_features_tournament_tier_in_features():
    df = _make_df(
        ("A", "B", 1, 0, True, "FIFA World Cup"),
        ("A", "B", 1, 0, True, "Friendly"),
    )
    X, _, _, _, _ = build_features(df)
    assert X.iloc[0]["tournament_tier"] == 5
    assert X.iloc[1]["tournament_tier"] == 1

def test_build_features_rank_diff_is_away_minus_home():
    rk = _make_rank_df([
        ("2019-01-01", "A", 5,  1600.0),
        ("2019-01-01", "B", 20, 1400.0),
    ])
    rank_idx = build_rank_index(rk)
    df = _make_df(("A", "B", 1, 0, True, "FIFA World Cup"))
    X, _, _, _, _ = build_features(df, rank_index=rank_idx)
    assert X.iloc[0]["rank_diff"] == 15   # 20 - 5


# ── serialize_stats ───────────────────────────────────────────────────────────

def test_serialize_deques_to_lists():
    t = _new_team()
    t["recent"].extend([3, 1, 3])
    t["recent_5"].extend([3, 0])
    t["recent_gd"].extend([2, -1])
    ts, _ = serialize_stats({"A": t}, {})
    assert isinstance(ts["A"]["recent"],    list)
    assert isinstance(ts["A"]["recent_5"],  list)
    assert isinstance(ts["A"]["recent_gd"], list)
    assert ts["A"]["recent"]   == [3, 1, 3]
    assert ts["A"]["recent_5"] == [3, 0]

def test_serialize_preserves_numeric_fields():
    t = _new_team()
    t["wins"] = 5; t["comp_wins"] = 3; t["comp_losses"] = 1
    ts, _ = serialize_stats({"X": t}, {})
    assert ts["X"]["wins"]        == 5
    assert ts["X"]["comp_wins"]   == 3
    assert ts["X"]["comp_losses"] == 1

def test_serialize_h2h_roundtrips():
    from collections import defaultdict
    h2h = defaultdict(lambda: {"a_wins": 0, "draws": 0, "b_wins": 0})
    h2h[("A", "B")]["a_wins"] = 3
    h2h[("A", "B")]["draws"]  = 1
    _, h2h_out = serialize_stats({}, h2h)
    assert h2h_out[("A", "B")]["a_wins"] == 3
    assert h2h_out[("A", "B")]["draws"]  == 1

def test_serialize_empty_inputs():
    ts, h2h = serialize_stats({}, {})
    assert ts == {}
    assert h2h == {}


# ── build_feature_vector ──────────────────────────────────────────────────────

def test_build_feature_vector_has_all_cols():
    fv = build_feature_vector("A", "B", True, {}, {})
    assert set(fv.keys()) == set(FEATURE_COLS)

def test_build_feature_vector_unknown_teams_zero():
    fv = build_feature_vector("X", "Y", True, {}, {})
    assert fv["home_win_rate"]   == 0.0
    assert fv["away_win_rate"]   == 0.0
    assert fv["h2h_total_games"] == 0
    assert fv["home_elo"]        == _ELO_INIT
    assert fv["away_elo"]        == _ELO_INIT
    assert fv["elo_diff"]        == 0.0
    assert fv["abs_elo_diff"]    == 0.0

def test_build_feature_vector_elo_diff_sign():
    elos = {"A": 1700.0, "B": 1400.0}
    fv_ab = build_feature_vector("A", "B", True, {}, {}, elo_ratings=elos)
    fv_ba = build_feature_vector("B", "A", True, {}, {}, elo_ratings=elos)
    assert fv_ab["elo_diff"]     == pytest.approx(300.0)
    assert fv_ba["elo_diff"]     == pytest.approx(-300.0)
    assert fv_ab["abs_elo_diff"] == pytest.approx(300.0)
    assert fv_ba["abs_elo_diff"] == pytest.approx(300.0)

def test_build_feature_vector_rank_diff():
    fv = build_feature_vector("A", "B", True, {}, {}, home_rank=5, away_rank=20)
    assert fv["rank_diff"] == 15

def test_build_feature_vector_neutral_flag():
    assert build_feature_vector("A", "B", True,  {}, {})["is_neutral"] == 1
    assert build_feature_vector("A", "B", False, {}, {})["is_neutral"] == 0

def test_build_feature_vector_win_rates_sum_to_one():
    _, _, team_stats, h2h, elos = build_features(_make_df(
        ("A", "B", 1, 0, False, "FIFA World Cup"),
        ("A", "B", 0, 1, False, "FIFA World Cup"),
        ("A", "B", 1, 1, False, "FIFA World Cup"),
    ))
    ts, h2h_s = serialize_stats(team_stats, h2h)
    fv = build_feature_vector("A", "B", True, ts, h2h_s, elo_ratings=elos)
    assert fv["home_win_rate"] + fv["home_draw_rate"] + fv["home_loss_rate"] == pytest.approx(1.0)

def test_build_feature_vector_elo_none_uses_init_rating():
    # When elo_ratings=None both teams get the initial ELO and diff is zero
    fv = build_feature_vector("A", "B", True, {}, {}, elo_ratings=None)
    assert fv["home_elo"]     == pytest.approx(_ELO_INIT)
    assert fv["away_elo"]     == pytest.approx(_ELO_INIT)
    assert fv["elo_diff"]     == pytest.approx(0.0)
    assert fv["abs_elo_diff"] == pytest.approx(0.0)

def test_build_feature_vector_reflects_serialized_stats():
    _, _, team_stats, h2h, elos = build_features(_make_df(
        ("A", "B", 2, 0, False, "FIFA World Cup"),
        ("A", "B", 2, 0, False, "FIFA World Cup"),
    ))
    ts, h2h_s = serialize_stats(team_stats, h2h)
    fv = build_feature_vector("A", "B", True, ts, h2h_s, elo_ratings=elos)
    assert fv["home_win_rate"]   == pytest.approx(1.0)
    assert fv["h2h_total_games"] == 2
    assert fv["home_elo"]        > _ELO_INIT
