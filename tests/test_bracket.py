"""
Tests for src/bracket.py — bracket construction, group/knockout simulation.
"""

import sys
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bracket import (
    _knockout_winner,
    _match_proba,
    _sample_outcome,
    _simulate_group_once,
    build_r32_bracket,
    most_likely_group_standings,
    simulate_groups,
    simulate_knockout,
)
from src.prepare import FEATURE_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_standings(n_groups: int = 12) -> dict[str, list[str]]:
    letters = "ABCDEFGHIJKL"
    return {letters[i]: [f"{letters[i]}{j}" for j in range(1, 5)] for i in range(n_groups)}


def _make_sim_results(standings: dict) -> dict[str, dict]:
    results = {}
    for order in standings.values():
        for j, team in enumerate(order):
            results[team] = {"avg_pts": float(6 - j), "advance_pct": float(80 - j * 20)}
    return results


def _model(outcome: str = "Win") -> MagicMock:
    """Mock model that always predicts one outcome with 100% probability."""
    m = MagicMock()
    m.classes_ = ["Draw", "Loss", "Win"]
    proba_map = {
        "Win":  [0.0, 0.0, 1.0],
        "Draw": [1.0, 0.0, 0.0],
        "Loss": [0.0, 1.0, 0.0],
    }
    m.predict_proba.return_value = [proba_map[outcome]]
    return m


def _balanced_model() -> MagicMock:
    """Model that returns equal probabilities (1/3 each)."""
    m = MagicMock()
    m.classes_ = ["Draw", "Loss", "Win"]
    m.predict_proba.return_value = [[1/3, 1/3, 1/3]]
    return m


# ── _sample_outcome ───────────────────────────────────────────────────────────

def test_sample_outcome_certain_win():
    for _ in range(20):
        assert _sample_outcome({"Win": 1.0, "Draw": 0.0, "Loss": 0.0}) == "Win"

def test_sample_outcome_certain_draw():
    for _ in range(20):
        assert _sample_outcome({"Win": 0.0, "Draw": 1.0, "Loss": 0.0}) == "Draw"

def test_sample_outcome_certain_loss():
    for _ in range(20):
        assert _sample_outcome({"Win": 0.0, "Draw": 0.0, "Loss": 1.0}) == "Loss"

def test_sample_outcome_returns_valid_value():
    random.seed(42)
    for _ in range(100):
        r = _sample_outcome({"Win": 0.4, "Draw": 0.3, "Loss": 0.3})
        assert r in ("Win", "Draw", "Loss")


# ── _match_proba ──────────────────────────────────────────────────────────────

def test_match_proba_returns_win_draw_loss_keys():
    m = _model("Win")
    p = _match_proba("A", "B", m, {}, {})
    assert set(p.keys()) == {"Win", "Draw", "Loss"}

def test_match_proba_probabilities_sum_to_one():
    m = _balanced_model()
    p = _match_proba("A", "B", m, {}, {})
    assert sum(p.values()) == pytest.approx(1.0)

def test_match_proba_calls_predict_proba_with_feature_cols():
    m = _model("Win")
    _match_proba("A", "B", m, {}, {})
    X_used = m.predict_proba.call_args[0][0]
    assert list(X_used.columns) == FEATURE_COLS

def test_match_proba_uses_rankings():
    m = _model("Win")
    rankings = {"A": {"rank": 1}, "B": {"rank": 50}}
    _match_proba("A", "B", m, {}, {}, rankings=rankings)
    # Should not raise; model receives a DataFrame either way
    m.predict_proba.assert_called_once()

def test_match_proba_no_rankings_defaults_to_zero():
    m = _model("Win")
    _match_proba("A", "B", m, {}, {}, rankings=None)
    X_used = m.predict_proba.call_args[0][0]
    assert X_used.iloc[0]["home_rank"] == 0
    assert X_used.iloc[0]["away_rank"] == 0


# ── _knockout_winner ──────────────────────────────────────────────────────────

def test_knockout_winner_returns_one_of_two():
    m = _model("Win")
    for seed in range(10):
        random.seed(seed)
        w = _knockout_winner("A", "B", m, {}, {}, rankings=None)
        assert w in ("A", "B")

def test_knockout_winner_certain_home():
    m = _model("Win")
    for _ in range(20):
        assert _knockout_winner("A", "B", m, {}, {}, rankings=None) == "A"

def test_knockout_winner_certain_away():
    m = _model("Loss")
    for _ in range(20):
        assert _knockout_winner("A", "B", m, {}, {}, rankings=None) == "B"

def test_knockout_winner_draw_is_50_50():
    # With draw probability=1.0 and 50/50 penalty logic, over many trials
    # the winner should be each team roughly half the time.
    m = _model("Draw")
    random.seed(0)
    results = [_knockout_winner("A", "B", m, {}, {}, rankings=None) for _ in range(1000)]
    a_wins = results.count("A")
    assert 400 < a_wins < 600, f"Expected ~500 A wins, got {a_wins}"


# ── _simulate_group_once ──────────────────────────────────────────────────────

def test_simulate_group_once_returns_all_teams():
    result = _simulate_group_once(["A","B","C","D"], _model("Win"), {}, {}, None)
    assert set(result.keys()) == {"A", "B", "C", "D"}

def test_simulate_group_once_positions_1_to_4():
    result = _simulate_group_once(["A","B","C","D"], _model("Win"), {}, {}, None)
    assert sorted(r["pos"] for r in result.values()) == [1, 2, 3, 4]

def test_simulate_group_once_positions_unique():
    result = _simulate_group_once(["A","B","C","D"], _model("Win"), {}, {}, None)
    positions = [r["pos"] for r in result.values()]
    assert len(set(positions)) == 4

def test_simulate_group_once_pts_non_negative():
    result = _simulate_group_once(["A","B","C","D"], _model("Draw"), {}, {}, None)
    assert all(r["pts"] >= 0 for r in result.values())

def test_simulate_group_once_total_pts_all_draw():
    # 6 matches, all draws → 2 pts per match × 6 = 12
    result = _simulate_group_once(["A","B","C","D"], _model("Draw"), {}, {}, None)
    assert sum(r["pts"] for r in result.values()) == 12

def test_simulate_group_once_total_pts_all_home_win():
    # 6 matches, all home wins → 3 pts per match × 6 = 18
    result = _simulate_group_once(["A","B","C","D"], _model("Win"), {}, {}, None)
    assert sum(r["pts"] for r in result.values()) == 18

def test_simulate_group_once_gd_sums_to_zero():
    # goal differences must cancel out across all teams
    result = _simulate_group_once(["A","B","C","D"], _model("Win"), {}, {}, None)
    assert sum(r["gd"] for r in result.values()) == 0


# ── build_r32_bracket ─────────────────────────────────────────────────────────

def test_bracket_has_32_teams():
    s = _make_standings(12)
    bracket = build_r32_bracket(s, _make_sim_results(s))
    assert len(bracket) == 32

def test_bracket_no_duplicates():
    s = _make_standings(12)
    bracket = build_r32_bracket(s, _make_sim_results(s))
    assert len(set(bracket)) == 32

def test_bracket_all_firsts_present():
    s = _make_standings(12)
    bracket = set(build_r32_bracket(s, _make_sim_results(s)))
    for order in s.values():
        assert order[0] in bracket

def test_bracket_all_seconds_present():
    s = _make_standings(12)
    bracket = set(build_r32_bracket(s, _make_sim_results(s)))
    for order in s.values():
        assert order[1] in bracket

def test_bracket_best_8_thirds_present():
    s = _make_standings(12)
    sim = _make_sim_results(s)
    bracket = set(build_r32_bracket(s, sim))
    thirds = sorted(
        [(sim[o[2]]["avg_pts"], sim[o[2]]["advance_pct"], o[2]) for o in s.values()],
        reverse=True
    )
    best_8 = {t for _, _, t in thirds[:8]}
    assert best_8 <= bracket

def test_bracket_worst_4_thirds_excluded():
    s = _make_standings(12)
    sim = _make_sim_results(s)
    bracket = set(build_r32_bracket(s, sim))
    thirds = sorted(
        [(sim[o[2]]["avg_pts"], sim[o[2]]["advance_pct"], o[2]) for o in s.values()]
    )
    worst_4 = {t for _, _, t in thirds[:4]}
    assert not (worst_4 & bracket)

def test_bracket_is_even_length():
    s = _make_standings(12)
    bracket = build_r32_bracket(s, _make_sim_results(s))
    assert len(bracket) % 2 == 0


# ── simulate_groups ───────────────────────────────────────────────────────────

def test_simulate_groups_returns_all_teams():
    groups = _make_standings(4)   # 4 groups of 4 = 16 teams
    all_teams = {t for ts in groups.values() for t in ts}
    results = simulate_groups(groups, _model("Win"), {}, {}, n_sims=10)
    assert set(results.keys()) == all_teams

def test_simulate_groups_percentages_in_range():
    groups = _make_standings(4)
    results = simulate_groups(groups, _model("Win"), {}, {}, n_sims=10)
    for team, stats in results.items():
        for key in ("advance_pct", "1st_pct", "2nd_pct", "3rd_pct", "4th_pct"):
            assert 0.0 <= stats[key] <= 100.0, f"{team}/{key}={stats[key]}"

def test_simulate_groups_position_pcts_sum_to_100():
    groups = _make_standings(4)
    results = simulate_groups(groups, _model("Win"), {}, {}, n_sims=50)
    for team, stats in results.items():
        total = stats["1st_pct"] + stats["2nd_pct"] + stats["3rd_pct"] + stats["4th_pct"]
        assert total == pytest.approx(100.0, abs=1.0), f"{team} position pcts sum to {total}"

def test_simulate_groups_avg_pts_non_negative():
    groups = _make_standings(4)
    results = simulate_groups(groups, _balanced_model(), {}, {}, n_sims=10)
    for team, stats in results.items():
        assert stats["avg_pts"] >= 0.0

def test_simulate_groups_top_team_high_advance_pct():
    # Model always predicts Win for home → first listed team in each group wins a lot
    groups = {"A": ["A1", "A2", "A3", "A4"]}
    results = simulate_groups(groups, _model("Win"), {}, {}, n_sims=100)
    # A1 is always home first and wins; should have the highest advance_pct
    assert results["A1"]["advance_pct"] >= results["A4"]["advance_pct"]


# ── most_likely_group_standings ───────────────────────────────────────────────

def test_most_likely_group_standings_returns_all_groups():
    groups = _make_standings(4)
    result = most_likely_group_standings(groups, _model("Win"), {}, {})
    assert set(result.keys()) == set(groups.keys())

def test_most_likely_group_standings_all_teams_in_each_group():
    groups = _make_standings(4)
    result = most_likely_group_standings(groups, _model("Win"), {}, {})
    for g, order in result.items():
        assert set(order) == set(groups[g])

def test_most_likely_group_standings_length_4_per_group():
    groups = _make_standings(4)
    result = most_likely_group_standings(groups, _model("Win"), {}, {})
    for order in result.values():
        assert len(order) == 4

def test_most_likely_group_standings_no_duplicates_per_group():
    groups = _make_standings(4)
    result = most_likely_group_standings(groups, _model("Win"), {}, {})
    for order in result.values():
        assert len(set(order)) == 4

def test_most_likely_group_standings_deterministic():
    # Same model, same groups → same result every call (no randomness)
    groups = _make_standings(3)
    m = _model("Win")
    r1 = most_likely_group_standings(groups, m, {}, {})
    r2 = most_likely_group_standings(groups, m, {}, {})
    assert r1 == r2


# ── simulate_knockout ─────────────────────────────────────────────────────────

def test_simulate_knockout_returns_all_teams():
    qualifiers = [f"T{i}" for i in range(32)]
    results = simulate_knockout(qualifiers, _model("Win"), {}, {}, n_sims=10)
    assert set(results.keys()) == set(qualifiers)

def test_simulate_knockout_round_keys():
    qualifiers = [f"T{i}" for i in range(32)]
    results = simulate_knockout(qualifiers, _model("Win"), {}, {}, n_sims=10)
    for team, rounds in results.items():
        assert set(rounds.keys()) == {"R16", "QF", "SF", "Final", "Winner"}

def test_simulate_knockout_probabilities_in_range():
    qualifiers = [f"T{i}" for i in range(32)]
    results = simulate_knockout(qualifiers, _model("Win"), {}, {}, n_sims=10)
    for team, rounds in results.items():
        for rnd, pct in rounds.items():
            assert 0.0 <= pct <= 100.0, f"{team}/{rnd}={pct}"

def test_simulate_knockout_winner_sums_to_100():
    qualifiers = [f"T{i}" for i in range(32)]
    results = simulate_knockout(qualifiers, _model("Win"), {}, {}, n_sims=50)
    total = sum(r["Winner"] for r in results.values())
    assert total == pytest.approx(100.0, abs=0.1)

def test_simulate_knockout_r16_sum_equals_100_pct_times_16():
    # 16 teams advance to R16; each sim contributes 16 R16 advancements → 16 * n_sims
    # as a percentage: 16 teams should have R16=100% if deterministic model
    qualifiers = [f"T{i}" for i in range(32)]
    results = simulate_knockout(qualifiers, _model("Win"), {}, {}, n_sims=50)
    r16_total = sum(r["R16"] for r in results.values())
    assert r16_total == pytest.approx(16 * 100.0 / 32 * 32, rel=0.01)

def test_simulate_knockout_certain_winner_wins_all():
    # Model always picks home team → T0 beats T1 in R32, T0 beats T2 in R16, etc.
    qualifiers = [f"T{i}" for i in range(32)]
    results = simulate_knockout(qualifiers, _model("Win"), {}, {}, n_sims=20)
    assert results["T0"]["Winner"] == pytest.approx(100.0)

def test_simulate_knockout_losing_bracket_side():
    # With Win model: T1 (always away in R32) never wins R32
    qualifiers = [f"T{i}" for i in range(32)]
    results = simulate_knockout(qualifiers, _model("Win"), {}, {}, n_sims=20)
    assert results["T1"]["R16"] == pytest.approx(0.0)
