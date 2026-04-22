"""
Tests for src/bracket.py — bracket construction, group simulation, knockout helpers.
"""

import sys
import random
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bracket import (
    _knockout_winner,
    _sample_outcome,
    _simulate_group_once,
    build_r32_bracket,
    simulate_knockout,
)
from src.prepare import FEATURE_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_group_standings(n_groups: int = 12):
    """Create synthetic most_likely_group_standings for n_groups groups of 4."""
    letters = "ABCDEFGHIJKL"
    standings = {}
    for i in range(n_groups):
        g = letters[i]
        standings[g] = [f"{g}1", f"{g}2", f"{g}3", f"{g}4"]
    return standings


def _make_group_sim_results(standings: dict):
    """Create matching group_sim_results (avg_pts + advance_pct for each team)."""
    results = {}
    for order in standings.values():
        for j, team in enumerate(order):
            results[team] = {
                "avg_pts":      6.0 - j,
                "advance_pct":  80.0 - j * 20,
            }
    return results


def _deterministic_model(outcome: str = "Win"):
    """
    Mock model that always predicts the same outcome with 100% probability.
    predict_proba returns [draw, loss, win] in alphabetical class order.
    """
    model = MagicMock()
    model.classes_ = ["Draw", "Loss", "Win"]
    proba_map = {"Win": [0.0, 0.0, 1.0], "Draw": [1.0, 0.0, 0.0], "Loss": [0.0, 1.0, 0.0]}
    model.predict_proba.return_value = [proba_map[outcome]]
    return model


# ── build_r32_bracket ─────────────────────────────────────────────────────────

def test_bracket_has_exactly_32_teams():
    standings = _make_group_standings(12)
    sim_results = _make_group_sim_results(standings)
    bracket = build_r32_bracket(standings, sim_results)
    assert len(bracket) == 32


def test_bracket_has_no_duplicate_teams():
    standings = _make_group_standings(12)
    sim_results = _make_group_sim_results(standings)
    bracket = build_r32_bracket(standings, sim_results)
    assert len(set(bracket)) == 32


def test_bracket_contains_all_firsts_and_seconds():
    standings = _make_group_standings(12)
    sim_results = _make_group_sim_results(standings)
    bracket = set(build_r32_bracket(standings, sim_results))
    for order in standings.values():
        assert order[0] in bracket, f"Group winner {order[0]} missing from bracket"
        assert order[1] in bracket, f"Group runner-up {order[1]} missing from bracket"


def test_bracket_contains_best_8_thirds():
    standings = _make_group_standings(12)
    sim_results = _make_group_sim_results(standings)
    bracket = set(build_r32_bracket(standings, sim_results))
    all_thirds = [(sim_results[o[2]]["avg_pts"], sim_results[o[2]]["advance_pct"], o[2])
                  for o in standings.values()]
    all_thirds.sort(reverse=True)
    best_8 = {t for _, _, t in all_thirds[:8]}
    assert best_8 <= bracket, f"Best 8 third-place teams not all in bracket"


def test_bracket_length_is_even_for_pairing():
    standings = _make_group_standings(12)
    sim_results = _make_group_sim_results(standings)
    bracket = build_r32_bracket(standings, sim_results)
    assert len(bracket) % 2 == 0


def test_bracket_excludes_worst_thirds():
    standings = _make_group_standings(12)
    sim_results = _make_group_sim_results(standings)
    bracket = set(build_r32_bracket(standings, sim_results))
    all_thirds = sorted(
        [o[2] for o in standings.values()],
        key=lambda t: sim_results[t]["avg_pts"],
    )
    worst_4 = all_thirds[:4]
    for t in worst_4:
        assert t not in bracket, f"Worst third-place team {t} should not be in bracket"


# ── _sample_outcome ───────────────────────────────────────────────────────────

def test_sample_outcome_certain_win():
    random.seed(0)
    for _ in range(20):
        assert _sample_outcome({"Win": 1.0, "Draw": 0.0, "Loss": 0.0}) == "Win"


def test_sample_outcome_certain_loss():
    random.seed(0)
    for _ in range(20):
        assert _sample_outcome({"Win": 0.0, "Draw": 0.0, "Loss": 1.0}) == "Loss"


def test_sample_outcome_returns_valid_outcome():
    random.seed(42)
    for _ in range(100):
        r = _sample_outcome({"Win": 0.4, "Draw": 0.3, "Loss": 0.3})
        assert r in ("Win", "Draw", "Loss")


# ── _knockout_winner ──────────────────────────────────────────────────────────

def test_knockout_winner_returns_one_of_two_teams():
    model = _deterministic_model("Win")
    for seed in range(10):
        random.seed(seed)
        w = _knockout_winner("TeamA", "TeamB", model, {}, {}, rankings=None)
        assert w in ("TeamA", "TeamB")


def test_knockout_winner_certain_home_win():
    model = _deterministic_model("Win")
    random.seed(0)
    for _ in range(20):
        w = _knockout_winner("TeamA", "TeamB", model, {}, {}, rankings=None)
        assert w == "TeamA", "p(Win)=1.0 should always yield home team"


def test_knockout_winner_certain_away_win():
    model = _deterministic_model("Loss")
    random.seed(0)
    for _ in range(20):
        w = _knockout_winner("TeamA", "TeamB", model, {}, {}, rankings=None)
        assert w == "TeamB", "p(Loss)=1.0 should always yield away team"


# ── _simulate_group_once ──────────────────────────────────────────────────────

def test_simulate_group_once_returns_all_teams():
    model = _deterministic_model("Win")
    teams = ["A", "B", "C", "D"]
    result = _simulate_group_once(teams, model, {}, {}, rankings=None)
    assert set(result.keys()) == set(teams)


def test_simulate_group_once_positions_are_1_to_4():
    model = _deterministic_model("Win")
    teams = ["A", "B", "C", "D"]
    result = _simulate_group_once(teams, model, {}, {}, rankings=None)
    positions = sorted(r["pos"] for r in result.values())
    assert positions == [1, 2, 3, 4]


def test_simulate_group_once_pts_are_non_negative():
    model = _deterministic_model("Draw")
    teams = ["A", "B", "C", "D"]
    result = _simulate_group_once(teams, model, {}, {}, rankings=None)
    for team, stats in result.items():
        assert stats["pts"] >= 0, f"{team} has negative points"


def test_simulate_group_once_total_pts_correct():
    # Each match awards 3 points total (win) or 2 (draw) or 3 (win).
    # 6 matches in a 4-team group, all draws → 6×2 = 12 total points
    model = _deterministic_model("Draw")
    teams = ["A", "B", "C", "D"]
    result = _simulate_group_once(teams, model, {}, {}, rankings=None)
    assert sum(r["pts"] for r in result.values()) == 12


# ── simulate_knockout ─────────────────────────────────────────────────────────

def test_simulate_knockout_returns_all_qualifier_teams():
    model = _deterministic_model("Win")
    qualifiers = [f"T{i}" for i in range(32)]
    results = simulate_knockout(qualifiers, model, {}, {}, n_sims=10, rankings=None)
    assert set(results.keys()) == set(qualifiers)


def test_simulate_knockout_probabilities_between_0_and_100():
    model = _deterministic_model("Win")
    qualifiers = [f"T{i}" for i in range(32)]
    results = simulate_knockout(qualifiers, model, {}, {}, n_sims=10, rankings=None)
    for team, rounds in results.items():
        for round_name, pct in rounds.items():
            assert 0.0 <= pct <= 100.0, f"{team} {round_name}: {pct} out of range"


def test_simulate_knockout_certain_winner_always_wins():
    # Model always predicts Win for home team. First qualifier should win everything.
    model = _deterministic_model("Win")
    qualifiers = [f"T{i}" for i in range(32)]
    results = simulate_knockout(qualifiers, model, {}, {}, n_sims=20, rankings=None)
    # T0 always wins (always home in each pair), so winner prob should be 100%
    assert results["T0"]["Winner"] == pytest.approx(100.0)
