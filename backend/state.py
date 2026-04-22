"""
Global app state — artifacts loaded once at startup, shared across all requests.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from src.scraper import fetch_fifa_rankings

ARTIFACTS = Path("artifacts")
DATA = Path("data")


@dataclass
class AppState:
    model:        object
    team_stats:   dict
    h2h:          dict
    elo_ratings:  dict
    teams:        list[str]
    rankings:     dict   # {team: {rank, points}}
    groups:       dict   # {group_name: [team, ...]}


_state: AppState | None = None


def load_state() -> AppState:
    global _state
    model       = joblib.load(ARTIFACTS / "model.pkl")
    team_stats  = joblib.load(ARTIFACTS / "team_stats.pkl")
    h2h         = joblib.load(ARTIFACTS / "h2h.pkl")
    elo_ratings = joblib.load(ARTIFACTS / "elo_ratings.pkl")
    teams       = joblib.load(ARTIFACTS / "teams.pkl")
    rankings    = fetch_fifa_rankings()
    with open(DATA / "wc2026_groups.json") as f:
        groups = json.load(f)["groups"]
    _state = AppState(model, team_stats, h2h, elo_ratings, teams, rankings, groups)
    return _state


def get_state() -> AppState:
    if _state is None:
        raise RuntimeError("App state not initialised — call load_state() first")
    return _state
