"""
Global app state — artifacts loaded once at startup, shared across all requests.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import shap

from src.scraper import fetch_fifa_rankings

log = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts")
DATA = Path("data")


@dataclass
class AppState:
    model:        object
    explainer:    object                    # shap.TreeExplainer, cached at startup
    team_stats:   dict[str, dict]
    h2h:          dict
    elo_ratings:  dict[str, float]
    teams:        list[str]
    rankings:     dict[str, dict]          # {team: {rank, points}}
    groups:       dict[str, list[str]]     # {group_name: [team, ...]}


_state: AppState | None = None


def load_state() -> AppState:
    global _state
    t0 = time.perf_counter()
    log.info("Loading artifacts from %s …", ARTIFACTS)

    model       = joblib.load(ARTIFACTS / "model.pkl")
    log.info("Model loaded: %s", type(model).__name__)

    explainer   = shap.TreeExplainer(model._clf)
    team_stats  = joblib.load(ARTIFACTS / "team_stats.pkl")
    h2h         = joblib.load(ARTIFACTS / "h2h.pkl")
    elo_ratings = joblib.load(ARTIFACTS / "elo_ratings.pkl")
    teams       = joblib.load(ARTIFACTS / "teams.pkl")
    log.info("Artifacts loaded: %d teams, %d h2h pairs, %d elo ratings",
             len(teams), len(h2h), len(elo_ratings))

    rankings    = fetch_fifa_rankings()
    log.info("FIFA rankings loaded: %d teams ranked", len(rankings))

    with open(DATA / "wc2026_groups.json") as f:
        groups = json.load(f)["groups"]
    log.info("Groups loaded: %d groups", len(groups))

    _state = AppState(model, explainer, team_stats, h2h, elo_ratings, teams, rankings, groups)
    log.info("Startup complete in %.2fs", time.perf_counter() - t0)
    return _state


def get_state() -> AppState:
    if _state is None:
        raise RuntimeError("App state not initialised — call load_state() first")
    return _state
